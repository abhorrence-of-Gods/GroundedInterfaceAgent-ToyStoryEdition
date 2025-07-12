"""Streaming RL trainer for the Meta-Cognitive Dreamer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import os
from torch.utils.tensorboard import SummaryWriter
import lpips

from data.dataloader import create_dataloader
from models.gia_agent import GiaAgent
from models.goal_bank import GoalLatentBank
from models.meta_loss_net import MetaLossNet
from models.utils.meta_loss_utils import kl_divergence_gaussian, lambda_return


# --- Loss Utility Functions ---
def kl_divergence_gaussian(mu, logvar):
    """KL divergence between N(mu, exp(logvar)) and N(0, 1)."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


def lambda_return(rewards, values, gamma, lmbda):
    """Computes lambda-returns for an imagined trajectory."""
    returns = torch.zeros_like(rewards)
    last_val = values[-1]
    for t in reversed(range(rewards.size(0))):
        last_val = rewards[t] + gamma * (1 - lmbda) * values[t] + gamma * lmbda * last_val
        returns[t] = last_val
    return returns


# --- Main Trainer Class ---
class StreamingTrainer:
    def __init__(self, cfg: DictConfig, model: GiaAgent):
        self.cfg = cfg
        self.model = model
        self.device = next(model.parameters()).device
        self.glb = GoalLatentBank().to(self.device)
        self.meta_net = MetaLossNet().to(self.device)
        self.dataloader = create_dataloader(batch_size=cfg.training.batch_size)
        
        # Combine all trainable parameters
        params = (
            list(self.model.parameters())
            + list(self.glb.parameters())
            + list(self.meta_net.parameters())
        )
        self.opt = hydra.utils.instantiate(cfg.training.optimizer, _convert_="object", params=params)
        
        self.start_epoch = 0
        self.writer = SummaryWriter(log_dir=self.cfg.get("log_dir", "runs/streaming_rl"))
        
        # Initialize LPIPS loss
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device).eval()
        # Freeze LPIPS model
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
            
        # Checkpoint loading
        self._load_checkpoint()

    def _get_kl_beta(self, epoch: int) -> float:
        """Linear KL annealing."""
        anneal_epochs = self.cfg.training.get("kl_anneal_epochs", 0)
        if anneal_epochs == 0:
            return 1.0
        return min(1.0, epoch / anneal_epochs)

    def _save_checkpoint(self, epoch: int):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.checkpoint_dir, f"epoch_{epoch}.pt")
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "glb_state_dict": self.glb.state_dict(),
            "meta_net_state_dict": self.meta_net.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
        }
        torch.save(state, path)
        print(f"[Trainer] Checkpoint saved to {path}")

    def _load_checkpoint(self):
        # Find the latest checkpoint
        if not os.path.isdir(self.cfg.checkpoint_dir):
            return
        checkpoints = [f for f in os.listdir(self.cfg.checkpoint_dir) if f.endswith('.pt')]
        if not checkpoints:
            return
        
        latest_ckpt = max(checkpoints, key=lambda f: int(f.split('_')[1].split('.')[0]))
        path = os.path.join(self.cfg.checkpoint_dir, latest_ckpt)

        try:
            state = torch.load(path, map_location=self.device)
            self.start_epoch = state["epoch"] + 1
            self.model.load_state_dict(state["model_state_dict"])
            self.glb.load_state_dict(state["glb_state_dict"])
            self.meta_net.load_state_dict(state["meta_net_state_dict"])
            self.opt.load_state_dict(state["optimizer_state_dict"])
            print(f"[Trainer] Resumed from checkpoint {path} at epoch {self.start_epoch}")
        except (FileNotFoundError, KeyError) as e:
            print(f"[Trainer] Could not load checkpoint from {path}: {e}")

    def get_loss_weights(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Returns raw loss weights from the meta-net."""
        return self.meta_net(state, goal)

    def train(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.cfg.training.num_epochs):
            beta = self._get_kl_beta(epoch)
            self.writer.add_scalar("train/kl_beta", beta, epoch)

            for i, batch in enumerate(self.dataloader):
                screenshot = batch["screenshot"].to(self.device)
                instr = batch["instruction_text"]
                goal = batch["goal"].to(self.device) if "goal" in batch else None

                # Get latent distributions and sample z0, u0
                z0, u0, (z_mu, z_logvar), (u_mu, u_logvar) = self.model.encode_state(
                    instr, screenshot, goal_vec=goal
                )
                
                # Use a consistent goal for the entire sequence
                glb_goal, _ = self.glb(z0.detach())

                # ---------- Dream roll-out ----------
                H = self.cfg.training.get("horizon", 8)
                imagined_z, imagined_a, imagined_r, imagined_v = self.model.plan_in_dream(z0, u0, horizon=H)

                # ---------- Actor-Critic Loss ----------
                action_dist = self.model.action_tower.get_action_dist(imagined_z)
                
                returns = lambda_return(
                    imagined_r.squeeze(-1),
                    torch.cat([imagined_v, imagined_v[-1:]], 0),
                    gamma=self.cfg.training.gamma,
                    lmbda=self.cfg.training.lambda_ac,
                )
                advantage = (returns - imagined_v).detach()
                
                # Policy loss with entropy regularization
                policy_log_prob = action_dist.log_prob(imagined_a)
                policy_loss = -(advantage * policy_log_prob).mean()
                
                entropy_loss = -action_dist.entropy().mean()
                
                value_loss = F.mse_loss(imagined_v, returns)

                # ---------- Meta ELBO Loss ----------
                # Image Reconstruction Loss using LPIPS
                recon_img = self.model.perception_decoder(z0)
                # LPIPS expects images in [-1, 1] range. Assuming input is [0, 1].
                img_recon_loss = self.lpips_loss(recon_img * 2 - 1, screenshot * 2 - 1).mean()
                
                # KL Divergence Losses with annealing
                kl_z_loss = kl_divergence_gaussian(z_mu, z_logvar).mean()
                kl_u_loss = kl_divergence_gaussian(u_mu, u_logvar).mean()

                # Reward Reconstruction Loss (placeholder)
                reward_recon_loss = torch.tensor(0.0, device=self.device) # Needs real rewards to compare

                # ---------- Total Loss with Dynamic Weights ----------
                w = self.meta_net(z0, glb_goal).mean(0)

                loss = (
                    w[0] * policy_loss
                    + w[1] * value_loss
                    + self.cfg.entropy_coeff * entropy_loss
                    + w[2] * img_recon_loss
                    + w[3] * beta * kl_z_loss
                    + w[4] * beta * kl_u_loss
                    + w[5] * reward_recon_loss
                )
                
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
                self.opt.step()

                # Decay goal bank gates
                self.glb.decay_gates()

                # Log to TensorBoard
                if i % self.cfg.log_interval == 0:
                    global_step = epoch * len(self.dataloader) + i
                    self.writer.add_scalar("train/total_loss", loss.item(), global_step)
                    self.writer.add_scalar("train/policy_loss", policy_loss.item(), global_step)
                    self.writer.add_scalar("train/value_loss", value_loss.item(), global_step)
                    self.writer.add_scalar("train/entropy_loss", entropy_loss.item(), global_step)
                    self.writer.add_scalar("train/img_recon_loss", img_recon_loss.item(), global_step)
                    self.writer.add_scalar("train/kl_z_loss", kl_z_loss.item(), global_step)
                    self.writer.add_scalar("train/kl_u_loss", kl_u_loss.item(), global_step)

            print(f"epoch {epoch} loss={loss.item():.4f}")
            self._save_checkpoint(epoch) 