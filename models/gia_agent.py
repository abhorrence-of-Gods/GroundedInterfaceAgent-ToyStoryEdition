import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from torch.utils import checkpoint as cp
import time
import os

from .dreamer.meta_transition_model import MetaTransitionModel
from .dreamer.hyper_reward_head import HyperRewardHead
from .goal_bank import GoalLatentBank
from .meta_loss_net import MetaLossNet
from .slw_transformer import SLWTransformer
from .towers.macro_action_tower import MacroActionTower
from models.transformers import UnifiedLatentWorldTransformer
from models.expressive.led_controller import LEDController  # optional use
from models.expressive.voice_controller import VoiceController  # optional use
from .memory_store import MemoryStore
from .hierarchical_memory import HierarchicalMemory
from .sst_transformer import SlowStateTransformer  # NEW: long-range context module
from models.warp_modules import GoalWarp, GoalWarpTasked, DeltaWarpCF, SpaceWarp, TimeWarp  # warp modules

class GiaAgent(nn.Module):
    """
    The GIA Agent, Toy-Story Edition.
    A central brain featuring a meta-cognitive dreamer for multi-robot control.
    """
    def __init__(self, model_config: DictConfig):
        super().__init__()
        # --- World Model Components (Encoders/Decoders) ---
        self.language_tower: LanguageTower = hydra.utils.instantiate(model_config.language_tower)
        self.perception_tower: PerceptionTower = hydra.utils.instantiate(model_config.perception_tower)
        self.action_tower: ActionTower = hydra.utils.instantiate(model_config.action_tower)
        self.bridge: GroundingBridge = hydra.utils.instantiate(model_config.bridge)
        self.spacetime_encoder: SpacetimeEncoder = hydra.utils.instantiate(model_config.spacetime_encoder)
        self.perception_decoder: PerceptionDecoder = hydra.utils.instantiate(model_config.perception_decoder)
        self.spacetime_decoder: SpacetimeDecoder = hydra.utils.instantiate(model_config.spacetime_decoder)

        # --- Meta-Cognitive Dreamer Components (Toy-Story Edition) ---
        self.meta_transition_model: MetaTransitionModel = hydra.utils.instantiate(model_config.meta_transition_model)
        self.hyper_reward_head: HyperRewardHead = hydra.utils.instantiate(model_config.hyper_reward_head)
        self.value_head: ValueHead = hydra.utils.instantiate(model_config.value_head) # Critic remains
        self.macro_action_tower: MacroActionTower = hydra.utils.instantiate(model_config.macro_action_tower)

        # --- Goal & Meta-Learning Components ---
        self.goal_bank = GoalLatentBank(num_slots=32, goal_dim=16)
        self.meta_loss_net = MetaLossNet()
        self.slw_transformer = SLWTransformer()
        # NEW: Slow-State Transformer (long-term)
        sst_cfg = model_config.get("sst", {})
        self.sst_transformer = SlowStateTransformer(**sst_cfg)  # default args if none provided
        self._sst_buffer: list[torch.Tensor] = []  # will store past latent states
        self.sst_max_len = sst_cfg.get("max_len", 512)
        # Projection from SST slow_state to latent state dimension
        self.slow_state_proj = nn.Linear(self.sst_transformer.embed_dim, model_config.bridge.hidden_dim)
        # Current aggregated slow state (initially zeros)
        self.register_buffer("slow_state", torch.zeros(1, self.sst_transformer.embed_dim))
        # Persona vector (EMA of slow_state)
        self.register_buffer("persona_vec", torch.zeros(1, self.sst_transformer.embed_dim))
        init_tau = float(model_config.get("persona_tau", 0.995))
        # Learnable τ parameter in unconstrained space; sigmoid→(0,1)
        self._logit_tau = nn.Parameter(torch.logit(torch.tensor(init_tau)))
        self._sst_min_compute = max(4, self.sst_max_len // 8)
        # NEW: Unified Latent-World Transformer for reward+feedback integration
        ulwt_cfg = model_config.get("ulwt", {})  # expects DictConfig or dict
        self.ulwt = UnifiedLatentWorldTransformer(**ulwt_cfg)  # default args if not specified
        # Projection from latent state dimension to ULWT embed dim for HFP when tokens are unavailable
        self.latent_token_proj = nn.Linear(model_config.bridge.hidden_dim, self.ulwt.embed_dim)

        # Projection for conditioning low-level policy on subgoals
        self.subgoal_projection = nn.Linear(
            model_config.macro_action_tower.subgoal_dim,
            model_config.bridge.hidden_dim
        )

        # Encoder for initial value latent u_0 from z_0
        self.value_latent_encoder = nn.Sequential(
            nn.Linear(model_config.bridge.hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, model_config.meta_transition_model.value_latent_dim * 2) # mu and logvar
        )

        self.action_projection = nn.Linear(self.action_tower.latent_action_dim, model_config.bridge.hidden_dim)

        # ---------------- Warp Modules ---------------------------
        warp_cfg = model_config.get("warp", {})
        warp_type = warp_cfg.get("type", "goal")  # goal, goal_tasked, delta, none
        if warp_type == "goal":
            self.goal_warp = GoalWarp(
                latent_dim=model_config.bridge.hidden_dim,
                goal_dim=warp_cfg.get("goal_dim", 16),
                num_flows=warp_cfg.get("num_flows", 4),
                hidden=warp_cfg.get("hidden", 1024),
            )
        elif warp_type == "goal_tasked":
            self.goal_warp = GoalWarpTasked(
                latent_dim=model_config.bridge.hidden_dim,
                goal_dim=warp_cfg.get("goal_dim", 16),
                n_tasks=warp_cfg.get("n_tasks", 10),
                num_flows=warp_cfg.get("num_flows", 4),
                hidden=warp_cfg.get("hidden", 1024),
            )
        elif warp_type == "delta":
            self.goal_warp = DeltaWarpCF(latent_dim=model_config.bridge.hidden_dim)
        else:
            # Identity warp if disabled
            self.goal_warp = nn.Identity()

        # Optional coordinate/time warps (additive bias)
        if warp_cfg.get("use_spacewarp", False):
            self.space_warp = SpaceWarp(output_dim=model_config.bridge.hidden_dim)
        else:
            self.space_warp = None
        if warp_cfg.get("use_timewarp", False):
            self.time_warp = TimeWarp(output_dim=model_config.bridge.hidden_dim)
        else:
            self.time_warp = None

        # --- Emotion latent ---------------------------------------------------------
        self.emotion_dim = model_config.get("emotion_dim", 32)
        self.emotion_proj = nn.Linear(model_config.bridge.hidden_dim, self.emotion_dim)
        self.emotion_to_state = nn.Linear(self.emotion_dim, model_config.bridge.hidden_dim)

        # Optional expressive controllers (not mandatory in forward path)
        self.led_controller = LEDController(self.emotion_dim)
        self.voice_controller = VoiceController(self.emotion_dim)

        # --- Memory Store (Long-Term) --------------------------------------------------
        mem_cfg = model_config.get("memory", {})
        if mem_cfg:
            mem_type = mem_cfg.get("type", "flat")  # 'flat' or 'hierarchical'
            if mem_type == "hierarchical":
                self.memory = HierarchicalMemory(
                    embed_dim=mem_cfg.get("embed_dim", 128),
                    episodic_max=mem_cfg.get("episodic_max", 10000),
                    semantic_db_path=mem_cfg.get("semantic_db_path", "semantic_memory_store.pt"),
                    flush_size=mem_cfg.get("flush_size", 1024),
                )
            else:
                self.memory = MemoryStore(
                    embed_dim=mem_cfg.get("embed_dim", 128),
                    db_path=mem_cfg.get("db_path", "memory_store.pt"),
                    flush_size=mem_cfg.get("flush_size", 1024),
                )
            # key projection from latent h  to memory key space
            self.key_proj = nn.Linear(model_config.bridge.hidden_dim, self.memory.embed_dim)
            self.memory_topk = mem_cfg.get("topk", 16)
        else:
            self.memory = None
            self.key_proj = nn.Linear(model_config.bridge.hidden_dim, 128)
            self.memory_topk = 0

        # Differentiable Episodic Memory (small GPU buffer)
        diff_mem_cfg = mem_cfg.get("differentiable", {}) if mem_cfg else {}
        self.diff_mem_max = diff_mem_cfg.get("max_entries", 512)
        diff_key_dim = self.key_proj.out_features if mem_cfg else 128
        self.register_buffer("diff_keys", torch.empty(0, diff_key_dim))
        self.register_buffer("diff_vals", torch.empty(0, model_config.bridge.hidden_dim))

        # --- Device & Precision Management ---
        self.device = next(self.language_tower.parameters()).device
        # ... (move all modules to device)
        self.to(self.device)

        # --- Warp modules ---
        # ... (warp module initializations remain the same) ...

        self._use_grad_ckpt = bool(model_config.get("use_grad_checkpoint", False))
        self._latest_z: torch.Tensor | None = None  # last RSSM latent state
        self._latest_u: torch.Tensor | None = None  # last value latent state

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from a Gaussian."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _update_slow_state(self):
        """Compute aggregated slow_state from the SST buffer and cache it."""
        if len(self._sst_buffer) < self._sst_min_compute:
            # Not enough history yet; keep previous slow_state
            return
        # Build sequence tensor (L, D) and add batch dim
        seq = torch.stack(self._sst_buffer, dim=0).unsqueeze(0).to(self.slow_state.device)
        with torch.no_grad():
            out = self.sst_transformer(seq)
            self.slow_state = out["slow_state"].detach()

    def encode_state(
        self,
        instruction_text: list[str],
        screenshot: torch.Tensor,
        goal_vec: torch.Tensor | None = None,
        # ... other args
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Encodes raw observation into a pair of latent states (z_t, u_t)
        and their distribution parameters.
        Returns: z, u, (z_mu, z_logvar), (u_mu, u_logvar)
        """
        if self._use_grad_ckpt:
            # ...
            z_mu, z_logvar = cp.checkpoint(lambda img: self.perception_tower(img), screenshot)
        else:
            # ...
            z_mu, z_logvar = self.perception_tower(screenshot)

        # Reparameterize to get stochastic z
        z_t = self._reparameterize(z_mu, z_logvar)

        # ... (language tower, bridge, warps) ...
        # Note: apply warps to the sampled z_t
        latent_state_warped = self.goal_warp(z_t, goal_vec) # example

        # Compute emotion latent e_t from state (no grad to emotion for now)
        e_t = torch.tanh(self.emotion_proj(latent_state_warped.detach()))  # (B,E)

        # Finally, encode initial value latent u_0 from the final z_0
        u_params = self.value_latent_encoder(latent_state_warped.detach())
        u_mu, u_logvar = torch.chunk(u_params, 2, dim=-1)
        u_t = self._reparameterize(u_mu, u_logvar)

        # -------------------- Differentiable Episodic Memory ------------------
        if self.diff_keys.numel() > 0:
            # cosine similarity attention over diff_keys (N,K) vs key_vec (B,K)
            q = torch.nn.functional.normalize(self.key_proj(latent_state_warped), dim=-1)  # (B,K)
            k = torch.nn.functional.normalize(self.diff_keys.to(q.device), dim=-1)  # (N,K)
            sim = torch.matmul(q, k.t())  # (B,N)
            attn = torch.softmax(sim, dim=-1)  # (B,N)
            mem_vals = self.diff_vals.to(latent_state_warped.device)  # (N,D)
            m_diff = torch.matmul(attn, mem_vals)  # (B,D)
            latent_state_warped = latent_state_warped + m_diff

        # -------------------- Long-Term Memory Integration (non-diff) ---------
        if self.memory is not None:
            # 1. Build key vector (B, K)
            key_vec = torch.nn.functional.normalize(self.key_proj(latent_state_warped.detach()), dim=-1)
            # 2. Search memory and build aggregated vector m_t per sample
            m_vec = []
            for i in range(key_vec.size(0)):
                results = self.memory.search(key_vec[i], topk=self.memory_topk)
                if results:
                    vals = torch.stack([
                        torch.as_tensor(r[0]["h"], device=latent_state_warped.device, dtype=latent_state_warped.dtype)
                        if not isinstance(r[0]["h"], torch.Tensor) else r[0]["h"].to(latent_state_warped.device)
                        for r in results
                    ], dim=0)
                    # Emotion values: may be missing in older records
                    e_vals = torch.stack([
                        torch.as_tensor(r[0].get("e", torch.zeros(self.emotion_dim)), device=e_t.device, dtype=e_t.dtype)
                        if not isinstance(r[0].get("e"), torch.Tensor) else r[0]["e"].to(e_t.device)
                        for r in results
                    ], dim=0)
                    e_vals_state = self.emotion_to_state(e_vals)  # project to state dim
                    vals = vals + e_vals_state
                    dists = torch.tensor([r[1] for r in results], device=latent_state_warped.device, dtype=latent_state_warped.dtype)
                    attn = torch.softmax(-dists, dim=0).unsqueeze(-1)  # (k,1)
                    m_i = (attn * vals).sum(dim=0)
                else:
                    m_i = torch.zeros_like(latent_state_warped[i])
                m_vec.append(m_i)
            m_vec = torch.stack(m_vec, dim=0)
            # Simple additive fusion
            latent_state_warped = latent_state_warped + m_vec

            # 3. Write current step to memory (detach to avoid grads)
            ts_now = time.time()
            value_dicts = [
                {
                    "h": latent_state_warped[i].detach().cpu(),
                    "e": e_t[i].detach().cpu(),
                    "ts": ts_now,
                }
                for i in range(latent_state_warped.size(0))
            ]
            self.memory.write(key_vec.detach().cpu(), value_dicts)

        # Update differentiable memory buffer with current step (detach values to stop gradient through memory)
        new_keys = self.key_proj(latent_state_warped).detach()  # (B,K)
        new_vals = latent_state_warped.detach()  # (B,D)
        self.diff_keys = torch.cat([self.diff_keys, new_keys.cpu()], dim=0)[-self.diff_mem_max:]
        self.diff_vals = torch.cat([self.diff_vals, new_vals.cpu()], dim=0)[-self.diff_mem_max:]

        # -------------------- Slow-State Transformer Buffer ----------------------
        # Store the latest latent state for long-range temporal modelling.
        # We detach to avoid backprop through historical states.
        if self.sst_transformer is not None:
            # `latent_state_warped` shape: (B, D). We store per-sample tensors.
            # To support batched observations, we split along batch dim.
            for i in range(latent_state_warped.size(0)):
                self._sst_buffer.append(latent_state_warped[i].detach())
            # Truncate buffer if it exceeds capacity (measured in *steps*, not batches).
            if len(self._sst_buffer) > self.sst_max_len:
                over = len(self._sst_buffer) - self.sst_max_len
                self._sst_buffer = self._sst_buffer[over:]

        # Update slow_state after modifying buffer
        if self.sst_transformer is not None:
            self._update_slow_state()
            # ----- Persona update (EMA) -----
            with torch.no_grad():
                tau_val = torch.sigmoid(self._logit_tau)
                self.persona_vec = self.persona_vec * tau_val + self.slow_state.detach() * (1 - tau_val)

        # Fuse slow_state into current latent state
        if self.sst_transformer is not None:
            latent_state_warped = latent_state_warped + self.slow_state_proj(self.slow_state).to(latent_state_warped.device)

        # After computations, cache latest latents
        self._latest_z = latent_state_warped.detach()
        self._latest_u = u_t.detach()
        return latent_state_warped, u_t, (z_mu, z_logvar), (u_mu, u_logvar)

    def plan_in_dream(self, initial_state_z: torch.Tensor, initial_state_u: torch.Tensor, macro_horizon: int = 3, micro_horizon: int = 5):
        """
        Hierarchical planning in dream. Proposes a subgoal, then executes.
        """
        z_list, u_list, a_list = [], [], []
        current_z, current_u = initial_state_z, initial_state_u
        main_goal, _ = self.goal_bank(current_z.detach())

        for _ in range(macro_horizon):
            # 1. Planner (MacroActionTower) proposes a subgoal
            g_sub, _ = self.macro_action_tower(current_z.detach(), main_goal.detach())
            g_sub_proj = self.subgoal_projection(g_sub)

            # 2. Controller (ActionTower) executes micro-actions
            for _ in range(micro_horizon):
                # Condition the low-level policy on the projected subgoal
                state_for_policy = current_z + g_sub_proj
            action_dist = self.action_tower.get_action_dist(state_for_policy)
            latent_action = action_dist.rsample()

            z_list.append(current_z)
            u_list.append(current_u)
            a_list.append(latent_action)

            # World model dynamics are conditioned on the main goal
            next_z, next_u = self.meta_transition_model(
                current_z, current_u, latent_action, main_goal
            )

            current_z, current_u = next_z, next_u

        imagined_z = torch.stack(z_list)
        imagined_u = torch.stack(u_list)
        imagined_a = torch.stack(a_list)

        imagined_rewards = self.hyper_reward_head(imagined_z, imagined_u)
        imagined_values = self.value_head(imagined_z).mean

        return imagined_z, imagined_a, imagined_rewards, imagined_values

    def forward(self, instruction_text: list[str], screenshot: torch.Tensor, target_action: torch.Tensor, target_warp: torch.Tensor, target_goal: torch.Tensor | None = None):
        """
        A grand forward pass for SUPERVISED PRE-TRAINING.
        This does not use the meta-cognitive dreamer loop.
        """
        # ... (this part of the code remains largely unchanged for now)
        # It's used for imitation/generative learning, not RL.
        # ... (rest of the forward method)

    def predict_human_feedback(self, obs_act_tokens: torch.Tensor, reward_prev: torch.Tensor) -> torch.Tensor:
        # Utility wrapper to get like-probability from ULWT.
        with torch.no_grad():
            out = self.ulwt(obs_act_tokens, reward_prev)
        return out["hfp_prob"]

    def predict_human_feedback_from_latent(self, z_latent: torch.Tensor, reward_prev: torch.Tensor) -> torch.Tensor:
        """Predict like-probability from a latent state z when full token sequence is unavailable."""
        token = self.latent_token_proj(z_latent).unsqueeze(1)  # (B,1,E)
        out = self.ulwt(token, reward_prev)
        return out["hfp_prob"]

    # ------------------------------ Persistence API ------------------------------
    def save_latent_state(self, filepath: str):
        """Save the latest RSSM latent states and SST buffer to disk."""
        state_dict: dict[str, torch.Tensor] = {}
        if self._latest_z is not None and self._latest_u is not None:
            state_dict["z"] = self._latest_z.cpu()
            state_dict["u"] = self._latest_u.cpu()
        # save SST buffer if non-empty
        if self._sst_buffer:
            state_dict["sst_buffer"] = torch.stack(self._sst_buffer).cpu()
        torch.save(state_dict, filepath)

    def load_latent_state(self, filepath: str, map_location: str | torch.device | None = None):
        """Load RSSM latent states and SST buffer from disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)
        state_dict = torch.load(filepath, map_location=map_location if map_location is not None else self.device)
        self._latest_z = state_dict.get("z")
        self._latest_u = state_dict.get("u")
        if "sst_buffer" in state_dict:
            self._sst_buffer = [t.to(self.device) for t in state_dict["sst_buffer"]]
            # keep within max_len
            if len(self._sst_buffer) > self.sst_max_len:
                self._sst_buffer = self._sst_buffer[-self.sst_max_len:]
            self._update_slow_state()
        # Ensure tensors on device
        if self._latest_z is not None:
            self._latest_z = self._latest_z.to(self.device)
        if self._latest_u is not None:
            self._latest_u = self._latest_u.to(self.device)

    # -------------------- Persona API --------------------
    def get_persona_vector(self, batch_size: int = 1) -> torch.Tensor:
        """Return persona vector repeated to batch size."""
        return self.persona_vec.to(self.device).repeat(batch_size, 1)

    # -------------------- Learnable Tau --------------------
    @property
    def persona_tau(self) -> torch.Tensor:
        """Return τ in (0,1) as sigmoid(logit_tau)."""
        return torch.sigmoid(self._logit_tau)
