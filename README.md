# GIA-Warp: Toy-Story Edition (Central Brain for Multi-Robot Coordination)

This repository contains the **Toy-Story Edition** of the Grounded Interface Agent (GIA-Warp), a framework for a central "brain" that learns to coordinate a team of simple, physical robots to perform complex tasks.

The core is a **Dreamer-style world model** that fuses multi-robot sensory input into a shared latent space. This "central brain" uses a novel set of **Warp modules** to dynamically reshape its internal understanding of time, space, and goals, enabling efficient real-time planning and collaboration.

The original GIA-Warp, a SOTA UI automation agent, forms the foundation. This edition extends it to the physical world, managing multiple edge devices (robots) from a single, powerful model.

[![Lint](https://github.com/your-org/GroundedInterfaceAgent/actions/workflows/lint.yml/badge.svg)](https://github.com/your-org/GroundedInterfaceAgent/actions/workflows/lint.yml)

## Key Concepts

- **Central Brain Architecture**: A single, powerful `GiaAgent` model runs on a central server, processing latent state information streamed from multiple, simpler "edge" robots.
- **Shared Latent Workspace (SLW)**: A Transformer that fuses real-time observations (`REAL` tokens) from robots with imagined future trajectories (`IMAG` tokens) from the world model.
- **Meta-Cognitive Dreamer**: The core learning engine. It includes:
    - **World Model**: Learns a predictive model of the environment's dynamics in a latent space (`z_t`).
    - **Value Function (`ValueHead`)**: Estimates the long-term value (expected future rewards) of any given state `z_t`. This is critical for the policy to make far-sighted decisions.
    - **Dynamic Reward Model (`HyperRewardHead`)**: A hyper-network that learns the reward function itself, conditioned on a latent "value state" (`u_t`).
- **Dynamic Goal Management**: A `GoalLatentBank` acts as a differentiable memory for task goals, which can be updated both by external instruction (top-down) and by the agent's own discoveries (bottom-up).
- **Advanced Warp Modules**: A suite of learned, invertible functions that warp the latent space for more efficient learning and control.

---

## Transformer-Dreamer Warp with Human Feedback

This release integrates the **Unified Latent-World Transformer (ULWT)** and a
self-supervised **Human-Feedback Predictor (HFP)** so that the agent can
leverage Like/Dislike signals while keeping real-time Dreamer planning.

### Quick Start
```bash
# install extra deps
pip install -r requirements.txt  # adds flask, streamlit, gymnasium, pytest

# run Like/Dislike HTTP server (port 8001)
python -m environments.human_feedback_interface &

# launch attention dashboard (optional)
streamlit run dashboard/attention_dashboard.py &

# training with ULWT config
python main.py model=gia_tse_ulwt
```

The new config `configs/model/gia_tse_ulwt.yaml` enables:
* Reward token embedding + ULWT sequence builder
* Human-Feedback BCE & self-labeling (`pseudo_threshold`, `pseudo_lambda`)
* Curriculum scheduling for `w_hfp`
* KL style regularizer against frozen BC policy

See `engine/dreamer_trainer.py` for implementation details.

##  Toy-Story Edition: Quick Start

This workflow runs the central brain server and connects a dummy robot client.

### 1. Environment Setup
```bash
# Install base requirements
pip install -r requirements.txt

# Install Toy-Story specific dependencies
pip install cbor2
```

### 2. Launch Central Brain
In one terminal, start the streaming server. This loads the `GiaAgent` and waits for robot connections.
```bash
# The server listens on 0.0.0.0:8765 by default
python -m engine.streaming_runner
```

### 3. Connect a Dummy Robot Client
In a second terminal, run the reference client. It will send random latent vectors and receive action commands from the central brain.
```bash
# This script simulates a robot sending its state and receiving actions
python - <<'PY'
from communication.bridge_client import BridgeClient
import time, random
cli = BridgeClient(robot_id="bot1")
print("Connecting to central brain...")
for t in range(32):
    # In a real robot, `z` would be encoded from sensors
    z = [random.random() for _ in range(4096)]
    print(f"Tick {t}: Sending REAL latent state...")
    cli.send_latent(z, token_type=0, delta_tau=0.0)
    act = cli.recv_action()
    print(f"Tick {t}: Received action -> {act.action[:4]}")
    time.sleep(0.05)
PY
```

### 4. Online RL Training
To train the central brain using data collected from robots (or datasets), run the `train_streaming` mode. This uses the `StreamingTrainer` which incorporates the `GoalLatentBank` and `MetaLossNet`.
```bash
python main.py mode=train_streaming \
  --config-path configs/training \
  --config-name streaming_rl
```

---

## Architecture Overview

The Toy-Story Edition extends the GIA-Warp agent with modules for multi-robot communication and coordination.

```mermaid
graph TD
    %% Edge Robots (Multiple)
    subgraph "Edge Robots (Multiple)"
        direction LR
        Robot1_Sensors --> R1_Encoder[Encoder]
        Robot2_Sensors --> R2_Encoder[Encoder]
        R1_Encoder -->|"z_real (bot1)"| StreamingServer
        R2_Encoder -->|"z_real (bot2)"| StreamingServer
    end

    %% Central Brain (Server)
    subgraph "Central Brain (Server)"
        direction TB

        %% A. Latent Space Fusion & World Model
        subgraph "A. Latent Space Fusion & World Model"
            StreamingServer[Streaming Server] --> SLW[SLW-Transformer]
            WorldModel_Dreamer[Dreamer World Model] -->|"z_imag, u_imag"| SLW
        end

        %% B. Policy & Value Estimation
        subgraph "B. Policy & Value Estimation"
            SLW --> Fused_Context
            Goal[Goal Latent Bank] --> Fused_Context
            Fused_Context --> Policy[Actor-Critic]
            Fused_Context --> ValueHead[Value Function V(z)]
            Policy --> Action_Vecs
        end

        %% C. Dynamic Learning & Reward
        subgraph "C. Dynamic Learning & Reward"
            MetaLoss[Meta Loss Net] -->|"loss weights"| WorldModel_Dreamer
            Fused_Context --> MetaLoss
            SLW --> HyperReward[Hyper Reward Head r(z,u)]
        end

        Action_Vecs --> StreamingServer
    end

```

### Component Map

| Component | File | Purpose |
|---|---|---|
| 3D-SpaceWarp++ | `models/warp_modules.SpaceWarp3D` | Inject `(x,y,z,h_e)` spatial bias |
| Uncertainty-TimeWarp | `models/warp_modules.TimeWarpUnc` | Time resolution ∝ sensor noise |
| Tasked Goal-Warp | `models/warp_modules.GoalWarpTasked` | Real-NVP conditioned on task-ID |
| Counterfactual-ΔWarp | `models/warp_modules.DeltaWarpCF` | Fast latent shift for what-if sims |
| SLW-Transformer | `models/slw_transformer.py` | REAL/IMAG token fusion |
| Goal Latent Bank | `models/goal_bank.py`

## Long-Term State Mechanisms (v0.4)

The June-2025 release introduces *multi-time-scale context* for better task continuity across long GUI episodes:

1. **Slow-State Transformer (SST)** – aggregates hundreds of latent states into a `slow_state` embedding that is
   fused back into the Dreamer world model each step.
2. **Hierarchical Memory** – two-tier k-NN store (episodic RAM + semantic on-disk) enabling cross-session recall.
3. **Persona Vector** – EMA of `slow_state` capturing stable traits; **self-consistency loss** encourages the agent’s
   recent behaviour to align with this persona.
4. **RSSM Persistence** – latest `(z,u)` latents + SST buffer are checkpointed (`*_latents.pt`) and auto-restored.

### Configuration Quick-Start
```yaml
# configs/model/gia_8b.yaml
sst:
  embed_dim: 4096
  n_layers: 6
  max_len: 512
memory:
  type: hierarchical
  embed_dim: 128
  episodic_max: 10000
persona_tau: 0.995  # EMA rate
```

### Training Loss Weights
```yaml
# configs/training/pretrain_grounding.yaml
loss_weights:
  self_consistency: 0.05  # 1-cos similarity
```

### Checkpoint Files
| File | Contents |
|------|-----------|
| `best.pt` | Model weights & metadata |
| `best_latents.pt` | `z`, `u`, `sst_buffer` |
| `semantic_memory.pt` | Semantic memory vectors |

For details see `docs/LongTermState.md`.