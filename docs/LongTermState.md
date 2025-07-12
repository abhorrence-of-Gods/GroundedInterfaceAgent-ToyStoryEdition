# Long-Term State & Memory Architecture

This document describes the new modules enabling **multi-episode reasoning**.

```mermaid
graph TB
  subgraph Fast Loop (25-30 Hz)
    A[Dreamer RSSM z<sub>t</sub>] -->|append| B[SST Buffer]
    A --> F[Key Projection]
    F --> G[Hierarchical Memory]
    G -->|m_t| A
  end

  subgraph Slow Loop (~1 Hz)
    B --> C[Slow-State Transformer]
    C -->|slow_state s_t| A
    C --> D[Persona EMA]
  end

  B -. flush episodic -> G
  D -->|self-consistency loss| Loss
```

### Components
• **SST Buffer**: Rolling window of recent latent states (length ≤ `max_len`).
• **Slow-State Transformer**: Transformer encoder with optional CLS; outputs `s_t`.
• **Persona EMA**: `persona_vec ← τ·persona_vec + (1-τ)·s_t`.
• **Hierarchical Memory**: Episodic RAM store & semantic disk store.

### Training Objective
`L_total = L_base + w_uncert·L_uncertainty + w_sc·(1 - cos(s_t , persona_vec)) + …`

See `configs/model/gia_8b.yaml` and `configs/training/*` for default hyper-parameters. 