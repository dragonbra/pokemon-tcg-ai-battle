# Pokemon TCG AI: BC + RL policy

Self-contained PPO specialist for exact deck `007` (Dragapult ex).

The focal Actor contains the frozen semantic backbone, rank-16 attention/FFN LoRA, final Option LayerNorm, Action Decoder, and Allocation Head. Critic outputs never enter Actor inference. The training-only Critic retains the Value/Prize/opponent-Meta topology.

Public entry points:

- `pokemon_tcg_ai.model`: model definitions and Actor/Critic ownership
- `pokemon_tcg_ai.training.train`: reproducible Policy-0814 → PPO trainer
- `pokemon_tcg_ai.inference`: portable and public-routed inference
- `pokemon_tcg_ai.evaluation`: deployment and benchmark contracts

Verify the policy hard gates:

```bash
PYTHONPATH=src:. pytest -q src/pokemon_tcg_ai/tests
python3 -m pokemon_tcg_ai.training.train
```

Formal long-run command (not launched automatically):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:. \
python3 -m pokemon_tcg_ai.training.train \
  --launch-formal \
  --wandb-mode online \
  --updates 10
```

The public run builds U0 directly from the committed Policy-0814 Actor and Value assets, uses that
immutable U0 as its KL reference, and trains against independently materialized Policy-0814
opponents. It collects 512 stochastic official-engine games per update and runs the fixed greedy
CUDA-512 exact-deck evaluation every five saved updates.

All runtime imports resolve within `pokemon_tcg_ai` or approved shared infrastructure. Historical numeric identifiers remain only where they are part of immutable checkpoint, schema, or provenance contracts; no executable numbered training project is imported.
