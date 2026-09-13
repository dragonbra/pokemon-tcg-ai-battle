# Pokemon TCG AI: BC + RL policy

Self-contained PPO specialist for exact deck `007` (Dragapult ex).

The focal Actor contains the frozen semantic backbone, final Option-block Q/V LoRA (`r=4`, `alpha=8`), Action Decoder, and Allocation Head. Critic outputs never enter Actor inference. The training-only Critic retains the 0044 Value/Prize/opponent-Meta topology.

Authoritative design and evidence:

- `docs/model/DESIGN.md`
- `docs/model/DESIGN.html`
- `runs/versions/V1_minimal_lora_dragapult_007/artifact/`

Verify the policy hard gates:

```bash
PYTHONPATH=src:. pytest -q src/pokemon_tcg_ai/tests
python3 -m pokemon_tcg_ai.training.run_v1
```

Formal long-run command (not launched automatically):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src:. \
python3 -m pokemon_tcg_ai.training.run_v1 \
  --launch-formal \
  --wandb-mode online \
  --u0-checkpoint runs/versions/V1_minimal_lora_dragapult_007/checkpoint/update-000000.pt
```

The run uses immutable `Frozen-0045-Init` as its specialist KL reference and immutable `Champion-G2` as opponent. It collects 512 stochastic rollout games per update and runs deployment-effective greedy CUDA-512 Benchmark Tiny V2 every five saved updates; U10/U20/... also serve as the ten-update evaluation points.

All runtime imports resolve within `pokemon_tcg_ai` or approved shared infrastructure. Historical numeric identifiers remain only where they are part of immutable checkpoint, schema, or provenance contracts; no executable numbered training project is imported.
