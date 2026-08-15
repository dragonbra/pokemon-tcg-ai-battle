# 0047 Single-Deck Expert Minimal-LoRA

Self-contained PPO specialist for exact deck `007` (Dragapult ex).

The focal Actor contains the frozen semantic backbone, final Option-block Q/V LoRA (`r=4`, `alpha=8`), Action Decoder, and Allocation Head. Critic outputs never enter Actor inference. The training-only Critic retains the 0044 Value/Prize/opponent-Meta topology.

Authoritative design and evidence:

- `experiments/0047_meta_routed_moe_rl/DESIGN.md`
- `experiments/0047_meta_routed_moe_rl/DESIGN.html`
- `rl_runs/0047_meta_routed_moe_rl/versions/V1_minimal_lora_dragapult_007/artifact/`

Verify current hard gates:

```bash
PYTHONPATH=. pytest -q train/0047_meta_routed_moe_rl/tests
python3 -m train.0047_meta_routed_moe_rl.training.run_v1
```

Formal long-run command (not launched automatically):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
python3 -m train.0047_meta_routed_moe_rl.training.run_v1 \
  --launch-formal \
  --wandb-mode online \
  --u0-checkpoint rl_runs/0047_meta_routed_moe_rl/versions/V1_minimal_lora_dragapult_007/checkpoint/update-000000.pt
```

The run uses immutable `Frozen-0047-Init` as its specialist KL reference and immutable `Champion-G2` as opponent. It collects 512 stochastic rollout games per update and runs deployment-effective greedy CUDA-512 Benchmark Tiny V2 every five saved updates; U10/U20/... also serve as the ten-update evaluation points.

All runtime imports resolve within 0047 or approved shared infrastructure. Copied 0044 assets retain provenance metadata, but no executable 0044 training module is imported.
