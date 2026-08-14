# 0045 Single-Deck Expert Minimal-LoRA

Self-contained PPO specialist for exact deck `007` (Dragapult ex).

The focal Actor contains the frozen semantic backbone, final Option-block Q/V LoRA (`r=4`, `alpha=8`), Action Decoder, and Allocation Head. Critic outputs never enter Actor inference. The training-only Critic retains the 0044 Value/Prize/opponent-Meta topology.

Authoritative design and evidence:

- `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`
- `rl_runs/0045_single_deck_expert_minimal_lora/versions/V1_minimal_lora_dragapult_007/artifact/`

Verify current hard gates:

```bash
PYTHONPATH=. pytest -q train/0045_single_deck_expert_minimal_lora/tests
python3 -m train.0045_single_deck_expert_minimal_lora.training.run_v1
```

Formal long-run command (not launched automatically):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
python3 -m train.0045_single_deck_expert_minimal_lora.training.run_v1 \
  --launch-formal \
  --wandb-mode online \
  --u0-checkpoint rl_runs/0045_single_deck_expert_minimal_lora/versions/V1_minimal_lora_dragapult_007/checkpoint/update-000000.pt
```

The run uses immutable `Frozen-0045-Init` as its specialist KL reference and immutable `Champion-G2` as opponent. It collects 512 stochastic rollout games per update and runs deployment-effective greedy CUDA-512 Benchmark Tiny V2 every five saved updates; U10/U20/... also serve as the ten-update evaluation points.

All runtime imports resolve within 0045 or approved shared infrastructure. Copied 0044 assets retain provenance metadata, but no executable 0044 training module is imported.
