# 0036 dedicated action Value network

The authoritative model, label and experiment contracts are in
`experiments/0036_dedicated_action_value_network/DESIGN.md`.

The admitted dataset is the all-date, recency-weighted Dragapult focal corpus described in the
design. Its source gzip/JSON shards remain the semantic audit artifact. Before training, convert
them once to the compact mmap format:

```bash
python3 -m train.0036_dedicated_action_value_network.training.materialized \
  --source rl_runs/0036_dedicated_action_value_network/datasets/dragapult_focal_all_dates_weighted_exact10k \
  --output rl_runs/0036_dedicated_action_value_network/datasets/dragapult_focal_all_dates_weighted_exact10k_tensor_v1 \
  --workers 4
```

Formal ablations use the same dataset and frozen source:

```bash
python3 -m train.0036_dedicated_action_value_network.run_ablation \
  --version V2_latent_value_archetype_diff --preset latent_value_archetype_diff \
  --dataset rl_runs/0036_dedicated_action_value_network/datasets/dragapult_focal_all_dates_weighted_exact10k_tensor_v1 \
  --source-dataset rl_runs/0036_dedicated_action_value_network/datasets/dragapult_focal_all_dates_weighted_exact10k \
  --source-checkpoint rl_runs/0036_dedicated_action_value_network/source/friend_0806_epoch11/model.pt
```

`V1_raw_pool_mlp` is a retained, interrupted I/O-preflight version and cannot be reused.
Subsequent ablations always receive the next unused version. Formal runs force W&B online
mirroring to the repository-standard private project while preserving local JSONL/checkpoints
if mirroring fails. Every epoch retains a model-only checkpoint.
