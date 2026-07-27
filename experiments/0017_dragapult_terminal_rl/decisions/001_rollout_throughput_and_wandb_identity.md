# Decision 001: Rollout throughput and W&B identity

Date: 2026-07-28

## Evidence

`V1_contract_probe` completed 260/260 official-engine games with no errors or discarded episodes.
It established a 38W-221L-1D baseline (14.62%) but measured only about 0.84 episodes/s with eight
workers across the full 26-opponent pool.

`V2_value_calibration` was stopped after 490/2,000 episodes before backward. The original batch
would have kept the GPU in low-utilization inference-only collection for roughly half an hour.
`V3_value_calibration_512` was then stopped after 10 episodes when the user requested a worker
throughput calibration before full training. Neither failed version produced a training checkpoint.

A representative 40-game benchmark used the exact CUDA centralized-inference collector, all 20
training opponents, balanced seats, isolated official-engine processes, and one CPU thread per
worker:

| Workers | Episodes/s | Decisions/s | Inference p50 ms | Errors |
|---:|---:|---:|---:|---:|
| 4 | 0.750 | 59.43 | 10.22 | 0 |
| 8 | 1.117 | 87.26 | 5.51 | 0 |
| 12 | 1.397 | 101.27 | 3.65 | 0 |
| 16 | 1.317 | 102.63 | 2.78 | 0 |

## Decision

- Use 12 rollout workers for the next formal run; 16 crossed the episode-throughput optimum.
- Use 512 episodes for `V4_value_calibration_512`, followed immediately by four value epochs.
- Use 256 fresh episodes per PPO update in `V5_ppo_pilot`, alternating collection and backward.
- Keep per-episode process isolation and single-threaded opponent inference for correctness.
- W&B display names use `<project number> · <project tag> · <version>`; group remains the full
  project ID and stable run ID remains derived from the full project ID and version.

These are throughput and observability changes. The terminal reward, action contract, model graph,
frozen opponent split, and model-only checkpoint contract are unchanged.
