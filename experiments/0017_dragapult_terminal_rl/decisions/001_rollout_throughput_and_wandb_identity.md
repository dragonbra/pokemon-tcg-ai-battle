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

## PPO guard follow-up

V5 and V6 each stopped after update 1 because behavior KL reached 0.0494 and 0.0504. Reducing
actor LR from `1e-5` to `2e-6` did not resolve the breach. The implementation checked KL only
after all minibatches in an epoch, so roughly twenty optimizer steps could occur before stopping.
V7 checks behavior KL before every minibatch backward and rejects the remaining steps immediately
when the current policy exceeds 0.02. A 32-episode CUDA smoke completed 1,972 decisions and 124
minibatches with KL 0.000127, clip fraction 0.00260, ratio mean 0.99996, and no invalid episode.

V7 then showed KL 0.0493 on the first formal minibatch before any optimizer step. From V8 onward,
each update freezes an exact behavior snapshot and recomputes denominator log-prob in the same
training minibatch. Stored rollout log-prob remains only as `ppo/rollout_log_prob_mae`; the fixed
BC reference remains a separate long-horizon anchor.

V8 completed ten controlled updates at actor LR `2e-6`. Behavior KL stayed between roughly
`4e-6` and `1e-5`, clip fraction remained effectively zero, and the BC-reference surrogate reached
only 0.000326. Rolling-2,000 win rate ended at 10.5%, so the configuration proved the corrected PPO
contract but moved too slowly. V9 warm-starts from V8 update 10 with fresh optimizer/rollout and
raises actor LR to `1e-5`, retaining both frozen-behavior and minibatch-KL guards.
