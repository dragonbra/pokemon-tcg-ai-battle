# 003 · V1 Throughput Baseline and V2 Worker Scaling

**Date:** 2026-07-31

**Decision:** Stop `V1_dragapult_focal_20h` after its first complete PPO update and launch a new immutable V2 with 128 Torch-light official-engine workers and 5 ms request coalescing.

## V1 complete boundary

V1's update-0 Frozen gate completed 96/96 games with zero errors in 162.07 seconds and a 51.04% greedy win rate. Its first sampled rollout completed 512/512 games with zero errors, 53,983 focal decisions, and produced `checkpoint/focal/update-000001.pt`. Provenance aligned exactly: `rollout/source_policy_update=0`, `trainer/update=1`, and `checkpoint/update=1`.

The PPO update was numerically healthy: behavior KL 0.0001603, clip fraction 0.00270, gradient norm 0.1982, actor relative L2 delta 0.002306, and no non-finite values. The sampled 28.91% rollout win rate is a behavior-policy diagnostic, not checkpoint-1 strength evidence.

The rollout required 803.39 seconds, or 0.6373 games/s. It issued 93,950 policy requests in 13,584 batches: mean batch 6.92, maximum 8. GPU peak allocation was about 2.49 GiB and the version occupied about 213 MiB, so neither GPU memory nor SSD was limiting. V1 was manually stopped after update 1; a partial update-2 in-memory rollout was discarded and no partial checkpoint was written. W&B synced successfully.

## Root cause

The initial formal command used 8 workers and 0.5 ms coalescing, starving the resident GPU service. A first 128-worker attempt then triggered the Linux OOM killer because Python `spawn` imported `train.0022_league_training.rollout.__init__`, which eagerly imported the Torch-heavy collector in every engine-only child.

The worker path was changed to lazy-load the parent-only collector and make protocol tensor annotations type-check-only. A subprocess regression test now requires importing `rollout.worker` without adding `torch` to `sys.modules`. The official-engine 4-game smoke remained 4/4 finished with zero errors after this change.

## Equal-contract scaling gate

All successful rows use the 0019 Epoch 13 Foundation, 256 official-engine sampled games, identical schedule/seed, 5 ms coalescing, focal trajectory retention, and zero engine errors. JSON artifacts are under `.tmp/evaluation/0022_worker_scaling/`.

| Workers | Finished | Games/s | Mean batch | Max batch | Parent max RSS | CUDA peak allocation |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 256/256 | 1.4081 | 53.21 | 64 | 2.26 GiB | 208.2 MiB |
| 128 | 256/256 | 2.4506 | 90.28 | 128 | 2.28 GiB | 334.5 MiB |
| 256 | 256/256 | 2.7039 | 140.48 | 256 | 2.27 GiB | 487.4 MiB |

The selection rule was the smallest configuration within 10% of best measured throughput. The 128-worker result is 90.63% of the 256-worker result, so it passes that rule while halving simultaneous engine process pressure. It is 3.85x faster than V1's actual 512-game collector rate. The 256-worker tier is only 10.33% faster than 128 and is not selected for the first long run.

## V2 contract

- 128 Torch-light official-engine workers;
- 5 ms GPU request coalescing;
- unchanged 512 games/update, balanced seats, deck schedule, Foundation, reward, PPO, Frozen evaluation, checkpoint retention, W&B, and SSD guards;
- SIGINT/SIGTERM records a parent stop request and completes the in-flight update before exit; engine children ignore the terminal SIGINT broadcast but retain SIGTERM for collector cleanup. Genuine exceptions still mark the version failed.

Changing worker/coalescing settings is a versioned throughput variable, so no additional training is appended to V1.
