# 0030 experiment decisions

## 2026-08-04: 0019 and 0028 both enter the training opponent pool

- Every PPO update uses the same 51 exact frozen decks under two immutable shared-policy foundations: legacy 0019 Epoch-13 and semantic 0028.
- The 512-game schedule is exactly 256 games per foundation and, within each foundation, 128 focal-first plus 128 focal-second games. The focal policy always uses the current 0028 representation and trainable decoder/value.
- Both frozen foundations remain resident on the same GPU. Engine workers stay Torch-light and hold no model weights; there is one shared opponent service per foundation, never one model per deck.
- Every fifth update runs 102 fixed-seed greedy games per foundation. Metrics remain under separate `eval/foundation_0019/*` and `eval/foundation_0028/*` namespaces and are never combined.
- The 0019 evaluation rate is the checkpoint-selection anchor for comparison with 0022. The 0028 rate is retained independently to measure behavior against the newer foundation.
- The accepted V2 update-5 checkpoint established the transition baseline: 55-47 (53.92%) against 0019 and 61-41 (59.80%) against 0028, zero errors over 204 official-engine games.
- This changes the opponent curriculum relative to V2. V3 must therefore use a fresh optimizer and newly collected on-policy data even though it initializes decoder/value weights from V2 update 5.

## 2026-08-04: pre-launch mixed-routing performance gate

- The optimized collector completed the exact 512-game schedule in 195.14 seconds (2.624 games/s) at 128 workers: 256 games against each foundation, zero official-engine errors.
- Bulk GPU-to-CPU action materialization reduced rollout time from 229.84 seconds (2.228 games/s) to 195.14 seconds without changing action selection or feature schema. PPO still made zero representation calls.
- This is 2.3% faster than the recorded 0022 2.565 games/s reference. V3 can launch with 128 workers; 192 workers was rejected after a 256-game calibration fell to 1.958 games/s.

## 2026-08-04: V2 update 5 boundary and lightweight-worker transition

- `V2_trimmed_fp16_prototype_cache` was intentionally stopped only after update 5 had written both its model-only checkpoint and its balanced 102-game frozen greedy evaluation.
- The accepted branch checkpoint is `update-0005.pt`, SHA-256 `7065bdd8fa49472b64c871a1c4e470c58a390ed5710968550386e297d1b7a7bf`.
- Update 5 sampled rollout was 208-304 (40.62%). Its checkpoint's frozen greedy evaluation was 60-42 (58.82%), with 64.71% first-player and 52.94% second-player win rates. These are different contracts and are not interchangeable strength estimates.
- The launcher began spawning update 6 workers before the update 5 record was observed. The launcher and child were interrupted immediately; no update 6 PPO step, metric record, or checkpoint exists, and the incomplete rollout is discarded.
- Root-cause evidence showed that each official-engine worker imported Torch, CUDA, and NumPy because `rollout/__init__.py` eagerly imported `collector`. At 64 workers this dominated host RAM and caused swap pressure; feature compilation and model inference remain parent-only.
- The next formal run is `V3_lightweight_engine_workers`. It must begin from the V2 update-5 model-only weights with a fresh optimizer and newly collected on-policy episodes. Worker import isolation and official-engine concurrency benchmarks are required before launch.
