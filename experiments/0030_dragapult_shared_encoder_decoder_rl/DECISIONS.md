# 0030 experiment decisions

## 2026-08-04: V2 update 5 boundary and lightweight-worker transition

- `V2_trimmed_fp16_prototype_cache` was intentionally stopped only after update 5 had written both its model-only checkpoint and its balanced 102-game frozen greedy evaluation.
- The accepted branch checkpoint is `update-0005.pt`, SHA-256 `7065bdd8fa49472b64c871a1c4e470c58a390ed5710968550386e297d1b7a7bf`.
- Update 5 sampled rollout was 208-304 (40.62%). Its checkpoint's frozen greedy evaluation was 60-42 (58.82%), with 64.71% first-player and 52.94% second-player win rates. These are different contracts and are not interchangeable strength estimates.
- The launcher began spawning update 6 workers before the update 5 record was observed. The launcher and child were interrupted immediately; no update 6 PPO step, metric record, or checkpoint exists, and the incomplete rollout is discarded.
- Root-cause evidence showed that each official-engine worker imported Torch, CUDA, and NumPy because `rollout/__init__.py` eagerly imported `collector`. At 64 workers this dominated host RAM and caused swap pressure; feature compilation and model inference remain parent-only.
- The next formal run is `V3_lightweight_engine_workers`. It must begin from the V2 update-5 model-only weights with a fresh optimizer and newly collected on-policy episodes. Worker import isolation and official-engine concurrency benchmarks are required before launch.
