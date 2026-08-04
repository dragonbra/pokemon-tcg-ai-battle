# 0030 experiment decisions

## 2026-08-04: Kaggle raw-exec fix and successful one-shot resubmissions

- Root cause of refs `55226701` and `55226710` was the package entrypoint's unconditional `Path(__file__)`: Kaggle's raw `exec(code_object, env)` loader does not define `__file__`. Local evaluation had used `importlib`, which supplied it and created a false-positive validation boundary.
- Added an isolated Kaggle-equivalent raw-exec gate and made the 0030 exporter normalize copied entrypoints to a CWD-based root with an explicit `/kaggle_simulations/agent` fallback. The exporter now fails closed before publishing a package if raw-exec initialization cannot return the exact deck.
- Preserved both failed archives and created new `kaggle_exec_fix` identities. Directory validation, extracted-tar validation, exact 60-card initialization, model hash preservation, member safety, 26 shared package tests, and 28 project tests all passed.
- Per user instruction, fixed U50 was submitted first exactly once as ref `55227026`, followed by fixed U81 exactly once as ref `55227052`; no automatic retry was issued. Both reached `COMPLETE`. Each initially reported `600.0`; at receipt finalization the live rolling scores were U50 `703.4` and U81 `698.2`, which are point-in-time observations rather than immutable offline metrics.
- Exact archive hashes, messages, timestamps, command counts, local reports, and Kaggle refs are recorded in `submission_receipt_u50_u81_kaggle_exec_fix.json`.

## 2026-08-04: U50 and U81 one-shot Kaggle submissions

- The two self-contained flat-root archives passed local package validation, exact 60-card initialization, archive traversal/symlink/forbidden-file checks, and exported-model hash verification before submission.
- Per explicit user instruction, U50 was submitted first and U81 second, with exactly one submission command per archive and no automatic retry. U50 is ref `55226701`; U81 is ref `55226710`.
- Both submissions reached terminal `ERROR` with Kaggle's only exposed diagnostic `Validation Episode failed.` and no public score. The one-shot constraint remains binding, so neither archive was resubmitted.
- The immutable receipt, archive hashes, exact messages, local Frozen 0019 results, and Kaggle timestamps are recorded in `submission_receipt_u50_u81.json`.

## 2026-08-04: V3 stopped at update 83 and checkpoint retention retired

- The user requested a normal stop after update 83 had fully written its model-only checkpoint and metrics. V3 is recorded as interrupted by `user_requested_stop`; no optimizer or rollout state is retained.
- Root-cause audit found that V3 configured `checkpoint_retention=6` and called a deletion function after every update. It preserved the best checkpoint plus recent checkpoints but deleted intermediate files, including requested updates 53 and 75. Those weights were not uploaded to W&B and cannot be reconstructed from scalar metrics.
- Surviving V3 weights at stop time are update 50 and updates 79-83. Update 50 already has a 510-game formal Frozen 0019 report. The high rollout record at `trainer/update=82` explicitly reports `rollout/source_policy_update=81`, so checkpoint 81, not checkpoint 82, is the selected recent comparison point; checkpoint 82 was produced only after consuming that rollout batch.
- This retention policy is retired. Subsequent 0030 runs must record `checkpoint_retention: all`, preserve every atomic model-only checkpoint, and contain no automatic checkpoint deletion path. Cleanup requires prior explicit user authorization.

## 2026-08-04: V3 update 5 independent Frozen 0019 package evaluation

- Exported `evaluation/arena/candidates/0030_dragapult_ex_001_v3_update5` from model-only checkpoint `15fd16529e55ba673a4f4e282181940d63b9ae0a9011af9a8053cb43b8a8baf0`. The self-contained actor package embeds decoder hash `bb28e23748c7493db21f1b5e6f12392834bed5536ff16892280e7e8fd4daa466` and exported model hash `81ef9630a52a211fd6985b3cef19b578c924ef52bcf38151ed09444fb910e2b1`.
- Standard package validation passed: exact 60-card `dragapult_ex_001`, compatible official `cg` tree, no symlinks, bytecode or optimizer state.
- An independent local evaluation used the standard 51-deck Frozen 0019 catalog, 10 games per deck, balanced seats, two persistent GPU inference services and isolated official-engine workers. It completed 510/510 games with zero errors or unfinished games.
- Result: 294-216 (57.65%); focal-first 155-100 (60.78%); focal-second 139-116 (54.51%). The approximate 95% binomial interval is 53.3%-61.9%.
- The independent estimate supports a real above-50% checkpoint but is 3.13 points below the training-time 102-game frozen snapshot (60.78%). Treat the 102-game value as a noisy checkpoint-selection signal; the current larger-sample strength estimate is approximately 57%-58%.
- Local report: `.tmp/evaluation/0030_update5_independent_frozen0019/run-b53bdf32b8734ab5b80333d413b1cce4/report.html`. This diagnostic ran concurrently with V3 training and did not replace either frozen evaluation namespace.

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
