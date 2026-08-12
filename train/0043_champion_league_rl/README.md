# 0043 Champion League RL

0043 implements the self-contained Opponent League and Promote Champion V2 project defined by:

- `docs/rl/0043_Project_Charter_Codex_Handoff.md`
- `docs/rl/0043_Promote_Champion_V2_Protocol.md`
- `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`

Current status is `READY_AWAITING_USER_LAUNCH`. The immutable opponent pool is exactly Policy-0809 plus Champion-G1; the mutable V1 focal seed lives only under `rl_runs/0043_champion_league_rl/versions/V1_focal_002_007/`. CUDA Engine 2.0, complete effective-policy isolation, 67-deck PFSP routing, own-deck 29-row initialization, frozen 15-way opponent Meta, PPO/W&B logging and launch readiness gates pass. No long rollout or PPO update has started.

## Readiness and launch commands

```bash
python3 -m train.0043_champion_league_rl.training.run_v1 \
  --readiness-output .tmp/evaluation/0043_initial_run_acceptance/readiness.json

# Run only after explicit user authorization:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m train.0043_champion_league_rl.training.run_v1 --launch-formal

python3 -m train.0043_champion_league_rl.import_assets --all
PYTHONPATH=. /home/cyd/.cache/uv/archive-v0/uIcEF-JYHod8LomN/bin/pytest -q train/0043_champion_league_rl/tests
PYTHONPATH=. python3 -m train.0043_champion_league_rl.preflight --gpu-smoke --output .tmp/evaluation/0043_preflight/report.json
python3 -m train.0043_champion_league_rl.cuda_engine_2.build print
python3 -m train.0043_champion_league_rl.cuda_engine_2.build build
python3 -m train.0043_champion_league_rl.cuda_engine_2.build smoke \
  --rule-pack .tmp/cuda_0032_rules/official_rules.bin \
  --output .tmp/evaluation/0043_cuda_engine_2/build_smoke.json
python3 -m train.0043_champion_league_rl.cuda_engine_2.smoke \
  --rule-pack .tmp/cuda_0032_rules/official_rules.bin \
  --output .tmp/evaluation/0043_cuda_engine_2/policy_transition_smoke.json
python3 -m train.0043_champion_league_rl.cuda_engine_2.performance \
  --old-binary engine_cuda/build/ptcg_cuda_smoke \
  --new-binary engine_cuda_2_0/build/native/ptcg_cuda_smoke \
  --extension-dir engine_cuda_2_0/build/native \
  --output .tmp/evaluation/0043_cuda_engine_2/performance.json
```

The import command is the only code allowed to read the approved historical sources. It freezes both immutable assets and the semantic inference source into 0043. Runtime consumers use only project-local registries, semantic policy directories and `semantic_runtime/`.

## Asset domains

- `assets/decks/`: 67 immutable exact-deck definitions with canonical IDs and directories `001`–`067`. FrozenMeta256-V1 remains the original `001`–`055`; `056`–`067` are training-only until a separately versioned evaluation contract admits them.
- `assets/policies/`: complete Policy-0809 base plus Champion-G1 delta/portable artifacts under semantic directories `policy_0809` and `champion_g001`.
- `assets/evaluation/`: FrozenMeta256-V1 composition and its versioned seed contract metadata.

The implementation plan is `docs/superpowers/plans/2026-08-12-0043-champion-league-rl.md`; the authoritative current design and audit are under `experiments/0043_champion_league_rl/`.

The 0043 build entrypoint intentionally builds only `ptcg_cuda_smoke` and `_ptcg_cuda`, always with `--parallel 1`. CUDA 12.8 compilation of one optional continuation diagnostic translation unit was observed at roughly 30 GiB RSS, so the CMake `all` target is not part of the safe 0043 build path. This restriction does not remove a production runtime target.

The production performance contract keeps `official_engine_kernels.cu`, `semantic0031_bridge.py`, and `semantic0031_resident.py` byte-identical to the audited 0042 path. Features enter through the device-resident `encode_semantic0031_v2_lanes` API and are consumed directly as CUDA `DecisionBatch` tensors; importing the CPU canonical feature compiler from the CUDA integration is a hard failure.

SHA-256 values remain mandatory registry integrity fields. They are not deck IDs, policy IDs, or meaningful folder names.
