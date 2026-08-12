# 0043 Champion League RL

0043 implements the self-contained Opponent League and Promote Champion V2 project defined by:

- `docs/rl/0043_Project_Charter_Codex_Handoff.md`
- `docs/rl/0043_Promote_Champion_V2_Protocol.md`
- `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`

Current status is `BLOCKED_FORMAL_TRAINING_PREFLIGHT`. Project-local policy reconstruction/no-hybrid isolation, sampler, PFSP, telemetry, frozen schedule construction and the manual Promote workflow are implemented. Formal training still waits for official-engine smoke, official-observation CPU/CUDA first-divergence parity, and a fresh version/W&B preflight.

## Phase 1 commands

```bash
python3 -m train.0043_champion_league_rl.import_assets --all
PYTHONPATH=. /home/cyd/.cache/uv/archive-v0/uIcEF-JYHod8LomN/bin/pytest -q train/0043_champion_league_rl/tests
PYTHONPATH=. python3 -m train.0043_champion_league_rl.preflight --gpu-smoke --output .tmp/evaluation/0043_preflight/report.json
```

The import command is the only code allowed to read the approved historical sources. It freezes both immutable assets and the semantic inference source into 0043. Runtime consumers use only project-local registries, semantic policy directories and `semantic_runtime/`.

## Asset domains

- `assets/decks/`: 55 immutable exact-deck definitions with canonical IDs and directories `001`–`055`; future decks append from `056`.
- `assets/policies/`: complete Policy-0809 base plus Champion-G1 delta/portable artifacts under semantic directories `policy_0809` and `champion_g001`.
- `assets/evaluation/`: FrozenMeta256-V1 composition and its versioned seed contract metadata.

The implementation plan is `docs/superpowers/plans/2026-08-12-0043-champion-league-rl.md`; the authoritative current design and audit are under `experiments/0043_champion_league_rl/`.

SHA-256 values remain mandatory registry integrity fields. They are not deck IDs, policy IDs, or meaningful folder names.
