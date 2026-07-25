# Repository Structure Cleanup Design

**Date:** 2026-07-26  
**Status:** Approved in conversation

## Goal

Simplify the repository by relocating raw official data, flattening the documentation tree,
retiring obsolete Kaggle/BC assets, and removing local generated files. Preserve stable runtime
boundaries and avoid all files owned by the concurrent experiment-project architecture work.

## Target Boundaries

- Keep `evaluation/` as the stable Python package, CLI, and arena asset root.
- Keep `engine/` at repository root. `engine/source/` remains read-only and `engine/build/`
  remains the build-output boundary.
- Use `data/official/` for official card references, `data/raw/` for immutable source datasets,
  and `data/processed/` for derived machine-readable data.
- Use `docs/` directly as the human-readable documentation root; remove the redundant
  `docs/reports/` layer.

## Data Migration

Move the ignored local replay dataset without duplicating its approximately 11 GB payload:

```text
replays/raw/archives/       -> data/raw/episodes/archives/
replays/raw/download-logs/  -> data/raw/episodes/download_logs/
replays/raw/index/          -> data/raw/episodes/index/
replays/raw/patches/        -> data/raw/episodes/patches/
replays/raw/source_manifest.json -> data/raw/episodes/source_manifest.json
```

Create documented placeholders/contracts for:

```text
data/raw/leaderboards/
data/processed/environment_daily/
docs/environment-daily_kaggle_top100/daily/
```

This change does not implement downloading, leaderboard collection, aggregation, scheduling, or
daily-report generation.

## Documentation Migration

Move every file under `docs/reports/` into a direct topic under `docs/`, then remove
`docs/reports/`:

```text
docs/environment-daily_kaggle_top100/  dated Kaggle Top 100 environment-analysis HTML reports
docs/evaluation/       evaluation methods, metrics, and conclusions
docs/training/         BC, reward/value, RL, and training research
docs/decks/            deck research
docs/rules/            official rules and rule evidence
docs/research/         general and external research
docs/implementation/   completed implementation notes
docs/references/       reference material
docs/history/          index/status for superseded lines of work
docs/superpowers/      unchanged specs and plans
docs/README.md         primary documentation navigation
```

Top100 and Top20 assets become discoverable frozen snapshots under
`docs/environment-daily_kaggle_top100/daily/`. Historical conclusions remain unchanged; only brief status labels
and repaired links may be added. Update repository-local Markdown/HTML/code references to moved
paths. Do not keep compatibility copies or symlinks.

## Obsolete Kaggle/BC Retirement

Delete:

- `notebooks/kaggle_bc_worker/`;
- `train/kaggle_bc_top20/`;
- implementation-specific tests, configs, imports, and ignore rules with no remaining consumer;
- obsolete `rank_*` and `meta*` packages in `evaluation/arena/candidates/`;
- obsolete submissions and derived model packages.

Preserve raw official episodes, leaderboard/source research, and tests that still protect raw-data
integrity, replay visualization, evaluation, or shared infrastructure. Preserve candidates that
clearly belong to projects `0009` through `0012`. Do not modify opponents, catalog, or combat-mat
assets.

## Submission Archive

Keep only these packages, renamed with project IDs:

```text
archive/submission/0010_alakazam_sota_model_v4_loss_best/
archive/submission/0011_alakazam_sota_reward_weighted_bc_v3_loss_best/
archive/submission/0012_alakazam_sota_feature_engineering_v9_v1_exact_best/
```

Keep only their proven matching archives in `archive/submission/dist/`, renamed with the same project-ID
prefix. If an archive cannot be mapped confidently, report it rather than creating or guessing a
replacement.

## Local Cleanup

Delete local generated state:

- `.venv/`;
- `.tmp/` contents;
- `pokemon_tcg_ai_battle_research.egg-info/`;
- `.pytest_cache/`;
- all `__pycache__/` directories and `.pyc` files;
- the empty `replays/` tree after migration.

Retain ignore rules for regenerable caches and temporary output. Do not delete `.claude/` or
`.superpowers/`.

## Concurrent-Work Exclusion

Do not edit, move, or delete:

- `rl_environment/runs.py`;
- `tests/test_experiment_projects.py`;
- `docs/superpowers/specs/2026-07-26-experiment-project-architecture-design.md`;
- `docs/superpowers/plans/2026-07-26-experiment-project-architecture.md`;
- the concurrent SDD ledger;
- any new `experiments/` project structure;
- historical `rl_runs` assets for projects `0001` through `0012`;
- `engine/source/`.

## Lightweight Verification

Favor speed while retaining basic safety:

1. Compare replay-data file counts and total bytes before and after the move.
2. Search for live references to removed paths and repair applicable references.
3. Confirm the three retained submission packages and matching archives exist.
4. Run focused tests for directly affected surviving contracts, plus a Python syntax check where
   practical. Do not run official-engine battle evaluation because strategy semantics do not
   change.
