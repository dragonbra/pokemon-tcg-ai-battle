# Repository Structure Cleanup — Completion Notes

Completed on 2026-07-26.

- Moved 33 official episode files (11,069,771,476 bytes) from `replays/raw/` to
  `data/raw/episodes/` without copying the payload.
- Flattened `docs/reports/` into direct topic directories and added navigation indexes.
- Added `docs/environment-daily_kaggle_top100/index.html` as the primary dated **battle-environment report** index and
  archived the 2026-07-25 Top 100 analysis at `docs/environment-daily_kaggle_top100/daily/2026-07-25.html`; this is not
  a project-progress diary.
- Removed the retired generic Kaggle BC notebook, package, exclusive tests, and obsolete
  `rank_*`/`meta*` candidates.
- Retained project candidates for `0009`–`0012`.
- Retained project-numbered submission packages under `archive/submission/`.
- Matching dist archives exist for projects `0010` and `0012`. No proven matching `0011` archive
  existed, and its ignored `strategy/model.bin` was also absent; the retained `0011` directory is
  therefore metadata/source only and is not currently runnable. No replacement was created or guessed.
- Python compilation passed. Twenty-one focused full-action/evaluation asset/opponent tests passed.
- Archived packages `0010` and `0012` passed `python3 -m evaluation validate`; `0011` failed validation
  specifically because the absent ignored `strategy/model.bin` could not be loaded.
- Removed `.venv/`, `.tmp/`, egg-info, pytest caches, Python bytecode, and the empty `replays/`
  directory.

Concurrent experiment-project architecture files and historical `rl_runs` assets were excluded.
