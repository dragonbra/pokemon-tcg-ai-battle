# 0034 Zero-Shot Exact007 / Exact009 Kaggle Packages Plan

**Goal:** Produce two flat, self-contained Kaggle submission archives from the audited 0806 zero-shot checkpoint: Dragapult ex exact007 and Mega Lucario ex / Solrock exact009.

**Source contract:** Reuse `train/0034_dragapult_third_large_model_rl/export_zero_shot_candidate.py`, the immutable 0806 checkpoint, the official runtime bundle, and the exact decks frozen in the 0034 league catalog. Do not modify `engine/source/` or submit anything externally.

**Outputs:**

- `archive/submission/0034_dragapult_ex_007_zero_shot_0806/`
- `archive/submission/dist/0034_dragapult_ex_007_zero_shot_0806.tar.gz`
- `archive/submission/0034_mega_lucario_ex_solrock_009_zero_shot_0806/`
- `archive/submission/dist/0034_mega_lucario_ex_solrock_009_zero_shot_0806.tar.gz`

## Task 1: Export the two self-contained packages

- Confirm both output names are unused.
- Export each exact 60-card deck with the audited 0806 model-only checkpoint and official `cg/` runtime.
- Update only package-local manifests so deck IDs, exact-deck hashes, and submission names are auditable.
- Confirm no symlinks, caches, optimizer state, or other forbidden files exist.

## Task 2: Build and inspect flat archives

- Create each `.tar.gz` with `main.py`, `deck.csv`, `cg/`, and `strategy/` directly at extraction root.
- Extract each archive into a fresh temporary directory.
- Verify exact 60-card identity, archive shape, file types, and checksum.

## Task 3: Run submission compatibility gates

- Run `python3 -m evaluation validate` on each extracted root.
- Dynamically execute each extracted `main.py` without defining `__file__` and without repository `PYTHONPATH`.
- Confirm `read_deck_csv()` and `agent({"select": null})` both return the exact packaged 60-card deck.

## Task 4: Run official-engine smoke evaluation

- Use each extracted package as the candidate against one fixed catalog opponent for 10 official-engine games.
- Store reports under `.tmp/evaluation/0034_zero_shot_kaggle_package_smoke/`.
- Require 10/10 finished and zero errors for each archive; fail closed otherwise.

## Task 5: Handoff

- Report clickable archive and smoke-report paths, archive size, SHA-256, exact deck identity, and gate results.
- Explicitly state that no Kaggle submission was performed.
