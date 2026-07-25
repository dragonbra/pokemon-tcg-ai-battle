# Historical Submission Archive

This directory contains the final, project-numbered Kaggle submission payloads retained as immutable
historical archives:

- `0010_alakazam_sota_model_v4_loss_best/`
- `0011_alakazam_sota_reward_weighted_bc_v3_loss_best/` — source metadata retained, but its ignored `strategy/model.bin` and packaged `.tar.gz` were not present during cleanup, so this package is not currently runnable.
- `0012_alakazam_sota_feature_engineering_v9_v1_exact_best/`

Each package remains self-contained with `main.py`, `deck.csv`, and its own `cg/` runtime. They are
not current training entry points. New trainable candidates belong in `evaluation/arena/candidates/`.
Matching locally packaged archives, when available, live in `dist/`.
