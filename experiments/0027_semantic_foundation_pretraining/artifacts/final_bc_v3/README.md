# 0027 final BC v3 artifact

`best_model.pt` is the exact model-only checkpoint downloaded from Kaggle kernel
`horizen12/ptcg-0027-05-train-dual-t4`, completed on 2026-08-03.

- bytes: 129,009,623
- SHA-256: `274407890d17b70a5a331cac1855a7a2438946e4c5a2989a99345f1ab4af5dfc`
- Git storage: LFS
- architecture: `CanonicalSemanticPolicy`
- parameters: 21,837,082
- actor contract: all 22 canonical features
- initialization: random
- optimizer/resume state: not present

`model_contract.json`, `training_report.json`, and
`official_public_prototypes_v1.json` are the exact accompanying Kaggle outputs.
`official_full_engine_prototypes_v1.json` is the matching immutable 0025
vendored sidecar required by `PrototypeIndex` for a self-contained strict load.
`evaluation/` contains the raw local BC pool checks for Dragapult and Raging
Bolt. See `evaluation_summary.json` for the aggregate and per-opponent results.
