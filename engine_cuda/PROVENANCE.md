# Engine CUDA Provenance

`engine_cuda/` was imported on 2026-07-30 from:

- source repository: `/home/cyd/repos/pokemon-tcg-ai`
- source commit: `bf56dfb4b1d58279f0cd1e49a646b46174d8a589`
- source subtree: `cuda_engine/`
- destination subtree: `engine_cuda/`

Only files tracked by that commit were imported. Local build directories,
Python caches, profiler reports, generated private rule packs, checkpoints, and
other ignored artifacts were excluded.

This snapshot is a CUDA-only engine prototype, not an official-engine
replacement. The unmodified official runtime under `engine/` and standard
evaluation packages remains the gameplay oracle until the promotion gates in
`docs/implementation_plan.md` pass. The 2026-07-30 local audit is recorded in
`experiments/0020_pluggable_deck_rl/CUDA_ENGINE_AUDIT.md`.

The imported `configs/` files retain source-repository paths for provenance and
are examples only. They are not a valid 0020 opponent catalog in this repository.
