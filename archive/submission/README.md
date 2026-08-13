# Historical Submission Archive

This directory contains the final, project-numbered Kaggle submission payloads retained as immutable
historical archives:

- `0010_alakazam_sota_model_v4_loss_best/`
- `0011_alakazam_sota_reward_weighted_bc_v3_loss_best/` — source metadata retained, but its ignored `strategy/model.bin` and packaged `.tar.gz` were not present during cleanup, so this package is not currently runnable.
- `0012_alakazam_sota_feature_engineering_v9_v1_exact_best/`
- `0013_v5_m0_epoch14_loss_best/` - M0 epoch 14 validation-loss-best self-contained payload and local official-engine evaluation candidate.
- `0022_frozen_festival_lead_dipplin_001/` - 0019 epoch-13 neutral zero-shot policy with the exact 0022 Festival Lead / Dipplin 001 deck.
- `0022_frozen_mega_kangaskhan_ex_crustle_004/` - 0019 epoch-13 neutral zero-shot policy with the exact 0022 Mega Kangaskhan ex / Crustle 004 deck.
- `0043_dragapult_ex_007_v2_u136_fp16_storage_fp32_runtime/` — 0043 V2 U136 with exact deck 007, materialized under `kaggle_fp16_storage_fp32_runtime_v1`; its formal Frozen Policy-0809 CUDA-2048 evidence is 1301-746-1 (63.53%).

Each package remains self-contained with `main.py`, `deck.csv`, and its own `cg/` runtime. They are
not current training entry points. New trainable candidates belong in `evaluation/arena/candidates/`.
All newly selected long-term payloads must be archived here under a project-numbered name; the
repository-root `submission/` path is retired. Matching locally packaged archives, when available,
live in `dist/`.

## PyTorch package load-order contract

Every newly built PyTorch package must declare the following fields in its root `manifest.json`:

```json
{
  "runtime_framework": "pytorch",
  "native_runtime_load_order": "torch_before_cg"
}
```

Its `main.py` startup order is normative:

1. Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
   `NUMEXPR_NUM_THREADS` before importing numerical libraries.
2. Resolve the package root without assuming `__file__` exists.
3. Import `torch` eagerly.
4. Only then import policy modules or anything that can import `cg`, call `ctypes.CDLL`, or load
   `cg/libcg.so`.
5. Load the model and expose `read_deck_csv()` plus `agent(observation)`.

Do not defer the Torch import until the first non-registration action. Loading `libcg` first and
Torch second in one process can terminate the worker with a native segmentation fault during
`torch._ops` or quantization initialization. That crash usually appears as a step-0
`worker_crash`; it is not a normal Python exception and cannot be treated as an inconclusive smoke.

The final gate always starts from a fresh extraction of the exact archive in `dist/`. Run package
validation without repository `PYTHONPATH`, exercise Kaggle-style raw `exec` without `__file__`,
and run at least one official-engine opponent for 10 games. The evaluation worker must preload
Torch from the manifest before loading `cg/libcg`; all 10 games must finish with zero errors. A
source-directory import, one manually preloaded game, shared-inference-only run, or a crash also
seen in an older package does not satisfy this gate. Use `PYTHONFAULTHANDLER=1` to diagnose a
step-0 native crash, correct the load order, rebuild the archive, and repeat every extraction gate.

The local runner defaults to a 30-second per-game worker timeout. Large CPU models can exceed that
limit without crashing, especially under process contention. Inspect the retained trace error: a
literal `worker timed out` is a timeout, while a non-zero process exit or SIGSEGV without a Python
error is a native crash. Calibrate slow-model smoke explicitly, for example with
`--worker-timeout-seconds 120 --workers 1 --worker-cpu-threads 1`, and record those values with the
gate result. Increasing the timeout must never be used to hide a native crash, and the final result
must still be 10/10 finished with zero errors.
