# 008 - Bind all RNGs before model initialization

Date: 2026-07-26
Status: accepted

Formal `V3_m0_atomic_checkpoint_fix` completed one full epoch with valid metrics and checkpoint
selection, proving the atomic pointer fix. Comparing its epoch-1 result with V2 exposed that the
configured seed controlled feature-shard and option permutation order but was never applied to
Python, NumPy, PyTorch CPU, or PyTorch CUDA before model construction. V2 and V3 therefore began
from different unrecorded parameter states.

V3 was stopped after epoch 1 and is preserved failed with failure class `unbound_model_rng_seed`.
Its W&B URL, canonical metrics, and checkpoint remain diagnostic evidence, not a formal M0 result.

Formal startup now, before dataset source or model creation:

- sets Python `random`, NumPy, PyTorch CPU, and all CUDA RNGs to the declared seed;
- sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- disables cuDNN benchmark and enables cuDNN deterministic behavior;
- enables PyTorch deterministic algorithms;
- computes a canonical initial model state SHA-256 over sorted state-dict names, dtypes, shapes, and
  raw tensor bytes;
- records the complete reproducibility contract and initial model digest in both training config
  and model contract.

Two independent processes initialized M0 with seed 20260726 and both produced initial model SHA-256
`e591bbe16390aa0ab52023294f7f3ea80aa946ea510b828e999f93375dfb0145`. A real CUDA
forward/backward smoke passed with deterministic algorithms enabled. The next formal attempt is V4
and must restart all three epochs from this bound initial state.
