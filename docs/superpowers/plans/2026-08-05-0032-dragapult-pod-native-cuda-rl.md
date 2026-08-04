# 0032 Dragapult POD-Native CUDA RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a self-contained Dragapult actor-critic that consumes the CUDA engine's resident `PolicyCodecV1` tensors, imports only explicitly compatible 0031 weights, and is ready for a separately versioned PPO run after correctness and throughput gates pass.

**Architecture:** The actor consumes the fixed 12-tensor POD codec contract directly. Field-aware embeddings produce global, entity, and option tokens; an entity Transformer and option cross-attention feed the frozen-copy ordered pointer decoder, while a separate scalar value head reads the state summary. The 0031 checkpoint is an initialization source only, with an exact hash and explicit source-to-destination tensor map.

**Tech Stack:** Python 3.11, PyTorch 2.11/CUDA 12.8, the repository `engine_cuda` PyTorch extension, unittest, W&B only for later formal training.

## Global Constraints

- Never modify `engine/source/`.
- `train/0032_dragapult_pod_native_cuda_rl/` must not import executable code from another numbered training project.
- The actor-visible contract contains only POD `PolicyCodecV1` state; source/team identity is provenance only.
- V1 is an adapter-validation version, not a formal PPO run; do not fabricate training metrics or checkpoints.
- All formal checkpoints are model-only, atomically written, and retained.
- Do not claim 0031 feature or behavior parity; report transferred tensors and newly initialized tensors explicitly.
- PPO may start only after CPU contract tests, codec parity, resident action legality, and measured throughput gates pass.
- The formal opponent snapshot is the 38-deck admitted subset recorded by the frozen51 audit, with `dragapult_ex_001` as focal deck.

---

### Task 1: Freeze The POD Actor Contract And Project Record

**Files:**
- Create: `train/0032_dragapult_pod_native_cuda_rl/contract.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_contract.py`
- Create: `experiments/0032_dragapult_pod_native_cuda_rl/manifest.json`
- Create: `experiments/0032_dragapult_pod_native_cuda_rl/DECISIONS.md`
- Create: `rl_runs/0032_dragapult_pod_native_cuda_rl/versions/V1_pod_native_adapter/artifact/status.json`

**Interfaces:**
- Consumes: the 12 named tensors emitted by CUDA `policy_codec_v1()`.
- Produces: `PodNativeBatch.from_mapping(mapping)` and stable width/capacity constants.

- [ ] Write tests covering required keys, shapes, dtypes, parent bounds, selection bounds, and CPU/CUDA validation behavior.
- [ ] Run `python3 -m unittest -v train.0032_dragapult_pod_native_cuda_rl.tests.test_contract` and verify the tests fail before implementation.
- [ ] Implement the immutable mapping wrapper without device synchronization in the CUDA validation path.
- [ ] Rerun the contract tests and record their result in V1 status.

### Task 2: Implement The POD-Native Actor-Critic

**Files:**
- Create: `train/0032_dragapult_pod_native_cuda_rl/model/config.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/model/action_decoder.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/model/actor_critic.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_actor_critic.py`

**Interfaces:**
- Consumes: `PodNativeBatch` or a compatible tensor mapping.
- Produces: `encode`, `logits`, `greedy`, `act_device`, and `value` methods; `act_device` returns `[B,S]` indices and `[B]` lengths entirely on the input device.

- [ ] Test output shapes, padding invariance, legal STOP/min/max behavior, no duplicate selections, gradients, and CUDA device residency when available.
- [ ] Implement separate field embeddings for bounded categorical fields and MLP projections for numeric fields.
- [ ] Implement parent-aware entity tokens, masked entity Transformer state, option-to-state cross-attention, the ordered pointer decoder, and scalar value head.
- [ ] Run the actor tests and a forward/backward smoke on CPU and CUDA.

### Task 3: Audit 0031 Weight Transfer

**Files:**
- Create: `train/0032_dragapult_pod_native_cuda_rl/transfer.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_transfer.py`
- Create during execution: `rl_runs/0032_dragapult_pod_native_cuda_rl/versions/V1_pod_native_adapter/artifact/weight_transfer.json`

**Interfaces:**
- Consumes: checkpoint SHA-256 `05f98e2a562f0037713ebd4b96a5a324151c91b187d7edaa879cc1c024134a51` and an uninitialized 0032 actor.
- Produces: a strict transfer report listing every copied source/destination tensor, every newly initialized tensor, parameter totals, and transfer fraction.

- [ ] Test rejection of the wrong checkpoint hash, missing source tensors, duplicate destinations, and shape mismatches.
- [ ] Define an explicit semantic map for the decoder, compatible Transformer blocks, and overlapping card/attack identity rows; prohibit general shape-based matching.
- [ ] Apply transfer to a fresh model, save the JSON audit, and verify copied tensors elementwise.

### Task 4: Build A Bounded POD Adaptation Corpus And BC Smoke

**Files:**
- Create: `train/0032_dragapult_pod_native_cuda_rl/data/build_adaptation_corpus.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/training/bc_smoke.py`
- Create: `train/0032_dragapult_pod_native_cuda_rl/tests/test_adaptation_data.py`

**Interfaces:**
- Consumes: raw 0031 JSONL.GZ rows with `actor_observation`, `ordered_action`, split, Episode, and source provenance.
- Produces: sharded `PolicyCodecV1` corpus plus manifest, without putting source identity into actor tensors.

- [ ] Test option-index preservation, exact ordered labels, Episode split preservation, provenance counts, and deterministic bounded sampling.
- [ ] Materialize a small V1 diagnostic corpus using the CPU reference codec.
- [ ] Run a short transferred-vs-random BC optimization smoke and report loss, exact action, token accuracy, examples/s, and peak CUDA memory as diagnostics only.

### Task 5: Build And Validate The PyTorch CUDA Extension

**Files:**
- Create during execution: `.tmp/cuda_0032_build/toolkit/` normalized toolkit view.
- Create during execution: `.tmp/cuda_0032_build/build/` isolated build tree.
- Create: `engine_cuda/tools/build_torch_extension_isolated.py` only if the normalized CMake invocation cannot be expressed reproducibly as a command.

**Interfaces:**
- Consumes: real CUDA compiler `/home/cyd/.local/cuda-12.8/bin/nvcc` and installed Torch CMake config.
- Produces: ABI-loadable `_ptcg_cuda` without modifying system symlinks or `engine/source/`.

- [ ] Create a normalized, local toolkit view and run CMake configure with SM 120 and the exact Torch/Python locations.
- [ ] Build the extension and verify import, engine construction, CUDA tensor ownership, and one reset/codec/action cycle.
- [ ] Run CPU `PolicyCodecV1` versus GPU codec elementwise parity on captured decisions.

### Task 6: Resident Legality And Throughput Gates

**Files:**
- Create: `train/0032_dragapult_pod_native_cuda_rl/benchmark_resident.py`
- Create during execution: `evaluation/arena/combat_mat/0031_latest_frozen_test/0032_pod_native_throughput.json`
- Modify: `evaluation/arena/combat_mat/0031_latest_frozen_test/THROUGHPUT.md`

**Interfaces:**
- Consumes: the admitted 38-deck snapshot, focal Dragapult deck, built extension, and transferred/adapted actor.
- Produces: action legality/error totals, decisions/s, completed episodes/s, p50/p95 decision latency, peak memory, batch/lane count, and CPU reference comparison.

- [ ] Run real resident CUDA games across admitted Dragapult matchups and fail on any invalid action, engine error, timeout, or host transfer in the hot path.
- [ ] Benchmark representative lane counts after warmup, synchronize timing boundaries, and preserve raw measurements.
- [ ] Profile the best stable lane count with NCU if a kernel bottleneck needs optimization; preserve reports under a timestamped `.tmp` run.
- [ ] Update the throughput report using only measured resident results and state the exact comparison contract.

### Task 7: Synchronize Design And Gate Formal PPO

**Files:**
- Create: `experiments/0032_dragapult_pod_native_cuda_rl/DESIGN.md`
- Create: `experiments/0032_dragapult_pod_native_cuda_rl/DESIGN.html`
- Modify: `experiments/0032_dragapult_pod_native_cuda_rl/DECISIONS.md`
- Modify: `rl_runs/0032_dragapult_pod_native_cuda_rl/versions/V1_pod_native_adapter/artifact/status.json`

**Interfaces:**
- Consumes: implemented tensor contract, model parameter inventory, transfer report, codec parity, legality, and throughput evidence.
- Produces: synchronized authoritative design documents and an explicit pass/fail decision for creating V2 PPO.

- [ ] Cross-check all documented shapes, parameter counts, hashes, and gates against code and artifacts.
- [ ] Render matching Markdown and HTML design documents with evidence links and the 0031 non-parity boundary.
- [ ] If every gate passes, allocate a fresh `V2_*` with a fresh optimizer, W&B online run, fixed opponent snapshot, model-only checkpoint retention, foreground watchdog, and separate rollout/eval namespaces; otherwise stop at V1 with blockers recorded.
