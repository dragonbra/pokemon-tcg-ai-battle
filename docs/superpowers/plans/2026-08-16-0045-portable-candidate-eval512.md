# 0045 Portable Candidate Eval512 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the exact complete policy stored in an exported 0045 Kaggle `model.bin` under the existing Benchmark Tiny V2 CUDA-512 contract without rebuilding its frozen actor from a mutable base asset.

**Architecture:** Add a fail-closed loader that strict-loads every portable actor, Value, allocation, adapter, and Option-LoRA tensor into the current CUDA collector model while deriving identity from the sealed FP16 artifact. Extend the existing Tiny V2 runner with a mutually exclusive portable input mode and preserve the existing checkpoint mode for training-time evaluation.

**Tech Stack:** Python 3.11, PyTorch, pytest, CUDA Engine 2.0, existing 0045 evaluation and identity helpers.

## Global Constraints

- The evaluated deployment-effective hash must equal the package manifest value `ce72121faa62801133354ab5a7c8a0e1c394cf77dc89267a43bd19b1572e9178` before games start.
- The portable artifact remains FP16 storage and is strict-loaded into FP32 runtime modules.
- The opponent remains the complete immutable `Champion-G2` policy with no focal/opponent tensor storage aliasing.
- The fixed `0045_benchmark_tiny_v2_core16_meta_balanced_common_seeds_cuda512_v1` schedule, greedy selection, official engine, and 512-game health gates remain unchanged.
- No optimizer state, training behavior, official engine source, or submitted archive is modified.

---

### Task 1: Fail-Closed Portable Candidate Loader

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/evaluation/candidate.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_policy_only_export.py`

**Interfaces:**
- Consumes: a sealed `0045_minimal_lora_candidate_v1` artifact, exact deck, deck/taxonomy identity, expected source hash, expected portable hash, and expected effective hash.
- Produces: `load_portable_candidate(...) -> tuple[SemanticActorCritic, CandidateAudit]`.

- [ ] **Step 1: Write a failing portable identity test**

Create a test that passes a mismatched expected hash and asserts the loader raises before returning a model, then passes matching hashes and asserts all floating runtime tensors are FP32 and the returned effective hash is exact.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
python3 -m pytest train/0045_single_deck_expert_minimal_lora/tests/test_policy_only_export.py -q
```

Expected: FAIL because `load_portable_candidate` does not exist.

- [ ] **Step 3: Implement strict portable loading**

Load the artifact with `weights_only=True`, validate schema/metadata/deck/update and file SHA-256, build the collector-compatible `SemanticActorCritic`, strict-load all five state dictionaries, convert only runtime tensors to FP32, compute `_tensor_hash`, and reject any expected identity mismatch.

- [ ] **Step 4: Run the focused test and verify pass**

Run the same pytest command and expect PASS.

### Task 2: Tiny V2 Portable Input Mode

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/evaluation/run_benchmark_tiny_v2.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/evaluation/render_benchmark_tiny_v2.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_minimal_optimizer_and_tiny_v2.py`

**Interfaces:**
- Consumes: either the existing source checkpoint inputs or `--portable-candidate` plus manifest-bound expected identities.
- Produces: the unchanged Tiny V2 report schema with a candidate audit that identifies the exact package policy.

- [ ] **Step 1: Add a failing CLI/input-contract test**

Assert checkpoint and portable modes are mutually exclusive and portable mode requires source, portable, and effective SHA-256 values.

- [ ] **Step 2: Run the focused runner tests and verify failure**

```bash
python3 -m pytest train/0045_single_deck_expert_minimal_lora/tests/test_minimal_optimizer_and_tiny_v2.py -q
```

- [ ] **Step 3: Wire portable mode into the existing run boundary**

Select `load_portable_candidate` before schedule construction, retain the same opponent materialization, storage-alias gate, jobs, collector, and report validation, and leave checkpoint mode behavior unchanged.

- [ ] **Step 4: Run focused tests and verify pass**

Run both focused test files and expect PASS.

### Task 3: Exact Package CUDA-512 Evaluation

**Files:**
- Create: `.tmp/evaluation/0045_003_final_dance_v1_exact_package_tiny_v2_cuda512/report.json`
- Create: `.tmp/evaluation/0045_003_final_dance_v1_exact_package_tiny_v2_cuda512/report.html`

**Interfaces:**
- Consumes: `0045-003-FINAL_DANCE_V1-MEGA_LOPUNNY_EX.tar.gz` and its verified extracted `strategy/model.bin`.
- Produces: 512-game official-engine diagnostic evidence bound to the package effective identity.

- [ ] **Step 1: Run the exact portable candidate**

Invoke Tiny V2 with Deck `003`, U385, the package portable artifact, and the three expected manifest hashes.

- [ ] **Step 2: Validate all report gates**

Require 512 entries, 512 valid terminal games, zero errors/unfinished/routing failures/D2H bytes, zero shared parameter storages, CUDA identity PASS, opponent identity PASS, and exact package deployment hash.

- [ ] **Step 3: Render the report**

Render the PASS JSON to `report.html` and verify the displayed summary matches JSON.

- [ ] **Step 4: Report the valid result and quarantine the invalid reconstruction run**

Clearly label the earlier `468dcc...` reconstruction result as non-package diagnostic evidence and report only the exact portable run as this archive's Eval512 score.
