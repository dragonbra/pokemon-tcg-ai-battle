# 0037 Update 32 Kaggle Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Execute this plan task-by-task inline; repository policy forbids subagent delegation unless the user explicitly requests it. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export 0037 V5 Update 32 as a self-contained Kaggle submission directory and root-level `.tar.gz`, preserving its trained action decoder and merged Last Option Q/V LoRA.

**Architecture:** Start from the audited 0034 Dragapult 007 zero-shot portable package. Merge each rank-4 Q/V LoRA delta into the final Option TransformerDecoder block's self-attention and cross-attention `in_proj_weight`, replace the action decoder tensors, and exclude the critic. Package the result under `archive/submission/0037_dragapult_ex_007_update32_last_option_qv_lora/`, archive its contents without an extra directory layer, then validate only from a fresh extraction.

**Tech Stack:** Python 3.11, PyTorch model-only checkpoints, repository evaluation CLI, official engine runtime, `tar.gz`.

## Global Constraints

- Do not modify `engine/source/`.
- Package root must directly contain `main.py`, `deck.csv`, `cg/`, `strategy/`, and `manifest.json`.
- `deck.csv` must be exact 60 and initialization `agent({"select": null})` must return the same list.
- Package must contain no symlink, cache, optimizer, rollout, trace, or repository runtime dependency.
- Update 32 checkpoint schema must be `0037_value_initialized_adapted_model_only_v2`, Update must equal 32, and its SHA sidecar must match.
- The actor action decoder and both Last Option block Q/V LoRA modules are deployed; Value head is excluded.
- Final gate uses a fresh extraction and official engine for at least one opponent ×10 games with 10/10 finished and 0 errors.

---

### Task 1: Adapted checkpoint export

**Files:**
- Modify: `train/0037_dragapult_value_initialized_rl/export_full_semantic_candidate.py`
- Create: `train/0037_dragapult_value_initialized_rl/tests/test_export_full_semantic_candidate.py`

**Interfaces:**
- Consumes: source portable `strategy/model.bin` and Update 32 adapted model-only checkpoint.
- Produces: `export_candidate(source: Path, checkpoint: Path, output: Path) -> dict[str, Any]` with merged Q/V weights and action decoder.

- [ ] **Step 1: Add a failing merge-contract test**

Construct a minimal merged QKV tensor plus q/v A/B factors and assert that only Q and V rows receive `(B @ A) * alpha/rank`, K remains byte-identical, and malformed shapes fail closed.

- [ ] **Step 2: Run the focused test and confirm the old exporter rejects the adapted schema**

Run: `python3 -m unittest -v train.0037_dragapult_value_initialized_rl.tests.test_export_full_semantic_candidate`

Expected: failure because adapted checkpoint merge support is absent.

- [ ] **Step 3: Implement the adapted exporter**

Validate checkpoint SHA sidecar, schema, Update 32 metadata, rank 4 / alpha 8 / Option block 1 / LayerNorm disabled, exact trainable tensor inventory, and source checkpoint schema. Merge self-attention and cross-attention Q/V deltas into their corresponding portable `in_proj_weight`; replace `action_decoder.*`; explicitly ignore `value_head.*`; write deployment provenance and tensor hashes to `manifest.json`.

- [ ] **Step 4: Run exporter tests**

Run: `python3 -m unittest -v train.0037_dragapult_value_initialized_rl.tests.test_export_full_semantic_candidate train.0037_dragapult_value_initialized_rl.tests.test_adaptation`

Expected: all pass.

### Task 2: Package materialization and design synchronization

**Files:**
- Modify: `experiments/0037_dragapult_value_initialized_rl/DESIGN.md`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DESIGN.html`
- Create: `archive/submission/0037_dragapult_ex_007_update32_last_option_qv_lora/`

**Interfaces:**
- Consumes: Task 1 exporter, Update 32 checkpoint, audited 0034 zero-shot package.
- Produces: self-contained source package with portable merged actor.

- [ ] **Step 1: Export to the final archive source directory**

Run the exporter with source `evaluation/arena/candidates/0034_dragapult_third_large_model_zero_shot`, checkpoint `rl_runs/0037_dragapult_value_initialized_rl/versions/V5_last_option_qv_lora_r4_eval5_50u/checkpoint/update-000032.pt`, and final archive source directory.

- [ ] **Step 2: Audit package contents**

Verify top-level files, exact 60 deck, no symlink/cache, no forbidden recovery state, no absolute runtime path, and manifest checkpoint/portable hashes.

- [ ] **Step 3: Synchronize authoritative DESIGN documents**

Record Update 32 as the exported research checkpoint, document Q/V LoRA merge into portable Option block weights, state that the critic is training-only and excluded, and record the final package boundary and path.

### Task 3: Archive and fresh-extraction gates

**Files:**
- Create: `archive/submission/dist/0037_dragapult_ex_007_update32_last_option_qv_lora.tar.gz`
- Create: `.tmp/evaluation/0037_update32_submission_gate/run-*/report.html`

**Interfaces:**
- Consumes: Task 2 package directory.
- Produces: submission archive plus fresh-extraction validation evidence.

- [ ] **Step 1: Create a root-content tarball**

Archive the contents of the package directory so `main.py`, `deck.csv`, `cg/`, `strategy/`, and `manifest.json` appear at tar root, excluding caches and symlinks.

- [ ] **Step 2: Extract into a fresh temporary directory and inspect structure**

Reject any extra wrapper directory, symlink, cache, missing model/runtime asset, or non-60-card deck.

- [ ] **Step 3: Run package validation**

Run: `python3 -m evaluation validate <fresh-extracted-root>`

Expected: validation succeeds.

- [ ] **Step 4: Run no-`__file__` dynamic execution**

Execute root `main.py` with `__file__` absent and no repository `PYTHONPATH`, call initialization observation, and assert the returned deck exactly matches all 60 lines.

- [ ] **Step 5: Run official-engine 10-game smoke**

Run the evaluation CLI against one enabled opponent using the freshly extracted root, writing the report below `.tmp/evaluation/0037_update32_submission_gate/`; require 10 finished, 0 errors.

- [ ] **Step 6: Final hash and size report**

Report the package directory, `.tar.gz` path, archive SHA-256, archive size, extracted validation result, and official-engine smoke report path.
