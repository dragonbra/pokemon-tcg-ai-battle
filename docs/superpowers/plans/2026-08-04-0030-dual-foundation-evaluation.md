# 0030 Dual Foundation Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train 0030 equally against legacy 0019 and semantic 0028 frozen opponent suites, then evaluate every checkpoint against each suite separately on the same fixed 51-deck schedule.

**Architecture:** The live focal policy always uses the current 0028-based 0030 checkpoint. Every training update contains 256 games against 51 decks served by one shared legacy 0019 policy and 256 games against the same decks served by one shared frozen 0028 policy. At update 0 and every five updates, both persistent GPU services run separate 102-game balanced evaluations with identical decks, seeds and seats.

**Tech Stack:** Python 3.11, PyTorch, `multiprocessing` spawn, official engine runtime, immutable `evaluation/arena/frozen` assets, `unittest`, W&B online logging.

## Global Constraints

- Do not modify `engine/source/`.
- Keep `train/0030_dragapult_shared_encoder_decoder_rl/` free of executable imports from another numbered training project.
- Treat the 0019 model and ontology under `evaluation/arena/frozen/_policy/strategy/` as immutable shared evaluation assets and verify exact SHA-256 values before loading.
- Use all 51 decks from pool `0019_foundation_51_exact_decks_v4` for both evaluation branches.
- Use identical opponent order, focal seat assignment and engine seeds for both branches.
- Keep the focal candidate, feature schema, trainable parameters and PPO loss/reward unchanged; explicitly version the new equal-weight 102-identity training opponent schedule.
- Keep engine workers Torch-light; both foundation models and feature compilers run only in the parent GPU service.
- Record sampled rollout, 0019 frozen greedy and 0028 frozen greedy as distinct contracts.
- Select best checkpoints by the legacy 0019 Frozen Arena win rate; retain the 0028 result as a co-equal diagnostic, not a replacement value.

---

### Task 1: Bind The Legacy Foundation Asset

**Files:**
- Create: `train/0030_dragapult_shared_encoder_decoder_rl/legacy_foundation.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/tests/test_contract.py`

**Interfaces:**
- Produces: `LegacyFoundationIdentity`, `verify_legacy_foundation()`, and `LegacyFoundationService`.
- Consumes: exported model SHA `2a3e3224b9fbc0bda95de45583faa1f3923d37bc55e26c6279315cc720d7f918`, ontology SHA `8144c63e512a2a00fabaaf2b19cd002c5b59c25fb0763a112256848460113a4d`, and provenance weight SHA `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`.

- [ ] **Step 1: Write the identity failure test**

Assert the loader reports asset ID `0019-0730-epoch13`, deployment source ID `0`, the three exact hashes above, and rejects an alternate asset root.

- [ ] **Step 2: Run the focused test before implementation**

Run: `python3 -m unittest -v train.0030_dragapult_shared_encoder_decoder_rl.tests.test_contract.ProjectContractTest.test_legacy_foundation_identity`
Expected: FAIL because `legacy_foundation.py` does not exist.

- [ ] **Step 3: Implement strict verification and loading**

Load `PortablePolicy` from the frozen evaluation asset, move its actor to the requested device, freeze all parameters, and expose its `OnlineCausalEncoder` class. Do not import `train.0019_*` or `train.0022_*`.

- [ ] **Step 4: Run the identity and numbered-import tests**

Expected: exact hashes pass and no cross-numbered executable imports appear.

### Task 2: Add Batched Legacy Greedy Inference

**Files:**
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/legacy_foundation.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/rollout/collector.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/tests/test_contract.py`

**Interfaces:**
- Produces: `LegacyFoundationService.new_encoder(actor, deck)` and `LegacyFoundationService.greedy(rows) -> list[tuple[int, ...]]`.
- Consumes: legacy online batches with variable second dimensions and batched `deterministic_action_tensors`.

- [ ] **Step 1: Write collate and freeze tests**

Test padding for every legacy variable-width field, all model parameters `requires_grad=False`, and output count equal to input row count.

- [ ] **Step 2: Confirm focused failure**

Expected: missing service/collator methods.

- [ ] **Step 3: Implement parent-only batching**

Pad legacy `entity`, `option`, registered-card, ledger, event and known-hand dimensions, concatenate rows, move once to GPU, call the actor's batched deterministic decoder, and return each row's real sequence length.

- [ ] **Step 4: Route collector opponents explicitly per job**

Add `RolloutJob.opponent_foundation` and a mixed collector mode. Focal rows always use 0028. Each opponent row uses its job's selected service. Reject mixed/0019 collection without a verified service.

### Task 3: Build The Equal-Weight Mixed Training Schedule

**Files:**
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/rollout/protocol.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/training/run.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/tests/test_contract.py`

**Interfaces:**
- Produces: 102 immutable opponent identities `(deck_id, foundation)` and exactly 256 games per foundation in every 512-game update.

- [ ] **Step 1: Test identity and seat counts**

Assert 51 distinct decks per foundation, 256 episodes per foundation, and 128 focal-first plus 128 focal-second episodes inside each foundation.

- [ ] **Step 2: Implement deterministic rotation**

Iterate each rotated deck through foundations `(0019, 0028)` and seats `(first, second)` before advancing. Keep rollout seeds update-dependent and frozen-evaluation seeds checkpoint-independent.

- [ ] **Step 3: Add foundation rollout metrics**

Log `rollout/foundation_0019/*` and `rollout/foundation_0028/*` episode results alongside overall rollout diagnostics; do not merge these with `eval/*`.

### Task 4: Make Frozen Evaluation Dual And Comparable

**Files:**
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/training/run.py`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/tests/test_contract.py`

**Interfaces:**
- Produces: metric namespaces `eval/foundation_0019/*` and `eval/foundation_0028/*`, plus shared `eval/checkpoint_update`.
- Consumes: one job list from `build_jobs(..., greedy_eval=True)` reused by both collectors.

- [ ] **Step 1: Test namespace and schedule identity**

Monkey-patch the two collector calls and assert both receive jobs with identical `(opponent_id, focal_first, seed, deck hashes)` tuples; assert no unqualified `eval/win_rate` is emitted.

- [ ] **Step 2: Confirm focused failure**

Expected: current `_evaluate` returns only one unqualified namespace.

- [ ] **Step 3: Implement `_evaluate_dual`**

Run legacy 0019 first and semantic 0028 second from the same immutable job tuple. Prefix all episode and inference metrics by foundation identity. Return the legacy win rate as the checkpoint-selection score and both wall times independently.

- [ ] **Step 4: Preserve metric timing semantics**

At update `k`, log `eval/checkpoint_update=k`, both greedy branches, and `rollout/source_policy_update=k-1` without shifting or rewriting prior metrics.

### Task 5: Verify Official-Engine Mixed Training And Dual Evaluation

**Files:**
- Create: `.tmp/evaluation/0030_dual_foundation_smoke/result.json`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/benchmark.py`

**Interfaces:**
- Produces: official-engine evidence for both branches, worker PSS, GPU memory and zero-error gates.

- [ ] **Step 1: Run two games per foundation**

Use identical Dragapult/opponent jobs and two seats. Require all four official-engine games to finish with no worker EOF, timeout or engine error.

- [ ] **Step 2: Run the full 102+102 update-0 gate**

Use V2 update-5 candidate weights, 128 workers, fixed seed `20260804`, and both persistent GPU models. Record each branch's win rate, wall time, inference requests and combined GPU peak memory.

- [ ] **Step 3: Run a full 512-game mixed rollout gate**

Require 256 games from each foundation, balanced seats within each, no engine errors, no worker Torch/CUDA mappings, no swap growth and no material throughput regression.

- [ ] **Step 4: Re-run worker import tests**

Expected: importing the engine worker still leaves Torch absent; live worker maps contain `libcg` but not Torch/CUDA.

### Task 6: Synchronize Authority Documents And Launch V3

**Files:**
- Modify: `experiments/0030_dragapult_shared_encoder_decoder_rl/DESIGN.md`
- Modify: `experiments/0030_dragapult_shared_encoder_decoder_rl/DESIGN.html`
- Modify: `experiments/0030_dragapult_shared_encoder_decoder_rl/DECISIONS.md`
- Modify: `train/0030_dragapult_shared_encoder_decoder_rl/INFERENCE_PERFORMANCE.md`

**Interfaces:**
- Produces: auditable dual-foundation provenance, metric interpretation, primary selection rule and current stage.

- [ ] **Step 1: Update both DESIGN formats**

Document that universal official rules and engine behavior are unchanged; the project-level opponent policy variable is legacy 0019 versus semantic 0028. Include asset hashes, metric namespaces, 204-game evaluation cost and unchanged PPO contract.

- [ ] **Step 2: Run all 0030 tests and diff checks**

Run the full project unittest discovery, JSON validation, `git diff --check`, and the official-engine dual gate.

- [ ] **Step 3: Commit only 0030 files**

Exclude all existing 0031 worktree changes and record the resulting source commit in the V3 decision entry.

- [ ] **Step 4: Launch formal V3**

Start from V2 update-5 model-only weights with a fresh optimizer, 512 training games/update, 128 workers, dual evaluation every five updates and online W&B in the foreground watchdog session.
