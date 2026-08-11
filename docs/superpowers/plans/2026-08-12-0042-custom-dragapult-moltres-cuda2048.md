# 0042 Custom Dragapult/Moltres CUDA-2048 Evaluation Plan

> **For Codex:** Execute this plan sequentially. Do not modify official engine sources or any PPO training semantics.

**Goal:** Canonicalize the supplied 60-card Dragapult/Moltres list, verify its CUDA rule coverage, then evaluate both V5 update 130 and the zero-gate Policy-0809 baseline against the immutable full Policy-0809 CUDA-2048 panel.

**Architecture:** Extend only the candidate deployment/evaluation surface so a caller may supply an audited exact-60 focal deck. Default 007 export behavior and hashes remain unchanged. Each candidate is materialized as FP16 storage and strict-loaded for FP32 runtime, while the opponent is resolved independently as full immutable Policy-0809. The two formal schedules retain the canonical identity-bound seed semantics and therefore have separate schedule hashes.

**Tech Stack:** Python, PyTorch, 0042 candidate exporter, resident CUDA official engine, unittest/pytest.

---

### Task 1: Freeze the card-ID and CUDA support audit

**Files:**
- Create: `.tmp/evaluation/0042_custom_dragapult_moltres/card_id_audit.json`
- Read: `data/official/EN_Card_Data.csv`
- Read: `.tmp/cuda_0032_rules/official_rules.json`
- Read: `engine_cuda/include/ptcg_cuda/official_targets_pod.cuh`
- Read: `engine_cuda/include/ptcg_cuda/official_effects_pod.cuh`

1. Resolve every supplied printing to the canonical numeric engine Card ID.
2. Assert the expanded deck contains exactly 60 cards and compute its exact-deck SHA-256.
3. Assert every canonical Card ID exists in the compiled CUDA rule inventory.
4. Record Moltres PFL 14 as Card ID 791, HP 120, Fighting Wings attack ID 1143, including the `kEx` condition and `kAttackDamageChange(+90)` interpreter paths.
5. Fail closed before evaluation if any identity or rule inventory assertion fails.

### Task 2: Add custom exact-deck candidate materialization without changing defaults

**Files:**
- Modify: `train/0042_full_model_design/export_full_semantic_candidate.py`
- Modify: `train/0042_full_model_design/candidate_deployment.py`
- Test: `train/0042_full_model_design/tests/test_export_full_semantic_candidate.py`
- Test: `train/0042_full_model_design/tests/test_candidate_deployment.py`

1. Add optional exact-deck/deck-ID/display/source arguments to deployment export and materialization.
2. Preserve byte-for-byte/default semantic behavior for the canonical Frozen 007 path.
3. For a custom focal deck, write its exact 60 IDs into `deck.csv`, classify its own-archetype independently, and bind the exact-deck SHA-256 into portable semantic metadata and the effective deployment identity.
4. Require manifest, portable metadata, and strict-loaded runtime to agree on custom deck identity.
5. Add regression tests for custom-deck hashing, own-archetype classification, and unchanged default behavior.

### Task 3: Add a formal custom-deck CUDA evaluator

**Files:**
- Create: `train/0042_full_model_design/diagnostics/fp16_cuda2048_custom_deck_policy0809.py`
- Test: `train/0042_full_model_design/tests/test_custom_deck_cuda_evaluator.py`

1. Load an exact-60 deck manifest and verify its hash/ID.
2. Materialize the checkpoint via `kaggle_fp16_storage_fp32_runtime_v1` using that deck.
3. Independently resolve full immutable Policy-0809 opponent weights and require identity audit PASS.
4. Build eight canonical 256-game units and run one resident 256-lane CUDA batch per unit.
5. Record per-game outcomes, opponent exact-deck hashes, first-player evidence, unsupported effects, semantic fallbacks, feature residency, throughput, deployment identity, and schedule hash.
6. Require 2,048 terminal games, zero error, zero unsupported effect, zero semantic fallback (apart from explicitly allowed chance boundaries), and passing lane-routing identity.

### Task 4: Preflight and execute the paired evaluations

**Files:**
- Output: `.tmp/evaluation/0042_custom_dragapult_moltres/v5_u130_cuda2048/`
- Output: `.tmp/evaluation/0042_custom_dragapult_moltres/policy0809_u0_cuda2048/`

1. Run unit tests and a 256-game custom-deck CUDA smoke on V5 U130.
2. If healthy, run V5 U130 for all eight canonical units.
3. Run the V1 update-0 zero-gate checkpoint as the Policy-0809-equivalent 0042 baseline on the same custom deck for all eight units.
4. Do not reuse focal/opponent mutable modules or storage and do not modify PPO state.
5. Report both results separately, including the reason their identity-bound formal schedule hashes differ.

### Task 5: Final audit report

1. Provide the full requested-print to canonical-ID mapping, distinguishing exact-print matches from canonical reprint aliases.
2. State the Moltres CPU card-text fact, CUDA implementation fact, and remaining evidence boundary separately.
3. Report V5 U130 and Policy-0809 baseline W-L-D, win rate, Wilson interval, first/second rates, throughput, health gates, exact focal deck hash, candidate identity, opponent identity, and schedule hash.
4. Make no promotion decision and do not claim paired-game deltas because formal schedules are deployment-identity bound.
