# 0043 Champion-G2 Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the human-approved U407 candidate into an immutable `Champion-G2`, register it as the latest champion, and make it safely available to future CPU/CUDA opponent pools for exact decks 001–067.

**Architecture:** A fail-closed promotion command validates the complete Full67 evidence and explicit human decision, freezes source and portable artifacts under `assets/policies/definitions/champion_g002/`, writes a bound promotion record and manifest, then atomically replaces the registry. The existing resolver and CUDA resident cache become registry-driven for admitted compound champions while preserving Policy-0809, Champion-G1, and all historical run manifests.

**Tech Stack:** Python 3, PyTorch model-only checkpoints, JSON manifests, pytest, CUDA Engine 2.0.

## Global Constraints

- Follow `RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1` and `0043_Promote_Champion_V2_Protocol` fail-closed.
- Preserve `Policy-0809` and `Champion-G1` as immutable independent identities; never overwrite or alias their tensor storage.
- Store G2 under semantic path `assets/policies/definitions/champion_g002/`; hashes are identity fields, never directory names.
- The admitted artifact must use `kaggle_fp16_storage_fp32_runtime_v1`: FP16 storage, strict FP32 runtime.
- Only own-deck conditioning uses the 29-class vocabulary; the frozen 15-way opponent Meta head remains unchanged.
- Exact deck IDs remain `001`–`067`; one resident G2 model serves deck-routed batches without per-deck checkpoint reload.
- Do not modify `engine/source/`, historical run configurations, or the evaluation opponent catalog.

---

### Task 1: Promotion gate and immutable record

**Files:**
- Create: `train/0043_champion_league_rl/promote/champion_g2.py`
- Modify: `train/0043_champion_league_rl/promote/__init__.py`
- Modify: `train/0043_champion_league_rl/tests/test_promote.py`

**Interfaces:**
- Consumes: U407 model-only checkpoint, Full67 per-deck reports, authoritative HTML evidence, and a bound human `PROMOTE` decision.
- Produces: `promote_champion_g2(repository_root: Path, decision: Mapping[str, Any]) -> dict[str, Any]` and a dry-run validation result.

- [ ] Add failing tests that reject incomplete reports, wrong U407 SHA, non-PROMOTE decisions, deck/effective identity mismatches, and an already-used generation.
- [ ] Run `python3 -m pytest train/0043_champion_league_rl/tests/test_promote.py -q` and confirm the new tests fail.
- [ ] Implement Full67 evidence collection, candidate freeze, decision binding, and atomic artifact/manifest/registry writes.
- [ ] Re-run the promotion tests and confirm all pass.

### Task 2: Registry-driven identity and runtime loading

**Files:**
- Modify: `train/0043_champion_league_rl/assets.py`
- Modify: `train/0043_champion_league_rl/policy_identity.py`
- Modify: `train/0043_champion_league_rl/runtime.py`
- Modify: `train/0043_champion_league_rl/tests/test_assets.py`
- Modify: `train/0043_champion_league_rl/tests/test_policy_identity.py`
- Modify: `train/0043_champion_league_rl/tests/test_runtime.py`

**Interfaces:**
- Consumes: admitted policy manifest roles, generation, portable/source artifact schemas, and 29-way own taxonomy.
- Produces: generic immutable compound-champion materialization and exact-deck runtime loading for G1 and G2.

- [ ] Add failing tests for a three-policy active pool, G1 preservation, G2 FP16/FP32 identity, and deck-specific G2 own IDs.
- [ ] Run the targeted tests and confirm the old two-policy assumptions fail.
- [ ] Generalize registry validation and compound policy loading without weakening artifact path/hash/provenance checks.
- [ ] Re-run the targeted tests and verify storage isolation across all three frozen policies.

### Task 3: CUDA resident routing and PPO identity acceptance

**Files:**
- Modify: `train/0043_champion_league_rl/cuda_engine_2/inference.py`
- Modify: `train/0043_champion_league_rl/rollout/cuda_collector.py`
- Modify: `train/0043_champion_league_rl/training/batch_full_semantic.py`
- Modify: `train/0043_champion_league_rl/preflight.py`
- Modify: `train/0043_champion_league_rl/tests/test_cuda_engine_2_policy_pool.py`

**Interfaces:**
- Consumes: `AssetRegistry.active_policy_ids` and compound-policy capability rather than a hard-coded G1 ID.
- Produces: one resident load per active policy, per-row own-deck IDs, and identity-complete PPO trajectory acceptance for G2.

- [ ] Add failing tests proving G2 is warmed once, deck IDs remain row-aligned, and compound strategy routing applies to both champions.
- [ ] Run the targeted CUDA/pool tests and confirm failure before implementation.
- [ ] Replace hard-coded admitted IDs with registry-derived IDs and route compound policies by capability.
- [ ] Run CPU tests plus a small CUDA forward/resident smoke with bounded memory.

### Task 4: Execute promotion and publish audit trail

**Files:**
- Create: `train/0043_champion_league_rl/assets/policies/definitions/champion_g002/source_update_000407.pt`
- Create: `train/0043_champion_league_rl/assets/policies/definitions/champion_g002/model.bin`
- Create: `train/0043_champion_league_rl/assets/policies/manifests/champion_g002.json`
- Create: `experiments/0043_champion_league_rl/promotion/champion_g002_u407.json`
- Modify: `train/0043_champion_league_rl/assets/policies/registry.json`
- Modify: `rl_runs/0043_champion_league_rl/versions/V8_u407_g2_candidate_full67_cuda2048/artifact/state.json`
- Modify: `rl_runs/0043_champion_league_rl/versions/V8_u407_g2_candidate_full67_cuda2048/artifact/evaluation.json`

**Interfaces:**
- Consumes: Tasks 1–3 and the user's explicit human `PROMOTE` decision.
- Produces: immutable Champion-G2 assets, latest pointer `Champion-G2`, active pool `[Policy-0809, Champion-G1, Champion-G2]`, and an evidence-bound decision record.

- [ ] Run the promotion command once with the exact evidence SHA and human decision fields.
- [ ] Validate all asset hashes, reconstruct G2 for every deck 001–067, and compare each deployment-effective hash with its evaluated report.
- [ ] Confirm a second promotion attempt fails closed instead of overwriting G2.
- [ ] Record V8 as `PROMOTED_AS_CHAMPION_G2` and retain its original evaluation evidence.

### Task 5: Authoritative documentation and final verification

**Files:**
- Modify: `experiments/0043_champion_league_rl/DESIGN.md`
- Modify: `experiments/0043_champion_league_rl/DESIGN.html`
- Modify: `experiments/0043_champion_league_rl/DECISIONS.md`

**Interfaces:**
- Consumes: final manifest hashes, promotion record, active pool, latest pointer, and CUDA smoke results.
- Produces: synchronized human-readable policy lifecycle and future-training semantics.

- [ ] Document U407 → Champion-G2, the three-policy pool, G2 latest-champion pressure, 29-way own-only conditioning, and the unchanged 15-way opponent head.
- [ ] Run all 0043 policy/promotion/routing tests and `python3 -m train.0043_champion_league_rl.preflight --gpu-smoke`.
- [ ] Inspect `git diff --check`, verify no unrelated files or official engine source changed, and report exact asset/evidence hashes.
