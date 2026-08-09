# 0809 Paired Policy and Value Pretrained Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute these checked steps in order and preserve every source checkpoint byte-for-byte.

**Goal:** Audit `new-model-0809.tar.gz` and publish its paired 0031 policy and 0036 Value network as one self-contained, hash-committed pretrained release.

**Architecture:** Preserve the two supplied model-only checkpoints without serialization changes. Reuse the frozen 0031 model implementation and prototype assets already proven by the 0806 release, add the matching 0036 latent-query Value implementation, and expose strict loaders that bind the critic to the exact policy SHA-256 declared by its metadata.

**Tech Stack:** Python 3.11, PyTorch model-only checkpoints, JSON manifests, SHA-256 verification.

## Global Constraints

- Do not modify `engine/source/`.
- Do not create or overwrite an `rl_runs` version for externally trained V5/V9 assets.
- Preserve both source checkpoint files byte-for-byte and reject optimizer, scheduler, scaler, RNG, rollout, or replay state.
- Treat offline validation metrics as imitation/calibration evidence, not official-engine policy-strength evidence.
- Keep actor source identity out of the forward path.

---

### Task 1: Audit checkpoint identity and compatibility

**Files:**
- Read: `new-model-0809.tar.gz`
- Read: `train/0031_rule_faithful_semantic_foundation_pretraining/`
- Read: `train/0036_dedicated_action_value_network/`

**Interfaces:**
- Consumes: the two tar members and their embedded metadata.
- Produces: immutable archive/member hashes, schema identities, source-policy linkage, tensor shapes, and model configs used by Tasks 2–3.

- [x] Verify the tar has exactly `V5_gsb_best_validation.pt` and `V9_value_head_best_validation_value_loss.pt`.
- [x] Load both with `torch.load(..., weights_only=True)` and reject unexpected top-level keys.
- [x] Confirm the policy is 0031 V5/epoch 20 and the critic is 0036 V9/epoch 9.
- [x] Confirm the critic's `source_checkpoint_sha256` equals the policy member SHA-256.

### Task 2: Build the self-contained 0809 pretrained release

**Files:**
- Create: `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/README.md`
- Create: `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/manifest.json`
- Create: `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/checkpoint_metadata.json`
- Create: `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/loader.py`
- Create: `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/verify_archive.py`
- Create: `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt`
- Create: `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/value_head.pt`
- Copy: 0031 model source, contracts, and prototype assets into the release.
- Copy: the 0036 latent-query Value implementation into `value_source/` with its model-only schema boundary.

**Interfaces:**
- Consumes: hashes and metadata from Task 1.
- Produces: `load_policy(device)` and `load_value_network(device)` strict loader functions.

- [x] Copy both checkpoints without re-saving them and verify copied SHA-256 values.
- [x] Snapshot only the implementation/assets needed to reconstruct policy and critic inference.
- [x] Implement strict identity, schema, source-link, parameter-count, and state-dict loading checks.
- [x] Write a manifest that distinguishes 0031 policy provenance from 0036 critic provenance.

### Task 3: Verify the release contract

**Files:**
- Test: `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/verify_archive.py`

**Interfaces:**
- Consumes: the complete release from Task 2.
- Produces: a passing hash/model-only/strict-load/minimal-forward verification result.

- [x] Verify every manifest file hash and both implementation bundle hashes.
- [x] Reject symlinks and forbidden resumable-training fields.
- [x] Strict-load the policy and paired critic on CPU.
- [x] Run a minimal 0031 policy forward and critic forward; require finite policy logits, value logit, win probability/value, archetype logits, and final-difference logits with exact expected shapes.
- [x] Run focused 0031/0036 model and checkpoint tests to ensure the release did not alter project contracts.
