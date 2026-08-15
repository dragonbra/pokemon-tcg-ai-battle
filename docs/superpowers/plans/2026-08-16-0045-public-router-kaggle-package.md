# 0045 Public Router Kaggle Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the accepted Deck-007 public-information expert router as one compact, self-contained Kaggle `.tar.gz` using FP16 storage and FP32 inference.

**Architecture:** Store the complete U282 portable candidate once as the immutable default/shared backbone, then store only Policy Q/V LoRA, Action Decoder, and Allocation Head FP16 state for U40/U90/U200. The CPU Agent strict-loads the U282 candidate into FP32, creates three same-shape FP32 head modules, loads the delta states, verifies all source/component/composite identities, and routes only from public official observations; every unrecognized case remains U282.

**Tech Stack:** Python 3, PyTorch, existing 0045 semantic runtime, official packaged `cg` runtime, pytest, tar/gzip.

## Global Constraints

- Do not modify `engine/source/` or CUDA engine source.
- Keep Deck 007 exact 60-card identity unchanged.
- Use `kaggle_fp16_storage_fp32_runtime_v1`: source FP32 → FP16 artifact → strict FP32 runtime.
- Route only the explicitly accepted public rules; every other observation remains U282.
- Share modules only after exact effective-weight equality has been proven.
- Do not overwrite existing archive/submission assets.
- Do not submit to Kaggle; provide the command for the user.

---

### Task 1: Compact routed artifact contract

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/public_router_candidate.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/semantic_runtime/deployment/public_meta_router.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_public_router_candidate.py`

**Interfaces:**
- Consumes four existing `0045_minimal_lora_candidate_v1` FP16 portable checkpoints.
- Produces `0045_public_meta_router_compact_candidate_v1` and `PublicRoutedCompoundPolicy.from_compact_checkpoint(path, deck)`.

- [ ] Write tests that reject missing updates, FP32 stored tensors, non-identical shared tensors, modified routing manifest, and malformed delta keys.
- [ ] Implement exact tensor partitioning into one U282 default payload plus U40/U90/U200 routed head deltas.
- [ ] Hash source artifacts, shared tensors, each routed head, routing rules, and the final composite identity.
- [ ] Strict-load the compact artifact into FP32 modules without constructing four complete policies.
- [ ] Prove every loaded runtime tensor is FP32 and every stored floating tensor is FP16.
- [ ] Run focused tests.

### Task 2: Formal Kaggle exporter and entrypoint

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/export_public_router_kaggle.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_public_router_kaggle_export.py`

**Interfaces:**
- Consumes source checkpoints U40/U90/U200/U282 and the V12 qualifying identity report.
- Produces a self-contained candidate directory plus `.tar.gz`, manifest schema `0045_public_meta_router_kaggle_package_v1`, and an Agent entrypoint loading the compact policy.

- [ ] Write an isolated-import test proving the generated `main.py` loads the routed policy, returns the exact 60-card deck, and resets memory at a new game.
- [ ] Implement export through the existing per-checkpoint materializer, then compact only those exact FP16 artifacts.
- [ ] Bind the manifest to source hashes, portable hashes, component hashes, rule hash, composite identity, package inventory, Deck 007, and V12 evidence.
- [ ] Create a deterministic archive without overwriting any existing submission asset.
- [ ] Run package validator and isolated archive import.

### Task 3: Runtime and official-engine gates

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/tests/test_public_meta_router_v1.py`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/decisions.md`

**Interfaces:**
- Consumes the final unpacked package.
- Produces validation receipts and the final archive/hash/size/command handoff.

- [ ] Verify every card outside the accepted trigger manifest remains U282; Meta 28 receives no separate benchmark requirement.
- [ ] Verify U40/U90/U200/U282 compact heads equal their qualifying portable sources tensor-for-tensor after FP16→FP32 load.
- [ ] Verify routing reset, monotonicity, Critic-free Actor behavior, and decoder state locality.
- [ ] Run the full 0045 regression suite relevant to deployment.
- [ ] Run `evaluation validate` and a short official-engine package smoke.
- [ ] Audit archive inventory, compressed size, import RSS/load time, and SHA-256.
- [ ] Synchronize DESIGN/decisions with the final schema and validation evidence.
- [ ] Deliver the `.tar.gz` path and exact Kaggle submission command without executing it.
