# Policy-0814 Internal Pretrained Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish `n-m-0814.zip` as a self-contained immutable internal `Policy-0814` BC actor plus its exactly bound pretrained Value network.

**Architecture:** Mirror the established `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/` release boundary. Preserve both supplied model-only checkpoints byte-for-byte, vendor the compatible 0031 inference source and 0036 Value source, and make the release fail closed on file hashes, actor/Value schemas, exact actor-to-critic binding, 0809 architecture parity, component identity, and minimal forwards.

**Tech Stack:** Python 3, PyTorch model-only checkpoints, JSON manifests, SHA-256, repository-local self-contained loaders.

## Global Constraints

- `Policy-0814` is a new immutable identity; existing `Policy-0809`, Frozen policies, champions, and numbered-project registries remain unchanged.
- BC/source identity remains provenance only and is not added to actor-visible inputs.
- Actor identity covers the complete effective 0031 inference policy; the separate Value network is a paired critic artifact and is not silently folded into actor logits.
- Both supplied checkpoint files remain byte-for-byte identical to the members of `n-m-0814.zip`.
- No optimizer, scheduler, scaler, RNG, replay, rollout buffer, or exact-resume state is admitted.
- Offline BC/Value metrics are not official-engine policy-strength evidence.
- No Git commit or external publication is performed automatically.

---

### Task 1: Freeze the source audit and release boundary

**Files:**
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/checkpoint_metadata.json`
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/contracts/local_0031_model_contract.json`

**Interfaces:**
- Consumes: `n-m-0814.zip` members `best_validation_loss_0814.pt` and `best_validation_value_loss_0814.pt`.
- Produces: immutable policy/value identities and the exact 0031 architecture contract consumed by the loader and manifest.

- [x] Verify archive path safety and member integrity with `python3 -m zipfile -t n-m-0814.zip`.
- [x] Load both members with `torch.load(..., weights_only=True)` and reject any non-model-only top-level keys.
- [x] Compare actor and Value state keys, shapes, and dtypes against the internal 0809 release; require exact equality of the structural inventories.
- [x] Record policy epoch 23/version `V6_medal_zone_gsb_all_days_to_0810`, Value epoch 13/version `V10_v6_e23_best_val_encoder_latent_value_archetype_diff`, validation metrics, dataset identities, byte sizes, and SHA-256 values.
- [x] Record the exact 0031 config (`d_model=320`, 8 heads, four state layers, two option layers) and parameter contracts (actor 56,352,322; Value 3,407,389).

### Task 2: Build the self-contained release

**Files:**
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/model.pt`
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/value_head.pt`
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/model_source/**`
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/value_source/**`
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/assets/**`
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/loader.py`

**Interfaces:**
- Consumes: Task 1 identities and the inference-compatible 0031/0036 source layout established by the 0809 release.
- Produces: `load_policy_checkpoint()`, `load_value_checkpoint()`, `load_policy()`, `load_value_network()`, and `load_pair()`.

- [x] Copy the supplied checkpoints byte-for-byte as `model.pt` and `value_head.pt`.
- [x] Vendor physical copies of the 0031 model/feature/contract/domain/knowledge source, current official prototype assets, and the 0036 Value implementation; exclude caches and symlinks.
- [x] Implement strict SHA/schema/metadata checks for `Policy-0814` and require the Value `source_checkpoint_sha256` to equal the released BC checkpoint hash.
- [x] Strict-load every actor and Value tensor and require the declared/actual parameter counts.
- [x] Keep `load_pair()` modules independent while binding both to the same immutable actor weights.

### Task 3: Publish immutable identity and documentation

**Files:**
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/manifest.json`
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/README.md`

**Interfaces:**
- Consumes: all Task 2 release files.
- Produces: a human-readable release contract plus machine-auditable file, implementation-bundle, actor-component, effective-policy, and actor/Value pair hashes.

- [x] Compute all file SHA-256 values and tree hashes over `model_source` and `value_source`.
- [x] Compute 0031 component hashes for prototype encoder, state encoder, option input, both option transformer layers, final norm, empty LoRA, and action decoder.
- [x] Register the immutable actor identity as `Policy-0814`; record the companion Value checkpoint separately with exact actor-source binding.
- [x] Document architecture parity with 0809 as exact state-key/shape/dtype parity, not weight equality or strength evidence.
- [x] State that project-local policy resolvers must explicitly import/register this identity before it can be used as an RL focal/opponent or Frozen benchmark.

### Task 4: Add and run fail-closed release verification

**Files:**
- Create: `archive/pretrained/0031_friend_0814_gsb_v6_value_v10/verify_archive.py`

**Interfaces:**
- Consumes: the completed release directory and manifest.
- Produces: a zero-exit verification receipt printed only after every invariant passes.

- [x] Verify every manifest file hash, implementation tree hash, absence of symlinks/caches, and absence of resumable training state.
- [x] Recompute actor component hashes/effective identity and require exact manifest equality.
- [x] Re-run actor/Value structural parity against the declared 0809 architecture contract without importing a numbered project at runtime.
- [x] Strict-load both networks, require the encoder to remain frozen inside the Value network, and run finite minimal actor and Value forwards with exact output shapes.
- [x] Run `python3 verify_archive.py` from the release directory and inspect `git status --short` to confirm no existing 0809/frozen/champion asset was modified.
