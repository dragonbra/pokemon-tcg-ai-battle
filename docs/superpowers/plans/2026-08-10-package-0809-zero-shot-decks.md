# 0809 Zero-Shot Kaggle Packages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute these checked steps in order; do not submit any package to Kaggle.

**Goal:** Build self-contained Kaggle-ready 0809 zero-shot packages for Frozen-0806 exact deck slots 003, 007, and 009.

**Architecture:** Reuse the audited 0031 portable export path because the 0809 checkpoint has the same model configuration, implementation identity, feature schema, and action contract as 0806. Export the actor with fp16 storage and fp32 runtime, bind one exact 60-card deck per package, physically copy the standard `cg/` runtime, and create root-layout tarballs under `archive/submission/dist/`.

**Tech Stack:** Python 3.11, PyTorch, official `cg` runtime, repository evaluation validator, gzip tar archives.

## Global Constraints

- Do not modify `engine/source/`.
- Do not include the 0036 Value network because zero-shot action inference consumes only the 0031 actor policy.
- Each package must be self-contained and contain no symlink, absolute runtime path, optimizer state, persisted cache tensor, trace, or bytecode. The model-owned, non-persistent Frozen Prototype Embedding Cache runtime implementation is mandatory.
- `main.py`, `deck.csv`, `cg/`, `strategy/`, and `manifest.json` must be at the tar extraction root.
- `deck.csv` must contain exactly 60 legal card-ID rows, and the registration callback must return the same ordered deck.
- Creating these local packages does not authorize a Kaggle upload or submission.

---

### Task 1: Freeze source identities

**Files:**
- Read: `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt`
- Read: `evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks/003_mega_lopunny_ex_mega_froslass_ex/deck.csv`
- Read: `evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks/007_dragapult_ex/deck.csv`
- Read: `evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks/009_mega_lucario_ex_solrock/deck.csv`

**Interfaces:**
- Consumes: audited 0809 checkpoint metadata and frozen deck manifests.
- Produces: immutable checkpoint/deck IDs used by all three exports.

- [x] Confirm the policy schema is `0031_model_only_checkpoint_v1` and the implementation hash equals the admitted 0031 runtime.
- [x] Confirm slots 003, 007, and 009 each have exactly 60 rows and match their logical exact-deck SHA-256 identities.

### Task 2: Export three self-contained payload directories

**Files:**
- Create: `archive/submission/0031_zero_shot_0809_003_mega_lopunny_ex_mega_froslass_ex_fp16_storage_fp32_runtime/`
- Create: `archive/submission/0031_zero_shot_0809_007_dragapult_ex_fp16_storage_fp32_runtime/`
- Create: `archive/submission/0031_zero_shot_0809_009_mega_lucario_ex_solrock_fp16_storage_fp32_runtime/`

**Interfaces:**
- Consumes: `export_candidate(checkpoint, deck_path, cg_source, output, deck_id, storage_dtype="fp16", runtime_dtype="fp32")`.
- Produces: three runnable package roots with independent physical `cg/` and `strategy/` trees.

- [x] Run the 0031 exporter once per exact deck with the 0809 checkpoint.
- [x] Confirm each manifest records checkpoint SHA-256 `926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f`, epoch 20, global step 115820, fp16 storage, fp32 runtime, and no optimizer state.
- [x] Confirm the portable actor checkpoint is identical across the three packages and only `deck.csv`, deck identity, and candidate name differ.

### Task 3: Validate package runtime contracts

**Files:**
- Test: each package directory created in Task 2.

**Interfaces:**
- Consumes: the three exported payload roots.
- Produces: validator results proving layout, exact deck registration, imports, and model initialization.

- [x] Run `python3 -m evaluation validate <package>` for all three packages.
- [x] Independently import each `main.py`, call `read_deck_csv()` and `agent({"select": null})`, and require identical 60-card lists.
- [x] Reject symlinks, `__pycache__`, `.pyc`, optimizer/replay artifacts, and repository absolute paths in runtime metadata.
- [x] Require `frozen_model_owned_v1`, verify one build plus cross-forward hits, and prove cached/direct logits and greedy actions are unchanged.

### Task 4: Build and audit final tarballs

**Files:**
- Create: `archive/submission/dist/0031_zero_shot_0809_003_mega_lopunny_ex_mega_froslass_ex_fp16_storage_fp32_runtime.tar.gz`
- Create: `archive/submission/dist/0031_zero_shot_0809_007_dragapult_ex_fp16_storage_fp32_runtime.tar.gz`
- Create: `archive/submission/dist/0031_zero_shot_0809_009_mega_lucario_ex_solrock_fp16_storage_fp32_runtime.tar.gz`

**Interfaces:**
- Consumes: validated directories from Task 3.
- Produces: three Kaggle-ready root-layout archives and their SHA-256 checksums.

- [x] Create each gzip tar from inside its package root so no enclosing project directory is added.
- [x] List every archive and require root-level `main.py`, `deck.csv`, `manifest.json`, `cg/`, and `strategy/model.bin`.
- [x] Extract each archive to an isolated temporary directory and rerun registration/import validation without repository `PYTHONPATH` assistance.
- [x] Record final byte sizes and SHA-256 checksums for delivery.

Final archive SHA-256 after the cache-contract rebuild:

- 003: `de4e1b895db94105ed414dcca7daeaefba26a770c2c1d1c5888eefa9ca3943d6`
- 007: `ac0b700c81cf6ee763d207d8e1cf55b354353d8dcb34852dab4af407fcb64ba2`
- 009: `4c2583e86000ef0805803769845d73bff2c730b059bdc99e535c8a3f35c84395`
