# 0019 Universal Winner BC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained winner-perspective universal BC project over every official Episode from 2026-07-10 onward, with explicit deck/source conditioning, compressed streaming materialization, loss-based early stopping, and hard disk guards.

**Architecture:** Freeze the 0016 R15 feature and action contracts into `train/0019_universal_winner_bc/` without runtime imports from another numbered project. Stream immutable daily ZIP members, retain exactly one completed winning trajectory per decisive Episode, condition the policy on exact registered deck plus normalized team/source identity, and write bounded gzip JSONL shards directly to `rl_runs/0019_universal_winner_bc/dataset/` without full extraction.

**Tech Stack:** Python 3.11, PyTorch, standard-library ZIP/gzip/JSON, unittest, TensorBoard, W&B online.

## Global Constraints

- Do not modify `engine/source/`.
- Do not interrupt the active 0018 PPO process or contend for its GPU.
- Official source archives remain compressed and immutable; no persistent full extraction.
- Final 0019 dataset must remain below 100 GiB and stop before C: free space falls below 30 GiB.
- Mixed decks and teams require explicit `deck_manifest` and `source_id` conditioning plus grouped metrics.
- Dataset rows use winner perspective only; draws, incomplete Episodes, malformed registrations, and invalid actions fail closed.
- BC selection uses full-validation loss with per-epoch validation and early stopping; online train metrics do not substitute for validation.
- Formal runs use private W&B project `dragon_bra/pokemon-tcg-policy-learning` and project/version run names.

---

### Task 1: Freeze The 0019 Project Contract

**Files:**
- Create: `train/0019_universal_winner_bc/`
- Create: `experiments/0019_universal_winner_bc/manifest.json`
- Create: `experiments/0019_universal_winner_bc/DESIGN.md`
- Create: `experiments/0019_universal_winner_bc/DESIGN.html`
- Test: `train/0019_universal_winner_bc/tests/test_project_contract.py`

**Interfaces:**
- Consumes: the immutable 0016 R15 feature/model implementation as source provenance only.
- Produces: self-contained imports under `train.0019_universal_winner_bc`, `PROJECT_ID = "0019_universal_winner_bc"`, and authoritative design documents.

- [ ] Copy only the tracked 0016 model, feature, knowledge, training, and data implementation files into the 0019 namespace; exclude caches, decks, checkpoints, and generated datasets.
- [ ] Replace every executable 0016 project/path/schema identifier with an explicit 0019 identifier while recording 0016 provenance in the manifest.
- [ ] Add a test that walks 0019 Python AST imports and rejects any `train.0016_*` or `train.0017_*` runtime dependency.
- [ ] Run `python3 -m unittest discover -s train/0019_universal_winner_bc/tests -p 'test_*.py'` and require success.

### Task 2: Build The Universal Winner Catalog

**Files:**
- Create: `train/0019_universal_winner_bc/data/replay_catalog.py`
- Create: `train/0019_universal_winner_bc/data/tests/test_replay_catalog.py`
- Create: `experiments/0019_universal_winner_bc/data_audit/archive_scan.json`

**Interfaces:**
- Consumes: `data/raw/episodes/archives/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD.zip` and optional audited patches.
- Produces: `rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner/source_catalog.json` with one winning trajectory per decisive Episode, exact 60-card deck hash, normalized source identity, seat, date, payload hash, and grouped split.

- [ ] Test decisive winner selection, draw exclusion, duplicate Episode de-duplication, 60-card registration validation, source normalization, and deterministic grouped split.
- [ ] Scan ZIP central directories first and record archive count, member count, compressed bytes, declared uncompressed bytes, date coverage, duplicates, and missing members without parsing all decisions.
- [ ] Parse Episodes serially while 0018 runs, assign validation by stable `(deck_hash, source_id, episode_id)` grouping, and fail closed on identity drift.
- [ ] Write the catalog atomically and include a canonical content SHA-256.

### Task 3: Stream Compressed Training Shards Under Disk Guards

**Files:**
- Create: `train/0019_universal_winner_bc/data/raw_dataset.py`
- Create: `train/0019_universal_winner_bc/storage.py`
- Create: `train/0019_universal_winner_bc/data/tests/test_raw_dataset.py`

**Interfaces:**
- Consumes: the immutable universal winner catalog.
- Produces: gzip JSONL train/validation shards plus `trajectory_index.json` and `dataset_reference.json` under `rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner_raw/`.

- [ ] Test winner-only rows, explicit `deck_manifest/source_id`, action ordering, per-trajectory split, 512 MiB target shard rollover, duplicate rejection, and partial-output cleanup.
- [ ] Add pre-write and per-shard guards for `<100 GiB` dataset bytes, Linux free space, and C: free space `>=30 GiB`.
- [ ] Stream one replay member at a time from ZIP to projected observation rows and directly into gzip output; never write expanded replay JSON to disk.
- [ ] Stop with an auditable `status.json` rather than leaving a dataset that looks complete after a guard or parse failure.

### Task 4: Add 0727 And 0728 Official Sources

**Files:**
- Create: `data/raw/episodes/archives/pokemon-tcg-ai-battle-episodes-2026-07-27.zip`
- Create after publication: `data/raw/episodes/archives/pokemon-tcg-ai-battle-episodes-2026-07-28.zip`
- Modify: `data/raw/episodes/source_manifest.json`
- Modify: `data/raw/episodes/index/manifest.csv`
- Modify: `data/raw/episodes/index/SHA256SUMS`

**Interfaces:**
- Consumes: private authenticated reads of official Kaggle datasets `kaggle/pokemon-tcg-ai-battle-episodes-YYYY-MM-DD`.
- Produces: immutable ZIPs, SHA-256, CRC result, member/byte audit, and credential-free download logs.

- [ ] Query dataset availability before downloading and make no external write or publication action.
- [ ] Download 0727 now; retry 0728 only after 2026-07-29 09:00 Asia/Shanghai if the official dataset exists.
- [ ] Validate archive CRC, JSON member count, declared uncompressed bytes, and exact date binding before updating manifests.
- [ ] Never record credentials, signed URLs, cookies, or access tokens.

### Task 5: Estimate Epoch Cost And Freeze The Training Entry

**Files:**
- Create: `train/0019_universal_winner_bc/run_r15.py`
- Create: `experiments/0019_universal_winner_bc/data_audit/training_estimate.json`
- Modify: `experiments/0019_universal_winner_bc/DESIGN.md`
- Modify: `experiments/0019_universal_winner_bc/DESIGN.html`

**Interfaces:**
- Consumes: dataset decision counts, shard bytes, 0016 measured decisions/s, and current model contract.
- Produces: epoch decisions, batches, projected read volume, train/validation minutes, early-stopping configuration, and tomorrow's launch command.

- [ ] Calculate exact train/validation decision and token counts after materialization.
- [ ] Estimate one epoch from 0016 measured throughput with low/base/high I/O scenarios and state every assumption.
- [ ] Configure loss-best checkpoint selection, full validation every epoch, early-stopping patience 5 and minimum delta 0.001 as the initial baseline.
- [ ] Keep checkpoint payload model-only by default and record W&B name `0019_universal_winner_bc · V1_<tag>`.

### Task 6: Audit And Remove Unreferenced Historical Checkpoints

**Files:**
- Create: `experiments/0019_universal_winner_bc/data_audit/checkpoint_cleanup.json`

**Interfaces:**
- Consumes: all repository path/hash references to 0016 and 0017 checkpoints.
- Produces: an exact keep/delete manifest, recovered byte count, and post-delete free-space snapshot.

- [ ] Resolve the two 0016 checkpoints and two 0017 VC inputs by exact path and SHA-256 from code, manifests, packages, and run metadata.
- [ ] Refuse deletion if the keep set is ambiguous or any proposed deletion is referenced by a current model/package/run.
- [ ] Record every deletion target and byte count before deleting only the authorized 0016/0017 checkpoint files and their sidecar metadata.
- [ ] Confirm all four keep files still hash correctly and record Linux/C: free space after cleanup.

### Task 7: Review And Handoff

**Files:**
- Create: `experiments/0019_universal_winner_bc/CODE_REVIEW.md`

**Interfaces:**
- Consumes: all 0019 implementation, tests, data audits, and smoke outputs.
- Produces: severity-ordered findings, resolved defects, remaining risks, tomorrow's exact training command, and go/no-go criteria.

- [ ] Review for source/deck leakage, split leakage, duplicate Episodes, action-contract drift, ZIP bomb risk, disk-guard races, W&B naming, checkpoint retention, and cross-project imports.
- [ ] Run focused 0019 tests, `python3 -m compileall -q train/0019_universal_winner_bc`, and a bounded end-to-end data smoke.
- [ ] Update both DESIGN documents with actual counts, shapes, stage, estimates, and evidence paths.
- [ ] Commit and immediately push the completed preparation without adding raw data, generated datasets, credentials, or model binaries.
