# 0019 Universal Winner Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a validated, model-ready 0019 universal winner-perspective BC dataset from every official Episode archive dated 2026-07-10 through 2026-07-28, with reproducible manifests and a measured training-time estimate.

**Architecture:** Official ZIP archives remain immutable inputs and are read without full extraction. A deterministic catalog assigns trajectory split, exact-deck ID, and Unicode-normalized source/team ID over the complete archive set; raw causal decision shards and model-ready tensor shards are then rebuilt atomically from that frozen catalog. Because adding an archive can reorder source IDs, the final 0710-0728 dataset is rebuilt as one unit rather than appended to the 0710-0727 tensors.

**Tech Stack:** Python 3.11, PyTorch, official Kaggle Episode ZIPs, repository-local 0019 replay compiler, JSON/JSONL manifests, SHA-256 and ZIP CRC validation.

## Global Constraints

- Do not interrupt `0018_alakazam_terminal_rl` PPO or contend for its GPU.
- Use winner perspectives only; exclude Episodes without exactly one positive-reward winner.
- Condition on exact 60-card deck and NFC-normalized exact source/team; reserve source ID 0 for neutral deployment.
- Split entire trajectories deterministically with SHA-256 into 90% train and 10% validation.
- Keep final model-ready data below 100 GiB, Linux free space above 50 GiB, and C drive free space above 30 GiB.
- Never extract all archives to disk; stream Episode JSON members from ZIP files.
- Save model-only checkpoints; do not save optimizer, RNG, or resume state.
- Do not modify `engine/source/`.
- Do not upload data, create a Kaggle submission, or perform another externally visible competition action.

---

### Task 1: Finish The 0710-0727 Baseline Materialization

**Files:**
- Verify: `rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner_raw/`
- Verify: `rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner/`
- Verify: `rl_runs/0019_universal_winner_bc/dataset/source_catalog.json`

**Interfaces:**
- Consumes: the already-running materialized builder process and 0710-0727 frozen catalog.
- Produces: an immutable baseline audit used to measure actual tensor bytes and build throughput.

- [x] **Step 1: Let the active builder exit naturally**

Poll the existing exec session and process tree; do not signal or restart it.

- [x] **Step 2: Validate every output shard**

Run:

```bash
python3 -m train.0019_universal_winner_bc.training.materialized validate \
  rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner
```

Expected: exit code 0, all manifest hashes and tensor contracts valid.

- [x] **Step 3: Record exact baseline size and elapsed time**

Write the measured 0710-0727 counts, decisions, bytes, and stage durations into the project data audit before replacing the reproducible outputs.

### Task 2: Download And Verify The 0728 Official Archive

**Files:**
- Create: `data/raw/episodes/archives/pokemon-tcg-ai-battle-episodes-2026-07-28.zip`
- Create: `data/raw/episodes/download_logs/pokemon-tcg-ai-battle-episodes-2026-07-28.log`
- Modify: `data/raw/episodes/source_manifest.json`
- Modify: `data/raw/episodes/index/SHA256SUMS`
- Modify only if an official index row exists: `data/raw/episodes/manifest.csv`

**Interfaces:**
- Consumes: Kaggle dataset `kaggle/pokemon-tcg-ai-battle-episodes-2026-07-28` after 2026-07-29 09:00 CST.
- Produces: one credential-free, CRC-verified official archive entry with SHA-256, JSON member count, and declared uncompressed bytes.

- [x] **Step 1: Check availability after 09:00 CST**

Use the authenticated Kaggle CLI only after the release window. Do not submit or upload anything.

- [x] **Step 2: Download into the archive directory**

Download the official dataset payload to `data/raw/episodes/archives/` without unpacking the full JSON corpus.

- [x] **Step 3: Verify ZIP integrity and inventory**

Run full CRC validation, count JSON members, sum declared uncompressed bytes, and calculate SHA-256. Reject partial or corrupt downloads.

- [x] **Step 4: Update local provenance manifests**

Record only non-secret dataset identifiers, hashes, counts, bytes, timestamps, and file paths. Never write the Kaggle access token into logs.

### Task 3: Rebuild The Final 0710-0728 Dataset

**Files:**
- Replace reproducibly: `rl_runs/0019_universal_winner_bc/dataset/source_catalog.json`
- Replace reproducibly: `rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner_raw/`
- Replace reproducibly: `rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner/`

**Interfaces:**
- Consumes: all verified official archives from 0710 through 0728 plus the audited replay gap patch.
- Produces: stable catalog, compressed causal rows, and immutable tensor shards with IDs derived from the complete source universe.

- [x] **Step 1: Preserve the completed baseline audit**

Confirm exact 0710-0727 counts, content hashes, apparent bytes, and build durations are recorded before removing reproducible outputs.

- [x] **Step 2: Remove only the three reproducible 0019 data products**

Delete the catalog, raw dataset, and model-ready dataset by their exact paths. Preserve official ZIPs, experiment documentation, audits, 0015 retained checkpoint, and every 0018 checkpoint.

- [x] **Step 3: Build the complete catalog**

Run:

```bash
python3 -m train.0019_universal_winner_bc.data.replay_catalog \
  --archives-dir data/raw/episodes/archives \
  --start-date 2026-07-10 \
  --end-date 2026-07-28 \
  --patches-dir data/raw/episodes/patches \
  --workers 4 \
  --output rl_runs/0019_universal_winner_bc/dataset/source_catalog.json
```

Expected: all Episode IDs unique, exact winner and 60-card deck contracts enforced, excluded Episodes explicitly counted.

- [x] **Step 4: Build compressed raw causal rows**

Run:

```bash
python3 -m train.0019_universal_winner_bc.data.raw_dataset \
  --catalog rl_runs/0019_universal_winner_bc/dataset/source_catalog.json \
  --output rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner_raw \
  --workers 8 \
  --shard-size 100000
```

Expected: atomic output, complete trajectory index, and no actor leakage between perspectives.

- [x] **Step 5: Build immutable tensor shards**

Run:

```bash
python3 -m train.0019_universal_winner_bc.training.materialized build \
  --raw rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner_raw \
  --output rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner \
  --workers 8 \
  --shard-records 1024
```

Expected: hidden staging directory is atomically renamed only after all shards and hashes are complete.

### Task 4: Validate, Benchmark, And Estimate Training

**Files:**
- Modify: `experiments/0019_universal_winner_bc/data_audit/training_estimate.json`
- Modify: `experiments/0019_universal_winner_bc/manifest.json`

**Interfaces:**
- Consumes: final model-ready manifest and, after 0018 releases the GPU, one bounded R15 training smoke.
- Produces: exact dataset scale, bytes per decision, measured decisions/second, epoch duration, and 8/15-epoch wall-clock ranges.

- [x] **Step 1: Run full model-ready validation**

```bash
python3 -m train.0019_universal_winner_bc.training.materialized validate \
  rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner
```

Expected: exit code 0 and no hash, dtype, shape, split, source, or deck failures.

- [x] **Step 2: Run a bounded GPU throughput smoke after PPO exits**

```bash
python3 -m train.0019_universal_winner_bc.run_r15 \
  --smoke \
  --smoke-tag final_0710_0728_throughput \
  --batch-size 128 \
  --validation-batch-size 256
```

Expected: forward, backward, checkpoint, logger, and neutral validation paths complete without altering a formal training version.

- [x] **Step 3: Calculate ranges from measured throughput**

Use full train and validation decision counts and smoke throughput. Report data-build time separately from training time; include early-stopping uncertainty rather than presenting a single false-precision ETA.

### Task 5: Synchronize Design And Verify The Project

**Files:**
- Modify: `experiments/0019_universal_winner_bc/DESIGN.md`
- Modify: `experiments/0019_universal_winner_bc/DESIGN.html`
- Modify: `experiments/0019_universal_winner_bc/CODE_REVIEW.md` only if new findings arise.

**Interfaces:**
- Consumes: final audit, current code, model shape, and checkpoint contract.
- Produces: matching Markdown/HTML authority documents and a tested, pushed repository state.

- [x] **Step 1: Synchronize exact data facts and stage status**

Update both DESIGN documents with final 0710-0728 counts, tensor bytes, conditioning contract, validation semantics, checkpoint policy, training estimate, and next single-variable experiments.

- [x] **Step 2: Run project tests**

```bash
python3 -m unittest -v \
  train.0019_universal_winner_bc.data.tests.test_replay_catalog \
  train.0019_universal_winner_bc.tests.test_project_contract \
  train.0019_universal_winner_bc.tests.test_storage \
  train.0019_universal_winner_bc.tests.test_checkpoints
python3 -m compileall -q train/0019_universal_winner_bc
```

Expected: every test passes and compileall exits 0.

- [ ] **Step 3: Commit and push tracked audit/document changes**

```bash
git add docs/superpowers/plans/2026-07-29-0019-universal-winner-data.md \
  experiments/0019_universal_winner_bc
git commit -m "data: 完成0019通用胜者数据准备"
git push
```

Expected: `origin/dev/cyd_main` contains the tracked experiment audit and design state. The local
credential-free source manifest and checksum index are updated but remain ignored with the official
ZIPs, tensor shards, W&B staging, and model checkpoints; never force-add those assets.
