# 0015 0726 Episode Data Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download and audit the official 2026-07-26 Episode dataset, then build a deduplicated 0015 source overlay for the named Dragapult experts.

**Architecture:** Keep the official Kaggle ZIP immutable in `data/raw/episodes/archives/`, index its Episode IDs without full extraction, and use the submission-scoped Episode API only for completed public replays absent from that ZIP. Preserve exact submission, team, player index, 60-card deck hash and build label for every player-side trajectory; create lightweight 0015 manifests rather than copying the official archive.

**Tech Stack:** Kaggle CLI/SDK, Python 3.11, ZIP/JSON/SHA-256, existing archived exact-expert replay downloader.

## Global Constraints

- This task downloads data only; it does not train, evaluate, submit, upload, commit or push.
- Never print or persist Kaggle credentials or signed URLs.
- Official archives are immutable and remain compressed; do not fully extract the roughly 350 GB logical dataset.
- Preserve unrelated dirty-worktree changes and do not modify `engine/source/`.
- Use all qualified natural data; deduplicate by Episode ID and content hash.
- Treat Dragapult+Blaziken as the T1 pure-Dragapult-side auxiliary subgroup while preserving its own build label and exact deck hash.

---

### Task 1: Verify and download the official 0726 dataset

**Files:**
- Create: `data/raw/episodes/archives/pokemon-tcg-ai-battle-episodes-2026-07-26.zip`
- Create: `data/raw/episodes/download_logs/pokemon-tcg-ai-battle-episodes-2026-07-26.log`
- Modify: `data/raw/episodes/index/manifest.csv`
- Modify: `data/raw/episodes/index/SHA256SUMS`
- Modify: `data/raw/episodes/source_manifest.json`

**Interfaces:**
- Consumes: Kaggle dataset `kaggle/pokemon-tcg-ai-battle-episodes-2026-07-26` and the local OAuth access token.
- Produces: one CRC-checked immutable ZIP plus episode/member counts, bytes, SHA-256 and provenance.

- [ ] **Step 1: Query dataset metadata**

Authenticate without echoing the token and confirm the dataset exists, its file/member metadata and download size. Save only non-secret metadata in the log.

- [ ] **Step 2: Download to an isolated staging path**

Use the existing bounded-range downloader or Kaggle CLI, retain resumable partials, and never overwrite an already validated archive.

- [ ] **Step 3: Validate and publish the archive**

Run ZIP CRC validation, index every JSON Episode ID/member, compute SHA-256, move the completed archive into the canonical path and update the three source indexes atomically.

### Task 2: Freeze the named source identities and builds

**Files:**
- Create: `experiments/0015_dragapult_conditioned_bc/data_audit/0726_named_sources.json`
- Create: `rl_runs/0015_dragapult_conditioned_bc/dataset/sources/0726_named_sources/<source>/deck.csv`
- Create: `rl_runs/0015_dragapult_conditioned_bc/dataset/sources/0726_named_sources/<source>/source_manifest.json`

**Interfaces:**
- Consumes: the frozen 0727 leaderboard snapshot and exact replay deck evidence.
- Produces: team ID, submission ID, exact 60-card multiset/hash and build assignment for each named source.

- [ ] **Step 1: Record the four named teams**

Freeze `LumenLiquidity`, `Oshbocker`, `THIRD PTCG Club` and `flg` with current submission IDs and exact probe decks.

- [ ] **Step 2: Classify without guessing**

Assign THIRD PTCG Club to target Dragapult+Dusknoir; LumenLiquidity and Oshbocker to the T1 Dragapult+Blaziken subgroup; `flg` to Marnie's Grimmsnarl/Munkidori T4, not to a Dragapult overlay.

### Task 3: Download only additional named Dragapult replays

**Files:**
- Create: `rl_runs/0015_dragapult_conditioned_bc/dataset/sources/targeted_overlay/<source>/episode-<id>-replay.json`
- Create: `rl_runs/0015_dragapult_conditioned_bc/dataset/sources/targeted_overlay/<source>/manifest.json`
- Create: `experiments/0015_dragapult_conditioned_bc/data_audit/0726_targeted_overlay_audit.json`

**Interfaces:**
- Consumes: submission-scoped latest completed/public Episode lists, exact source manifests and official 0726 Episode-ID index.
- Produces: only replay JSON absent from the official archive, with exact deck/player audit and base-versus-overlay counts.

- [ ] **Step 1: List completed public episodes per Dragapult submission**

Fetch at most the API-visible latest 1,000 episodes, resolve the player by exact submission ID and freeze the list before replay download.

- [ ] **Step 2: Reuse official 0726 coverage and fetch gaps**

Mark Episode IDs already present in the official ZIP as `official_base`; download only missing IDs for the three Dragapult sources, with bounded workers/retries and resumable files.

- [ ] **Step 3: Audit every replay**

Require matching Episode ID, team/submission/player index, complete public state, exact 60-card deck hash and usable visualization/action trace. Quarantine mismatches instead of training on them.

- [ ] **Step 4: Publish the overlay audit**

Report eligible, official-base, newly downloaded, duplicate, mismatched/quarantined and usable counts by source/build. Do not build model-ready features until the user supplies the final model contract.

### Task 4: Verify the ingestion handoff

**Files:**
- Read: all files created by Tasks 1–3

**Interfaces:**
- Consumes: archive/source/overlay manifests.
- Produces: a clickable, reproducible data-preparation handoff for the next model step.

- [ ] **Step 1: Cross-check hashes and counts**

Verify archive SHA-256/index membership, 60-card deck hashes, no duplicate overlay Episode IDs and no overlap counted twice between base and overlay.

- [ ] **Step 2: Validate repository hygiene**

Run JSON parsing, ZIP CRC, `git diff --check` and confirm large raw/overlay data remain Git-ignored. Record exact storage paths and remaining model-dependent work.
