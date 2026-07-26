# 2026-07-26 Environment Daily 0726 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a reproducible `0726` Top 100 environment snapshot that binds every current leaderboard team to its final listed submission, then analyzes that submission's official completed Episode Meta in the same contract and presentation as the 2026-07-25 report.

**Architecture:** Reuse the 2026-07-25 report's leaderboard → final submission → official completed Episode Meta → representative replay/deck contract. Fetch current leaderboard rows and submission-scoped metadata from official Kaggle endpoints, retain only completed records for the currently listed submission ID, deduplicate shared games, then render every table from that frozen campaign without overwriting the 2026-07-25 snapshot.

**Tech Stack:** Kaggle CLI/API, Python 3.11 standard library, official Episode JSON/ZIP archives, static HTML/CSS/JavaScript.

## Global Constraints

- Official engine/source is read-only; this task does not modify it.
- Raw official Meta/replay data remains immutable; any local campaign cache records its endpoint, capture time, submission ID, and pagination boundary.
- The report must state its capture time, submission-scoped Meta input, source URLs, endpoint limits, and evidence limits.
- The final report is `docs/environment-daily_kaggle_top100/daily/2026-07-26.html` and the index is date-descending.
- Do not submit to Kaggle or publish/upload any dataset.
- Preserve unrelated dirty-worktree changes and do not commit unless explicitly requested.

### Task 1: Inspect the prior report and source contract

**Files:**
- Read: `docs/environment-daily_kaggle_top100/daily/2026-07-25.html`
- Read: `data/raw/episodes/source_manifest.json`
- Read: `data/raw/episodes/index/manifest.csv`

- [x] Extract the report sections, embedded data fields, and previous-day comparison inputs needed for the 0726 report.
- [x] Confirm the new source boundary is the complete official 2026-07-25 daily dataset and record the prior report's format without altering it.

### Task 2: Verify and download the official 2026-07-25 dataset

**Files:**
- Create: `data/raw/episodes/archives/pokemon-tcg-ai-battle-episodes-2026-07-25.zip`
- Create: `data/raw/episodes/download_logs/pokemon-tcg-ai-battle-episodes-2026-07-25.log`
- Modify: `data/raw/episodes/index/manifest.csv`
- Modify: `data/raw/episodes/index/SHA256SUMS`
- Modify: `data/raw/episodes/source_manifest.json`

- [x] Query Kaggle metadata for `kaggle/pokemon-tcg-ai-battle-episodes-2026-07-25` and confirm the official dataset exists, its file list, size, and version metadata.
- [x] Download the exact official dataset into a temporary isolated directory, validate archive CRC and expected manifest, then move the validated ZIP into the canonical archive path.
- [x] Record the download command's non-secret log, SHA256, byte count, episode count, and coverage in the source indexes.

### Task 3: Rebuild the 0726 campaign from current submissions and official Meta

**Files:**
- Create: `docs/environment-daily_kaggle_top100/daily/2026-07-26.html`
- Create or update: repository-local derived analysis artifacts only if the existing workflow requires them.

- [ ] Capture the current official Top100 leaderboard and resolve every row's final submission ID from the current leaderboard record.
- [ ] Fetch paginated completed Meta records scoped to each final submission, retain only records whose own submission IDs still match the frozen leaderboard, and record the API limit/truncation status per player.
- [ ] Fetch one or more representative replay JSON files per final submission for exact 60-card deck, actual firstPlayer and final engine turn; deduplicate shared replay files while retaining both policy views.
- [ ] Recompute W/L/D, Top100-vs-other splits, wall-clock, per-player and archetype matchups, card pool, turn-order description, and the 20-player schedule from this submission-bound campaign only.
- [ ] Preserve the previous report's full sections and visual language while updating all date, counts, rankings, deck hashes, matchup values, and comparison labels.

### Task 4: Publish and validate the archive entry

**Files:**
- Modify: `docs/environment-daily_kaggle_top100/index.html`
- Modify: `docs/environment-daily_kaggle_top100/README.md`

- [x] Add 2026-07-26 above older reports with the name `0726` in the visible report metadata/title where the existing format supports it.
- [ ] Verify HTML references, 100 frozen leaderboard rows, final-submission binding, Meta pagination/limit records, replay/deck evidence, and date ordering.
- [ ] Run HTMLParser, Top100, five-day manifest, and `git diff --check` checks and report the final clickable paths plus key findings.
