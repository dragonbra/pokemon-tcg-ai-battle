# Frozen League Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the immutable 48-deck Frozen Arena as a complete directed 48×48×10 official-engine league and publish a durable, auditable dashboard centered on `dragapult_ex_001` while retaining generic metrics for every deck.

**Architecture:** A resumable orchestrator runs one 480-game row per candidate deck against the fixed Frozen catalog. Both seats route their exact 60-card identity through one shared 0019 Foundation GPU service when policy hashes match; isolated CPU workers continue to own all official-engine state transitions. A dedicated Frozen League aggregator validates all 48 source reports, builds `matrix.json`, and renders a self-contained dashboard at `evaluation/arena/frozen_league/index.html`.

**Tech Stack:** Python 3.11, existing `evaluation.runner.batch`, PyTorch CUDA inference server, official engine runtime, stdlib JSON/HTML/JavaScript, `unittest`.

## Global Constraints

- The workload is exactly 48×48×10 = 23,040 official-engine games, including 10 self-play games per deck and five games in each candidate seat per directed cell.
- Foundation identity is 0019 Epoch 13, deployment `source_id=0`, SHA-256 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`.
- `engine/source/` is read-only; only the official engine runtime may produce strength evidence.
- Every inference request carries the exact 60-card deck. Frozen-vs-Frozen may share one GPU model service, but never one causal session between games.
- A completed row is immutable and resumable only when pool ID, catalog hash, policy hash, engine hash, game count, seat schedule, and report identity all match.
- Long-lived assets go under `evaluation/arena/frozen_league/`; temporary smoke artifacts go under `.tmp/evaluation/frozen_league/`.
- The primary ranking is uniform-schedule aggregate win rate with Wilson interval. No order-dependent Elo score may replace raw outcomes.
- The focal deck is `dragapult_ex_001`; the dashboard must show its display name, representative cards, and exact grouped 60-card construction before aggregate tables.
- Every row uses `league_deck_quality` revision 1. All 48 deck IDs must resolve to exactly one of the 21 audited strategy-category profiles in `evaluation/metrics/league_profiles.py`.

---

### Task 1: Shared-policy Frozen candidate adapter

**Files:**
- Modify: `evaluation/frozen.py`
- Modify: `evaluation/runner/batch.py`
- Test: `tests/test_evaluation_frozen_catalog.py`
- Test: `tests/test_evaluation_batch.py`

**Interfaces:**
- Consumes: `FrozenCatalog.policy`, `FrozenCatalog.opponents`, `BatchConfig`.
- Produces: `FrozenCatalog.candidate(deck_id: str) -> SubmissionPackage` and `BatchConfig.share_policy_inference_server: bool`.

- [ ] **Step 1: Write failing candidate-identity tests**

Assert that `catalog.candidate("dragapult_ex_001")` keeps the exact deck identity hash/name/display metadata, uses `_policy` as runtime root, uses the Foundation entrypoint and `cg_manifest`, and rejects unknown deck IDs.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python3 -m unittest -v tests.test_evaluation_frozen_catalog`

Expected: failure because `FrozenCatalog.candidate` does not exist.

- [ ] **Step 3: Implement the immutable adapter**

Add a method that locates exactly one Frozen identity and returns a `dataclasses.replace` copy whose `root` is `policy.root` while retaining the deck identity's name, exact deck, hashes, display name, representative cards, and deck manifest.

- [ ] **Step 4: Write failing shared-socket lifecycle tests**

Configure a Frozen `BatchConfig` with identical candidate/opponent policy hashes and `share_policy_inference_server=True`; patch `_policy_inference_server` and assert it is entered once and the same socket reaches both worker roles. Assert mismatched roots or policy hashes fail closed.

- [ ] **Step 5: Implement one-service Frozen-vs-Frozen inference**

Add `share_policy_inference_server: bool = False`. When enabled, start one candidate policy server and pass its socket as both candidate and opponent sockets. Validate both inference roots resolve to the same policy root and both devices are identical. Preserve the existing two-service path for trained candidate versus Frozen Foundation.

- [ ] **Step 6: Run focused tests**

Run: `python3 -m unittest -v tests.test_evaluation_frozen_catalog tests.test_evaluation_batch tests.test_evaluation_inference_server`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add evaluation/frozen.py evaluation/runner/batch.py tests/test_evaluation_frozen_catalog.py tests/test_evaluation_batch.py
git commit -m "feat(evaluation): share frozen league GPU policy"
```

### Task 2: Resumable 48-row league orchestrator

**Files:**
- Create: `evaluation/frozen_league.py`
- Create: `tests/test_evaluation_frozen_league.py`
- Create: `evaluation/arena/frozen_league/README.md`

**Interfaces:**
- Consumes: `load_frozen_catalog`, `FrozenCatalog.candidate`, `run_batch`, `BatchConfig`.
- Produces: `FrozenLeagueConfig`, `run_frozen_league(config: FrozenLeagueConfig) -> Path`, CLI `python3 -m evaluation.frozen_league run`.

- [ ] **Step 1: Write failing schedule and state tests**

Use a two-deck fixture and assert deterministic row order, 10 games per directed cell, alternating five/five seats, and state fields `pool_id`, `catalog_sha256`, `policy_hash`, `engine_hash`, `games_per_cell`, `completed_rows`, `started_at`, and `updated_at`.

- [ ] **Step 2: Run the test and verify failure**

Run: `python3 -m unittest -v tests.test_evaluation_frozen_league`

Expected: import failure because `evaluation.frozen_league` does not exist.

- [ ] **Step 3: Implement validated configuration and immutable run state**

Define a frozen dataclass with catalog path, output root, games per cell fixed to 10, workers, worker CPU threads, CUDA device, batch size, batch wait, timeout, and focal deck. Refuse any value other than 10 games per cell or a focal deck outside the catalog. Write state atomically through a sibling temporary file and `os.replace`.

- [ ] **Step 4: Write failing resume tests**

Create one valid completed row report and assert it is skipped. Mutate catalog hash, policy hash, engine hash, opponent order, total games, or completion count and assert the orchestrator refuses to reuse it. Assert interrupted/incomplete rows are rerun into a new `run-<id>` without deleting evidence.

- [ ] **Step 5: Implement one full row per candidate**

For each catalog deck, construct the candidate adapter, call `run_batch` with all 48 Frozen opponents, `games_per_opponent=10`, `workers=128` by default, `worker_cpu_threads=1`, `candidate_inference_device=opponent_inference_device="cuda:0"`, and `share_policy_inference_server=True`. Store reports under `evaluation/arena/frozen_league/reports/<deck_id>/run-<id>/report.html`; append a row to state only after 480/480 complete with zero error and zero unfinished.

- [ ] **Step 6: Add resource and interruption handling**

Before each row require at least 80 GiB free on the output filesystem. SIGINT/SIGTERM sets a stop request, allows the active row to finish and publish, writes state, then exits before the next row. Never retain traces unless `--keep-temp` is explicitly supplied for a smoke.

- [ ] **Step 7: Add CLI**

Support `run`, `status`, and `render` subcommands. `run` accepts `--workers`, `--worker-cpu-threads`, `--device`, `--batch-size`, `--batch-wait-ms`, `--focal-deck`, and `--row-limit` for smoke only. `status` reports completed rows, games, errors, elapsed wall time, and remaining rows.

- [ ] **Step 8: Run tests**

Run: `python3 -m unittest -v tests.test_evaluation_frozen_league`

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add evaluation/frozen_league.py tests/test_evaluation_frozen_league.py evaluation/arena/frozen_league/README.md
git commit -m "feat(evaluation): add resumable frozen league runner"
```

### Task 3: Generic Frozen League data model

**Files:**
- Create: `evaluation/frozen_league_report.py`
- Create: `tests/test_evaluation_frozen_league_report.py`

**Interfaces:**
- Consumes: 48 complete row reports, Frozen catalog, official card catalog, focal deck ID.
- Produces: `build_frozen_league_data(...) -> dict[str, object]`, `wilson_interval(wins: int, games: int) -> tuple[float, float]`.

- [ ] **Step 1: Write failing aggregation tests**

Build a three-deck synthetic directed matrix and assert total games, W-L-D, completion, first/second splits, average complete rounds, per-cell results, row aggregate, Wilson bounds, hardest/easiest non-self matchups, self-play audit, reciprocal-role delta, and archetype aggregation.

- [ ] **Step 2: Run the test and verify failure**

Run: `python3 -m unittest -v tests.test_evaluation_frozen_league_report`

Expected: import failure because the report module does not exist.

- [ ] **Step 3: Implement strict report ingestion**

Reuse the embedded `report-data` contract. Require catalog-order opponents, exactly 480 games per row for the real catalog, 10 games per cell, five candidate-first and five candidate-second, matching Frozen pool/catalog/policy hashes, one official engine hash, zero missing rows, and unique run IDs.

- [ ] **Step 4: Implement generic metrics**

Compute uniform aggregate win rate and 95% Wilson interval; W-L-D; first/second win rates; average complete rounds; hardest and easiest five non-self matchups; per-deck matchup spread; diagonal self-play seat balance; reciprocal role delta between `A→B` and the inverted `B→A` evidence; archetype weighted outcomes; global errors, unfinished, throughput, and wall time. Keep Elo out of the primary contract because the schedule is already uniform and Elo introduces order/model assumptions.

- [ ] **Step 5: Implement exact deck presentation data**

For every deck, count duplicate card IDs and enrich them with official name, expansion, collection number, image URL, and one of `Pokémon`, `Trainer`, or `Energy`. For the focal deck, include grouped counts and verify the group totals sum to exactly 60.

- [ ] **Step 6: Run tests**

Run: `python3 -m unittest -v tests.test_evaluation_frozen_league_report`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add evaluation/frozen_league_report.py tests/test_evaluation_frozen_league_report.py
git commit -m "feat(evaluation): aggregate frozen league evidence"
```

### Task 4: League dashboard UI

**Files:**
- Modify: `evaluation/frozen_league_report.py`
- Test: `tests/test_evaluation_frozen_league_report.py`

**Interfaces:**
- Consumes: `build_frozen_league_data` payload.
- Produces: `render_frozen_league(data: Mapping[str, object]) -> str`, `write_frozen_league(...)`.

- [ ] **Step 1: Write failing HTML contract tests**

Assert presence of focal-deck hero, exact grouped 60-card construction with card images, summary cards, sortable strength table, W-L-D and Wilson interval, first/second split, focal matchup panel, package win-rate heatmap, first/second heatmaps, archetype heatmap, average-round heatmap, self-play/reciprocal audit, methodology warning, embedded JSON, sticky matrix labels, and responsive CSS.

- [ ] **Step 2: Run the test and verify failure**

Run: `python3 -m unittest -v tests.test_evaluation_frozen_league_report`

Expected: failure because the renderer is absent.

- [ ] **Step 3: Implement focal deck hero**

Render `Dragapult ex 001` display name, internal deck ID, representative Pokémon images, Foundation/pool identity, and three grouped construction columns. Each card row shows quantity, image, official name, set, and collection number. This section must be the first substantive content after the title.

- [ ] **Step 4: Implement generic overview and ranking**

Render 48 decks, 23,040 games, completion/errors, total wall time, games/s, average complete rounds, and Foundation hash. The sortable ranking shows rank, deck identity, W-L-D, win rate, 95% Wilson interval, first/second rates, matchup spread, average rounds, and source report link.

- [ ] **Step 5: Implement focal matchup explorer**

Show the focal deck's 48 matchups sorted hardest-to-easiest with opponent cards, W-L-D, total/first/second win rates, average rounds, and a 10-game sample warning. Add a client-side opponent-name filter.

- [ ] **Step 6: Implement matrices and audits**

Render sticky 48×48 package total/first/second win-rate heatmaps, archetype weighted heatmap, and package average-round heatmap. Cell tooltips contain W-L-D, seat counts, rounds, and both deck names. Add diagonal self-play and reciprocal-role audit tables so infrastructure asymmetry is visible instead of silently averaged away.

- [ ] **Step 7: Implement evidence boundary**

State that 10 games per directed cell is noisy, Wilson intervals describe row aggregates rather than individual cells, raw win rate measures adaptation to this fixed Foundation/deck environment, and results do not prove global optimality or Kaggle performance.

- [ ] **Step 8: Run tests**

Run: `python3 -m unittest -v tests.test_evaluation_frozen_league_report tests.test_evaluation_combat_matrix`

Expected: all pass and the legacy Combat Matrix UI remains unchanged.

- [ ] **Step 9: Commit**

```bash
git add evaluation/frozen_league_report.py tests/test_evaluation_frozen_league_report.py
git commit -m "feat(evaluation): render frozen league dashboard"
```

### Task 5: Official-engine smoke and full publication

**Files:**
- Create: `evaluation/arena/frozen_league/state.json` through the runner
- Create: `evaluation/arena/frozen_league/reports/**/report.html` through the runner
- Create: `evaluation/arena/frozen_league/matrix.json` through the renderer
- Create: `evaluation/arena/frozen_league/index.html` through the renderer
- Modify: `evaluation/README.md`
- Modify: `experiments/0022_league_training/DESIGN.md`
- Modify: `experiments/0022_league_training/DESIGN.html`

**Interfaces:**
- Consumes: completed Tasks 1–4 and RTX 5080 CUDA device.
- Produces: durable full league evidence and documentation links.

- [ ] **Step 1: Run a two-row official-engine GPU smoke**

Run:

```bash
python3 -m evaluation.frozen_league run \
  --output .tmp/evaluation/frozen_league/smoke \
  --row-limit 2 \
  --workers 32 \
  --worker-cpu-threads 1 \
  --device cuda:0
```

Expected: 960/960 finished, zero error, one shared GPU policy server per row, balanced seats, and two complete row reports.

- [ ] **Step 2: Render and inspect the smoke UI**

Run `python3 -m evaluation.frozen_league render --output .tmp/evaluation/frozen_league/smoke` and inspect the generated `index.html`; verify exact focal deck total 60 and all tested matrix cells contain 10 games.

- [ ] **Step 3: Run the complete resumable league**

Run:

```bash
python3 -m evaluation.frozen_league run \
  --output evaluation/arena/frozen_league \
  --workers 128 \
  --worker-cpu-threads 1 \
  --device cuda:0 \
  --batch-size 256 \
  --batch-wait-ms 5
```

Expected: 48 complete rows, 23,040/23,040 finished, zero error and zero unfinished. If interrupted, rerun the identical command and resume only verified complete rows.

- [ ] **Step 4: Publish the final report**

Run `python3 -m evaluation.frozen_league render --output evaluation/arena/frozen_league` and verify `matrix.json` and `index.html` report exactly 48×48×10 games and link every source row report.

- [ ] **Step 5: Update authority documentation**

Document the command, pool/policy/catalog hashes, total games, completion, wall time, throughput, focal deck, and report link in both 0022 DESIGN formats and `evaluation/README.md`. Do not describe ranking as global strength.

- [ ] **Step 6: Run full verification**

Run:

```bash
python3 -m unittest discover -v -s tests
python3 -m unittest discover -v -s train/0022_league_training/tests -t .
git diff --check
```

Expected: evaluation and 0022 tests pass; any unrelated environment snapshot failure is reported separately with its exact missing asset.

- [ ] **Step 7: Commit**

```bash
git add evaluation/arena/frozen_league evaluation/README.md experiments/0022_league_training/DESIGN.md experiments/0022_league_training/DESIGN.html
git commit -m "docs(evaluation): publish frozen league report"
```
