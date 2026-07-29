# Limitless TEF-POR Environment Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable TEF-POR Limitless tournament and matchup analysis, compare it with the 2026-07-30 Kaggle Top 100 snapshot, and publish `docs/environment/limitless.html`.

**Architecture:** A standard-library Python package fetches and caches read-only Limitless pages, parses filter/deck/tournament/decklist/pairing facts into typed records, validates invariants, aggregates statistics, and renders one self-contained HTML report from a compact frozen JSON snapshot. Collection, analysis, and rendering stay separate so tests can recompute every displayed claim without parsing prose.

**Tech Stack:** Python 3.11+ standard library, `unittest`, HTML/CSS/vanilla JavaScript, Playwright or an installed Chromium-compatible screenshot tool for final visual verification.

## Global Constraints

- Never modify `engine/source/`.
- Limitless filter is exactly `time=all&type=all&format=TEF-POR&region=all&division=all`.
- Kaggle comparison source is `docs/environment-daily_kaggle_top100/daily/2026-07-30.html`.
- Network responses are cached only under ignored `.tmp/environment_limitless/`.
- Missing pairings are missing evidence, never zero matches or 0% win rate.
- Do not infer a counter from a cell with fewer than 15 matches; require `n >= 30` and a 95% interval excluding 50% for a strong signal.
- Do not commit or push unless the user explicitly requests it.

---

### Task 1: Filter and Source Audit

**Files:**
- Create: `data/processed/environment_limitless/__init__.py`
- Create: `data/processed/environment_limitless/models.py`
- Create: `data/processed/environment_limitless/fetch.py`
- Create: `data/processed/environment_limitless/parsers.py`
- Test: `tests/test_environment_limitless.py`

**Interfaces:**
- Produces: `fetch_text(url: str, cache_dir: Path, refresh: bool = False) -> FetchResult`
- Produces: `parse_filter_audit(html: str) -> FilterAudit`
- Produces: `parse_tournament_index(html: str) -> list[Tournament]`
- Produces: `parse_deck_distribution(html: str) -> list[DeckShare]`

- [ ] **Step 1: Write failing parser tests**

Use compact HTML fixtures inside `tests/test_environment_limitless.py` to assert exact active filter values, tournament IDs and player totals, deck points and shares, HTML entity decoding, and rejection of a missing `tef-por` label.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest -v tests.test_environment_limitless`
Expected: import failure because the package does not exist.

- [ ] **Step 3: Implement typed source records and parsers**

Use frozen dataclasses for `FetchResult`, `FilterAudit`, `Tournament`, and `DeckShare`; use `urllib.request` with a bounded retry count and `html.parser.HTMLParser` subclasses. Store response SHA-256 and final URL with every fetch.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m unittest -v tests.test_environment_limitless`
Expected: all Task 1 tests pass.

- [ ] **Step 5: Run the live filter audit**

Fetch both target URLs, assert exactly eight tournaments and 10,923 players unless the live page has changed, and print an explicit diff if it has. Confirm the eight IDs are the same IDs rendered in the report.

### Task 2: Decklists, Variants, and Player Evidence

**Files:**
- Modify: `data/processed/environment_limitless/models.py`
- Modify: `data/processed/environment_limitless/parsers.py`
- Create: `data/processed/environment_limitless/classification.py`
- Modify: `tests/test_environment_limitless.py`

**Interfaces:**
- Produces: `parse_tournament_standings(html: str, tournament: Tournament) -> list[Standing]`
- Produces: `parse_decklist(html: str, decklist_id: int) -> Decklist`
- Produces: `classify_deck(standing: Standing, decklist: Decklist | None) -> Classification`
- Produces: `validate_decklist(decklist: Decklist) -> None`

- [ ] **Step 1: Add failing classification and 60-card tests**

Cover Dragapult base, Dusknoir, Dudunsparce, Blaziken, conflicting labels, missing lists, duplicate shared decklists, and rejection of non-60-card lists.

- [ ] **Step 2: Verify RED**

Run the focused unittest methods and confirm missing interfaces fail.

- [ ] **Step 3: Implement standings/decklist parsing and two-level classification**

Preserve Limitless primary deck ID and variant ID as authoritative labels; use actual card names only as an audit cross-check. Record conflicts instead of silently relabeling.

- [ ] **Step 4: Verify GREEN and inspect live samples**

Parse at least one list for every nonzero Dragapult variant and print its exact card count, label, player, tournament, and key variant cards.

### Task 3: Pairings and Matchup Statistics

**Files:**
- Create: `data/processed/environment_limitless/pairings.py`
- Create: `data/processed/environment_limitless/stats.py`
- Modify: `data/processed/environment_limitless/models.py`
- Modify: `tests/test_environment_limitless.py`

**Interfaces:**
- Produces: `discover_pairing_source(tournament_html: str) -> PairingSource`
- Produces: `parse_pairings(source: PairingSource, payload: str) -> list[Match]`
- Produces: `aggregate_matchups(matches: Iterable[Match], classifications: Mapping[str, Classification]) -> MatchupMatrix`
- Produces: `wilson_interval(effective_wins: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]`

- [ ] **Step 1: Add failing matchup tests**

Test win/loss mirror symmetry, draw symmetry, duplicate-match rejection, byes exclusion, unresolved players exclusion with audit counts, effective win rate, Wilson intervals, evidence grades, and self-matchup exclusion from counter ranking.

- [ ] **Step 2: Verify RED**

Run focused tests and confirm undefined pairing/statistics interfaces fail.

- [ ] **Step 3: Implement source adapters and aggregation**

Support only observed public source structures from Limitless Labs/RK9. Do not invent a generic adapter for unavailable pages. Preserve tournament, round, table, player IDs, result, and source URL for deduplication.

- [ ] **Step 4: Audit coverage per tournament**

For each of the eight events report whether standings, deck identity, pairings, rounds, and results are available. Reconcile parsed match counts against published round/player counts where possible.

- [ ] **Step 5: Verify GREEN**

Run all environment-limitless tests and inspect matrix invariants on live aggregates.

### Task 4: Kaggle Snapshot and Comparative Metrics

**Files:**
- Create: `data/processed/environment_limitless/kaggle.py`
- Modify: `data/processed/environment_limitless/stats.py`
- Modify: `tests/test_environment_limitless.py`

**Interfaces:**
- Produces: `parse_kaggle_snapshot(html: str) -> KaggleSnapshot`
- Produces: `compare_environments(limitless: EnvironmentSnapshot, kaggle: KaggleSnapshot) -> EnvironmentComparison`

- [ ] **Step 1: Add failing Kaggle parser and metric tests**

Assert 100 leaderboard rows, key archetype counts, exact source date, HHI, Top-2/Top-5 concentration, normalized distributions, and Jensen-Shannon divergence behavior on identical and disjoint fixtures.

- [ ] **Step 2: Verify RED**

Run focused tests and confirm missing comparison interfaces fail.

- [ ] **Step 3: Implement local HTML parsing and metrics**

Read the existing 0730 report without changing it. Keep explicit mapping records for comparable archetypes and list unmapped categories in both directions.

- [ ] **Step 4: Verify GREEN and cross-check headline figures**

Confirm Kaggle has 100 rows, Grimmsnarl/Froslass has 56 seats, and Dragapult-related categories total two seats before analysis prose is generated.

### Task 5: Frozen Snapshot and Reproducible Generator

**Files:**
- Create: `data/processed/environment_limitless/generate.py`
- Create: `data/processed/environment_limitless/snapshot.json`
- Modify: `data/processed/environment_limitless/README.md`
- Modify: `tests/test_environment_limitless.py`

**Interfaces:**
- CLI: `python3 -m data.processed.environment_limitless.generate --refresh --output docs/environment/limitless.html`
- Produces: schema-versioned JSON with `sources`, `filter_audit`, `events`, `shares`, `variants`, `players`, `decklists`, `matchup_coverage`, `matchups`, `kaggle`, `comparison`, and `review_checks`.

- [ ] **Step 1: Add failing schema and invariant tests**

Require source hashes/timestamps, eight event audit entries, points conservation, all displayed decklists totaling 60, matchup symmetry, no missing-as-zero cells, and claim IDs that link prose conclusions to metrics.

- [ ] **Step 2: Verify RED**

Run the snapshot tests before generating the file.

- [ ] **Step 3: Implement CLI orchestration and atomic writes**

Collect into memory, validate all invariants, write temporary files beside targets, and replace targets only after successful validation.

- [ ] **Step 4: Generate the full frozen snapshot**

Run live collection with refresh, record all source coverage, and inspect unresolved rows and classification conflicts before accepting the snapshot.

- [ ] **Step 5: Verify GREEN**

Run all focused tests against the generated snapshot.

### Task 6: Rich Self-Contained HTML Report

**Files:**
- Create: `data/processed/environment_limitless/render.py`
- Create: `docs/environment/limitless.html`
- Modify: `tests/test_environment_limitless.py`

**Interfaces:**
- Produces: `render_report(snapshot: EnvironmentSnapshot) -> str`
- Required section IDs: `summary`, `filter-audit`, `metagame`, `kaggle-gap`, `dragapult-core`, `variants`, `players`, `matchup-matrix`, `dragapult-matchups`, `counter-evidence`, `causes`, `training`, `methodology`.

- [ ] **Step 1: Add failing report contract tests**

Assert required IDs, source links, eight event rows, headline distributions, heatmap dimensions, `W-L-D/n` cell details, confidence/evidence legends, representative deck sections, card images with text fallback, filter controls, and no `n=0` rendered as 0%.

- [ ] **Step 2: Verify RED**

Run the report contract tests before implementing the renderer.

- [ ] **Step 3: Implement semantic HTML, restrained CSS, and vanilla JS**

Follow the 0726 daily report information density and navigation while creating a distinct historical-comparison page. Use sticky table headers, responsive overflow, sortable tables, accessible heatmap labels, tooltips, card preview dialog, and print styles.

- [ ] **Step 4: Write evidence-linked analysis prose**

Explain environment formation, matchup mechanisms, Dragapult variants, player/build evidence, and training implications. Every quantitative claim includes a sample size or references a visible metric; causal wording is limited to supported card/rule mechanisms.

- [ ] **Step 5: Verify report contract GREEN**

Run focused tests and `git diff --check`.

### Task 7: Independent Data, Conclusion, and Visual Review

**Files:**
- Modify if needed: files from Tasks 1-6
- Create only under ignored paths: `.tmp/environment_limitless/review/`

**Interfaces:**
- Produces: machine-readable review checks in `snapshot.json` and screenshots under ignored `.tmp`.

- [ ] **Step 1: Re-fetch filter pages independently**

Compare final URL, active filter, event IDs, players, deck points/share, and split variants with the frozen snapshot. Fail on unexplained differences.

- [ ] **Step 2: Recompute statistics independently**

Use a separate verification path to recompute distributions, matchup totals, mirrored cells, Wilson intervals, HHI, concentration, and Jensen-Shannon divergence.

- [ ] **Step 3: Review every conclusion against evidence**

Create a claim checklist covering each headline, counter claim, representative-player statement, and training recommendation. Downgrade or remove claims whose evidence is weak, confounded, or missing.

- [ ] **Step 4: Render desktop and mobile screenshots**

Open the final HTML at desktop and mobile viewports, inspect screenshots, browser console errors, overflow, sticky navigation, heatmap readability, tooltip behavior, and card-preview behavior. Fix visible defects and repeat.

- [ ] **Step 5: Run final verification**

Run:

```bash
python3 -m unittest -v tests.test_environment_limitless
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q data/processed/environment_limitless
git diff --check
```

Expected: all commands exit 0. If unrelated pre-existing tests fail, isolate and report them with evidence instead of claiming a clean full suite.

- [ ] **Step 6: Audit final requirements and hand off**

Verify the objective, design spec, and this plan requirement by requirement; report the exact source coverage, generated file path, validation results, and any residual limitations.
