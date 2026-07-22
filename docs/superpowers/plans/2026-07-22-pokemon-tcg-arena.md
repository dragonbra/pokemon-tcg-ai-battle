# Kaggle 公共卡组 Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前仓库建立可恢复的 Kaggle 公共卡组 Arena，收集并兼容公开 Code，运行本地官方引擎循环对战，保存 Rating/胜率事实并生成 All/Eligible 双版本 HTML 报告。

**Architecture:** 新增顶层 `arena` Python package 作为调用层，使用 SQLite 保存索引、JSONL 保存可审计事实，使用独立 package 目录保存每个卡组。Arena 复用 `evaluation` 的 package validator、CG 兼容检查和隔离 worker；评价层只提供对局能力，Arena 负责来源、调度、Rating、低质量筛选和综合报告。

**Tech Stack:** Python 3.11+、标准库 `sqlite3`/`json`/`csv`/`tarfile`/`zipfile`/`ast`/`html`、现有 Kaggle CLI、现有官方 `cg` runtime、标准库 `unittest`。

## Global Constraints

- 所有新增 Arena 内容位于仓库顶层 `arena/`；设计和计划文档使用中文。
- 公开来源主 competition 固定为 `pokemon-tcg-ai-battle`，只补充 challenge-strategy 中实际包含 submission 压缩包的方案。
- 适配只修复路径、导入、入口函数和 package 外壳，不修改策略决策逻辑；原始来源永不覆盖。
- 标准 package 必须包含 `main.py`、恰好 60 行 `deck.csv` 和物理独立的完整 `cg/`。
- 默认 Rating 为 `official_gaussian_approx`，初始 `mu=600`、`sigma=200`；低质量阈值为 `mu<300`。
- 同时从相同逐局事实重算初始分 600、K=32、平局按 0.5 的 `elo_compat`。
- 至少 20 场有效局且连续两个 checkpoint 低于 300 才标记 `demoted`；All 和 Eligible 报告都必须保留其事实。
- `work/alakazam_v8_current` 是 `internal_reference`，参加对局但不计入公开卡组汇总，也不自动修改 evaluation catalog。
- 初筛每对 10 局、双方先后手各 5 局；正式默认每对 20 局，可配置到 50 局。
- 不修改 `engine/source/`，不执行 Kaggle submission，不写入凭据、`.env` 或官方数据副本。
- 除非用户明确要求，不执行 `git commit`；所有验证必须在报告完成前用新命令执行。

---

### Task 1: Arena 数据模型和 SQLite/JSONL 存储

**Files:**
- Create: `arena/__init__.py`
- Create: `arena/models.py`
- Create: `arena/storage.py`
- Create: `arena/paths.py`
- Create: `tests/test_arena_storage.py`

**Interfaces:**
- `SourceRecord`、`DeckRecord`、`RunRecord`、`GameRecord`、`RatingEvent` 使用 frozen dataclass，并提供 `to_json()` 返回 JSON-safe `dict[str, object]`。
- `ArenaStore(root: Path)` 在 `root/db/arena.sqlite3` 建表，并提供 `initialize() -> None`、`upsert_source(record: SourceRecord) -> None`、`upsert_deck(record: DeckRecord) -> None`、`create_run(record: RunRecord) -> None`、`record_game(record: GameRecord) -> None`、`record_rating_event(record: RatingEvent) -> None`、`completed_game_ids(run_id: str) -> set[str]`、`load_games(run_id: str | None = None) -> tuple[dict[str, object], ...]`、`load_latest_ratings() -> tuple[dict[str, object], ...]`。
- `arena.paths.arena_root(repo_root: Path) -> Path`、`source_root(repo_root: Path) -> Path`、`package_root(repo_root: Path) -> Path`、`run_root(repo_root: Path) -> Path`、`report_root(repo_root: Path) -> Path` 只负责路径解析，不创建用户数据。
- SQLite 使用 `sources(source_id PRIMARY KEY, competition, ref, version, title, author, url, metadata_json, source_hash, status, error, collected_at)`、`decks(deck_id PRIMARY KEY, source_id, display_name, archetype, primary_pokemon_json, package_path, package_hash, role, status, metadata_json)`、`runs(run_id PRIMARY KEY, phase, config_json, started_at, finished_at, status)`、`games(game_id PRIMARY KEY, run_id, player_a, player_b, player_a_first, winner, result, status, error_kind, error, steps, payload_json)`、`rating_events(event_id PRIMARY KEY, run_id, game_id, engine, revision, player_a, player_b, before_json, outcome, expected_a, delta_a, delta_b, after_json, created_at)`、`checkpoints(checkpoint_id PRIMARY KEY, run_id, sequence, ratings_json, created_at)` 六张表。

- [ ] **Step 1: Write the failing storage tests**

```python
class ArenaStoreTests(unittest.TestCase):
    def test_round_trips_source_deck_game_and_rating_event(self) -> None:
        store = ArenaStore(self.root)
        store.initialize()
        store.upsert_source(SourceRecord("source-1", "pokemon-tcg-ai-battle", "u/k", 1, "Title", "u", "https://www.kaggle.com/code/u/k", {}, "hash", "exact_submission", None, "2026-07-22T00:00:00Z"))
        store.upsert_deck(DeckRecord("deck-1", "source-1", "Lucario - u", "Lucario", (1, 2), "arena/packages/deck-1", "pkg", "public", "exact_submission", {}))
        store.create_run(RunRecord("run-1", "smoke", {"seed": 7}, "2026-07-22T00:00:00Z", None, "running"))
        store.record_game(GameRecord("game-1", "run-1", "deck-1", "internal", True, "deck-1", "win", "finished", None, None, 4, {"candidate_first": True}))
        store.record_rating_event(RatingEvent("event-1", "run-1", "game-1", "elo_compat", 1, "deck-1", "internal", {"mu": 600}, "win", 0.5, 16, -16, {"mu": 616}, "2026-07-22T00:00:01Z"))
        self.assertEqual(store.completed_game_ids("run-1"), {"game-1"})
        self.assertEqual(store.load_games("run-1")[0]["result"], "win")
        self.assertEqual(store.load_latest_ratings()[0]["event_id"], "event-1")

    def test_completed_game_ids_are_unique_and_json_is_safe(self) -> None:
        store = ArenaStore(self.root)
        store.initialize()
        record = GameRecord("game-1", "run-1", "a", "b", True, None, "error", "error", "worker", "boom", 0, {})
        store.record_game(record)
        store.record_game(record)
        self.assertEqual(store.completed_game_ids("run-1"), {"game-1"})
```

- [ ] **Step 2: Run the focused tests and verify the expected missing-module failure**

Run: `python3 -m unittest -v tests.test_arena_storage`

Expected: FAIL because `arena.models` and `arena.storage` do not yet exist.

- [ ] **Step 3: Implement the models, schema and atomic JSONL helpers**

Implement `ArenaStore.initialize()` with `CREATE TABLE IF NOT EXISTS`, `INSERT ... ON CONFLICT DO UPDATE`, JSON serialization using `ensure_ascii=False`, and a unique `game_id` constraint. Every mutating operation must commit immediately so a process interruption cannot lose a completed game. Store JSONL facts at `arena/runs/<run_id>/games.jsonl` and `rating_events.jsonl` in addition to SQLite.

- [ ] **Step 4: Run the focused tests and inspect the schema**

Run: `python3 -m unittest -v tests.test_arena_storage`

Expected: PASS with 2 tests; `python3 -c 'from arena.storage import ArenaStore; from pathlib import Path; ArenaStore(Path("/tmp/arena-schema")).initialize()'` exits 0.

- [ ] **Step 5: Refactor only after green**

Run: `git diff --check` and `python3 -m compileall -q arena tests/test_arena_storage.py`.

Expected: exit 0 and no whitespace diagnostics.

### Task 2: Kaggle Code/Discussion 收集和 package 适配

**Files:**
- Create: `arena/collect.py`
- Create: `arena/catalog.py`
- Create: `arena/adapters.py`
- Create: `arena/discussions.py`
- Create: `arena/catalog/overrides.json`
- Create: `tests/test_arena_collect.py`
- Create: `tests/test_arena_adapters.py`

**Interfaces:**
- `KernelMetadata` has `from_mapping(value: Mapping[str, object]) -> KernelMetadata` and `source_id() -> str`; it preserves `ref`, `title`, `author`, `last_run_time`, `total_votes`, `competition`, `collected_at` and raw metadata.
- `collect_kernel_index(competition: str, destination: Path, *, page_size: int = 200, max_pages: int | None = None, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> tuple[KernelMetadata, ...]` invokes `kaggle kernels list --competition <competition> --page <n> --page-size <page_size> --format json` and writes one immutable metadata JSON per source.
- `collect_kernel_files(metadata: KernelMetadata, destination: Path, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> CollectionResult` calls `kaggle kernels pull <ref> --path <dir> --metadata` and `kaggle kernels output <ref> --path <output-dir> --quiet`, records command failures, and never raises for one unavailable source.
- `collect_discussion_index(competition: str, destination: Path, *, pages: int = 1) -> tuple[DiscussionRecord, ...]` uses `kaggle competitions topics list <competition> --format json` and `topics show` to save official Discussion title, author, URL, votes, created time and comment tree.
- `safe_extract_archive(archive_path: Path, destination: Path) -> tuple[Path, ...]` rejects absolute paths and `..` traversal before extracting `.tar`, `.tar.gz` or `.zip` files.
- `reconstruct_notebook_package(source_dir: Path, package_dir: Path, baseline_cg: Path) -> AdapterResult` concatenates code cells from `.ipynb`, removes notebook-only magic lines, finds a literal 60-card list from `DECK`, `my_deck`, `MY_DECK` or `deck`, copies `main.py`/`deck.csv` and physically copies baseline `cg`; it marks compile/deck/entrypoint failures as `invalid` with an exact reason.
- `classify_deck(deck: list[int], card_metadata: Mapping[int, Mapping[str, str]], overrides: Mapping[str, object]) -> tuple[str, tuple[int, ...], str]` returns `(archetype, primary_pokemon_ids, display_name)` and falls back to `Unknown Archetype` without guessing.

- [ ] **Step 1: Write failing tests for pagination, safe extraction and notebook deck extraction**

```python
class CollectionTests(unittest.TestCase):
    def test_kernel_metadata_keeps_votes_and_builds_stable_source_id(self) -> None:
        metadata = KernelMetadata.from_mapping({"ref": "u/k", "title": "Title", "author": "U", "totalVotes": 8})
        self.assertEqual(metadata.source_id(), "pokemon-tcg-ai-battle__u__k")
        self.assertEqual(metadata.total_votes, 8)

    def test_archive_extraction_rejects_parent_traversal(self) -> None:
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../../escape.txt", "bad")
        with self.assertRaises(ValueError):
            safe_extract_archive(archive, self.root / "out")

class AdapterTests(unittest.TestCase):
    def test_reconstructs_literal_deck_and_copies_cg(self) -> None:
        notebook = {"cells": [{"cell_type": "code", "source": ["my_deck = [7] * 60\n", "def agent(obs):\n", "    return my_deck if obs.get('select') is None else [0]\n"]}]}
        (self.source / "agent.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
        result = reconstruct_notebook_package(self.source, self.package, self.baseline_cg)
        self.assertEqual(result.status, "notebook_reconstructed")
        self.assertEqual(len((self.package / "deck.csv").read_text().splitlines()), 60)
        self.assertTrue((self.package / "cg" / "game.py").is_file())
```

- [ ] **Step 2: Run tests and verify they fail for missing Arena collectors/adapters**

Run: `python3 -m unittest -v tests.test_arena_collect tests.test_arena_adapters`

Expected: FAIL because the collector and adapter interfaces do not yet exist.

- [ ] **Step 3: Implement collection with immutable snapshots and failure records**

Parse only JSON returned by Kaggle CLI, preserve raw entries, paginate until an empty page or `max_pages`, and write source directories using a safe slug derived from owner/kernel/version. Run both Code competitions but only retain challenge-strategy artifacts when an archive is actually found. Store command stdout/stderr hashes and failures in the source record instead of aborting the complete collection.

- [ ] **Step 4: Implement secure archive and Notebook adaptation**

Use `tarfile.is_tarfile`/`zipfile.ZipFile`, reject traversal before extraction, search extracted trees for a complete top-level or nested package, and copy only the selected package into `arena/packages/<deck-id>`. For Notebook reconstruction, use `json.load`, `ast.parse`, and literal evaluation; do not execute untrusted Notebook code during collection. Copy the physical `cg` baseline only after the strategy source has been written. Run package validation later in a subprocess.

- [ ] **Step 5: Implement metadata/card classification and Discussion snapshots**

Read `data/official/EN_Card_Data.csv` with `csv.DictReader`; identify Pokémon rows from `Stage (Pokémon)/Type (Energy and Trainer)` and use card-name/count ordering for primary Pokémon. Preserve manual overrides. Save Discussion topic trees as JSON and an index Markdown with links, title, author, votes and collection time.

- [ ] **Step 6: Run focused tests and static checks**

Run: `python3 -m unittest -v tests.test_arena_collect tests.test_arena_adapters`

Expected: all focused tests pass; `python3 -m compileall -q arena` exits 0; no test executes downloaded Notebook code.

### Task 3: Gaussian/Elo Rating and match scheduler

**Files:**
- Create: `arena/rating.py`
- Create: `arena/scheduler.py`
- Create: `tests/test_arena_rating.py`
- Create: `tests/test_arena_scheduler.py`

**Interfaces:**
- `RatingState(name: str, mu: float = 600.0, sigma: float = 200.0, elo: float = 600.0, games: int = 0, status: str = "active", below_threshold_checkpoints: int = 0)` is immutable and serializable.
- `GaussianRatingEngine(initial_mu: float = 600.0, initial_sigma: float = 200.0, beta: float = 200.0, update_scale: float = 80.0, min_sigma: float = 40.0, revision: int = 1)` provides `expected(a: RatingState, b: RatingState) -> float` and `update(a: RatingState, b: RatingState, outcome: float) -> RatingUpdate`.
- `EloRatingEngine(initial: float = 600.0, k_factor: float = 32.0)` provides `expected(a: RatingState, b: RatingState) -> float` and `update(a: RatingState, b: RatingState, outcome: float) -> RatingUpdate`.
- `RatingUpdate` contains `before_a`, `before_b`, `after_a`, `after_b`, `expected_a`, `delta_a`, `delta_b`, `outcome` and `revision`.
- Gaussian expected result uses `0.5 * (1 + erf((mu_a - mu_b) / (sqrt(sigma_a**2 + sigma_b**2 + 2*beta**2) * sqrt(2))))`; the symmetric mu update is `update_scale * min(1, (sigma_a + sigma_b)/(2*initial_sigma)) * (outcome - expected_a)`; both sigma values shrink by `5% * abs(outcome - expected_a)` down to `min_sigma`.
- `eligible_for_demotion(state: RatingState, checkpoint_games: int) -> bool` requires `games >= 20` and two consecutive checkpoint observations below 300; `apply_checkpoint_status(states, threshold=300, minimum_games=20) -> tuple[RatingState, ...]` updates `status` without deleting state.
- `build_smoke_pairs(deck_ids: Sequence[str]) -> tuple[Pairing, ...]` returns every unordered pair; `build_rating_pairs(states, completed_counts, seed, limit) -> tuple[Pairing, ...]` prioritizes nearest mu, then underplayed pairs, then lower game counts, with deterministic seeded tie-breaking.

- [ ] **Step 1: Write failing Rating tests**

```python
class RatingTests(unittest.TestCase):
    def test_gaussian_starts_at_600_and_win_moves_winner_up(self) -> None:
        engine = GaussianRatingEngine()
        a, b = RatingState("a"), RatingState("b")
        update = engine.update(a, b, 1.0)
        self.assertEqual(update.before_a.mu, 600.0)
        self.assertGreater(update.after_a.mu, 600.0)
        self.assertLess(update.after_b.mu, 600.0)
        self.assertLess(update.after_a.sigma, 200.0)

    def test_draw_moves_equal_ratings_toward_each_other_without_prize_margin(self) -> None:
        update = GaussianRatingEngine().update(RatingState("a", 700), RatingState("b", 500), 0.5)
        self.assertLess(update.after_a.mu, 700)
        self.assertGreater(update.after_b.mu, 500)

    def test_elo_compat_is_replayable_from_same_result(self) -> None:
        update = EloRatingEngine().update(RatingState("a"), RatingState("b"), 1.0)
        self.assertEqual(update.after_a.elo, 616.0)
        self.assertEqual(update.after_b.elo, 584.0)

    def test_demotion_requires_twenty_games_and_two_checkpoints(self) -> None:
        state = RatingState("bad", 299, 120, games=19)
        self.assertFalse(eligible_for_demotion(state, 299))
        state = replace(state, games=20, below_threshold_checkpoints=2)
        self.assertTrue(eligible_for_demotion(state, 299))
```

- [ ] **Step 2: Run focused Rating tests and verify the expected missing-module failure**

Run: `python3 -m unittest -v tests.test_arena_rating`

Expected: FAIL because `arena.rating` does not yet exist.

- [ ] **Step 3: Implement both engines and status transitions**

Use only `math.erf`, `math.sqrt`, dataclasses and explicit constants. Do not depend on SciPy. Record algorithm revision and all before/after fields in `RatingUpdate`. Clamp sigma and never mutate caller-owned `RatingState` objects.

- [ ] **Step 4: Write and run scheduler tests**

```python
class SchedulerTests(unittest.TestCase):
    def test_smoke_pairs_cover_each_unordered_pair_once(self) -> None:
        pairs = build_smoke_pairs(("a", "b", "c"))
        self.assertEqual(pairs, (Pairing("a", "b", 0), Pairing("a", "c", 0), Pairing("b", "c", 0)))

    def test_rating_scheduler_prefers_closest_ratings_and_underplayed_pairs(self) -> None:
        states = (RatingState("a", 600), RatingState("b", 605), RatingState("c", 1000))
        pairs = build_rating_pairs(states, {("a", "b"): 0, ("a", "c"): 5, ("b", "c"): 5}, 7, 1)
        self.assertEqual(pairs[0].players, ("a", "b"))
```

Run: `python3 -m unittest -v tests.test_arena_scheduler`

Expected: FAIL before `arena.scheduler` exists, then PASS after the deterministic scheduler is implemented.

- [ ] **Step 5: Run combined Rating/scheduler tests**

Run: `python3 -m unittest -v tests.test_arena_rating tests.test_arena_scheduler`

Expected: all tests pass and no random seed changes the selected pair for the same input.

### Task 4: Single-game evaluation bridge and Arena runner

**Files:**
- Create: `evaluation/runner/single.py`
- Modify: `evaluation/runner/batch.py`
- Create: `arena/runner.py`
- Create: `tests/test_arena_runner.py`
- Modify: `evaluation/runner/__init__.py`

**Interfaces:**
- `evaluation.runner.single.run_game(request: GameRequest, *, temp_root: Path, timeout_seconds: float = 30.0) -> tuple[GameResult, dict[str, object]]` invokes the existing isolated worker once and returns the result plus trace payload without creating a batch report.
- `ArenaRunner(store: ArenaStore, repo_root: Path, config: ArenaRunConfig)` provides `run_validation(deck: DeckRecord) -> ValidationResult`, `run_initial_round(deck_ids: Sequence[str]) -> None`, `run_continuous_round(limit: int) -> None`, `run_one(pair: Pairing, game_number: int) -> GameRecord`, and `checkpoint() -> None`.
- `ArenaRunConfig` contains `run_id`, `phase`, `games_per_pair`, `max_steps`, `worker_timeout_seconds`, `seed`, `visualize`, `rating_method`, `include_demoted` and `checkpoint_every`.
- Each pair runs two physical seat orientations. The candidate/opponent names passed to `GameRequest` are remapped back to Arena `player_a`/`player_b`; winner is stored as `player_a`, `player_b`, `draw`, or `None` for error/unfinished.

- [ ] **Step 1: Write the single-game bridge regression test**

```python
class SingleGameBridgeTests(unittest.TestCase):
    def test_single_game_delegates_to_isolated_worker_and_keeps_trace(self) -> None:
        request = make_fixture_request()
        with patch("evaluation.runner.single._run_worker") as worker:
            worker.return_value = (fixture_result(request), {"trace": [], "result": {"winner": 0}})
            result, trace = run_game(request, temp_root=self.root / "temp")
        self.assertEqual(result.game_id, request.game_id)
        self.assertEqual(trace["result"]["winner"], 0)
        worker.assert_called_once()
```

- [ ] **Step 2: Run the bridge test and verify it fails before implementation**

Run: `python3 -m unittest -v tests.test_arena_runner.SingleGameBridgeTests`

Expected: FAIL because `evaluation.runner.single` does not yet exist.

- [ ] **Step 3: Extract the existing worker call into a public single-game function**

Refactor only shared request/result parsing from `evaluation.runner.batch`; preserve existing batch behavior and tests. The single-game function must use a run-scoped temporary directory, return an error `GameResult` for timeout/worker crash, and never expose private temp files as long-term traces unless the Arena explicitly stores the selected payload.

- [ ] **Step 4: Implement ArenaRunner with idempotent game IDs and rating events**

Before launching a game, query `ArenaStore.completed_game_ids(run_id)`. After a finished game, write the game record, apply both Rating engines, write two rating events, update pair counts, and call `checkpoint()` every `checkpoint_every` games. A failed worker writes a game fact with `status=error` and does not update Rating.

- [ ] **Step 5: Add runner tests for seat alternation, resume and failed games**

Run: `python3 -m unittest -v tests.test_arena_runner`

Expected: tests verify 10-game smoke pairs alternate first player 5/5, re-running an existing game ID does not call the worker, and a worker error is retained without changing either rating.

### Task 5: All/Eligible aggregation and HTML/Markdown reporting

**Files:**
- Create: `arena/reports.py`
- Create: `tests/test_arena_reports.py`
- Create: `arena/templates/README.md`

**Interfaces:**
- `build_report_snapshot(store: ArenaStore, run_id: str, *, include_demoted: bool) -> dict[str, object]` returns `ratings`, `pair_matrix`, `deck_rows`, `source_rows`, `summary`, `config`, `generated_at` and `quality_filter`.
- `render_html(snapshot: Mapping[str, object]) -> str` returns an escaped standalone `<!doctype html>` with embedded JSON data, sortable ranking table, All/Eligible label, W/L/D heatmap, source metadata table, primary Pokémon image cards, and low-quality appendix.
- `render_markdown(snapshot: Mapping[str, object]) -> str` returns Chinese Markdown with the same ranking, aggregate and provenance facts.
- `write_reports(snapshot, destination: Path, *, latest: bool = True) -> tuple[Path, Path]` writes an immutable `run_id` directory and atomically replaces `arena/reports/latest.html`/`latest.md` through temporary sibling files.

- [ ] **Step 1: Write failing report tests**

```python
class ReportTests(unittest.TestCase):
    def test_eligible_report_excludes_demoted_but_keeps_all_report_rows(self) -> None:
        snapshot = fixture_snapshot()
        all_html = render_html(snapshot | {"quality_filter": "all"})
        eligible_html = render_html(snapshot | {"quality_filter": "eligible"})
        self.assertIn("低质量附录", all_html)
        self.assertIn("bad-deck", all_html)
        self.assertNotIn('data-deck-id="bad-deck"', eligible_html)

    def test_html_escapes_source_title_and_contains_heatmap_and_image_url(self) -> None:
        html = render_html(fixture_snapshot())
        self.assertIn("&lt;unsafe&gt;", html)
        self.assertIn("克制关系热力图", html)
        self.assertIn("loading=\"lazy\"", html)
```

- [ ] **Step 2: Run report tests and verify the expected missing-module failure**

Run: `python3 -m unittest -v tests.test_arena_reports`

Expected: FAIL because `arena.reports` does not yet exist.

- [ ] **Step 3: Implement aggregation from stored facts**

Compute overall W/L/D, completion/error rates, per-deck win rates, first-player split, pairwise matrix cells with numerator/denominator, current Gaussian and Elo rankings, sample sufficiency labels, and source metadata. Apply only the requested quality filter to ranking/aggregate rows; preserve demoted deck details in the All report appendix.

- [ ] **Step 4: Implement standalone escaped report rendering**

Use `html.escape` for all text and JSON `<`/`>` escaping for embedded data. Use inline CSS and no CDN dependency. Show official image URLs only when present; otherwise show card name and ID. Include a visible note that `official_gaussian_approx` is not the undisclosed official formula.

- [ ] **Step 5: Run focused report tests and inspect a fixture HTML**

Run: `python3 -m unittest -v tests.test_arena_reports && python3 -m compileall -q arena`

Expected: all report tests pass, generated HTML contains one ranking table and one pairwise matrix, and no raw `<unsafe>` source title is rendered as markup.

### Task 6: Arena CLI, README, live collection and verification

**Files:**
- Create: `arena/cli.py`
- Modify: `arena/__main__.py`
- Create: `arena/README.md`
- Create: `tests/test_arena_cli.py`
- Modify: `pyproject.toml` only if package discovery needs `arena*`

**Interfaces:**
- CLI commands are `collect`, `validate`, `run`, `report` and `discussions`; parser errors use exit code 2 and do not swallow collection failures.
- `collect` supports `--repo-root`, `--competition`, `--include-challenge-archives`, `--max-pages`, `--no-download` and `--discussion-pages`.
- `validate` supports `--repo-root` and records every package as `valid`, `invalid` or `error` without stopping at the first bad source.
- `run` supports `--phase smoke|formal|continuous`, `--games`, `--limit`, `--resume`, `--include-demoted`, `--seed`, `--max-steps`, `--timeout` and `--checkpoint-every`.
- `report` supports `--run-id`, `--include-demoted` and `--output`; it can rebuild reports without a new game.
- `discussions` supports `--competition`, `--pages` and `--refresh` and writes official Code/Discussion research index files.

- [ ] **Step 1: Write failing CLI dispatch tests**

```python
class ArenaCliTests(unittest.TestCase):
    def test_parser_exposes_all_commands_and_defaults_smoke_to_ten_games(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--phase", "smoke"])
        self.assertEqual(args.phase, "smoke")
        self.assertEqual(args.games, 10)

    def test_report_command_does_not_require_kaggle_credentials(self) -> None:
        with patch("arena.cli.write_report_command") as command:
            self.assertEqual(main(["report", "--run-id", "run-1"]), 0)
            command.assert_called_once()
```

- [ ] **Step 2: Run CLI tests and verify the expected missing-command failure**

Run: `python3 -m unittest -v tests.test_arena_cli`

Expected: FAIL because `arena.cli` and its parser do not yet exist.

- [ ] **Step 3: Implement CLI and package discovery**

Use the repository root from `Path(__file__).resolve().parents[1]`, create data directories only for commands that write data, and initialize storage before every data operation. `collect` performs Code metadata/file/output/discussion capture. `validate` loads official card IDs and calls `load_submission_package` plus `assert_cg_compatible`. `run` loads active packages, adds `work/alakazam_v8_current`, creates a run manifest and invokes `ArenaRunner`. `report` reads stored facts only.

- [ ] **Step 4: Write Chinese README and operator workflow**

Document the five commands, source status meanings, official Rating evidence, exact unknown-formula caveat, data retention policy, smoke/formal/continuous phases, resume behavior, and the fact that local Arena results do not replace Kaggle leaderboard results.

- [ ] **Step 5: Run the full relevant test suite and repository checks**

Run:

```bash
python3 -m unittest -v tests.test_arena_storage tests.test_arena_collect tests.test_arena_adapters tests.test_arena_rating tests.test_arena_scheduler tests.test_arena_runner tests.test_arena_reports tests.test_arena_cli
python3 -m compileall -q arena evaluation scripts
python3 scripts/check_assets.py
```

Expected: all Arena tests pass, compileall exits 0, and the existing asset checker remains green.

- [ ] **Step 6: Collect live official sources and run the first smoke Arena**

Run:

```bash
python3 -m arena collect --include-challenge-archives --discussion-pages 2
python3 -m arena validate
python3 -m arena run --phase smoke --games 10 --checkpoint-every 10 --seed 20260722
python3 -m arena report
```

Expected: collection leaves a source status for every listed kernel, validation records every failure without aborting the set, smoke writes `arena/runs/<run-id>/` and reports write `arena/reports/latest.html` and `latest.md`. Any unavailable Kaggle source remains in the catalog with a failure reason.

- [ ] **Step 7: Start resumable continuous evaluation after smoke evidence**

Run the continuous scheduler with a bounded process session so it can be inspected and resumed:

```bash
python3 -m arena run --phase continuous --games 20 --checkpoint-every 10 --seed 20260722 --limit 100000
```

Keep the process alive while resources are available, poll its output, and use `python3 -m arena report` to inspect checkpoints. Do not install a launchd job or submit to Kaggle without a separate explicit request.

## Plan self-review

- Spec coverage: source collection, challenge archive scan, metadata, package adaptation, validation, initial 10-game matrix, formal 20-50-game matrix, continuous nearest-rating scheduling, Gaussian/Elo results, 300-point demotion, All/Eligible reports, images, Code/Discussion capture, resume and long-running CLI are covered by Tasks 1-6.
- Placeholder scan: the plan contains no `TODO`, `TBD`, unspecified “appropriate” behavior, or references to undefined interfaces.
- Type consistency: `ArenaStore`, `RatingState`, `RatingUpdate`, `Pairing`, `ArenaRunConfig`, `GameRecord` and all report/CLI entrypoints are introduced before later tasks consume them.
- Safety review: downloaded archives are path-checked before extraction; Notebook code is never executed during collection; validation and game execution stay in subprocess workers; no credentials or Kaggle submission command is invoked.
- Repository review: no existing evaluation opponent catalog is changed; `engine/source/` is untouched; new generated data is confined to `arena/`.

The plan is ready for inline implementation in the current `to_better_opponents` branch. Per the repository instruction, no automatic commit is part of execution.
