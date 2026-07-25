# Alakazam V7 AutoIter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定 V7 `deck.csv` 的前提下，实现一个能读取隔壁评测框架 trace、提取规则与卡组指标、生成具体失败 case，并比较 control/candidate 的分析式 AutoIter 工具。

**Architecture:** 在当前仓库新增 `scripts/alakazam_auto_iter.py`，内部使用纯 Python 数据类和标准库 JSON 解析，隔离 trace 规范化、指标提取、case 生成、候选比较和外部 evaluator 命令构造。工具不修改策略代码，也不修改卡组；它为每轮提供可复现的 manifest、指标 JSON、case JSONL、Markdown 分析和 candidate/control 决策。

**Tech Stack:** Python 3.11+ 标准库（`argparse`、`dataclasses`、`json`、`subprocess`、`pathlib`、`unittest`）；隔壁仓库的 `eval/alakazam_replay.py` 作为可选外部 trace 生成器。

## Global Constraints

- 固定 `submission/alakazam_v7_auto_iter/deck.csv`，并在工具运行前验证它与 V7/V6 卡组一致。
- 只分析 simulator 已提供的合法 options；工具不得生成或替换 agent action。
- `Powerful Hand` 指 `attackId=1072`，先手检查 engine turn 3，后手检查 engine turn 4。
- `empty_bench_run_away_draw_count` 必须将 Active Dudunsparce、空 Bench、实际选择 `Run Away Draw` 视为硬错误，目标为 0。
- 打手断档同时输出事件级分母和对局级分母，不用败局数量作为唯一跨版本指标。
- candidate/control 使用相同对手、局数和先后手协议；seed 只降低噪声，不能成为唯一晋级证据。
- 原始 trace 留在隔壁评测项目的 reports 目录；当前仓库保存精炼指标、case 和分析结论，不提交大批逐局 JSON。
- 不执行 Kaggle 提交，不自动执行 git commit。

## File Map

- Create: `scripts/alakazam_auto_iter.py` — trace 规范化、指标提取、case 输出、control/candidate 比较和 CLI。
- Create: `tests/test_alakazam_auto_iter.py` — synthetic trace fixtures 和 CLI/domain regression tests。
- Modify: `submission/alakazam_v7_auto_iter/README.md` — AutoIter 命令、指标口径和报告目录说明。
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-00-baseline/` — 初始 V7 AutoIter 分析的精炼产物。

## Task 1: Trace Normalization and Core Metrics

**Files:**
- Create: `tests/test_alakazam_auto_iter.py`
- Create: `scripts/alakazam_auto_iter.py`

**Interfaces:**
- Consumes: 一个评测报告目录，包含 `summary.json` 和递归分布的 `game_*.json`；trace 可以使用隔壁 evaluator 的 raw `observation` 或 summary `players/select` 结构。
- Produces: `EvaluationMetrics`、`CaseRecord`、`AnalysisResult` 数据类，以及 `analyze_report(report_dir: Path, agent_label: str | None = None) -> AnalysisResult`。

- [ ] **Step 1: Write failing tests for second-turn attack and trace normalization**

在 `tests/test_alakazam_auto_iter.py` 建立最小 raw observation fixture，覆盖 raw `select.option` 和 evaluator summary `select.options` 两种格式。先写：

```python
def test_counts_powerful_hand_only_on_agent_turn_three_or_four():
    result = analyze_records([
        make_game(
            label="alakazam_v7",
            alakazam_index=0,
            steps=[agent_step(turn=3, attack_id=1072), opponent_step(turn=3)],
        ),
        make_game(
            label="alakazam_v7",
            alakazam_index=1,
            steps=[agent_step(turn=4, attack_id=1072)],
        ),
    ])
    assert result.metrics.games == 2
    assert result.metrics.second_turn_powerful_hand_games == 2


def test_does_not_count_powerful_hand_on_later_turn():
    result = analyze_records([
        make_game(label="alakazam_v7", alakazam_index=0,
                  steps=[agent_step(turn=5, attack_id=1072)]),
    ])
    assert result.metrics.second_turn_powerful_hand_games == 0
```

The helper fixtures must encode the selected option index, not merely put `attackId=1072` in an unselected option; this proves the analyzer measures the actual action.

- [ ] **Step 2: Run the focused tests and verify the expected RED failure**

Run:

```bash
python3 -m unittest tests.test_alakazam_auto_iter.TestTraceMetrics.test_counts_powerful_hand_only_on_agent_turn_three_or_four tests.test_alakazam_auto_iter.TestTraceMetrics.test_does_not_count_powerful_hand_on_later_turn -v
```

Expected result: import failure because `scripts.alakazam_auto_iter` and `analyze_records` do not exist yet.

- [ ] **Step 3: Implement normalization and summary metrics**

Add these exact data structures and functions:

```python
@dataclass(frozen=True)
class EvaluationMetrics:
    games: int
    wins: int
    losses: int
    draws: int
    errors: int
    win_rate: float
    meta_weighted_win_rate: float
    second_turn_powerful_hand_games: int
    second_turn_powerful_hand_rate: float
    post_ko_count: int
    post_ko_zero_ready_count: int
    post_ko_zero_ready_event_rate: float
    games_with_post_ko_break: int
    games_with_post_ko_break_rate: float
    empty_bench_run_away_draw_count: int
    first_alakazam_turns: tuple[int, ...]


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    source: dict[str, Any]
    failure_class: str
    state_summary: dict[str, Any]
    legal_options: list[dict[str, Any]]
    actual_action: list[int]
    expected_action: list[int]
    expected_reason: str
    case_status: str


@dataclass(frozen=True)
class AnalysisResult:
    metrics: EvaluationMetrics
    cases: tuple[CaseRecord, ...]
    source_files: tuple[str, ...]


def analyze_records(records: list[dict[str, Any]], agent_label: str | None = None) -> AnalysisResult:
    """Analyze already-loaded evaluator records without touching the engine."""


def analyze_report(report_dir: Path, agent_label: str | None = None) -> AnalysisResult:
    """Load summary/game JSON files recursively and analyze their full traces."""
```

Normalize `select.option`/`select.options`, `players`, raw `observation`, and evaluator `role` before metric logic. Count an actual attack only when the selected action index points at the option carrying `attackId=1072`. Determine the agent's physical player from `alakazamPhysicalIndex` and use the record label for role matching.

Detect a post-KO event from an observation log entry with `type=6`, `playerIndex` equal to the Alakazam physical index, `fromArea` in `{4, 5}`, `toArea=3`, and a tracked Abra-line card. Deduplicate by `(serial, toArea)` within a game. Count immediate ready attackers as Bench/Active `Kadabra` or `Alakazam` with Psychic Energy; separately preserve Abra-line state in the case summary so an energized but not immediately attacking Abra is not mistaken for a ready attacker.

Count an empty-Bench `Run Away Draw` case only when the agent's selected main option has `type=10`, card id `66`, the current Active id is `66`, and the current Bench is empty. This is a hard error independent of game result.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python3 -m unittest tests.test_alakazam_auto_iter -v
```

Expected result: all current trace normalization and core metric tests pass.

## Task 2: Failure Cases and Reports

**Files:**
- Modify: `tests/test_alakazam_auto_iter.py`
- Modify: `scripts/alakazam_auto_iter.py`

**Interfaces:**
- Consumes: `AnalysisResult` from Task 1.
- Produces: `write_analysis(result: AnalysisResult, output_dir: Path, manifest: dict[str, Any]) -> None` and Markdown/JSON/JSONL reports.

- [ ] **Step 1: Write failing case and report tests**

Add tests for the two V7 baseline problems:

```python
def test_empty_bench_run_away_draw_creates_hard_case():
    result = analyze_records([make_empty_bench_run_away_game()])
    assert result.metrics.empty_bench_run_away_draw_count == 1
    assert result.cases[0].failure_class == "empty_bench_run_away_draw"
    assert result.cases[0].case_status == "fail"


def test_post_ko_metrics_use_events_and_games_as_separate_denominators():
    result = analyze_records([make_post_ko_game(ready_count=0), make_post_ko_game(ready_count=1)])
    assert result.metrics.post_ko_count == 2
    assert result.metrics.post_ko_zero_ready_count == 1
    assert result.metrics.post_ko_zero_ready_event_rate == 0.5
    assert result.metrics.games_with_post_ko_break == 1
    assert result.metrics.games_with_post_ko_break_rate == 0.5


def test_write_analysis_creates_compact_machine_and_human_reports(tmp_path):
    result = analyze_records([make_empty_bench_run_away_game()])
    write_analysis(result, tmp_path, {"label": "alakazam_v7", "games": 1})
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "cases.jsonl").exists()
    assert (tmp_path / "analysis.md").exists()
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_alakazam_auto_iter.TestCaseReports -v
```

Expected result: failures because case extraction and report writing are not implemented.

- [ ] **Step 3: Implement case extraction and reports**

Create cases for:

- missing second-turn `Powerful Hand` when no actual attack action is present;
- every post-KO zero-ready event;
- every empty-Bench `Run Away Draw` hard error.

Each case must include source opponent/game/turn, Active/Bench/hand/deck/Prize summary, normalized legal options, selected action, and a concrete failure reason. The analyzer must not invent an expected action for an unclassified random-draw failure; use `expected_action=[]` and `case_status="diagnostic"` until a human strategy review assigns the expected action.

`write_analysis` writes:

```text
output_dir/
  metrics.json
  cases.jsonl
  analysis.md
```

`metrics.json` contains all `EvaluationMetrics` fields and source coverage. `cases.jsonl` contains one compact JSON object per case. `analysis.md` contains the sample size, win/error counts, second-turn rate, event/game post-KO rates, hard Run Away Draw count, case table, and explicit trace coverage limitations.

- [ ] **Step 4: Run all analyzer tests and verify GREEN**

Run:

```bash
python3 -m unittest tests.test_alakazam_auto_iter -v
```

Expected result: all tests pass with no warnings.

## Task 3: Control/Candidate Comparison and Evaluator Integration

**Files:**
- Modify: `tests/test_alakazam_auto_iter.py`
- Modify: `scripts/alakazam_auto_iter.py`
- Modify: `submission/alakazam_v7_auto_iter/README.md`

**Interfaces:**
- Consumes: two `metrics.json`/`analysis.md` directories and optional external evaluator root.
- Produces: `compare_reports(control_dir: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]`, `build_replay_command(evaluator_root, agent, label, opponents, games, output, cg_path) -> list[str]`, and CLI subcommands `analyze`, `compare`, `run`.

- [ ] **Step 1: Write failing tests for soft promotion gates and command construction**

Add tests:

```python
def test_candidate_passes_when_case_improves_and_win_rate_is_flat():
    decision = decide_promotion(control_metrics(win_rate=0.60, second_turn=0.27),
                                candidate_metrics(win_rate=0.60, second_turn=0.28,
                                                  empty_bench_draws=0),
                                target_case_improved=True)
    assert decision.status == "accept"


def test_candidate_is_rejected_after_two_independent_win_rate_declines():
    decision = decide_promotion(
        control_metrics(win_rate=0.60),
        candidate_metrics(win_rate=0.55),
        target_case_improved=True,
        independent_pairs=[(0.60, 0.55), (0.61, 0.56)],
    )
    assert decision.status == "reject"
    assert "win_rate" in decision.reasons


def test_replay_command_keeps_fixed_protocol():
    command = build_replay_command(
        evaluator_root=Path("/tmp/ptcg-agent-kaggle"),
        agent=Path("submission/alakazam_v7_auto_iter/main.py"),
        label="alakazam_v7_auto_iter",
        opponents=["kiyotah_dragapult"],
        games=10,
        output=Path("/tmp/iter-focus"),
        cg_path=Path("submission/alakazam_v7_auto_iter"),
    )
    assert "--save-traces" in command
    assert "--games" in command and "10" in command
    assert "kiyotah_dragapult" in command
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_alakazam_auto_iter.TestComparisonAndRunner -v
```

Expected result: failures because promotion decisions and evaluator command construction do not exist.

- [ ] **Step 3: Implement comparison and runner**

Add:

```python
@dataclass(frozen=True)
class PromotionDecision:
    status: str  # "accept", "observe", or "reject"
    reasons: tuple[str, ...]
    regressions: tuple[str, ...]


def decide_promotion(
    control: EvaluationMetrics,
    candidate: EvaluationMetrics,
    target_case_improved: bool,
    independent_pairs: list[tuple[float, float]] | None = None,
) -> PromotionDecision:
    """Apply correctness, case, signal and soft outcome gates."""


def compare_reports(control_dir: Path, candidate_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write comparison.json and decision.md for one candidate/control pair."""


def build_replay_command(
    evaluator_root: Path,
    agent: Path,
    label: str,
    opponents: list[str],
    games: int,
    output: Path,
    cg_path: Path,
) -> list[str]:
    """Build, but do not execute, the external alakazam_replay.py command."""
```

The `run` subcommand executes only this explicit external evaluator command and writes a manifest containing command, agent path, deck hash, opponents, game count, swap policy, and seed policy. It never submits to Kaggle. `compare` applies these rules:

- any correctness error or nonzero empty-Bench Draw count is `reject`;
- case improvement with flat outcome metrics can be `accept`;
- a single noisy outcome decline is `observe`;
- candidate outcome lower than control in both independent batches is `reject`;
- a candidate that does not improve its target case cannot be `accept`.

The command builder must use `PYTHONDONTWRITEBYTECODE=1` when invoked by the CLI and pass `--save-traces`, `--games`, `--opponents`, `--output`, `--cg-path`, and `--agent` exactly to `eval/alakazam_replay.py`.

- [ ] **Step 4: Update AutoIter README and verify the complete test module**

Add a Chinese usage section to `submission/alakazam_v7_auto_iter/README.md`:

```bash
python3 scripts/alakazam_auto_iter.py analyze \
  --report-dir /path/to/reports/alakazam_v7 \
  --output-dir docs/history/kaggle/alakazam-v7-auto-iter/iter-00-baseline

python3 scripts/alakazam_auto_iter.py compare \
  --control docs/history/kaggle/alakazam-v7-auto-iter/iter-01/control \
  --candidate docs/history/kaggle/alakazam-v7-auto-iter/iter-01/candidate \
  --output-dir docs/history/kaggle/alakazam-v7-auto-iter/iter-01/comparison
```

Explain that raw traces stay in the adjacent evaluation repository, the deck is fixed, V7 metrics are diagnostic plus guardrails, and each strategy edit must be accompanied by a case hypothesis.

Run:

```bash
python3 -m unittest tests.test_alakazam_auto_iter
python3 -m unittest tests.test_alakazam_v7_strategy tests.test_alakazam_v6_strategy
```

Expected result: all AutoIter, V7 and V6 tests pass.

## Task 4: Initial V7 Baseline Iteration

**Files:**
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-00-baseline/metrics.json`
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-00-baseline/cases.jsonl`
- Create: `docs/history/kaggle/alakazam-v7-auto-iter/iter-00-baseline/analysis.md`

**Interfaces:**
- Consumes: the available V7 detailed evaluator report and its trace files under `/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7`.
- Produces: the first machine-readable V7 AutoIter baseline and a concrete list of strategy cases for the next iteration.

- [ ] **Step 1: Run the analyzer against the available V7 trace report**

Run:

```bash
python3 scripts/alakazam_auto_iter.py analyze \
  --report-dir /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7 \
  --output-dir docs/history/kaggle/alakazam-v7-auto-iter/iter-00-baseline \
  --agent-label alakazam_v7
```

The report must state the number of trace files actually analyzed. It must not claim coverage of all 170 games if the adjacent report directory contains fewer complete trace files.

- [ ] **Step 2: Review the generated cases against V7 EVAL_RESULT.md**

Check that the baseline report exposes the two known V7 issues when their source traces are available: empty-Bench `Run Away Draw` cases and post-KO zero-ready cases. Compare the generated second-turn count with the trace coverage, not blindly with `46/170` when only a subset of traces is present.

- [ ] **Step 3: Write the first analysis-driven iteration brief**

Add to `analysis.md` a short next-iteration brief selecting one minimal hypothesis. The initial candidate should target the hard error first: prevent Active Dudunsparce from using `Run Away Draw` when Bench is empty, then rerun the affected cases before touching attack-line energy priorities. Do not modify `deck.csv` or claim a win-rate improvement from this baseline-only pass.

- [ ] **Step 4: Run repository validation**

Run:

```bash
python3 scripts/check_assets.py
python3 -m compileall -q scripts submission
git diff --check
```

Expected result: all checks pass; no Kaggle command is executed.
