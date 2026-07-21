# Alakazam V9 Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定 V8 60 张卡表的前提下，实现 Alakazam V9 的显式回合计划、两只 Abra/一只 Dunsparce 场面优先级、可用 Fezandipiti Ability、Lana's Aid 接力恢复、攻击优先 Energy 预留，以及可持续的 AutoIteration HTML 历史。

**Architecture:** 保留 `work/alakazam_v9/main.py` 的单文件 Kaggle runtime 契约，在内部增加不可变 `TurnPlan` 和场面缺口/攻击路线 helper；`_main_action`、`_choose_energy_option`、`_select_effect` 只使用 simulator 提供的合法 option。扩展现有 `scripts/auto_iteration_report.py`，把 native evaluation run 的 `summary.json`/`metrics.json` 规范化成 iteration sample 的 `result.json`，每轮生成详情页和外层趋势页。

**Tech Stack:** Python 3.11+、标准库 `unittest`/`dataclasses`/`json`/`pathlib`、官方 cg runtime、`python3 -m evaluation`、`auto_iteration_v8_setup_relay` revision 2、静态 HTML。

## Global Constraints

- V9 固定使用 `work/alakazam_v9/deck.csv` 的现有 60 张卡表；不得修改卡表内容或顺序。
- 不修改 `engine/source/`、官方 cg runtime、`evaluation/configs/opponents.json` 或 profile 分母。
- 当前目标场面是 Active + Bench 至少两只 Abra 系列，再至少一只 Dunsparce 系列；达到最低配置后仍可继续准备。
- 当前回合攻击优先；只有不破坏当前攻击路线的最低配置动作可以在攻击前执行。
- simulator 提供的 option 是唯一合法动作来源；不得伪造 option index 或绕过规则。
- Fezandipiti、Lana's Aid、Energy 和场面缺口行为必须先写失败测试，再写生产代码。
- full evaluation 固定 17 个启用 opponent、每个 10 局，共 170 局；profile 为 `auto_iteration_v8_setup_relay` revision 2。
- baseline 和每个 iteration 使用独立随机批次，不能描述为逐局配对 A/B。
- correctness gate 必须先于任何策略 promotion；agent error 非零时不能晋级。
- 临时 trace 写入 `/tmp` 或 evaluation 临时目录，长期 history 只保存轻量报告和框架选中的 trace。
- 默认不执行 `git commit` 或 `git push`；只有用户明确要求时才执行，并且 commit 后立即 push。
- 每项实现完成后运行对应定向测试；最终声明前必须运行 fresh verification 命令并读取完整结果。

---

## 文件结构

- Modify: `work/alakazam_v9/main.py`：TurnPlan、场面缺口、攻击路线、Energy、Fezandipiti、Lana's Aid 和主动作排序。
- Create: `tests/test_alakazam_v9_strategy.py`：V9 专属 fixture 与行为回归测试。
- Modify: `scripts/auto_iteration_report.py`：native evaluation 输出适配与 V9 iteration 页面 metadata。
- Modify: `tests/test_auto_iteration_report.py`：native run 适配和跨 iteration HTML 测试。
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/baseline/`：V9 固定卡表基线报告。
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/iteration-001/`：Energy/攻击路线候选报告。
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/iteration-002/`：场面不变量/恢复候选报告。
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/iteration-003/`：Fezandipiti 候选报告。
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/index.html`：baseline 到所有 iteration 的趋势页。
- Modify: `work/auto-iteration/history_iterations/alakazam_v9_strategy/DESIGN.md`：每轮完成后追加事实、假设、结果、Decision 和下一轮问题。
- Create or modify: `.superpowers/sdd/progress.md`：记录 task 完成、测试证据和 review 状态；该文件是 scratch ledger，不执行 commit。

### Task 0: 锁定固定卡表与未修改策略 baseline

**Files:**
- Read only: `work/alakazam_v9/main.py`
- Read only: `work/alakazam_v9/deck.csv`
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/baseline/`

**Interfaces:**
- Consumes: untouched `work/alakazam_v9`, fixed opponent catalog, revision 2 metric profile.
- Produces: source/deck SHA-256 record and one native 17 x 10 baseline run.

- [ ] **Step 1: Record immutable-input hashes**

Run `shasum -a 256 work/alakazam_v9/main.py work/alakazam_v9/deck.csv` and record both hashes in
`.superpowers/sdd/progress.md`. Every later iteration must retain the same deck hash; the baseline
manifest must identify the pre-change candidate.

- [ ] **Step 2: Validate before evaluation**

Run:

```bash
python3 -m evaluation validate work/alakazam_v9
python3 -m py_compile work/alakazam_v9/main.py
```

Expected: the existing V9 package validates and compiles without changing either input file.

- [ ] **Step 3: Run the unchanged 170-game baseline**

Run:

```bash
python3 -m evaluation run \
  --candidate work/alakazam_v9 \
  --opponents all \
  --games 10 \
  --no-visualize \
  --metric-profile auto_iteration_v8_setup_relay \
  --output work/auto-iteration/history_iterations/alakazam_v9_strategy/baseline
```

Read the generated run's `manifest.json`, `summary.json`, and `metrics.json`. Confirm 17 opponents,
170 completed games, revision 2, matching candidate hash, and zero agent errors before any
`main.py` edit begins. Do not normalize the run yet; Task 7 performs that after the adapter exists.

### Task 1: 建立 V9 测试骨架和失败 fixture

**Files:**
- Create: `tests/test_alakazam_v9_strategy.py`
- Test: same file

**Interfaces:**
- Consumes: `work/alakazam_v9/main.py` through `importlib.util.spec_from_file_location`.
- Produces: `load_main()`、`card()`、`state()`、`observation()`、`play_option()`、`attack_option()` fixture helpers and failing tests used by Tasks 2-5.

- [ ] **Step 1: Write the failing tests and fixtures**

Add a V9 module loader and helpers equivalent to the existing V8 strategy tests, but point
`MAIN_PATH` to `work/alakazam_v9/main.py`. Include these tests before implementation:

```python
def test_active_fezandipiti_gets_energy_for_ready_bench_alakazam_handoff():
    player = state(active=card(FEZANDIPITI_EX, serial=1),
                   bench=[card(ALAKAZAM, serial=2, energies=[BASIC_PSYCHIC])],
                   hand=[BASIC_PSYCHIC])
    options = [energy_option(BASIC_PSYCHIC, area=4, index=0),
               energy_option(BASIC_PSYCHIC, area=5, index=0),
               {"type": 12}, attack_option(), {"type": 14}]
    assert main._main_action(observation(player=player, options=options)) == [0]

def test_active_abra_attack_route_beats_bench_kadabra_attachment():
    player = state(active=card(ABRA, serial=1, energies=[BASIC_PSYCHIC]),
                   bench=[card(KADABRA, serial=2)],
                   hand=[RARE_CANDY, ALAKAZAM, BASIC_PSYCHIC])
    options = [energy_option(BASIC_PSYCHIC, area=5, index=0),
               {"type": 9, "cardId": ALAKAZAM, "inPlayArea": 4, "inPlayIndex": 0},
               attack_option(), {"type": 14}]
    assert main._main_action(observation(player=player, options=options)) == [1]

def test_lanas_aid_selects_abra_and_basic_psychic_for_missing_successor():
    player = state(active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
                   discard=[ABRA, BASIC_PSYCHIC])
    select = recovery_select(LANAS_AID, [ABRA, BASIC_PSYCHIC], min_count=1, max_count=2)
    assert main._select_effect(observation(player=player, select=select)) == [0, 1]

def test_field_gap_prioritizes_abra_before_dunsparce_and_other_pokemon():
    player = state(active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
                   bench=[card(ABRA, serial=2)],
                   hand=[ABRA, DUNSPARCE, FEZANDIPITI_EX])
    options = [play_option(DUNSPARCE), play_option(ABRA), play_option(FEZANDIPITI_EX), {"type": 14}]
    assert main._main_action(observation(player=player, options=options)) == [1]

def test_legal_fezandipiti_ability_is_used_without_local_ko_log():
    player = state(active=card(ALAKAZAM, serial=1, energies=[BASIC_PSYCHIC]),
                   bench=[card(ABRA, serial=2), card(DUNSPARCE, serial=3)],
                   hand=[900, 901], deck_count=30)
    options = [{"type": 10, "cardId": FEZANDIPITI_EX, "inPlayArea": 5, "index": 0},
               attack_option(), {"type": 14}]
    assert main._main_action(observation(player=player, options=options)) == [0]
```

The fixture must set `turn=3`, `firstPlayer=0`, `supporterPlayed=False`,
`energyAttached=False`, `retreated=False`, `benchMax=5`, and include explicit `handCount`,
`deckCount`, `discard`, `prize`, `energies`, and `energyCards` fields so the failure is caused by
missing V9 behavior rather than malformed observation data.

- [ ] **Step 2: Run the V9 tests to verify RED**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v9_strategy
```

Expected: the new handoff, field-gap, or ability tests fail against the V8-copy implementation;
fixture/import errors must be fixed until the failure names describe missing behavior.

- [ ] **Step 3: Do not implement production code in this task**

Keep the test module as the red contract. Do not change `deck.csv` or `main.py` until Task 2.

- [ ] **Step 4: Record the RED evidence**

Append the command and failing test names to `.superpowers/sdd/progress.md` without committing.

### Task 2: Implement TurnPlan, field counters, and current-attack reservation

**Files:**
- Modify: `work/alakazam_v9/main.py` near `TurnMemory`, field helper functions, and `_main_action`.
- Test: `tests/test_alakazam_v9_strategy.py`

**Interfaces:**
- Consumes: `_active`, `_bench`, `_field_pokemon`, `_has_psychic_energy`, `_retreat_available`, `_has_attack_option`, and simulator option decoder helpers.
- Produces:
  - `@dataclass(frozen=True) class TurnPlan`
  - `_abra_series_count(player: dict[str, Any]) -> int`
  - `_dunsparce_series_count(player: dict[str, Any]) -> int`
  - `_field_gaps(player: dict[str, Any]) -> tuple[int, int]`
  - `_reserve_bench_slots(player: dict[str, Any]) -> int`
  - `_build_turn_plan(current: dict[str, Any], player: dict[str, Any], options: list[dict[str, Any]], select: dict[str, Any]) -> TurnPlan`

- [ ] **Step 1: Add the minimal TurnPlan and counter tests**

Add tests for `main._abra_series_count`, `main._dunsparce_series_count`, `_field_gaps`, and
`_reserve_bench_slots` using fields with evolved and unevolved instances. Assert that one Active
Alakazam plus one Bench Abra counts as two Abra-series instances, and that one Dunsparce is the
only Dunsparce-series instance.

- [ ] **Step 2: Run the new counter tests to verify RED**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v9_strategy.V9StrategyTest.test_field_counters
```

Expected: `AttributeError` for the absent helpers or an assertion failure naming the missing count.

- [ ] **Step 3: Implement the minimal TurnPlan API**

Use this exact shape and recompute it on every main-action observation:

```python
@dataclass(frozen=True)
class TurnPlan:
    attack_now: bool
    attack_route: str | None
    reserved_energy_id: int | None
    reserved_energy_target_serial: int | None
    reserved_supporter_id: int | None
    abra_count: int
    dunsparce_count: int
    missing_abra: int
    missing_dunsparce: int
    reserved_bench_slots: int
    safe_draw: bool
    terminal_attack: bool
```

`_build_turn_plan` must first identify terminal attack, then direct/evolution/attachment/retreat
routes, then calculate field gaps. It may mark `attack_now=True` only when the current option list
contains the route's legal action or a legal sequence step. It must never add an option to the list.

- [ ] **Step 4: Route `_main_action` through TurnPlan**

Call `_build_turn_plan` immediately after `_TURN_MEMORY.sync`. Replace the broad boolean
`preparation_due` check for route-critical decisions with `plan.attack_route`, while leaving
existing special legality filters in place until Tasks 3-5 move their behavior over. Return the
lowest-ranked legal option and keep the existing forbidden Abra/Trading Places fallback.

- [ ] **Step 5: Run the counter and current-route tests to verify GREEN**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v9_strategy
```

Expected: the TurnPlan, field-counter, and current-route tests introduced for Task 2 pass with no
illegal option index. The Lana's Aid and Fezandipiti contracts from Task 1 may remain RED until
Tasks 4 and 5. Run the existing V8 strategy module as a regression check:

```bash
python3 -m unittest -v tests.test_alakazam_v8_sol_deck_opt_strategy
```

### Task 3: Make Energy and Retreat follow TurnPlan

**Files:**
- Modify: `work/alakazam_v9/main.py` in `_choose_energy_option` and the `option_type == 8/12` branches of `_main_action`.
- Test: `tests/test_alakazam_v9_strategy.py`

**Interfaces:**
- Consumes: `TurnPlan` from Task 2 and `_pokemon_from_option`/`_field_option_area`.
- Produces: `_energy_option_matches_plan(option, plan, current) -> bool` and deterministic scoring that ranks current-route Energy before future Bench Energy.

- [ ] **Step 1: Add failing Energy/Retreat tests**

Add tests for Active Fezandipiti handoff, Active Abra direct attack, final-Prize attack before
future Bench attachment, and Telepath Energy on Active when it is the only way to attack this turn.
The tests must include both Energy options and an attack/end option in the same observation.

- [ ] **Step 2: Run the Energy tests to verify RED**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v9_strategy -k energy
```

Expected: at least the Active handoff or direct-route test fails by choosing the Bench option.

- [ ] **Step 3: Implement route-preserving Energy ranking**

Pass `plan` into `_choose_energy_option` (update both its direct callers). Rank options in this
order: plan-reserved Active target, plan-reserved Bench target needed before Retreat, Telepath
Energy that creates the route, Basic Psychic for an Active attack, explicit evolution target,
future Abra with a concrete evolution route, safe Enriching Energy for Dudunsparce, then other
legal targets. If `plan.terminal_attack` is true, score all optional future attachments after the
attack option. Preserve the existing opponent Special Energy removal behavior.

- [ ] **Step 4: Implement Retreat gate**

Return a good Retreat score only if `_retreat_available(current)` is true,
`plan.attack_route` is one of `"retreat_ready_bench"` or `"retreat_after_attach"`, and the target
Bench Pokémon is the planned ready Alakazam. A dead Retreat remains below END and attack. Do not
turn Dunsparce `Trading Places` into a Retreat action.

- [ ] **Step 5: Run Energy and full strategy regressions**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v9_strategy tests.test_alakazam_v8_sol_deck_opt_strategy
```

Expected: all V9 Energy tests and all existing V8 regression tests pass.

### Task 4: Enforce Abra-first/Dunsparce-second field invariant and Lana recovery

**Files:**
- Modify: `work/alakazam_v9/main.py` in Poffin/Basic Pokémon/Lana's Aid/Night Stretcher selection and `_recovery_needs`.
- Test: `tests/test_alakazam_v9_strategy.py`

**Interfaces:**
- Consumes: `_field_gaps`, `_reserve_bench_slots`, `TurnPlan`, `_discard_ids`, `_recovery_can_complete_route`.
- Produces:
  - `_field_anchor_priority(card_id, player) -> tuple[int, int]`
  - `_recovery_selection_ids(effect_id, options, current, player, max_count) -> list[int]`
  - shared gap behavior for Poffin, Telepath Energy, Night Stretcher, Sacred Ash, Lana's Aid and Basic Pokémon play.

- [ ] **Step 1: Add failing field/recovery tests**

Cover: one Abra-series instance plus hand Abra/Dunsparce/Fezandipiti; two Abra instances with no
Dunsparce; Lana's Aid with discard Abra + Basic Psychic; Lana's Aid with only one missing resource;
and Boss/Hilda Supporter reservation. Assert selected card IDs and selection order.

- [ ] **Step 2: Run the field/recovery tests to verify RED**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v9_strategy -k field
python3 -m unittest -v tests.test_alakazam_v9_strategy -k lana
```

Expected: the old independent scoring chooses a lower-priority Pokémon or only one recovery card.

- [ ] **Step 3: Implement shared field anchor priority**

Use priority `0` for Abra when `_abra_series_count(player) < 2`, priority `1` for Dunsparce when
Abra count is at least 2 and Dunsparce count is zero, priority `2` for additional Abra/Dunsparce,
and priority `3` for Fezandipiti/Shaymin/other Pokémon. Add `missing_abra` slots to the Bench
reservation guard; never spend a non-anchor play that consumes a reserved slot.

- [ ] **Step 4: Implement Lana's Aid maximum useful selection**

When effect ID is `LANAS_AID`, inspect all visible options and select up to `max_count` in this
order: an available Abra that fills `missing_abra`, Basic Psychic needed by the recovered route,
Kadabra/Alakazam that continues a visible line, then Dunsparce only after Abra minimum is met.
Select two cards for the explicit Abra + Energy fixture, but do not select unrelated cards merely
to reach `max_count`. Keep Supporter budget checks in `_main_action` so Boss/Hilda current attack
routes outrank Lana.

- [ ] **Step 5: Re-run recovery and existing V8 effect tests**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v9_strategy tests.test_alakazam_v8_sol_deck_opt_strategy
```

Expected: all V9 field/recovery tests pass; existing Sacred Ash and Lana tests retain their prior
ordering unless the new shared field invariant intentionally makes the expected route stricter.

### Task 5: Make Fezandipiti Ability option-driven and safe

**Files:**
- Modify: `work/alakazam_v9/main.py` in `_fezandipiti_needed`, `_fezandipiti_play_is_safe`, `option_type == 10`, and effect context 43.
- Test: `tests/test_alakazam_v9_strategy.py`

**Interfaces:**
- Consumes: `TurnPlan.safe_draw`, `TurnPlan.terminal_attack`, simulator Ability options, `_v6_draw_is_blocked`, `_previous_turn_had_knockout`.
- Produces: `_fezandipiti_ability_allowed(current, player, plan, gain=3) -> bool` and option-driven Ability acceptance.

- [ ] **Step 1: Add failing Ability tests**

Cover legal Ability with no local KO log, legal Ability after KO, a non-KO attack where Ability
must still happen, low deck refusal without terminal closure, and terminal attack precedence.

- [ ] **Step 2: Run Ability tests to verify RED**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v9_strategy -k fezandipiti
```

Expected: the old `fezandipiti_needed` log gate rejects at least the no-log legal option or delays it behind attack.

- [ ] **Step 3: Implement option-driven Ability selection**

When the main option is a legal Fezandipiti Ability, accept it unless `plan.terminal_attack` is
true or `_v6_draw_is_blocked` rejects the draw. Do not require `_previous_turn_had_knockout` for
an already offered Ability option. Keep deck-after-draw and hand-size guards unchanged.

- [ ] **Step 4: Implement hand-to-Bench Fezandipiti guard**

For a Fezandipiti play option, require a confirmed KO trigger and either an already satisfied
minimum board, enough unreserved Bench slots, or a concrete route where the Ability is the only
visible way to find a missing anchor. A Fezandipiti play must not consume a slot reserved for the
second Abra or first Dunsparce.

- [ ] **Step 5: Run Ability and regression tests**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v9_strategy tests.test_alakazam_v8_sol_deck_opt_strategy
```

Expected: all V9 Ability tests pass, no V8 regression test reports an illegal or forbidden action.

### Task 6: Adapt native evaluation artifacts and render V9 history

**Files:**
- Modify: `scripts/auto_iteration_report.py`
- Modify: `tests/test_auto_iteration_report.py`
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/index.html` (generated)

**Interfaces:**
- Consumes: native evaluation `summary.json`, `metrics.json`, `manifest.json`, run path, iteration metadata.
- Produces:
  - `build_native_iteration(summary_path: Path, metrics_path: Path, history_root: Path, *, iteration_id: str, label: str, change_summary: str, decision: str, agent_label: str, control_id: str | None = None) -> dict[str, Any]`
  - `render_iteration_html` support for native-run links and hypothesis metadata.
  - `render_history_index` rows for baseline and `iteration-*` records with stable detail links.

- [ ] **Step 1: Write failing report-adapter tests**

Create temporary native-shaped `summary.json` and `metrics.json` fixtures with outcome,
correctness, powerful_hand, post_ko_relay, attack_quality, and library_pressure payloads. Assert
that `build_native_iteration` writes `result.json`, `iteration.md`, `index.html`, and root
`index.html`; assert that the root page contains iteration IDs, win rates, errors, Powerful Hand,
Post-KO and Decision text.

- [ ] **Step 2: Run report tests to verify RED**

Run:

```bash
python3 -m unittest -v tests.test_auto_iteration_report
```

Expected: the new native adapter test fails because the function and native metric mapping are absent.

- [ ] **Step 3: Implement native metric normalization**

Map native fields into the existing iteration sample shape without changing profile definitions:

```python
def build_native_iteration(
    summary_path: Path,
    metrics_path: Path,
    history_root: Path,
    *,
    iteration_id: str,
    label: str,
    change_summary: str,
    decision: str,
    agent_label: str,
    control_id: str | None = None,
) -> dict[str, Any]:
    """Normalize one native evaluation run into result.json and HTML artifacts."""
```

Preserve native `manifest.json`/`metrics.json` under the iteration run directory. Use the same
metric IDs and denominators, include `control_id` and a `sample_type` field, and represent missing
payloads explicitly rather than silently dropping them.

- [ ] **Step 4: Implement detail and history HTML**

Keep pages static and dependency-free. Each detail page links the native `run-*/report.html` and
its Markdown record. The history table displays baseline and all iteration rows, actual sample
size, first/second turn results, Powerful Hand, Post-KO, attack-quality penalty, correctness errors,
change summary, and Decision.

- [ ] **Step 5: Run report tests and existing report regressions**

Run:

```bash
python3 -m unittest -v tests.test_auto_iteration_report tests.test_evaluation_reporting
```

Expected: all existing report tests and new native adapter tests pass.

### Task 7: Normalize Task 0 baseline and execute V9 AutoIteration rounds

**Files:**
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/baseline/`
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/iteration-001/`
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/iteration-002/`
- Create: `work/auto-iteration/history_iterations/alakazam_v9_strategy/iteration-003/`
- Modify: `work/auto-iteration/history_iterations/alakazam_v9_strategy/DESIGN.md`

**Interfaces:**
- Consumes: validated `work/alakazam_v9`, `evaluation/configs/opponents.json`, report adapter from Task 6.
- Produces: native run artifacts, normalized result JSON, Markdown notes, detail pages, root trend page, and per-round Decision.

- [ ] **Step 1: Verify and normalize the immutable Task 0 baseline**

Locate the single native run produced by Task 0 and re-read its `summary.json`, `metrics.json`,
`manifest.json`, and `report.html`. Confirm its recorded candidate hash still matches Task 0 and the
deck hash still matches the current fixed deck. Call the native adapter with
`iteration_id="baseline"`, `label="V9 fixed-deck baseline"`,
`decision="observe"`, and `agent_label="alakazam_v9_baseline"`.

- [ ] **Step 2: Do not re-run baseline**

Treat Task 0 as the sole unchanged control sample. A second baseline after any `main.py` edit would
not be a baseline and must not replace or merge with the Task 0 run.

- [ ] **Step 3: Run focused evaluation after Task 3**

Use `crustle_wall,kacchan_anti_wall,kiyotah_abomasnow` for four games each, write output below
`/tmp/ptcg-v9-iteration-001-focused`, and record it as mechanism evidence only. Do not compare
focused win rate to full baseline.

- [ ] **Step 4: Run iteration-001 full evaluation**

Run the same 17x10 command with candidate `work/alakazam_v9`, write output to
`iteration-001/`, normalize it with the hypothesis and `control_id="baseline"`, and decide
`promote`, `observe`, or `reject` from correctness, attack-route cases, profile targets, and result guardrails.

- [ ] **Step 5: Run iteration-002 and iteration-003 sequentially**

After each candidate code change passes focused and regression tests, run the same full command to
the next iteration directory, normalize the native artifacts, update `DESIGN.md` with facts and
next question, and rebuild the root `index.html`. Never change deck.csv, opponent count, games,
profile, or metric denominator between these rounds.

### Task 8: Final verification and handoff

**Files:**
- Verify: `work/alakazam_v9/main.py`, `work/alakazam_v9/deck.csv`, generated V9 history, all touched tests/scripts.

**Interfaces:**
- Consumes: all completed tasks and generated evaluation records.
- Produces: verified V9 strategy state, linked static dashboard, and concise user handoff with exact evidence.

- [ ] **Step 1: Run the complete automated test suite**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q evaluation scripts work/alakazam_v9
```

Expected: zero test failures and compileall exit 0.

- [ ] **Step 2: Re-run repository and package gates**

Run:

```bash
python3 scripts/check_assets.py
python3 -m evaluation validate work/alakazam_v9
```

Expected: all repository assets and the fixed 60-card candidate validate. Do not create or submit
an archive unless the user separately requests packaging or Kaggle submission.

- [ ] **Step 3: Inspect generated HTML and metric artifacts**

Verify that `work/auto-iteration/history_iterations/alakazam_v9_strategy/index.html` links to
baseline and every completed iteration, every iteration detail page links its native report, and
each `result.json` preserves sample size, errors, first/second turn values, profile revision, and Decision.

- [ ] **Step 4: Review the final diff and status**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Confirm `work/alakazam_v9/deck.csv` is unchanged, no Kaggle credentials or generated binary data
were added, and no unrelated user changes were reverted. Report any unavailable full evaluation or
failed gate explicitly instead of claiming completion.
