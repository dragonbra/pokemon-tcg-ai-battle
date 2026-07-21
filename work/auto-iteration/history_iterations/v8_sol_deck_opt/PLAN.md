# V8 Sol Deck Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `work/alakazam_v8_current/main.py` 中实现 deck notes 驱动的 V8 策略语义，并用统一 evaluation profile 验证 setup、攻击和接力能力。

**Architecture:** 保留现有单文件 agent、60 张卡组和官方 cg runtime；通过少量事实/判定 helper 与 `_main_action`、`_select_effect` 的优先级分支表达回合阶段和效果目标。每个行为改动先由独立 fixture 测试锁定，再以 17×10 evaluation 检查真实对局指标。

**Tech Stack:** Python 3.11+、标准库 `unittest`、官方 cg runtime、`python3 -m evaluation`、`auto_iteration_v8_setup_relay` revision 2。

## Global Constraints

- 只修改 `work/alakazam_v8_current/main.py`、新增/修改针对性测试和本目录记录。
- 不修改 `deck.csv`、`cg/`、`evaluation/`、`engine/source/` 或历史 submission。
- 不执行 `kaggle competitions submit`、`git commit` 或 `git push`。
- 所有策略动作必须来自 simulator 提供的合法 options，并保持确定性。
- 攻击是回合终止提交；`Trading Places` 永远不选；低阶段非 KO 攻击只能是最后手段。
- 每一轮完整评测使用全部 17 个 opponent、每个 10 局、`--no-visualize` 和 profile revision 2。

---

### Task 1: 建立新鲜 baseline 与本轮评测记录

**Files:**
- Create: `work/auto-iteration/history_iterations/v8_sol_deck_opt/baseline-2026-07-21.md`
- Create: `work/auto-iteration/history_iterations/v8_sol_deck_opt/baseline/`（evaluation 输出）

**Interfaces:**
- Consumes: `work/alakazam_v8_current`、`evaluation/configs/opponents.json` 和 profile `auto_iteration_v8_setup_relay` revision 2。
- Produces: 可复核的 `manifest.json`、`summary.json`、`metrics.json`、`games.jsonl`、`cases.jsonl`、`report.md` 与 `report.html`。

- [x] **Step 1: Validate the candidate package.**

Run:

```bash
python3 -m evaluation validate work/alakazam_v8_current
```

Expected: candidate、60 张 deck、cg tree 和运行时依赖全部通过。

- [x] **Step 2: Run the fresh 17×10 baseline.**

Run:

```bash
python3 -m evaluation run \
  --candidate work/alakazam_v8_current \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output work/auto-iteration/history_iterations/v8_sol_deck_opt/baseline
```

Expected: exit 0，生成一个 `run-*/` 目录且 candidate error 为 0；若出现错误，记录实际状态，不把它当作策略改动结果。

- [x] **Step 3: Extract the baseline headline metrics.**

Read the generated `summary.json` and `metrics.json`，在 `baseline-2026-07-21.md` 记录 run id、总体/先后手 W-L-D、Powerful Hand、Post-KO relay、attack quality、deck pressure、空 Bench Run Away Draw、逐 opponent 胜率和 errors。

### Task 2: 建立行为 fixture 与第一组失败测试

**Files:**
- Create: `tests/test_alakazam_v8_sol_deck_opt_strategy.py`
- Modify: `work/auto-iteration/history_iterations/v8_sol_deck_opt/PLAN.md`

**Interfaces:**
- Consumes: `work/alakazam_v8_current/main.py` 的 `agent`、`_main_action`、`_select_effect` 和现有 observation schema。
- Produces: 可独立加载当前 main、清空 `_TURN_MEMORY`、构造合法最小 observation，并断言 action/effect index 的行为测试。

- [x] **Step 1: Write RED tests for immutable rules and mandatory effects.**

测试至少覆盖：

```python
def test_never_submits_trading_places():
    # Active Dunsparce + Bench Alakazam 时，attack options 含 Trading Places，结果不能选它。
    self.assertNotEqual(main._main_action(obs), [trading_places_index])

def test_fezandipiti_draw_is_selected_after_previous_turn_knockout():
    # 上回合己方 Pokémon 被 KO，Flip the Script 的 yes/no option 合法时必须选 yes。
    self.assertEqual(main._main_action(obs), [fezandipiti_option_index])

def test_enhanced_hammer_selects_active_protective_special_energy_first():
    # 对手 Active 有 Mist/Rock 特殊能量、Bench 也有特殊能量时选择 Active 目标。
    self.assertEqual(main._select_effect(obs), [active_energy_index])

def test_run_away_draw_is_used_to_release_ready_bench_alakazam():
    # Active Dunsparce + Bench 带 Psychic 的 Alakazam 时优先使用 Dudunsparce Ability。
    self.assertEqual(main._main_action(obs), [run_away_draw_index])

def test_sacred_ash_selects_all_legal_attack_line_cards_before_other_pokemon():
    # maxCount=5 时，按 Abra/Kadabra/Alakazam 接力优先并尽可能选满。
    self.assertEqual(main._select_effect(obs), expected_five_indices)
```

- [x] **Step 2: Run only the new tests and confirm the intended RED failures.**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v8_sol_deck_opt_strategy
```

Expected: 新增行为测试失败在现有策略分支，而不是 fixture import/schema 错误。

### Task 3: 实现 Dudunsparce/攻击终止/资源消耗阶段

**Files:**
- Modify: `work/alakazam_v8_current/main.py`
- Test: `tests/test_alakazam_v8_sol_deck_opt_strategy.py`

**Interfaces:**
- Consumes: Task 2 的 observation fixture、`_main_action` 的合法 option 列表和 `_TURN_MEMORY`。
- Produces: `_dudunsparce_handoff_due(current, player, options)`、`_pre_attack_preparation_due(current, player, options, select)` 两个确定性判定，供 `_main_action` 在攻击分支前使用。

- [x] **Step 1: Implement only the minimum Dudunsparce handoff predicate.**

要求：Active 为 Dunsparce、Bench 存在带 Psychic 的 Alakazam，且本回合存在 Dudunsparce evolution/Ability route 时返回 true；Bench 为空或没有接班者时返回 false。

- [x] **Step 2: Re-run the Dudunsparce RED test and confirm GREEN.**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v8_sol_deck_opt_strategy.V8SolDeckOptStrategyTest.test_run_away_draw_is_used_to_release_ready_bench_alakazam
```

Expected: PASS。

- [x] **Step 3: Implement the pre-attack gate.**

在攻击排序前阻止仍有合法收益的攻击，包括合法进化、Fezandipiti Ability、Dudunsparce draw/handoff、有效的检索/回收/附能和必要的 Bench Kadabra draw；保留 `Powerful Hand` 的最终提交位置。

- [x] **Step 4: Add and run tests for Bench Kadabra preparation and forbidden attacks.**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v8_sol_deck_opt_strategy
```

Expected: 新增测试全部 PASS。

### Task 4: 实现 V8 卡牌效果目标与资源保留

**Files:**
- Modify: `work/alakazam_v8_current/main.py`
- Test: `tests/test_alakazam_v8_sol_deck_opt_strategy.py`

**Interfaces:**
- Consumes: `_select_effect`、`_choose_card_option`、`_choose_energy_option` 的当前 selection schema。
- Produces: Fezandipiti、Enhanced Hammer、Nighttime Mine、Sacred Ash、Lana’s Aid、Dawn/Hilda/Poké Pad 的 deck-notes 目标选择。

- [x] **Step 1: Implement and verify Fezandipiti Ability and Nighttime Mine action priority.**

合法 Ability 在上一回合 KO 后必须被接受；Nighttime Mine 在手牌且 simulator 提供合法 play option 时，不依赖 Tera 可见性才允许被选择。

- [x] **Step 2: Implement and verify Enhanced Hammer target ordering.**

先 Active 的 Mist/Rock 等保护性特殊能量，再 Active 其它特殊能量，再 Bench 特殊能量；没有特殊能量目标时不伪造选择。

- [x] **Step 3: Implement and verify maximum useful recovery selection.**

Sacred Ash 在 `maxCount` 范围内尽量选满，Abra 系列按接力密度优先；Lana’s Aid、Dawn、Hilda 和 Poké Pad 的目标按当前攻击/进化/换位路线排序，不把“找得到”当成“值得现在找”。

- [x] **Step 4: Run the effect-focused tests and full current V8 test modules.**

Run:

```bash
python3 -m unittest -v tests.test_alakazam_v8_sol_deck_opt_strategy
python3 -m unittest discover -s tests -p 'test_*.py'
```

Expected: 两条命令均 exit 0。

### Task 5: 运行 focused/full evaluation 并按 evidence 迭代

**Files:**
- Create: `work/auto-iteration/history_iterations/v8_sol_deck_opt/iteration-001/`
- Create: `work/auto-iteration/history_iterations/v8_sol_deck_opt/iteration-002/`（如需要）
- Create: `work/auto-iteration/history_iterations/v8_sol_deck_opt/iteration-*.md`

**Interfaces:**
- Consumes: baseline 与每次 candidate 的 `summary.json`、`metrics.json`、`games.jsonl`、`cases.jsonl`。
- Produces: 每轮变更、证据、指标比较和下一步 decision；不复制策略代码。

- [x] **Step 1: Run the first candidate evaluation with the same 17×10 profile.**

Run:

```bash
python3 -m evaluation run \
  --candidate work/alakazam_v8_current \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output work/auto-iteration/history_iterations/v8_sol_deck_opt/iteration-001
```

Expected: 生成独立 run 目录；报告中 correctness 无 candidate error，或明确记录错误分类。

- [x] **Step 2: Compare metrics without changing the profile.**

按 outcome → Powerful Hand → Post-KO relay → attack quality 的顺序比较；若只有随机波动且没有新的行为 evidence，不据此增加无关规则。

- [x] **Step 3: If a concrete failure remains, write one new RED fixture before the next code edit.**

每次后续改动只处理一个首个行为断点，先运行目标测试失败，再运行目标测试和回归集通过；最多保留与 deck notes 直接相关的迭代。

### Task 6: 最终验证与交付记录

**Files:**
- Modify: `work/auto-iteration/history_iterations/v8_sol_deck_opt/FINAL.md`

- [x] **Step 1: Run package and syntax checks.**

```bash
python3 scripts/check_assets.py
python3 -m compileall -q work/alakazam_v8_current evaluation scripts tests
git diff --check
```

- [x] **Step 2: Run the final 17×10 evaluation after the last code change.**

使用新的 `final/` 输出目录，不复用之前的 run，确保最终指标有新鲜证据。

- [x] **Step 3: Write FINAL.md.**

记录修改文件、baseline 与 final run id、完整指标、已实现 deck 语义、未解决的失败分类和未执行 commit/push/Kaggle submission 的状态。
