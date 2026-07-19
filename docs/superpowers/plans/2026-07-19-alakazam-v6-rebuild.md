# Alakazam V6 Strategy Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定 `v5_auto_iter/deck.csv` 的前提下，重构 Alakazam V6 的回合状态、攻击终止语义、进化/能量/Supporter 编排、资源预算和效果选择。

**Architecture:** V6 复用 V5 AutoIter 的 Kaggle 入口、官方 `cg/` runtime 和合法 option 选择器，但把主行动决策改为显式的 `TurnMemory` + `V6Plan` 顺序。规则预算和进化时钟作为硬约束，攻击只在准备计划完成后提交；牌库、Prize、恢复和对手可见事实作为动作价值输入。HTML 流程图与代码共享同一套中文策略定义，便于 replay 对照。

**Tech Stack:** Python 3.11+、标准库 `dataclasses`/`unittest`、现有官方 simulator runtime、HTML5/CSS/SVG/原生 JavaScript。

## Global Constraints

- V6 必须固定复用 `submission/alakazam_v5_auto_iter/deck.csv`，不得修改 60 张卡牌组成或顺序。
- V6 submission 目录必须自包含 `main.py`、`deck.csv`、`cg/`；只返回 simulator 当前提供的合法 option。
- 攻击、Supporter、手动附能、Retreat 和进化时机必须遵循 simulator 与官方规则；Trading Places 在 V6 初版禁用。
- 牌库正好 10 张进入中局保护；低于 10 张时只有能在本回合完成最后 Prize 闭环才允许继续消耗牌库。
- 有 Enriching Energy、Active Alakazam 已能攻击且场上有 Dudunsparce 时，默认优先启动 Dudunsparce 过牌；没有 Enriching Energy 时能量优先补未充能 Abra 线。
- 当前 Active 不能 KO 而 Bench 有确定 KO 时，Boss 优先；多个合法 KO 目标按剩余 HP 从高到低选择。
- Active Alakazam 已能攻击、无 Boss KO 目标、对手手牌至少 6 张且 Active 已受伤未 KO 时，Xerosic 优先于攻击。

---

### Task 1: V5/V6 策略流程图

**Files:**
- Create: `submission/alakazam_v6/strategy-flow.html`
- Modify: `submission/alakazam_v6/REFACTOR_DESIGN.md` only if the rendered flow exposes a contradiction

**Interfaces:**
- Consumes: V5 `agent(obs)`/`_main_action`/`_select_effect` and the confirmed V6 rules in `REFACTOR_DESIGN.md`.
- Produces: A self-contained local HTML page with V5 and V6 columns, action priority lanes, state memory, and clickable node details.

- [ ] **Step 1: Write the V5/V6 comparison skeleton**

  Include `agent → turn synchronization → main-action planner → effect selector → legal action`, with V5 shown as score ordering and V6 shown as phase ordering.

- [ ] **Step 2: Add the confirmed V6 rules**

  Cover attack termination, evolution clock, one Supporter/manual Energy/Retreat budget, Active Kadabra natural evolution, Dudunsparce/Enriching Energy, Boss HP target rule, Xerosic six-card threshold, recovery order, ten-card deck reserve, ledger/history, and disabled Trading Places.

- [ ] **Step 3: Validate the static page**

  Run an HTML parser, check all required markers, reject external URLs and external script/style references, and run `git diff --check`.

### Task 2: V6 behavior tests before implementation

**Files:**
- Create: `tests/test_alakazam_v6_strategy.py`
- Test: `submission/alakazam_v6/main.py`

**Interfaces:**
- Consumes: Pure helper functions and deterministic `_main_action`/effect-selection behavior from V6.
- Produces: Regression coverage for the rules that differ from V5.

- [ ] **Step 1: Write tests for the V6 decision surface**

  Cover: Trading Places is never selected; a natural Active Kadabra evolution precedes a non-KO attack; Bench Abra can evolve to Kadabra after Active Alakazam; Enriching Energy targets Dudunsparce when the Active Alakazam is ready; Xerosic requires six opposing cards and a wounded non-KO Active; Boss targets the highest-HP visible KO Bench target; deck protection blocks non-terminal draw at ten cards; Dudunsparce net deck change accounts for attached cards; and `TurnMemory` records turn-start Pokémon/evolution/supporter/energy/retreat/attack state.

- [ ] **Step 2: Run the new test file and verify expected RED failures**

  Run: `python3 -m unittest -v tests.test_alakazam_v6_strategy`

  Expected: import or assertion failures because V6 behavior/helpers do not exist yet.

### Task 3: Implement V6 runtime and strategy

**Files:**
- Create: `submission/alakazam_v6/main.py`
- Create: `submission/alakazam_v6/deck.csv` by exact copy of `submission/alakazam_v5_auto_iter/deck.csv`
- Create: `submission/alakazam_v6/cg/` by exact copy of the V5 AutoIter runtime
- Modify: `tests/test_alakazam_v6_strategy.py` only when a test assertion is proven wrong by the simulator contract

**Interfaces:**
- Consumes: V5 helper/effect-selection implementation and the Task 2 tests.
- Produces: A self-contained `agent(obs_dict)` entry point and explicit `TurnMemory`/planner helpers.

- [ ] **Step 1: Copy the self-contained V5 runtime assets and verify deck equality**

  Use `cmp` for `deck.csv` and `cg/` file hashes before editing Python behavior.

- [ ] **Step 2: Add `TurnMemory` and turn synchronization**

  Track per-game reset, turn-start serials/snapshots, played/evolved turn markers, supporter/manual-energy/retreat budgets, attack submission, and visible zone counts for Abra/Kadabra/Alakazam/Dunsparce/Energy. Reconcile simulator booleans before scoring actions.

- [ ] **Step 3: Add the V6 deck/prize/draw route helpers**

  Implement explicit effective attack, current/next attacker, Retreat handoff, Boss KO target, six-card Xerosic, Lana/Night recovery path, Dudunsparce net deck delta, terminal Prize closure, and ten-card protection checks.

- [ ] **Step 4: Replace main-action scoring with ordered V6 planning**

  Order legal options as: mandatory win/KO preparation, required Active evolution and support setup, handoff/energy, safe non-Supporter draw, targeted Supporter, low-stage fallback attack, then END. Assign Trading Places a lower priority than END. Ensure attack is selected only after pre-attack options are exhausted or it has direct Prize value.

- [ ] **Step 5: Update effect selection to preserve the plan**

  Choose evolution targets by current turn-start legality and Active/Bench route, Boss targets by KO then highest remaining HP, recovery targets by explicit missing route, and discard/retrieval choices by future attack value. Keep effect serial reset on `select=None`.

- [ ] **Step 6: Run the focused tests and verify GREEN**

  Run: `python3 -m unittest -v tests.test_alakazam_v6_strategy`

  Expected: all V6 behavior tests pass with no warnings.

### Task 4: Submission documentation and verification

**Files:**
- Create: `submission/alakazam_v6/README.md`
- Create: `submission/alakazam_v6/STRATEGY.md`
- Modify: `submission/alakazam_v6/REFACTOR_DESIGN.md` to mark implemented behaviors and leave replay-calibration items explicit

- [ ] **Step 1: Document V6 scope and V5 comparison**

  State the fixed deck boundary, the rule-semantic strategy order, the explicit state model, and the known experimental assumptions.

- [ ] **Step 2: Run repository validation**

  Run: `python3 scripts/check_assets.py`; `python3 -m compileall -q scripts submission`; `git diff --check`.

- [ ] **Step 3: Run a local legality smoke test and package inspection**

  Write local battle output only to `/tmp`, then run `bash scripts/package_submission.sh alakazam_v6` and `tar -tzf dist/alakazam_v6.tar.gz`. Confirm archive root contains `main.py`, `deck.csv`, and `cg/`.

- [ ] **Step 4: Perform a final code review**

  Check the diff against this plan and the confirmed V6 design, with special attention to illegal post-attack preparation, accidental second Supporter/manual Energy/Retreat, hidden Prize assumptions, and Trading Places leakage.
