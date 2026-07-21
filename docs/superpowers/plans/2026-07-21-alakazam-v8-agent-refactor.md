# Alakazam V8 Agent Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `work/alakazam_v8_current/main.py` 重构为显式 TurnPlan 驱动的多模块 Agent，完整实现 V8 Deck Notes 语义，并让命名策略 profile 可以安全组合与评测。

**Architecture:** 顶层 `main.py` 只暴露 evaluator 契约；`strategy/` 将 observation 依次转换为 Facts、Routes、TurnPlan、SemanticOption 和 Decision。主动作通过固定阶段与 plan goal 选择，不使用跨领域全局 score；effect selector 只执行已经选定的计划目的。

**Tech Stack:** Python 3.11+、标准库 `dataclasses`/`enum`/`collections`、现有 `unittest`、Kaggle `cg/` runtime、Bash/tar。

## Global Constraints

- 策略真值优先级以 `docs/superpowers/specs/2026-07-21-alakazam-v8-agent-refactor-design.md` 为准。
- 保持 `DECK` 和 `agent(obs_dict) -> list[int]` evaluator 契约。
- 只返回 simulator 提供的合法 option index；`Trading Places` 永远不返回。
- 不添加 Kaggle 现场需要联网安装的第三方依赖。
- Python 使用 4 空格、类型注解、100 字符行宽；repo-facing 注释和说明优先中文。
- 不修改用户当前正在进行的 `evaluation/`、`work/auto-iteration/` 或无关 notes 改动。
- 按仓库规则不自动 commit；每个任务以测试和 diff checkpoint 结束，只有用户明确要求才提交。

---

## File Map

**Create:**

- `work/alakazam_v8_current/strategy/__init__.py`：包入口。
- `work/alakazam_v8_current/strategy/cards.py`：V8 卡牌常量与 `DeckSpec`。
- `work/alakazam_v8_current/strategy/model.py`：领域枚举和不可变 dataclass。
- `work/alakazam_v8_current/strategy/memory.py`：跨 observation 状态。
- `work/alakazam_v8_current/strategy/facts.py`：observation 适配。
- `work/alakazam_v8_current/strategy/options.py`：原始 option 语义化。
- `work/alakazam_v8_current/strategy/routes.py`：攻击、接班、恢复、换位路线。
- `work/alakazam_v8_current/strategy/profiles.py`：不可变策略 profile。
- `work/alakazam_v8_current/strategy/planner.py`：TurnPlan 生成与提交闸门。
- `work/alakazam_v8_current/strategy/orchestrator.py`：运行时编排。
- `work/alakazam_v8_current/strategy/policies/{__init__,setup,continuity,resources,control,commit}.py`：主动作策略。
- `work/alakazam_v8_current/strategy/effects/{__init__,dispatcher,search,recovery,targeting}.py`：effect 选择。
- `tests/alakazam_v8_fixtures.py`：共享 observation/option fixture builders。
- `tests/test_alakazam_v8_facts.py`、`tests/test_alakazam_v8_planner.py`、`tests/test_alakazam_v8_effects.py`、`tests/test_alakazam_v8_agent.py`：分层测试。

**Modify:**

- `work/alakazam_v8_current/main.py`：切换到 orchestrator 并删除旧策略实现。
- `tests/test_alakazam_v8_luna_deck_opt_strategy.py`：迁移为新公共接口/兼容行为测试。
- `scripts/package_submission.sh`：把 `strategy/` 纳入归档。
- `tests/test_package_submission.py`：验证多模块归档和解包导入。

---

### Task 1: 锁定 V8 反例与 Characterization

**Files:**
- Create: `tests/alakazam_v8_fixtures.py`
- Create: `tests/test_alakazam_v8_agent.py`
- Modify: `tests/test_alakazam_v8_luna_deck_opt_strategy.py`

**Interfaces:**
- Produces: `pokemon()`, `player()`, `main_obs()`, `effect_obs()` fixture builders。
- Produces: V8C-01 至 V8C-10 的 option identity 断言，不依赖稳定数组位置之外的策略内部实现。

- [ ] **Step 1: 提取共享 fixture builders**

```python
def pokemon(
    card_id: int,
    serial: int,
    *,
    hp: int = 100,
    energies: list[int] | None = None,
    energy_cards: list[int] | None = None,
) -> dict:
    return {
        "id": card_id,
        "serial": serial,
        "energies": list(energies or []),
        "energyCards": [
            {"id": energy_id, "serial": serial * 100 + index}
            for index, energy_id in enumerate(energy_cards or [])
        ],
        "appearThisTurn": False,
        "preEvolution": [],
        "hp": hp,
        "maxHp": hp,
    }

def player(
    *,
    active: dict | None = None,
    bench: list[dict] | None = None,
    hand: list[int] | None = None,
    discard: list[int] | None = None,
    deck_count: int = 30,
) -> dict:
    hand_ids = list(hand or [])
    return {
        "active": [active] if active else [],
        "bench": list(bench or []),
        "benchMax": 5,
        "hand": [{"id": card_id, "serial": 9000 + index}
                 for index, card_id in enumerate(hand_ids)],
        "handCount": len(hand_ids),
        "deckCount": deck_count,
        "discard": [{"id": card_id, "serial": 7000 + index}
                    for index, card_id in enumerate(discard or [])],
        "prize": [{} for _ in range(6)],
    }

def main_obs(me: dict, opponent: dict, options: list[dict], *, turn: int = 5) -> dict:
    return {
        "current": {
            "turn": turn,
            "yourIndex": 0,
            "firstPlayer": 0,
            "players": [me, opponent],
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
        },
        "logs": [],
        "select": {"type": 0, "context": 0, "minCount": 1, "maxCount": 1,
                   "option": options},
    }

def effect_obs(
    me: dict,
    opponent: dict,
    options: list[dict],
    *,
    effect_id: int,
    context: int,
    min_count: int = 1,
    max_count: int = 1,
) -> dict:
    obs = main_obs(me, opponent, options)
    obs["select"] = {
        "type": 1,
        "context": context,
        "effect": {"id": effect_id, "serial": 5000},
        "minCount": min_count,
        "maxCount": max_count,
        "option": options,
    }
    return obs
```

- [ ] **Step 2: 写三个已确认失败的回归测试**

```python
def test_nonterminal_ko_waits_for_bench_kadabra_evolution(self):
    me = player(
        active=pokemon(ALAKAZAM, 1, energies=[PSYCHIC_ENERGY_TYPE]),
        bench=[pokemon(KADABRA, 2)],
        hand=[ALAKAZAM, 900, 901],
    )
    opponent = player(active=pokemon(900, 3, hp=40))
    options = [
        {"type": 9, "cardId": ALAKAZAM, "inPlayArea": 5, "inPlayIndex": 0},
        {"type": 13, "attackId": POWERFUL_HAND_ATTACK},
        {"type": 14},
    ]
    self.assertEqual(MODULE.agent(main_obs(me, opponent, options)), [0])

def test_dudunsparce_search_requires_ready_bench_attacker(self):
    me = player(
        active=pokemon(DUNSPARCE, 1),
        bench=[pokemon(ABRA, 2)],
        hand=[POKE_PAD],
    )
    opponent = player(active=pokemon(900, 3, hp=220))
    options = [
        {"type": 1, "cardId": KADABRA},
        {"type": 1, "cardId": DUDUNSPARCE},
    ]
    obs = effect_obs(me, opponent, options, effect_id=POKE_PAD, context=7)
    self.assertEqual(MODULE.agent(obs), [0])

def test_sacred_ash_recovers_complete_lines_before_duplicate_stages(self):
    discard = [ABRA, ABRA, KADABRA, KADABRA, ALAKAZAM, ALAKAZAM]
    me = player(active=pokemon(DUNSPARCE, 1), discard=discard)
    opponent = player(active=pokemon(900, 2, hp=220))
    options = [{"type": 3, "area": 3, "index": index} for index in range(6)]
    obs = effect_obs(
        me,
        opponent,
        options,
        effect_id=SACRED_ASH,
        context=9,
        min_count=1,
        max_count=5,
    )
    chosen = MODULE.agent(obs)
    chosen_ids = [discard[index] for index in chosen]
    self.assertEqual(chosen_ids, [ABRA, KADABRA, ALAKAZAM, ABRA, KADABRA])
```

断言分别为：先选 Bench 进化；无 ready Bench 时不把 Dudunsparce 作为换位目标；六张
`Abra/Kadabra/Alakazam` 候选选五张时顺序为一条完整线后开始第二条线。

- [ ] **Step 3: 写事实层反例测试**

覆盖 Dudunsparce 固定数量为 2、未知 Prize 不等于 known deck、旧 KO 不重复触发、
`stadiumPlayed` 独立预算和没有固定手牌上限。

- [ ] **Step 4: 运行测试并确认当前实现失败**

Run: `python3 -m unittest tests.test_alakazam_v8_agent -v`

Expected: 至少三个策略反例 FAIL；失败原因分别指向提前攻击、过宽 Dudunsparce 路线和
Sacred Ash 选择顺序。

Checkpoint: 只新增/整理测试，不修改生产策略。

---

### Task 2: 建立 Cards、Model、Memory 与 Facts

**Files:**
- Create: `work/alakazam_v8_current/strategy/{__init__,cards,model,memory,facts}.py`
- Create: `tests/test_alakazam_v8_facts.py`

**Interfaces:**
- Produces: `DeckSpec.from_deck(deck: list[int]) -> DeckSpec`。
- Produces: `GameMemory.sync(current, player, logs) -> None` 和 `reset() -> None`。
- Produces: `build_turn_facts(obs, memory, deck_spec) -> TurnFacts`。

- [ ] **Step 1: 写 DeckSpec 与资源边界失败测试**

```python
self.assertEqual(spec.count(DUDUNSPARCE), 2)
self.assertNotIn(WONDROUS_PATCH, spec.card_ids)
self.assertEqual(facts.resources.known_in_deck[KADABRA], 0)
self.assertGreater(facts.resources.unknown_deck_or_prize[KADABRA], 0)
```

- [ ] **Step 2: 实现领域类型**

`model.py` 定义 `PlanKind`、`RouteCertainty`、`ActionKind`、`DecisionPhase`、
`PokemonRef`、`PlayerFacts`、`TurnBudget`、`ResourceLedger`、`TurnFacts`、`AttackRoute`、
`HandoffRoute`、`PlanGoal`、`TurnPlan`、`SemanticOption`、`ActionIntent` 和 `Decision`。

所有事实/计划 dataclass 使用 `frozen=True`；可变计数在构建时转成只读 mapping 或 tuple。

- [ ] **Step 3: 实现 DeckSpec**

从实际 `deck.csv` 读取的 60 张 ID 生成计数，不在 memory 中重复写死数量。V8 卡牌常量
只包含实际卡表；对手保护性 Energy 等跨卡组事实单独定义。

- [ ] **Step 4: 实现 GameMemory**

记录 turn key、turn-start serial、进化 serial、Supporter/Energy/Retreat/Stadium、攻击、
Item Lock、effect session 和上一对手回合 KO。KO 优先使用 `TURN_START` 日志边界；简化
fixture 没有 turn marker 时，同一事件 signature 只能触发一次。

- [ ] **Step 5: 实现 Facts adapter**

将 raw observation 转换为 `PokemonRef` 和 `PlayerFacts`；分别维护可见 zone、已知 Prize、
已知 Deck 与 `unknown_deck_or_prize`，不把未知资源声明为确定可搜索。

- [ ] **Step 6: 运行事实层测试**

Run: `python3 -m unittest tests.test_alakazam_v8_facts -v`

Expected: PASS，且覆盖两局连续 reset、同回合进化和 Stadium 预算。

---

### Task 3: 建立 SemanticOption 边界

**Files:**
- Create: `work/alakazam_v8_current/strategy/options.py`
- Modify: `tests/test_alakazam_v8_facts.py`

**Interfaces:**
- Produces: `decode_options(select: dict, facts: TurnFacts) -> Sequence[SemanticOption]`。
- Produces: `match_intent(intent, options) -> SemanticOption | None`。
- Produces: `required_effect_fallback(select, options) -> Sequence[int]`。

- [ ] **Step 1: 写协议解析测试**

覆盖 PLAY hand index、EVOLVE `inPlayArea/inPlayIndex`、ATTACH、ABILITY、Hammer attached
energy、deck/discard card selection、ATTACK 和 END。

- [ ] **Step 2: 实现 option decoder**

所有裸 `type/context/area` 整数只存在于 `options.py` 的协议适配层。每个
`SemanticOption` 保存原始合法 index、动作类型、card ID、source/target key、energy ID、
attack ID 和 effect ID。

- [ ] **Step 3: 实现确定性 fallback**

主动作 fallback 只选择合法 END；永久禁止攻击不作为 fallback。required effect 解析失败
时返回满足 min/max 的确定性合法 index，并产生 `fallback.effect_required` reason。

- [ ] **Step 4: 运行 adapter 测试**

Run: `python3 -m unittest tests.test_alakazam_v8_facts -v`

Expected: PASS；测试不再直接调用旧 `_option_card_id()`。

---

### Task 4: 建立 Routes、Profiles 与 TurnPlan

**Files:**
- Create: `work/alakazam_v8_current/strategy/{routes,profiles,planner}.py`
- Create: `tests/test_alakazam_v8_planner.py`

**Interfaces:**
- Produces: `analyze_routes(facts, options) -> RouteAnalysis`。
- Produces: `build_turn_plan(facts, routes, profile, previous=None) -> TurnPlan`。
- Produces: `BASELINE_PROFILE: StrategyProfile` 和
  `with_variant(profile: StrategyProfile, **changes: object) -> StrategyProfile`。

- [ ] **Step 1: 写路线失败测试**

覆盖 Active Alakazam、Active Kadabra 自然进化、Active Abra Rare Candy、Active
Dunsparce + ready Bench Alakazam、无 ready Bench、CONFIRMED/POSSIBLE/BLOCKED 和最后
Prize。

- [ ] **Step 2: 实现 StrategyProfile**

使用枚举字段定义 `attack_preparation`、`dudunsparce`、`fezandipiti`、`hammer`、
`recovery`、`stadium`、`deck_safety`。候选用 `dataclasses.replace()` 生成；不接收任意
dict 或动态执行策略字符串。

- [ ] **Step 3: 实现路线分析**

路线对象必须列出 attacker、进化、Psychic、换位、目标、伤害、Prize 和 certainty。
未知 Deck/Prize 只能形成 POSSIBLE 搜索路线，不能计为 ready handoff。

- [ ] **Step 4: 实现 planner**

生成 `VICTORY`、`ATTACK` 或 `BUILD_SURVIVE`；分配唯一 Supporter 和 Energy purpose；
建立 must goals；攻击提交条件返回未满足 goal 的稳定 reason code。

- [ ] **Step 5: 运行 planner 测试**

Run: `python3 -m unittest tests.test_alakazam_v8_planner -v`

Expected: PASS；Active Dunsparce 无 ready Bench 时没有换位 goal，Bench Kadabra 进化是
非终局攻击前 must goal。

---

### Task 5: 实现阶段式主动作 Policies

**Files:**
- Create: `work/alakazam_v8_current/strategy/policies/{__init__,setup,continuity,resources,control,commit}.py`
- Create: `work/alakazam_v8_current/strategy/orchestrator.py`
- Modify: `tests/test_alakazam_v8_agent.py`

**Interfaces:**
- Produces: 每个 policy 的
  `propose(facts, plan, routes, options, profile) -> Sequence[ActionIntent]`。
- Produces: `StrategyOrchestrator.choose_main(obs) -> Decision`。

- [ ] **Step 1: 写阶段冲突测试**

覆盖：硬禁止优先；当前攻击路线优先于可选建设；非终局 must goal 优先攻击；唯一
Supporter 由 plan 决定；出牌后重新计算手牌伤害；满足 commit condition 后攻击。

- [ ] **Step 2: 实现 setup 与 continuity policy**

实现 Poffin/Telepath/手牌 Basic 建场、进化时钟、主攻击 Abra 保护、Bench Kadabra
进化、ready attacker 接班和 Dudunsparce 换位。Dudunsparce 搜索 intent 必须引用
confirmed handoff route ID。

- [ ] **Step 3: 实现 resources policy**

实现 Fezandipiti/Kadabra/Alakazam/Dudunsparce Ability、牌库三段预算、Energy purpose、
Lana/Night Stretcher/Sacred Ash 使用条件和 Poké Pad 延迟。

- [ ] **Step 4: 实现 control policy**

实现 Hammer、Nighttime Mine、Boss 和 Xerosic。非终局 Hammer/Stadium 是 must action；
终局只执行维持或提高终局确定性的控制动作。

- [ ] **Step 5: 实现 commit policy 与 orchestrator**

按 HARD_RULES、VICTORY、CURRENT_ATTACKER、MUST_PREPARE、RESOURCE_ALLOCATION、
OPTIONAL_PREPARE、ATTACK_COMMIT、END 顺序匹配第一个合法 intent。每次调用重新构建 facts
和 plan，返回带 `rule_id` 的 `Decision`。

- [ ] **Step 6: 运行主动作测试**

Run: `python3 -m unittest tests.test_alakazam_v8_agent -v`

Expected: PASS；Task 1 的三个红测转绿，并且没有数字 score API。

---

### Task 6: 实现 Effect Selectors

**Files:**
- Create: `work/alakazam_v8_current/strategy/effects/{__init__,dispatcher,search,recovery,targeting}.py`
- Create: `tests/test_alakazam_v8_effects.py`

**Interfaces:**
- Produces: `select_effect(obs, facts, plan, memory, profile) -> Decision`。
- Consumes: `TurnPlan.supporter_purpose`、`energy_purpose` 和 route ID。

- [ ] **Step 1: 写 effect 红测**

按 card ID 覆盖 Dawn、Hilda、Poffin、Telepath、Poké Pad、Lana、Night Stretcher、Sacred
Ash、Hammer、Boss、Rare Candy、Dudunsparce Yes/No 和 COUNT。

- [ ] **Step 2: 实现 search selector**

Dawn 三阶段选择服务同一 plan；Active Dunsparce 只有 confirmed handoff 时找
Dudunsparce；首回合 Poké Pad 不浪费在冗余 Basic；Telepath 只找 Basic Psychic。

- [ ] **Step 3: 实现 recovery selector**

Lana Pokémon-first 并填满有价值名额；Sacred Ash 先一条完整 Abra 线、再第二条、最后
Dunsparce 线；Rare Candy 只改变 Kadabra 必要度，不允许孤立 Stage 2。

- [ ] **Step 4: 实现 targeting 与 dispatcher**

Hammer 四级顺序、Boss 确定 KO、己方换位 ready Alakazam、required/optional fallback。
effect session 使用 `(serial, effect_id)`，新局清空。

- [ ] **Step 5: 运行 effect 测试**

Run: `python3 -m unittest tests.test_alakazam_v8_effects -v`

Expected: PASS；选择数量和 card IDs 均精确匹配。

---

### Task 7: 切换 main.py 并删除 Legacy 策略

**Files:**
- Modify: `work/alakazam_v8_current/main.py`
- Modify: `tests/test_alakazam_v8_luna_deck_opt_strategy.py`

**Interfaces:**
- Produces: `DECK: list[int]`、`agent(obs_dict) -> list[int]`。
- Consumes: `StrategyOrchestrator` 和 `BASELINE_PROFILE`。

- [ ] **Step 1: 写入口契约测试**

验证 deck request 返回 60 张卡、新局 reset、main/effect dispatch、只返回合法 index 和跨局
effect serial 不污染。

- [ ] **Step 2: 将 main.py 缩减为入口**

```python
import sys
from pathlib import Path

ROOT = Path(globals().get("__file__", "/kaggle_simulations/agent/main.py")).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy.cards import load_deck
from strategy.orchestrator import StrategyOrchestrator
from strategy.profiles import BASELINE_PROFILE

DECK = load_deck()
_ORCHESTRATOR = StrategyOrchestrator(DECK, BASELINE_PROFILE)

def agent(obs_dict: dict) -> list[int]:
    if obs_dict.get("select") is None:
        _ORCHESTRATOR.reset()
        return list(DECK)
    return list(_ORCHESTRATOR.choose(obs_dict).option_indexes)
```

- [ ] **Step 3: 删除旧实现**

删除 TurnMemory 旧类、全局 score、`HAND_DRAW_STOP`、V7 移除卡牌、opponent-ID 分支、重复
route helper 和 legacy fallback。最终 `main.py` 不包含卡牌策略 if/elif。

- [ ] **Step 4: 运行全部 V8 测试**

Run: `python3 -m unittest tests.test_alakazam_v8_facts tests.test_alakazam_v8_planner tests.test_alakazam_v8_effects tests.test_alakazam_v8_agent tests.test_alakazam_v8_luna_deck_opt_strategy -v`

Expected: 全部 PASS，0 failures/errors。

---

### Task 8: 多模块 Submission 打包

**Files:**
- Modify: `scripts/package_submission.sh`
- Modify: `tests/test_package_submission.py`

**Interfaces:**
- Produces: `submission/dist/<name>.tar.gz`，顶层包含 `main.py`、`deck.csv`、`strategy/`、`cg/`。

- [ ] **Step 1: 写失败的归档结构测试**

临时候选增加 `strategy/__init__.py` 和 `strategy/policy.py`，断言归档包含它们，且没有候选
父目录、`__pycache__` 或 `*.pyc`。

- [ ] **Step 2: 修改打包脚本**

新候选存在 `strategy/` 时把它加入 tar；历史单文件 submission 没有该目录时仍保持原有
三项归档。脚本使用显式数组构造输入：

```bash
archive_entries=(main.py deck.csv cg)
if [[ -d strategy ]]; then
  archive_entries+=(strategy)
fi
tar --exclude='__pycache__' --exclude='*.pyc' \
  -czf "$out_file" "${archive_entries[@]}"
```

- [ ] **Step 3: 增加解包导入测试**

把归档解到临时 `kaggle_simulations/agent`，分别以该目录为 cwd 导入、以及从其它 cwd 对
`main.py` 执行 raw-exec，断言 `len(DECK) == 60` 且不访问网络或外部 site-packages。

- [ ] **Step 4: 运行打包测试并生成真实归档**

Run: `python3 -m unittest tests.test_package_submission -v`

Run: `bash scripts/package_submission.sh alakazam_v8_current`

Run: `tar -tzf submission/dist/alakazam_v8_current.tar.gz`

Expected: 测试 PASS；归档顶层只有必要入口/运行时目录，包含完整 `strategy/`。

---

### Task 9: 全量验证与 Review

**Files:**
- Modify only if verification reveals a scoped defect in files above.

**Interfaces:**
- Produces: correctness、语法、资产、入口和归档的最新证据。

- [ ] **Step 1: 运行策略与仓库相关测试**

```bash
python3 -m unittest \
  tests.test_alakazam_v8_facts \
  tests.test_alakazam_v8_planner \
  tests.test_alakazam_v8_effects \
  tests.test_alakazam_v8_agent \
  tests.test_alakazam_v8_luna_deck_opt_strategy \
  tests.test_package_submission -v
python3 scripts/check_assets.py
```

Expected: 0 failures/errors；资产检查退出码 0。

- [ ] **Step 2: 运行语法与格式验证**

```bash
python3 -m compileall work/alakazam_v8_current tests
git diff --check
```

Expected: compileall 成功，diff check 无输出。

- [ ] **Step 3: 验证真实归档与 raw-exec**

重新打包，检查 `tar -tzf`，解包到临时目录并从目录内执行 `main.py` deck request。禁止把
临时输出写入仓库。

- [ ] **Step 4: 规格逐项自审**

逐项核对设计文档第 14 节完成标准；确认无全局 score、无旧 V7 卡牌、无固定手牌上限，
并列出尚未进行的 Kaggle focused/full evaluation，不把重构测试通过声称为胜率提升。

Checkpoint: 报告修改文件、测试数量、归档内容和剩余外部评测风险；不自动 commit/push。
