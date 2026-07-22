# Alakazam V8 Agent 多模块重构设计

## 1. 文档定位

本文档是 `work/docs/REFACTOR_DESIGN.md` 的实现架构补充，定义
`work/alakazam_v8_current/` 下一轮重构的模块边界、决策模型、策略组合方式、迁移顺序
和验收标准。

策略语义的优先级保持不变：

1. 官方规则与模拟器卡牌事实；
2. `docs/reports/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`；
3. `work/docs/REFACTOR_DESIGN.md`；
4. `work/docs/DECK_NOTES.md` 中用户补充的 V8 建议；
5. `work/docs/CURRENT_OBJECTIVES.md` 的 V8C-01 至 V8C-10；
6. 当前 `main.py` 行为和单局实验结果。

如果当前实现或测试与上述高优先级来源冲突，应修正实现或测试，不能反向修改设计以
适应旧代码。

## 2. 目标与非目标

### 2.1 目标

1. 把 3300 行单文件拆成职责明确的多模块源码；顶层 `main.py` 只保留 evaluator 入口、
   `DECK` 暴露和依赖装配。
2. 用显式 `TurnFacts` 和 `TurnPlan` 取代 `_main_action()` 中隐含的布尔变量与全局 tuple
   score。
3. 让攻击成为满足 `attack_commit_condition` 后的不可逆提交，而不是普通候选动作。
4. 让 V8 Deck Notes 中的攻击准备、Dudunsparce 换位、Fezandipiti、Hammer、恢复、
   Poké Pad、Dawn 和 Nighttime Mine 语义拥有独立实现与独立测试。
5. 允许 AutoIteration 用命名策略 profile 组合不同策略，不再复制并修改整份
   `main.py`。
6. 每个决策输出稳定的 `rule_id`、`plan_kind`、`purpose` 和未选择原因，使 focused
   evaluation 可以直接统计策略行为。
7. 让 `scripts/package_submission.sh` 把 `strategy/` 作为顶层目录一并打包，并验证
   Kaggle 解包到 `/kaggle_simulations/agent/` 后可以离线导入。

### 2.2 非目标

- 本轮不修改 60 张 V8 卡表。
- 本轮不改变 `agent(obs_dict) -> list[int]` 和 `DECK` 的 evaluator 契约。
- 本轮不实现单文件合并器；Kaggle submission 直接携带 `strategy/` Python 包。
- 本轮不引入概率搜索、蒙特卡洛树搜索、通用行为树或 opponent ID 脚本。
- 本轮不把所有卡牌做成一张卡一个类；模块按策略责任划分，不按卡牌数量机械拆分。
- 本轮不以本地单局胜率替代 correctness gate 和固定 control/candidate 对比。
- 本轮不增加需要 Kaggle 现场联网安装的第三方依赖。

## 3. 当前实现审计结论

### 3.1 已经正确覆盖的行为

- `Trading Places` 已从主动作候选中永久过滤。
- 非终局场景下，Fezandipiti ex 的 post-KO 抽牌可以先于 `Powerful Hand`。
- Enhanced Hammer 主动作可以先于普通攻击，目标顺序已区分保护性特殊能量和
  Active/Bench。
- 首回合已有两只 Abra 和一只 Dunsparce 时可以保留 Poké Pad。
- Sacred Ash 已能使用最多五个合法选择名额。
- Nighttime Mine 已能先于普通攻击和铺场动作使用。

这些行为必须通过 characterization tests 保留，除非其触发条件本身与设计冲突。

### 3.2 必须修正的行为缺口

1. Active Alakazam 可以非终局 KO 时，未充能 Bench Kadabra 的合法
   `Kadabra -> Alakazam -> Psychic Draw` 会被跳过。
2. Poké Pad 和 Dawn 只因 Active 是 Dunsparce 就寻找 Dudunsparce，没有要求 Bench
   上存在本回合可接班的攻击者。
3. Sacred Ash 使用全局阶段排序，六张候选为两组 Abra/Kadabra/Alakazam 时会选择
   `Abra, Abra, Kadabra, Kadabra, Alakazam`，没有优先恢复完整攻击链。
4. Lana's Aid 同时缺 Pokémon 和 Basic Psychic、但弃牌区只有一只合适 Pokémon 时，
   可能只选择 Pokémon，浪费其余合法选择名额。
5. `HAND_DRAW_STOP = 20` 把经验手牌数写成通用硬门槛，与“没有通用手牌上限”的规则
   结论冲突。
6. Fezandipiti 的 KO 检测扫描日志但没有明确限制到对手上一回合，存在旧 KO 事件被重复
   解释的风险。

### 3.3 必须修正的事实层缺口

- V8 实际只有两张 Dudunsparce，当前资源总量仍写成三张。
- 当前代码仍包含 Psyduck、Wondrous Patch 和 Battle Cage 的 V7 逻辑。
- `resource_unknown` 混合未知 Prize 与未知 Deck，不能作为“目标卡确定仍在牌库”的证据。
- `TurnMemory` 没有维护 `stadiumPlayed`，与规则和 `TurnPlan` 所需预算不完整。
- 主动作、效果上下文和 option 类型大量使用裸整数，策略代码同时承担协议解析责任。

### 3.4 测试缺口

当前 17 个 V8 专项测试全部通过，但主要验证单步 option index。重构前必须补充：

- 完整动作链与每一步重规划；
- `TurnPlan` 的当前攻击者、接班者和 attack commit condition；
- 同一局跨 observation 的资源、进化时钟和 KO 事件；
- 同一场景在不同 `StrategyProfile` 下只改变目标策略维度；
- effect 的选择数量、选择身份和顺序，而不只检查返回列表长度。

## 4. 方案选择

### 4.1 不采用：只拆文件并保留全局 score

把 helper 和 score 分支搬到多个文件不能解决策略冲突。攻击、Supporter、恢复和过牌仍
通过不可解释的数字竞争，新策略仍可能无意覆盖旧策略。

### 4.2 采用：显式 TurnPlan 与可组合策略组件

事实、路线、计划、动作匹配和效果选择使用稳定接口连接。策略组件只负责自己的决策
维度，冲突通过计划阶段和资源所有权解决，不使用跨领域的全局数字分数。

### 4.3 暂不采用：行为树或多步状态搜索

当前存在隐藏 Prize/Deck 边界，且 observation fixture 和完整动作链测试仍不足。通用
搜索会放大状态模型误差，也会增加 AutoIteration 的归因难度；应在确定性规划器稳定后
再单独评估。

## 5. 目标目录与职责

```text
work/alakazam_v8_current/
├── main.py
├── deck.csv
├── cg/
└── strategy/
    ├── __init__.py
    ├── cards.py
    ├── model.py
    ├── memory.py
    ├── facts.py
    ├── options.py
    ├── routes.py
    ├── planner.py
    ├── orchestrator.py
    ├── profiles.py
    ├── policies/
    │   ├── __init__.py
    │   ├── setup.py
    │   ├── continuity.py
    │   ├── resources.py
    │   ├── control.py
    │   └── commit.py
    └── effects/
        ├── __init__.py
        ├── dispatcher.py
        ├── search.py
        ├── recovery.py
        └── targeting.py
```

| 模块 | 单一职责 |
|---|---|
| `main.py` | 暴露 `DECK` 和 `agent()`，识别新局并调用 orchestrator |
| `cards.py` | V8 卡牌 ID、类型、卡组数量和卡牌角色；数量从 `deck.csv` 校验 |
| `model.py` | 不依赖 observation 字典的不可变领域类型 |
| `memory.py` | 跨 observation 的进场、进化、回合预算、KO、Item Lock 和 effect session |
| `facts.py` | 把 observation 与 memory 转换为稳定 `TurnFacts` |
| `options.py` | 把原始 option 解码为 `SemanticOption`，并把决定映射回合法 index |
| `routes.py` | 计算终局、当前攻击、接班、换位、恢复和牌库路线 |
| `planner.py` | 根据事实与 profile 生成 `TurnPlan`，动作后重算剩余目标 |
| `orchestrator.py` | 串联 Facts、Plan、Policy、Effects 与 Memory，不包含卡牌特例 |
| `profiles.py` | 定义命名策略 profile 和可组合策略维度 |
| `policies/` | 按建设、接班、资源、控制和攻击提交产生动作意图 |
| `effects/` | 处理卡牌搜索、恢复、多选和目标排序，不重新制定 TurnPlan |

`policies/` 不按一张卡一个文件拆分。会共同影响同一资源或同一路线的卡牌放在同一
策略模块中，避免模块数量失控。

顶层 `main.py` 使用 `from strategy.orchestrator import ...` 形式导入。只有
`strategy/` 包内部使用 `from .model import ...` 等相对导入，顶层入口不使用
`from .strategy ...`。

## 6. 核心领域模型

### 6.1 TurnFacts

`TurnFacts` 是一次 observation 的只读事实快照，至少包含：

- 当前回合、己方回合数、先后手；
- 双方 Active、Bench、手牌数、牌库数、Prize 数、弃牌区、能量、伤害和 Stadium；
- 每只己方 Pokémon 的稳定 serial、进场回合、进化回合和本回合可进化状态；
- Supporter、手填 Energy、Retreat、Stadium 和受限 Ability 的独立预算；
- Item Lock、上一对手回合己方 KO、已使用 effect 和攻击是否已经提交；
- V8 固定资源在可见区域中的数量；
- `known_in_deck`、`known_in_prize`、`unknown_deck_or_prize` 的明确边界。

Facts 不包含“是否值得使用”或“优先级”。模拟器提供的合法 option 是合法性最终权威，
Facts 只补充策略需要的时点和资源语义。

### 6.2 Route

路线是可验证的资源链，不是单个布尔值：

```text
AttackRoute
  attacker -> evolution requirement -> energy requirement -> switch requirement
  -> expected damage -> target -> prize value -> certainty

HandoffRoute
  successor -> setup timing -> evolution path -> energy path -> search/recovery entry
  -> certainty
```

路线确定性分为：

- `CONFIRMED`：资源已经可见，合法 option 或时点可以验证；
- `POSSIBLE`：目标可能在未知 Deck/Prize 中，需要搜索动作后才能确认；
- `BLOCKED`：时点、资源、Item Lock、Bench 空间或牌库预算使路线不可达。

`POSSIBLE` 不能冒充 ready attacker，但可以成为使用 Poké Pad、Dawn 或 Hilda 探索路线
的理由。

### 6.3 TurnPlan

`TurnPlan` 至少包含：

- `kind`: `VICTORY`、`ATTACK` 或 `BUILD_SURVIVE`；
- `victory_route`；
- `primary_attacker`；
- `attack_target`；
- `handoff_route`；
- `must_goals`：攻击前必须完成的状态目标；
- `supporter_purpose`：本回合唯一 Supporter 服务的目标；
- `energy_purpose`：本回合手填 Energy 服务的目标；
- `deck_budget`；
- `attack_commit_condition`；
- `revision` 和稳定 `plan_id`。

计划保存目标而不是一串固定 option index。抽牌、搜索或进化后，planner 使用新事实重算
达到相同目标的下一步；如果原路线已经不可达，则显式变更 plan 并记录原因。

### 6.4 SemanticOption、ActionIntent 与 Decision

`SemanticOption` 对原始 option 提供卡牌、来源、目标、能量、攻击和 effect 上下文的
语义视图。策略模块不能直接依赖 option 数组下标。

`ActionIntent` 描述某项计划希望完成的动作，至少包含：

- `rule_id`；
- `phase`；
- `action_kind`；
- `card_id`、source/target 条件；
- `purpose`；
- `required_before_attack`；
- `satisfied_when`。

`Decision` 保存最终合法 option index，以及它满足的 plan、rule 和 purpose。评测可以
直接使用这些稳定字段，不需要从 score 或注释推断行为。

## 7. 决策阶段与冲突解决

主动作不使用全局分数。每次 observation 按以下阶段选择第一个与当前计划一致的合法
动作：

1. **硬规则过滤**：非法时点、Item Lock、重复 Supporter/附能/Retreat/Stadium、
   `Trading Places`、空 Bench 的危险 `Run Away Draw`；
2. **确定胜利或避免立即失败**：建立不会被后续动作破坏的终局路线；
3. **确定当前攻击者**：完成必要进化、Psychic、Retreat 或 Dudunsparce 换位；
4. **完成 must goals**：Fezandipiti、合法进化 Ability、接班打手、Hammer、Stadium、
   必要搜索与恢复；
5. **分配一次性资源**：统一选择 Supporter、手填 Energy、Retreat 和 Stadium 的目的；
6. **可选建设动作**：只执行能改善当前/下一攻击路线且符合牌库预算的动作；
7. **攻击提交检查**：重新计算伤害、Prize、未完成目标和接班路线；
8. **攻击或 END**。

同一阶段的冲突不靠数字决定：

- Supporter 由 planner 在 Boss、Lana、Dawn、Hilda、Xerosic 中选择一个 purpose；
- Energy 由当前攻击、接班攻击、Dudunsparce 引擎三个互斥目的分配；
- effect selector 只执行已经选定的 purpose；
- 终局计划跳过只有未来价值的准备，但不能跳过完成终局所需的 Boss、换位、能量或
  解保护动作；
- Hammer 和 Nighttime Mine 在不破坏确定终局的前提下视为立即动作。非终局有合法
  Hammer 目标或合法 Nighttime Mine 时，必须在攻击前使用。

## 8. V8 策略的明确落点

### 8.1 攻击前资源与接班

- 非终局攻击前，所有合法且能改善当前攻击或确认接班路线的进化和 Ability 都进入
  `must_goals`。
- Bench Kadabra 即使未充能，只要能合法进化为 Alakazam 并抽牌、建立下一只 Stage 2，
  也不能被当前 KO 跳过。
- “能 KO”只证明攻击可行；只有最后 Prize 或其它确定胜利才能跳过纯未来准备。

### 8.2 Dudunsparce

- `Trading Places` 永远不产生 ActionIntent。
- `Dudunsparce -> Run Away Draw` 作为主动换位路线时，必须存在 Bench ready attacker，
  或本回合可以确定完成 Bench attacker 的进化和 Psychic。
- Poké Pad、Dawn 或 Hilda 只有在上述路线成立时才把 Dudunsparce 作为换位目标。
- 作为普通过牌使用时独立评估牌库净变化、Bench 接班和手牌价值，不能借用“换位”理由。

### 8.3 Fezandipiti ex

- memory 只把对手紧邻上一回合发生的己方 KO 标记为触发事实。
- 非终局且 Ability 合法时，`Flip the Script` 是 must goal，不因已有普通 Bench 或当前
  可以攻击而跳过。
- 确定终局不需要额外三张牌时，不为未来价值延迟攻击。

### 8.4 Enhanced Hammer 与 Nighttime Mine

- 非终局时，有合法 Special Energy 目标就使用 Hammer。
- Hammer 目标顺序为 Active 保护性、Bench 保护性、Active 其它、Bench 其它；同级使用
  稳定 option 顺序。
- 非终局时，合法 Nighttime Mine 立即使用；它可以覆盖对手 Stadium，并且不会限制
  V8 自身的非 Tera 攻击线。
- 两者都必须重新计算出牌后的 Alakazam 手牌伤害，不能沿用旧的 KO 判断。

### 8.5 恢复与检索

- Lana's Aid 默认 Pokémon-first，并在需要接班时尽量使用三个选择名额；Basic Psychic
  只在它补齐当前确定攻击或唯一现实接班路线时加入。
- Sacred Ash 尽量选五张。选择目标按“先形成一条完整 Abra 线，再形成下一条线，最后
  补 Dunsparce/其它资源”组织；Rare Candy 充足时可以降低 Kadabra，但不能产生只有
  Stage 2 没有底座的恢复组合。
- 首回合已有两只 Abra 和一只 Dunsparce，且没有立即换位或进化用途时保留 Poké Pad。
- Dawn 每个 Basic/Stage 1/Stage 2 选择都服务同一 TurnPlan；Active Dunsparce 只有在
  ready attacker 接班路线存在时才提高 Dudunsparce 优先级。

### 8.6 牌库预算

- 删除固定手牌上限。
- `> 15`、`11-15`、`<= 10` 保留为牌库策略区间，而不是规则限制。
- 每个抽牌/检索动作声明 `deck_cost`，Dudunsparce 使用抽三减洗回数量的净变化。
- `<= 10` 时只有确定终局或避免立即失败所需的 deck 消耗可以越过保护线。

## 9. AutoIteration 组合接口

`StrategyProfile` 是不可变配置，按决策维度使用枚举或命名策略，不使用任意字典和大量
互相依赖的布尔开关。基线 profile 至少包含：

```text
attack_preparation = ALL_REAL_VALUE
dudunsparce = HANDOFF_WITH_READY_ATTACKER
fezandipiti = USE_AFTER_CONFIRMED_KO
hammer = ALWAYS_IF_LEGAL_NONTERMINAL
recovery = POKEMON_FIRST
stadium = PLAY_IMMEDIATELY_NONTERMINAL
deck_safety = V8_THRESHOLDS
```

AutoIteration 使用 `dataclasses.replace(BASELINE_PROFILE, dimension=new_value)` 生成候选。
每轮只改变一个主要策略维度；多个已独立验证的策略可以组合成命名 profile。profile 名、
各维度值和代码版本必须进入评测记录。

策略组件输出稳定 reason code，例如：

- `continuity.evolve_bench_kadabra`；
- `handoff.search_dudunsparce`；
- `recovery.lana_pokemon_first`；
- `control.hammer_active_protective`；
- `commit.blocked_pending_must_goal`；
- `commit.powerful_hand_terminal`。

## 10. 错误处理和合法性边界

- 永远只返回模拟器提供的 option index，不构造不存在的动作。
- main selection 没有匹配计划动作时，优先返回合法 END；如果只剩永久禁止攻击，则返回
  空选择并记录 `fallback.forbidden_only`。
- required effect 若 `minCount > 0` 但没有 options，继续抛出明确异常；这属于 simulator
  契约错误。
- required effect 有 options 但策略解析失败时，选择满足 min/max 的确定性合法 fallback，
  同时记录 `fallback.effect_required`，不能静默把 fallback 当作正常策略。
- optional effect 在没有策略价值时选择 NO 或空选择。
- 新局通过 deck request 重置 memory 和 effect session；旧 effect serial 不得跨局泄漏。

## 11. Submission 打包契约

Kaggle submission 使用 `.tar.gz`，解包根目录为 `/kaggle_simulations/agent/`。归档顶层
必须至少包含：

```text
main.py
deck.csv
strategy/
cg/
```

`scripts/package_submission.sh` 必须从完整候选目录打包 `main.py`、`deck.csv`、`cg/` 和
`strategy/`，不能遗漏策略包，也不能在归档中额外包一层 `work/alakazam_v8_current/`。
对没有 `strategy/` 的历史单文件 submission，脚本继续只打包原有三项；多模块目录是
新候选的可选附加运行时，不应使历史归档失效。

打包测试必须验证：

- `tar -tzf submission/dist/alakazam_v8_current.tar.gz` 包含顶层 `strategy/__init__.py`
  和所有必要策略模块；
- 归档不包含 `__pycache__`、`*.pyc`、本地 trace、测试文件或开发文档；
- 从候选目录直接 import，以及把归档解到临时 `/kaggle_simulations/agent` 等价目录后的
  raw-exec/import 都能读取 `DECK` 并调用入口；
- 运行时只依赖 Python 标准库、归档内源码和 `cg/`，不尝试联网安装依赖。

## 12. 测试设计

### 12.1 纯单元测试

- `facts`：区域、进化时钟、预算、KO、Item Lock、Stadium 和未知 Prize/Deck 边界；
- `routes`：当前攻击、接班、Dudunsparce 换位、恢复和终局；
- `planner`：三类 plan、Supporter/energy purpose 和 attack commit condition；
- `options`：裸 option 到语义动作及合法 index 的双向映射；
- `effects`：按 card ID 验证选择数量、卡牌身份与目标顺序；
- `profiles`：候选 profile 只改变声明的策略维度。

### 12.2 决策链测试

每个 case 提供一组连续 observation，断言：

1. 初始 `TurnPlan`；
2. 每一步 `Decision.rule_id`；
3. 动作后 plan revision；
4. 最终攻击或 END；
5. 整条链中没有未完成 must goal、非法动作或目标漂移。

### 12.3 V8 correctness cases

- Active Alakazam 非终局 KO 前进化 Bench Kadabra；
- Trading Places + END，以及只有 Trading Places；
- Active Dunsparce 有/没有 ready Bench attacker 的正反换位路线；
- Fezandipiti 紧邻上一回合 KO、旧 KO、终局三个边界；
- Hammer 四级目标顺序和出牌后伤害重算；
- Lana 一至三只 Pokémon、必要 Energy 和 Supporter 机会成本；
- Sacred Ash 两条完整 Abra 线、Rare Candy 充足/不足；
- 首回合 Poké Pad 保留与立即 Dudunsparce 换位搜索；
- Dawn 三阶段选择服务同一 plan；
- Nighttime Mine 非终局立即使用、终局不破坏；
- 牌库 16、15、11、10 张和 Dudunsparce 净洗回边界。

### 12.4 回归与集成

- 运行全部策略测试和 `python3 scripts/check_assets.py`；
- 运行 `py_compile`/`compileall`；
- 打包并用 `tar -tzf` 验证 `strategy/` 位于归档顶层；
- 在归档解包后的临时目录执行 raw-exec/import，验证 `DECK` 与 `agent` 入口；
- 本地对局只用于合法性和崩溃 smoke test，输出写入 `/tmp`；
- promotion 仍使用固定 focused evaluation、full control/candidate 和 Kaggle Episode/replay。

## 13. 渐进迁移

1. 先把本次 review 的反例写成失败测试，并为当前已确认行为补 characterization tests。
2. 建立 `model.py`、`cards.py`、`memory.py`、`facts.py` 和 `options.py`，保持外部行为不变。
3. 建立纯 `routes.py` 与 `planner.py`，先让测试直接验证 TurnPlan，不切换生产入口。
4. 建立 policies、effects 和 orchestrator，以 `V8_REFACTOR_PROFILE` 完整覆盖主动作与效果
   选择。
5. 把 `main.py` 切换到 orchestrator；迁移期间允许显式 legacy fallback，最终验收前必须
   删除 fallback。
6. 修正 V8C-01 至 V8C-10 的行为和错误测试，删除全局 score、手牌硬上限、旧卡逻辑与
   重复 helper。
7. 修改 `scripts/package_submission.sh` 和打包测试，把 `strategy/` 纳入归档并完成解包后
   的离线导入验证。
8. 通过全量 correctness gate 后再运行 focused/full comparison；重构本身不因测试通过
   自动声明胜率改善。

每个迁移任务必须有可单独评审的测试结果。当前 dirty worktree 中与 evaluation framework
相关的改动不属于本重构范围，不修改、不删除，也不纳入本重构提交。

## 14. 完成标准

- `main.py` 只包含 evaluator 入口与装配，不包含卡牌策略分支。
- Facts、Plan、Actions、Effects 四层可独立测试，策略层不读取 option 数组下标。
- 主动作中不存在跨领域全局数字 score。
- V8 卡组数量从实际 deck 校验，不再出现移除卡牌策略。
- V8C-01 至 V8C-10 的正反 case 和完整动作链全部通过。
- 没有固定通用手牌上限；牌库预算符合 V8 三段保护线。
- 每次决策都有稳定 plan/rule/purpose，可供 AutoIteration 统计。
- baseline profile 与单维候选 profile 可以组合和复现。
- 非法动作、Trading Places、Item Lock 违规、跨局状态污染和 agent error 均为 0。
- submission 归档顶层包含 `main.py`、`deck.csv`、`strategy/` 和 `cg/`，没有额外父目录。
- 解包环境不联网安装依赖即可导入并运行 Agent。
- `check_assets.py`、策略测试、语法检查、归档内容和入口契约验证全部通过。
