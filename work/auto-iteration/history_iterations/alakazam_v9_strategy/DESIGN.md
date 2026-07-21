# Alakazam V9 固定卡表策略优化设计

## 1. 目标与范围

V9 固定使用 `work/alakazam_v9/deck.csv` 的现有 60 张卡表，只修改策略代码、测试、
评测记录和 AutoIteration 可视化产物。设计依据是：

- 用户在 [`../../../docs/ANALYSIS.md`](../../../docs/ANALYSIS.md) 中记录的 Kaggle 实战现象；
- 仓库的官方规则约束与 V6 已确认策略语义；
- 当前 `work/alakazam_v9/main.py` 的真实决策逻辑；
- `auto_iteration_v8_setup_relay` revision 2 的现有评测口径。

本轮不修改卡表、官方 engine、opponent catalog、metric profile、评测分母或 submission
运行时结构，不执行 Kaggle submission。

## 2. 问题归因

### 2.1 Fezandipiti Ability 没有形成稳定行为

当前代码只有在本地日志判断“上一回合己方 Pokémon 被击倒”时，才认为 Fezandipiti
值得出场或使用。Ability 的合法选项已经由 simulator 提供，再叠加一层容易漏判的日志门控，
会让真实合法机会被跳过。同时，旧的准备门控没有把 Fezandipiti 的抽牌明确放入当前回合
攻击路线，导致它即使能扩展手牌，也可能无法穿过主动作排序。

### 2.2 Lana's Aid 没有服务统一的场面目标

当前恢复逻辑围绕若干局部 handoff 条件展开。Lana's Aid、Night Stretcher、Poffin 和
Telepath Energy 分别判断是否值得使用，没有共同消费一个明确的场面缺口。因此，出现
“场上只有一只 Abra 系列、弃牌区有 Abra 和 Basic Psychic、Bench 有空间”的局面时，
Lana's Aid 仍可能被其它 Supporter 或未来准备动作压过。

### 2.3 能量附着没有先保留本回合攻击资源

现有实现对 Active、Bench、Psychic、Telepath 和 Enriching Energy 设置了大量局部分数，
但没有先确定本回合攻击路线并锁定唯一手填 Energy。这会让“给 Bench Kadabra 做未来准备”
压过“给 Active 支付撤退成本并换入已准备好的 Alakazam”，也会在过牌前过早消耗本回合
唯一附能机会。

### 2.4 准备动作和攻击目标缺少统一层级

当前 `preparation_due` 能阻止过早攻击，但会把不同价值的进化、过牌、铺场和接力混为一类。
V9 需要同时满足两个边界：

1. 不破坏本回合攻击的必要准备，应在攻击提交前完成；
2. 如果准备动作会消耗当前攻击所需的 Energy、Supporter、Bench slot 或撤退预算，则当前
   回合攻击优先。

## 3. 方案选择

采用显式 `TurnPlan`，不继续扩张全局分数特例，也不枚举完整回合动作树。

`TurnPlan` 保留在现有 `main.py` 内部，每次 simulator 返回新 observation 时重新计算，
不缓存一条可能被抽牌、检索或进化改变的固定动作序列。它至少表达：

- 当前是否存在合法 Alakazam 攻击；
- 是否能通过进化、附能或撤退在本回合形成攻击；
- 为该路线保留的 Energy、Supporter、Bench slot 和撤退预算；
- Active + Bench 中 Abra 系列的实例数；
- 是否已有 Dunsparce/Dudunsparce；
- 当前允许的过牌与牌库保护状态；
- 本动作属于当前攻击、场面最低配置、机会扩展还是软性准备。

现有合法选项仍是唯一动作来源。`TurnPlan` 只比较 simulator 已提供的 option，不生成不存在
的动作，不绕过规则和 runtime 合法性。

## 4. 决策层级

每次主动作按以下层级解释，高层资源保留约束所有低层动作：

1. **确定的最终取胜路线**：能在本回合拿完最后 Prize 时，保留完成路线需要的攻击、Boss、
   Energy 和撤退预算。
2. **本回合 Alakazam 攻击路线**：优先让 Alakazam 在本回合发起攻击；先完成不会破坏该
   路线的合法收益动作，再提交攻击。
3. **两只 Abra 系列最低配置**：Active + Bench 中 Abra、Kadabra、Alakazam 的独立实例
   总数至少为 2。只要不会牺牲本回合攻击，就先补足该缺口。
4. **一只 Dunsparce 系列最低配置**：满足两只 Abra 系列后，Active + Bench 中至少保留一只
   Dunsparce 或 Dudunsparce。
5. **机会扩展与接力**：合法 Fezandipiti Ability、Lana's Aid、Night Stretcher、进化和安全
   过牌，用于找到当前攻击缺件或准备下一只打手。
6. **软性准备**：最低配置满足后仍可继续铺 Abra/Dunsparce、附能和整理资源，但这些动作
   不得阻挡明确的当前攻击。
7. **攻击提交或结束回合**：攻击仍是不可逆的回合终止动作；低阶段攻击和 Trading Places
   继续作为严格受限的最后手段。

“满足最低配置后仍可继续准备”表示最低配置是硬下限，不是铺场上限。

## 5. 场面不变量

### 5.1 计数规则

- Abra、Kadabra、Alakazam 每个在场实例各计一只 Abra 系列；进化堆叠只按当前顶层实例
  计数一次。
- Dunsparce、Dudunsparce 每个在场实例各计一只 Dunsparce 系列。
- 只统计 Active 和 Bench；手牌、弃牌区和牌库中的卡不是已经建立的场面。

### 5.2 Bench slot 预留

先计算达到“两只 Abra + 一只 Dunsparce”还需要的 Bench slot。Fezandipiti、Shaymin 和
额外冗余 Pokémon 不能占用这些保留位置。若已经有足够空位同时容纳最低配置和
Fezandipiti，合法 Ability 路线可以作为过牌桥梁提前使用。

### 5.3 共用缺口

Poffin、Telepath Energy、Night Stretcher、Lana's Aid、Sacred Ash 和从手牌铺 Basic
必须读取同一个场面缺口：

1. Abra 系列少于 2 时，先寻找或恢复可落场的 Abra；
2. Abra 系列达到 2、Dunsparce 系列为 0 时，再寻找或恢复 Dunsparce；
3. 最低配置满足后，再比较额外 Abra/Dunsparce 和其它 Pokémon 的具体收益。

## 6. 本回合攻击路线

按确定性从高到低检查以下路线：

1. Active Alakazam 已有 Psychic Energy 且 simulator 提供 Powerful Hand；
2. Active Abra/Kadabra 可合法进化为 Alakazam，并保留或取得 Psychic Energy；
3. 给 Active Abra/Kadabra/Alakazam 附 Psychic Energy 后可在本回合攻击；
4. Bench 已有带 Psychic Energy 的 Alakazam，Active 可直接撤退；
5. Bench Alakazam 可通过本回合手填 Psychic Energy 变为 ready，随后 Active 可撤退；
6. Active Fezandipiti/Shaymin 需要一次附能支付撤退，Bench 已有 ready Alakazam；
7. 合法的 Kadabra、Dudunsparce 或 Fezandipiti 过牌可以寻找唯一缺件，再重新计算路线。

路线一旦成立，`TurnPlan` 标记必须保留的 Energy、Supporter、Bench slot 和撤退预算。
任何未来准备动作如果会消耗这些资源，必须排在攻击之后，实际上也就不会在该回合执行。

Boss 只在改变 Prize race 或完成终局时占用 Supporter；Hilda 只在它是当前攻击必要组件时
占用 Supporter。其它情况下 Supporter budget 可以交给 Lana's Aid。

## 7. Fezandipiti 策略

### 7.1 Ability 已经可用

当 simulator 提供合法 `Flip the Script` Ability option 时，以 simulator 合法性为准，
不再要求本地日志再次证明上一回合 KO。除以下情况外，在攻击前使用：

- 抽牌后会进入牌库保护禁区，且不能形成同回合终局闭环；
- 手牌已经达到既有保护上限；
- 当前已有无需额外动作的确定最终取胜攻击。

Ability 即使不能立即让对手 Active 被击倒，也仍然有价值，因为抽到的进化、Energy、
检索或下一段过牌会让本回合攻击路线在下一次 observation 中重新成立。

### 7.2 Fezandipiti 尚在手牌

只有在 TurnMemory 确认 KO 触发对当前回合仍有效时，才把 Fezandipiti 作为 Ability 路线
铺到 Bench。铺场前必须满足以下任一条件：

- 场面最低配置已经满足；
- 使用后仍有足够 Bench slot 完成缺失的 Abra/Dunsparce；
- Ability 是寻找当前攻击或最低配置缺件的唯一可见机会，且不会永久封死保留位置。

需要用真实 trace fixture 验证 KO 日志在 turn boundary 上的识别，不能只依赖人工构造的
单步 observation。

## 8. Lana's Aid 与恢复策略

当 Abra 系列少于 2、弃牌区存在 Abra 且 Bench 有空间时，Lana's Aid 是明确的接力动作。
它的选择顺序为：

1. 可直接落场的 Abra；
2. 为当前或新恢复攻击线所需的 Basic Psychic；
3. 能继续该路线的 Kadabra/Alakazam；
4. 最低 Abra 配置已满足后，才恢复 Dunsparce 系列或其它合法 Pokémon。

Lana's Aid 最多恢复三张，不因为 simulator 的 `minCount` 为 1 就只拿一张；但也不为凑满
数量恢复与当前路线无关的资源。

Supporter 冲突规则：

- Boss 能取得关键 Prize 或终局时，Boss 优先；
- Hilda 是本回合形成 Alakazam 攻击的必要组件时，Hilda 优先；
- 否则 Lana's Aid 可以在攻击前补接力，因为使用 Supporter 本身不会终止攻击窗口；
- Night Stretcher 是 Item，可和其它 Supporter 组合，并在只缺一个资源时优先节省 Lana。

## 9. Energy 策略

### 9.1 当前回合保留

唯一手填 Energy 必须先服务第 6 节的当前攻击路线：

1. 让 Active Abra/Kadabra/Alakazam 具备 Psychic 攻击条件；
2. 给挡在 Active 的 Fezandipiti/Shaymin 支付撤退成本，换入 ready Alakazam；
3. 给 Bench Alakazam 补 Psychic 后完成同回合撤退接力；
4. 只有上述路线都不存在时，才进入下一回合准备。

Telepath Energy 能提供 Psychic 攻击机会时，攻击价值高于它作为普通 Bench 搜索工具的
价值。Enriching Energy 不能支付 Abra 系列的 Psychic 攻击。

### 9.2 附能时机

可选的未来附能推迟到本回合过牌、检索和攻击机会已经确认之后。特别是手中仍有
Fezandipiti、Dudunsparce、Kadabra Ability 或能形成 Rare Candy + Alakazam 的路线时，
不得先给 Bench Kadabra/Abra 附能并消耗本回合机会。

### 9.3 没有当前攻击时

下一回合准备顺序为：

1. 有明确进化路线的 Abra/Kadabra/Alakazam；
2. 满足两只 Abra 场面后的第二打手；
3. 已满足两只 Abra 后，给 Dunsparce/Dudunsparce 使用能形成安全抽牌的 Enriching Energy；
4. 其它 Pokémon 只在不占用最低配置和攻击资源时考虑。

## 10. 测试设计

所有策略行为使用 TDD。每个实现轮次先加入能在旧代码上以预期原因失败的最小 fixture，
再修改策略。

### 10.1 TurnPlan 与 Energy

- Active Fezandipiti 缺一次撤退 Energy、Bench 有 ready Alakazam 时，附能目标必须是 Active，
  随后选择撤退并攻击；
- Active Abra 可通过 Psychic + Rare Candy + Alakazam 本回合攻击时，不能先给 Bench Kadabra
  附能；
- 当前已存在确定终局攻击时，未来附能、过牌和铺场不能阻挡攻击；
- 可选附能不能消耗 Hilda/Boss 所保留的当前路线资源。

### 10.2 场面不变量与恢复

- 场上只有一只 Abra 系列时，Poffin/Telepath/Night Stretcher/Lana's Aid 都先补 Abra；
- 达到两只 Abra 后才补 Dunsparce；
- 缺两个最低配置位置时，Fezandipiti/Shaymin 不能占用最后的 Bench slot；
- Lana's Aid 能从弃牌区恢复 Abra + Basic Psychic 时，两张都选中并随后铺场；
- Boss/Hilda 是本回合攻击必要组件时，Lana's Aid 不抢 Supporter。

### 10.3 Fezandipiti

- simulator 提供合法 Ability option 时，即使当前攻击不能 KO，也必须选择 Ability；
- 已有非终局攻击时，安全 Ability 在攻击前使用；
- 牌库保护线内且不存在终局闭环时拒绝抽牌；
- 真实 KO trace 的下一个己方回合能正确铺 Fezandipiti 并触发 Ability；
- 新对局会清空 KO trigger 和 effect serial，不能跨局泄漏。

## 11. AutoIteration 协议

历史根目录为 `work/auto-iteration/history_iterations/alakazam_v9_strategy/`。每轮包含
`result.json`、`iteration.md`、`index.html` 和 evaluation 原生 `run-*/` 目录。

### baseline

不修改 V9 策略，使用固定 17 个 opponent、每个 10 局、
`auto_iteration_v8_setup_relay` revision 2 建立 170 局基线。

### iteration-001：当前攻击资源预留

主要假设：显式 TurnPlan 能避免错误附能和漏掉撤退接力，从而提高本回合 Alakazam 攻击
完成率，且不损害 correctness。

### iteration-002：场面最低配置与恢复

主要假设：两只 Abra 优先、一只 Dunsparce 次之的统一缺口，配合 Lana's Aid/Night
Stretcher，能改善 Post-KO 接力并减少可恢复资源漏做。

### iteration-003：Fezandipiti 机会扩展

主要假设：合法 Flip the Script 在安全范围内进入真实 action 流，可以扩展当前攻击与接力
机会，且不引入牌库耗尽和攻击延误。

出现可归因的新问题时才追加 iteration-004；不为凑版本数量继续迭代。

## 12. 每轮评测和晋级

每轮按固定顺序执行：

1. 定向 RED/GREEN 单元测试；
2. V9/V8 相关策略回归和完整 `unittest`；
3. `python3 scripts/check_assets.py`、语法、package 和 raw-exec 检查；
4. 少量 focused 对局确认行为进入真实 action 流；
5. 同 profile、同 17x10 matrix 的 fresh full evaluation；
6. 记录事实、解释、主要假设、结果护栏和 `promote`/`observe`/`reject`。

晋级至少要求：

- candidate agent error 为 0；
- 目标 focused case 命中预期动作；
- 主要假设对应的 profile 信号或可复核 case 得到支持；
- 总体、先手、后手胜率没有无法解释的明显回退；
- 未改变卡表、profile、对手矩阵、分母和异常归属。

各 full run 是独立随机批次，不描述为逐局配对 A/B。局部过程指标改善不能覆盖 correctness
失败或明显结果回退。

## 13. 可视化产物

扩展现有 `scripts/auto_iteration_report.py`，增加 evaluation 原生
`summary.json`/`metrics.json` 到 iteration sample `result.json` 的适配，不另造新的评测口径。

### 单轮页面

每个 `iteration-NNN/index.html` 展示：

- control、candidate、主要假设和变更摘要；
- correctness、总体/先手/后手结果；
- 二回合 Powerful Hand、Post-KO relay、attack quality 和牌库健康；
- 关键 focused/full case、限制和 Decision；
- 原生 `run-*/report.html` 链接。

### 汇总页面

根 `index.html` 展示 baseline 到所有 iteration 的：

- 样本量与评测口径；
- 总体/先手/后手胜率；
- 二回合攻击、Post-KO、攻击质量等趋势；
- 每轮假设、主要变化、Decision 和 control 关系；
- 指向每轮详情页的稳定链接。

页面为静态 HTML、无外部依赖。每次 full evaluation 完成后立即重建，因此同一个入口始终
反映最新理解和实际结果。

## 14. 错误处理与边界

- simulator option 是合法性的最终来源；本地事实不足时不得生成不存在的 index；
- 必选 effect 没有选项时保持现有显式错误，不静默伪造结果；
- TurnMemory 在新对局清空 KO trigger、动作历史和 effect serial；
- focused evaluation 只证明机制，不单独支持 promotion；
- evaluation 或报告适配失败时，不运行后续策略晋级判断；
- 完整临时 trace 仍按现有生命周期写入临时目录并清理，长期只保留框架选择的少量 trace；
- 不下载或伪造缺失的 Kaggle replay；本地 full evaluation 是策略比较证据，官方 Kaggle
  Episode 仍是后续外部验证来源。

## 15. 2026-07-21 实际迭代结果

七轮均使用固定 `deck.csv` SHA-256
`0598646548d081832ec311c15fdc369b32c6f5e63175b0cfd1904d21fd082451`、17 个对手、
每个 10 局和 `auto_iteration_v8_setup_relay` revision 2。各轮是独立随机批次，下表只描述
同口径趋势，不视为逐局配对 A/B。

| 轮次 | 胜率 | 先手 / 后手胜场 | 二回合 PH | Post-KO | 可回收漏做 | 全攻击未拿奖赏 | PH 未拿奖赏 | errors | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 115/170 | 60/85 · 55/85 | 29/170 | 185/529 | 99 | 122/605 | 66/546 | 3 | observe |
| iteration-001 | 124/170 | 61/85 · 63/85 | 38/170 | 171/449 | 74 | 106/567 | 70/527 | 2 | promote |
| iteration-002 | 120/170 | 64/85 · 56/85 | 50/170 | 166/486 | 83 | 132/613 | 74/547 | 1 | observe |
| iteration-003 | 117/170 | 60/85 · 57/85 | 50/170 | 139/471 | 80 | 134/590 | 81/529 | 0 | observe |
| iteration-004 | 97/170 | 55/85 · 42/85 | 52/170 | 122/563 | 146 | 126/556 | 68/491 | 1 | reject |
| iteration-005 | 117/170 | 59/85 · 58/85 | 35/170 | 170/468 | 87 | 115/599 | 63/541 | 2 | promote |
| iteration-006 | 121/170 | 62/85 · 59/85 | 29/170 | 170/483 | 89 | 109/589 | 64/538 | 2 | promote |
| iteration-007 | 121/170 | 65/85 · 56/85 | 33/170 | 160/486 | 84 | 121/598 | 63/535 | 1 | promote |

### 15.1 baseline

- Run：`run-837ae45f0b944adc8202f8157cdb5f13`。
- 未修改策略的 `main.py` SHA-256 为
  `8e2ab06705368f789d09fa97bcb4de5d7171fd3dce109da6750e1b3ff5cdaab6`。
- 三个 error 均来自 `yakitori_raging_bolt` 的 opponent action；candidate-side agent error 为 0，
  但框架 correctness 总口径仍保留 `3/170`。

### 15.2 iteration-001：TurnPlan 与 Energy

- Run：`run-f911eabbcec8447c94c77ce191bd84db`。
- 显式处理 Active 进化、Boss 终局、当前 Energy、先附能后撤退以及 ready Alakazam 换入。
- 胜率、二回合 Powerful Hand、Post-KO 和总体攻击质量相对 baseline 同时改善，因此
  Decision 为 `promote`。
- 两个 error 仍是同一 opponent-side `IndexError`，不是候选合法性回归。

### 15.3 iteration-002：场面不变量与 Lana

- Run：`run-01c6110c386c4d269399e305ba639db3`。
- 手牌铺场、Poffin、Night Stretcher 和 Lana's Aid 统一读取“两 Abra 后一 Dunsparce”缺口；
  Lana 能按 Abra、Basic Psychic、进化资源顺序取回最大有用集合。
- 二回合 Powerful Hand 明显上升，但 Post-KO 与攻击质量没有延续 iteration-001 的改善，
  因而 Decision 为 `observe`。该行为仍属于用户明确要求，不回退。
- 唯一 error 仍是 opponent-side `IndexError`。

### 15.4 iteration-003：Fezandipiti

- Run：`run-067f8bcdea9747719acff6290b9c0d11`。
- 已提供的合法 Flip the Script option 不再依赖本地 KO 日志；低牌库、手牌保护和确定终局
  攻击仍会拒绝。手牌铺 Fez 继续要求真实 KO trigger，并服从 Bench slot 预留。
- `170/170` 全部完成且 correctness error 为 0；二回合 Powerful Hand 保持 `50/170`。
- 胜率略高于 baseline，但 Post-KO 与攻击质量低于前轮。由于批次不配对且没有 Fez 专属
  聚合 metric，Decision 为 `observe`，不把总体差异直接归因给 Ability。

### 15.5 iteration-004：reviewer 反例修复的过宽版本

- Run：`run-3d201c3bb6f244c494b9c25d5e2cdfd1`。
- 首次补齐精确 KO、Poffin 批内缺口、已撤退后死接力和 Night Stretcher
  当前攻击 Energy 四个 reviewer 反例，但把“当前可 KO”过宽地提升为绝对优先。
- 二回合 Powerful Hand 升到 `52/170`，但总体胜场降到 `97/170`、后手仅
  `42/85`，Post-KO 也降到 `122/563`，明确触发结果护栏，Decision 为 `reject`。
- 唯一 error 仍是 `yakitori_raging_bolt` 的 opponent-side engine error。

### 15.6 iteration-005：收窄精确 KO 保护

- Run：`run-9d3c9b201764417c80c11e4f05cd4141`。
- 只阻止会让 Powerful Hand 掉出当前 KO 阈值的 Pokémon 铺场；安全的
  Fezandipiti Ability 仍允许在非终局 KO 前使用。
- 胜场恢复为 `117/170`，Post-KO 改善为 `170/468`，全攻击未拿奖赏降为
  `115/599`。相比 iteration-004 通过结果护栏，Decision 为 `promote`。
- 两个 error 均是 `yakitori_raging_bolt` 的 opponent-side engine error，candidate action error 为 0。

### 15.7 iteration-006：冻结候选边界收口

- Run：`run-a2611d7873a346a4a40828cbee605656`。
- 进一步覆盖 Poffin 自身离手造成的精确 KO 丢失、已使用 Retreat 后所有
  Energy 类型的死接力，以及 Night Stretcher 在“Abra 已满两只”后先补
  Dunsparce、再考虑未来 Energy 的顺序。
- focused 临时评测（不纳入长期 history）为 `9/12`、error 0；full 为 `121/170`，
  先手 / 后手 `62/85 · 59/85`，
  Post-KO `170/483`，全攻击未拿奖赏 `109/589`。总体和先后手结果都高于
  baseline，因此 Decision 为 `promote`。
- 两个 error 均是 `yakitori_raging_bolt` 的 opponent-side engine error，candidate action error 为 0。

### 15.8 iteration-007：Nighttime Mine 精确 KO 收口

- Run：`run-250289e1b92e48738482e79077d485ab`；策略 package hash 为
  `752028e581184d703f43fb7ee48b018f5f7c47df8780d01e75ab9094d584d233`。
- reviewer 复现了手牌三张、对手 Active 60 HP 时，先打出 Nighttime Mine 会把
  Powerful Hand 从 60 降到 40 的确定奖赏丢失。修复把保护条件统一为“主动作净消耗
  手牌后是否丢失当前确定 KO”，不再依赖 Pokémon/Poffin 白名单；非 KO Stadium 路线
  保持可用。
- focused `run-ccdcd929384e4fd2a3449385683a9fd2` 为 `9/12`、error 0；该报告位于临时
  评测目录，不纳入长期 history。full 为 `121/170`，先手 / 后手
  `65/85 · 56/85`，二回合 Powerful Hand `33/170`。
- 相比 iteration-006，总胜场持平且二回合 Powerful Hand 上升；Post-KO
  `160/486` 和全攻击未拿奖赏 `121/598` 是混合信号。各轮是独立随机批次，不能把这些
  差异直接归因给单一动作修复。确定 KO 反例已由单元测试锁定，结果护栏未回退，因此
  Decision 为 `promote`。
- 唯一 error 是 `yakitori_raging_bolt-001` 的 opponent-side engine error，candidate
  action error 为 0。

### 15.9 当前结论

最终 V9 保留固定卡表下的 TurnPlan、Abra-first/Dunsparce-second、Lana/Night Stretcher
共用缺口、Fez option-driven Ability 和当前攻击 Energy 预留。iteration-004 作为过宽
KO 规则的明确 reject 样本保留；iteration-005/006 证明收窄后能恢复结果护栏，
iteration-007 则补齐 Nighttime Mine 的精确 KO 边界，并在总体胜场持平的 full run 中
通过结果护栏。冻结候选为 iteration-007 对应策略；`main.py` SHA-256 为
`3bab23664379a11d63604da769eca4c04bc8fc83c8a9e4ea34c7c1ac44857f0f`。根
`index.html` 是 baseline 到 iteration-007 的统一趋势入口，每轮页面链接对应
evaluation 原生报告。所有原生报告路径已相对化，`result.json` 也以机器可读字段明确
区分语义 Post-KO 成功率与 legacy zero-ready 失败率。
