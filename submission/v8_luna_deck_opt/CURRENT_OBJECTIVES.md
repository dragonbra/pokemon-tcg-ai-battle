# 本轮目标

1. 在不改变卡组与运行时契约的前提下，建立可重复的 Auto Iteration 记录流程。
2. 选择一个主要策略假设，先通过 correctness gate，再进行 focused evaluation。
3. 用明确分母记录第二回合攻击、打手接力、牌库安全和结果 guardrail。
4. 只有 control/candidate 对比达到 promotion 规则时，才把候选晋级为新的基线。

## v8_luna_deck_opt：Deck Notes review 任务

本轮 review 将 `work/docs/DECK_NOTES.md` 中新增的卡牌使用说明转化为
v8_luna_deck_opt 的可执行任务。目标不是单独提高某张牌的使用次数，而是让每个动作
都符合以下顺序：先利用本回合仍然合法且有长期价值的资源，再提交攻击；攻击
提交后本回合立即结束，不能把攻击当成普通的最后一个排序选项。

用户在 Deck Notes 中使用 `Arven`、`Nest Ball`、`Fusion Energy` 举例说明
Dudunsparce 换位路线。这些牌不在 V8 实际卡表中，只作为概念示例；实现时必须
映射到 V8 实际拥有的 `Poké Pad`、`Dawn`、`Hilda`、`Buddy-Buddy Poffin`、
`Telepath Psychic Energy` 等资源，不能把示例牌加入卡组或代码。

### P0：先修正动作语义

| ID | 主题 | 当前审计结论 | 具体任务与 correctness gate |
|---|---|---|---|
| V8C-01 | Alakazam 攻击前资源利用 | `_main_action` 已有进化、过牌和接力闸门，但仍需验证“当前可 KO”不会压过本回合可完成的进化、Ability、检索、Supporter 或打手接力。 | 建立“Active Alakazam 可 KO、Bench 仍有合法 Abra/Kadabra 接力动作”的 fixture；必须先完成有现实收益的动作，再选择攻击。最终 Prize 只在没有更高价值的合法当前回合动作时优先。 |
| V8C-02 | Dunsparce `Trading Places` | `_is_forbidden_terminal_attack` 已尝试过滤该攻击，但排序列表为空时的安全兜底仍可能返回它。 | 将 `Trading Places` 视为永远不可用的策略动作；若同时存在 `END`，选择 `END`，不得因排序兜底选中该攻击。测试只有 `Trading Places` 与 `END`、以及它和正常动作并存的场面。 |
| V8C-03 | Dudunsparce 主动换位 | 现有 `_dudunsparce_deck_delta` 只计算牌库变化，`Run Away Draw` 的换位价值没有被明确绑定到“洗回后由可攻击的 Bench Alakazam 接班”。 | 覆盖 Active Dunsparce、Bench 已具备 Alakazam + Psychic、手中能找到/进化 Dudunsparce 的二回合路线；必须先合法进化、使用 Ability、完成换位，再攻击。空 Bench 和牌库保护仍禁止无意义 Ability。 |
| V8C-04 | Fezandipiti ex `Flip the Script` | `_fezandipiti_play_is_safe` 会因已有普通 Bench 或可攻击线路过度压制 Fezandipiti；这与“上一回合被击倒且 Ability 合法时应把它作为重要补手牌”不符。 | 在上一回合己方 Pokémon 被击倒、Fezandipiti 已可上场/已在场、Ability 可触发的 fixture 中，优先完成抽 3；只有没有合法触发条件、没有合法 Bench，或抽牌后仍会错过最后 Prize 闭环时跳过。 |
| V8C-05 | Enhanced Hammer 主动拆能 | 主行动排序在当前攻击可 KO 时可能把合法 Hammer 放到攻击之后；能量选择还会让 Active 普通特殊能量压过 Bench 保护性特殊能量。 | 只要对手实际存在 Special Energy 目标且 Item 未被锁定，就优先使用 Hammer。目标顺序固定为 Active 保护性特殊能量、Bench 保护性特殊能量、Active 其它特殊能量、Bench 其它特殊能量；测试 Rock Fighting Energy、Mist Energy 和普通 Special Energy。 |

### P1：补齐接力和检索资源管理

| ID | 主题 | 当前审计结论 | 具体任务与 correctness gate |
|---|---|---|---|
| V8C-06 | Lana’s Aid Pokémon-first | `_recovery_needs` 和 `_select_effect` 当前按 Pokémon/Energy 缺口做最小选择，并把 Basic Psychic 放在很高优先级；这会削弱它作为 Alakazam 接力入口的作用。 | 当需要新打手时，优先从弃牌区恢复 Abra 线 Pokémon，并在合法范围内使用最多三张；只有 Basic Psychic 是完成当前攻击的必要缺口时才加入 Energy。保留唯一 Supporter 和最后 Prize 例外测试。 |
| V8C-07 | Sacred Ash 最大化恢复 | `_select_effect` 在 `minCount=1` 时可能只选一张，且通用排序偏向高阶段 Pokémon，不符合“尽量选满、先建 Abra 底座”的接力目标。 | 在最多五张合法选择中尽量选满；优先组织两条 Abra → Kadabra → Alakazam 接力，基础阶段优先，但结合 Rare Candy 和当前场况降低不必要 Kadabra；之后再补 Dunsparce 线或其它 Pokémon。 |
| V8C-08 | Poké Pad 延迟使用 | 当前第一回合分支可能把 Poké Pad 当作普通 Basic 搜索，未充分判断“两只 Abra + 一只 Dunsparce”是否已经完成。 | 第一回合已有现实的两只 Abra 和一只 Dunsparce，且没有立即进化/换位收益时，不使用 Poké Pad；保留给 Kadabra、Alakazam、Dudunsparce 等关键目标。对应场面必须优先选择其它合法动作或结束回合。 |
| V8C-09 | Dawn 的 Dudunsparce 路线 | Dawn 的类别排序主要按固定 Basic/Stage 1/Stage 2 分组，没有根据场上是否缺 Dunsparce、Active 是否为 Dunsparce 来提高 Dudunsparce 换位资源的优先级。 | 无 Dunsparce 时应考虑取 Dunsparce；Active Dunsparce 且 Bench Alakazam 可接班时，应考虑取 Dudunsparce；每个合法类别都应完成有价值的选择，不能明明能拿三张却只拿一张。 |

### P2：新增 Stadium 的实际价值

| ID | 主题 | 当前审计结论 | 具体任务与 correctness gate |
|---|---|---|---|
| V8C-10 | Nighttime Mine | v8_luna_deck_opt 尚未为 Nighttime Mine 建立独立的 Stadium 优先级；通用卡牌排序可能让它落后于攻击或无关动作。 | 只要 Nighttime Mine 在手牌中且当前可以合法打出，就优先打出，用于覆盖对手 Stadium；我方没有 Tera Pokémon，因此不应为我方攻击增加成本。若观察到对手 Tera Pokémon，再验证其攻击费用被增加。 |

### 验证与晋级标准

每个 P0/P1 任务都必须先有最小 observation fixture，再运行策略单测；测试至少覆盖
合法性、动作顺序、选择数量/目标排序和牌库保护边界。测试不得把用户举例中的
V8 不存在卡牌伪造进运行时。

focused evaluation 统一记录以下指标：

- 第二回合实际选择 `Powerful Hand` 的次数与“到达己方第二回合且存在合法攻击选项”的分母；
- 二回合攻击前漏掉的合法进化、过牌、接力、Fezandipiti、Hammer 和 Dudunsparce 换位动作；
- Alakazam 被击倒后的下一只可攻击打手形成率，以及 Lana’s Aid/Sacred Ash 恢复目标分布；
- `Trading Places` 选择次数、Hammer 合法目标拆除率、Nighttime Mine 可用时使用率；
- 低牌库保护触发次数、错误过牌次数、错误延迟终局攻击次数和 agent/evaluator error。

候选版本只有在 correctness gate 全部通过、没有新增非法动作或崩溃，并且在固定
control/candidate 对比中改善目标指标且不违反胜率、终局攻击和牌库安全 guardrail
时，才能晋级为 v8_luna_deck_opt 新基线。单个本地对局不能作为晋级依据，正式结论优先
使用 Kaggle Episode/replay 数据。
