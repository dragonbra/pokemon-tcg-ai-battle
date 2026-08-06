# 0032 最终特征表

状态说明：本表中的“保留”字段全部经过 `compiler -> batch -> forward` 校验；删除项不会进入 actor batch。

## 全局决策特征

| 字段 | 中文名称 | 含义 |
|---|---|---|
| `select_type` | 选择类型 | 当前引擎要求玩家完成的选择类别 |
| `select_context` | 选择上下文 | 当前选择发生在哪种规则上下文 |
| `relative_first_player` | 相对先手方 | 先手是自己还是对手 |
| `supporter_played` | 本回合已用支援者 | 本回合是否已经使用支援者卡 |
| `stadium_played` | 本回合已出竞技场 | 本回合是否已经打出竞技场 |
| `energy_attached` | 本回合已手贴能量 | 本回合通常能量贴附机会是否已使用 |
| `retreated` | 本回合已撤退 | 本回合撤退机会是否已使用 |
| `turn` | 回合编号 | 当前总回合编号 |
| `turn_action_count` | 回合内操作数 | 当前回合已执行的操作数量 |
| `own_deck_count` | 己方牌库张数 | 己方牌库剩余卡数 |
| `opponent_deck_count` | 对手牌库张数 | 对手牌库剩余卡数 |
| `opponent_hand_count` | 对手手牌数 | 对手当前手牌数量 |
| `own_prize_count` | 己方奖赏卡数 | 己方剩余奖赏卡数量 |
| `opponent_prize_count` | 对手奖赏卡数 | 对手剩余奖赏卡数量 |
| `own_bench_max` | 己方后场上限 | 当前规则效果下己方后场容量 |
| `opponent_bench_max` | 对手后场上限 | 当前规则效果下对手后场容量 |
| `remaining_damage_counter` | 剩余伤害指示物 | 当前多步选择尚需分配的伤害指示物数 |
| `remaining_energy_cost` | 剩余能量费用 | 当前多步选择尚需处理的能量费用 |

## 卡牌实例特征

| 字段 | 中文名称 | 含义 |
|---|---|---|
| `card_id` | 卡牌原型编号 | 用于唯一查找卡牌 prototype；不再额外做第二套身份 embedding |
| `relative_owner` | 相对持有者 | 该实体属于自己、对手或未知方 |
| `zone` | 所在区域 | owner-neutral 的前场、后场、手牌、弃牌等区域 |
| `zone_slot` | 区域槽位 | 同一区域内的准确位置 |
| `status_bits` | 特殊状态位 | 睡眠、灼伤、混乱、麻痹、中毒；明确无状态与不适用不同 |
| `appeared_this_turn_state` | 本回合登场状态 | 未登场、已登场、未知或不适用 |
| `current_hp` | 当前 HP | observation 中的当前 HP |
| `maximum_hp` | 当前最大 HP | 计入已生效规则后的 observation 最大 HP |
| `resolved_energy_type_0_count` | 有效无色能量份数 | 引擎已解析为 Colorless 的份数 |
| `resolved_energy_type_1_count` | 有效草能量份数 | 引擎已解析为 Grass 的份数 |
| `resolved_energy_type_2_count` | 有效火能量份数 | 引擎已解析为 Fire 的份数 |
| `resolved_energy_type_3_count` | 有效水能量份数 | 引擎已解析为 Water 的份数 |
| `resolved_energy_type_4_count` | 有效雷能量份数 | 引擎已解析为 Lightning 的份数 |
| `resolved_energy_type_5_count` | 有效超能量份数 | 引擎已解析为 Psychic 的份数 |
| `resolved_energy_type_6_count` | 有效斗能量份数 | 引擎已解析为 Fighting 的份数 |
| `resolved_energy_type_7_count` | 有效恶能量份数 | 引擎已解析为 Darkness 的份数 |
| `resolved_energy_type_8_count` | 有效钢能量份数 | 引擎已解析为 Metal 的份数 |
| `resolved_energy_type_9_count` | 有效龙能量份数 | 引擎已解析为 Dragon 的份数 |
| `resolved_energy_type_10_count` | 有效全属性能量份数 | 引擎已解析为 All 的份数 |
| `resolved_energy_type_11_count` | 有效超/恶复合能量份数 | 引擎已解析为 Psychic or Darkness 的复合份数 |

## 己方资源账本特征

| 字段 | 中文名称 | 含义 |
|---|---|---|
| `card_id` | 资源卡牌编号 | 当前账本条目对应的卡牌 prototype |
| `deck_knowledge` | 牌库知识状态 | 当前牌库数量是可见、记忆、精确推断、有界还是未知 |
| `prize_knowledge` | 奖赏区知识状态 | 当前奖赏区数量的知识状态 |
| `initial_count` | 注册卡组初始张数 | 60 卡注册卡组中该卡的初始数量 |
| `visible_playing` | 结算中可见数 | 该卡当前处于规则结算上下文的数量 |
| `deck_value` | 牌库精确数 | 精确已知时该卡在牌库的数量 |
| `deck_lower` | 牌库数量下界 | 不完全信息下该卡在牌库的最小可能数量 |
| `deck_upper` | 牌库数量上界 | 不完全信息下该卡在牌库的最大可能数量 |
| `prize_value` | 奖赏区精确数 | 精确已知时该卡在奖赏区的数量 |
| `prize_lower` | 奖赏区数量下界 | 该卡在奖赏区的最小可能数量 |
| `prize_upper` | 奖赏区数量上界 | 该卡在奖赏区的最大可能数量 |
| `deck_information_age` | 牌库信息年龄 | 该牌库信息距当前决策的事件年龄 |
| `prize_information_age` | 奖赏区信息年龄 | 该奖赏区信息距当前决策的事件年龄 |

## 因果事件特征

| 字段 | 中文名称 | 含义 |
|---|---|---|
| `log_type` | 日志事件类型 | 摸牌、移动、贴附、攻击、HP 变化等事件类别 |
| `relative_actor` | 相对执行者 | 事件由自己、对手或未知方执行 |
| `card_id` | 事件来源卡 | 事件直接关联的来源卡牌 prototype |
| `target_card_id` | 事件目标卡 | 事件直接关联的目标卡牌 prototype |
| `attack_id` | 事件攻击编号 | 攻击事件所使用的 attack prototype |
| `from_area` | 来源区域 | 卡牌移动前所在的引擎区域 |
| `to_area` | 目标区域 | 卡牌移动后所在的引擎区域 |
| `active_card_id` | 前场卡编号 | 切换事件中的前场卡 prototype |
| `bench_card_id` | 后场卡编号 | 切换事件中的后场卡 prototype |
| `before_card_id` | 变化前卡编号 | 进化或变化前的卡牌 prototype |
| `after_card_id` | 变化后卡编号 | 进化或变化后的卡牌 prototype |
| `is_recover` | 是否回复 | HP 变化是否为回复 |
| `put_damage_counter` | 是否放置伤害指示物 | 官方日志的 `putDamageCounter` 布尔标志；未知、否、是三态分开编码 |
| `coin_head` | 投币是否正面 | 官方日志单次投币的 `head` 布尔结果；未知、反面、正面三态分开编码 |
| `special_condition_type` | 特殊状态类型 | 事件施加或处理的特殊状态 |
| `result_type` | 结果类型 | 引擎公开日志中的离散结果编号 |
| `reason_type` | 原因类型 | 引擎公开日志中的离散原因编号 |
| `age` | 事件年龄 | 该事件距最新可见事件的距离 |
| `value` | 事件数值 | 日志公开的通用数值 |
| `count` | 事件计数 | 日志公开的 count 参数 |
| `number` | 事件编号值 | 日志公开的 number 参数 |

## 合法选项特征

| 字段 | 中文名称 | 含义 |
|---|---|---|
| `action_type` | 操作类型 | 当前合法选项对应的引擎操作类别 |
| `source_owner` | 来源持有者 | 操作来源属于自己还是对手 |
| `source_area` | 来源区域 | 操作来源所在引擎区域 |
| `target_owner` | 目标持有者 | 操作目标属于自己还是对手 |
| `target_area` | 目标区域 | 操作目标所在引擎区域 |
| `source_card_id` | 来源卡牌编号 | 来源卡牌 prototype；关系缺失时仍可保留语义 |
| `target_card_id` | 目标卡牌编号 | 目标卡牌 prototype |
| `attack_id` | 攻击编号 | 该选项对应的 attack prototype |
| `special_condition_type` | 特殊状态选项 | 该选项选择的特殊状态类型 |
| `context_card_id` | 上下文卡牌编号 | 发起当前选择的上下文卡牌 prototype |
| `effect_card_id` | 生效卡牌编号 | 当前结算关联的卡牌 prototype，不是 effect node |
| `number` | 选项数值 | 引擎选项公开的 number 参数 |
| `count` | 选项计数 | 引擎选项公开的 count 参数 |

## 关系特征

| 字段 | 中文名称 | 含义 |
|---|---|---|
| `card_parent` | 卡牌附属关系 | Energy、Tool、进化卡具体附属于哪只宝可梦 |
| `event_source` | 事件来源实例 | 事件来源绑定到当前 observation 中的具体卡实例 |
| `event_target` | 事件目标实例 | 事件目标绑定到具体卡实例 |
| `option_source` | 选项来源实例 | Attach 等选项来自哪张具体能量卡 |
| `option_target` | 选项目标实例 | Attach 等选项要作用于哪只具体宝可梦 |

## Prototype 特征

- 卡牌：卡种、宝可梦属性、进化阶段、进化来源、规则/流派标志、9 位供能属性、9 位弱点、9 位抗性、基础 HP、撤退费用、引擎能量份数、卡牌到技能、卡牌到攻击。
- 攻击：20 位攻击规则标志、5 个有序需求能量槽、取消失败行为、基础伤害；所属卡牌由卡牌到攻击的单向关系恢复。
- 技能：技能类别、25 位生效区域、8 个技能标志、最多两个触发器；所属卡牌由卡牌到技能的单向关系恢复。
- 每个触发器：触发类型、目标玩家、`not_me`、`skip_enemy_target`、两个有序区域、两个有序筛选条件；每个条件含类型、比较符、名称、value、value2。
- 引擎静态 `Trigger` 只有 `trigger_type` 和一份完整的 `subject` 目标规则，没有独立静态 `trigger_target`。运行时真正命中的 `TriggerInfo.subject/object` 不在官方 actor observation 中，不能伪造；公开日志中可见的具体实例由 `event_source/event_target` 表达。
- 卡牌规则/流派标志逐项为：`can_play_first_turn` 首回合可用、`can_trash` 可主动弃置、`transform_only` 仅变身使用、`trash_my_turn_end` 回合末弃置、`cannot_to_hand_or_deck_in_trash` 弃牌区不可回收、`tera` 太晶、`to_bench` 直接去后场、`to_battle_field_only_setup` 开局仅前场、`to_active_only_setup` 开局仅战斗场、`no_prize` 不取奖赏、`only_team_rocket` 仅火箭队、`ancient` 古代、`future` 未来、`hop` 赫普、`lillie` 莉莉艾、`iono` 奇树、`n` N、`ethan` 阿响、`cynthia` 竹兰、`misty` 小霞、`arven` 派帕、`steven` 大吾、`marnie` 玛俐、`erika` 莉佳、`larry` 青木、`team_rocket` 火箭队、`ace_spec` ACE SPEC、`can_use` 可使用标志。
- 技能标志逐项为：`main_ability` 主动特性、`once_turn` 每回合一次、`select_activation` 可选择触发、`not_stack` 不叠加、`activate_in_discard` 弃牌区生效、`attach_bench` 贴附到后场、`ko_self` 自身昏厥、`lucky_bonus` 幸运奖励。

## 明确删除

`effect_node_graph`、`is_condition`、运行时 trigger subject/target、单独附着能量卡计数、有效能量总份数、有效能量类型 mask、贴附后供能属性/份数、攻击/撤退能量缺口、newly enabled、伤害后 HP、KO 答案、卡牌 serial 数值、无法从 observation 证实的牌库顺序、owner-specific zone、重复 option 选择上下文、可由卡牌关系重建的 option-skill 关系、反向 attack/skill-to-card 关系、重复身份 embedding、队伍/玩家 persona 均不进入模型。
