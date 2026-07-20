# Alakazam V8 卡组笔记设计

## 目标

把 `submission/alakazam_v8/DECK_NOTES.md` 从空白备注入口完善为一份可供人工复核、后续策略迭代使用的卡牌语义记录。它不是运行时配置，也不直接改变 agent 的动作选择。

## 输入与边界

- 以 `submission/alakazam_v8/deck.csv` 的实际 60 张卡为准，逐一覆盖 22 种卡牌。
- 卡面事实以 `data/official/EN_Card_Data.csv` 为准。
- 策略背景以 `submission/alakazam_v7_auto_iter/` 的 AutoIter 语义、iter-46 记录和 V8 当前说明为准。
- 说明使用中文，并保留卡牌英文名、ID、攻击名和 Ability 名，便于和 replay、代码及卡表对照。
- 不预设对手构筑，不把单局观察写成普遍规则；需要看到实际场面才能触发的效果要明确说明。

## 输出结构

1. 在逐卡表格前增加简短的阅读说明，解释“卡面事实”“V7 已关注/实现的策略语义”和“V8 下一步建议”的区别。
2. 增加卡组级使用边界，记录攻击终止回合、Supporter/手填能量/Retreat 预算、进化时机、牌库保护线和手牌伤害管理。
3. 保留现有逐卡表格和数量，将每一行状态改为 `已填写`，备注至少覆盖：
   - 这张牌在本卡组中的主要职责；
   - 何时优先使用，以及它服务哪条攻击或过牌路线；
   - 规则限制、资源代价和不应使用的情形；
   - 与 V8 构筑变化或 V7 策略的关系。
4. 在表格后增加 V8 构筑差异小结，明确 `Enhanced Hammer` 增至 4 张、加入 `Nighttime Mine`、`Dudunsparce` 减至 2 张，以及移除 `Wondrous Patch` 和 `Battle Cage` 对资源循环的影响。

## 逐卡内容重点

- Abra 线：区分主攻击者、自然进化过牌目标和接力攻击者；明确 Abra 攻击及 `Trading Places` 的策略禁区。
- Dunsparce 线：说明 `Enriching Energy`、`Run Away Draw`、Bench 接力和牌库净变化；空 Bench 时不洗回唯一 Active。
- Fezandipiti ex 与 Shaymin：说明 Rule Box、Prize 风险、Ability/保护效果和非攻击者定位。
- 三类能量：区分 Psychic 攻击需求、`Enriching Energy` 的抽牌触发和 `Telepath Psychic Energy` 的检索价值。
- Trainer：分别记录搜索、恢复、拆特殊能量、换位、手牌压制和 Stadium 的 Supporter/Item/牌库代价，避免把不同卡的区域限制混为一谈。

## 表达约定

- `V7 已实现/已验证`：当前继承策略明确要求的行为，例如攻击是回合终止提交、非终局不使用低阶段弱攻、Item Lock 下改走自然进化，以及低牌库时停止非终局过牌。
- `V8 建议`：根据 V8 实际卡表和规则推导出的候选理解，例如提高 Enhanced Hammer 密度后优先处理实际存在的特殊能量，或仅在观察到 Tera Pokémon 时发挥 Nighttime Mine。
- 如果某条建议尚未落到代码或测试，不写成已经自动执行的承诺。

## 验收标准

- 表格仍与 V8 `deck.csv` 的卡牌种类、ID 和数量一致，总数量为 60。
- 22 种卡牌均有非空备注，且没有把移除的 V7 卡牌误列入 V8 卡表。
- Markdown 表格列数一致，状态和备注没有错位。
- 备注没有违反官方规则或 V6/V7 已确认边界，尤其是攻击终止回合、进化时机、Supporter 次数、手填能量次数和牌库保护线。
