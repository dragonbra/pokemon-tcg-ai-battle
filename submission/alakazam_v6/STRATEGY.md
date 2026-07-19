# Alakazam V6 策略说明

## 1. 回合状态

`TurnMemory` 在新对局时清空，在每个己方回合开始保存场上 Pokémon 的 serial。它同时维护：

- `turn_start_serials`、`played_at_turn`、`evolved_at_turn`；
- Supporter、手填 Energy、Retreat、攻击和主动结束的本回合预算；
- Abra/Kadabra/Alakazam/Dunsparce 资源在手牌、场上、弃牌、牌库和已知 Prize 中的可见账本，以及未知数量边界。

引擎的 `appearThisTurn`、`supporterPlayed`、`energyAttached` 和 `retreated` 仍是合法性校验来源；本地历史只补充时间语义，不替代引擎。

## 2. 主行动顺序

主行动先执行 `preparation_is_due()` 的硬性前置判断，再在同一阶段的合法 option 内
进行确定性排序；因此攻击不会因为一个较低的 tuple score 而抢在仍有必要的准备动作之前。
实际顺序为：

1. 当前攻击或 Boss 能否在本回合制造确定 Prize；Active 不能 KO 时，Boss 从可 KO Bench 中选剩余 HP 最高者。
2. 完成 Active 的合法进化。带 Psychic Energy 的 Active Kadabra 有 Alakazam 时，先自然进化。
3. 为后续攻击线进化合法 Bench Abra 到 Kadabra，并在攻击前使用其 Ability；即使
   Active 当前已经能 KO，也不会跳过这条已经可执行的接力准备。
4. 分配唯一的手填 Energy。Active Alakazam 已能攻击时，Enriching Energy + Dudunsparce 默认转为过牌；否则补没有 Psychic Energy 的 Abra 线。
5. 只有能补齐明确攻击路线时才用 Night Stretcher/Lana's Aid；需要同时拿回 Pokémon 和 Basic Energy 时，Lana's Aid 优先。
6. 在牌库预算允许时使用 Dudunsparce、Kadabra、Alakazam 或其他过牌。牌库剩余 10 张或更少时，只有抽牌后可以完成本回合最后 Prize 闭环才继续。
7. 没有 Boss KO 时，且对手 Active 已受伤未 KO、手牌至少 6 张、我方 Active Alakazam 已能攻击，使用 Xerosic。
8. 完成所有准备后攻击；低阶段攻击只作为最后手段，Trading Places 永远低于 END。

## 3. 效果选择

效果选择继续使用 V5 的合法 option 映射，但 V6 修改了关键状态：Rare Candy 和普通
进化目标都受回合开始快照与本回合进化历史约束；进化后新 serial 通过
`preEvolution`/动作历史关联，Boss 目标按打出 Boss 后的手牌伤害计算并在确定 KO 集合
中按剩余 HP 降序；恢复选择从下一次现实攻击路线的缺口倒推，并在 Lana's Aid 中为
Basic Psychic 预留槽位。

## 4. 实验边界

V6 不预设 opponent ID 或完整构筑，只使用 observation 中实际看到的 Active、Bench、伤害、能量、手牌、Prize 和牌库状态。Land Collapse 仍然只作为已观察到的牌库压力，不建立 matchup 特例。真实胜率和节奏需要 Kaggle 官方 replay 评估；本地 battle 只用于发现非法 option、崩溃和状态泄漏。
