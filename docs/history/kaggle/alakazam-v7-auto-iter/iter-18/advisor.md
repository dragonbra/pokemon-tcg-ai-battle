# Iter-18 Strategy Advisor

## 复核对象

本轮只复核 `kiyotah_dragapult/game_008.json` 的第二个己方回合，以及对应的
`firstPlayer` swap 计数问题。卡组和 `deck.csv` 均未修改。

## 卡牌与规则核验

- `Abra (741)` 是 Basic；带有 Psychic Energy 的 Active Abra 可以在允许的己方
  回合从手牌进化成 `Kadabra (742)`。
- `Kadabra (742)` 从手牌进化时可使用 `Psychic Draw` 抽 2 张；它的
  `Super Psy Bolt` 需要 `{P}`，造成 30 点伤害。
- `Budew (235)` 的 `Itchy Pollen` 只限制对手下一回合从手牌使用 Item。它会阻止
  `Rare Candy`，但不会阻止从手牌自然进化成 Kadabra。
- 宝可梦 TCG 的共享回合计数不能直接假定 player index 0 是先手。对方交换先手时，
  `firstPlayer` 才能确定某个物理 index 在共享 turn 3 是否已经进入自己的第二回合。
- 进化和攻击是同一回合内的先后操作；一旦选择攻击，本回合立即结束。因此本 case
  应先进化 Active Abra，再在后续 observation 中用 Kadabra 攻击，而不是直接 END。

## 边界

本轮没有把“所有 Active Abra 都优先进化”泛化为硬规则：只有当进化合法、手中有
Kadabra、Active 已有 Psychic Energy，并且进化后的 Kadabra 能对当前 Active 形成
确定 KO 时，才提升该 Active 进化的优先级。其它场面继续保留原有 Rare Candy、Bench
接力和牌库保护判断。
