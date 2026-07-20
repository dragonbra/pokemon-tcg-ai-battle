# Alakazam V8 卡组说明预留

这份文件是 V8 的人工说明入口。卡表已经切换为排行榜对局中玩家 0
`Yushin Ito` 的 60 张构筑；策略采用 `alakazam_v7_auto_iter_iter46`。

请在“你的特殊用法说明”一列补充你希望 agent 了解的卡牌语义、使用时机、
优先级或组合。填写后可以把“说明状态”从 `待填写` 改为 `已填写`；没有额外
用法的卡可以标记为 `无需特殊说明`。

| 类型 | 卡牌 | ID | 数量 | 说明状态 | 你的特殊用法说明 |
|---|---|---:|---:|---|---|
| Pokémon | Abra | 741 | 4 | 待填写 | |
| Pokémon | Kadabra | 742 | 4 | 待填写 | |
| Pokémon | Alakazam | 743 | 4 | 待填写 | |
| Pokémon | Dunsparce | 305 | 3 | 待填写 | |
| Pokémon | Dudunsparce | 66 | 2 | 待填写 | |
| Pokémon | Fezandipiti ex | 140 | 1 | 待填写 | |
| Pokémon | Shaymin | 343 | 1 | 待填写 | |
| Energy | Basic {P} Energy | 5 | 2 | 待填写 | |
| Energy | Enriching Energy | 13 | 1 | 待填写 | |
| Energy | Telepath Psychic Energy | 19 | 4 | 待填写 | |
| Trainer | Rare Candy | 1079 | 3 | 待填写 | |
| Trainer | Enhanced Hammer | 1081 | 4 | 待填写 | |
| Trainer | Buddy-Buddy Poffin | 1086 | 4 | 待填写 | |
| Trainer | Night Stretcher | 1097 | 1 | 待填写 | |
| Trainer | Sacred Ash | 1129 | 1 | 待填写 | |
| Trainer | Poké Pad | 1152 | 4 | 待填写 | |
| Trainer | Boss’s Orders | 1182 | 3 | 待填写 | |
| Trainer | Lana’s Aid | 1184 | 1 | 待填写 | |
| Trainer | Xerosic’s Machinations | 1197 | 3 | 待填写 | |
| Trainer | Hilda | 1225 | 4 | 待填写 | |
| Trainer | Dawn | 1231 | 4 | 待填写 | |
| Trainer | Nighttime Mine | 1266 | 2 | 待填写 | |

## 使用边界

- 当前版本不会自动读取本文件，也不会因为 `待填写` 状态改变动作选择。
- 你补充说明后，再逐条把明确的 Trainer 时机、卡牌组合和例外条件转成 V8
  策略代码或测试。
- 卡牌数量以 `deck.csv` 为准；本表每一行代表一种卡牌，不代表单张实体。
