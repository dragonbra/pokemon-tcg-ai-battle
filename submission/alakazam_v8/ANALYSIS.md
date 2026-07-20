# Alakazam V8 卡组差异分析

> V8 已切换为排行榜对局中玩家 0 `Yushin Ito` 的 60 张卡表；运行时策略采用
> `alakazam_v7_auto_iter_iter46`。卡牌特殊用法的人工说明入口见
> [`DECK_NOTES.md`](DECK_NOTES.md)。

## 1. 分析范围与数据来源

- 对局来源：`/Users/hejinyu/Downloads/87100444.json`
- Episode：`87100444`
- 目标卡组：初始 `visualize` 快照中的玩家 0，`Yushin Ito`，本局奖励为 `+1`
- 对照卡组：当前 `submission/alakazam_v7/deck.csv`
- V7、`alakazam_v7_auto_iter`、`alakazam_v7_auto_iter_iter46` 和
  `alakazam_v7_auto_iter_submit_v15` 的卡表当前一致，因此这里用 V7 作为代表。
- 卡牌名称、ID、类别和效果：`data/official/EN_Card_Data.csv`

本文件只研究卡组构筑差异。JSON 中的对局行动不作为逐回合复盘依据，也不据此
推断这些卡在更多对局中的胜率。

## 2. 构筑组成概览

两副牌都是 60 张，能量数量完全相同；变化集中在 Pokémon 和 Trainer：

| 类别 | V7 | Yushin Ito | 变化 |
|---|---:|---:|---:|
| Pokémon | 20 | 19 | -1 |
| Energy | 7 | 7 | 0 |
| Trainer | 33 | 34 | +1 |
| 合计 | 60 | 60 | 0 |

核心进攻线没有变化：`Abra ×4`、`Kadabra ×4`、`Alakazam ×4`。主要过牌和检索
骨架也没有变化，包括 `Dunsparce ×3`、`Poké Pad ×4`、`Buddy-Buddy Poffin ×4`、
`Hilda ×4`、`Dawn ×4`、`Rare Candy ×3` 和 `Xerosic’s Machinations ×3`。

这不是简单的“用两张 Enhanced Hammer 换掉两张 Battle Cage”。真实构筑调整是：

- `Enhanced Hammer` 增加 2 张，达到 4 张。
- 新增 `Nighttime Mine` 2 张。
- `Battle Cage` 2 张全部移除。
- `Wondrous Patch` 移除 1 张。
- `Dudunsparce` 减少 1 张，由 3 张变为 2 张。

换句话说，V7 的资源恢复与后场保护，换成了更高密度的特殊能量干扰，以及针对
Tera Pokémon 的 Stadium 干扰。

## 3. 完整卡表对照

`Delta` 为 Yushin Ito 卡表减去 V7 卡表；负数表示 Yushin Ito 少了该卡。

### Pokémon

| 卡牌 | ID | V7 | Yushin Ito | Delta |
|---|---:|---:|---:|---:|
| Abra | 741 | 4 | 4 | 0 |
| Kadabra | 742 | 4 | 4 | 0 |
| Alakazam | 743 | 4 | 4 | 0 |
| Dunsparce | 305 | 3 | 3 | 0 |
| Dudunsparce | 66 | 3 | 2 | -1 |
| Fezandipiti ex | 140 | 1 | 1 | 0 |
| Shaymin | 343 | 1 | 1 | 0 |

### Energy

| 卡牌 | ID | V7 | Yushin Ito | Delta |
|---|---:|---:|---:|---:|
| Basic {P} Energy | 5 | 2 | 2 | 0 |
| Enriching Energy | 13 | 1 | 1 | 0 |
| Telepath Psychic Energy | 19 | 4 | 4 | 0 |

### Trainer

| 卡牌 | ID | V7 | Yushin Ito | Delta |
|---|---:|---:|---:|---:|
| Rare Candy | 1079 | 3 | 3 | 0 |
| Enhanced Hammer | 1081 | 2 | 4 | +2 |
| Buddy-Buddy Poffin | 1086 | 4 | 4 | 0 |
| Night Stretcher | 1097 | 1 | 1 | 0 |
| Sacred Ash | 1129 | 1 | 1 | 0 |
| Wondrous Patch | 1146 | 1 | 0 | -1 |
| Poké Pad | 1152 | 4 | 4 | 0 |
| Boss’s Orders | 1182 | 3 | 3 | 0 |
| Lana’s Aid | 1184 | 1 | 1 | 0 |
| Xerosic’s Machinations | 1197 | 3 | 3 | 0 |
| Hilda | 1225 | 4 | 4 | 0 |
| Dawn | 1231 | 4 | 4 | 0 |
| Battle Cage | 1264 | 2 | 0 | -2 |
| Nighttime Mine | 1266 | 0 | 2 | +2 |

## 4. 实际 Diff

下面是两副 60 张卡表之间所有非零差异，没有把相同卡牌列入“变化”：

| 变化 | 卡牌 | ID | 构筑含义 |
|---:|---|---:|---|
| +2 | Enhanced Hammer | 1081 | 特殊能量拆除密度从 2 张提高到 4 张 |
| +2 | Nighttime Mine | 1266 | 新增针对 Tera Pokémon 攻击费用的 Stadium |
| -1 | Dudunsparce | 66 | 少一条可用 `Run Away Draw` 的 Dudunsparce 资源 |
| -1 | Wondrous Patch | 1146 | 放弃 1 张从弃牌区给后场 Psychic Pokémon 回收 Basic Psychic Energy 的 Item |
| -2 | Battle Cage | 1264 | 放弃后场伤害指示物保护，全部换出 Stadium 槽位 |

### 新增卡的数据库效果

以下内容直接读取自 `data/official/EN_Card_Data.csv`。

#### Enhanced Hammer（ID 1081，TWM 148）×2

- 类别：Item
- 效果：`Discard a Special Energy from 1 of your opponent’s Pokémon.`
- 中文释义：将对手一只 Pokémon 身上的 1 张特殊能量弃置。
- 限制：只能处理特殊能量，不能处理 Basic Energy。

相对 V7，这张卡达到 4 张后，卡组更稳定地寻找并使用特殊能量拆除。它的价值
取决于对手是否实际使用特殊能量；如果目标只有 Basic Psychic Energy 等基础能量，
这张卡不能完成拆能。

#### Nighttime Mine（ID 1266，ASC 197）×2

- 类别：Stadium
- 效果：`Attacks used by each Tera Pokémon in play (both yours and your opponent’s)
  cost {C} more.`
- 中文释义：场上双方每一只 Tera Pokémon 使用招式时，费用增加 `{C}`。

这张 Stadium 不是 Alakazam 自身的通用增伤卡；它主要改变含有 Tera Pokémon 的
对局中的攻击门槛，而且效果同时作用于双方。若对手没有 Tera Pokémon，或者当前
攻击不受增加一格无色费用影响，它的即时收益会降低。

## 5. 初步构筑结论

1. **Alakazam 主轴保持不变。** 4-4-4 进化线、7 张能量、过牌检索 Trainer 和
   `Xerosic’s Machinations ×3` 都没有调整，说明变化不是重做引擎，而是在稳定
   主轴上增加针对性卡位。
2. **最明确的方向是特殊能量对策。** `Enhanced Hammer` 从 2 张升到满编 4 张，
   是本次 Diff 中最直接的密度提升。
3. **Stadium 由防守转为对局限制。** `Battle Cage ×2` 和 `Wondrous Patch ×1`
   被移除，换入 `Nighttime Mine ×2`；这降低了后场伤害防护和弃牌区能量恢复，
   但对 Tera 对手提供了攻击费用压力。
4. **过牌回收能力略有下降。** `Dudunsparce` 少 1 张意味着 `Run Away Draw`
   的可重复资源减少。该牺牲与 `Wondrous Patch` 同时减少，说明这套构筑更偏向
   主动干扰对手，而不是延长自己的资源循环。

以上是构筑层面的研究假设，不是基于单个 Episode 得出的胜率结论。下一步若要继续
迭代 V8，应把这 5 个非零 Diff 作为独立候选变更逐项验证，而不是一次
性把整副牌的差异归因于某一张卡。
