# V7 AutoIter iter-44 独立规则 / 卡牌 Advisor

## 本轮结论

terminal Powerful Hand 的边界成立：攻击宣告会立即结束回合，因此当攻击已经拿完
最后奖赏时，Fezandipiti ex 的可选抽牌不应抢先。该判断必须同时检查实际伤害、目标 HP、
保护能量和剩余 Prize 数，不能只看“攻击是 KO”。

## 下一轮已确认 case：Telepath + Poké Pad

来源：
`/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-43-control-20260720/zoli_dragapult/game_004.json`，raw[75]。

- Active Alakazam `743` 已有 Basic Psychic；Bench 有两只未附能 Abra。
- 合法 options 同时包含同一张 Telepath Energy 给 Active（option 6）或 Bench Abra
  （option 7/8/9），也包含 `Powerful Hand`。
- 手牌含 Telepath Energy、Basic Psychic、Rare Candy 和 Poké Pad `1152`，而不是直接含
  Kadabra。
- 实际 control 给 Active Alakazam 贴 Telepath；随后才用 Poké Pad 找 Kadabra，再把 Bench
  Abra 进化成 Kadabra，但它没有 Energy。Active 后续被击倒，接班失败。
- Poké Pad `1152` 的官方卡面是“从牌库检索一张没有 Rule Box 的 Pokémon”；Kadabra
  `742` 符合条件。原始 trace 随后证明该路线实际可执行：raw[77] 通过 Poké Pad 找到
  Kadabra，raw[78] 让 Bench Abra 进化。

## 策略边界

只把以下条件视为可提升到“确定 handoff”优先级：

1. 当前攻击不是最后奖赏闭环；
2. Active 已经有 Psychic，不需要依靠这张 Energy 才能完成本回合攻击；
3. Bench Abra 可接受 Telepath；
4. 手牌有 Poké Pad，且资源账本显示仍有未知的 Kadabra 来源（牌库/奖赏区），或手里
   直接有 Kadabra；
5. 当前回合没有 Budew `Itchy Pollen` 造成的 Item Lock。

不能把“手里有 Poké Pad”无条件当作成功：若资源账本已知四张 Kadabra 都在手牌、场上、
弃牌或奖赏，或者牌库没有可检索目标，就必须回到普通攻击/铺场排序。该建议只改策略，
不改固定卡组。

## 依据

- `data/official/EN_Card_Data.csv:1854`：Rare Candy 的进化时机限制。
- `data/official/EN_Card_Data.csv` 中 card `1152`：Poké Pad 的无 Rule Box Pokémon
  检索效果。
- `data/official/EN_Card_Data.csv` 中 card `19`：Telepath Energy 提供 Psychic Energy；
  其额外搜索效果由运行时处理。
