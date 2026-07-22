# Kaggle 公开 Alakazam code 扩展调研

- 调研日期：2026-07-19（北京时间）
- 比赛 Code 页面：https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/code
- 检索方式：Kaggle Code 页面内搜索 `alakazam`，然后使用 Kaggle 公开 pull endpoint 获取 notebook 原文。
- 本轮没有使用 Kaggle token，没有发布、提交、复制发布或修改 Kaggle 资源。

## 总结结论

本轮找到并整合了 5 份直接与 Alakazam 相关的公开 notebook，此外确认其中有 3 份是同一套/同一作者代码的不同载体或改版，不能按 5 个独立 agent 直接计数。

最有价值的独立参考大致分成三类：

1. **高成绩 rule-based 完整 agent**：Ryotasueyoshi 的 “Best: 5th” Alakazam。
2. **高成绩/对局归因与 deck evolution**：Naoto714 的 mirror setup-speed 实验，以及 no-tech pivot 版本。
3. **策略框架说明**：Tien N. 的 search-augmented heuristic agent、Heiseimikiko 的 Alakazam baseline 说明。

其中能直接拿来和当前仓库 V2/V3 做代码级对比的，优先级如下：

1. Ryotasueyoshi `rule-based-not-psychic-alakazam-best-5th`：完整规则 agent，曾位于 leaderboard 第 5。
2. Tien N. `search-augmented-heuristic-agent-alakazam`：完整 heuristic + 2-ply search agent，作者自报旧 checkpoint 1034.6 Elo（上一份报告已保存）。
3. Naoto714 `alakazam-mirror-setup-speed-en`：完整提交代码和明确 deck evolution 实验，包含 600 局 self-play、34 局 live 样本的报告。
4. Naoto714 `alakazam-no-tech-pivot-ja`：完整提交代码，报告 submission `53966602` 的 publicScore `908.2` 和公开 6 局 5 胜 1 负。
5. Heiseimikiko `why-alakazam-is-a-good-baseline-for-ai`：不是完整 agent，而是 top 3 经验和策略抽象。
6. Ryotasueyoshi `alakazam-deck-best-5th-place`：主要是 deck image renderer，但包含第 5 名版本的 60 张卡表；属于辅助资料，不是独立策略 agent。

## 逐份分析

### 🤖 Rule-based, not psychic: Alakazam (Best: 5th)
- URL: https://www.kaggle.com/code/ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th
- 作者：sue124；版本：4；notebook ID：123851559
- 60-card code：是
- deck：Basic Psychic Energy ×2，Enriching Energy ×1，Telepath Psychic Energy ×4，Dudunsparce ×2，Fezandipiti ex ×1，Genesect ×1，Dunsparce ×3，Shaymin ×1，Abra ×4，Kadabra ×4，Alakazam ×3，Psyduck ×1，Rare Candy ×3，Enhanced Hammer ×3，Buddy-Buddy Poffin ×4，Night Stretcher ×1，Sacred Ash ×1，Poké Pad ×4，Lucky Helmet ×3，Boss's Orders ×2，Hilda ×4，Dawn ×4，Battle Cage ×4

### Alakazam Deck (Best: 5th place)
- URL: https://www.kaggle.com/code/ryotasueyoshi/alakazam-deck-best-5th-place
- 作者：sue124；版本：1；notebook ID：123691735
- 60-card code：是
- deck：Basic Psychic Energy ×2，Enriching Energy ×1，Telepath Psychic Energy ×4，Dudunsparce ×2，Fezandipiti ex ×1，Genesect ×1，Dunsparce ×3，Shaymin ×1，Abra ×4，Kadabra ×4，Alakazam ×3，Psyduck ×1，Rare Candy ×3，Enhanced Hammer ×3，Buddy-Buddy Poffin ×4，Night Stretcher ×1，Sacred Ash ×1，Poké Pad ×4，Lucky Helmet ×3，Boss's Orders ×2，Hilda ×4，Dawn ×4，Battle Cage ×4

### Why Alakazam is a Good Baseline for AI
- URL: https://www.kaggle.com/code/heiseimikiko/why-alakazam-is-a-good-baseline-for-ai
- 作者：Heisei；版本：2；notebook ID：123787825
- 60-card code：否/未找到
- deck：无完整 deck

### Alakazam Mirror Setup Speed EN
- URL: https://www.kaggle.com/code/naoto714/alakazam-mirror-setup-speed-en
- 作者：Naoto；版本：2；notebook ID：124705390
- 60-card code：是
- deck：Basic Psychic Energy ×3，Enriching Energy ×1，Telepath Psychic Energy ×4，Dudunsparce ×2，Fezandipiti ex ×1，Dunsparce ×3，Abra ×4，Kadabra ×4，Alakazam ×4，Rare Candy ×4，Enhanced Hammer ×3，Buddy-Buddy Poffin ×4，Night Stretcher ×1，Sacred Ash ×1，Poké Pad ×4，Lucky Helmet ×2，Boss's Orders ×3，Hilda ×4，Dawn ×4，Battle Cage ×4

### Alakazam No Tech Pivot JA
- URL: https://www.kaggle.com/code/naoto714/alakazam-no-tech-pivot-ja
- 作者：Naoto；版本：1；notebook ID：124503192
- 60-card code：是
- deck：Basic Psychic Energy ×3，Enriching Energy ×1，Telepath Psychic Energy ×4，Dudunsparce ×2，Fezandipiti ex ×1，Genesect ×1，Dunsparce ×3，Abra ×4，Kadabra ×4，Alakazam ×3，Rare Candy ×3，Enhanced Hammer ×3，Buddy-Buddy Poffin ×4，Night Stretcher ×1，Sacred Ash ×1，Poké Pad ×4，Lucky Helmet ×3，Boss's Orders ×3，Hilda ×4，Dawn ×4，Battle Cage ×4

## 1. Ryotasueyoshi：Best: 5th rule-based agent

页面标题明确写着：`Rule-based, not psychic: Alakazam (Best: 5th)`。作者说明这是比赛第二天曾达到 leaderboard 第 5 的 Alakazam agent，并公开了完整 `deck.csv` 和 `main.py`。作者同时承认文档中的 playing principles 并没有完全准确地反映在代码里，因此不能把自然语言原则当成代码已实现行为。

核心初期计划：

- 第一回合优先在 Bench 放 1 Abra + 1 Dunsparce；
- 再放两个 Abra；
- 有条件时用 Telepath Psychic Energy 触发额外 Psychic Basic 搜索；
- 剩余 Poffin 再补 Dunsparce；
- 用 Poké Pad 找 Dudunsparce 和 Kadabra；
- 尽量保留一个 Bench 空位。

核心中后期计划：

- 对手 Active 剩余 HP ≤30 时，允许 Kadabra 用 Super Psy Bolt 收尾；
- 否则计算 Powerful Hand 的当前/最大伤害；
- 追踪场上、手牌和牌库剩余的 Abra/Kadabra/Alakazam；
- 目标带 Mist Energy 或 Rock Fighting Energy 时，先比较 Enhanced Hammer 数量，再决定是否 Boss；
- 不让牌库剩余张数低于剩余 Prize 数量；如果本回合能拿完所有 Prize，则允许牌库到 0；
- Fezandipiti ex 不是无条件下场，而是只有补伤害能形成击倒时才使用；
- Dunsparce/Dudunsparce 抽牌在不需要增加伤害时主动停止。

卡组：

- 4 Abra、4 Kadabra、3 Alakazam、3 Dunsparce、2 Dudunsparce；
- 1 Fezandipiti ex、1 Genesect、1 Psyduck、1 Shaymin；
- 4 Poké Pad、4 Poffin、3 Rare Candy、1 Night Stretcher、1 Sacred Ash；
- 3 Lucky Helmet、3 Enhanced Hammer、2 Boss's Orders、4 Dawn、4 Hilda、4 Battle Cage；
- 4 Telepath Energy、2 Basic Psychic、1 Enriching Energy。

这个版本与当前仓库 V2/V3 的差异很有意思：

- 它只有 3 张 Alakazam，而不是我们当前的 4 张；
- 只有 2 张 Dudunsparce，而我们的 V2/V3 是 3 张；
- 加入了 3 张 Lucky Helmet、1 张 Genesect；
- 使用 1 张 Night Stretcher 和 1 张 Sacred Ash；
- 使用 Fezandipiti、Psyduck、Shaymin 三种 matchup tech；
- 4 张 Battle Cage 与我们的 V2/V3 相同。

最值得吸收的不是具体 card count，而是它把以下东西联动起来：

```text
开局铺场目标
→ Telepath 额外搜索
→ Dunsparce / Dudunsparce 抽牌
→ Powerful Hand 伤害
→ Enhanced Hammer 对特殊能量
→ 牌库不能低于剩余 Prize
```

这是一条比当前 V2 更完整的资源约束链。

## 2. Naoto714：Alakazam mirror setup-speed

这份 notebook 的核心判断是：Alakazam mirror 本质上是 setup-speed race。

作者给出的推理：

- Alakazam 140 HP；手牌达到 7 张时 Powerful Hand 已经能一击击倒 Alakazam；
- 中期两边通常都有 10+ 手牌；
- 因此 mirror 中最关键的不再是继续增加伤害，而是谁先把 Alakazam 放上场并先攻击；
- 失败样本主要表现为“慢一回合”，不是复杂操作错误。

作者做了两张牌的变化：

- Alakazam：3 → 4；
- Rare Candy：3 → 4；
- 删除 Genesect；
- Lucky Helmet：3 → 2。

报告的本地实验：

- mirror：55.0% → 60.5%；
- Lucario：51.9% → 57.4%；
- Iono：38.6% → 46.8%；
- meta-weighted：62.2% → 66.0%；
- Dragapult：45.2% → 44.4%。

作者也给出了 34 局 live 样本：22–12，总 ladder rating 940.5；Alakazam mirror 4–3（57%）。但这些数据仍然是作者报告的有限样本，不是独立验证 benchmark。

对当前项目的直接启示：

- V2 把 Psyduck 换成第 4 张 Alakazam，是有公开实验支持的方向；
- Rare Candy 第 4 张可能比额外 tech 更直接地提升 setup speed；
- 如果目标是 mirror 或 setup race，应把“首只 Alakazam turn”作为核心指标；
- 不能因为中后期手牌多就继续抽牌，超过 7 张后的收益常常已经接近 0；
- Enhanced Hammer 不宜轻易砍掉，因为对 Mist Energy / Crustle 型对手仍有约束价值。

## 3. Naoto714：no-tech pivot

这份 notebook 是另一套完整提交代码，作者明确记录：

- submission：`53966602`；
- publicScore：`908.2`（2026-06-23 20:05 JST）；
- 可确认 public episode：6 局；
- replay reward：5 胜 1 负。

它与 Best: 5th 版本的主线卡组很接近，但调整为：

- 去掉 Psyduck、Shaymin；
- 增加 Basic Psychic Energy 和 Boss's Orders；
- 保留 Fezandipiti、Genesect、Dunsparce ×3、Dudunsparce ×2、Alakazam ×3；
- 目标是 Hop 混成、Crustle/wall 派生和 baseline-like Lucario。

作者的 200 局本地比较显示：

- Hop：84.5%；
- Day-2 Crustle：93.5%；
- Abomasnow：70.0%；
- Iono：38.0%；
- Dragapult：48.0%；
- Lucario search：50.5%。

这份代码说明 Alakazam 并非只有一个固定最优构筑，而是可以根据 meta 在“tech 型”“no-tech 一致性型”“特殊能量处理型”之间切换。

## 4. Heiseimikiko：为什么 Alakazam 是好 baseline

这份不是可运行代码，但作者自述在 Alakazam 上达到 top 3，并归纳了最值得用于规则 agent 的原因：

- game plan 清晰；
- 进化链可显式建模；
- Psychic Energy 资源简单；
- 需要持续准备后续攻击手；
- Powerful Hand 把手牌资源转成可计算伤害；
- 适合作为第一套 rule-based baseline。

这和我们当前项目的研究路线相符：先把确定性资源管理做对，再考虑 search、MCTS 或学习型策略。

## 5. 两份 Ryotasueyoshi notebook 的关系

`alakazam-deck-best-5th-place` 主要是 deck image renderer：

- 它不包含完整 agent policy；
- 但包含与 Best: 5th agent 相同的 60 张 deck；
- 可以作为卡表视觉核验和卡牌 ID 复查资料。

因此它不应被当作第二个独立 Alakazam agent。

## 综合比较

| Reference | 类型 | 成绩证据 | 主要价值 |
|---|---|---|---|
| Ryotasueyoshi Best: 5th | 完整 rule-based agent | 作者称 leaderboard 第 5 | 最接近可直接复刻的高成绩简单策略 |
| Tien N. Search-Augmented | 完整 heuristic + search | 自报 1034.6 Elo 历史 checkpoint | 2-ply search、belief sampling、显式资源状态 |
| Naoto714 mirror speed | 完整 agent + 实验 | 600 self-play、34 live、940.5 | 证明 setup speed 和 4 Alakazam/4 Rare Candy 的价值 |
| Naoto714 no-tech pivot | 完整 agent + submission 记录 | publicScore 908.2、6 局 5-1 | meta-specific deck pivot、crash-safe 规则 agent |
| Heiseimikiko baseline essay | 策略说明 | 作者称 top 3 | 为什么 Alakazam 适合 rule-based 建模 |
| Ryotasueyoshi deck renderer | 卡组辅助 notebook | 无独立 agent 成绩 | 核对 Best: 5th 卡表 |

## 对我们 V3 的建议

### 优先吸收的代码结构

1. 使用实例计数，不使用 set 判断场面数量：

```python
abra_line_on_field = sum(field_counts[x] for x in ABRA_LINE)
dunsparce_line_on_field = sum(field_counts[x] for x in DUNSPARCE_LINE)
```

2. 把牌库保护放到共同状态变量：

```python
safe_draws = deck_count - prize_count - 1
```

3. 显式估算当前/潜在手牌伤害：

```python
max_damage = max_hand_size * 20
```

4. 把 `need_dudunsparce_draw`、`need_fez`、`need_retreat_energy` 作为动作评分的上游状态，而不是在单个 card branch 内各自猜测。

5. 用“首只 Alakazam 的 turn”和“首个有效 KO 的 turn”作为核心实验指标。

### 具体卡组实验矩阵

建议不要直接宣布某一套是 V3 最优，而是至少比较：

- 当前 V2/V3：4 Alakazam、3 Dunsparce、3 Dudunsparce、Fezandipiti；
- Best: 5th：3 Alakazam、3 Dunsparce、2 Dudunsparce、Genesect/Psyduck/Shaymin tech；
- Mirror speed：4 Alakazam、4 Rare Candy、去 Genesect、Lucky Helmet 减少；
- no-tech pivot：去 Psyduck/Shaymin，补 Basic Psychic 和 Boss；
- Tien reference：混合 TEF/JTG Dunsparce、4 Dudunsparce、3 Night Stretcher、Xerosic。

每套固定同一 agent policy，再固定同一卡组比较 policy，避免把 deck effect 和 policy effect 混在一起。

### 需要谨慎的地方

- 各 notebook 的成绩大多是作者自报，样本量不统一；
- 公开代码版本可能已经不是作者当前线上版本；
- Kaggle Code 页面搜索结果包含复制 notebook，不能按标题数量统计独立方案；
- 公开 notebook 中有些自然语言原则明确未完全落地到代码；
- 不能直接复制公开代码中的 `cg/` 或违反 Competition Use Only 的运行时资产约束；
- `Dunsparce` ID 65 与 305 是不同卡，必须按官方卡表分别建模。

## 已保存文件

本轮新增 notebook 原文件：

- `docs/reports/kaggle/notebooks/ryotasueyoshi-rule-based-not-psychic-alakazam-best-5th-v4.ipynb`
- `docs/reports/kaggle/notebooks/ryotasueyoshi-alakazam-deck-best-5th-place-v1.ipynb`
- `docs/reports/kaggle/notebooks/heiseimikiko-why-alakazam-is-a-good-baseline-for-ai-v2.ipynb`
- `docs/reports/kaggle/notebooks/naoto714-alakazam-mirror-setup-speed-en-v2.ipynb`
- `docs/reports/kaggle/notebooks/naoto714-alakazam-no-tech-pivot-ja-v1.ipynb`

上一轮已保存：

- `docs/reports/kaggle/notebooks/tientrum-search-augmented-heuristic-agent-alakazam-v26.ipynb`

## 来源

- https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/code
- https://www.kaggle.com/code/ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th
- https://www.kaggle.com/code/ryotasueyoshi/alakazam-deck-best-5th-place
- https://www.kaggle.com/code/heiseimikiko/why-alakazam-is-a-good-baseline-for-ai
- https://www.kaggle.com/code/naoto714/alakazam-mirror-setup-speed-en
- https://www.kaggle.com/code/naoto714/alakazam-no-tech-pivot-ja
- https://www.kaggle.com/api/v1/kernels/pull/{owner}/{slug}
