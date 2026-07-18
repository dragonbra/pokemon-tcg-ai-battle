# Kaggle 公开 Alakazam agent 调研参考

- 调研日期：2026-07-19（北京时间）
- 比赛：The Pokémon Company - PTCG AI Battle Challenge Simulation
- 公开 notebook：<https://www.kaggle.com/code/tientrum/search-augmented-heuristic-agent-alakazam>
- 作者：Tien N.
- Kaggle ref：`tientrum/search-augmented-heuristic-agent-alakazam`
- 当前公开版本：Version 26 of 26
- Kaggle notebook ID：`124138025`
- notebook 原文件：[`notebooks/tientrum-search-augmented-heuristic-agent-alakazam-v26.ipynb`](notebooks/tientrum-search-augmented-heuristic-agent-alakazam-v26.ipynb)

## 结论

有，而且目前公开代码中发现了一套明确值得作为 Alakazam 参考基线的 agent：Tien N. 的 **Search-Augmented Heuristic Agent (Alakazam)**。

作者在 notebook 中声称，这个较早、已经被后续版本取代的 checkpoint 曾在真实 ladder submission 上达到 **1034.6 Elo**（2026-07-05 记录）。作者没有提供当时的精确名次，但 notebook 写明当时 top 100 cutoff 约为 980–1000，因此这套 Alakazam 曾经达到相当有竞争力的水平。

这个数字需要谨慎解释：

1. 它是作者自报的历史 checkpoint 结果，不是当前实时 leaderboard 结果。
2. 该 notebook 明确说它不是作者当前 live submission，而是旧版本公开分享。
3. notebook 页面目前显示的是公开代码的可运行版本，不等于我们能直接验证当时完整 ladder 样本、对手分布和 Elo 计算过程。
4. 但从代码完整度、策略细节和公开成绩描述看，它已经不是普通的 starter/demo agent，而是目前值得直接复用和对比的 Alakazam reference。

## 公开页面和文件核验

通过 Kaggle 公开 competition Code 页面检索到与 Alakazam 直接相关的 notebook：

- 标题：`Search-Augmented Heuristic Agent (Alakazam)`
- 页面显示：5 天前更新、307 views、3 votes、6 copies
- 最近一次公开运行：约 21 秒，状态 successful
- 输入 competition：`pokemon-tcg-ai-battle`
- 输入 strategy competition：`pokemon-tcg-ai-battle-challenge-strategy`
- notebook 是公开资源，非 GPU、非 TPU、无 Internet 依赖
- 公开 pull 接口返回 Version 26 notebook 原文，已保存到本目录

本次调研重点是 competition Code 页面和公开 notebook 内容；没有使用账户凭据，也没有提交、复制发布或执行 Kaggle 账户写操作。

## 卡组结构

公开 notebook 的 60 张卡如下。它和当前仓库 V2/V3 的核心卡组相似，但不是同一套 60 张构筑：

| 数量 | 卡牌 | Card ID |
|---:|---|---:|
| 2 | Basic Psychic Energy | `5` |
| 4 | Telepath Psychic Energy | `19` |
| 2 | Dunsparce | `65` |
| 4 | Dudunsparce | `66` |
| 1 | Card ID 305 | `305` |
| 4 | Abra | `741` |
| 4 | Kadabra | `742` |
| 4 | Alakazam | `743` |
| 4 | Rare Candy | `1079` |
| 3 | Enhanced Hammer | `1081` |
| 4 | Buddy-Buddy Poffin | `1086` |
| 3 | Night Stretcher | `1097` |
| 1 | Sacred Ash | `1129` |
| 4 | Poké Pad | `1152` |
| 3 | Boss’s Orders | `1182` |
| 1 | Lana’s Aid | `1184` |
| 2 | Xerosic | `1197` |
| 4 | Hilda | `1225` |
| 1 | Lillie’s Determination | `1227` |
| 4 | Dawn | `1231` |
| 1 | Neutralization Zone | `1247` |

关键结构差异：

- 4 Alakazam、4 Kadabra、4 Abra、4 Dudunsparce、4 Rare Candy、4 Poffin、4 Hilda、4 Dawn、4 Poké Pad。
- 只有 1 张 `Dunsparce` (`305`)；当前仓库 V2/V3 是 3 张。
- 3 张 `Enhanced Hammer`；当前仓库是 2 张。
- 3 张 `Night Stretcher`；当前仓库是 1 张。
- 1 张 `Lana’s Aid` (`1184`)，与当前仓库 V2/V3 相同。
- 使用了 `Dunsparce` ID `65` 两张（TEF 128，60 HP，`Gnaw`/`Dig`），以及 1 张 `Dunsparce` ID `305`（JTG 120，70 HP，`Trading Places`/`Ram`）；当前仓库 V2/V3 只使用 ID `305`。这是一个必须核对官方规则和策略意图的实现细节。
- 使用 `Xerosic`、`Lillie’s Determination`、`Neutralization Zone` 等当前 V2/V3 没有的卡。
- 没有仓库 V2/V3 使用的 `Fezandipiti ex` (`140`)、`Battle Cage` (`1264`)、`Wondrous Patch` (`1146`)。
- 使用 4 张 Telepath Psychic Energy 和 2 张 Basic Psychic Energy，与 V2/V3 相同。

## 策略实现的主要价值

### 1. 不是简单的 card-aware priority，而是两层策略

代码主体先为每个 simulator 合法 option 计算启发式分数，再在主决策上叠加一个轻量 2-ply search：

- 只取启发式排名靠前的 top-K candidate，控制分支数。
- 对隐藏信息做若干 determinizations。
- 模拟我方一回合、对手的 greedy reply，以及后续少量动作。
- 用 Prize differential、场上 HP、能量数量和 Active 缺失等指标做 leaf evaluation。
- 只有 search 相对 heuristic 有足够明显优势时才覆盖 heuristic 首选。
- 每次决策的 search budget 是 0.8 秒。
- 默认 3 个 determinizations，最多 3 个对手分支，最多 40 个 rollout substeps。

这是最值得我们借鉴的架构结论：作者自己明确表示，search layer 带来的胜率提升比继续手调 heuristic 权重更大。

### 2. 对 Alakazam / Dunsparce 的状态建模比当前 V2 更接近我们的目标

代码显式统计：

- Active + Bench 上的 Abra/Kadabra/Alakazam 实际数量；
- Dunsparce/Dudunsparce 实际数量；
- 牌库剩余量、手牌量、Prize 数量；
- 当前目标是否能击倒；
- 最大可能手牌增量和 Powerful Hand 最大伤害；
- 是否需要 Dudunsparce 抽牌；
- 是否需要 Fezandipiti 补伤害；
- 是否需要撤退能量；
- 弃牌区可恢复的攻击线资源。

例如它使用的是数量计数，而不是把场上卡牌压成集合：

```python
abra_line_on_field = sum(field_counts[x] for x in ABRA_LINE)
dunsparce_line_on_field = dunsparce_on_field + field_counts[Dudunsparce]
```

这与我们当前 V3 文档中“至少三只 Abra 系列”的目标直接对应。

### 3. Poffin 已经有分阶段目标

公开 agent 的启发式对 Poffin 的判断不是固定拿某一种宝可梦：

- 早期：当 Abra 系列少于 3 只，或 Dunsparce 引擎还没有建立时，优先使用。
- 后期：当 Abra 系列仍不足 3 只或 Dunsparce 系列不足 2 只时，继续使用。
- 场面已经比较完整且当前存在击倒收益时，才把 Poffin 作为 fallback。

这与我们刚刚讨论的“先满足三只 Abra 系列，再主动扩大 Dunsparce”高度相似，但公开实现采用了更柔性的双条件，而不是简单硬切换。

### 4. 对牌库风险有显式 `safe_draws`

代码定义了近似的安全抽牌量：

```python
safe_draws = deck_count - my_prize_count - 1 if not can_win_this_turn else 999
```

大量抽牌、进化、Supporter 和能量动作都会检查这个变量。例如：

- Dunsparce 能力：牌库不足时禁用；只有需要抽牌或手牌不够时才使用。
- Hilda：牌库安全时才允许使用。
- Dawn：牌库安全时才允许使用，手牌不足时可以走 emergency 分支。
- Telepath Energy / Enriching Energy：牌库过低时限制使用。
- Rare Candy / Alakazam 进化：也有最小安全牌库限制。

它不完全等同于我们提出的“牌库 10 张安全线”，但提供了一个重要参考：公开高水平 agent 把牌库风险作为所有动作的共同条件，而不是只在 Dudunsparce ability 处阻止抽牌。

### 5. Rare Candy 保护和主动撤退已有部分处理

公开代码对 Kadabra 进化有类似保护逻辑：如果手里同时有 Rare Candy 和 Alakazam，某些 Kadabra 进化会被扣分，避免破坏直接 Alakazam 路线。

它也会判断：

- Active 是否已经是带 Psychic Energy 的 Alakazam；
- Bench 是否有备用攻击手；
- 是否需要为撤退补能量；
- 当前是否因为对手 Active HP 很低而优先 Kadabra finish。

不过这并不代表它完全实现了我们要求的“没有明确收益就不 Retreat”。它仍然保留了 `retreat_kadabra`、`retreat_promote` 等正向分数，因此需要逐 replay 验证，而不能直接照搬。

### 6. 恢复逻辑比当前 V2 更丰富

公开版本中：

- Night Stretcher 在弃牌区有 Abra/Kadabra/Alakazam 时优先恢复宝可梦；没有攻击线宝可梦时才考虑恢复 Psychic Energy。
- Lana’s Aid 根据弃牌区可恢复资源数量判断是否值得使用。
- Sacred Ash 根据弃牌区攻击线数量决定优先级。

这与我们 V3 当前提出的“Night Stretcher / Lana’s Aid 应服务于完整进化链，而不是固定只拿 Abra”一致。

## 与本仓库 V2/V3 的对比

| 维度 | Kaggle 公开 Alakazam reference | 本仓库当前 V2/V3 |
|---|---|---|
| 卡组 | 4 Alakazam，4/4 Abra/Kadabra，4 Dudunsparce；更多恢复和 tech | 4 Alakazam，4/4 Abra/Kadabra，3 Dudunsparce，3 Dunsparce |
| 主策略 | heuristic + 2-ply belief-sampled search | deterministic heuristic |
| Abra 计数 | 实际实例数量 | 当前核心判断部分使用集合，数量语义不足 |
| Dunsparce 计数 | 显式实际数量 | 当前 V3 还没有实现 |
| 牌库保护 | 统一 `safe_draws` 约束 | V2 只有较局部的低 deckCount 保护 |
| Telepath 顺序 | 偏好给 Abra，但仍需 replay 验证先后动作 | 当前 V3 文档已明确要求先铺 Abra，代码尚未实现 |
| Rare Candy | 通过条件和扣分保护进化路线 | V2 有基础逻辑，V3 目标更明确但未实现 |
| 恢复 | Night Stretcher/Lana/Sacred Ash 有较细分支 | V2 主要是粗粒度 recovery score |
| search | 有 0.8 秒 2-ply search | 当前没有 search layer |
| 结果证据 | 作者自报历史 1034.6 Elo | V2 当前公开样本为 1 胜 2 负、public score 611.8 |

## 对我们下一步的建议

### 建议一：先复刻“状态量”，不要先复刻整套权重

优先从这份公开代码吸收：

1. `abra_line_on_field`；
2. `dunsparce_line_on_field`；
3. `safe_draws` / deck protection；
4. `max_hand_size` 和 `max_damage`；
5. `need_dudunsparce_draw`；
6. `need_fez`；
7. `need_retreat_energy`。

这些状态量正好对应我们从 V2 replay 发现的失败模式。

### 建议二：先做 deterministic V3，再评估 search

公开 notebook 的 search layer 很有吸引力，但它依赖：

- `cg.api` 中的 `search_begin/search_step/search_end`；
- 完整的 typed observation API；
- 隐藏信息 determinization；
- 额外的时间预算和 fallback。

建议顺序是：

1. 先把 Telepath、三只 Abra、Retreat、牌库保护和恢复逻辑实现成 deterministic V3；
2. 用 replay/本地对局验证动作合法性和状态指标；
3. 再单独引入 2-ply search，避免同时改变 heuristic 和 search，无法归因。

### 建议三：重点核对 ID 65 与 ID 305

公开 deck 使用 `Dunsparce` ID `65`，而本仓库 V1/V2/V3 使用 ID `305`。两者可能是不同版本/不同卡名映射，也可能涉及官方卡池中的两张不同 Dunsparce。不能直接替换，必须查 `data/official/EN_Card_Data.csv` 确认名称、HP、技能和合法性。

### 建议四：把 notebook 当作 reference，不要当作已验证事实

它最可靠的价值是：

- 公开可读的实现；
- 明确的策略状态建模；
- 2-ply search 的工程模式；
- 一个作者声称曾达到较高 Elo 的 Alakazam checkpoint。

它不够可靠的部分是：

- 没有完整 ladder submission ID 和原始评分页面；
- 没有公开独立样本统计；
- search 的“measurable win-rate improvement”没有给出表格；
- notebook 已明确是旧版本；
- 没有证明当前版本仍能达到 1034.6 Elo。

## 调研边界

本次先检查了 Kaggle 比赛 Code 页面中公开可见的 notebooks，并对直接命名 Alakazam 的 notebook 做了深度读取。当前公开页面中发现的直接相关高质量候选是上述 1 份；其它高分 notebook 多为不同 archetype、通用 engine 或 meta/replay 分析，不应在没有读取其 deck 和策略之前被归类为 Alakazam agent。

后续如果需要更完整的 exhaustive scan，应继续分页抓取 Kaggle Code 列表，并逐个下载标题、正文或 deck 中包含 Alakazam (`743`) 的公开 notebook 做统一筛选。

## 验证命令/接口

本报告来自以下只读来源：

- Kaggle competition Code 页面：<https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/code>
- Kaggle notebook 页面：<https://www.kaggle.com/code/tientrum/search-augmented-heuristic-agent-alakazam>
- Kaggle 公开 pull endpoint：`https://www.kaggle.com/api/v1/kernels/pull/tientrum/search-augmented-heuristic-agent-alakazam`

没有使用 Kaggle token，没有发布、提交或修改 Kaggle 资源。
