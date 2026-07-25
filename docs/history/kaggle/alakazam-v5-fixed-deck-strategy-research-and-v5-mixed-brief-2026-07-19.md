# V5 固定卡组下的 Alakazam 规则策略混合调研与 v5_mixed 任务书

- 调研日期：2026-07-19（北京时间）
- 目标：保持 `submission/alakazam_v5/deck.csv`、V5 卡组内容和官方 `cg/` runtime 不变，只吸收公开 Alakazam code 中可以迁移的规则策略，形成适合 Codex 实现的 `v5_mixed` 方案。
- 相关公开 notebook 原文件：[`notebooks/`](notebooks/)
- 直接参考报告：[`alakazam-public-code-survey-2026-07-19.md`](alakazam-public-code-survey-2026-07-19.md)
- 当前 V5 官方对局复盘：[`alakazam-v3-episodes-2026-07-19.md`](alakazam-v3-episodes-2026-07-19.md)；V5 自身策略说明：`../../submission/alakazam_v5/STRATEGY.md`

## 0. 给 Codex 的一句话任务定义

在不改变 V5 卡组、`deck.csv`、`cg/`、公开 API 和确定性要求的前提下，复制 V5 为 `submission/alakazam_v5_mixed/`，只改策略代码：把 V5 的进化/Telepath/Mist/Fezandipiti 安全规则，与公开高水平 Alakazam agent 的“攻击连续性状态、目标驱动抽牌、牌库安全、特殊能量处理、恢复链和可验证指标”混合起来；不要直接复制 notebook 的整套权重，也不要引入 search、随机性或未验证卡牌假设。

## 1. 最终判断

公开代码对 V5 最有价值的不是换卡，而是三个层次的策略思想：

1. **把“建立攻击线”改成可计数、可预测的连续性状态。**
   公开 Best: 5th、Naoto mirror-speed、Tien search agent 都显式统计场上 Abra/Kadabra/Alakazam，而不是只判断是否存在某一类卡。它们还分别统计 Dunsparce/Dudunsparce 引擎。V5 已经有 `attack_line_count`，但仍把“数量达到三只”当作很强的单一门槛，没有把“当前攻击、下回合攻击、第三只攻击线、牌库循环线”拆开。

2. **把抽牌当作实现目标的工具，而不是默认优先动作。**
   公开 agent 会先估算当前/潜在手牌和 Powerful Hand 伤害，再决定 Dudunsparce、Kadabra、Alakazam、Dawn/Hilda 是否值得消耗牌库。V5 已经有 20 张手牌和 10 张牌库保护，但 `draw_is_blocked` 仍然偏硬，且没有把“当前击倒、下一只打手、恢复后续攻击”区分成不同目标。

3. **把一次动作是否改变 Prize race 作为统一判断。**
   Poffin、Hilda、Rare Candy、Telepath、Fezandipiti、Boss、Xerosic、Retreat 都应回答同一个问题：这一步是否使本回合或下回合的有效攻击更可靠，是否保护了奖赏卡交换，是否只是因为选项存在而执行。

因此，`v5_mixed` 不应该是“把公开 agent 的分数抄进 V5”，而应是：

```text
V5 的已验证规则
+ 公开 agent 的显式状态模型
+ 公开 agent 的目标驱动资源管理
+ replay 可观测的动作指标
```

## 2. 研究材料和可信度分层

### A. Ryotasueyoshi：Best: 5th rule-based agent

URL：<https://www.kaggle.com/code/ryotasueyoshi/rule-based-not-psychic-alakazam-best-5th>

作者称该版本在比赛第二天达到 leaderboard 第 5，并公开完整 `deck.csv` 与 `main.py`。它是最适合研究确定性规则优先级的 reference，但作者也明确说自然语言 playing principles 没有完全准确地反映在代码中。

值得迁移的规则：

- 首回合优先 Abra + Dunsparce，再补 Abra；
- Telepath 用于补 Psychic Basic；
- 追踪场上、手牌、牌库中的剩余进化线；
- Powerful Hand 先算当前和最大伤害，再决定抽牌；
- Fezandipiti 只有在能改变击倒结果时才使用；
- 对手带 Mist/Rock Fighting 等特殊能量时，先判断 Hammer 是否足够；
- 牌库不要低于剩余 Prize 数量，除非本回合可以结束比赛；
- Dunsparce 抽牌在不改变伤害/攻击结果时停止。

### B. Tien N.：Search-Augmented Heuristic Agent

URL：<https://www.kaggle.com/code/tientrum/search-augmented-heuristic-agent-alakazam>

作者自报旧 checkpoint 1034.6 Elo；该数字不是独立验证的实时 benchmark。代码价值主要在：

- 显式统计 `abra_line_on_field`、`dunsparce_line_on_field`；
- 估算 `max_hand_size`、`max_damage`；
- 计算 `need_dudunsparce_draw`、`need_fez`、`need_retreat_energy`；
- 用 `safe_draws` 约束几乎所有抽牌和进化动作；
- heuristic 之上再加 2-ply search。

`v5_mixed` 第一版不引入它的 search layer。search 依赖 typed API、隐藏信息 determinization 和时间预算，应该作为后续独立实验，不与本轮规则混合。

### C. Naoto714：Alakazam Mirror Setup Speed

URL：<https://www.kaggle.com/code/naoto714/alakazam-mirror-setup-speed-en>

作者用 600 局 self-play 和 34 局 live 样本论证 mirror 更像 setup-speed race：达到 7 张手牌后，Alakazam 已经足以互相击倒，关键是首只 Alakazam 的落地和首次攻击回合。

对 V5 的启发：

- 不能把“手牌越大越好”当成永远成立；
- `first_alakazam_turn`、`first_effective_attack_turn`、`first_ko_turn` 应成为策略指标；
- 当前已能击倒时，额外抽牌通常没有价值；
- 4 Alakazam/4 Rare Candy 是卡组实验结论，不能迁移到本任务，因为 V5 卡组禁止改变；但“直接攻击线优先于非 combo tech”这一策略思想可以迁移。

### D. Naoto714：Alakazam No-Tech Pivot

URL：<https://www.kaggle.com/code/naoto714/alakazam-no-tech-pivot-ja>

作者记录 submission `53966602` 的 publicScore `908.2`，并报告可确认 public 6 局 5 胜 1 负。它的价值不是绝对成绩，而是展示了固定主线下根据 meta 改变 tech 的方法。V5_mixed 不改变卡组，因此只吸收：

- 用目标对手的明确状态决定 Boss/Hammer/Xerosic 是否值得；
- 不让 matchup tech 破坏主线攻击连续性；
- 通过 crash-safe、合法选项、确定性排序维持稳定。

### E. Heiseimikiko：Why Alakazam is a Good Baseline for AI

URL：<https://www.kaggle.com/code/heiseimikiko/why-alakazam-is-a-good-baseline-for-ai>

这是策略说明，不是完整 agent。作者强调 Alakazam 的核心是：进化线、能量、后续打手、手牌到伤害的转换。它支持我们把 V5_mixed 定义为“攻击连续性和资源时序策略”，而不是继续堆卡牌特例。

## 3. V5 当前已经做对的部分

V5 不是从零开始；以下规则已经吸收了不少公开研究，`v5_mixed` 不应破坏：

### 3.1 进化时序

- 首回合不为不能进化的 Kadabra 浪费 Poké Pad；
- Active Abra 有合法 Rare Candy + Alakazam 直通路线时，Bench Abra 的 Kadabra 进化优先保留 Active 攻击机会；
- 第三回合以后，Active Kadabra 有能量和 Alakazam 时优先自然进化；
- Rare Candy 不是每回合固定使用的动作。

### 3.2 Telepath 和能量上限

- Telepath 贴到 Psychic Pokémon 才触发搜索；
- 有手牌 Abra 且场上没有 Psychic 目标时，先铺 Abra 再贴 Telepath；
- Abra/Kadabra/Alakazam 默认每只只保留一张 Psychic Energy；
- Active 非攻击手、Bench 有准备好的 Alakazam 时，才允许为明确的攻击交接支付 Retreat。

### 3.3 当前 runtime 的 Mist 事实

V5 已根据 V3 replay 和官方 engine 行为将 Alakazam 对带 Mist Energy 目标的预期伤害视为 0。这是本仓库的运行时事实，不应被公开 notebook 的一般规则覆盖。

### 3.4 Fezandipiti 的两奖风险

V5 已经开始限制 Fezandipiti：上一回合己方被击倒、当前没有高价值 Alakazam 路线且手牌较少时才考虑，并在有普通 Bench、对手剩两张 Prize 时避免无条件下场。这直接回应了 V3 public Episode `86732591` 的失败。

## 4. V5 当前仍可改进的地方

### 4.1 “三只 Abra 系列”还不是完整攻击连续性

V5 的：

```python
attack_line_count = _attack_line_count(player)
attack_line_target_reached = attack_line_count >= 3
setup_needed = not attack_line_target_reached
```

有两个问题：

1. 三只数量相同，不代表价值相同。
   - `Abra + Abra + Abra`：资源需求大，短期攻击弱；
   - `Kadabra + Abra + Abra`：有自然进化通道；
   - `Alakazam(active) + Kadabra(bench) + Abra(bench)`：攻击连续性明显更高。

2. 三只数量达到后，`setup_needed=False` 太早结束了攻击线建设。
   - 还需要区分下一只是否带能量；
   - 是否有可用 Alakazam/Kadabra/Abra 进化牌；
   - 是否有恢复资源；
   - 是否只是有三只未充能 Abra，但没有实际交接能力。

`v5_mixed` 应保留“三只实际数量”的硬事实，但另外维护攻击线质量：

```text
attackers_ready_now
attackers_ready_next_turn
attackers_future_count
unenergized_attack_line_count
recovery_attack_count
```

### 4.2 Dunsparce 引擎状态没有单独维护

公开 agent 把 Abra 线和 Dunsparce 线分开统计。V5 只用 `_can_use_enriching_for_draw()` 判断一次 Enriching Energy 是否可以给 Dudunsparce，缺少：

```text
dunsparce_basic_count
active_dudunsparce_count
available_run_away_draw_count
```

建议不要把 Dunsparce 简单提升到攻击线同等优先级，而是只在以下条件成立时建立/使用：

- 当前没有本回合必需的进化/能量/击倒动作；
- 当前攻击已经安全，或下回合攻击已经有明确路径；
- 手牌或伤害仍然不足；
- 牌库大于安全线；
- 使用一次抽牌不会让下一次攻击线断裂。

### 4.3 抽牌保护需要从“硬阈值”升级为“目标驱动”

V5 当前：

```python
hand_count > 20 or deck_count <= 10 or _can_knockout(...)
```

直接阻止抽牌。它比 V2 更安全，但仍缺少三个区分：

- 当前攻击已经 lethal；
- 抽牌可以在本回合形成 lethal；
- 抽牌不能本回合 lethal，但可以保住下回合攻击。

公开 Best: 5th 使用“牌库不能低于剩余 Prize”的安全思想；Tien 使用 `safe_draws`；Naoto 使用“7 张手牌已足以击倒 Alakazam”的 mirror 阈值。综合后，V5_mixed 应使用以下优先级：

```text
P0：当前已有有效击倒
    → 立即攻击；不再抽牌、进化或使用非必要 Trainer。

P1：当前伤害不足，但一次明确的 draw/search/进化可以形成本回合击倒
    → 允许该动作；计算动作的实际 hand_gain 和 deck_cost。

P2：本回合不能击倒，但当前攻击手会被击倒或下回合没有攻击手
    → 只做建立下一只攻击手所必需的动作。

P3：本回合和下回合都安全，且牌库高于安全线
    → 才考虑 Dudunsparce / Psychic Draw / Hilda 的额外价值。
```

手牌 20 和牌库 10 应作为保护线，不应独立取代 P0/P1/P2/P3 判断。任何允许低牌库继续抽牌的例外必须满足“动作后可以在牌库耗尽前完成对局”。

### 4.4 Fezandipiti 的判断需要显式区分“能抽牌”和“值得暴露”

公开 Best: 5th 的 Fez 规则强调：如果不需要它就不使用。V5 已经有安全判断，但 `_fezandipiti_needed()` 仍主要依赖上一回合 KO、当前能否 KO 和手牌数量，建议增加：

```text
opponent_prizes
opponent_has_visible_boss_risk
ordinary_bench_available
current_or_next_attack_is_safe
fez_hand_gain_is_necessary_for_ko
```

尤其注意对手剩 2 Prize 时：如果 Fez 下场不会直接让本回合获得 Prize，且普通 Bench 已经存在，应强烈禁止放下。

### 4.5 Xerosic 需要变成“资源交换判断”，而不是单一手牌门槛

V5 文档规定对手手牌至少 4 张时考虑 Xerosic。公开代码和 no-tech pivot 的启发是：Xerosic 是为了压低对手 Alakazam 的手牌伤害，或破坏对手下一回合 setup，不是“对手手牌 ≥4 就自动使用”。

建议 Xerosic 只在以下至少一个成立时前置：

- 对手 Active Alakazam 的 Powerful Hand 伤害可以因为手牌减少而从 lethal 降为 non-lethal；
- 对手手牌明显包含下一回合进化/能量/攻击的连续性资源，且当前我方没有更高优先动作；
- 对手是 Alakazam mirror，且我方下一次交换落后，Xerosic 能改变 Prize race。

否则保留 Xerosic，避免把 Supporter 轮次用于没有可证明收益的手牌减少。

### 4.6 恢复选择要看“攻击链缺口”，不只是卡牌类型

V5 已经在没有 Abra 时优先恢复 Abra，Lana's Aid 尽量多取回。但公开 agent 的可迁移思想更进一步：恢复目标应根据下一次合法攻击路径选择。

建议恢复评分顺序：

1. 如果场上没有 Abra 系列：Abra 是硬前置；
2. 如果 Active/Bench 有可进化 Abra、手里有 Alakazam/Rare Candy：恢复缺的中间件或 Alakazam；
3. 如果已有完整宝可梦但缺 Psychic Energy：比较恢复能量与恢复宝可梦；
4. 如果本回合无法攻击、但下回合可以攻击：取回下回合最短路径；
5. Lana's Aid 在合法范围内取回多张，但不要为了“取满”牺牲恢复后仍然无法形成攻击的牌库或手牌安全。

## 5. 建议的 v5_mixed 状态模型

Codex 不需要一次性重写所有分支，先在 `_main_action()` 前计算一个只读策略状态对象或一组局部变量。建议字段如下：

```text
turn_number
hand_count
deck_count
prize_count
opponent_prize_count

abra_line_count
abra_count
kadabra_count
alakazam_count
ready_active_attack
ready_bench_attack
future_attack_line_count
unenergized_attack_line_count
recovery_attack_line_count

has_current_ko
has_ko_after_one_draw
current_target_is_mist_blocked
has_non_mist_ko_target

has_dunsparce
has_dudunsparce
can_run_away_draw
can_use_enriching_for_draw

previous_turn_had_ko
fez_draw_is_required
fez_exposure_is_forbidden
xerosic_changes_opponent_attack
boss_changes_prize_race
retreat_changes_attack
```

这些字段应该由 observation 直接推导，不要引入随机隐藏信息，不要假定对手手牌内容。

## 6. 建议的统一优先级

### 第一层：必胜/立即兑现

1. 合法且有效的击倒；
2. 一次明确的进化、能量、Hilda 或搜索可以在本回合形成击倒；
3. Trading Places 能把已充能 Alakazam 换到 Active 并立即攻击；
4. Mist 阻挡时，先使用能解除阻挡的合法动作，或寻找无 Mist 的有效目标。

### 第二层：保持攻击连续性

5. Active Kadabra → Alakazam 自然进化；
6. Active Abra 的 Rare Candy + Alakazam 直通；
7. 保护 Active 直通路线，把 Kadabra 进化到 Bench Abra；
8. 为下一只攻击手贴第一张 Psychic Energy；
9. 没有 Abra 时用恢复资源先补 Abra，不要先拿无前置意义的 Alakazam。

### 第三层：建立抽牌引擎

10. 三只 Abra 系列数量未达到时，Poffin/Telepath 优先补攻击线；
11. 攻击线满足本回合和下回合后，Poffin/Hilda 再补 Dunsparce/Dudunsparce；
12. Dudunsparce 只有在 hand_gain 对当前/下回合目标有意义且牌库安全时使用；
13. Enriching Energy 给 Dudunsparce 只作为额外循环，不得抢占下一只打手的能量。

### 第四层：辅助、压制与恢复

14. Fezandipiti 只有伤害/攻击必要且暴露风险可接受时使用；
15. Xerosic 只有能改变对手攻击或 Prize race 时使用；
16. Boss 只拉出可击倒或 Prize 价值更高的目标；
17. Night Stretcher/Lana/Sacred Ash 按攻击链缺口恢复；
18. 无明确收益时跳过 Retreat、非致命攻击和额外抽牌。

## 7. V5 与公开策略的保留/吸收/拒绝清单

| 规则 | 处理 | 原因 |
|---|---|---|
| Abra/Kadabra/Alakazam 实例计数 | 吸收并加强 | 与 V5“三只攻击线”一致，但需要质量和下一回合状态 |
| Dunsparce/Dudunsparce 单独计数 | 吸收 | 公开多个高水平 agent 都使用 |
| 首只 Alakazam 尽早落地 | 保留并量化 | mirror-speed 数据支持，但不能破坏 Bench 备用手 |
| 4 Alakazam / 4 Rare Candy | 拒绝卡组变更 | 本任务明确固定 V5 卡组 |
| `safe_draws` / 牌库不低于 Prize | 吸收思想 | 比固定 10 张更有上下文，但仍保留 V5 10 张硬保护 |
| Telepath 先铺 Abra | 保留 V5 | 已有明确逻辑，增加回归验证即可 |
| Mist 视为当前 runtime 的 0 伤害 | 保留 V5 | 有官方 replay 和 engine source 证据 |
| Fez 必须补伤害才下场 | 加强 | V3 已出现两奖暴露失败 |
| Xerosic 压低对手手牌 | 加强 | 只在能改变对手攻击/交换时使用 |
| Retreat 只为攻击交接 | 保留 V5 | 用户明确要求，公开 agent 的宽松 retreat 不应覆盖 |
| 2-ply search | 暂不吸收 | 另立实验，避免规则改动无法归因 |
| 随机 tie-break | 拒绝 | V5 要求确定性，且随机会增加 replay 归因难度 |
| 公开 notebook 自带卡组 | 拒绝 | 不改变 V5 deck.csv |
| Dunsparce ID 65 | 暂不吸收 | V5 使用 ID 305；两者是不同卡，不能仅凭公开 agent 使用就替换 |

## 8. Codex 实现边界

### 必须保持不变

- `submission/alakazam_v5/deck.csv` 的 60 张卡；
- V5 原目录中的 `cg/` 和 simulator runtime；
- V5 对 Mist Energy 的 runtime 特殊处理；
- 只从当前 simulator option 中返回合法 index；
- 确定性排序；
- 不修改 `submission/alakazam_v5/` 原文件；
- 不引入 Kaggle notebook 的未验证依赖、网络访问、随机性或 search API。

### 允许新增/修改

- 新建 `submission/alakazam_v5_mixed/`，复制 V5 的完整提交结构；
- 修改 `v5_mixed/main.py` 的策略逻辑；
- 新增 `v5_mixed/README.md` 和 `v5_mixed/STRATEGY.md`；
- 新增只读策略单元 probe 或放在 `/tmp` 的验证脚本；
- 新增报告/实验记录，不改变原 V5 卡组。

### 不要做

- 不要直接复制某个 notebook 的 3 万行 `main.py`；
- 不要在第一版加入 2-ply search；
- 不要把公开作者的 Elo、top 3/top 5 或 publicScore 当作本地已验证结果；
- 不要把“手牌 >20”简单改成永久禁止所有抽牌；
- 不要把“场上有三只 Abra 系列”简单改成永久禁止所有 Dunsparce；
- 不要把 `set` 继续用于需要实例数量的判断；
- 不要为了测试在 repo 生成大体积 replay，应写到 `/tmp`。

## 9. 建议的实现顺序

### Phase 1：策略状态和最小高置信修正

1. 复制 V5 到 `v5_mixed`，确认 deck hash 与 V5 完全一致；
2. 引入独立的 attack continuity 状态计算；
3. 引入 Dunsparce engine 状态计算；
4. 把 `setup_needed` 从单一布尔值改为由当前/下一回合攻击安全和未来攻击手缺口决定；
5. 不改变现有 Telepath、Rare Candy、Mist、Retreat、Fez 安全规则。

### Phase 2：统一 draw gate

6. 实现 `draw_is_required_for_current_ko`；
7. 实现 `draw_is_required_for_next_attack`；
8. 实现 `draw_is_safe`，结合 `hand_count`、`deck_count`、`prize_count`；
9. 将 Dudunsparce、Kadabra、Alakazam、Dawn、Hilda、Fezandipiti 的抽牌判断统一接入；
10. 增加当前已有 KO 时不继续抽牌的回归 probe。

### Phase 3：资源动作

11. 重新定义 Poffin：攻击线不足时 Abra，攻击线安全后 Dunsparce；
12. 重新定义 Hilda：先补当前/下回合攻击缺口，再在安全时补 Dudunsparce/Enriching 线；
13. 强化 Xerosic 的对手手牌/攻击伤害收益判断；
14. 强化 Night Stretcher/Lana 的攻击链恢复目标选择；
15. 确认 Fez 暴露条件与对手剩余 Prize 的交互。

### Phase 4：可观测验证

至少记录以下指标：

- `first_alakazam_turn`；
- `first_effective_attack_turn`；
- `first_ko_turn`；
- 每回合 Abra/Kadabra/Alakazam 数量；
- 每回合准备好的 Active/Bench 攻击手数量；
- 每回合 Dunsparce/Dudunsparce 数量；
- 每次抽牌前后 hand/deck count；
- 抽牌动作是否直接改变 KO；
- Mist 目标上的 0 伤害次数；
- Fezandipiti 下场时我方/对手 Prize；
- Xerosic 是否改变对手可估计的 Powerful Hand lethal；
- Retreat/Trading Places 是否确实带来当回合 Alakazam 攻击。

## 10. 验收标准

Codex 完成后，至少满足：

```bash
python3 scripts/check_assets.py
python3 -m compileall -q scripts submission
bash scripts/package_submission.sh alakazam_v5_mixed
```

另外需要通过静态/策略 probe：

1. V5 与 v5_mixed 的 `deck.csv` 字节或 SHA-256 完全一致；
2. V5 原目录没有被修改；
3. 首回合有 Poké Pad 时，不搜索不能进化的 Kadabra；
4. 场上没有 Psychic Pokémon、手里有 Abra + Telepath 时，先铺 Abra；
5. Active Abra 有直通路线且 Bench 有 Abra 时，Kadabra 优先进化 Bench；
6. Active Kadabra 有能量和 Alakazam 时，自然进化优先于浪费 Rare Candy；
7. 当前 Alakazam 已能有效击倒时，不触发额外 Dudunsparce/Psychic Draw；
8. 牌库 ≤10 且抽牌不能直接结束比赛时，不继续非必要过牌；
9. Mist 阻挡当前 Alakazam 时，不重复执行同一无效攻击；
10. 对手剩 2 Prize、有普通 Bench 且不需要 Fez 形成击倒时，不放下 Fezandipiti；
11. 无明确攻击交接收益时不 Retreat；
12. 所有返回值都来自 simulator 当前 options。

## 11. 需要 Codex 回报的内容

Codex 不应只回报“完成修改”。应明确返回：

- 修改文件的绝对路径；
- V5 与 v5_mixed 的 deck hash；
- 各项 probe 的输入状态和选择结果；
- 运行过的命令和真实输出；
- 哪些规则已实现，哪些因 simulator observation 不足而保留为 fallback；
- 是否运行本地 battle；若运行，输出必须在 `/tmp`；
- 不要声称 Kaggle 成绩提升，除非有新的官方 submission/replay 证据。

## 12. 当前最重要的研究假设

本报告不主张公开 agent 的所有规则都正确。当前最值得验证的三个假设是：

1. **V5 的主要剩余问题不是“有没有三只 Abra”，而是三只之后能否把攻击手连续性交接下去。**
2. **抽牌动作的正确价值不是“增加手牌”，而是让当前或下一回合的 Prize race 更可靠。**
3. **在固定 V5 卡组下，公开高分 agent 最可能带来收益的地方是状态分解和动作 gate，而不是简单调整单卡权重。**

如果这三个假设得到局部 probe 和 replay 支持，`v5_mixed` 才值得继续进入新的 Kaggle submission；否则应回到具体失败 episode，而不是继续扩展规则数量。
