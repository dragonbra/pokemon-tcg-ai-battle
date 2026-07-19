# Alakazam AutoIter 设计约定

## 1. 目标与边界

AutoIter 的目标不是让程序盲目尝试大量动作权重，也不是让某一批随机对局的总分不断上涨。它的目标是：从具体 replay 中发现策略问题，把失败局面转化为可复盘、可回归的 case，再用最小且可解释的策略改动逐步改善整体策略。

AutoIter 采用以下固定边界：

- `deck.csv` 固定，不进行换牌实验；
- 每轮只提出一个主要策略假设，并围绕该假设修改策略；
- 规则语义和 simulator 合法性是硬边界，不能用胜率提升换取规则回退；
- 策略改动由分析者根据 trace 和规则做出，不使用不可解释的随机 mutation；
- 胜率仍然是重要的整体质量指标，但不要求每一轮都上涨。

`submission/alakazam_v7_auto_iter/` 是 V7 策略的 AutoIter 起点。后续版本应保留其固定卡组和规则语义，只在明确记录假设的前提下迭代策略。

## 2. 核心概念

### Control 与 candidate

- **control**：最近一个已接受的策略版本，作为本轮所有比较的基线；
- **candidate**：针对一个具体假设修改后的策略版本；
- candidate 不能只与自己比较，至少要和同一 control 比较合法性、case、核心节奏指标和整体结果。

### Case

Case 是一个发生在具体对局中的、可以解释的决策问题，而不是一句“这局输了”。一个有效 case 至少要能回答：

1. 行动前局面是什么；
2. simulator 当时提供了哪些合法选项；
3. control 实际选择了什么；
4. 按规则和卡组计划，应该优先选择什么；
5. 这个选择如何影响后续行动链或胜负；
6. candidate 是否已经做出更合理的选择。

Case 可能属于策略错误、规则保护过严、资源确实不可得，或者只是随机发牌无法解决。只有前两类通常适合直接改策略；后两类应记录为边界，不能为了消除它们而过度放宽策略。

## 3. 单轮迭代流程

每一轮 AutoIter 按以下顺序执行：

### 3.1 保存基线

记录 control 的版本标识、策略代码 hash、固定卡组 hash、对手列表、每个对手的局数、先后手交换规则、trace 配置和随机性配置。上一轮的已知 case 也要随本轮配置保存。

### 3.2 运行正确性检查

先验证 agent 合同、60 张卡组、runtime 导入、合法 option 选择和跨局状态清理。出现我方 action error、非法动作、必需选择缺失或跨局状态泄漏时，不进入策略效果比较。

### 3.3 运行发现批次

用当前对手 registry 的完整矩阵发现问题。发现批次可以使用较小局数，例如 17 个对手各 10 局；它的主要作用是暴露有代表性的失败模式，不要求每轮都带来胜率提升。

### 3.4 提取和分类 case

从完整 trace 中优先提取以下问题：

- 第二回合没有实际使用 Alakazam 的 `Powerful Hand`；
- Abra → Kadabra → Alakazam 进化链断裂；
- 当前 Active 被击倒后没有 ready attacker 接班；
- 能量贴给了无法及时形成攻击的目标；
- 过早攻击、过早抽牌或错误保留行动机会；
- 违反 Supporter、手填能量、Retreat、进化时钟或 Item Lock 约束；
- Dudunsparce 在空 Bench 时使用 `Run Away Draw`；
- 牌库保护导致主计划停止，或抽牌/恢复导致 deck-out。

每个 case 必须有明确的主要失败分类，不能只依靠总胜负标签。

### 3.5 提出一个策略假设

分析者先阅读行动前状态、合法 options、上一回合历史和后续结果，再写出一个最小假设。例如：

> 在 Active Alakazam 可能被击倒前，策略把能量投入到了不能及时接力的 Pokémon，导致下一次 Active 为空；本轮只调整接力打手的能量优先级。

一个 candidate 不应同时修改多个互不相关的策略主题，否则无法判断 case 改善来自哪里。

### 3.6 先做 case 回归

将旧 case 转成最小 observation/合法 option fixture，先验证 control 与 candidate 的实际动作差异。case 通过的标准是动作或行动链更符合预期，不要求这一个 case 的整局一定变成胜局。

### 3.7 运行 focused 对照

针对 case 涉及的 4–6 个对手保存完整 trace，使用相同的对手、局数和先后手交换协议比较 control 与 candidate。重点观察目标 case、第二回合攻击、进化链和接力指标，而不是只看一行总胜率。

### 3.8 运行晋级批次

如果 case 回归和 focused 对照没有发现硬性回退，再运行较大规模的完整矩阵，例如当前 registry 的 17 个对手各 30 局，或多个独立批次。晋级结果必须同时记录 case 变化、策略信号和胜率变化。

## 4. 指标体系

### 4.1 Correctness gates

以下指标是硬门槛：

- 我方 action error；
- 非法 option 或必需 selection 失败；
- 跨局状态泄漏；
- 我方异常结束或 simulator contract 失败；
- 我方 deck-out；
- 已有规则回归测试失败。

对手侧异常要单独记录，不能伪装成我方策略收益或损失；但也不能在报告中从分母中悄悄删除，必须明确说明。

### 4.2 Case resolution

每个 case 记录 control/candidate 的动作、预期动作和行动链结果。case resolution 是本轮改动是否解决目标问题的第一证据。若 candidate 没有改善目标 case，即使总胜率偶然上涨，也不能把它视为本轮策略修复成功。

### 4.3 卡组核心节奏信号

#### 第二回合 Powerful Hand

只统计我方实际选择 Alakazam 的 `Powerful Hand`（`attackId=1072`）：先手使用 engine turn 3，后手使用 engine turn 4。分母为全部对局数。

这是前期铺场和进化策略的观察信号，不是唯一优化目标。不能为了提高该数字而牺牲能量接力、Prize race 或牌库安全。

#### 击倒后的打手接力

每次记录我方 Pokémon 被击倒后的攻击线状态，并同时报告事件级和对局级指标：

- `post_ko_zero_ready_count`：击倒后 Active/Bench 没有可立即形成主要攻击的 ready attacker 的次数；
- `post_ko_zero_ready_event_rate`：该次数 / 我方 Pokémon 被击倒事件数；
- `games_with_post_ko_break_rate`：出现过至少一次打手断档的对局数 / 总对局数；
- 按被击倒成员拆分 Alakazam、Kadabra、Abra 的接力结果；
- 单独记录仍有 Abra 进化线、但尚不能立即攻击的情况。

`ready attacker` 的默认定义是 Active 或 Bench 中已经具备主要攻击所需 Psychic Energy、且按当前 simulator 状态可以形成合法攻击路径的 Abra 进化线 Pokémon。只有 Abra 但仍需进化的场面不应和已经能立即使用 `Powerful Hand` 或 `Super Psy Bolt` 的场面混为一谈。

报告可以保留“败局中有多少局属于打手断档”作为诊断摘要，但跨版本比较必须优先使用事件级分母和总对局级分母，不能只用败局数作分母。

#### 空 Bench 的 Run Away Draw

`empty_bench_run_away_draw_count` 定义为：Active 是 Dudunsparce、Bench 为空，且 agent 实际选择了 `Run Away Draw` 的次数。

这是明确的局部策略错误，晋级硬门槛为 `0`。每次触发都必须保存行动前状态、合法 options 和实际 action；禁止只把这个 action 过滤掉，却不检查替代动作是否丢失了主要游戏计划。

### 4.4 Outcome guardrails

以下指标用于检查局部修复是否损害整体计划：

- 总胜率；
- Meta 加权胜率；
- 对手分组胜率，特别是 case 来源 matchup；
- 胜局/败局/未完成/错误的完整分母；
- 主要攻击节奏与 Prize race。

胜率不要求每轮上涨，但 candidate 不能在多批独立评测中持续、明显低于 control。case 已解决但胜率连续下降时，必须回到 trace 分析是否损害了主要游戏计划；不能因为局部 case 通过就强行保留。

## 5. 随机样本与对照

candidate/control 尽量使用相同的对手集合、局数和先后手交换协议。若评测器能够控制随机 seed，可以使用相同 seed 降低对照噪声，但 seed 只用于配对比较，不能成为策略优化目标，也不能只凭一个固定 seed 晋级。

每个重要 candidate 至少需要一个不同随机性的独立批次复核。以下情况应视为疑似 overfit：

- 只在一个 seed 或一批发牌中改善；
- case 动作没有真正改变，只是 opponent 发牌变好；
- 第二回合指标上涨，但接力、Prize race 或胜率在独立批次持续下降。

## 6. Candidate 晋级与回退

### 接受为下一轮 control

candidate 同时满足以下条件时，才可以作为下一轮 control：

1. 所有 correctness gates 通过；
2. 目标 case 的动作或行动链得到改善；
3. `empty_bench_run_away_draw_count` 没有新增，且目标错误为零时才算完全修复；
4. 第二回合攻击、打手接力和进化链没有出现无法解释的核心回退；
5. 胜率和 Meta 加权胜率在独立批次中没有持续、明显低于 control。

case 已解决但整体胜率基本持平，可以先保留观察；case 已解决但整体结果在多个独立批次下降，应回退或重新设计假设。

### 拒绝或回退

出现以下任一情况时拒绝 candidate：

- 引入规则或运行时错误；
- 目标 case 没有改善；
- 通过删除合法动作、过早结束或牺牲主要攻击计划来掩盖问题；
- 空 Bench `Run Away Draw` 仍然发生；
- 打手断档或主要 outcome 在独立批次持续恶化。

## 7. Case 与评测产物

每个迭代目录至少应包含以下内容：

```text
iter-XX-name/
  manifest.json          # control/candidate、卡组、对手、局数、swap、随机性配置
  control-summary.json
  candidate-summary.json
  cases.jsonl             # 精炼 case，不直接复制全部 observation
  analysis.md             # 逐 case 的状态、动作和原因分析
  decision.md             # 保留、观察或回退的结论
  traces/                 # 只保存需要复盘的完整 trace
```

原始评测 trace 继续保存在隔壁 `ptcg-agent-kaggle` 项目的 report 目录。当前仓库只保留精炼 case、指标摘要、结论和必要的回归 fixture，避免把大批逐局 JSON 混入 submission。

每个 case 的精炼结构至少包含：

```json
{
  "case_id": "v7-iter-01-001",
  "source": {"opponent": "kiyotah_dragapult", "game": 4, "turn": 6},
  "failure_class": "post_ko_no_ready_attacker",
  "state_summary": {
    "active": {"id": 66, "energy_count": 0},
    "bench": [],
    "hand_count": 7,
    "deck_count": 18,
    "prize_count": 3
  },
  "legal_options": [],
  "control_action": [],
  "candidate_action": [],
  "expected_action": [],
  "expected_reason": "当前没有可接班的主要攻击者，应避免把抽牌动作误认为攻击接力已经完成",
  "case_status": "pass"
}
```

## 8. V7 基线

V7 当前 17 个对手各 10 局、共 170 局的详细评测记录位于 `submission/alakazam_v7/EVAL_RESULT.md`，基线为：

- 胜率：106/170 = 62.4%；
- 第二回合 `Powerful Hand`：46/170 = 27.1%；
- 败局主因中，19/64 局属于击倒后打手断档，占全部对局 11.2%；
- 4/64 局出现 Active Dudunsparce 在空 Bench 时使用 `Run Away Draw`，占全部对局 2.4%。

其中 19/64 和 4/64 是本批败局的诊断摘要，不替代未来的事件级断档率和零错误计数。V6 的第二回合基线目前没有可靠同口径数据，不在 AutoIter 设计中填写估计值。

## 9. 后续实现边界

实现阶段优先完成四个独立组件：

1. 评测封装：统一 control/candidate、对手、局数、swap 和报告 manifest；
2. trace 指标提取：第二回合攻击、击倒后接力、空 Bench `Run Away Draw` 和 correctness；
3. case fixture：从精炼 state/options 建立确定性动作回归；
4. 迭代报告模板：记录假设、改动、case 结果、策略信号、胜率 guardrail 和最终决策。

在这四个组件完成前，不进行大规模自动参数搜索，也不把单次矩阵胜率直接作为策略改动依据。
