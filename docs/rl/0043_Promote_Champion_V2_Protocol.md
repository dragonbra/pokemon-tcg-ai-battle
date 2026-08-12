# Promote Champion V2 Protocol
## 0043 起执行的 PTCG RL 训练、对手进化、评测与 Champion 治理协议

**Protocol Version:** V2.0  
**Baseline Project:** 0043  
**Effective Date:** 2026-08-12  
**Status:** 0043 起正式执行  
**Audience:** GPT / Codex / 训练操作者 / 评测与导出脚本维护者

---

## 0. 协议目的

Promote Champion V2 的目的，是把当前 PTCG RL 训练从“持续对一个陈旧 Frozen Policy 做局部优化”升级为一个可持续进化、可审计、可复现的 League 体系。

V2 解决的核心问题是：

1. **弱对手无法继续提供足够细粒度的惩罚。**  
   Policy0809 在当前阶段已经可能允许 focal policy 通过一些粗糙、赌博性或明显次优的操作仍然取得胜利，因此单纯继续 exploit 0809 不能保证策略继续朝更准确、更细致的方向进化。

2. **训练分布与最终验收必须解耦。**  
   Rollout 需要动态地寻找当前弱点；Evaluation 需要保持冻结，才能作为历史坐标系与最终验收基准。

3. **Deck、Policy、Evaluation 必须成为 0043 内部的正式资产。**  
   0043 不允许依赖散落在旧实验目录中的临时权重、卡组列表或 seed 文件来重建正式训练与评测语义。

4. **训练主循环不能因为 V2 立项而被顺手重调。**  
   当前 256-game rollout 与现有 PPO / optimizer 配置已经表现出健康的 behavior KL 变化。V2 的第一版只改变 opponent curriculum、资产治理、监控与 Promotion 工作流，不重做训练超参数实验。

---

# 1. 协议权威与术语

## 1.1 V2 继承并强化 V1

Promote Champion V2 完整继承 Promote Champion / Frozen Policy V1 的核心治理原则：

- 每个 Policy 的身份由其**完整有效推理权重（full effective inference weights）**定义。
- focal model 的改变不得静默改变任何 Frozen opponent。
- 禁止 `focal trunk + old opponent head` 等 Hybrid Policy。
- Frozen Policy 必须能从其正式资产中精确重建。
- Promoted Champion 一经冻结即成为不可变快照。
- CPU 官方引擎仍是语义 ground truth。
- 与提交相关的 deploy-effective inference 语义必须被显式审计，不得用“同一个 checkpoint 名称”替代真正的 effective-weight identity。

若 V1 与 V2 在资产组织、采样或评测调度上存在冲突，以 V2 为准；V1 的 Frozen Policy 身份与防 Hybrid 原则不得被削弱。

## 1.2 术语

- **Focal Policy**：当前正在被更新的模型。
- **Focal Deck**：当前训练阶段由 focal policy 操作的主视角卡组。
- **Opponent Deck Pool**：训练过程中可由对手使用的 Deck 资产集合。
- **Opponent Policy Pool**：训练过程中可作为对手权重使用的 Policy 资产集合。
- **Evaluation Pool**：冻结的最终验收 / 历史体检资产集合。
- **Champion**：通过当前正式 Promotion 流程、被冻结并登记的 Policy 快照。
- **Generation**：仅在 Champion 成功晋升后递增。普通 update 数量不构成新 Generation。
- **Generalist Phase**：focal deck 从更广泛的卡组池中采样、用于提升共享 Policy 基础能力的训练阶段。该术语不得与 Champion Generation 混用。
- **PFSP**：Prioritized Fictitious Self-Play。在本项目中指根据当前 focal policy 的弱点动态提高对应 opponent deck / opponent policy 的采样概率。

---

# 2. 0043 自包含资产原则

## 2.1 硬性要求

0043 运行正式训练、正式评测、Champion Promotion 或 Champion 重建时：

> **不得依赖 0036 / 0040 / 0042 / 0809 等历史工程目录中的运行时文件。**

历史资产允许执行一次性 import，但 import 后必须：

- 复制或内容寻址存入 0043；
- 生成正式 manifest；
- 记录 SHA256 / 等价内容哈希；
- 使用 0043 内相对路径引用；
- 通过重建与有效权重审计。

任何正式资产在 0043 内部无法独立重建，则不得被视为 V2 合法资产。

## 2.2 推荐资产布局

```text
0043/
├── assets/
│   ├── decks/
│   │   ├── registry.json
│   │   └── ...
│   ├── policies/
│   │   ├── registry.json
│   │   ├── blobs/
│   │   ├── policy0809/
│   │   ├── champion_g001/
│   │   └── ...
│   └── evaluation/
│       ├── registry.json
│       ├── manifests/
│       └── seeds/
├── league/
│   ├── pfsp_config.*
│   ├── pfsp_state/
│   └── curriculum_manifests/
├── train/
├── evaluate/
├── promote/
└── reports/
```

实际目录名可由 Codex 结合仓库结构调整，但上述资产边界不得改变。

---

# 3. Deck 资产协议

## 3.1 Deck 身份

每套正式 Deck 必须至少包含：

- `deck_id`
- 人类可读名称 / archetype
- exact 60-card list
- card IDs / 数量
- deck content hash
- source / provenance
- 加入时间
- 所属集合：training / evaluation / both
- 可选标签：meta / rogue / semantic-coverage / exploit / experimental

**禁止原地修改 Deck。**

只要 exact list 发生变化，就必须产生新的 deck version / deck hash。

## 3.2 Training Deck Pool

Training Deck Pool 是**可进化、可追加**的。

初始基础包括现有 55 套 Meta Deck；后续允许加入：

- 低分 Rogue Deck；
- 包含模型较少见卡牌、场地、Ability、检索或特殊语义的 Deck；
- 曾在线上或专项测试中暴露模型理解盲区的 Deck；
- 新发现的高价值 matchup。

一套 Deck 是否值得加入训练池，不只由真实环境出现频率决定。  
**Card-semantic coverage 本身就是合法的加入理由。**

## 3.3 Evaluation Deck Pool

现阶段冻结的 55 套 Meta Evaluation Deck 与其历史环境频率保持不变。

新增训练 Deck：

> **不得自动进入 Evaluation Pool。**

Evaluation Pool 只有在显式发布新 benchmark version 时才允许变更；变更后必须保留旧版本，禁止静默覆盖。

---

# 4. Policy 资产协议

## 4.1 Policy 身份

每个正式 Policy 至少必须记录：

- `policy_id`
- role：anchor / champion / archive / candidate
- source checkpoint / import provenance
- 完整重建 manifest
- full effective weight hashes
- base / decoder / LoRA / adapter / value 等实际参与推理的来源
- inference precision / deploy-effective metadata
- compatible model architecture version
- freeze timestamp
- parent champion（若有）

正式对手 Policy 不得仅凭文件名或旧 checkpoint 名称识别。

## 4.2 初始 Active Policy Pool

0043 V2 初始至少包含：

- **Policy0809**：Historical Anchor。  
  保留其历史坐标意义，但不再被视为足够强的唯一训练老师。

- **G1**：Current / Latest Champion。  
  在 G1 最终冻结后导入 0043，成为更强的主要 opponent pressure 来源。

未来每次 Promotion 成功：

- 新 Champion 加入 Active / Champion Registry；
- 旧 Champion 不删除；
- 旧 Champion 可继续留在 active pool 或进入 archive / hall-of-fame 状态；
- 任何被保留用于正式训练或评测的 Policy 仍必须完整存在于 0043 内部。

## 4.3 Policy 与 Deck 的关系

V2 **不建立硬性的 `Policy × Deck` compatibility cell 限制**。

当前工作假设是：

> PTCG 中更强的通用 Policy 能力与具体 Deck identity 具有较强正交性；在 BC 已经形成基础卡组语言后，进一步 RL 得到的决策能力可以跨 Deck 迁移。

因此：

- opponent deck 与 opponent policy 可以独立采样；
- 新 Champion 可以带入不同 opponent deck；
- 不因为 Champion 主要在多龙或胡地阶段进化就限制其操作其他 Deck。

但必须保留按 `deck × policy` 的联合统计作为 diagnostic，以便发现未来可能存在的异常 specialization。  
若 G1 全面体检显示该假设明显不成立，再发布 V2.x 修订；不得在没有证据的情况下提前施加 cell 限制。

---

# 5. Rollout 与训练协议

## 5.1 256-game Rollout 不变

0043 第一版保持：

> **每次 rollout = 256 games。**

不得为了 V2：

- 改成 512；
- 硬切出额外 reference probe lanes；
- 调整 PPO epoch / optimizer budget；
- 顺手改现有 LR、GAE、KL、batch 语义或其他已验证健康的训练参数。

若未来要改变这些内容，必须作为独立实验，不得混入 V2 League 立项。

## 5.2 Focal Deck 调度独立于 Opponent Sampler

Opponent V2 sampler 负责：

- opponent deck
- opponent policy

Focal Deck 由外层训练阶段决定。

例如：

- Dragapult focal phase
- Generalist phase
- Alakazam focal phase

不得把 focal deck 的阶段切换与 opponent PFSP 的 50/25/25 配额混为同一概念。

---

# 6. Opponent Sampling V2：50 / 25 / 25

每个 256-game rollout 使用固定三路 curriculum mixture：

| 分支 | 比例 | 256 局对应配额 | 目的 |
|---|---:|---:|---|
| PFSP Weakness Exploitation | 50% | 128 | 主动挖掘当前 focal 的薄弱 opponent deck / policy |
| Uniform Coverage | 25% | 64 | 防止训练池中低频或特殊语义 Deck/Policy 被遗忘 |
| Latest Champion Pressure | 25% | 64 | 始终保持来自当前最强 Frozen Champion 的高质量惩罚 |

除非后续 V2.x 明确修订，0043 初版默认使用**精确 128 / 64 / 64 配额**，而不是仅以 multinomial 期望实现 50/25/25。

## 6.1 PFSP 分支

PFSP 分支中：

- opponent deck 根据当前 focal 对各 Deck 的 smoothed weakness 独立采样；
- opponent policy 根据当前 focal 对各 Policy 的 smoothed weakness独立采样；
- 两者不做硬绑定。

第一版优先使用简单、可解释的 weakness score，不追求复杂 PFSP 数学。

推荐基础估计：

```text
smoothed_wr = (wins + 1) / (games + 2)
weakness   = 1 - smoothed_wr
```

即 Beta(1,1) / Laplace 平滑，保证新对象初始均值为 50%。

允许增加：

- minimum sampling floor；
- maximum concentration cap；
- recent-window / EMA；

但这些必须显式配置并记录，不得藏在代码常量中。

## 6.2 Uniform Coverage 分支

- opponent deck：从完整 Training Deck Pool 均匀采样；
- opponent policy：从 Active Policy Pool 均匀采样。

该分支的目的不是模拟真实 Meta，而是保证训练覆盖。

因此 V2 中不把它命名为 `P_meta`。

真实环境 Meta frequency 仍作为重要 metadata 与 Frozen Evaluation 资产保留，但不直接决定这 25% 的 curriculum。

## 6.3 Latest Champion Pressure 分支

- opponent policy 固定为当前 Latest Frozen Champion；
- opponent deck 从 Training Deck Pool 广泛采样，默认 uniform。

目的：

> 防止系统再次出现“旧老师过弱，因此大量次优操作无法得到足够惩罚”的情况。

Latest Champion 必须是 Frozen snapshot；不得指向正在更新中的 focal weights。

---

# 7. PFSP 更新节奏

## 7.1 Piecewise-stationary Curriculum

PFSP 权重默认：

> **每 10 updates 刷新一次。**

在同一个 curriculum version 内，采样权重冻结。

目的：

- 让 focal 有时间适应当前老师；
- 让 W&B 能看到“课程切换后的胜率下降 → 同一课程内逐步恢复”；
- 防止每个 256 rollout 后立刻反馈导致 sampling oscillation；
- 便于复现与审计。

每次刷新必须：

- 产生新的 `curriculum_version`；
- 保存 deck sampling weights；
- 保存 policy sampling weights；
- 保存用于计算这些权重的统计快照；
- 在 W&B 中记录 version boundary。

PFSP refresh 周期属于 V2 初始默认值；若未来调整，必须版本化配置并在实验记录中声明。

---

# 8. Rollout W&B 强度监控协议

PFSP 上线后：

> **raw rollout win rate 不再等同于绝对实力。**

因为 opponent distribution 会随 focal 变强而变难。

因此每个 update 至少记录以下三类指标。

## 8.1 A. 当前课程表现

- `rollout/raw_win_rate`
- `rollout/games = 256`
- `rollout/curriculum_version`
- `rollout/opponent_deck_distribution`
- `rollout/opponent_policy_distribution`
- `rollout/deck_smoothed_win_rate/*`
- `rollout/policy_smoothed_win_rate/*`

用途：判断 focal 是否正在适应当前 curriculum。

## 8.2 B. 对局质量 / 强度代理

至少记录：

- `strength/terminal_prize_margin`
  - 我方终局拿奖数 − 对手终局拿奖数

- `strength/prizes_taken_on_loss`
  - 仅统计 loss；输了时我方平均已经拿了多少 prize

- `strength/opponent_prizes_taken_on_win`
  - 仅统计 win；赢局中对手平均已经拿了多少 prize

- `strength/turns_to_win`
  - 仅统计 win

- `strength/turns_to_loss`
  - 仅统计 loss

推荐同时记录：

- first-prize / first-KO 等引擎可稳定定义的先手优势事件；
- terminal prize margin 的分布 / 分位数；
- decisive-win / close-loss 比例。

解释原则：

- `turns_to_win ↓` 单独看不能证明变强；
- `turns_to_loss ↑` 单独看也不能证明变强；
- 更有意义的联合趋势是：
  - 赢得更快；
  - 赢局让对手拿的 prize 更少；
  - 输局自己能拿更多 prize；
  - 输局能坚持更久或至少不更快崩盘。

## 8.3 C. Curriculum / Coverage 健康度

至少记录：

- `pfsp/sampling_entropy`
- `pfsp/curriculum_version`
- `pfsp/deck_difficulty_distribution`
- `pfsp/policy_difficulty_distribution`
- `pfsp/worst_decks`
- `pfsp/worst_policies`
- `coverage/deck_p10_win_rate`
- `coverage/policy_p10_win_rate`
- `coverage/red_deck_count`
- `coverage/red_policy_count`

可选：

- `pfsp/hard_mass`
- current opponent generation / rating summary
- joint `deck × policy` diagnostic matrix

## 8.4 禁止制造一个不透明“总强度分”

V2 初版不允许用一个未经验证的手工 composite score 替代 win rate / prize / turn / coverage 明细。

W&B 的职责是提供多视角健康信号；正式 Champion 资格仍由 Frozen Evaluation / Promotion 决定。

---

# 9. Evaluation Pool 协议

## 9.1 现阶段不大改 Frozen Evaluation

当前已有：

> 55 套 Meta Deck 按历史环境分布组成固定 256-game mini-batch。

它继续作为冻结历史体检标准。

正式命名建议：

`FrozenMeta256-V1`

如当前完整评测使用 8 个 256 单元，则保留：

`FrozenMeta2048-V1 = 8 × FrozenMeta256-compatible seed blocks`

具体 seed manifest 必须进入 0043 evaluation assets。

## 9.2 Policy0809 的新定位

Policy0809：

> 从“主要训练老师 / 进化目标”降级为“Historical Anchor / Regression Check”。

它继续回答：

- 新 Policy 是否连历史能力都丢失；
- G1 / G2 / 后续 Champion 相比历史 checkpoint 是否有整体迁移；
- 不同 focal deck 带入同一 Policy 后，跨 Deck 能力如何变化。

它不再单独回答：

> “当前 Candidate 是否已经达到新一代最强水平？”

## 9.3 G1 全面体检

G1 冻结后，应执行一次跨 Deck 全面体检：

- 使用冻结的评测协议；
- 将 G1 带入不同 focal deck；
- 与 Policy0809 基线对比；
- 输出 aggregate + per-deck delta；
- 重点统计：
  - improved deck count
  - roughly-flat deck count
  - regressed deck count
  - worst regression
  - overall delta

该体检用于验证 V2 的重要工作假设：

> 在主卡组上进化的更强 Policy 是否能较广泛迁移到其他 Deck。

0043 的立项不需要等待该体检结束；二者可以并行。

---

# 10. Evaluation 频率与 Scout

## 10.1 不再固定高频跑完整 Frozen Evaluation

V2 默认：

> **不在每 N 个 update 自动跑完整 0809 / 2048 体检。**

原因：

- 0809 已经偏旧；
- PFSP rollout 本身将提供大量更高价值的训练强度信息；
- 高频独立 Eval 会显著侵占真正用于 update 的时间。

## 10.2 Scout 降级为按需工具

Scout Evaluation：

- 不是固定周期任务；
- 可以非常精简；
- 只在 operator / W&B 显示 candidate 值得进一步检查时触发；
- 不能因为“到了 25/50/100 update”就机械执行。

## 10.3 Full Evaluation / Promotion Check

仅在以下类型事件出现时考虑冻结 Candidate 并跑正式验收：

- 当前 focal 在固定 curriculum version 内明显恢复并进入平台；
- prize / turn / tail metrics 同步改善；
- Generalist phase 后回到主 focal 出现新突破；
- operator 判断已出现有提交价值的 candidate；
- 需要发布新的 Champion Generation。

V2.0 初期 Promotion 保留人工触发，不自动化为单一阈值。

---

# 11. Promote Champion V2 工作流

## 11.1 Candidate Freeze

一旦决定进入正式 Promotion：

1. 立即冻结 Candidate；
2. 生成不可变 candidate manifest；
3. 记录完整 effective-weight identity；
4. 后续评测不得边测边继续更新同一 Candidate；
5. Candidate 若继续训练，必须产生新的 candidate ID。

## 11.2 Hard Audit Gate

任何 Champion 晋升前必须通过：

- asset completeness audit
- policy registry audit
- full effective-weight audit
- no-hybrid opponent audit
- checkpoint reconstruction audit
- inference/deploy semantic audit
- required engine semantic / parity regression tests

任何一项失败：

> 不得 Promote。

## 11.3 Performance Review

V2.0 初期至少保留：

- Frozen historical evaluation；
- Policy0809 historical anchor comparison；
- per-deck regression / generalization report；
- 当前训练期间 W&B evolution report。

Latest Champion / previous Champion head-to-head、专项 matchup gate、Shadow Benchmark 等可以作为 V2.x 扩展加入，但在尚未建立稳定 baseline 与方差估计前，不在 V2.0 随意硬编码数值门槛。

## 11.4 Promotion

只有人工明确宣布 Promotion 成功后：

- Generation +1；
- Candidate 改为 immutable Champion；
- 新 Champion 写入 Policy Registry；
- 新 Champion 成为 `Latest Champion Pressure` 分支的固定 teacher；
- 上一代 Champion 不删除；
- 下一 Generation 的训练从新的 League 状态开始。

Update 100 / 200 / 300 本身永远不等于 Generation +1。

---

# 12. 今晚及近期推荐进化路线

V2 支持但不强制固定 update 数量的阶段式训练：

```text
G1
 ↓
Dragapult focal + stronger opponent league
 ↓
在同一 curriculum version 中观察：WR 下滑 → 恢复 → 平台
 ↓
Generalist Phase（随机 / 广泛 focal deck）
 ↓
提升共享 Policy 基础能力
 ↓
回到 Dragapult，测试能否突破旧 plateau
 ↓
Alakazam focal
 ↓
重复同样的 League 进化逻辑
```

预期单阶段约 100–200 updates 可以作为操作参考，但：

> **平台信号优先于机械 update 数量。**

核心实验目标是验证：

1. 更强 opponent pool 是否能把主 focal 从 0809 的“宽松最优区”推向更精细策略；
2. Generalist phase 是否能提高跨 Deck 共享能力；
3. 回到 Dragapult 后能否突破此前 plateau；
4. Alakazam 能否在同一强对手 League 下获得类似收益；
5. 两个主提交 Deck 是否具备共同使用一个进化 Policy League 的资格。

---

# 13. 禁止事项

0043 V2 明确禁止：

1. 正式 runtime 依赖旧项目散落资产。
2. 未版本化地修改已登记 Deck 60-card list。
3. 未版本化地覆盖 Frozen Evaluation composition / seeds。
4. focal 权重静默泄漏到 Frozen opponent。
5. Hybrid Policy。
6. 将 Latest Champion 指向正在更新的 focal model。
7. 因 V2 立项顺便调整已验证的 PPO / optimizer / rollout 主参数。
8. 将 PFSP raw win rate 当作绝对实力唯一指标。
9. 高频机械跑旧 0809 Full Eval 侵占训练预算。
10. 因某个 Rogue Deck 真实 Meta 占比低就拒绝其进入 training pool。
11. 将真实 Meta frequency 与 curriculum learning value 视为同一概念。
12. 仅凭 checkpoint 名称、目录名或 metadata 中的 source hash 判断 Policy 身份，而不审计实际 effective weights。
13. 未冻结 Candidate 就执行正式 Promotion Evaluation。
14. 在没有显式 benchmark version bump 的情况下改变 Evaluation Pool。

---

# 14. V2.0 已冻结决策 vs 待验证假设

## 已冻结决策

- 0043 自包含资产。
- 三大资产域：Opponent Deck Pool / Opponent Policy Pool / Evaluation Pool。
- Rollout 固定 256。
- 不改当前已验证训练参数。
- Deck 与 Policy 独立采样，不做 cell compatibility 绑定。
- Training Deck Pool 可追加。
- Frozen Evaluation 暂不改。
- Opponent sampling：50% PFSP / 25% Uniform Coverage / 25% Latest Champion Pressure。
- 256 局初版按 128 / 64 / 64 实现。
- PFSP 默认每 10 updates 刷新。
- Policy0809 降级为 Historical Anchor。
- Latest Champion 为 Frozen teacher。
- W&B 加入 prize、win/loss turn、tail、PFSP coverage 等强度监控。
- Full Evaluation 降低频率；Scout 按需；Promotion 初期人工触发。
- Champion 成功 Promotion 才增加 Generation。

## 待验证 / 可在 V2.x 修订

- G1 的跨 Deck 泛化是否足以长期支持完全独立的 Deck × Policy 采样假设。
- PFSP concentration cap / minimum floor 的最优数值。
- PFSP 是否需要从 `1 - WR` 升级到 hard-but-learnable weighting。
- Scout 的最佳局数与触发条件。
- 是否建立固定 incumbent head-to-head hard gate。
- 是否引入 Shadow Evaluation Pool。
- 是否引入 rollout 内 reference estimator / reference slice。
- 何时把 Promotion 从人工触发升级为半自动 / 自动触发。

所有待验证项目都不得在未记录实验结果的情况下静默变为硬协议。

---

# 15. 协议一句话摘要

> **0043 让训练对手快速进化，让正式资产完全自包含，让 Frozen Evaluation 保持历史可比；用 PFSP 找弱点、用 Uniform 保覆盖、用 Latest Champion 提供持续惩罚，并用高信息量 Rollout telemetry 取代无意义的高频旧基准体检。**

从 0043 起，我们不再只训练一个 checkpoint；我们维护的是一个可持续进化、可重建、可审计的 Champion League。
