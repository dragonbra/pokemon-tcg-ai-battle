# 0043 Project Charter & Codex Handoff
## Promote Champion V2 / League Training 项目立项文档

**Project ID:** 0043  
**Date:** 2026-08-12  
**Primary Goal:** 建立可持续进化的 Opponent League，并为 Dragapult / Alakazam 等主提交 Deck 提供更强、更细粒度的训练压力。  
**Protocol Dependency:** `0043_Promote_Champion_V2_Protocol.md`

---

# 0. 给 Codex 的执行摘要

0043 不是一次新的模型架构实验，也不是重新调 PPO 参数。

0043 的任务是：

1. 把正式 Deck、Policy、Evaluation 资产完整收进 0043；
2. 建立不可变、可审计的 Deck / Policy Registry；
3. 接入新的 Opponent League；
4. 实现 256-game rollout 下的 50/25/25 sampler；
5. 实现按 10 updates 刷新的 PFSP curriculum；
6. 增加能反映真实训练强度的 W&B telemetry；
7. 保留现有 Frozen Evaluation 作为历史坐标系；
8. 建立 Candidate Freeze / Promote Champion V2 的正式入口；
9. **不改变当前已经验证健康的 PPO / optimizer / rollout 训练主参数。**

Codex 开工时先做 repo audit，再实现。  
不要根据本文猜旧文件路径；找到当前真实的 active training / evaluation 入口后再导入。

---

# 1. 为什么要做 0043

当前已经出现一个明确训练瓶颈：

> Policy0809 对现阶段 focal policy 的惩罚能力不足。

在 0809 上，模型可能：

- 浪费攻击轮次；
- 做风险很高的赌博性决策；
- 做局部明显次优但不会直接输掉对局的操作；
- 仍然依靠整体能力优势拿到胜利。

这意味着：

`win vs 0809` 已不再等价于 `操作足够准确`。

如果继续只 exploit 0809，RL 很可能进入一个很宽的“能赢旧老师就行”的策略区域，而不是继续逼近更精细、更稳健的最优操作。

0043 的核心假设是：

> **提升 teacher / opponent quality，会给当前 Policy 提供更高分辨率的惩罚信号。**

---

# 2. 当前已知项目状态

Codex 应先审计当前 repo 并确认以下事实与真实实现一致。

## 2.1 Rollout

- 当前正式 rollout：**256 games / update**。
- 该规模不要改。
- 现有训练配置已经让 behavior KL 真正发生健康变化。
- 0043 不做 512 rollout，不为 reference probe 切走部分 lanes。

## 2.2 训练参数

原则：

> **clone 当前最新批准的 active training config；不要根据旧文档重新手填一套“看起来一样”的参数。**

Codex 在 Phase 0 应：

1. 定位当前实际训练 entrypoint；
2. 导出完整 config snapshot；
3. 记录 config hash；
4. 建立 regression guard；
5. 0043 初版只允许 opponent sampler / asset routing / telemetry / promote governance 变化。

如代码中已有 PPO epochs、optimizer-step budget、LR、GAE、KL、Value、LoRA、Adapter 等参数，一律从当前 active config 继承。

除非用户显式要求：

> 不得顺手重调。

## 2.3 Engine / Frozen Policy 治理

必须继承之前已经建立的规则：

- CPU 官方引擎是语义 ground truth。
- Frozen opponent 使用自己的完整有效权重。
- 禁止 focal trunk + old head 的 Hybrid。
- Champion / anchor 要能完整重建。
- 同源 checkpoint metadata 不能代替 effective-weight equality audit。
- 与提交相关的 deploy-effective policy 需要保持可审计。

---

# 3. 0043 资产边界

## 3.1 Hard Requirement：0043 自包含

正式运行不得依赖：

```text
../0036/...
../0040/...
../0042/...
../0809/...
```

或者任何其他历史实验目录作为运行时资产源。

历史文件可以一次性导入，但之后：

- copy / content-address 到 0043；
- 写 manifest；
- 记录 hash；
- 只使用 0043 内相对路径。

验收时应支持：

> 把 0043 复制到一个不包含旧实验目录的 clean environment 后，仍然能够加载正式 Deck、0809、G1、Frozen Evaluation，并重建合法对手。

## 3.2 建议目录

Codex 可以根据 repo 结构调整，但建议：

```text
0043/
├── README.md
├── PROTOCOL.md
├── assets/
│   ├── decks/
│   │   ├── registry.json
│   │   └── definitions/
│   ├── policies/
│   │   ├── registry.json
│   │   ├── blobs/
│   │   └── manifests/
│   └── evaluation/
│       ├── registry.json
│       ├── manifests/
│       └── seeds/
├── league/
│   ├── sampler.py
│   ├── pfsp.py
│   ├── pfsp_config.yaml
│   ├── pfsp_state/
│   └── curriculum_manifests/
├── train/
├── evaluate/
├── promote/
├── tests/
└── reports/
```

---

# 4. Deck Registry

## 4.1 初始内容

至少导入：

1. 当前 55 套 Meta Deck；
2. 当前准备加入 training pool 的 Rogue / low-score / semantic-coverage Deck；
3. Dragapult 主 Deck；
4. Alakazam 主 Deck；
5. 其他当前训练或评测脚本实际引用的正式 Deck。

不要自行猜 exact 60-card list。  
必须从当前已批准资产中提取，并生成 hash。

## 4.2 Registry Schema 建议

```json
{
  "deck_id": "deck_xxx_v001",
  "name": "human readable name",
  "archetype": "string",
  "cards": [
    {"card_id": 123, "count": 4}
  ],
  "card_count": 60,
  "content_sha256": "...",
  "roles": ["training"],
  "tags": ["rogue", "semantic-coverage"],
  "source": {
    "imported_from": "...",
    "imported_at": "..."
  }
}
```

要求：

- `card_count == 60`
- 同一 `deck_id` 内容不可变
- 内容改变必须新 version
- 训练池与 Frozen Eval 池可以引用同一个 immutable deck asset，但 pool manifest 必须分开

---

# 5. Policy Registry

## 5.1 初始正式 Policy

### Policy0809

角色：

`historical_anchor`

用途：

- 历史回归；
- Frozen Evaluation；
- G1 全面体检对比；
- 可参与 Active Policy Pool 的 uniform / PFSP 分支。

它不再是唯一或最高质量 teacher。

### G1

角色：

`latest_champion`

G1 当前正在 / 即将进行全面体检。  
0043 可与体检并行开工。

当用户最终指定 G1 正式 checkpoint 后：

1. import 到 0043；
2. 生成 complete reconstruction manifest；
3. effective-weight audit；
4. freeze；
5. 成为 Latest Champion Pressure 的 teacher。

Codex 不得猜 G1 路径或 checkpoint 编号；必须从当前 repo / 用户指定结果确认。

## 5.2 Policy Manifest 至少记录

```json
{
  "policy_id": "champion_g001",
  "role": "champion",
  "generation": 1,
  "architecture_version": "...",
  "parent_policy": "...",
  "components": {
    "base": "...",
    "decoder": "...",
    "lora": "...",
    "adapter": "...",
    "value": "..."
  },
  "effective_weight_hashes": {
    "...": "..."
  },
  "inference_semantics": {
    "storage_dtype": "...",
    "runtime_dtype": "...",
    "deploy_mode": "..."
  },
  "frozen": true
}
```

具体字段应按真实模型结构适配。

## 5.3 No Hybrid Test

必须提供自动测试 / audit：

- focal 权重变化不能改变 0809 effective tensors；
- focal 权重变化不能改变 G1 effective tensors；
- opponent loader 输出必须完全来自该 policy manifest；
- 正式对手不得共享 focal 中会被更新的可变参数引用。

---

# 6. Opponent Deck 与 Policy 独立采样

0043 不实现强制 `deck × policy cell compatibility`。

原因是当前路线假设：

- BC 已经给每个 Deck 建立了基础操作语言；
- 强 Policy 的进一步 RL 能力具有较高跨 Deck 迁移性；
- PTCG 不依赖 bluffing / 针对低水平玩家的 exploit style；
- opponent strength 主要体现为每一步组合优化是否更准确、更能惩罚错误。

因此 sampler 分别选择：

```text
opponent_deck
opponent_policy
```

然后组合成实际对手。

仍需记录 joint matrix：

```text
focal_deck × opponent_deck × opponent_policy
```

仅用于诊断，不用于 V2.0 compatibility gate。

---

# 7. 256-game Opponent Sampler

每个 rollout 必须保持 256 games。

初版精确分配：

```text
128 games  PFSP weakness exploitation
 64 games  Uniform coverage
 64 games  Latest Champion pressure
--------------------------------------
256 games  total
```

如现有 rollout framework 使用 lane queue / 256 environment slots，优先在构建 queue 时一次性生成上述 quota，再 shuffle lane order，避免位置偏差。

如存在先后手 / seed 平衡逻辑，必须继承，不得被新 sampler 破坏。

---

# 8. PFSP V1 实现

## 8.1 统计维度

至少分别维护：

### Opponent Deck Stats

对当前 focal / 当前训练 context：

```text
deck_id
games
wins
losses
smoothed_win_rate
weakness
last_updated
```

### Opponent Policy Stats

```text
policy_id
games
wins
losses
smoothed_win_rate
weakness
last_updated
```

可同时维护更细 joint stats，但初版 sampler 不依赖 joint compatibility。

## 8.2 初始平滑

默认：

```python
smoothed_wr = (wins + 1) / (games + 2)
weakness = 1.0 - smoothed_wr
```

新 Deck / Policy 在 0 局时：

```text
smoothed_wr = 0.5
```

避免少量随机输赢直接劫持 curriculum。

## 8.3 Sampling

PFSP 128 lanes：

- Deck weights ∝ weakness，经 floor / cap 处理；
- Policy weights ∝ weakness，经 floor / cap 处理；
- 两者独立采样。

Floor / cap 不要写死在函数内部。  
写入 config，并在 report 中回显实际值。

如果当前数据不足以合理设定 cap：

- 先提供安全 default；
- 加单元测试保证任一对象不会因为数值问题出现 NaN / zero-all；
- 把 cap 标记为 tunable，不要把其值写成不可变协议。

---

# 9. Uniform Coverage 64 lanes

Uniform branch：

```text
opponent_deck   ~ Uniform(TrainingDeckPool)
opponent_policy ~ Uniform(ActivePolicyPool)
```

注意：

这不是历史 Meta distribution。

现有 55 Meta 的环境频率继续保存在 evaluation assets / metadata，但这 64 局用于：

- 保证 Rogue Deck 仍有机会出现；
- 保证特殊卡牌语义被覆盖；
- 防止 PFSP 因短期胜率而完全丢掉某些对象；
- 防止只练最难 matchup 导致 coverage collapse。

---

# 10. Latest Champion Pressure 64 lanes

```text
opponent_policy = LatestFrozenChampion
opponent_deck   ~ Uniform(TrainingDeckPool)
```

如果 0043 刚初始化、G1 尚未正式导入：

- 不得偷偷用正在更新的 focal 代替；
- 可暂时保持该 branch disabled / 指向经确认的 latest frozen champion；
- 必须在 config / W&B 明确标记状态。

G1 导入后，正式切到 G1。

未来 G2 Promote 成功：

```text
latest_champion = G2
```

G1 仍保留在 registry，不删除。

---

# 11. PFSP Curriculum Version

默认：

```text
refresh_every_updates = 10
```

流程：

```text
Update 000–009 -> Curriculum C000
refresh
Update 010–019 -> Curriculum C001
refresh
Update 020–029 -> Curriculum C002
...
```

每次 refresh 保存：

```json
{
  "curriculum_version": "Cxxx",
  "created_at_update": 10,
  "deck_weights": {...},
  "policy_weights": {...},
  "stats_snapshot": {...},
  "config_hash": "...",
  "training_deck_pool_hash": "...",
  "active_policy_pool_hash": "..."
}
```

训练日志必须能回溯：

> 某一个 rollout 当时到底面对了什么 curriculum。

---

# 12. W&B Telemetry

0043 的目标不是再找一个 Value 指标，而是让每 256 局 rollout 本身产生足够丰富的“真实对局强度”信息。

## 12.1 必做

```text
rollout/raw_win_rate
rollout/games
rollout/curriculum_version

strength/terminal_prize_margin
strength/prizes_taken_on_loss
strength/opponent_prizes_taken_on_win
strength/turns_to_win
strength/turns_to_loss

pfsp/sampling_entropy
pfsp/deck_difficulty/*
pfsp/policy_difficulty/*
coverage/deck_p10_win_rate
coverage/policy_p10_win_rate
coverage/red_deck_count
coverage/red_policy_count
```

同时保留已有：

- behavior KL
- reference KL / 现有 KL 指标
- policy / value losses
- value explained variance
- entropy
- optimizer step / coverage
- engine turns 等已有健康指标

## 12.2 Per-Deck / Per-Policy Panels

W&B 至少应提供：

- top-k easiest opponent decks
- top-k hardest opponent decks
- top-k easiest policies
- top-k hardest policies
- sampling probability vs observed WR
- curriculum version boundary

不要只把几十个 series 混成不可读的一张图。  
可以使用 table / grouped panel / artifact report。

## 12.3 Win / Loss 拆分

已有 average engine turn 指标必须拆成：

```text
turns_to_win
turns_to_loss
```

Prize 至少拆成：

```text
prizes_taken_on_loss
opponent_prizes_taken_on_win
```

因为总平均回合数或总平均 prize 很容易被 changing PFSP distribution 混淆。

---

# 13. Frozen Evaluation 迁移

## 13.1 现有评测不要改语义

当前：

- 55 Meta Deck；
- 历史环境分布；
- 组成固定 256 mini-batch；
- 用于 rollout 历史方案与最终评测；
- Policy0809 是主要历史 anchor。

0043 要做的是：

> **把它剥离成正式 Frozen Evaluation asset。**

不是重新设计它。

要求：

- 保存 exact deck composition；
- 保存 frequency / lane counts；
- 保存 seed protocol；
- 保存对手 Policy ID；
- 保存 inference mode；
- 给 benchmark version；
- 运行结果与旧入口做 regression 对比。

## 13.2 推荐命名

```text
FrozenMeta256-V1
FrozenMeta2048-V1   # 若正式体检为 8 × 256
```

实际名称可变，但必须版本化。

---

# 14. G1 全面体检任务

G1 出来后并行执行。

目标不是要求所有 55 Deck 都上涨，而是回答：

> 主要在多龙上进化出来的 G1，是否把更强的 Policy 能力迁移到了广泛 Deck？

建议报告：

```text
G1 vs 0809 baseline

overall delta
per-deck WR
per-deck delta
improved count
flat count
regressed count
worst regression
best improvement
confidence / sample count
```

将结果写入：

```text
0043/reports/g1_cross_deck_audit.*
```

若结果显示：

- 大多数 Deck 提升或持平；
- 少数下降接近统计噪声；
- overall 有明显正向变化；

则视为支持“Deck 与 Policy 能力近似正交”的工作假设。

若出现系统性大规模跨 Deck regression：

> 停止把该假设当作既定事实，提交 V2.x 设计讨论，不要 Codex 自行加 compatibility hack。

---

# 15. Promote Champion V2 工程入口

需要一个明确 promote command / workflow，名字按 repo 风格确定。

逻辑：

```text
candidate checkpoint
        ↓
freeze manifest
        ↓
asset completeness audit
        ↓
effective-weight audit
        ↓
no-hybrid audit
        ↓
reconstruction audit
        ↓
deploy / inference semantic audit
        ↓
Frozen Evaluation
        ↓
report
        ↓
manual promotion decision
        ↓
registry update + Generation increment
```

V2.0 不要求 Codex 自动决定“是否 Promote”。

Codex 只负责：

- 生成可靠结果；
- 防止非法 Candidate；
- 防止评测过程中权重变化；
- 生成可追踪报告；
- 在用户明确 Promote 后原子化更新 registry。

---

# 16. 今晚的操作路线

今晚主要不是自动实验框架，而是建立 0043 后开始第一轮真正 League Evolution。

## Phase A — G1 Health Check 与 0043 立项并行

并行执行：

### A1. G1 全面体检

- Frozen 0809
- 多 focal deck
- 检查 G1 是否广泛提升

### A2. Codex 建 0043

- asset import framework
- deck registry
- policy registry
- evaluation registry
- self-contained pathing
- tests

两者互不等待。

---

## Phase B — Dragapult Focal + Stronger Opponent Pool

Focal：

`Dragapult`

目标：

1. 切到新的更强 opponent league；
2. 预期 raw rollout WR 先明显下降；
3. 在 curriculum version 冻结的 10-update window 内观察恢复；
4. 希望恢复来自更准确的操作，而不是 exploit 陈旧 0809。

重点看：

```text
raw_win_rate
prizes_taken_on_loss
opponent_prizes_taken_on_win
turns_to_win
turns_to_loss
deck/policy tail
behavior KL
```

如果老师变强后 WR 没掉：

- 检查 new policy 是否真的被加载；
- 检查 sampler distribution；
- 检查 G1 effective weights；
- 不要先假设“模型太强所以没掉”。

---

## Phase C — Generalist Phase

当 Dragapult 进入平台后：

- focal deck 改为更广泛 / 随机 deck schedule；
- opponent league 继续使用 0043；
- 目标是提高共享 Policy 基础能力；
- 不删除 Dragapult 能力资产。

该阶段不是新 Champion Generation。  
它只是当前 Candidate 的 Generalist training phase。

---

## Phase D — 回到 Dragapult

最关键的实验之一：

> Generalist phase 后回到 Dragapult，是否能突破之前的 Dragapult plateau？

如果能：

- 支持 Generalist training 不是浪费主 Deck budget；
- 支持共享 Policy 能力可被更广泛环境抬高；
- 为后续循环提供强证据。

---

## Phase E — Alakazam Focal

之后把 focal 切到：

`Alakazam`

使用同一 Opponent League。

目标：

- 验证第二个主提交 Deck 是否也能从更强 teacher 中获得精细化操作收益；
- 验证 Dragapult / Alakazam 是否具备共同 League evolution 资格。

阶段预计可以在约 100–200 updates 量级观察，但：

> 不把 update 数作为强制停止条件。

优先看 plateau 与 W&B 质量信号。

---

# 17. Implementation Phases for Codex

## Phase 0 — Audit Only

先不要大改代码。

输出：

1. 当前 active training entrypoint
2. 当前 active config snapshot + hash
3. 当前 opponent loading path
4. 当前 55-deck mini-batch build path
5. 当前 evaluation entrypoint
6. 当前 W&B metrics
7. 当前 Policy0809 source
8. G1 source（如已明确）
9. 当前 asset dependencies
10. 0043 计划复制 / 重构清单

发现任何不确定项：

> 报告，不猜。

## Phase 1 — Self-contained Assets

实现：

- deck registry
- policy registry
- evaluation registry
- import command
- content hash
- relative path manifests
- asset integrity tests

验收：

- 0043 不依赖旧目录运行资产加载。

## Phase 2 — Opponent Sampler

实现：

- 128 PFSP
- 64 uniform
- 64 latest champion
- independent deck / policy sampling
- seeded reproducibility
- lane shuffle
- first/second-player semantic preservation

先写 sampler tests，再接训练。

## Phase 3 — PFSP State

实现：

- per-deck stats
- per-policy stats
- Beta(1,1) smoothing
- 10-update curriculum freeze
- curriculum version manifest
- configurable floor / cap
- resume / checkpoint of PFSP state

训练断点后：

> curriculum state 也必须可恢复。

## Phase 4 — W&B

加入协议定义的 telemetry。

确保：

- 不显著拖慢 rollout；
- 不需要额外 engine games；
- aggregation 在 256 games 完成后计算；
- 不把大量 per-game logging 直接刷到 W&B 导致吞吐下降。

## Phase 5 — Frozen Evaluation Assetization

把旧 55 Meta 评测剥离到 0043。

重点：

> regression parity，而不是重新调评测。

旧入口与新 0043 入口在相同资产 / seed / policy 下应输出一致或协议允许范围内一致的结果。

## Phase 6 — Promote Workflow

实现：

- freeze candidate
- audit
- evaluate
- report
- manual promote
- registry update
- generation increment
- latest champion pointer update

---

# 18. 必须写的测试

至少包括：

## Asset

- deck hash stable
- modified deck rejected under same ID
- missing policy blob rejected
- external path dependency rejected
- manifest reconstruction succeeds

## Policy

- 0809 effective-weight audit
- G1 effective-weight audit
- focal update does not mutate opponent
- no hybrid routing
- latest champion pointer references frozen policy only

## Sampler

在 256 batch：

```text
PFSP   == 128
Uniform == 64
Latest == 64
Total  == 256
```

- seeded run reproducible
- deck / policy independently sampled
- empty / one-item pool handled
- new item smoothed WR == 0.5
- no NaN weights
- concentration floor/cap respected

## Curriculum

- weights do not change inside 10-update window
- refresh exactly at version boundary
- resume reproduces same curriculum state

## Evaluation

- FrozenMeta256 composition unchanged
- seed manifest stable
- 0043 imported 0809 equals approved source effective policy
- old/new evaluation parity test

## Training Regression

最重要：

> 在关闭新 sampler / 使用 legacy-compatible config 时，0043 不得因为项目重构改变当前 PPO optimizer 行为。

至少检查：

- parameter groups
- learning rates
- trainable parameter set
- optimizer-step budget
- rollout games = 256
- behavior/action inference semantics
- loss config
- gradient flow

---

# 19. Codex 禁止自行决定的事项

Codex 不得未经用户明确授权：

1. 改 rollout 256。
2. 改 LR / PPO epoch / optimizer budget / GAE / KL 等主训练超参。
3. 将 G1 路径或版本“猜成”某个 checkpoint。
4. 修改 Frozen 55 Meta composition。
5. 将 Rogue Deck 自动加入 Frozen Evaluation。
6. 自动删除旧 Champion。
7. 自动 Promotion。
8. 将最新 focal 权重直接当 opponent。
9. 建立未经要求的 deck-policy compatibility whitelist。
10. 为了工程方便重新引入旧项目 runtime dependency。
11. 仅用 checkpoint source SHA 代替 effective-weight audit。
12. 在没有报告的情况下改变 FP16 / FP32 deploy semantics。
13. 因 W&B 指标太多而删掉现有训练健康指标。
14. 为采集 telemetry 额外跑大量游戏，侵占 update budget。

---

# 20. 0043 Definition of Done

0043 第一阶段可以宣布“可开始正式 League 训练”，必须同时满足：

- [ ] 0043 可以自包含加载所有正式训练与评测资产
- [ ] 55 Meta Frozen Evaluation 已版本化导入且语义未变
- [ ] Policy0809 已以完整 effective policy 身份导入
- [ ] G1 在确认后可被完整导入、冻结、重建
- [ ] No-Hybrid audit PASS
- [ ] 256 rollout 保持不变
- [ ] 当前训练超参数 regression audit PASS
- [ ] 128/64/64 sampler tests PASS
- [ ] Deck / Policy 独立采样 PASS
- [ ] PFSP 统计与 10-update curriculum version PASS
- [ ] W&B prize / win-loss turn / PFSP / tail metrics 可见
- [ ] PFSP state 可保存与恢复
- [ ] Frozen Evaluation 新旧入口 parity PASS
- [ ] Candidate freeze / audit / report workflow 可运行
- [ ] 文档明确区分 Training Deck Pool 与 Frozen Evaluation Pool
- [ ] 所有正式 asset / config / curriculum 都有可追踪 hash / version

---

# 21. 0043 第一轮成功信号

我们不要求所有信号同时完美，但希望看到：

## Dragapult

切强 opponent pool 后：

```text
raw WR ↓
```

随后在固定 curriculum 内：

```text
raw WR ↑
prizes_taken_on_loss ↑
opponent_prizes_taken_on_win ↓
turns_to_win ↓ 或更稳定
tail weakness 收缩
```

## Generalist Phase

希望：

```text
general deck coverage ↑
Dragapult historical ability 不出现明显结构性崩塌
```

## Return to Dragapult

最关键：

```text
new plateau > old plateau
```

## Alakazam

希望复制同样现象：

```text
stronger teacher
   ↓
initial difficulty
   ↓
policy refinement
   ↓
new frontier
```

---

# 22. 对今晚的定义

今晚的目标不是简单“多跑几百个 updates”。

真正希望完成的是：

> **让 G1 第一次从一个 checkpoint 变成 0043 League 的第一代强老师，让多龙和胡地开始在一个会持续进化、会真正惩罚错误的环境里训练。**

如果这条路线成立，0043 之后我们不再被一个陈旧 Policy0809 的能力上限困住。

训练的目标也不再只是：

> “还能不能赢这个旧对手？”

而是：

> “当老师越来越强、卡组语义越来越广时，我们的 Policy 能不能继续把每一步操作做得更准确？”

这就是 0043 的立项目标。
