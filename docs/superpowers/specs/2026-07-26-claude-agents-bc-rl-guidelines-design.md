# CLAUDE.md 与 AGENTS.md 统一及 BC + RL 路线优化设计

## 背景

仓库根目录目前只有 `AGENTS.md`，Codex 可以读取其中的项目级约定，但 Claude Code 的标准项目级入口是 `CLAUDE.md`。如果分别维护两个普通文件，内容容易漂移。

项目方向也已经发生明确变化：当前全面采用 BC + RL 路线。胡地卡组的单专家 BC 已经成功验证；接下来一方面建立 rollout、reward、value calibration 和 RL fine-tuning 闭环，另一方面使用相近的 BC 架构学习若干常见强力卡组，为后续 RL 提供更强且更多样的 arena opponents。

## 目标

1. 让 Claude Code 与 Codex 使用同一份项目级约定。
2. 保留现有仍然有效的官方引擎、评测、实验版本、规则和提交硬约束。
3. 明确 BC + RL 是当前主路线，以及胡地 BC、多卡组 BC 和 RL opponent 池之间的关系。
4. 防止多来源 BC 标签污染、弱 opponent 自动晋级和 opponent 池变化导致的 RL 实验不可复现。

## 非目标

- 不重构训练、评测或策略代码。
- 不修改 `engine/source/`。
- 不改变已有实验结果、路径、指标或已确认的宝可梦规则事实。
- 不在本次工作中训练新卡组、运行 RL 或调整 opponent catalog。
- 不自动提交或推送 Git 变更。

## 选定方案

采用“保守迁移 + 定向优化”：

```text
CLAUDE.md              # 唯一真实文件
AGENTS.md -> CLAUDE.md # 相对软链接
```

`AGENTS.md` 使用相对目标 `CLAUDE.md`，因此仓库移动后链接仍然有效。现有指向 `AGENTS.md` 的 README 链接无需调整，Claude Code 则直接读取标准名称 `CLAUDE.md`。

相比全面重构，该方案减少误改既有硬约束的风险；相比仅追加一段说明，它会定向修正与当前路线冲突或已经陈旧的措辞，使约定本身保持一致。

## 内容设计

### 1. 当前 BC + RL 路线

在 `CLAUDE.md` 中加入清晰的项目方向说明：

- BC + RL 是当前主要研发路线。
- 规则策略是历史资产、分析基线、回归参考或合法性保障，不再是主要扩展方向。
- 胡地单 deck、单 expert 的 BC 已完成成功验证，但不代表 RL 或跨卡组泛化已经完成。
- 当前并行推进两条主线：
  - 建立 rollout collection、reward、value calibration 和 RL fine-tuning 闭环；
  - 针对常见强力卡组分别训练可审计的 BC policy，验收后扩充正式 arena opponent 池。
- 更强且更多样的 BC opponents 是 RL curriculum 和训练环境的一部分，但其强度必须由官方引擎真实对局验证。

### 2. 多卡组 BC 隔离原则

保留现有单来源约束，并扩展到多卡组计划：

- 每套卡组保持明确的 deck、expert/team policy、dataset manifest、experiment/version 和 candidate package 边界。
- 不把不同 deck、不同 team 或不同实现逻辑的动作标签无条件混入同一个 BC policy。
- 默认优先训练独立的 deck-specific policy。
- 未来若采用共享模型，必须显式加入可审计的 deck/source conditioning，并设计冲突标签处理以及按 deck、source 分组的独立评测。
- 离线 exact-action 或 legal-action 指标只能证明模仿质量和动作合同，不能单独证明对战强度。

### 3. BC opponent 准入原则

- 新 BC policy 先产出自包含 candidate package。
- candidate 必须经过 package 验证和官方 engine runtime 的真实对局评测。
- 只有用户明确确认后，才能进入 `evaluation/arena/opponents/` 固定池。
- 不因训练完成、离线指标较高或单一 matchup 表现良好而自动晋级。
- 正式池命名、catalog、代表卡和资产测试继续遵循现有约定。

### 4. RL opponent 池可复现性

- 每个正式 RL run 必须记录实际使用的 opponent catalog、明确的 pool snapshot，或足以重建该集合的 package 标识与版本信息。
- opponent 池构成、采样权重或 curriculum 阶段发生变化时，必须作为显式实验变量记录。
- 不得让同一逻辑版本在未记录的情况下跨越不同 opponent 池继续训练或比较。
- 对手池增强的效果必须通过同口径评测验证，不能把并行吞吐、离线 BC accuracy 或静态分析当作策略强度证据。

### 5. 既有内容处理

- 保留官方引擎只读边界、真实对局证据要求、evaluation 目录约定、实验版本不可覆盖、逐 epoch 指标、Kaggle 显式授权和提交规范。
- 保留已确认的宝可梦规则与胡地策略语义；它们继续服务于专家行为分析、数据审计和回归检查。
- 仅修正与新路线直接冲突或明显陈旧的定位，不进行无关的大规模改写。
- README 当前链接到 `AGENTS.md`，软链接建立后仍然有效，因此本次无需修改 README。

## 实施步骤

1. 以当前 `AGENTS.md` 为完整基线生成 `CLAUDE.md`。
2. 在 `CLAUDE.md` 中加入上述路线和边界，并定向清理冲突措辞。
3. 删除普通文件形式的 `AGENTS.md`。
4. 创建相对软链接 `AGENTS.md -> CLAUDE.md`。
5. 检查工作区，确保未触碰用户已有的其他未提交改动。

## 验收与验证

本次仅涉及文档与软链接，不运行训练、正式 evaluation 或完整测试套件。验收证据包括：

1. `CLAUDE.md` 是普通文件。
2. `AGENTS.md` 是软链接，链接文本恰为 `CLAUDE.md`。
3. `AGENTS.md` 能正确解析到仓库根目录的 `CLAUDE.md`。
4. 从两个路径读取的内容完全一致。
5. `git diff` 只包含计划中的约定迁移和内容优化，不包含对现有其他工作区变更的改写。
6. 文档不存在互相矛盾的主路线描述，也不把胡地 BC 的成功扩大解释为 RL 已完成。

## 风险与缓解

- **平台软链接支持差异：** Git 可以记录软链接；目标使用相对路径，降低路径迁移问题。若某环境禁用 symlink，Git checkout 行为需由该环境配置处理，但仓库中的规范表示仍以 symlink 为准。
- **硬约束在编辑中丢失：** 以完整现有文件为基线，只做定向增补和必要措辞修正，完成后审阅 diff。
- **多卡组 BC 被误解为混合 policy：** 明确 deck-specific 默认和共享模型所需 conditioning/分组评测门槛。
- **RL 结果因 opponent 池漂移失真：** 要求每个正式 run 固化或可重建 opponent 集合，并将变化作为实验变量。
- **越过用户控制的外部动作：** 不自动 promote opponent、提交 Kaggle、commit 或 push。
