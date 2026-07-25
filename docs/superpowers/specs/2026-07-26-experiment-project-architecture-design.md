# 实验项目目录架构设计

**日期：** 2026-07-26
**状态：** 已确认，待实施计划
**适用范围：** 从 `0013` 起的新 BC、rollout、value calibration 与 RL 研究项目

## 目标

为每个研究假设建立可独立实现、可审计运行、可集中查阅的项目单元。人和 AI 应能从一个稳定入口找到项目设计、数据合同、版本决策、训练事实、candidate 以及官方引擎评测结论。

本设计不批量迁移或重写 `0001`–`0012` 的既有资产；它们保持原有路径并由兼容索引发现。

## 核心约定

### 项目 ID

项目使用全局递增 ID 加描述性 slug：

```text
0013_alakazam_rollout_value_calibration
```

一个项目表示一个明确的研究假设，不表示单一候选、checkpoint 或一次训练。项目内每次实际训练、校准、导出或评测迭代使用严格递增版本名：

```text
V1_initial_contract
V2_action_codec_fix
```

ID 和 slug 必须同时用于项目的实现、档案与运行资产目录。编号不可重用；失败、中止和确认有缺陷的版本同样保留。

### 三层职责

```text
rl_environment/                          # 无项目语义的共享基础设施
train/<project_id>/                      # 自包含的可运行训练实现
experiments/<project_id>/                # Git 跟踪的项目事实与导航
rl_runs/<project_id>/                    # 训练过程的物理运行资产
```

`rl_environment/` 仅提供跨项目且语义稳定的能力：项目与版本路径分配、原子占用检查、canonical JSONL/TensorBoard/W&B 日志协议、checkpoint 协议、流式数据原语、官方 engine adapter 和通用 rollout 容器。它不得包含 deck、expert/source、feature schema、模型、reward、loss 或具体训练算法。

`train/<project_id>/` 是独立训练实现。换卡组、重做 feature schema、模型结构或算法时，项目可以不依赖此前项目的业务代码。只有在两个及以上项目已验证语义相同的能力，才可迁入 `rl_environment/`。

`experiments/<project_id>/` 是项目的唯一人工与 AI 导航入口，保存长期可阅读、可审计的事实。它不保存大型训练数据、checkpoint、TensorBoard event 或可运行 candidate package 的复制品。

`rl_runs/<project_id>/` 是运行时资产的物理落盘区，不再承担项目首页、设计文档或正式评测的权威入口职责。

## 目标目录结构

### 共享基础设施

```text
rl_environment/
├── runs/                 # project/V 分配与路径保护
├── logging/              # JSONL → TensorBoard → W&B
├── runtime/              # official-engine adapter、rollout 协议
├── storage/              # checkpoint 与 artifact 公共契约
└── streaming/            # bounded-memory 数据原语
```

不为未来假设预建空模块。只有出现对应的跨项目稳定接口时才创建子包。

### 独立训练实现

```text
train/
└── 0013_alakazam_rollout_value_calibration/
    ├── README.md
    ├── features/
    ├── data/
    ├── model/
    ├── objective/
    ├── training/
    ├── configs/
    ├── export/
    └── tests/
```

各目录仅在有实际代码时创建：

- `features/`：observation、action 与 feature schema/codec。
- `data/`：项目特有的读写、验证、split 与数据转换。
- `model/`：网络和推理接口。
- `objective/`：BC loss、reward、value、PPO 等项目训练目标。
- `training/`：训练入口、loop、完整 train/validation epoch evaluation 与运行接入。
- `configs/`：与 `V<n>_<tag>` 对应的声明式配置。
- `export/`：candidate package 构建和验证，不保存 package 副本。
- `tests/`：项目特有单元及合同测试。

`README.md` 只说明如何运行该实现及其外部依赖；项目历史、实验结论和正式报告导航由 `experiments/<project_id>/README.md` 承担。

### 集中项目档案

```text
experiments/
├── INDEX.md
├── INDEX.html
├── 0013_alakazam_rollout_value_calibration/
│   ├── README.md
│   ├── DESIGN.md
│   ├── DESIGN.html
│   ├── manifest.json
│   ├── data_audit/
│   ├── decisions/
│   ├── versions/
│   │   ├── V1_initial_contract.md
│   │   └── V2_action_codec_fix.md
│   └── evaluation/
│       ├── index.html
│       └── V1_initial_contract.html
└── legacy/
    └── INDEX.md
```

`experiments/INDEX.md` 和 `INDEX.html` 是所有项目的总览，按项目编号排序并显示研究假设、当前状态、实现、设计、最新版本和最新正式评测链接。

项目 `README.md` 是唯一项目导航入口，至少链接：训练实现、项目设计、manifest、数据审计、决策记录、各版本状态、candidate manifest/hash、正式评测总览与对应 `rl_runs` artifact。

`DESIGN.md` 是可 diff、可审查的设计正文；`DESIGN.html` 是其对应的可视化当前状态。二者都位于 `experiments/<project_id>/`，使所有项目设计页集中且可按编号回溯。模型输入、网络、action contract、loss/reward/value/PPO 接口和项目阶段变更时，必须在同一工作中同步二者。

`manifest.json` 是项目身份和合同来源，至少声明项目 ID、研究假设、deck、expert/source policy、数据合同、官方 engine/runtime revision、opponent pool snapshot、训练实现路径和创建时 Git revision。项目启动后这些身份字段不可静默改写；若研究假设或关键合同改变，应新建项目。

`data_audit/` 保存可跟踪的数据来源、schema、规模、split、identity 与质量审计摘要，不保存大型原始语料。

`decisions/` 与 `versions/` 保存从一个版本到下一个版本的证据、失败原因、变量和后续动作。版本状态必须指向实际 artifact、candidate 或正式评测报告。

`evaluation/` 是正式官方 engine 评测 HTML 的权威位置。`index.html` 汇总项目全部版本的 candidate、run ID、局数、胜负、error、胜率、完成率及每份版本报告链接。每份报告的版本名必须与对应训练版本一致。

### 运行资产

```text
rl_runs/
└── 0013_alakazam_rollout_value_calibration/
    ├── dataset/                              # Git ignore
    └── versions/
        ├── V1_initial_contract/
        │   ├── artifact/                     # tracked 小型训练事实
        │   ├── checkpoint/                   # Git ignore 或 LFS 管理
        │   ├── tensorboard/                  # event 文件
        │   └── wandb/                        # 本地 staging，Git ignore
        └── V2_action_codec_fix/
            └── ...
```

`artifact/` 至少包含训练配置、逐 epoch `training_metrics.jsonl`、summary、status，以及正式评测报告的相对路径、内容 hash 与真实 `run_id`。在相同合同下，训练每个 epoch 必须完整评测 train 和 validation split，并把同名标量写入 JSONL 和 TensorBoard。

`dataset/`、`checkpoint/`、`wandb/` 以及适用的 TensorBoard event 文件遵循 Git ignore/LFS 策略；其路径仍必须由版本目录统一约束。不得按资产类型在 `rl_runs/artifact/`、`rl_runs/checkpoint/`、`rl_runs/tensorboard/` 等顶层目录分别建立项目树。

## Candidate 与正式 opponent

训练项目只能将可运行候选导出到：

```text
evaluation/arena/candidates/<candidate_name>/
```

项目档案和运行 artifact 记录 candidate 的路径、package hash、验证结果、评测链接和晋级决定，但不复制 package。只有用户明确确认后，candidate 才能按既有规则迁入 `evaluation/arena/opponents/` 并写入正式 catalog；训练完成或离线指标高不能自动晋级。

## 路径分配与守卫

`rl_environment.runs create` 负责创建新项目骨架和初始 manifest，而不是让使用者或 AI 手动推断路径。它必须：

1. 检查 `train/`、`experiments/`、`rl_runs/` 中均不存在指定项目 ID 或冲突 slug。
2. 同时创建并交叉写入同一 `<project_id>` 的三处目录。
3. 在项目 manifest 生成前拒绝创建可训练版本。
4. 验证 manifest 中 deck、expert/source、数据合同、engine revision 与 opponent pool snapshot 已填写。
5. 创建版本时验证版本名合法、严格递增、四类运行路径均不存在非空内容。
6. 在训练结束时检查 artifact 包含 config、逐 epoch metrics、summary/status。
7. 在正式评测落盘时更新 `experiments/<project_id>/evaluation/index.html`，并将报告路径、hash、run ID 反向记录到 artifact 和项目版本记录。

测试覆盖项目/版本路径分配、拒绝覆盖、ID/slug 一致性、manifest 必填合同、artifact 完整性和索引链接有效性。

## 历史项目过渡

`0001`–`0012` 保持现有 `rl_runs/artifact/`、`rl_runs/evaluation/`、`rl_runs/tensorboard/` 和相关训练目录不动，避免破坏已有报告、Git 历史和外部链接。

`experiments/legacy/INDEX.md` 为每个历史编号提供只读链接：已知训练实现路径、旧 artifact、旧 evaluation index 和关键结论。它是发现性兼容层，不伪造新结构缺失的 manifest、版本或数据审计。

当一个历史项目需要新的训练、维护或扩展时，才为该项目建立 `experiments/<project_id>/` 档案，并在新结构中记录后续版本；不对全仓执行高风险的批量路径迁移。

## 验收标准

从 `0013` 开始：

- 每个新项目同时拥有 `train/<project_id>/`、`experiments/<project_id>/` 和 `rl_runs/<project_id>/`。
- `experiments/<project_id>/README.md` 可定位设计、数据合同、版本、candidate、正式评测和运行 artifact。
- 所有 DESIGN 文档都集中于 `experiments/`，不再以 `train/` 作为权威位置。
- 每个训练版本只有一个同名运行目录，拒绝覆盖或复用。
- 正式评测 HTML 只以 `experiments/<project_id>/evaluation/` 为权威归档位置，并从 artifact 双向追溯。
- 新项目的 deck、expert/source、dataset contract、engine revision 和 opponent pool snapshot 都可审计。
- `0001`–`0012` 的现有文件和链接不被移动或破坏。
