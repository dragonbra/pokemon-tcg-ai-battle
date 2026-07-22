# 仓库职责拆分与工作目录重构设计

## 1. 目标

本次重构只调整仓库的职责边界和文件组织，不改变当前 Alakazam 策略的行为。
目标是把历史提交、当前工作版本、长期设计知识、自动迭代约定、评测框架和对局可视化拆开，
让每个目录都有明确的维护对象。

本次确认的核心决定是：

- `work/` 位于仓库根目录，表示当前正在开发的版本；
- `work/` 下按 `docs/`、`auto-iteration/` 和当前候选卡组目录三个主题拆分；
- 当前候选卡组目录必须保持纯净，只包含最终提交包需要的 `main.py`、`deck.csv` 和 `cg/`；
- `submission/` 保存历史完整提交，并新增 `submission/dist/` 保存打包产物；
- `codex/index/` 保存经过沉淀的长期项目知识和索引；
- 评测和可视化的核心代码逐步收回本仓库，详细结果和复盘资料继续放入 `docs/reports/`。

## 2. 目标目录结构

```text
.
├── AGENTS.md
├── README.md
│
├── codex/
│   ├── README.md
│   ├── index/
│   │   ├── README.md
│   │   ├── CURRENT_STATE.md
│   │   ├── DESIGN_PRINCIPLES.md
│   │   ├── RULES.md
│   │   ├── EVALUATION.md
│   │   └── CHANGELOG.md
│   └── design/
│       ├── specs/
│       └── plans/
│
├── work/
│   ├── README.md
│   ├── docs/
│   │   ├── README.md
│   │   ├── DECK_NOTES.md
│   │   ├── STRATEGY.md
│   │   ├── CURRENT_OBJECTIVES.md
│   │   └── decisions/
│   ├── auto-iteration/
│   │   ├── README.md
│   │   ├── GOALS.md
│   │   ├── METRICS.md
│   │   ├── ITERATION_RULES.md
│   │   ├── ITERATION_TEMPLATE.md
│   │   └── templates/
│   └── alakazam_v8_<candidate-name>/
│       ├── main.py
│       ├── deck.csv
│       └── cg/
│
├── submission/
│   ├── README.md
│   ├── dist/
│   │   └── alakazam_v8_<candidate-name>.tar.gz
│   ├── alakazam_v1/
│   ├── alakazam_v2/
│   ├── ...
│   └── alakazam_v8/
│
├── evaluation/
│   ├── README.md
│   ├── configs/
│   ├── runner/
│   ├── adapters/
│   ├── metrics/
│   ├── cases/
│   └── schemas/
│
├── visualization/
│   ├── README.md
│   ├── replay/
│   ├── analysis/
│   └── templates/
│
├── docs/reports/
├── replays/
├── data/official/
├── engine/
├── scripts/
└── tests/
```

## 3. 目录职责

### 3.1 `codex/`

`codex/` 是项目级 Codex 工作脚手架，不参与 Kaggle submission 打包。

`codex/index/` 只保存稳定且需要长期维护的信息：

- `CURRENT_STATE.md`：当前工作候选、已确认基线、最近结果和下一步目标；
- `DESIGN_PRINCIPLES.md`：从规则研究和历史实验中沉淀出的策略设计原则；
- `RULES.md`：官方规则和 simulator 行为的可执行摘要，详细证据链接到 `docs/reports/rules/`；
- `EVALUATION.md`：历史评测结果、关键指标、分母和报告链接；
- `CHANGELOG.md`：单一 Alakazam 卡组从 V1 到当前版本的重大变化和原因。

索引文件不复制完整 replay、逐局 trace 或长篇研究报告，而是提供结论、状态和来源链接。

`codex/design/` 用于集中保存设计规格和实施计划。现有 `docs/superpowers/` 内容后续可以
按主题迁移到这里；迁移前不改变已有路径，避免破坏历史引用。

### 3.2 `work/`

`work/` 是当前迭代的唯一工作入口，不保存历史候选版本。

#### `work/docs/`

存放当前候选版本的说明性文档，包括：

- 卡组构成和卡牌语义说明；
- 当前策略流程和设计假设；
- 本轮准备解决的问题和主要目标；
- 不适合写进通用策略文档的当前决策记录。

这些文件服务于设计、复盘和下一轮迭代，不进入最终提交包。

#### `work/auto-iteration/`

存放 Auto Iteration 的轻量流程约定，而不是每一轮的大型结果文件。至少定义：

- 每轮迭代的目标和单一主要假设；
- correctness gate、case resolution 和 outcome guardrail；
- 第二回合攻击、打手接力、牌库安全等核心指标；
- control、candidate、focused evaluation 和 promotion 的流程；
- iteration manifest、analysis、decision 和 case 的模板。

具体迭代结果继续放入 `docs/reports/kaggle/` 或其他报告目录，精炼后的结论登记到
`codex/index/EVALUATION.md` 和 `codex/index/CHANGELOG.md`。

#### `work/alakazam_v8_<candidate-name>/`

这是当前实际提交候选。目录必须保持与最终打包输入完全一致：

```text
work/alakazam_v8_<candidate-name>/
├── main.py
├── deck.csv
└── cg/
```

目录内不放 `README.md`、`STRATEGY.md`、分析 JSON、replay、`__pycache__` 或其他说明文件。
所有说明性内容统一放到 `work/docs/`。

### 3.3 `submission/`

`submission/` 保存历史完整提交。每个历史版本仍然是一个自包含目录，继续满足：

```text
submission/<name>/
├── main.py
├── deck.csv
└── cg/
```

历史版本原则上只读，不把当前工作的临时改动直接写回历史版本。

`submission/dist/` 是唯一的本地打包产物目录。压缩包内部仍必须保持 Kaggle 要求的顶层结构：

```text
main.py
deck.csv
cg/
```

因此，打包脚本需要从当前 `work/<name>/` 读取候选，并把结果写入
`submission/dist/<name>.tar.gz`。

### 3.4 `evaluation/`

`evaluation/` 收回评测框架的核心代码，后续逐步承接当前散落在 `scripts/` 和外部评测仓库中的
可复用逻辑：

- `configs/`：对手 registry、发现/聚焦/晋级等评测配置；
- `runner/`：运行 control、candidate 和矩阵评测；
- `adapters/`：本地官方引擎、Kaggle replay 和其他输入格式适配；
- `metrics/`：统一指标定义、分母和聚合逻辑；
- `cases/`：case 提取、fixture 和回归输入；
- `schemas/`：manifest、summary、case 等结构约定。

评测框架不在仓库内默认保存大批逐局原始 trace。长期保留的结果应是摘要、精炼 case、分析和决定。

### 3.5 `visualization/`

`visualization/` 负责特定对局的展示和复盘分析：

- `replay/`：识别 replay、提取可视化帧、生成 viewer launcher；
- `analysis/`：生成行动时间线、策略指标和问题摘要；
- `templates/`：Markdown 或 HTML 复盘报告模板。

原始官方 replay 继续放在 `replays/`，生成的分析报告放在 `docs/reports/`，不把临时 HTML 或大型本地
可视化输出写入当前候选提交目录。

## 4. 迁移边界和顺序

后续实施按以下顺序进行：

1. 建立 `codex/index/`、根目录 `work/` 和 `submission/dist/` 的目录与入口文档；
2. 整理历史版本索引、规则摘要、评测摘要和 V1 至当前版本 changelog；
3. 把当前候选整理为 `work/alakazam_v8_<candidate-name>/`，确认其只包含可打包文件；
4. 修改打包脚本，使输入来自 `work/`，输出写入 `submission/dist/`；
5. 在不改变命令行为的前提下，把评测核心逻辑逐步迁移到 `evaluation/`；
6. 把可视化和复盘分析核心逻辑逐步迁移到 `visualization/`；
7. 最后更新 `AGENTS.md`、README、测试和所有路径引用。

本次结构重构不包括：

- 修改 Alakazam 策略行为；
- 修改卡组构成；
- 修改官方卡牌数据库或官方引擎源码；
- 把历史 replay 或大批评测 trace 迁移进 `work/`；
- 自动上传 Kaggle submission。

## 5. 验收标准

结构重构完成后应满足：

1. 根目录存在 `work/`，且其中只有 `docs/`、`auto-iteration/` 和当前候选卡组主题目录；
2. 当前候选目录可以直接通过资产检查和打包检查；
3. 当前候选目录内没有说明文档、trace 或临时文件；
4. 打包产物位于 `submission/dist/`，并且压缩包顶层仍是 `main.py`、`deck.csv`、`cg/`；
5. 历史提交仍可按原版本目录访问；
6. `codex/index/` 能够链接到规则证据、评测报告和历史版本变化；
7. 评测框架和可视化工具的职责边界写入 `AGENTS.md` 并有对应入口文档；
8. `python3 scripts/check_assets.py`、目标代码语法检查、打包检查和 `git diff --check` 通过。
