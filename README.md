# Pokémon TCG AI Battle

面向 Kaggle [Pokémon TCG AI Battle Challenge Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview)
的中文研究、训练与评测仓库。项目围绕胡地卡组维护三条可审计链路：规则策略历史、基于人类
专家 replay 的行为克隆（BC），以及使用官方引擎运行的本地闭环评测。

模拟器接口参考 [CABT 文档](https://matsuoinstitute.github.io/cabt/)。仓库中的性能结论必须来自
官方 engine runtime 的真实模拟对局；静态分析、单元测试和离线动作准确率只作为辅助证据。

## 当前状态

项目已经从规则代理迭代进入单专家纯 BC 验证阶段。当前可运行候选是
[`work/yushin_ito_exact_bc_v2`](work/yushin_ito_exact_bc_v2)，归档 baseline 是
[`work/alakazam_bc_v1`](work/alakazam_bc_v1)。两者都是标准 package，独立包含 `main.py`、
60 行 `deck.csv` 和官方 `cg/` runtime。

当前主实验 [`0003-yushin_ito_exact_bc_v2`](rl_runs/0003-yushin_ito_exact_bc_v2/) 的
`V2_card_token_fix` 使用同一位 Yushin Ito 专家的 1,000 场 replay、86,875 条决策记录：

| 项目 | 当前结果 |
|---|---:|
| 模型 | Universal full-action BC，5.19M 参数，无规则 fallback |
| 数据切分 | 800 / 100 / 100 个完整 episode（train / validation / test） |
| Validation exact action rate | 80.52% |
| Test exact action rate / legal action rate | 78.37% / 100% |
| 本地官方引擎评测 | 136–44–0，180/180 完成，0 errors，75.56% 胜率 |
| Kaggle | submission `54912599`，public score `713.2` |

完整事实和原始产物见：

- [RL 实验索引](rl_runs/INDEX.html)
- [0003 决策记录](rl_runs/0003-yushin_ito_exact_bc_v2/decisions.md)
- [V2 训练摘要](rl_runs/0003-yushin_ito_exact_bc_v2/V2_card_token_fix/training_summary.json)
- [V2 官方引擎评测报告](rl_runs/0003-yushin_ito_exact_bc_v2/evaluation/V2_card_token_fix/run-8688bfdaa9c44059a7fec383b5f64f6e/report.html)
- [当前 RL 模型设计](train/alakazam_bc_rl/DESIGN.html)

这些结果验证了单 deck、单 expert 的 BC inference 和 full-action contract，不代表已经完成跨卡组
泛化或强化学习。Rollout collection、reward/value calibration 和 PPO fine-tuning 仍是后续阶段。

## 核心工作流

```text
Kaggle expert replay
  → 来源冻结、整局 split 与 dataset audit
  → 行为克隆训练和 best-validation checkpoint
  → 标准 candidate package（main.py + deck.csv + cg/）
  → 固定 18-opponent 官方引擎 evaluation
  → rl_runs 中的 config、metrics、case 和 HTML 报告
  → 验收后进入 work/；历史正式 payload 进入 submission/
```

关键边界：

- 一个 BC dataset 默认只包含一个明确的 team/agent policy；不同专家不能无条件混合标签。
- 同一 experiment 的每次训练使用新的 `V<n>_<tag>`，run、checkpoint、TensorBoard 和
  evaluation 版本名保持一致，失败版本也不覆盖或删除。
- `work/` 只保存当前候选，`submission/` 保存历史正式 payload；训练 checkpoint 和研究
  candidate 不是 Kaggle submission。
- `evaluation/` 负责测量和报告，不自动执行 promote/reject，也不替代 Kaggle 官方成绩。

## 仓库结构

| 路径 | 职责 |
|---|---|
| [`work/`](work/) | 当前标准候选 package；候选目录只放可运行/可打包内容 |
| [`submission/`](submission/) | 历史提交源目录、规则代理归档和正式 BC payload |
| [`rl_environment/`](rl_environment/) | 与卡组无关的训练基础设施、run 分配和存储保护 |
| [`train/alakazam_bc_rl/`](train/alakazam_bc_rl/) | 胡地 BC/RL 的特征、模型、reward/loss 与训练入口 |
| [`train/kaggle_bc_top20/`](train/kaggle_bc_top20/) | Top-20 单专家 BC 数据与 Kaggle worker 归档项目 |
| [`rl_runs/`](rl_runs/) | 可审计的 config、metrics、status、TensorBoard 索引和 evaluation 报告 |
| [`evaluation/`](evaluation/) | 固定 opponent catalog、独立 worker、指标插件和 Markdown/HTML 报告 |
| [`visualization/`](visualization/) | Kaggle replay 与 engine `visualize` 帧的统一 viewer launcher |
| [`engine/`](engine/) | 官方引擎只读源码和本地构建边界；构建产物只写 `engine/build/` |
| [`data/official/`](data/official/) | 官方卡牌参考数据，只读使用，不直接打包 |
| [`docs/reports/`](docs/reports/) | 研究报告、规则证据、Kaggle 复盘和实现结论 |
| [`scripts/`](scripts/) | 只保留 TensorBoard 启动入口，不放一次性训练/规则脚本 |
| [`tests/`](tests/) | evaluation、资产、可视化和 RL 数据/训练回归测试 |

大体积 dataset、checkpoint 和临时 replay 不进入 Git。BC 数据和 checkpoint 分别位于
`rl_runs/dataset/` 与 `rl_runs/checkpoint/`；本地临时评测、trace 和 replay 应写入
项目 `.tmp/`；临时 Evaluation report 固定写入 `.tmp/evaluation/`。

## 环境安装

项目要求 Python 3.11+：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[rl]'
```

基础依赖包含 Kaggle 客户端和数据工具；`rl` extra 安装 PyTorch、NumPy 与 TensorBoard。
GPU/CUDA 版本应根据训练机环境选择兼容的 PyTorch wheel。

## 快速开始

### 查看实验与训练曲线

直接打开 [`rl_runs/INDEX.html`](rl_runs/INDEX.html) 查看已归档实验、离线指标和正式评测。
跨 run 启动 TensorBoard：

```bash
./scripts/start_tensorboard.sh
```

默认读取 `rl_runs/tensorboard` 并监听 `127.0.0.1:6006`。可通过
`TENSORBOARD_LOGDIR`、`TENSORBOARD_HOST`、`TENSORBOARD_PORT` 和 `PYTHON` 覆盖；远程机器
建议保持本地监听并使用 SSH 端口转发。

### 验证当前候选

```bash
python3 -m evaluation list-opponents
python3 -m evaluation validate work/yushin_ito_exact_bc_v2
```

`validate` 会检查 package 布局、60 张 deck、deck hash、`cg/` tree hash 和官方 card ID。

### 运行正式本地评测

```bash
python3 -m evaluation run \
  --candidate work/yushin_ito_exact_bc_v2 \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output /tmp/ptcg-yushin-v2-evaluation
```

正式入口固定使用全部 18 个 opponent，每个至少 10 局。每次 run 会建立独立目录并写入：

- `manifest.json`：candidate、catalog、profile、hash 和运行参数；
- `summary.json` / `games.jsonl`：总体与逐局事实；
- `metrics.json` / `cases.jsonl`：聚合指标和证据 case；
- `report.md` / `report.html`：可直接审阅的语义报告；
- `traces/`：默认最多长期保留三份被选中的完整 trace。

`auto_iteration_v8_setup_relay` 当前为 revision 2，除胜负和正确性外，还展示二回合 setup、
Powerful Hand、Post-KO relay、攻击质量和牌库健康等指标。完整调用边界见
[`evaluation/HANDOFF.md`](evaluation/HANDOFF.md)。`--control` 目前只记录对比包信息，不会替你
执行第二批 control 对局；真实比较需要分别运行两个 candidate。

### 查看 replay

```bash
python3 -m visualization.replay.cli path/to/replay.json
python3 -m visualization.replay.cli path/to/replay.json --no-open
```

CLI 支持 Kaggle replay，以及包含非空 `visualize` / `visualize_frames` 的本地 replay。只有
observation/action 的旧 trace 无法事后还原卡面。详细格式和 viewer 故障排查见
[`visualization/README.md`](visualization/README.md)。

## RL 实验约定

RL 子系统的目录关系如下：

```text
rl_runs/<000N-experiment>/<Vn_tag>/                 # config、metrics、summary/status
rl_runs/tensorboard/<000N-experiment>/<Vn_tag>/     # TensorBoard event
rl_runs/checkpoint/<000N-experiment>/<Vn_tag>/   # 大模型文件，Git 忽略
rl_runs/<000N-experiment>/evaluation/<Vn_tag>/      # 对应版本的真实对局报告
```

新项目先通过 `rl_environment.runs create` 分配全局编号；同一项目的新尝试严格递增 `V<n>`，不能把
event、metrics 或 checkpoint 继续写进已有版本。训练入口、dataset 构建方式和完整命名规则见：

- [通用 RL 基础设施](rl_environment/README.md)
- [RL run 归档规则](rl_runs/README.md)
- [胡地训练项目](train/alakazam_bc_rl/README.md)
- [模型输入、结构和阶段设计](train/alakazam_bc_rl/DESIGN.html)

任何涉及 feature schema、模型结构/output head、action contract、BC/RL loss、reward/value/PPO
接口或项目阶段的改动，都必须同步更新 `train/alakazam_bc_rl/DESIGN.html`。

## 标准 package 与提交边界

一个可运行 package 必须自包含：

```text
<candidate>/
├── main.py
├── deck.csv      # 恰好 60 行
└── cg/           # 物理复制的官方 runtime，不使用 symlink
```

神经网络提交还会包含 `strategy/` 和 `model.bin`。候选必须只返回 simulator 提供的合法选项，
并在新对局清理模块级状态。

Kaggle 登录、接受规则、打包和正式 submission 都是显式人工操作。仓库不会自动上传，也不会在
pending、失败或低分后自动重试。不要提交 Kaggle 凭据、`.env`、下载的比赛数据或未经许可的
新二进制资源。

## 官方引擎边界

[`engine/source/`](engine/source/) 是官方 C++ 源码的只读版本化输入，禁止为适配 agent、指标
或测试而修改。允许按照官方说明构建，但所有产物只能写入 `engine/build/` 或仓库外临时目录，
不能覆盖任何 candidate/submission 的 `cg/`。

用于判断策略实际能力的结果必须来自官方 runtime 真实对局。mock、静态检查和离线 BC 指标
不能替代 evaluation。具体构建与许可证边界见 [`engine/README.md`](engine/README.md)。

## 开发与验证

Python 使用 4 空格、类型注解和清晰的小函数；Ruff 行宽为 100。测试使用标准库
`unittest`：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q evaluation visualization rl_environment train
bash -n scripts/start_tensorboard.sh
```

evaluation 资产变更还应单独运行：

```bash
python3 -m unittest -v tests.test_evaluation_assets
```

新增 opponent 必须是独立标准 package，并更新唯一 catalog
[`evaluation/configs/opponents.json`](evaluation/configs/opponents.json)。不要使用 adapter、共享
`cg/`、symlink 或其他仓库的绝对路径。更完整的协作、规则和提交流程见
[`AGENTS.md`](AGENTS.md)。

## 文档入口

- [实验总览](rl_runs/INDEX.html)
- [RL 基础设施](rl_environment/README.md)
- [模型设计](train/alakazam_bc_rl/DESIGN.html)
- [Evaluation 使用说明](evaluation/README.md)
- [Evaluation 交接契约](evaluation/HANDOFF.md)
- [Replay 可视化](visualization/README.md)
- [官方引擎边界](engine/README.md)
- [研究报告索引](docs/reports/README.md)
- [历史 submission 说明](submission/README.md)
