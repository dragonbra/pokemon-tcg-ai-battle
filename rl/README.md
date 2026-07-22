# Pokémon TCG 学习框架

`rl/` 保存模型、训练工具和可审计的实验记录。可以打包的候选仍进入 `work/`，正式评测仍
通过仓库的 `evaluation` CLI 完成。

## 目录职责

```text
rl/
├── core/                         # 通用 loss、checkpoint、日志和 run 工具
├── model/                        # 特征、模型结构、推理和 MCTS
├── train/                        # 数据构建、审计、下载和训练入口
├── artifact/
│   ├── dataset/<run-name>/       # 仅 BC 使用，Git 忽略
│   └── checkpoint/<run-name>/<Vn_tag>/ # 每次训练独立模型大文件，Git 忽略
└── _runs/                        # 可提交、可比较的实验事实
    ├── INDEX.html
    ├── tensorboard/<run-name>/<Vn_tag>/
    └── <run-name>/
        ├── manifest.json
        ├── data_manifest.json
        ├── data_summary.json
        ├── V1_<tag>/             # config、metrics、summary/status
        ├── V2_<tag>/
        ├── evaluation/V1_<tag>/
        └── commands.md
```

同一实验使用一个全局名称，例如 `0003-next_bc`；实验内每次尝试再使用严格递增且不复用的
`V<n>_<snake_case_tag>`。run record、checkpoint、TensorBoard 和对应 evaluation 的版本名
必须完全一致。模型源码不再复制进每个 run：`model/`、`train/` 和 `core/` 是正式源码，
`manifest.json` 记录训练时的 Git commit，必要时额外保存 source revision patch。

Top-ladder 多专家项目先建立 source staging，不提前占用全局 run 编号：

```bash
python3 -m rl.train.prepare_top_ladder_bc \
  --top 20 \
  --raw-output data/replays/kaggle_top20_bc_20260723 \
  --dataset-output rl/artifact/dataset/top20_arena_20260723
```

该入口只冻结排行榜、保存每位专家的公开 Episode metadata、下载一局审计样本，并生成
`0001`–`0020` source slot 和原版 `deck.csv`；这些编号是 campaign 内 source ID，不是
`rl/_runs` 全局实验 ID，且状态为 `sample_only_not_train_ready`。完整 replay corpus 通过
审计后，再为每位专家分别执行 `rl.core.runs create`。即使两个专家使用完全相同的 60 张
构筑，也必须保持独立 dataset/run，不能无条件混合其动作标签。

BC 阶段可以生成 `artifact/dataset/`；后续 RL 阶段直接与真实 opponent 模拟对局，不创建
完整监督 label 数据集。TensorBoard event、metrics 和 evaluation 报告是重要的小型实验
证据，全部正常跟踪。

## 新实验

先分配编号：

```bash
PYTHONPATH=. python3.11 -m rl.core.runs create next_bc \
  --objective "Pure full-action BC on the frozen exact-deck replay pool"
```

BC 数据和训练分别写入约定位置：

```bash
PYTHONPATH=. python3.11 -m rl.train.build_kaggle_bc_dataset \
  replays/example \
  --manifest replays/example/manifest.json \
  --output rl/artifact/dataset/0003-next_bc/dataset.jsonl

PYTHONPATH=. python3.11 -m rl.train.train_full_action_bc \
  rl/artifact/dataset/0003-next_bc/dataset.jsonl \
  --output rl/_runs/0003-next_bc/V1_initial_contract \
  --device cuda
```

训练器会把 checkpoint 写到
`rl/artifact/checkpoint/0003-next_bc/V1_initial_contract/`，把 TensorBoard 写到
`rl/_runs/tensorboard/0003-next_bc/V1_initial_contract/`，并把 config、metrics 和 summary
写到同名 tracked version。任何已有文件都会阻止复用该版本。

固定评测写回同一个 run：

```bash
python3 -m evaluation run \
  --candidate work/<candidate-name> \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --output rl/_runs/0003-next_bc/evaluation/V1_initial_contract
```

跨实验查看 TensorBoard：

```bash
./scripts/start_tensorboard.sh
```

脚本默认监听 `127.0.0.1:6006`；远程训练机建议保留本地监听并使用 SSH 端口转发。
确需覆盖时可设置 `TENSORBOARD_LOGDIR`、`TENSORBOARD_HOST`、`TENSORBOARD_PORT` 或
`PYTHON`，额外命令行参数会原样传给 TensorBoard。

完整命名和归档规则见 [`RUNS.md`](RUNS.md)，实验对比入口见
[`_runs/INDEX.html`](_runs/INDEX.html)，模型设计背景见 [`DESIGN.md`](DESIGN.md)。
Top 20 单专家 BC 与 Arena 的项目拆分、样本牌组和 metadata 边界见
[`docs/reports/rl/top20-bc-arena-design-and-source-audit-20260723.md`](../docs/reports/rl/top20-bc-arena-design-and-source-audit-20260723.md)。
