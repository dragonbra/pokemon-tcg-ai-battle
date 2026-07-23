# 胡地 BC / RL 训练项目

本目录拥有胡地项目特有的 feature schema、policy/value 模型、MCTS、reward/loss、dataset
处理和训练入口。`training/` 里的模块是可组合的项目代码，不再是仓库级杂项脚本。

```text
train/alakazam_bc_rl/
├── features.py, card_metadata.py       # 项目输入合同
├── full_action_model.py, inference.py  # 模型与推理
├── mcts.py                             # 项目搜索实现
├── training/
│   ├── dataset.py, rewards.py          # 数据与训练目标
│   ├── build_*.py                      # BC/DAgger/MCTS 数据或 candidate
│   └── train_*.py, calibrate_value.py  # BC、过渡 RL 与校准入口
└── DESIGN.html                         # 当前模型与阶段事实
```

已有数据集与 checkpoint 未复制，完整保留在 `rl_runs/dataset/` 和
`rl_runs/checkpoint/`。例如：

```bash
PYTHONPATH=. python3.11 -m train.alakazam_bc_rl.training.train_full_action_bc \
  rl_runs/dataset/<experiment>/dataset.jsonl \
  --output rl_runs/<experiment>/V<n>_<tag> \
  --device cuda
```

模型输入、输出 head、action contract、loss/reward 或阶段变化时，必须同步更新
`DESIGN.html`。
