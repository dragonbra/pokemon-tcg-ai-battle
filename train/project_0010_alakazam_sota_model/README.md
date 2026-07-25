# Alakazam SOTA Model

这是一个独立的训练项目，用于忠实复刻 Kaggle Notebook
[`horizen12/ptcg-yushin-id-only-bc-v1`](https://www.kaggle.com/code/horizen12/ptcg-yushin-id-only-bc-v1)
的 ID-only pointer policy。`SOTA model` 是项目名；V4 已完成长曲线、官方 engine 固定池评测
与双 checkpoint Kaggle 提交，具体强度证据以归档报告和 submission record 为准。

模型、输入字段、现有 universal BC 对照和阶段状态见 [`DESIGN.html`](DESIGN.html)。

## 忠实复刻边界

- 只使用公开 observation 中的 card ID、owner、zone、slot、parent、状态和数值计数。
- 不使用官方 CSV category、卡文、规则、role、archetype、deck list 或 expert conditioning。
- 从随机初始化开始；不继承 0008/0009 checkpoint，不使用 reward weighting。
- 网络保持 Notebook 默认：`d_model=320`、8 heads、4 encoder layers、FFN `3x`、
  dropout `0.1`、GRU pointer decoder。
- 训练保持 Notebook 默认：batch 96、6 epochs、AdamW、LR `3e-4`、weight decay `0.02`、
  gradient clip `1.0`、seed `20260723`、AMP。
- 同时保留最低 validation token cross-entropy 的 `best_validation.pt` 和最高
  validation exact-action accuracy 的 `best_exact.pt`，再用官方 engine 评测决定策略版本。

仓库版只增加不改变梯度的工程观测：每个 epoch 写 TensorBoard，并额外执行完整 train-eval，
以便同时看到拟合和泛化曲线。

## 数据合同

0010 使用与 0009 完全相同的单一 expert corpus：

- team：`Yushin Ito`
- selection：2026-07-13 至 2026-07-22 daily winner-only
- validation：2026-07-22
- episode/player groups：3,104
- train：2,875 episodes / 221,289 decisions
- validation：229 episodes / 17,067 decisions
- source dataset SHA-256：
  `36415a61b93fdcdb2e1b17c0fbaebb9432f8d94ed0c203365b36cd3011b09bdf`

现有 0009 JSONL 只有 universal encoder 张量，无法还原 Notebook 的 owner/zone/parent 输入。
`build-dataset` 因此把冻结 JSONL 的 episode、player、step、targets 和 split 作为权威索引，
从十个保留 ZIP 读取对应原始 observation，再用 `IDOnlyCodec` 重编码。任何 actor、action、
source hash 或记录数不一致都会失败；不会重新筛选玩家或胜负，也不会静默跳过样本。

正式本地 shards 位于 `rl_runs/dataset/0010-alakazam_sota_model/`，gzip 后约 30.8 MB。

## 命令

```bash
python3 -m train.alakazam_sota_model build-dataset \
  --source-dataset rl_runs/dataset/0009-reward_weighted_bc/dataset.jsonl \
  --archive-root /mnt/d/pokemon-tcg-ai-battle-data/kaggle_yushin_ito_20260713_20260722 \
  --output rl_runs/dataset/0010-alakazam_sota_model \
  --config train/project_0010_alakazam_sota_model/configs/reference_notebook.json \
  --expected-source-sha256 36415a61b93fdcdb2e1b17c0fbaebb9432f8d94ed0c203365b36cd3011b09bdf

python3 -m train.alakazam_sota_model train \
  --dataset-root rl_runs/dataset/0010-alakazam_sota_model \
  --output rl_runs/artifact/0010-alakazam_sota_model/V2_empty_entity_fix \
  --config train/project_0010_alakazam_sota_model/configs/reference_notebook.json
```

对应 TensorBoard 与 checkpoint 必须使用同一版本名：

- `rl_runs/tensorboard/0010-alakazam_sota_model/V2_empty_entity_fix/`
- `rl_runs/checkpoint/0010-alakazam_sota_model/V2_empty_entity_fix/`

`V1_reference_notebook_reproduction` 在第 2 个 training batch 暴露空 public-entity
collate 缺陷后失败并保留。V2 只修复这个 padding 边界：至少分配一个被 mask 的 entity slot，
不改变任何非空样本编码、attention 或 loss 语义。

epoch 6 官方评测完成后，续训使用独立 V3，并在 validation exact 连续 5 轮未刷新历史最佳时停止：

```bash
python3 -m train.alakazam_sota_model train \
  --dataset-root rl_runs/dataset/0010-alakazam_sota_model \
  --output rl_runs/artifact/0010-alakazam_sota_model/V4_resume_rng_fix \
  --config train/project_0010_alakazam_sota_model/configs/continuation_until_patience5.json \
  --resume-checkpoint rl_runs/checkpoint/0010-alakazam_sota_model/V2_empty_entity_fix/latest.pt
```

V3 在任何 batch 开始前暴露 resume RNG device restore 缺陷并保留失败记录；V4 只把保存的
RNG ByteTensor 转回 CPU 后再恢复，仍从 V2 epoch 6 权重、optimizer 与 RNG 开始。

V4 在 epoch 21 刷新最高 validation exact 后，epoch 22–26 连续五轮未刷新，于 epoch 26
自然停止。最终并列保留并提交两类 checkpoint：

- loss-best：epoch 9，validation loss `0.232557`，Kaggle ref `54958731`，public score `944.2`；
- exact-best：epoch 21，validation exact `84.2562%`，Kaggle ref `54958951`，public score `600.0`。

虽然 exact-best 的本地固定池结果更高，但 Kaggle 结果显著更差；后续 0011 reward-weighted BC
以 loss-best 的训练阶段与强度证据作为 baseline，不以 teacher-forced exact 单独选策略。

## 模块

- `model.py`：ID-only codec、Transformer encoder、option cross-attention、GRU pointer decoder。
- `dataset.py`：冻结 identity 到 replay ZIP 的严格重编码，以及本地 gzip streaming dataset。
- `config.py`：模型与训练配置合同。
- `training.py`：纯 BC、完整 train/validation 观测、TensorBoard 与 checkpoint。
- `inference.py`：无标签 observation 编码与满足 min/max count 的 greedy pointer 解码。
- `export_candidate.py`：裁剪 optimizer 后导出自包含标准 candidate package。
- `__main__.py`：`build-dataset` / `train` 入口。
