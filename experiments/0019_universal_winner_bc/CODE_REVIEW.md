# 0019 数据与训练代码审查

审查时间：2026-07-29；2026-07-30 在全量数据与 V1 训练后复核

## 结论

0710–0728 全量数据已经完成构建并通过 materialized 全量验证。winner 选择、Episode 去重、
exact deck、整局 split、source/team conditioning、原始 payload commitment 和存储护栏均按
fail-closed 合同工作；最终规模为 89,503 Episodes / 7,347,132 decisions / 52.73 GiB。

V1 训练结果确认了此前的高优先级模型选择问题：训练和 validation 都使用真实 expert
`source_id`，而实际部署使用 `source_id=0`。后者的 embedding 固定为零，且没有训练样本直接
优化中性输出。因此，`bc/validation/loss` 只证明 expert-conditioned imitation，不能保证
被选择的 checkpoint 在中性部署模式下最好。

实证是：conditioned 指标在 Epoch 13 最优（loss `0.253080647`），neutral loss 则在
Epoch 9 最优（`0.365444490`）。这不是理论风险，已经成为 V2 必须处理的可测 gap。

## Findings

### P1：中性部署 persona 没有对应训练与选点指标

- 证据：`run_r15.py` 将 `deployment_source_id` 记录为 0；训练 batch 始终保留数据中的非零
  `source_id`；validation 同样使用非零 `source_id`。
- 影响：模型可以把一部分决策规律放进 persona residual。此时 conditioned validation loss
  下降，但 source 0 的共享 backbone 可能没有同步得到同等收益。
- 已完成训练前最低要求：增加一套不反向传播的 neutral validation（将同一 validation batch
  的 `source_id` 置 0），复用同一次 backbone encoding，并记录
  `bc/validation_neutral/*`。
- 建议实验：V1 保持当前 conditioned baseline；V2 对训练 batch 做 10%–20% deterministic
  source dropout，并以 neutral validation loss 选点。不要把两种语义写入同一个 version。

### P1：0728 加入后 source vocabulary 必须重新冻结

- 证据：`source_id` 根据最终 catalog 的 Unicode NFC team name 排序分配；新增日期可能增加
  team，并改变后续 ID。
- 影响：0727 catalog 生成的 raw/model-ready 数据不能与 0728 catalog 直接拼接，否则相同
  数字 ID 可能代表不同 team。
- 处理：当前流水线只作为预构建。若 09:00 后成功取得 0728，必须从最终 catalog 重新构建
  raw 与 model-ready，不能增量追加旧张量。

### P2：按 expert 的 validation 不能单独回答跨来源泛化

- 证据：split 单元是完整 Episode，但 train/validation 会共享 team 与 deck；这符合优化 loss
  和 early stopping 的目的。
- 影响：低 validation loss 可能包含同一 expert persona 的泛化，不是 unseen expert/deck
  泛化证据。
- 建议：保留当前 90/10 split 作为正式 early stopping surface；额外生成只读的 grouped
  diagnostics（按 source、deck、source 样本量分桶），不要为了研究严谨性丢弃有效训练数据。

### P2：model-ready shard 的重复目录扫描会增加构建尾部开销

- 证据：每关闭一个 tensor shard，storage guard 会递归统计整个 0019 dataset；当前每 shard
  1,024 decisions，预计会产生数千个文件。
- 影响：不影响正确性，但构建后半段会出现随 shard 数增长的元数据扫描成本。
- 建议：本次先保留 fail-closed guard；记录全量实测时间。后续若该部分显著占比，再改为
  “累计已提交 bytes + 定期全目录 audit”，并保留最终完整复核。

### P2：突然断电/重启会留下 JSONL 尾部半条记录

- 证据：系统在 Epoch 18 validation 时重启，`training_metrics.jsonl` 留下 205 bytes 的半条
  JSON；其 SHA-256 已记录在 V1 status，尾部恢复后 7,716 行均可解析。
- 影响：正常 Python exception 会进入 trainer 的 failed status，但机器级重启不会执行
  `__exit__` 或最终 summary，W&B 也会显示 crashed。
- 当前处理：保留系统启动时间、丢弃片段 hash、完整 Epoch 数、未完成 validation 进度和
  selected checkpoint；状态明确为 interrupted，不伪装为 complete/early-stopped。
- 后续建议：为通用 `TrainingLogger` 增加启动时“仅允许修复最后一个无换行且不可解析的
  fragment”的显式恢复入口，并给该合同增加断电模拟测试。由于本项目约定不恢复 optimizer
  轨迹，本次不从 Epoch 17 续训。

## 已验证合同

- 同一 Episode ID 的 payload hash 不一致时立即失败。
- winner 必须是唯一正 reward 且状态为 `DONE`；draw、active 和 malformed Episode 被排除。
- 注册卡组必须恰为 60 张，model input 保留 card ID 与 multiplicity。
- split 以完整 Episode 为单位，不会把同局决策拆到 train 与 validation。
- raw ZIP 逐 member 流式读取，不完整解压到磁盘。
- 0019 dataset 总目录（raw + model-ready + manifests）合计受 100 GiB 上限约束。
- Linux 保留至少 50 GiB，Windows C 盘保留至少 30 GiB。
- checkpoint 只保存模型权重与必要 metadata，不保存 optimizer/RNG/resume state。
- 正式 W&B 项目与 run name 合同为
  `dragon_bra/pokemon-tcg-policy-learning` 和
  `0019 · universal_winner_bc · V<n>_<tag>`。
- 项目生产模块均可 import，R15 teacher-forced 与 greedy 前向 smoke 已通过。

## 下一步建议顺序

1. 固化 Epoch 13 作为 V1 conditioned baseline；不要继续写入已经 interrupted 的版本。
2. V2 只增加 deterministic source dropout 或 neutral distillation 二者之一，不能同时更改。
3. 用 neutral validation 检查 source-0 gap，再以 deck-specific official-engine 评测判断强度。
4. V1/V2 之后再考虑 source/deck 分层采样或通用 Encoder + deck-specific adapter。
