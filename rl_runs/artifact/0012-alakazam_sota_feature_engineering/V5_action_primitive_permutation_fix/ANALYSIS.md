# V5_action_primitive_permutation_fix training analysis

## 1. 符合假设的结果

- 训练完成并保留 minimum-loss 与 maximum-exact 两套 checkpoint。
- best validation exact-action: `0.845023`（epoch 18）。
- best validation token loss: `0.216032`（epoch 11）。
- permutation validation exact-action: `0.844788`。
- 修复后的 primitive/context 表示小幅刷新 exact SOTA：比 V3 `0.843792`
  高 `0.001230`（约 0.12 个百分点）。
- 更清晰的收益来自分布质量：best-loss checkpoint 的 validation loss `0.216032`，
  比 V3 `0.220765` 下降 `0.004733`（约 2.14%），同时 exact 为
  `0.843851`，也未丢失命中率。
- 修复确实恢复了严格置换等变：semantic consistency `1.000000`、
  teacher-forced KL `0.000000`；V4 的编码污染不再存在。

## 2. 不符合或尚未证明假设的结果

- 单个版本的离线曲线不能证明策略强度，也不能把 BC 表示解释为真实 action value。
- primitive 对 exact 的收益只有 21/17,067 decisions，还不足以在未做 episode-group
  bootstrap 时声称统计稳健超过 V3。
- best exact 与 best loss 分别出现在 epoch 18/11；epoch 18 loss `0.237962`
  已比 epoch 11 恶化 10.15%。不能只根据 exact 将后期 checkpoint 解释为全面更强。
- epoch 23 train/validation exact 为 `0.917782/0.840570`，gap `0.077212`；
  validation loss 恶化到 `0.265809`，过拟合证据明确。

## 3. 下一步修改措施

- V5 保留两个晋级候选：`best_validation.pt` 用于分布质量/校准路线，
  `best_exact.pt` 用于严格按当前 BC exact 主指标的路线。官方 engine 评测
  将分开归档，不混淆 checkpoint。
- 运行 V6 `chosen_transition_aux`：使用同一 primitive 结构和 transition shard，
  只增加专家所选 action 到下一 decision 可观测 delta/turn-change 辅助目标。
- 后续统一运行 select context、option count、single/multi-action slice 与 episode-group
  bootstrap，再决定 V3/V5 哪个 checkpoint 作为最终 BC 主线。

## 4. 本版本具体执行内容

- experiment: `action_primitive_permutation_fix`
- remove_option_position: `True`
- role_separated_action: `True`
- permutation_augmentation: `True`
- seed: `20260723`；batch: `96`；LR: `0.0003`
- stopped_early: `True`；last epoch: `23`
- parameter_count: `7481666`；peak GPU MiB: `848.95`
- runtime: `7048.93s`；冻结 238,356 decision identity/split 不变，feature dataset audit
  与修复后 source hashes 均写入 `training_config.json`。
