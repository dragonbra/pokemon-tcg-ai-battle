# V1_baseline_control training analysis

## 1. 符合假设的结果

- 训练完成并保留 minimum-loss 与 maximum-exact 两套 checkpoint。
- best validation exact-action: `0.838929`（epoch 9）。
- best validation token loss: `0.223857`（epoch 9）。
- permutation validation exact-action: `0.751040`。
- baseline 对 simulator 候选顺序存在明显依赖：best epoch 在原始顺序下的 exact
  为 `0.838929`，随机置换后为 `0.751040`，下降 `0.087889`（10.48% relative）。
- early-stop 确实捕获到了过拟合区间：epoch 9 后连续 5 轮未创新高，但最后一轮
  train exact 已升至 `0.892593`，而 validation exact 为 `0.835003`，gap 为
  `0.057590`。

## 2. 不符合或尚未证明假设的结果

- 单个版本的离线曲线不能证明策略强度，也不能把 BC 表示解释为真实 action value。
- 原始顺序下的高 exact 并不等价于学到了 action 语义；随机置换后 teacher-forced KL
  为 `0.411231`，semantic prediction consistency 仅 `0.851409`，说明输出分布
  会随候选排列发生实质变化。
- 训练在 epoch 9 后仍继续降低 train loss，但 validation loss 从 `0.223857`
  恶化到 epoch 13 的 `0.247295`；更长训练不会自动消除位置捷径。

## 3. 下一步修改措施

- 立即运行 V2 `no_position_permutation`：去掉 option position embedding，并在训练时
  随机置换候选与同步重映射 expert index。
- V2 首要验证的不是原始顺序 exact 单点超过 V1，而是 permutation exact 与
  original exact 的 gap 接近 0、semantic consistency 接近 1、teacher-forced KL 接近 0；
  同时尽量保持原始顺序的模仿质量。
- V1 作为后续所有版本的冻结 control；待 V2–V6 离线对照完成后，对值得晋级的
  checkpoint 导出独立 candidate，再用官方 engine 固定池检验策略强度。

## 4. 本版本具体执行内容

- experiment: `baseline_control`
- remove_option_position: `False`
- role_separated_action: `False`
- permutation_augmentation: `False`
- seed: `20260723`；batch: `96`；LR: `0.0003`
- stopped_early: `True`；last epoch: `14`
- parameter_count: `7154562`；peak GPU MiB: `838.74`
- runtime: `4450.65s` GPU training/evaluation；dataset train/validation decisions:
  `221289/17067`；test split 为空，未伪造 test 指标。
- 输入身份已写入 `training_config.json`：dataset audit SHA-256、train/validation
  output SHA-256、训练源码 SHA-256 均可追溯。
