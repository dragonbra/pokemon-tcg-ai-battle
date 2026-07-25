# V2_no_position_permutation training analysis

## 1. 符合假设的结果

- 训练完成并保留 minimum-loss 与 maximum-exact 两套 checkpoint。
- best validation exact-action: `0.837698`（epoch 10）。
- best validation token loss: `0.229691`（epoch 7）。
- permutation validation exact-action: `0.837581`。
- 核心置换假设得到完整支持：best-exact checkpoint 的 original/permuted exact 仅相差
  `0.000117`，semantic prediction consistency 为 `1.000000`，teacher-forced KL
  为 `0.000000`。
- 去除位置捷径的原始顺序代价很小：V2 best exact 比 V1 的 `0.838929`
  低 `0.001230`（约 0.12 个百分点），而 permutation exact 比 V1 的 `0.751040`
  高 `0.086541`（约 8.65 个百分点）。
- 与 V1 一样，early-stop 捕获到过拟合：epoch 15 train exact `0.899656`，
  validation exact `0.833538`，gap `0.066118`。

## 2. 不符合或尚未证明假设的结果

- 单个版本的离线曲线不能证明策略强度，也不能把 BC 表示解释为真实 action value。
- best loss 与 best exact 分别出现在 epoch 7/10，说明 checkpoint 选择必须与目标对齐；
  BC 主对照使用 `best_exact.pt`，但保留 `best_validation.pt` 供校准研究。
- V2 不能单独证明 action feature 语义更强；它证明的是候选顺序不再改变预测。
  是否提升官方 engine 策略强度仍需真实对局。

## 3. 下一步修改措施

- 把“无 option position + permutation augmentation”作为后续 action-side 实验的新基线。
- 运行 V3 `role_separated_action`，只增加 source/target 角色投影，测试相同
  card/entity ID 在动作参数中的不同语义是否能恢复或超过 V1 的模仿上限。
- V2 已达到离线晋级条件，后续将导出 `best_exact.pt` candidate，并与其他晋级
  版本一起用官方 engine 固定池评测。

## 4. 本版本具体执行内容

- experiment: `no_position_permutation`
- remove_option_position: `True`
- role_separated_action: `False`
- permutation_augmentation: `True`
- seed: `20260723`；batch: `96`；LR: `0.0003`
- stopped_early: `True`；last epoch: `15`
- parameter_count: `7154562`；peak GPU MiB: `842.10`
- runtime: `5035.39s`；dataset/split/source hashes 与 V1 一致，并写入本版本
  `training_config.json`。
