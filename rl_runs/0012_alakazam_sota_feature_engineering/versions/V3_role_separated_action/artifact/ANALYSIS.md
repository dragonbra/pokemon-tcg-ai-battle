# V3_role_separated_action training analysis

## 1. 符合假设的结果

- 训练完成并保留 minimum-loss 与 maximum-exact 两套 checkpoint。
- best validation exact-action: `0.843792`（epoch 10）。
- best validation token loss: `0.220765`（epoch 10）。
- permutation validation exact-action: `0.843675`。
- source/target 角色分离假设得到支持：V3 比 V2 best exact `0.837698`
  提升 `0.006094`（约 0.61 个百分点），也比带位置捷径的 V1 `0.838929`
  提升 `0.004863`（约 0.49 个百分点）。
- epoch 10 同时是 best exact 和 best loss，避免了 V2 的 checkpoint 目标分歧。
- 置换性质没有因角色投影破坏：original/permuted exact 仅差 `0.000117`，
  semantic consistency `1.000000`，teacher-forced KL `0.000000`。

## 2. 不符合或尚未证明假设的结果

- 单个版本的离线曲线不能证明策略强度，也不能把 BC 表示解释为真实 action value。
- 角色分离增加了 `204800` 参数（V2 的 2.86%）；虽在预定容量带内，仍不能
  把全部收益无条件归因为纯语义而完全排除容量效应。
- epoch 10 后连续 5 轮未创新高；最后一轮 train/validation exact 为
  `0.891219/0.840394`，gap `0.050825`，过拟合仍存在。

## 3. 下一步修改措施

- 把 V3 作为当前离线 SOTA 和后续 action-side 表示基线。
- 运行 V4 `action_primitive_context`：在 V3 上加入 attack/effect/context card、
  count/energy/tool 等候选动作原语义，测试结构化 action feature 是否超过
  仅依靠 token ID + role 的表示。
- V3 已达离线晋级条件，将导出 `best_exact.pt` candidate 并进入官方 engine
  固定池评测。

## 4. 本版本具体执行内容

- experiment: `role_separated_action`
- remove_option_position: `True`
- role_separated_action: `True`
- permutation_augmentation: `True`
- seed: `20260723`；batch: `96`；LR: `0.0003`
- stopped_early: `True`；last epoch: `15`
- parameter_count: `7359362`；peak GPU MiB: `849.28`
- runtime: `4294.44s`；冻结 dataset/split/source hashes 已写入本版本
  `training_config.json`。
