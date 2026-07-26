# V4_action_primitive_context failure analysis

## 1. 符合假设的结果

- 结构化 primitive 字段能够进入模型并完成 full-train/full-validation epoch；
  epoch 1 original validation exact 为 `0.760122`。
- 永久 permutation consistency 指标成功把隐蔽的 option-feature 错配暴露出来，
  阻止了对一个无效实验继续消耗 GPU。

## 2. 不符合假设的结果

- V4 不是有效的 action primitive 消融。`permute_candidates()` 重排了
  `option_cat`，但没有同步重排 `option_primitive_cat`。
- 该缺陷同时污染了训练和评测：permutation augmentation 会把 attack/count/
  energy/tool primitive 配给另一个 candidate。
- epoch 1 semantic prediction consistency 降为 `0.991914`，teacher-forced KL
  升为 `0.000340252`；对于无 position 的模型，这与 V2/V3 的 `1.0/0.0`
  不一致，是直接失效证据。

## 3. 下一步修改措施

- 修复 candidate permutation，对所有 option-aligned tensor 使用同一
  `new_to_old` permutation；当前 schema 至少包括 `option_cat` 和
  `option_primitive_cat`。
- 增加单元测试，同时断言 primitive 与 candidate 的行对齐以及完整
  action-primitive 模型 logits 的 permutation equivariance。
- 新建 `V5_action_primitive_permutation_fix`，从随机初始化重新训练；不恢复、
  不覆盖 V4。

## 4. 本版本具体执行内容

- 完成 epoch 1 和 epoch 2 的前 1600/2306 batches 后人工中止。
- V4 checkpoint、training config、epoch-1 metrics 和 TensorBoard event 全部保留，
  仅用于失败审计，不得导出 candidate 或形成策略强度结论。
- 根因定位到 `train/project_0012_alakazam_sota_feature_engineering/batching.py`；回归测试在
  `tests/test_alakazam_sota_feature_engineering.py` 中覆盖。
