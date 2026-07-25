# V6_chosen_transition_aux failure analysis

## 1. 符合假设的结果

- transition shard、batch schema 和辅助 head 都成功初始化，失败发生在第一个
  GPU forward，没有污染任何 epoch 指标或 checkpoint。
- 失败被定位为 AMP dtype 屏蔽值问题，而不是 transition label 或 loss 设计问题。

## 2. 不符合假设的结果

- `teacher_logits_and_transition()` 为 stop tensor 填充了
  `torch.finfo(pointer.dtype).min`。在 CUDA autocast 下 pointer 与 stop 可以具有不同 dtype，
  FP32 minimum 无法写入 FP16 stop tensor，触发 overflow。
- V6 在首 batch 失败，不存在可解释的 BC 或 transition 指标。

## 3. 下一步修改措施

- stop mask 使用 `torch.finfo(stop.dtype).min`，与实际被写入 tensor 的 dtype 一致。
- 新增 CPU bfloat16 autocast 回归测试，覆盖 pointer/stop dtype 可不同时的安全
  masked fill；全部项目测试 10/10 通过。
- 新建 `V7_chosen_transition_amp_fix` 从随机初始化重训；history 版本顺延到 V8。

## 4. 本版本具体执行内容

- 已保留 `training_config.json`、空 `training_metrics.jsonl` 和 TensorBoard run 身份。
- 未创建 checkpoint，未完成 epoch，未产生可导出 candidate。
