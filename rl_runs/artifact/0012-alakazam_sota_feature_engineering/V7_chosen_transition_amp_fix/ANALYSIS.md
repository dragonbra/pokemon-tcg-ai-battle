# V7_chosen_transition_amp_fix training analysis

## 1. 符合假设的结果

- V6 的 AMP stop-mask overflow 修复有效；V7 完成 21 个完整 epoch，没有再次出现数值
  overflow，并保留 best-loss、best-exact、latest 三套可恢复 checkpoint。
- 候选置换等变在全部 epoch 保持 semantic consistency `1.000000`、KL `0.000000`。
- 辅助任务持续收敛：validation delta MSE 从 epoch 1 的 `0.001777` 降到 epoch 21
  的 `0.000795`；turn-change BCE 从 `0.036646` 降到 `0.017793`。
- best exact 为 epoch 19 的 `0.843851`，说明显式 chosen-transition 表征没有破坏
  基本 BC 模仿能力。

## 2. 不符合或尚未证明假设的结果

- V7 best exact `0.843851` 低于 V5 `0.845023`，与 V3 `0.843792` 仅相差
  1/17,067 decision；没有证据证明 transition auxiliary 提升 imitation 上限。
- best exact checkpoint 的 validation loss `0.250039`，显著差于 V3 `0.220765`
  和 V5 best-loss `0.216032`；exact 与概率质量分叉明显。
- epoch 19 train/validation exact 为 `0.908296/0.843851`，gap `0.064446`；到
  epoch 21 validation loss 已恶化到 `0.256106`，后期过拟合明确。
- 辅助标签只覆盖专家已选动作到下一 decision 的观测变化，不是未选动作反事实，不能
  解释为 Q-value 学习。

## 3. 下一步修改措施

- 按用户指令停止 V7，不继续 epoch 22，也不自主启动 V8。
- V1--V7 指标交由用户选择 SOTA 基座；之后只基于指定结构修复 action length >16
  的训练/推理合同并重训一份。
- 长 action 修复必须重新审计冻结 source corpus 中此前被过滤的 >16 标签，不能只做
  inference fallback 后声称训练问题已解决。

## 4. 本版本具体执行内容

- experiment: `chosen_transition_amp_fix`；随机初始化。
- 结构：V5 primitive/context + 32-d transition trunk + 11-d delta head + turn head。
- loss：policy CE + `0.10 * delta MSE + 0.05 * turn BCE`。
- 完成 21 个完整 epoch；累计 completed-epoch runtime `7942.14s`。
- 用户在 epoch 22、约 batch 1000 时要求停止；进程以 KeyboardInterrupt 退出。
- best exact：epoch 19；best loss：epoch 11；latest：epoch 21。
