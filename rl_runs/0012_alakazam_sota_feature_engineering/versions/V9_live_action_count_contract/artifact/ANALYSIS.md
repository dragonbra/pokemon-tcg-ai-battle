# V9_live_action_count_contract analysis

## 1. 符合假设的结果

- 首轮 V1/V2/V3/V5 official evaluation 的 `game_error` 可由 live observation 中
  `minCount=maxCount=17/20/22`、而 decoder 最多返回 16 个 option 解释。
- 保持 V1 checkpoint、deck 与 cg 完全不变，只恢复原始 min/max 并允许 pointer 继续
  自回归后，诊断评测从 196/200 finished、4 errors 变为 200/200 finished、0 errors。
- 新回归测试构造 `minCount=17`、20 options，确认 inference 返回 17 个不重复且范围合法
  的 index；0012 单测 11/11 通过。

## 2. 不符合或尚未证明假设的结果

- 冻结 dataset 仍过滤 action length >16 的记录，因此第 17 个及之后的排序只是 GRU
  权重外推，不等价于专家监督。
- 诊断 run 是独立随机 200 局，157 胜与旧 run 的胜局差不能解释为强度提升；V9 当前
  只证明 legality/completion 改善。
- opponent package 也可能在同类状态返回非法长度；candidate 修复不能替对手修复。

## 3. 下一步修改措施

- 用同一 V9 decoder 重导出 V1/V2/V3/V5 best-exact checkpoint，并分别跑完整正式
  catalog；只有 0 error 报告进入强度比较。
- 后续重新构建 dataset 时单独审计并纳入 >16 action 标签，再决定是否把训练
  `max_action_steps` 提升到 32；本版本不改变冻结 corpus。
- V7/V8 checkpoint 导出也统一使用此 live contract。

## 4. 本版本具体执行内容

- 训练参数与 checkpoint 权重：无变化。
- 推理输入：用 observation 原始 `minCount/maxCount` 覆盖训练 codec 的 16-step cap。
- 输出：GRU pointer 最多解码到真实 option count，并在 `minimum > maximum` 时 fail closed。
- 诊断 official engine：200 games，157-43，0 errors，0 unfinished；报告位于 `.tmp`，
  不替代后续正式 `rl_runs/evaluation/` 结果。
