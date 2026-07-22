# 旧 BC/RL 探索结论

本轮清理前的 `rl/runs` 生成数据已经删除。以下结论来自仍保留的设计记录和
evaluation 摘要，作为新 Yushin Ito baseline 的研究背景，不作为可复现实验输入。

## 已确认的事实

- 规则 teacher 的大数据 BC 可以把合法候选动作分类学到较高离线准确率，但离线
  accuracy 不是闭环策略强度。1 GB 规则 teacher 数据有 179,947 条主动作记录，
  validation accuracy 为 85.42%，teacher-fallback candidate 在固定 17×10 评测中为
  121/170，低于 teacher 的 124/170。
- v6 的 action-card 特征使输入描述更完整，但直接纯模型接管的小样本探索只有
  12/34；加 0.95 residual gate 后为 27/34，说明“更像输入”没有自动变成“更强策略”。
- 终局胜负粗粒度复制到每个 action、visible potential reweighting 和当前第一版
  terminal PPO 都没有稳定提升 primary outcome。reward shaping 指标改善不能替代
  固定对手矩阵的胜率、完成率和 engine error 护栏。
- DAgger/MCTS/PPO 的现有实现验证了接口和合法 action mask，但没有产生可晋级的
  checkpoint；主要问题是 teacher 分布与 candidate rollout 分布不同，以及当前 effect
  selection contract 仍不完整。

## 研究收获

1. BC baseline 必须以完整合法候选集合为条件，而不是学习全局动作编号。
2. 必须按整局划分 train/validation/test；随机拆 action 会高估泛化。
3. 纯模型实验必须把 effect、目标和多选动作纳入 action contract；只模仿主动作时，
   仍然依赖规则 handler，不能声称是纯神经策略。
4. 下一轮应先冻结一个来源清晰、数据版本清晰的专家 BC checkpoint，再逐个消融
   reward，避免数据、模型和 reward 同时变化。
