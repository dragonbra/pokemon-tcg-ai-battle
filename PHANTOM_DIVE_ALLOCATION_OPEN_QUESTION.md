# Phantom Dive Allocation：后续值得探讨的问题

日期：2026-08-09

## 当前结论

- Phantom Dive 的 6 个伤害指示物已建模为一个完整 macro action。
- 对手 Bench 有 `n=1..5` 个合法目标时，一次枚举 `1、7、28、84、210` 个 canonical allocation。
- Root Attack 与 allocation 使用层级概率，allocation 数量不会改变 Root softmax 的概率质量。
- 官方引擎仍逐次接收原格式 primitive `select`；相同 allocation 的最终权威状态、reward、terminal 和合法后继与旧路径一致。
- 六次内部 callback 不再形成六个 PPO timestep，也不重复计算 Policy/Value 或应用额外 gamma/lambda。

## 尚未完全一致的部分

新 allocation head 的初始选择偏好并不严格等于旧 Zero-Shot sequential policy。当前 BC 使用 sampled final allocation 标签，validation top-1 约为 49.2%。canonical 2,048 局观察为：

- legacy sequential：57.52%；
- forced shortcut、关闭 macro：57.52%；
- 完整 Phantom macro：56.74%；
- 旧 007 报告：57.32%。

因此，官方动作执行语义已经一致；剩余问题是新旧策略在 update-0 对合法 allocation 的概率分布不同，而不是训练和评测采用不同的游戏规则。

## PPO 可能带来的变化

穷举只负责提供全部合法完整分配，不会自动判断最优方案。PPO 会根据完整动作后的 advantage 更新：

```text
P(allocation | state, Phantom Dive)
```

如果某种分配持续带来更高胜率、立即奖赏或后续双奖机会，其概率应逐渐提高。它不保证与旧 sequential policy 越来越相似；目标是学习更有效的完整分配，而不是长期模仿旧策略。

## 建议继续验证

1. 将 allocation BC 从单个 sampled label 升级为 canonical soft-distribution distillation。
2. 把所有产生相同 count vector 的选择顺序聚合为同一 allocation 概率质量。
3. 固定状态集跟踪 allocation KL、top-1 一致率、count MAE、立即 KO/奖赏一致率。
4. 在 Frozen 2,048 中单独报告 Phantom Dive 子集的 paired outcome flips。
5. 确保 rollout、Frozen evaluation 和最终部署始终使用同一 macro planner 与 official primitive executor。

这是后续研究项，不改变当前官方协议，也不应把旧 007 与 macro update-0 之间的差异误记为 PPO 收益或损失。
