# AutoIter 27 Strategy Advisor

## 规则核验

iter-25 的第二回合 Powerful Hand 快速优先只应越过泛化 setup，不能越过本回合已经
观察到的确定奖赏路线。若对手 Active 不能被当前攻击击倒，而 Boss's Orders 可以把
对手 Bench 的可击倒宝可梦拉到 Active，则 Boss 是本回合的 Supporter 预算，应先完成
Boss 选择，再攻击。

本轮 fixture 构造为：

- Active Alakazam 已附 Psychic，当前为己方第二回合；
- Bench 已有 Kadabra，因而不存在“必须先建立接力”的理由；
- Active 目标 HP 高于 Powerful Hand 伤害，Bench 目标 HP 低于该伤害；
- Boss's Orders 与 Powerful Hand 都是合法选项，且 Boss KO 不是最后奖赏闭环。

当前旧排序会因为第二回合指标 gate 选择 Powerful Hand。本轮将 Boss 的确定 KO 路线
置于该 gate 之前；终局、空 Bench insurance、保护能量、Xerosic 以及普通攻击排序不变。

## 评测注意

iter-27 中 `yakitori_raging_bolt` 的 2 场 `IndexError` 出现在 opponent action，
不是我方 agent error。策略分析仍以我方 `errors=0` 为准，但这两局不能作为完整胜负
样本的强证据。
