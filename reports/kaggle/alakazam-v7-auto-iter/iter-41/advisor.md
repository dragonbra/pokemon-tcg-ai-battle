# iter-41 Strategy Advisor

## 规则核验

- 宝可梦 TCG 允许在攻击前手动附能，但附能不是强制动作；一旦宣告攻击，回合立即结束。
- 如果 `_v7_terminal_prize_closure` 已确认当前攻击能拿完最后奖赏，Bench 附能不会再产生本局收益，先攻击是合法且符合奖赏闭环的选择。
- 本轮只降低 `option_type == 8` 的优先级，没有禁止附能，也没有改变非终局的 Bench handoff。
- closure 应继续同时检查 Active、Psychic Energy、实际伤害和剩余奖赏，不能把“存在攻击选项”直接视为终局。

## 评测信号

iter-41 相对 iter-40：

- 胜率 60.0% → 60.0%，胜数均为 102；
- Meta 加权胜率 61.0% → 60.1%；
- 第二回合 Powerful Hand 23.5% → 21.8%；
- post-KO 无 ready attacker 62.4% → 62.0%，事件数 205 → 192；
- 对局级接力断档 38.2% → 31.8%；
- 我方 action error 仍为 0。

post-KO 和对局级接力方向略好，但没有同步带来胜率、Meta 或二回合改善，不能据此晋升。由于是独立随机批次，下降也不能单独证明本轮规则有害。

## 下一轮建议

1. 从 iter-41 trace 中筛选 closure 为真的局面，逐一确认是否确实为最后奖赏；重点看 Bench 有未充能 Kadabra/Alakazam 时是否存在误判。
2. 定位二回合 Powerful Hand 未发生的真实动作链，确认是否有本轮 Energy 低优先级影响，而不是把独立样本波动误认为因果。
3. 在证据明确前，保留终局攻击优先的规则边界，但不放宽二回合 Powerful Hand 或 Night/Lana 恢复 gate。

结论：**observe**；规则语义接受，默认策略晋升暂缓。
