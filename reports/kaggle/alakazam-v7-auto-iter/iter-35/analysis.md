# Alakazam AutoIter iter-35 分析摘要

## 样本与结果

本轮使用 evaluator 默认独立随机、swap 开启的 5 个对手 × 10 局 full-trace。完整 trace
只保存在 `/tmp/alakazam-v7-auto-iter-35-energy-handoff.TrjlYi/`，没有复制进主 repo。

| 指标 | 结果 |
|---|---:|
| 胜 / 负 / 平 | 25 / 25 / 0 |
| 胜率 | 50.0% |
| Meta 加权胜率 | 51.0% |
| 第二回合 Powerful Hand | 5/50 = 10.0% |
| 被击倒事件 | 98 |
| 击倒后无 ready attacker | 58/98 = 59.2% |
| 出现过打手断档的对局 | 29/50 = 58.0% |
| 我方 action error | 0 |
| 空 Bench Run Away Draw | 0 |

与 iter-34 focused 的 62.0% 胜率相比，本批样本明显更低；由于随机发牌和样本范围
不同，不能判定策略造成了 12 个百分点的因果下降。与目标相关的 post-KO event rate
从 75.3% 降至 59.2%，对局级断档从 60.0% 降至 58.0%，但第二回合指标也从 14.0% 降至
10.0%，所以当前不能晋升。

## 策略触发核验

重新读取 50 个 full trace，并用生产 helper 对每个主行动 observation 计算触发条件：

- 触发 32 次；
- 32 次实际选择均为 `inPlayArea=5` 的 Bench Psychic 目标；
- 其中目标包括 Kadabra/Alakazam，以及有 Telepath 路线的 Abra；
- 无触发时，Bench 不具备确定 handoff，策略保留 Active 附能。

这说明本轮改动真实地作用于 evaluator legal options，而不是只有 fixture 层面的死代码。

## 失败指标的解释边界

当前 analyzer 的 `post_ko_no_ready_attacker` 仍是“击倒后没有立即 ready”的诊断，不等于
击倒前每次都存在免费替代动作；本轮只把三个已 forensic 核验的 Active/Bench Energy
目标 case 转成策略规则。`bench_insurance_missed` 也同时包含 pass/fail，不能用总数
直接评价策略。

## 决策

**observe，不晋升。** 保留本轮窄规则和成对反例 fixture，继续观察其对接力事件的影响；
下一轮优先分析为什么第二回合指标下降，以及是否存在“Bench 目标虽然可充能，但本轮应
优先使用 Rare Candy/进化或攻击”的具体 case。不要无条件扩大 Bench 优先范围。
