# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：170
- 对局：170；胜 / 负 / 平：127 / 43 / 0
- 我方错误：0
- 胜率：74.7%
- Meta 加权胜率：74.7%
- 第二回合 Powerful Hand：40/170 (23.5%)
- 我方被击倒事件：169
- 击倒后无 ready attacker：109/169 (64.5%)
- 出现过打手断档的对局：59/170 (34.7%)
- 空 Bench Run Away Draw：0

## 口径说明

`bench_insurance_missed` 的 240 条记录包含 analyzer 判定为 `pass` 的动作：它们表示
当前动作前存在 Bench insurance 检查，但 trace 已在同一回合完成铺场，不能当作 240 次
策略失败。`second_turn_powerful_hand_missing` 也包含合法选项不可得的诊断样本，不能直接
等同于“本来可以在第二回合攻击却漏做”。本轮的可执行策略 case 由原始 trace 单独复核。

iter-44 的实际修复是 terminal Powerful Hand 优先级：当当前攻击已经闭合最后奖赏时，
不再让 Fezandipiti ex 的抽牌 Ability 抢在攻击前。该修复的代表性 raw case 和后续 Telepath
接力 case 见 `decision.md` 与 `advisor.md`。

## Case 摘要

| failure_class | 数量 |
|---|---:|
| bench_insurance_missed | 240 |
| post_ko_no_ready_attacker | 109 |
| second_turn_powerful_hand_missing | 130 |

## 评测配置

```json
{
  "agent_label": "iter-44-terminal-fezandipiti-20260720",
  "command": "analyze",
  "report_dir": "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-44-terminal-fezandipiti-20260720"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
