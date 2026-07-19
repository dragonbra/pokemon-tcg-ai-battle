# Alakazam AutoIter 分析

## 样本与结果

- 实际读取 trace 文件：74
- 对局：74；胜 / 负 / 平：32 / 41 / 1
- 我方错误：1
- 胜率：43.2%
- Meta 加权胜率：52.5%
- 第二回合 Powerful Hand：12/74 (16.2%)
- 我方被击倒事件：74
- 击倒后无 ready attacker：56/74 (75.7%)
- 出现过打手断档的对局：27/74 (36.5%)
- 空 Bench Run Away Draw：2

## Case 摘要

| failure_class | 数量 |
|---|---:|
| empty_bench_run_away_draw | 2 |
| post_ko_no_ready_attacker | 56 |
| second_turn_powerful_hand_missing | 62 |

## 第一轮迭代假设

本轮先处理一个明确的硬错误：当 Active 是 Dudunsparce 且 Bench 为空时，禁止选择
`Run Away Draw`。该动作会把唯一的 Active 洗回牌库，不能被解释为正常的抽牌节奏。

候选版本只改 `submission/alakazam_v7_auto_iter/main.py`：为这个局面提高该 option 的
拒绝优先级，并正确解析 evaluator 通过 `area/indexInArea` 指向 Active 的 option。
不修改 `deck.csv`，也不同时调整进化、能量或牌库保护策略。验收顺序为：case 回归为
零、我方 action error 不增加，再观察第二回合攻击、击倒后接力和胜率 guardrail。

## 评测配置

```json
{
  "agent_label": "alakazam_v7",
  "command": "analyze",
  "report_dir": "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7"
}
```

原始 trace 保留在评测框架 report 目录；本文件只记录实际读取的 trace 覆盖范围。
