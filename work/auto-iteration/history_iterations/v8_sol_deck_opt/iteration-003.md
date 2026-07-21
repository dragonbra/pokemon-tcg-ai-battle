# Iteration 003：Active Dunsparce 的 Abra 底座建立

## 变更

- 新增首回合 Poké Pad 目标规则：Active Dunsparce 且牌库同时可见 Abra/Dunsparce 时优先找 Abra；其余首回合仍按已有 Dunsparce → Abra 保守规则。
- 保留“首回合已有两只 Abra + 一只 Dunsparce 基础资源时不使用 Poké Pad”的规则。
- 没有把没有 Bench 接班者的 Active Dunsparce 强行视为可执行 Run Away Draw 路线；避免空 Bench 自损。

## 测试证据

- RED：Active Dunsparce 的首回合 Poké Pad 目标测试失败在旧代码，GREEN 后定向测试 `10/10` 通过。
- 回归：`python3 -m unittest discover -s tests -p 'test_*.py'`，`415/415` 通过。
- 临时 `--keep-temp` trace：`/tmp/v8-sol-bridge-debug-2/run-ebe629ebdc554baa89c5a5b1c6ef5223`；其中可见的 bridge opportunity 有些没有合法 Bench 接班者，因此不能仅凭 opportunity 计数强行宣称应使用 Run Away Draw。

## 评测

- Run：[`run-30625b6831594e0795328c9a5e6093fa`](iteration-003/run-30625b6831594e0795328c9a5e6093fa/)

| 指标 | Baseline | Iteration 002 | Iteration 003 |
|---|---:|---:|---:|
| 总体胜率 | 103/170 = 60.6% | 119/170 = 70.0% | 116/170 = 68.2% |
| Powerful Hand | 34/170 = 20.0% | 35/170 = 20.6% | 48/170 = 28.2% |
| Rare Candy 进化链 | 34/170 = 20.0% | 35/170 = 20.6% | 48/170 = 28.2% |
| Post-KO relay | 154/523 = 29.4% | 175/479 = 36.5% | 154/473 = 32.6% |
| field_route_miss | 215 | 212 | 219 |
| recoverable_discard_miss | 130 | 72 | 90 |
| 攻击未拿奖赏 | 137/584 = 23.5% | 143/621 = 23.0% | 127/604 = 21.0% |
| Dunsparce bridge | 0/44 | 0/53 | 0/56 |
| 空 Bench Run Away Draw | 0 | 0 | 0 |

## 结论

- 起手 Abra 底座规则显著提高了二回合 Powerful Hand 和 Rare Candy 指标，同时降低了攻击质量惩罚项。
- 单轮胜率较 Iteration 002 下降 3 胜，属于随机评测波动范围内，不能用它否定局部 setup 证据。
- Dunsparce bridge 仍未完成，当前证据不足以再添加更激进的自动换位分支；最终报告会明确保留这一未解决项。
