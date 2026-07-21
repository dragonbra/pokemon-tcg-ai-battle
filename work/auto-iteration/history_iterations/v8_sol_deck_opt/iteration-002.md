# Iteration 002：Fezandipiti 触发与 Dunsparce bridge 起点

## 变更

- Fezandipiti ex：上一回合己方 Pokémon 被 KO 时，始终优先使用 `Flip the Script`，不再因为两 Prize 风险、当前已有 KO 或终局判断跳过 Ability。
- Active Dunsparce：后场存在已带 Psychic 的 Alakazam 时，保留 Hilda → Dudunsparce → `Run Away Draw` 的接力路线，即使 Dunsparce 当回合可以取得低价值 KO。
- Poké Pad：第一回合已有两只 Abra 与一只 Dunsparce 的基础资源时保留 Poké Pad；Active Dunsparce 且牌库候选同时有 Abra/Dunsparce 时优先找 Abra 作为攻击线底座。

## 测试证据

- RED：Active Dunsparce 低价值 KO、首回合 Poké Pad 保留、Active Dunsparce 的 Poké Pad Abra 目标、Fezandipiti Prize 风险四项测试先失败。
- GREEN：定向测试 `10/10` 通过。
- 回归：`python3 -m unittest discover -s tests -p 'test_*.py'`，`415/415` 通过。

## 评测

- Run：[`run-124abc8103df4cee96c0c70463b40f28`](iteration-002/run-124abc8103df4cee96c0c70463b40f28/)

| 指标 | Baseline | Iteration 002 |
|---|---:|---:|
| 总体胜率 | 103/170 = 60.6% | 119/170 = 70.0% |
| Powerful Hand | 34/170 = 20.0% | 35/170 = 20.6% |
| Post-KO relay | 154/523 = 29.4% | 175/479 = 36.5% |
| recoverable_discard_miss | 130 | 72 |
| 攻击未拿奖赏 | 137/584 = 23.5% | 143/621 = 23.0% |
| Dunsparce bridge | 0/44 | 0/53 |
| 空 Bench Run Away Draw | 0 | 0 |
| Rare Candy 进化链 | 34/170 = 20.0% | 35/170 = 20.6% |

## 正确性与结论

- Iteration 002 的 1 个 error 仍为 `yakitori_raging_bolt` 对手侧 engine error；没有 candidate error。
- 评测总体结果继续上升，且恢复型接力失败下降；但 Powerful Hand 仅小幅上升，Dunsparce bridge 仍未完成。
- `--keep-temp` 调试显示部分 bridge opportunity 实际没有后场宝可梦，不能凭空执行 Run Away Draw；另一个重要失败是首回合 Active Dunsparce 时 Poké Pad 原先优先找 Dunsparce，导致攻击底座没有建立。本轮已修正目标选择，下一轮通过同口径评测确认。
