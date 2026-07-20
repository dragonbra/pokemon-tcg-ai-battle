# Alakazam AutoIter iter-37 分析摘要

## 本轮改动

iter-36 已修复“Active 本回合能完成 Alakazam 攻击时不能把唯一 Psychic 让给 Bench”的
边界。本轮继续收窄另一处 handoff 误判：Telepath→孤立 Bench Abra 不是确定的下一回合
打手，因为 Telepath 本身不能检索进化牌。

实现位置为 `submission/alakazam_v7_auto_iter/main.py`：

- `_bench_handoff_preparation_due()` 的直接能量目标判断；
- `_is_concrete_bench_psychic_handoff_option()` 的 Telepath/Abra 判定；
- 保留独立 Bench insurance 的 Telepath 路径。

## 评测结果

本轮使用固定 17 个对手 × 10 局，共 170 局 full trace：

| 指标 | iter-36 full | iter-37 full |
|---|---:|---:|
| 胜 / 负 / 平 | 105 / 64 / 1 | 117 / 50 / 3 |
| 胜率 | 61.8% | 68.8% |
| Meta 加权胜率 | 63.0% | 69.5% |
| 第二回合 Powerful Hand | 16.5% | 27.6% |
| post-KO 无 ready attacker | 59.6% | 59.0% |
| 对局级接力断档 | 39.4% | 30.0% |
| 空 Bench Run Away Draw | 0 | 0 |
| 我方 action error | 0 | 0 |

独立 focused 诊断批次（4 个对手 × 10 局）为 27/13/0，胜率 67.5%，第二回合
Powerful Hand 15.0%，post-KO 无 ready attacker 48.1%，对局级接力断档 55.0%，我方
action error=0。该 focused 批次仅用于分支执行确认，不与 full 矩阵混合解释。

## 行为验收

当前 full trace 中有 99 次“Telepath→无可见进化路线 Bench Abra”选项暴露，其中 28 次仍
由独立 Bench insurance 路径选择该目标；这条路径按设计没有被本轮修改。concrete handoff
gate 不再把它当作确定接力，4 个历史 case 的 old/candidate action 对比保存在 `cases.jsonl`
和 `decision.md`。

3 个未完成样本集中在 `yakitori_raging_bolt` 对手侧 `IndexError`，没有我方 action error。
完整原始 trace 位于隔壁评测仓库的 iter-37 目录；本 repo 不保存逐局 JSON。

## 限制与下一步

iter-37 与 iter-36、iter-15 都是独立随机批次，不能仅凭总胜率声称因果提升。当前策略
可作为下一轮 control，但仍需继续寻找“Bench 打手接力断档”的具体击倒前错误动作，并
同时守住第二回合 Powerful Hand 和当前回合攻击优先级。
