# alakazam_v8_luna_deck_opt 评测摘要

评测日期：2026-07-21

使用 `scripts/alakazam_auto_iter.py run` 调用隔壁
`ptcg-agent-kaggle/eval/alakazam_replay.py`，17 个 opponent 各运行 10 局，交替先后手，
共 170 局。完整 trace 仅保留在 `/tmp/alakazam-v8-current-review`，不进入仓库。

## 结果

- 胜 / 负 / 平：`103 / 65 / 2`
- 原始胜率：`60.6%`
- Meta 加权胜率：`60.2%`
- 我方 agent error：`0`
- 第二回合实际使用 `Powerful Hand`：`44/170 (25.9%)`
- KO 后无立即可用接班打手：`109/191 (57.1%)`
- 出现打手断档的对局：`67/170 (39.4%)`
- 空 Bench 使用 `Run Away Draw`：`0`

二回合中没有发现“攻击合法但 agent 漏用 Powerful Hand”的 case；未使用的 126 局
属于目标回合没有合法的 `Powerful Hand` option。主要待优化方向是 KO 后接班打手和
Bench 保险，而不是 Powerful Hand 的动作优先级。

`yakitori_raging_bolt` 的两局未完成对局是对手策略触发 `IndexError`，错误步骤角色
为 `opponent`，不计为我方 agent error。

## 对局信号

- `crustle_wall`、`crustle_v1`：`10/10`
- `penguin_915`、`zoli_dragapult`：`8/10`
- `pilkwang_v2`：`7/10`；`yakitori_raging_bolt`：`7/10`，另有两局对手 error
- `kokinn_search`、`nursrijan_lucario`、`maktha_1084`、`yanxiaohan`：`6/10`
- `kiyotah_lucario`、`kiyotah_abomasnow`、`sue_alakazam`：`5/10`
- `romanrozen_v9`、`kiyotah_iono`：`4/10`
- `kiyotah_dragapult`、`kacchan_anti_wall`：`3/10`

本摘要用于保存本轮实验背景，不替代 Kaggle 正式结果。
