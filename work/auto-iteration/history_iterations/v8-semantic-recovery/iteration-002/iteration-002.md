# iteration-002

## Context

- 类型：`strategy`
- control：iteration-001
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局 focused
- trace：`/tmp/ptcg-v8-iteration-002-focus.ZWQ6y9`

## Hypothesis and change

上一轮 correctness 已通过，因此先用小范围对局确认动作链是否已经开始工作，暂不运行
新的 170 局 full Sample。重点观察进化、Psychic attachment 和攻击提交。

## Result

- 对手结果：`romanrozen_v9 0W/4L/0D`、`crustle_v1 2W/2L/0D`、`maktha_1084 0W/4L/0D`
- 总体 W/L/D：`2/10/0`
- agent error：`0`
- 观察到 Attack：`20`
- 其中 `attackId=1070`：`16`；`attackId=1072`：`4`
- Attach：`12`；Evolve：`4`

## Interpretation and decision

进化和附能动作已经出现，但 Active Abra 的 Teleportation 仍被提交，和旧版
`_is_forbidden_terminal_attack()` 的语义冲突。`observe`。下一轮先固定“Abra 不主动
攻击，攻击是终止提交”的回归测试和最小 commit gate。
