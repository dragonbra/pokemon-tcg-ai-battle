# iteration-019

## Context

- 类型：`focused`
- control：iteration-018
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局
- trace：`/tmp/ptcg-v8-iteration-019-focus`

## Result

- W/L/D：`7/5/0`
- candidate errors：`0`
- 第二回合实际 `attackId=1072`：`0/12`
- post-KO：`10`；zero-ready：`8/10`
- 出现打手断档的对局：`3/12`
- analyzer cases：`36`

同批独立 Target control 为 `9/3/0`、第二回合 Powerful Hand `3/12`；该 control 只用于
语义对照，不是 candidate promotion 证据。

## Decision

focused 结果暂不晋级，继续追踪 candidate 与 oracle 的 first divergence。

