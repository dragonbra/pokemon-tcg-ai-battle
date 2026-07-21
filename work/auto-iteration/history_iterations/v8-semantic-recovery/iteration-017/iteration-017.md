# iteration-017

## Context

- 类型：`focused`
- control：iteration-016
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局
- trace：`/tmp/ptcg-v8-iteration-017-focus`

## Result

- W/L/D：`4/8/0`
- candidate errors：`0`
- 第二回合实际 `attackId=1072`：`1/12`
- post-KO：`15`；zero-ready：`12/15`
- 出现打手断档的对局：`7/12`
- analyzer cases：`32`

## Decision

focused 继续显示 post-KO 接力不稳定，记录为 `observe`，不能据此替换 full baseline。

