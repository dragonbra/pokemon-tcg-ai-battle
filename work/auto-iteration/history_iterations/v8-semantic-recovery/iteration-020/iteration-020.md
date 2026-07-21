# iteration-020

## Context

- 类型：`focused`
- control：iteration-019
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局
- trace：`/tmp/ptcg-v8-iteration-020-focus`

## Result

- W/L/D：`3/9/0`
- candidate errors：`0`
- 第二回合实际 `attackId=1072`：`0/12`
- post-KO：`14`；zero-ready：`14/14`
- 出现打手断档的对局：`6/12`
- analyzer cases：`42`

## Decision

本轮 focused 回退，说明现有 handoff/insurance 修复仍依赖起手资源，不能作为 full
晋级依据。随后运行的 full control 见 `iteration-020-full`。

