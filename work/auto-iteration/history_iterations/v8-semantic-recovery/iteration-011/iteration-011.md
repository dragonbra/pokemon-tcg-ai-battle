# iteration-011

## Context

- 类型：`focused`
- control：iteration-010
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局
- trace：`/tmp/ptcg-v8-iteration-011-focus`

## Result

- W/L/D：`5/7/0`
- candidate errors：`0`
- 第二回合实际 `attackId=1072`：`2/12`
- post-KO：`13`；zero-ready：`10/13`
- 出现打手断档的对局：`6/12`
- analyzer cases：`48`

## Interpretation and decision

correctness 保持通过，focused 的第二回合 Powerful Hand 有所出现，但 bench insurance
和 post-KO 资源缺口仍明显。记录为 `observe`，不能替代 full promotion Sample。

