# iteration-013

## Context

- 类型：`focused`
- control：iteration-012
- candidate：`work/alakazam_v8_current`
- 范围：`romanrozen_v9, crustle_v1, maktha_1084`，每个 4 局，共 12 局
- trace：`/tmp/ptcg-v8-iteration-013-focus`

## Hypothesis and change

补充 Active 为 Fezandipiti 等暂时不能攻击的 Pokémon 时，仍先完成合法 Bench
Abra/Kadabra/Dudunsparce 进化路线，再结束回合。

## Result

- W/L/D：`3/9/0`
- candidate errors：`0`
- 第二回合实际 `attackId=1072`：`0/12`
- post-KO：`5`；zero-ready：`5/5`
- 出现打手断档的对局：`3/12`
- analyzer cases：`22`

## Decision

回归测试覆盖了“不可攻击 Active 仍要建立 Bench 引擎”的语义，focused 运行无错误，
但小样本不能晋级，记录为 `observe`。

