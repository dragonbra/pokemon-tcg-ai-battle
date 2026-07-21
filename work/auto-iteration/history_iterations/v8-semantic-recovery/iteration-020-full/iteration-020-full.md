# iteration-020-full

## Context

- 类型：`full`
- control：iteration-020
- candidate：`work/alakazam_v8_current`
- 范围：17 × 10 = 170 局
- trace：`/tmp/ptcg-v8-iteration-020-full`

## Result

- W/L/D：`51/118/1`
- candidate errors：`0`
- 外部 evaluator error：`1`，`yakitori_raging_bolt` game 10 的对手侧 `effect 1197 IndexError`
- 原始胜率：`30.0%`
- Meta 加权胜率：`29.5%`
- 第二回合实际 `attackId=1072`：`4/170`
- post-KO：`214`；zero-ready：`173/214`
- 出现打手断档的对局：`76/170`
- analyzer cases：`469`

## Interpretation and decision

candidate correctness 仍为 0，但 full 指标与 Target 的 `118/50/2`、`69.4%` 仍有巨大
差距。外部对手异常单独标注，记录为 `observe`，不宣称达到 Target。

