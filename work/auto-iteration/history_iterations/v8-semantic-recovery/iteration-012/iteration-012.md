# iteration-012

## Context

- 类型：`full`
- control：iteration-011
- candidate：`work/alakazam_v8_current`
- 范围：17 × 10 = 170 局
- trace：`/tmp/ptcg-v8-iteration-012-full`

## Result

- W/L/D：`46/123/1`
- candidate errors：`0`
- 外部 evaluator error：`1`，`yakitori_raging_bolt` game 10 的对手侧 `effect 1197 IndexError`
- 原始胜率：`27.1%`
- Meta 加权胜率：`26.5%`
- 第二回合实际 `attackId=1072`：`3/170`
- post-KO：`156`；zero-ready：`130/156`
- 出现打手断档的对局：`69/170`
- analyzer cases：`487`

## Interpretation and decision

本轮没有 candidate correctness error；外部异常单独记录，不能作为候选策略通过或失败
证据。full 指标仍显著低于 Target，继续恢复旧版的后场接力语义，记录为 `observe`。

