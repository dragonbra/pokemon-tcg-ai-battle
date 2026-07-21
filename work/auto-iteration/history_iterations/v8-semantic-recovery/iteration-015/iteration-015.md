# iteration-015

## Context

- 类型：`full`
- control：iteration-014
- candidate：`work/alakazam_v8_current`
- 范围：17 × 10 = 170 局
- trace：`/tmp/ptcg-v8-iteration-015-full`

## Result

- W/L/D：`41/129/0`
- candidate errors：`0`
- 原始胜率：`24.1%`
- Meta 加权胜率：`23.4%`
- 第二回合实际 `attackId=1072`：`6/170`
- post-KO：`189`；zero-ready：`166/189`
- 出现打手断档的对局：`78/170`
- analyzer cases：`523`

## Decision

full 结果回退，说明 iteration-013/014 的 focused 改善不足以解释完整样本收益，继续
观察并定位更早的 setup/attachment 分歧。

