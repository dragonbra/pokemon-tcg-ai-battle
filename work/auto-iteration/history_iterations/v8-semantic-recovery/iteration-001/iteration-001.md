# iteration-001

## Context

- 类型：`strategy`
- control：Start Baseline，见 [start-baseline](start-baseline.md)
- candidate：`work/alakazam_v8_current`
- 范围：17 × 10 = 170 局 full Sample
- trace：`/tmp/ptcg-v8-iteration-001-full.HsH5U5`

## Hypothesis and change

先修复阻断所有策略观察的 effect dispatch correctness，并补齐手牌索引解析，使无
显式 `cardId` 的进化/附能 option 能映射到语义卡牌。对应改动为：

- `dispatcher.py` 补充 `KADABRA` import；
- `options.py` 对 `Area.HAND` 的进化和附能统一从己方手牌索引解析卡牌。

## Result

- W/L/D：`8/162/0`
- 原始胜率：`4.7%`
- agent error：`0`
- 第二回合实际 `attackId=1072`：`0/170`
- post-KO ready attacker：`0/145`

## Interpretation and decision

NameError 已消失，说明 correctness 假设成立；策略表现仍远低于 Target，说明语义
恢复尚未开始完成。`observe`。下一轮聚焦 Active/Bench Abra 的低阶段攻击和进化优先级。

