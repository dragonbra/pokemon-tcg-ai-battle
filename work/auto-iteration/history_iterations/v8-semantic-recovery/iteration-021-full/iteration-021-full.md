# iteration-021-full

- 变更说明：恢复 bench insurance / handoff 语义后的 full rerun；candidate errors 0
- 指标 profile：`v8-setup-relay`
- Decision：`observe`
- 样本：170 局，17 个对手；先手 95 局，后手 75 局

## 结果

- 总体胜率：34.1% (58/170)
- 先手胜率：35.8% (34/95)
- 后手胜率：32.0% (24/75)
- 二回合实际选择 `attackId=1072`：5.9% (10/170)
- 二回合实际攻击（先手 / 后手）：6.3% (6/95) / 5.3% (4/75)
- 四组件状态（仅观测）：Active Abra 89/170；Rare Candy 62/170；Alakazam 或检索路线 157/170；Psychic Energy 或 Hilda 127/170；四项同时满足 20/170
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 2.79 张 (475/170)；实际到达二回合 3.00 张 (471/157)
- 第二回合平均过牌张数（先手 / 后手）：2.77 张 (263/95) / 2.83 张 (212/75)
- 攻击但未拿奖赏（全部攻击）：41.1% (184/448)
- 攻击但未拿奖赏（Powerful Hand）：23.8% (82/345)
- 攻击结算审计：已结算 450 次，未完成 0 次，奖赏状态未知 2 次
- Post-KO 立即接力：4 / 103（3.9%）
- `recoverable_discard_miss`：96

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
