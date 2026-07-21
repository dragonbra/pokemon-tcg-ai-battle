# iteration-023

- 变更说明：恢复 Active Alakazam 下的 Bench Dudunsparce 进化与 Run Away Draw focused 验证
- 指标 profile：`v8-setup-relay`
- Decision：`observe`
- 样本：12 局，3 个对手；先手 8 局，后手 4 局

## 结果

- 总体胜率：58.3% (7/12)
- 先手胜率：62.5% (5/8)
- 后手胜率：50.0% (2/4)
- 二回合实际选择 `attackId=1072`：0.0% (0/12)
- 二回合实际攻击（先手 / 后手）：0.0% (0/8) / 0.0% (0/4)
- 四组件状态（仅观测）：Active Abra 3/12；Rare Candy 7/12；Alakazam 或检索路线 11/12；Psychic Energy 或 Hilda 11/12；四项同时满足 0/12
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 3.25 张 (39/12)；实际到达二回合 3.55 张 (39/11)
- 第二回合平均过牌张数（先手 / 后手）：3.75 张 (30/8) / 2.25 张 (9/4)
- 攻击但未拿奖赏（全部攻击）：34.9% (15/43)
- 攻击但未拿奖赏（Powerful Hand）：20.0% (7/35)
- 攻击结算审计：已结算 43 次，未完成 0 次，奖赏状态未知 0 次
- Post-KO 立即接力：0 / 9（0.0%）
- `recoverable_discard_miss`：9

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
