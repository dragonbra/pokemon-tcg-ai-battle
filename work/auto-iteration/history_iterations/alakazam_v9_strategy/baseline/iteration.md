# baseline

- 变更说明：固定 V8 最新版 60 张卡表与未修改策略的 17×10 基线
- 主要假设：固定 V8 最新版 60 张卡表与未修改策略的 17×10 基线
- 指标 profile：`auto_iteration_v8_setup_relay`
- profile revision：`2`
- 样本类型：`full`
- Control：`none`
- Decision：`observe`
- 原生报告：[Evaluation HTML](run-837ae45f0b944adc8202f8157cdb5f13/report.html)；[Evaluation Markdown](run-837ae45f0b944adc8202f8157cdb5f13/report.md)
- 样本：170 局，17 个对手；先手 85 局，后手 85 局

## 结果

- Correctness errors：3 / 170
- 总体胜率：67.6% (115/170)
- 先手胜率：70.6% (60/85)
- 后手胜率：64.7% (55/85)
- 二回合实际选择 `attackId=1072`：17.1% (29/170)
- 二回合实际攻击（先手 / 后手）：14.1% (12/85) / 20.0% (17/85)
- 四组件状态（仅观测）：Active Abra 103/170；Rare Candy 65/170；Alakazam 或检索路线 161/170；Psychic Energy 或 Hilda 114/170；四项同时满足 24/170
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 5.25 张 (893/170)；实际到达二回合 5.38 张 (893/166)
- 第二回合平均过牌张数（先手 / 后手）：4.88 张 (415/85) / 5.62 张 (478/85)
- 攻击但未拿奖赏（全部攻击）：20.2% (122/605)
- 攻击但未拿奖赏（Powerful Hand）：12.1% (66/546)
- 攻击结算审计：已结算 605 次，未完成 0 次，奖赏状态未知 0 次
- Post-KO 立即接力：185 / 529（35.0%）
- Post-KO 原始审计：native 顶层保留 legacy zero-ready 失败率 344 / 529；语义指标使用上一行 payload 成功率，不使用 legacy 失败率做 promotion。
- `recoverable_discard_miss`：99
- 低牌库区间消耗：3.223529411764706

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
