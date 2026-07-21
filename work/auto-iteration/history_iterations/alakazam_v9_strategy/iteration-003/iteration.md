# iteration-003

- 变更说明：合法 Flip the Script 直接信任 simulator option，并保留终局、手牌和牌库安全门控
- 主要假设：合法 Flip the Script 直接信任 simulator option，并保留终局、手牌和牌库安全门控
- 指标 profile：`auto_iteration_v8_setup_relay`
- profile revision：`2`
- 样本类型：`full`
- Control：`iteration-002`
- Decision：`observe`
- 原生报告：[Evaluation HTML](run-067f8bcdea9747719acff6290b9c0d11/report.html)；[Evaluation Markdown](run-067f8bcdea9747719acff6290b9c0d11/report.md)
- 样本：170 局，17 个对手；先手 85 局，后手 85 局

## 结果

- Correctness errors：0 / 170
- 总体胜率：68.8% (117/170)
- 先手胜率：70.6% (60/85)
- 后手胜率：67.1% (57/85)
- 二回合实际选择 `attackId=1072`：29.4% (50/170)
- 二回合实际攻击（先手 / 后手）：29.4% (25/85) / 29.4% (25/85)
- 四组件状态（仅观测）：Active Abra 101/170；Rare Candy 70/170；Alakazam 或检索路线 156/170；Psychic Energy 或 Hilda 139/170；四项同时满足 27/170
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 5.04 张 (856/170)；实际到达二回合 5.22 张 (856/164)
- 第二回合平均过牌张数（先手 / 后手）：5.13 张 (436/85) / 4.94 张 (420/85)
- 攻击但未拿奖赏（全部攻击）：22.7% (134/590)
- 攻击但未拿奖赏（Powerful Hand）：15.3% (81/529)
- 攻击结算审计：已结算 590 次，未完成 0 次，奖赏状态未知 0 次
- Post-KO 立即接力：139 / 471（29.5%）
- Post-KO 原始审计：native 顶层保留 legacy zero-ready 失败率 332 / 471；语义指标使用上一行 payload 成功率，不使用 legacy 失败率做 promotion。
- `recoverable_discard_miss`：80
- 低牌库区间消耗：2.111764705882353

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
