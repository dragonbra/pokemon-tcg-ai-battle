# iteration-007

- 变更说明：统一保护会使 Powerful Hand 丢失当前确定 KO 的手牌消耗动作，覆盖 Nighttime Mine，同时保留非 KO Stadium 路线
- 主要假设：统一保护会使 Powerful Hand 丢失当前确定 KO 的手牌消耗动作，覆盖 Nighttime Mine，同时保留非 KO Stadium 路线
- 指标 profile：`auto_iteration_v8_setup_relay`
- profile revision：`2`
- 样本类型：`full`
- Control：`iteration-006`
- Decision：`promote`
- 原生报告：[Evaluation HTML](run-250289e1b92e48738482e79077d485ab/report.html)；[Evaluation Markdown](run-250289e1b92e48738482e79077d485ab/report.md)
- 样本：170 局，17 个对手；先手 85 局，后手 85 局

## 结果

- Correctness errors：1 / 170
- 总体胜率：71.2% (121/170)
- 先手胜率：76.5% (65/85)
- 后手胜率：65.9% (56/85)
- 二回合实际选择 `attackId=1072`：19.4% (33/170)
- 二回合实际攻击（先手 / 后手）：15.3% (13/85) / 23.5% (20/85)
- 四组件状态（仅观测）：Active Abra 90/170；Rare Candy 55/170；Alakazam 或检索路线 157/170；Psychic Energy 或 Hilda 137/170；四项同时满足 21/170
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 4.65 张 (791/170)；实际到达二回合 4.71 张 (791/168)
- 第二回合平均过牌张数（先手 / 后手）：4.13 张 (351/85) / 5.18 张 (440/85)
- 攻击但未拿奖赏（全部攻击）：20.2% (121/598)
- 攻击但未拿奖赏（Powerful Hand）：11.8% (63/535)
- 攻击结算审计：已结算 598 次，未完成 0 次，奖赏状态未知 0 次
- Post-KO 立即接力：160 / 486（32.9%）
- Post-KO 原始审计：native 顶层保留 legacy zero-ready 失败率 326 / 486；语义指标使用上一行 payload 成功率，不使用 legacy 失败率做 promotion。
- `recoverable_discard_miss`：84
- 低牌库区间消耗：2.0352941176470587

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
