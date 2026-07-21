# iteration-006

- 变更说明：补齐 Poffin 精确 KO 阈值、已撤退后的全 Energy 死接力门控，以及 Night Stretcher 的 Abra→Dunsparce→未来 Energy 顺序
- 主要假设：补齐 Poffin 精确 KO 阈值、已撤退后的全 Energy 死接力门控，以及 Night Stretcher 的 Abra→Dunsparce→未来 Energy 顺序
- 指标 profile：`auto_iteration_v8_setup_relay`
- profile revision：`2`
- 样本类型：`full`
- Control：`baseline`
- Decision：`promote`
- 原生报告：[Evaluation HTML](run-a2611d7873a346a4a40828cbee605656/report.html)；[Evaluation Markdown](run-a2611d7873a346a4a40828cbee605656/report.md)
- 样本：170 局，17 个对手；先手 85 局，后手 85 局

## 结果

- Correctness errors：2 / 170
- 总体胜率：71.2% (121/170)
- 先手胜率：72.9% (62/85)
- 后手胜率：69.4% (59/85)
- 二回合实际选择 `attackId=1072`：17.1% (29/170)
- 二回合实际攻击（先手 / 后手）：14.1% (12/85) / 20.0% (17/85)
- 四组件状态（仅观测）：Active Abra 96/170；Rare Candy 53/170；Alakazam 或检索路线 154/170；Psychic Energy 或 Hilda 137/170；四项同时满足 17/170
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 5.19 张 (883/170)；实际到达二回合 5.29 张 (883/167)
- 第二回合平均过牌张数（先手 / 后手）：4.75 张 (404/85) / 5.64 张 (479/85)
- 攻击但未拿奖赏（全部攻击）：18.5% (109/589)
- 攻击但未拿奖赏（Powerful Hand）：11.9% (64/538)
- 攻击结算审计：已结算 589 次，未完成 0 次，奖赏状态未知 0 次
- Post-KO 立即接力：170 / 483（35.2%）
- Post-KO 原始审计：native 顶层保留 legacy zero-ready 失败率 313 / 483；语义指标使用上一行 payload 成功率，不使用 legacy 失败率做 promotion。
- `recoverable_discard_miss`：89
- 低牌库区间消耗：1.9176470588235295

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
