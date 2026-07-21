# iteration-002

- 变更说明：统一执行两只 Abra 系列优先、一只 Dunsparce 系列次之，并让 Lana/Night Stretcher 服务同一缺口
- 主要假设：统一执行两只 Abra 系列优先、一只 Dunsparce 系列次之，并让 Lana/Night Stretcher 服务同一缺口
- 指标 profile：`auto_iteration_v8_setup_relay`
- profile revision：`2`
- 样本类型：`full`
- Control：`iteration-001`
- Decision：`observe`
- 原生报告：[Evaluation HTML](run-01c6110c386c4d269399e305ba639db3/report.html)；[Evaluation Markdown](run-01c6110c386c4d269399e305ba639db3/report.md)
- 样本：170 局，17 个对手；先手 85 局，后手 85 局

## 结果

- Correctness errors：1 / 170
- 总体胜率：70.6% (120/170)
- 先手胜率：75.3% (64/85)
- 后手胜率：65.9% (56/85)
- 二回合实际选择 `attackId=1072`：29.4% (50/170)
- 二回合实际攻击（先手 / 后手）：24.7% (21/85) / 34.1% (29/85)
- 四组件状态（仅观测）：Active Abra 108/170；Rare Candy 64/170；Alakazam 或检索路线 162/170；Psychic Energy 或 Hilda 132/170；四项同时满足 26/170
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 5.54 张 (941/170)；实际到达二回合 5.60 张 (941/168)
- 第二回合平均过牌张数（先手 / 后手）：5.40 张 (459/85) / 5.67 张 (482/85)
- 攻击但未拿奖赏（全部攻击）：21.5% (132/613)
- 攻击但未拿奖赏（Powerful Hand）：13.5% (74/547)
- 攻击结算审计：已结算 613 次，未完成 0 次，奖赏状态未知 0 次
- Post-KO 立即接力：166 / 486（34.2%）
- Post-KO 原始审计：native 顶层保留 legacy zero-ready 失败率 320 / 486；语义指标使用上一行 payload 成功率，不使用 legacy 失败率做 promotion。
- `recoverable_discard_miss`：83
- 低牌库区间消耗：1.723529411764706

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
