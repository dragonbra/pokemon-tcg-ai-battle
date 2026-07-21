# iteration-001

- 变更说明：显式保留当前攻击的进化、Boss、Energy 与撤退接力路线
- 主要假设：显式保留当前攻击的进化、Boss、Energy 与撤退接力路线
- 指标 profile：`auto_iteration_v8_setup_relay`
- profile revision：`2`
- 样本类型：`full`
- Control：`baseline`
- Decision：`promote`
- 原生报告：[Evaluation HTML](run-f911eabbcec8447c94c77ce191bd84db/report.html)；[Evaluation Markdown](run-f911eabbcec8447c94c77ce191bd84db/report.md)
- 样本：170 局，17 个对手；先手 85 局，后手 85 局

## 结果

- Correctness errors：2 / 170
- 总体胜率：72.9% (124/170)
- 先手胜率：71.8% (61/85)
- 后手胜率：74.1% (63/85)
- 二回合实际选择 `attackId=1072`：22.4% (38/170)
- 二回合实际攻击（先手 / 后手）：23.5% (20/85) / 21.2% (18/85)
- 四组件状态（仅观测）：Active Abra 99/170；Rare Candy 64/170；Alakazam 或检索路线 156/170；Psychic Energy 或 Hilda 130/170；四项同时满足 25/170
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 5.25 张 (893/170)；实际到达二回合 5.45 张 (893/164)
- 第二回合平均过牌张数（先手 / 后手）：4.75 张 (404/85) / 5.75 张 (489/85)
- 攻击但未拿奖赏（全部攻击）：18.7% (106/567)
- 攻击但未拿奖赏（Powerful Hand）：13.3% (70/527)
- 攻击结算审计：已结算 569 次，未完成 0 次，奖赏状态未知 2 次
- Post-KO 立即接力：171 / 449（38.1%）
- Post-KO 原始审计：native 顶层保留 legacy zero-ready 失败率 278 / 449；语义指标使用上一行 payload 成功率，不使用 legacy 失败率做 promotion。
- `recoverable_discard_miss`：74
- 低牌库区间消耗：1.8058823529411765

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
