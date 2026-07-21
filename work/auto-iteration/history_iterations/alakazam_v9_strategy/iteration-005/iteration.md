# iteration-005

- 变更说明：保留四项反例修复，仅阻止会让 Powerful Hand 掉出 KO 阈值的宝可梦铺场；安全 Fez 抽牌仍可先做
- 主要假设：保留四项反例修复，仅阻止会让 Powerful Hand 掉出 KO 阈值的宝可梦铺场；安全 Fez 抽牌仍可先做
- 指标 profile：`auto_iteration_v8_setup_relay`
- profile revision：`2`
- 样本类型：`full`
- Control：`baseline`
- Decision：`promote`
- 原生报告：[Evaluation HTML](run-9d3c9b201764417c80c11e4f05cd4141/report.html)；[Evaluation Markdown](run-9d3c9b201764417c80c11e4f05cd4141/report.md)
- 样本：170 局，17 个对手；先手 85 局，后手 85 局

## 结果

- Correctness errors：2 / 170
- 总体胜率：68.8% (117/170)
- 先手胜率：69.4% (59/85)
- 后手胜率：68.2% (58/85)
- 二回合实际选择 `attackId=1072`：20.6% (35/170)
- 二回合实际攻击（先手 / 后手）：16.5% (14/85) / 24.7% (21/85)
- 四组件状态（仅观测）：Active Abra 104/170；Rare Candy 52/170；Alakazam 或检索路线 157/170；Psychic Energy 或 Hilda 136/170；四项同时满足 18/170
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 4.94 张 (839/170)；实际到达二回合 5.15 张 (839/163)
- 第二回合平均过牌张数（先手 / 后手）：4.49 张 (382/85) / 5.38 张 (457/85)
- 攻击但未拿奖赏（全部攻击）：19.2% (115/599)
- 攻击但未拿奖赏（Powerful Hand）：11.6% (63/541)
- 攻击结算审计：已结算 601 次，未完成 0 次，奖赏状态未知 2 次
- Post-KO 立即接力：170 / 468（36.3%）
- Post-KO 原始审计：native 顶层保留 legacy zero-ready 失败率 298 / 468；语义指标使用上一行 payload 成功率，不使用 legacy 失败率做 promotion。
- `recoverable_discard_miss`：87
- 低牌库区间消耗：2.052941176470588

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
