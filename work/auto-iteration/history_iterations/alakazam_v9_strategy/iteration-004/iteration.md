# iteration-004

- 变更说明：修复精确 KO、Poffin 批内缺口、已撤退门控与 Night Stretcher 当前攻击能量；初版将所有 KO 绝对提前，结果护栏回退
- 主要假设：修复精确 KO、Poffin 批内缺口、已撤退门控与 Night Stretcher 当前攻击能量；初版将所有 KO 绝对提前，结果护栏回退
- 指标 profile：`auto_iteration_v8_setup_relay`
- profile revision：`2`
- 样本类型：`full`
- Control：`baseline`
- Decision：`reject`
- 原生报告：[Evaluation HTML](run-3d201c3bb6f244c494b9c25d5e2cdfd1/report.html)；[Evaluation Markdown](run-3d201c3bb6f244c494b9c25d5e2cdfd1/report.md)
- 样本：170 局，17 个对手；先手 85 局，后手 85 局

## 结果

- Correctness errors：1 / 170
- 总体胜率：57.1% (97/170)
- 先手胜率：64.7% (55/85)
- 后手胜率：49.4% (42/85)
- 二回合实际选择 `attackId=1072`：30.6% (52/170)
- 二回合实际攻击（先手 / 后手）：28.2% (24/85) / 32.9% (28/85)
- 四组件状态（仅观测）：Active Abra 94/170；Rare Candy 68/170；Alakazam 或检索路线 159/170；Psychic Energy 或 Hilda 135/170；四项同时满足 26/170
- 第二回合平均过牌张数（能力/效果抽牌）：全部对局 4.32 张 (734/170)；实际到达二回合 4.40 张 (734/167)
- 第二回合平均过牌张数（先手 / 后手）：4.05 张 (344/85) / 4.59 张 (390/85)
- 攻击但未拿奖赏（全部攻击）：22.7% (126/556)
- 攻击但未拿奖赏（Powerful Hand）：13.8% (68/491)
- 攻击结算审计：已结算 556 次，未完成 0 次，奖赏状态未知 0 次
- Post-KO 立即接力：122 / 563（21.7%）
- Post-KO 原始审计：native 顶层保留 legacy zero-ready 失败率 441 / 563；语义指标使用上一行 payload 成功率，不使用 legacy 失败率做 promotion。
- `recoverable_discard_miss`：146
- 低牌库区间消耗：1.4352941176470588

## 口径限制

本报告只使用 trace 可见事实。四组件只记录第一回合起始资源状态，不参与主要评估门槛；Alakazam 组件包含 Alakazam、Poke Pad、Hilda、Dawn，Psychic Energy 组件包含 Psychic Energy 或 Hilda。过牌主指标排除回合开始的正常抽牌。攻击但未拿奖赏是结果惩罚项，不能单独证明是手牌不足、伤害不足、Boss 漏用或特殊能量造成；后续 evaluator 应补充选项级资源可达性。`recoverable_discard_miss` 是弃牌区出现攻击线资源的保守候选，不能单独证明当时一定有合法回收选项。
