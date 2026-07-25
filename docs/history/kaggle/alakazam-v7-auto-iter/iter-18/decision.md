# Iter-18 决策：observe

## 1. 本轮改动

- 修正 swap 对局中的己方回合计数：`_own_turn_number()` 改为依据
  `firstPlayer` 判断奇偶，而不是固定假设物理 index 0 先手。
- 增加 Active Abra → Kadabra 的确定 KO 路线：如果本回合自然进化合法、手牌有
  Kadabra、Active 有 Psychic，且 Kadabra 进化后能立即击倒对方 Active，则在
  Bench Abra 进化之前完成 Active 进化。
- 未修改 `deck.csv`、卡牌数量、攻击终止语义或 Supporter/手填能量次数限制。

## 2. 具体 case 验收

`kiyotah_dragapult/game_008.json` 的 shared turn 3、`trace[27]`：

- control 实际选择 `[6]`：`END`；
- 局面有 Active Abra + Psychic、手牌 Kadabra、对方 Budew 30 HP，且先前的
  `Itchy Pollen` 只锁 Item；
- candidate 在按 raw option index 复原后选择 `[1]`：Active Abra → Kadabra；
- 后续可使用 Kadabra 的 Psychic Draw，并用 `Super Psy Bolt` 形成确定 KO。

该 fixture 已加入 `tests/test_alakazam_v7_auto_iter_strategy.py`，并覆盖 swap 后第二
个己方回合的进化合法性。

## 3. 评测前后

control 是已接受的 `iter-15-patch-priority` full-trace 批次；candidate 是独立随机
的 17 个对手 × 10 局 full-trace 批次。两批不是逐局同随机样本，因此只作 guardrail，
不把胜率差异解释为单变量因果。

| 指标 | iter-15 control | iter-18 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 109 / 60 / 1 | 胜率下降 |
| 胜率 | 68.2% | 64.1% | -4.1pp |
| Meta 加权胜率 | 70.4% | 64.2% | -6.2pp |
| 第二回合 Powerful Hand | 27.1% | 28.8% | +1.8pp |
| post-KO 无 ready attacker | 68.5% | 64.0% | -4.5pp |
| 打手断档对局率 | 36.5% | 34.1% | -2.4pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

candidate 原始 summary 中有 1 个 `yakitori_raging_bolt/game_002` 的 `IndexError`，
trace role 标为 `opponent`；分析器未将其计入我方 action error。

## 4. 保留决策

`observe`，不覆盖当前 best。目标 fixture 已解决，第二回合和接力指标方向改善，但
完整独立批次的胜率和 Meta 加权胜率明显低于 iter-15。当前 08:05 定时提交仍使用
`BEST_STRATEGY.json` 指向的不可变 `iter-15-patch-priority` 归档。

下一轮应继续分析剩余的 Bench Kadabra → Alakazam 接力 case，而不是把 iter-18 直接
作为 Kaggle best 提交。
