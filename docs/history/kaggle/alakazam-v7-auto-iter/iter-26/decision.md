# AutoIter 26：无可见进化路线时不消耗 Basic Psychic

## 1. 本轮改动

- 直接把 Basic Psychic 贴给 Bench Abra 前，要求当前手牌能看见 Kadabra，或能看见
  Alakazam + Rare Candy 且没有 Item Lock。
- 同步收紧 Bench insurance 对 Basic Psychic→Abra 的识别，避免把不可验证的未来路线
  当作当前接力准备。
- Telepath Energy、Lana's Aid、Wondrous Patch、终局攻击与 `deck.csv` 均未改动。

## 2. 具体 case 验收

此前三个“Telepath 贴回 Active”的真实 replay case，在当前策略中都会选择 Bench
Abra 的合法 attachment 选项。新增 fixture 中，Active Alakazam 已有 Psychic、Bench
Abra 没有可见 Kadabra/Rare Candy 路线时，Basic Psychic attachment 被跳过，保留当前
攻击选项。

完整 trace 中观察到 67 次“Basic Psychic→无可见进化路线的 Bench Abra”选项，候选实际
选择 0 次；Telepath→Bench Abra 仍有 406 次机会、选择 95 次，说明 Telepath 的检索
型 Bench anchor 没有被误伤。

## 3. 评测前后

候选与当前 best 是独立随机批次，不能当作严格 A/B。当前 best 为 `iter-15-patch-priority`。

| 指标 | iter-15 immutable best | iter-26 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 116 / 52 / 2 | 111 / 59 / 0 | 方向下降 |
| 胜率 | 68.2% | 65.3% | -2.9pp |
| Meta 加权胜率 | 70.4% | 64.8% | -5.6pp |
| 第二回合 Powerful Hand | 27.1% | 19.4% | -7.7pp |
| post-KO 无 ready attacker | 68.5% | 69.8% | +1.3pp |
| 对局级打手断档 | 36.5% | 38.2% | +1.7pp |
| 空 Bench Run Away Draw | 0 | 0 | 持平 |
| 我方 action error | 0 | 0 | 持平 |

## 4. 决策

**observe，不晋升。** 本轮修复了一个真实且可复现的资源浪费边界，focused case 通过，
但独立 170 局没有显示整体收益，且三个核心 guardrail 低于 best。保留工作区候选供后续
组合分析，但 `BEST_STRATEGY.json` 继续指向 `iter-15-patch-priority`；08:05 定时任务
仍提交该 immutable archive。

下一轮只测试一个独立变量：第二回合 Powerful Hand 的快速优先不能越过可确认的 Boss
高奖赏 KO 路线。
