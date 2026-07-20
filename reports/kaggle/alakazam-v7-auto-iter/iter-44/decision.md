# AutoIter iter-44 决策：终局 Fezandipiti 让位于 Powerful Hand

## 1. 本轮尝试

- Control：iter-43 的 field option 解析修复版本。
- Candidate：当 Active Alakazam 的 `Powerful Hand` 已经能够击倒当前目标并拿完最后奖赏时，
  直接提交攻击；不再让可选的 Fezandipiti ex `Flip the Script` 抽牌动作先执行。
- 固定不变：`deck.csv`、Abra 不攻击、Trading Places 禁用、攻击宣告立即结束回合、
  Supporter/手动附能次数、Rare Candy 的 Budew `Itchy Pollen` 限制，以及 Mist/Rock
  Fighting Energy 的伤害保护规则。

## 2. 具体验收 case

来源：
`/Users/hejinyu/Documents/repos/ptcg-agent-kaggle/reports/alakazam_v7_auto_iter/iter-43-control-20260720/kacchan_anti_wall/game_010.json`，raw[151]。

当时 Active Alakazam 已有 Psychic Energy，手牌 27 张，剩余 Prize 为 1；对手 Active
为 340 HP 的 Mega Lucario ex。`Powerful Hand` 可造成 540 伤害并直接完成最后奖赏，
但旧排序先选择 Fezandipiti ex。正确动作是攻击，后续铺场已经没有收益。

## 3. 前后指标

两轮均为 17 个对手 × 10 局的完整 trace，但使用独立随机样本，不能视为严格逐局 A/B：

| 指标 | iter-43 control | iter-44 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 106 / 62 / 2 | 127 / 43 / 0 | +21 胜 |
| 胜率 | 62.35% | 74.71% | +12.35pp |
| Meta 加权胜率 | 61.29% | 74.65% | +13.36pp |
| 第二回合 Powerful Hand | 25.29% | 23.53% | -1.76pp |
| post-KO 无 ready attacker | 110/180 = 61.11% | 109/169 = 64.50% | +3.39pp |
| 对局级接力断档 | 34.12% | 34.71% | +0.59pp |
| 空 Bench Run Away Draw | 0 | 0 | 不变 |
| 我方 action error | 0 | 0 | 不变 |

胜率信号明显向好，但第二回合和 post-KO 事件率没有同步改善，且样本独立；因此不能
把 +12.35pp 宣称为本轮单变量的因果收益。

## 4. 采纳状态

**observe：作为下一轮探索 control，暂不更新 immutable `BEST_STRATEGY.json`。**

terminal attack 边界保留。下一轮处理已确认的 Telepath case：当手里有 Poké Pad、资源
账本仍显示牌库/未知区存在 Kadabra 时，给 Bench Abra 贴 Telepath Energy 比给已能攻击
的 Active Alakazam 多一层接力价值；仍需保留终局攻击优先和 Item Lock 边界。

完整 trace 只保留在外部评测目录；本 repo 只保留本报告和轻量摘要。
