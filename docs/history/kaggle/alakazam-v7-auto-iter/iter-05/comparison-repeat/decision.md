# AutoIter V5 独立 repeat 决策

## 本轮改动

前一版本是 `iter-04` repeat candidate；本轮只增加 Active Alakazam 空 Bench 时的
Poffin 保险，并保持 `deck.csv` 不变。终局攻击闭环仍优先于 Poffin。

## evaluate 前后

17 个对手 × 10 局，完整 trace，独立随机批次：

| 指标 | iter-04 repeat control | V5 repeat candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 106 / 63 / 1 | 97 / 72 / 1 | 胜 -9 |
| 胜率 | 62.4% | 57.1% | -5.3pp |
| Meta 加权胜率 | 61.8% | 57.8% | -3.9pp |
| 第二回合 Powerful Hand | 36/170 (21.2%) | 49/170 (28.8%) | +7.6pp |
| post-KO 无 ready attacker | 117/174 (67.2%) | 137/183 (74.9%) | +7.7pp |
| 打手断档对局率 | 40.0% | 37.1% | -2.9pp |
| Bench 保险 miss | 12/13 | 0/6 | 解决 |
| 我方 action error | 0 | 0 | 持平 |

## 决策

`observe`。具体 Bench case 全部解决，但胜率、Meta 和 post-KO ready attacker 退化；
该 repeat 不能支持晋级。上述触发局面数量来自独立发牌，不能解读为绝对频率变化。
