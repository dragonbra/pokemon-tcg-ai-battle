# 评测报告

运行 ID：run-01c6110c386c4d269399e305ba639db3

## Metric profile

| 项目 | 数值 |
| --- | --- |
| profile | auto_iteration_v8_setup_relay |
| revision | 2 |
| metrics | outcome, length, correctness, powerful_hand, rare_candy, post_ko_relay, run_away_draw, library_pressure, setup_relay, attack_quality |

## 总体结果

| 项目 | 数值 |
| --- | ---: |
| 总对局 | 170 |
| 胜 / 负 / 平 | 120 / 49 / 0 |
| 完成率 | 99.41% |
| 胜率 | 70.59% |
| 错误数 | 1 |
| 未完成 | 0 |
| 已完成 | 169 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_abomasnow | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_dragapult | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_iono | 2 | 8 | 0 | 0 | 0 | 20.00% |
| kiyotah_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kokinn_search | 5 | 5 | 0 | 0 | 0 | 50.00% |
| maktha_1084 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| nursrijan_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| penguin_915 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| pilkwang_v2 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 8 | 2 | 0 | 0 | 0 | 80.00% |
| yakitori_raging_bolt | 9 | 0 | 0 | 1 | 0 | 90.00% |
| yanxiaohan | 5 | 5 | 0 | 0 | 0 | 50.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 132 | 613 | 0.215334 |
| correctness | 1 | 170 | 0.00588235 |
| length | 19997 | 170 | 117.629 |
| library_pressure | 293 | 170 | 1.72353 |
| outcome | 120 | 170 | 0.705882 |
| post_ko_relay | 320 | 486 | 0.658436 |
| powerful_hand | 50 | 170 | 0.294118 |
| rare_candy | 50 | 170 | 0.294118 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 50 | 170 | 0.294118 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 1 |

## 重点案例

### 案例 1：yakitori_raging_bolt-004

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-004.json
- 证据步骤 84：

### 案例 2：kacchan_anti_wall-002

- 对手：kacchan_anti_wall
- failure_class：rare_candy_not_played
- trace：traces/kacchan_anti_wall-002.json
- 证据步骤 51：

### 案例 3：kiyotah_abomasnow-009

- 对手：kiyotah_abomasnow
- failure_class：no_legal_attack
- trace：traces/kiyotah_abomasnow-009.json
- 证据步骤 21：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 120 | 170 | 0.7058823529411765 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 19997 | 170 | 117.62941176470588 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 170 | 0.0058823529411764705 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 50 | 170 | 0.29411764705882354 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 50 | 170 | 0.29411764705882354 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 320 | 486 | 0.6584362139917695 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 293 | 170 | 1.723529411764706 |

## Setup and relay

- bridge opportunities: 39
- bridge completions: 0
- ability draws / game: 5.535294117647059
- normal draws / game: 2.0470588235294116

## Attack quality

- attack submissions: 613
- resolved: 613
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 132/613
