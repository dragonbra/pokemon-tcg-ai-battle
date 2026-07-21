# 评测报告

运行 ID：run-837ae45f0b944adc8202f8157cdb5f13

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
| 胜 / 负 / 平 | 115 / 52 / 0 |
| 完成率 | 98.24% |
| 胜率 | 67.65% |
| 错误数 | 3 |
| 未完成 | 0 |
| 已完成 | 167 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_abomasnow | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_dragapult | 3 | 7 | 0 | 0 | 0 | 30.00% |
| kiyotah_iono | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kokinn_search | 6 | 4 | 0 | 0 | 0 | 60.00% |
| maktha_1084 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| nursrijan_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| penguin_915 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| pilkwang_v2 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 7 | 3 | 0 | 0 | 0 | 70.00% |
| yakitori_raging_bolt | 7 | 0 | 0 | 3 | 0 | 70.00% |
| yanxiaohan | 9 | 1 | 0 | 0 | 0 | 90.00% |
| zoli_dragapult | 7 | 3 | 0 | 0 | 0 | 70.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 122 | 605 | 0.201653 |
| correctness | 3 | 170 | 0.0176471 |
| length | 21317 | 170 | 125.394 |
| library_pressure | 548 | 170 | 3.22353 |
| outcome | 115 | 170 | 0.676471 |
| post_ko_relay | 344 | 529 | 0.650284 |
| powerful_hand | 29 | 170 | 0.170588 |
| rare_candy | 29 | 170 | 0.170588 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 29 | 170 | 0.170588 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 3 |

## 重点案例

### 案例 1：yakitori_raging_bolt-007

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-007.json
- 证据步骤 76：

### 案例 2：yakitori_raging_bolt-008

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-008.json
- 证据步骤 112：

### 案例 3：yakitori_raging_bolt-010

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-010.json
- 证据步骤 56：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 115 | 170 | 0.6764705882352942 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 21317 | 170 | 125.39411764705882 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 3 | 170 | 0.01764705882352941 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 29 | 170 | 0.17058823529411765 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 29 | 170 | 0.17058823529411765 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 344 | 529 | 0.6502835538752363 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 548 | 170 | 3.223529411764706 |

## Setup and relay

- bridge opportunities: 43
- bridge completions: 0
- ability draws / game: 5.252941176470588
- normal draws / game: 2.0352941176470587

## Attack quality

- attack submissions: 605
- resolved: 605
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 122/605
