# 评测报告

运行 ID：run-9d3c9b201764417c80c11e4f05cd4141

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
| 胜 / 负 / 平 | 117 / 51 / 0 |
| 完成率 | 98.82% |
| 胜率 | 68.82% |
| 错误数 | 2 |
| 未完成 | 0 |
| 已完成 | 168 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_abomasnow | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_dragapult | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_iono | 1 | 9 | 0 | 0 | 0 | 10.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 6 | 4 | 0 | 0 | 0 | 60.00% |
| maktha_1084 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| nursrijan_lucario | 9 | 1 | 0 | 0 | 0 | 90.00% |
| penguin_915 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| pilkwang_v2 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yakitori_raging_bolt | 8 | 0 | 0 | 2 | 0 | 80.00% |
| yanxiaohan | 6 | 4 | 0 | 0 | 0 | 60.00% |
| zoli_dragapult | 7 | 3 | 0 | 0 | 0 | 70.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 115 | 599 | 0.191987 |
| correctness | 2 | 170 | 0.0117647 |
| length | 20161 | 170 | 118.594 |
| library_pressure | 349 | 170 | 2.05294 |
| outcome | 117 | 170 | 0.688235 |
| post_ko_relay | 298 | 468 | 0.636752 |
| powerful_hand | 35 | 170 | 0.205882 |
| rare_candy | 35 | 170 | 0.205882 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 35 | 170 | 0.205882 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 2 |

## 重点案例

### 案例 1：yakitori_raging_bolt-008

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-008.json
- 证据步骤 45：

### 案例 2：yakitori_raging_bolt-010

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-010.json
- 证据步骤 50：

### 案例 3：kacchan_anti_wall-003

- 对手：kacchan_anti_wall
- failure_class：rare_candy_not_played
- trace：traces/kacchan_anti_wall-003.json
- 证据步骤 27：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 117 | 170 | 0.6882352941176471 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 20161 | 170 | 118.59411764705882 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 2 | 170 | 0.011764705882352941 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 35 | 170 | 0.20588235294117646 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 35 | 170 | 0.20588235294117646 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 298 | 468 | 0.6367521367521367 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 349 | 170 | 2.052941176470588 |

## Setup and relay

- bridge opportunities: 39
- bridge completions: 0
- ability draws / game: 4.935294117647059
- normal draws / game: 2.0647058823529414

## Attack quality

- attack submissions: 601
- resolved: 601
- unresolved: 0
- unknown prize: 2
- non-prize attacks: 115/599
