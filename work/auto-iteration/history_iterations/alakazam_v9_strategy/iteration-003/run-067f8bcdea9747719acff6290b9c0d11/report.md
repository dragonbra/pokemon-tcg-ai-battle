# 评测报告

运行 ID：run-067f8bcdea9747719acff6290b9c0d11

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
| 胜 / 负 / 平 | 117 / 53 / 0 |
| 完成率 | 100.00% |
| 胜率 | 68.82% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 170 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_abomasnow | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kiyotah_dragapult | 3 | 7 | 0 | 0 | 0 | 30.00% |
| kiyotah_iono | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 6 | 4 | 0 | 0 | 0 | 60.00% |
| maktha_1084 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| nursrijan_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| penguin_915 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| pilkwang_v2 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| romanrozen_v9 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| sue_alakazam | 9 | 1 | 0 | 0 | 0 | 90.00% |
| yakitori_raging_bolt | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yanxiaohan | 4 | 6 | 0 | 0 | 0 | 40.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 134 | 590 | 0.227119 |
| correctness | 0 | 170 | 0 |
| length | 19603 | 170 | 115.312 |
| library_pressure | 359 | 170 | 2.11176 |
| outcome | 117 | 170 | 0.688235 |
| post_ko_relay | 332 | 471 | 0.704883 |
| powerful_hand | 50 | 170 | 0.294118 |
| rare_candy | 50 | 170 | 0.294118 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 50 | 170 | 0.294118 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：kacchan_anti_wall-001

- 对手：kacchan_anti_wall
- failure_class：rare_candy_not_played
- trace：traces/kacchan_anti_wall-001.json
- 证据步骤 35：

### 案例 2：kiyotah_iono-001

- 对手：kiyotah_iono
- failure_class：no_legal_attack
- trace：traces/kiyotah_iono-001.json
- 证据步骤 25：

### 案例 3：kiyotah_abomasnow-003

- 对手：kiyotah_abomasnow
- failure_class：rare_candy_not_played
- trace：traces/kiyotah_abomasnow-003.json
- 证据步骤 12：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 117 | 170 | 0.6882352941176471 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 19603 | 170 | 115.31176470588235 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

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
| 332 | 471 | 0.7048832271762208 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 359 | 170 | 2.111764705882353 |

## Setup and relay

- bridge opportunities: 51
- bridge completions: 0
- ability draws / game: 5.035294117647059
- normal draws / game: 2.0176470588235293

## Attack quality

- attack submissions: 590
- resolved: 590
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 134/590
