# 评测报告

运行 ID：run-3d201c3bb6f244c494b9c25d5e2cdfd1

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
| 胜 / 负 / 平 | 97 / 72 / 0 |
| 完成率 | 99.41% |
| 胜率 | 57.06% |
| 错误数 | 1 |
| 未完成 | 0 |
| 已完成 | 169 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_abomasnow | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_dragapult | 1 | 9 | 0 | 0 | 0 | 10.00% |
| kiyotah_iono | 3 | 7 | 0 | 0 | 0 | 30.00% |
| kiyotah_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kokinn_search | 6 | 4 | 0 | 0 | 0 | 60.00% |
| maktha_1084 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| nursrijan_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| penguin_915 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| pilkwang_v2 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| romanrozen_v9 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| sue_alakazam | 6 | 4 | 0 | 0 | 0 | 60.00% |
| yakitori_raging_bolt | 9 | 0 | 0 | 1 | 0 | 90.00% |
| yanxiaohan | 6 | 4 | 0 | 0 | 0 | 60.00% |
| zoli_dragapult | 6 | 4 | 0 | 0 | 0 | 60.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 126 | 556 | 0.226619 |
| correctness | 1 | 170 | 0.00588235 |
| length | 19686 | 170 | 115.8 |
| library_pressure | 244 | 170 | 1.43529 |
| outcome | 97 | 170 | 0.570588 |
| post_ko_relay | 441 | 563 | 0.783304 |
| powerful_hand | 52 | 170 | 0.305882 |
| rare_candy | 52 | 170 | 0.305882 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 52 | 170 | 0.305882 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 1 |

## 重点案例

### 案例 1：yakitori_raging_bolt-010

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-010.json
- 证据步骤 47：

### 案例 2：crustle_wall-004

- 对手：crustle_wall
- failure_class：rare_candy_not_played
- trace：traces/crustle_wall-004.json
- 证据步骤 23：

### 案例 3：kiyotah_lucario-005

- 对手：kiyotah_lucario
- failure_class：no_legal_attack
- trace：traces/kiyotah_lucario-005.json
- 证据步骤 38：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 97 | 170 | 0.5705882352941176 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 19686 | 170 | 115.8 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 170 | 0.0058823529411764705 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 52 | 170 | 0.3058823529411765 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 52 | 170 | 0.3058823529411765 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 441 | 563 | 0.783303730017762 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 244 | 170 | 1.4352941176470588 |

## Setup and relay

- bridge opportunities: 40
- bridge completions: 0
- ability draws / game: 4.317647058823529
- normal draws / game: 2.0764705882352943

## Attack quality

- attack submissions: 556
- resolved: 556
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 126/556
