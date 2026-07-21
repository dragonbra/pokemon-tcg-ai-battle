# 评测报告

运行 ID：run-250289e1b92e48738482e79077d485ab

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
| 胜 / 负 / 平 | 121 / 48 / 0 |
| 完成率 | 99.41% |
| 胜率 | 71.18% |
| 错误数 | 1 |
| 未完成 | 0 |
| 已完成 | 169 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_abomasnow | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_dragapult | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_iono | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kokinn_search | 6 | 4 | 0 | 0 | 0 | 60.00% |
| maktha_1084 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| nursrijan_lucario | 9 | 1 | 0 | 0 | 0 | 90.00% |
| penguin_915 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| pilkwang_v2 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 9 | 1 | 0 | 0 | 0 | 90.00% |
| yakitori_raging_bolt | 9 | 0 | 0 | 1 | 0 | 90.00% |
| yanxiaohan | 6 | 4 | 0 | 0 | 0 | 60.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 121 | 598 | 0.202341 |
| correctness | 1 | 170 | 0.00588235 |
| length | 20346 | 170 | 119.682 |
| library_pressure | 346 | 170 | 2.03529 |
| outcome | 121 | 170 | 0.711765 |
| post_ko_relay | 326 | 486 | 0.670782 |
| powerful_hand | 33 | 170 | 0.194118 |
| rare_candy | 33 | 170 | 0.194118 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 33 | 170 | 0.194118 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 1 |

## 重点案例

### 案例 1：yakitori_raging_bolt-001

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-001.json
- 证据步骤 26：

### 案例 2：crustle_v1-005

- 对手：crustle_v1
- failure_class：rare_candy_not_played
- trace：traces/crustle_v1-005.json
- 证据步骤 20：

### 案例 3：romanrozen_v9-006

- 对手：romanrozen_v9
- failure_class：no_legal_attack
- trace：traces/romanrozen_v9-006.json
- 证据步骤 40：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 121 | 170 | 0.711764705882353 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 20346 | 170 | 119.68235294117648 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 170 | 0.0058823529411764705 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 33 | 170 | 0.19411764705882353 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 33 | 170 | 0.19411764705882353 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 326 | 486 | 0.6707818930041153 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 346 | 170 | 2.0352941176470587 |

## Setup and relay

- bridge opportunities: 57
- bridge completions: 0
- ability draws / game: 4.652941176470589
- normal draws / game: 2.123529411764706

## Attack quality

- attack submissions: 598
- resolved: 598
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 121/598
