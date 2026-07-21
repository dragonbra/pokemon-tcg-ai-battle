# 评测报告

运行 ID：run-f911eabbcec8447c94c77ce191bd84db

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
| 胜 / 负 / 平 | 124 / 44 / 0 |
| 完成率 | 98.82% |
| 胜率 | 72.94% |
| 错误数 | 2 |
| 未完成 | 0 |
| 已完成 | 168 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_abomasnow | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_dragapult | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_iono | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kokinn_search | 8 | 2 | 0 | 0 | 0 | 80.00% |
| maktha_1084 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| nursrijan_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| penguin_915 | 3 | 7 | 0 | 0 | 0 | 30.00% |
| pilkwang_v2 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| romanrozen_v9 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| sue_alakazam | 9 | 1 | 0 | 0 | 0 | 90.00% |
| yakitori_raging_bolt | 8 | 0 | 0 | 2 | 0 | 80.00% |
| yanxiaohan | 5 | 5 | 0 | 0 | 0 | 50.00% |
| zoli_dragapult | 9 | 1 | 0 | 0 | 0 | 90.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 106 | 567 | 0.186949 |
| correctness | 2 | 170 | 0.0117647 |
| length | 19419 | 170 | 114.229 |
| library_pressure | 307 | 170 | 1.80588 |
| outcome | 124 | 170 | 0.729412 |
| post_ko_relay | 278 | 449 | 0.619154 |
| powerful_hand | 38 | 170 | 0.223529 |
| rare_candy | 38 | 170 | 0.223529 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 38 | 170 | 0.223529 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 2 |

## 重点案例

### 案例 1：yakitori_raging_bolt-008

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-008.json
- 证据步骤 10：

### 案例 2：yakitori_raging_bolt-010

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-010.json
- 证据步骤 31：

### 案例 3：crustle_wall-004

- 对手：crustle_wall
- failure_class：alakazam_not_active
- trace：traces/crustle_wall-004.json
- 证据步骤 21：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 124 | 170 | 0.7294117647058823 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 19419 | 170 | 114.22941176470589 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 2 | 170 | 0.011764705882352941 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 38 | 170 | 0.2235294117647059 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 38 | 170 | 0.2235294117647059 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 278 | 449 | 0.6191536748329621 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 307 | 170 | 1.8058823529411765 |

## Setup and relay

- bridge opportunities: 47
- bridge completions: 0
- ability draws / game: 5.252941176470588
- normal draws / game: 2.0294117647058822

## Attack quality

- attack submissions: 569
- resolved: 569
- unresolved: 0
- unknown prize: 2
- non-prize attacks: 106/567
