# 评测报告

运行 ID：run-a2611d7873a346a4a40828cbee605656

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
| 胜 / 负 / 平 | 121 / 47 / 0 |
| 完成率 | 98.82% |
| 胜率 | 71.18% |
| 错误数 | 2 |
| 未完成 | 0 |
| 已完成 | 168 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_abomasnow | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_dragapult | 3 | 7 | 0 | 0 | 0 | 30.00% |
| kiyotah_iono | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kokinn_search | 8 | 2 | 0 | 0 | 0 | 80.00% |
| maktha_1084 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| nursrijan_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| penguin_915 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| pilkwang_v2 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| romanrozen_v9 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| sue_alakazam | 8 | 2 | 0 | 0 | 0 | 80.00% |
| yakitori_raging_bolt | 8 | 0 | 0 | 2 | 0 | 80.00% |
| yanxiaohan | 5 | 5 | 0 | 0 | 0 | 50.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 109 | 589 | 0.185059 |
| correctness | 2 | 170 | 0.0117647 |
| length | 20089 | 170 | 118.171 |
| library_pressure | 326 | 170 | 1.91765 |
| outcome | 121 | 170 | 0.711765 |
| post_ko_relay | 313 | 483 | 0.648033 |
| powerful_hand | 29 | 170 | 0.170588 |
| rare_candy | 29 | 170 | 0.170588 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 29 | 170 | 0.170588 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 2 |

## 重点案例

### 案例 1：yakitori_raging_bolt-001

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-001.json
- 证据步骤 22：

### 案例 2：yakitori_raging_bolt-008

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：traces/yakitori_raging_bolt-008.json
- 证据步骤 60：

### 案例 3：kacchan_anti_wall-004

- 对手：kacchan_anti_wall
- failure_class：rare_candy_not_played
- trace：traces/kacchan_anti_wall-004.json
- 证据步骤 36：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 121 | 170 | 0.711764705882353 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 20089 | 170 | 118.17058823529412 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 2 | 170 | 0.011764705882352941 |

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
| 313 | 483 | 0.6480331262939959 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 326 | 170 | 1.9176470588235295 |

## Setup and relay

- bridge opportunities: 50
- bridge completions: 0
- ability draws / game: 5.194117647058824
- normal draws / game: 2.0588235294117645

## Attack quality

- attack submissions: 589
- resolved: 589
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 109/589
