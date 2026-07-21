# 评测报告

运行 ID：run-81560c84b71d4142af838cd51da7af78

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
| 胜 / 负 / 平 | 70 / 99 / 0 |
| 完成率 | 99.41% |
| 胜率 | 41.18% |
| 错误数 | 1 |
| 未完成 | 0 |
| 已完成 | 169 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kacchan_anti_wall | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_abomasnow | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_dragapult | 2 | 8 | 0 | 0 | 0 | 20.00% |
| kiyotah_iono | 0 | 10 | 0 | 0 | 0 | 0.00% |
| kiyotah_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kokinn_search | 3 | 7 | 0 | 0 | 0 | 30.00% |
| maktha_1084 | 3 | 7 | 0 | 0 | 0 | 30.00% |
| nursrijan_lucario | 4 | 6 | 0 | 0 | 0 | 40.00% |
| penguin_915 | 3 | 7 | 0 | 0 | 0 | 30.00% |
| pilkwang_v2 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| romanrozen_v9 | 3 | 7 | 0 | 0 | 0 | 30.00% |
| sue_alakazam | 2 | 8 | 0 | 0 | 0 | 20.00% |
| yakitori_raging_bolt | 8 | 1 | 0 | 1 | 0 | 80.00% |
| yanxiaohan | 1 | 9 | 0 | 0 | 0 | 10.00% |
| zoli_dragapult | 5 | 5 | 0 | 0 | 0 | 50.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 139 | 458 | 0.303493 |
| correctness | 1 | 170 | 0.00588235 |
| length | 17991 | 170 | 105.829 |
| library_pressure | 564 | 170 | 3.31765 |
| outcome | 70 | 170 | 0.411765 |
| post_ko_relay | 339 | 455 | 0.745055 |
| powerful_hand | 4 | 170 | 0.0235294 |
| rare_candy | 4 | 170 | 0.0235294 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 4 | 170 | 0.0235294 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 1 |

## 重点案例

### 案例 1：yakitori_raging_bolt-009

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/evaluation-sample/run-81560c84b71d4142af838cd51da7af78/traces/yakitori_raging_bolt-009.json
- 证据步骤 97：

### 案例 2：crustle_wall-002

- 对手：crustle_wall
- failure_class：rare_candy_not_played
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/evaluation-sample/run-81560c84b71d4142af838cd51da7af78/traces/crustle_wall-002.json
- 证据步骤 19：

### 案例 3：kacchan_anti_wall-010

- 对手：kacchan_anti_wall
- failure_class：alakazam_not_active
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/evaluation-sample/run-81560c84b71d4142af838cd51da7af78/traces/kacchan_anti_wall-010.json
- 证据步骤 48：

## Attack quality

- attack submissions: 458
- resolved: 458
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 139/458

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 170 | 0.0058823529411764705 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 17991 | 170 | 105.82941176470588 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 564 | 170 | 3.3176470588235296 |

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 70 | 170 | 0.4117647058823529 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 339 | 455 | 0.7450549450549451 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 4 | 170 | 0.023529411764705882 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 4 | 170 | 0.023529411764705882 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

## Setup and relay

- bridge opportunities: 59
- bridge completions: 0
- ability draws / game: 2.7705882352941176
- normal draws / game: 1.9823529411764707
