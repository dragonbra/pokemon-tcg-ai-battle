# 评测报告

运行 ID：run-835907c1bab4457b915fc5b8be8d9773

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
| 胜 / 负 / 平 | 103 / 66 / 0 |
| 完成率 | 99.41% |
| 胜率 | 60.59% |
| 错误数 | 1 |
| 未完成 | 0 |
| 已完成 | 169 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_abomasnow | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_dragapult | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_iono | 3 | 7 | 0 | 0 | 0 | 30.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 5 | 5 | 0 | 0 | 0 | 50.00% |
| maktha_1084 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| nursrijan_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| penguin_915 | 3 | 7 | 0 | 0 | 0 | 30.00% |
| pilkwang_v2 | 3 | 7 | 0 | 0 | 0 | 30.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 7 | 3 | 0 | 0 | 0 | 70.00% |
| yakitori_raging_bolt | 9 | 0 | 0 | 1 | 0 | 90.00% |
| yanxiaohan | 4 | 6 | 0 | 0 | 0 | 40.00% |
| zoli_dragapult | 6 | 4 | 0 | 0 | 0 | 60.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 137 | 584 | 0.234589 |
| correctness | 1 | 170 | 0.00588235 |
| length | 20317 | 170 | 119.512 |
| library_pressure | 482 | 170 | 2.83529 |
| outcome | 103 | 170 | 0.605882 |
| post_ko_relay | 369 | 523 | 0.705545 |
| powerful_hand | 34 | 170 | 0.2 |
| rare_candy | 34 | 170 | 0.2 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 34 | 170 | 0.2 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 1 |

## 重点案例

### 案例 1：yakitori_raging_bolt-001

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/v8_sol_deck_opt/baseline/run-835907c1bab4457b915fc5b8be8d9773/traces/yakitori_raging_bolt-001.json
- 证据步骤 54：

### 案例 2：crustle_v1-008

- 对手：crustle_v1
- failure_class：rare_candy_not_played
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/v8_sol_deck_opt/baseline/run-835907c1bab4457b915fc5b8be8d9773/traces/crustle_v1-008.json
- 证据步骤 18：

### 案例 3：kacchan_anti_wall-010

- 对手：kacchan_anti_wall
- failure_class：alakazam_not_active
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/v8_sol_deck_opt/baseline/run-835907c1bab4457b915fc5b8be8d9773/traces/kacchan_anti_wall-010.json
- 证据步骤 49：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 103 | 170 | 0.6058823529411764 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 20317 | 170 | 119.51176470588236 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 170 | 0.0058823529411764705 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 34 | 170 | 0.2 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 34 | 170 | 0.2 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 369 | 523 | 0.7055449330783938 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 482 | 170 | 2.835294117647059 |

## Setup and relay

- bridge opportunities: 44
- bridge completions: 0
- ability draws / game: 4.882352941176471
- normal draws / game: 2.011764705882353

## Attack quality

- attack submissions: 585
- resolved: 585
- unresolved: 0
- unknown prize: 1
- non-prize attacks: 137/584
