# 评测报告

运行 ID：run-30625b6831594e0795328c9a5e6093fa

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
| 胜 / 负 / 平 | 116 / 53 / 0 |
| 完成率 | 99.41% |
| 胜率 | 68.24% |
| 错误数 | 1 |
| 未完成 | 0 |
| 已完成 | 169 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_abomasnow | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_dragapult | 3 | 7 | 0 | 0 | 0 | 30.00% |
| kiyotah_iono | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 8 | 2 | 0 | 0 | 0 | 80.00% |
| maktha_1084 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| nursrijan_lucario | 4 | 6 | 0 | 0 | 0 | 40.00% |
| penguin_915 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| pilkwang_v2 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| romanrozen_v9 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| sue_alakazam | 5 | 5 | 0 | 0 | 0 | 50.00% |
| yakitori_raging_bolt | 9 | 0 | 0 | 1 | 0 | 90.00% |
| yanxiaohan | 6 | 4 | 0 | 0 | 0 | 60.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 127 | 604 | 0.210265 |
| correctness | 1 | 170 | 0.00588235 |
| length | 19972 | 170 | 117.482 |
| library_pressure | 535 | 170 | 3.14706 |
| outcome | 116 | 170 | 0.682353 |
| post_ko_relay | 319 | 473 | 0.674419 |
| powerful_hand | 48 | 170 | 0.282353 |
| rare_candy | 48 | 170 | 0.282353 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 48 | 170 | 0.282353 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 1 |

## 重点案例

### 案例 1：yakitori_raging_bolt-009

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/v8_sol_deck_opt/iteration-003/run-30625b6831594e0795328c9a5e6093fa/traces/yakitori_raging_bolt-009.json
- 证据步骤 103：

### 案例 2：kacchan_anti_wall-002

- 对手：kacchan_anti_wall
- failure_class：rare_candy_not_played
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/v8_sol_deck_opt/iteration-003/run-30625b6831594e0795328c9a5e6093fa/traces/kacchan_anti_wall-002.json
- 证据步骤 33：

### 案例 3：romanrozen_v9-010

- 对手：romanrozen_v9
- failure_class：no_legal_attack
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/v8_sol_deck_opt/iteration-003/run-30625b6831594e0795328c9a5e6093fa/traces/romanrozen_v9-010.json
- 证据步骤 51：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 116 | 170 | 0.6823529411764706 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 19972 | 170 | 117.48235294117647 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 170 | 0.0058823529411764705 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 48 | 170 | 0.2823529411764706 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 48 | 170 | 0.2823529411764706 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 319 | 473 | 0.6744186046511628 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 535 | 170 | 3.1470588235294117 |

## Setup and relay

- bridge opportunities: 56
- bridge completions: 0
- ability draws / game: 5.247058823529412
- normal draws / game: 2.023529411764706

## Attack quality

- attack submissions: 604
- resolved: 604
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 127/604
