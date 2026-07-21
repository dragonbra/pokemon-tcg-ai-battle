# 评测报告

运行 ID：run-4b116f7130c64dd08f84e0f55e28b943

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
| 胜 / 负 / 平 | 118 / 52 / 0 |
| 完成率 | 100.00% |
| 胜率 | 69.41% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 170 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kiyotah_abomasnow | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_dragapult | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_iono | 3 | 7 | 0 | 0 | 0 | 30.00% |
| kiyotah_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kokinn_search | 7 | 3 | 0 | 0 | 0 | 70.00% |
| maktha_1084 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| nursrijan_lucario | 4 | 6 | 0 | 0 | 0 | 40.00% |
| penguin_915 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| pilkwang_v2 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 9 | 1 | 0 | 0 | 0 | 90.00% |
| yakitori_raging_bolt | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yanxiaohan | 5 | 5 | 0 | 0 | 0 | 50.00% |
| zoli_dragapult | 9 | 1 | 0 | 0 | 0 | 90.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 136 | 658 | 0.206687 |
| correctness | 0 | 170 | 0 |
| length | 20605 | 170 | 121.206 |
| library_pressure | 451 | 170 | 2.65294 |
| outcome | 118 | 170 | 0.694118 |
| post_ko_relay | 277 | 469 | 0.590618 |
| powerful_hand | 41 | 170 | 0.241176 |
| rare_candy | 41 | 170 | 0.241176 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 41 | 170 | 0.241176 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：kiyotah_abomasnow-002

- 对手：kiyotah_abomasnow
- failure_class：rare_candy_not_played
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/alakazam_v8_luna_deck_opt_new_eval/final/run-4b116f7130c64dd08f84e0f55e28b943/traces/kiyotah_abomasnow-002.json
- 证据步骤 43：

### 案例 2：penguin_915-009

- 对手：penguin_915
- failure_class：no_legal_attack
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/alakazam_v8_luna_deck_opt_new_eval/final/run-4b116f7130c64dd08f84e0f55e28b943/traces/penguin_915-009.json
- 证据步骤 33：

### 案例 3：kiyotah_dragapult-003

- 对手：kiyotah_dragapult
- failure_class：rare_candy_not_played
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/alakazam_v8_luna_deck_opt_new_eval/final/run-4b116f7130c64dd08f84e0f55e28b943/traces/kiyotah_dragapult-003.json
- 证据步骤 27：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 118 | 170 | 0.6941176470588235 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 20605 | 170 | 121.20588235294117 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 41 | 170 | 0.2411764705882353 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 41 | 170 | 0.2411764705882353 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 277 | 469 | 0.5906183368869936 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 451 | 170 | 2.652941176470588 |

## Setup and relay

- bridge opportunities: 45
- bridge completions: 0
- ability draws / game: 5.447058823529412
- normal draws / game: 2.088235294117647

## Attack quality

- attack submissions: 658
- resolved: 658
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 136/658
