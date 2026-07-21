# 评测报告

运行 ID：run-1fe111daa83241a38a6e36b8d9233f8f

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
| 胜 / 负 / 平 | 111 / 57 / 0 |
| 完成率 | 98.82% |
| 胜率 | 65.29% |
| 错误数 | 2 |
| 未完成 | 0 |
| 已完成 | 168 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_abomasnow | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_dragapult | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_iono | 1 | 9 | 0 | 0 | 0 | 10.00% |
| kiyotah_lucario | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kokinn_search | 7 | 3 | 0 | 0 | 0 | 70.00% |
| maktha_1084 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| nursrijan_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| penguin_915 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| pilkwang_v2 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| romanrozen_v9 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| sue_alakazam | 6 | 4 | 0 | 0 | 0 | 60.00% |
| yakitori_raging_bolt | 6 | 2 | 0 | 2 | 0 | 60.00% |
| yanxiaohan | 7 | 3 | 0 | 0 | 0 | 70.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 128 | 561 | 0.228164 |
| correctness | 2 | 170 | 0.0117647 |
| length | 19593 | 170 | 115.253 |
| library_pressure | 496 | 170 | 2.91765 |
| outcome | 111 | 170 | 0.652941 |
| post_ko_relay | 299 | 469 | 0.637527 |
| powerful_hand | 37 | 170 | 0.217647 |
| rare_candy | 37 | 170 | 0.217647 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 37 | 170 | 0.217647 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 2 |

## 重点案例

### 案例 1：yakitori_raging_bolt-002

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/alakazam_v8_luna_deck_opt_new_eval/iteration-001/run-1fe111daa83241a38a6e36b8d9233f8f/traces/yakitori_raging_bolt-002.json
- 证据步骤 30：

### 案例 2：yakitori_raging_bolt-010

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/alakazam_v8_luna_deck_opt_new_eval/iteration-001/run-1fe111daa83241a38a6e36b8d9233f8f/traces/yakitori_raging_bolt-010.json
- 证据步骤 47：

### 案例 3：crustle_wall-006

- 对手：crustle_wall
- failure_class：rare_candy_not_played
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/alakazam_v8_luna_deck_opt_new_eval/iteration-001/run-1fe111daa83241a38a6e36b8d9233f8f/traces/crustle_wall-006.json
- 证据步骤 15：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 111 | 170 | 0.6529411764705882 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 19593 | 170 | 115.25294117647059 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 2 | 170 | 0.011764705882352941 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 37 | 170 | 0.21764705882352942 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 37 | 170 | 0.21764705882352942 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 299 | 469 | 0.6375266524520256 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 496 | 170 | 2.9176470588235293 |

## Setup and relay

- bridge opportunities: 42
- bridge completions: 0
- ability draws / game: 4.270588235294118
- normal draws / game: 2.1

## Attack quality

- attack submissions: 561
- resolved: 561
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 128/561
