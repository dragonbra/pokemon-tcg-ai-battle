# 评测报告

运行 ID：run-124abc8103df4cee96c0c70463b40f28

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
| 胜 / 负 / 平 | 119 / 50 / 0 |
| 完成率 | 99.41% |
| 胜率 | 70.00% |
| 错误数 | 1 |
| 未完成 | 0 |
| 已完成 | 169 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_abomasnow | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_dragapult | 2 | 8 | 0 | 0 | 0 | 20.00% |
| kiyotah_iono | 2 | 8 | 0 | 0 | 0 | 20.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 5 | 5 | 0 | 0 | 0 | 50.00% |
| maktha_1084 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| nursrijan_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| penguin_915 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| pilkwang_v2 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| romanrozen_v9 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| sue_alakazam | 8 | 2 | 0 | 0 | 0 | 80.00% |
| yakitori_raging_bolt | 8 | 1 | 0 | 1 | 0 | 80.00% |
| yanxiaohan | 10 | 0 | 0 | 0 | 0 | 100.00% |
| zoli_dragapult | 9 | 1 | 0 | 0 | 0 | 90.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 143 | 621 | 0.230274 |
| correctness | 1 | 170 | 0.00588235 |
| length | 20375 | 170 | 119.853 |
| library_pressure | 511 | 170 | 3.00588 |
| outcome | 119 | 170 | 0.7 |
| post_ko_relay | 304 | 479 | 0.634656 |
| powerful_hand | 35 | 170 | 0.205882 |
| rare_candy | 35 | 170 | 0.205882 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 35 | 170 | 0.205882 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 1 |

## 重点案例

### 案例 1：yakitori_raging_bolt-002

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/alakazam_v8_luna_deck_opt_new_eval/iteration-002/run-124abc8103df4cee96c0c70463b40f28/traces/yakitori_raging_bolt-002.json
- 证据步骤 29：

### 案例 2：kacchan_anti_wall-004

- 对手：kacchan_anti_wall
- failure_class：rare_candy_not_played
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/alakazam_v8_luna_deck_opt_new_eval/iteration-002/run-124abc8103df4cee96c0c70463b40f28/traces/kacchan_anti_wall-004.json
- 证据步骤 56：

### 案例 3：kiyotah_abomasnow-007

- 对手：kiyotah_abomasnow
- failure_class：alakazam_not_active
- trace：/Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/work/auto-iteration/history_iterations/alakazam_v8_luna_deck_opt_new_eval/iteration-002/run-124abc8103df4cee96c0c70463b40f28/traces/kiyotah_abomasnow-007.json
- 证据步骤 24：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 119 | 170 | 0.7 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 20375 | 170 | 119.8529411764706 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 170 | 0.0058823529411764705 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 35 | 170 | 0.20588235294117646 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 35 | 170 | 0.20588235294117646 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 304 | 479 | 0.6346555323590815 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 511 | 170 | 3.0058823529411764 |

## Setup and relay

- bridge opportunities: 53
- bridge completions: 0
- ability draws / game: 5.064705882352941
- normal draws / game: 2.0823529411764707

## Attack quality

- attack submissions: 621
- resolved: 621
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 143/621
