# 评测报告

运行 ID：run-f4c9cda8ce6a4dfc92e98b8db4c1fa00

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
| 胜 / 负 / 平 | 120 / 48 / 0 |
| 完成率 | 98.82% |
| 胜率 | 70.59% |
| 错误数 | 2 |
| 未完成 | 0 |
| 已完成 | 168 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kiyotah_abomasnow | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kiyotah_dragapult | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_iono | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_lucario | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kokinn_search | 6 | 4 | 0 | 0 | 0 | 60.00% |
| maktha_1084 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| nursrijan_lucario | 4 | 6 | 0 | 0 | 0 | 40.00% |
| penguin_915 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| pilkwang_v2 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| romanrozen_v9 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| sue_alakazam | 7 | 3 | 0 | 0 | 0 | 70.00% |
| yakitori_raging_bolt | 7 | 1 | 0 | 2 | 0 | 70.00% |
| yanxiaohan | 6 | 4 | 0 | 0 | 0 | 60.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 154 | 639 | 0.241002 |
| correctness | 2 | 170 | 0.0117647 |
| length | 22378 | 170 | 131.635 |
| library_pressure | 1031 | 170 | 6.06471 |
| outcome | 120 | 170 | 0.705882 |
| post_ko_relay | 275 | 446 | 0.616592 |
| powerful_hand | 46 | 170 | 0.270588 |
| rare_candy | 46 | 170 | 0.270588 |
| run_away_draw | 1 | 170 | 0.00588235 |
| setup_relay | 46 | 170 | 0.270588 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 2 |

## 重点案例

### 案例 1：yakitori_raging_bolt-005

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0001-yushin_ito_exact_bc_baseline/evaluation/run-f4c9cda8ce6a4dfc92e98b8db4c1fa00/traces/yakitori_raging_bolt-005.json
- 证据步骤 55：

### 案例 2：yakitori_raging_bolt-006

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0001-yushin_ito_exact_bc_baseline/evaluation/run-f4c9cda8ce6a4dfc92e98b8db4c1fa00/traces/yakitori_raging_bolt-006.json
- 证据步骤 7：

### 案例 3：crustle_v1-009

- 对手：crustle_v1
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0001-yushin_ito_exact_bc_baseline/evaluation/run-f4c9cda8ce6a4dfc92e98b8db4c1fa00/traces/crustle_v1-009.json
- 证据步骤 33：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 120 | 170 | 0.7058823529411765 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 22378 | 170 | 131.63529411764705 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 2 | 170 | 0.011764705882352941 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 46 | 170 | 0.27058823529411763 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 46 | 170 | 0.27058823529411763 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 275 | 446 | 0.6165919282511211 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 170 | 0.0058823529411764705 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1031 | 170 | 6.064705882352941 |

## Setup and relay

- bridge opportunities: 45
- bridge completions: 0
- ability draws / game: 7.329411764705882
- normal draws / game: 2.0823529411764707

## Attack quality

- attack submissions: 640
- resolved: 640
- unresolved: 0
- unknown prize: 1
- non-prize attacks: 154/639
