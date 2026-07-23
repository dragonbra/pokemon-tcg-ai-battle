# 评测报告

运行 ID：run-18b4105dc058495586334525e13b636c

## Metric profile

| 项目 | 数值 |
| --- | --- |
| profile | auto_iteration_v8_setup_relay |
| revision | 2 |
| metrics | outcome, length, correctness, powerful_hand, rare_candy, post_ko_relay, run_away_draw, library_pressure, setup_relay, attack_quality |

## 总体结果

| 项目 | 数值 |
| --- | ---: |
| 总对局 | 180 |
| 胜 / 负 / 平 | 128 / 52 / 0 |
| 完成率 | 100.00% |
| 胜率 | 71.11% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 180 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent_Aluxian | 8 | 2 | 0 | 0 | 0 | 80.00% |
| Agent_Lucario | 1 | 9 | 0 | 0 | 0 | 10.00% |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_abomasnow | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_dragapult | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_iono | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 7 | 3 | 0 | 0 | 0 | 70.00% |
| maktha_1084 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| nursrijan_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| penguin_915 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| pilkwang_v2 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 8 | 2 | 0 | 0 | 0 | 80.00% |
| yanxiaohan | 10 | 0 | 0 | 0 | 0 | 100.00% |
| zoli_dragapult | 9 | 1 | 0 | 0 | 0 | 90.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 173 | 684 | 0.252924 |
| correctness | 0 | 180 | 0 |
| length | 22291 | 180 | 123.839 |
| library_pressure | 1068 | 180 | 5.93333 |
| outcome | 128 | 180 | 0.711111 |
| post_ko_relay | 242 | 450 | 0.537778 |
| powerful_hand | 50 | 180 | 0.277778 |
| rare_candy | 50 | 180 | 0.277778 |
| run_away_draw | 1 | 180 | 0.00555556 |
| setup_relay | 50 | 180 | 0.277778 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-002

- 对手：Agent_Aluxian
- failure_class：no_legal_attack
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V3_b_d256_l2_lr5e4_s7/run-18b4105dc058495586334525e13b636c/traces/Agent_Aluxian-002.json
- 证据步骤 15：

### 案例 2：Agent_Lucario-001

- 对手：Agent_Lucario
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V3_b_d256_l2_lr5e4_s7/run-18b4105dc058495586334525e13b636c/traces/Agent_Lucario-001.json
- 证据步骤 18：

### 案例 3：crustle_wall-004

- 对手：crustle_wall
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V3_b_d256_l2_lr5e4_s7/run-18b4105dc058495586334525e13b636c/traces/crustle_wall-004.json
- 证据步骤 32：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 128 | 180 | 0.7111111111111111 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 22291 | 180 | 123.83888888888889 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 50 | 180 | 0.2777777777777778 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 50 | 180 | 0.2777777777777778 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 242 | 450 | 0.5377777777777778 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 180 | 0.005555555555555556 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1068 | 180 | 5.933333333333334 |

## Setup and relay

- bridge opportunities: 66
- bridge completions: 0
- ability draws / game: 7.266666666666667
- normal draws / game: 1.9944444444444445

## Attack quality

- attack submissions: 684
- resolved: 684
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 173/684
