# 评测报告

运行 ID：run-1c7ed4b7c6a04b0b87dec8449d4abda1

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
| 胜 / 负 / 平 | 134 / 46 / 0 |
| 完成率 | 100.00% |
| 胜率 | 74.44% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 180 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent_Aluxian | 7 | 3 | 0 | 0 | 0 | 70.00% |
| Agent_Lucario | 3 | 7 | 0 | 0 | 0 | 30.00% |
| crustle_v1 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_abomasnow | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_dragapult | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_iono | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 9 | 1 | 0 | 0 | 0 | 90.00% |
| maktha_1084 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| nursrijan_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| penguin_915 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| pilkwang_v2 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yanxiaohan | 9 | 1 | 0 | 0 | 0 | 90.00% |
| zoli_dragapult | 9 | 1 | 0 | 0 | 0 | 90.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 192 | 738 | 0.260163 |
| correctness | 0 | 180 | 0 |
| length | 23428 | 180 | 130.156 |
| library_pressure | 1032 | 180 | 5.73333 |
| outcome | 134 | 180 | 0.744444 |
| post_ko_relay | 285 | 470 | 0.606383 |
| powerful_hand | 61 | 180 | 0.338889 |
| rare_candy | 61 | 180 | 0.338889 |
| run_away_draw | 1 | 180 | 0.00555556 |
| setup_relay | 61 | 180 | 0.338889 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-003

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V9_w_d384_l2_lr5e4_s17/run-1c7ed4b7c6a04b0b87dec8449d4abda1/traces/Agent_Aluxian-003.json
- 证据步骤 25：

### 案例 2：Agent_Lucario-004

- 对手：Agent_Lucario
- failure_class：alakazam_not_active
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V9_w_d384_l2_lr5e4_s17/run-1c7ed4b7c6a04b0b87dec8449d4abda1/traces/Agent_Lucario-004.json
- 证据步骤 20：

### 案例 3：crustle_v1-010

- 对手：crustle_v1
- failure_class：no_legal_attack
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V9_w_d384_l2_lr5e4_s17/run-1c7ed4b7c6a04b0b87dec8449d4abda1/traces/crustle_v1-010.json
- 证据步骤 11：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 134 | 180 | 0.7444444444444445 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 23428 | 180 | 130.15555555555557 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 61 | 180 | 0.3388888888888889 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 61 | 180 | 0.3388888888888889 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 285 | 470 | 0.6063829787234043 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 180 | 0.005555555555555556 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1032 | 180 | 5.733333333333333 |

## Setup and relay

- bridge opportunities: 64
- bridge completions: 0
- ability draws / game: 8.027777777777779
- normal draws / game: 2.0

## Attack quality

- attack submissions: 738
- resolved: 738
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 192/738
