# 评测报告

运行 ID：run-477f2bfcad10439db8befe0810b00ecd

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
| 胜 / 负 / 平 | 127 / 53 / 0 |
| 完成率 | 100.00% |
| 胜率 | 70.56% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 180 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent_Aluxian | 7 | 3 | 0 | 0 | 0 | 70.00% |
| Agent_Lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| crustle_v1 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_abomasnow | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_dragapult | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_iono | 3 | 7 | 0 | 0 | 0 | 30.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 6 | 4 | 0 | 0 | 0 | 60.00% |
| maktha_1084 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| nursrijan_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| penguin_915 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| pilkwang_v2 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| romanrozen_v9 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| sue_alakazam | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yanxiaohan | 9 | 1 | 0 | 0 | 0 | 90.00% |
| zoli_dragapult | 9 | 1 | 0 | 0 | 0 | 90.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 197 | 719 | 0.273992 |
| correctness | 0 | 180 | 0 |
| length | 22390 | 180 | 124.389 |
| library_pressure | 859 | 180 | 4.77222 |
| outcome | 127 | 180 | 0.705556 |
| post_ko_relay | 323 | 476 | 0.678571 |
| powerful_hand | 46 | 180 | 0.255556 |
| rare_candy | 46 | 180 | 0.255556 |
| run_away_draw | 1 | 180 | 0.00555556 |
| setup_relay | 46 | 180 | 0.255556 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-009

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V1_b_d256_l2_lr1e4_s7/run-477f2bfcad10439db8befe0810b00ecd/traces/Agent_Aluxian-009.json
- 证据步骤 36：

### 案例 2：Agent_Lucario-006

- 对手：Agent_Lucario
- failure_class：alakazam_not_active
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V1_b_d256_l2_lr1e4_s7/run-477f2bfcad10439db8befe0810b00ecd/traces/Agent_Lucario-006.json
- 证据步骤 27：

### 案例 3：kacchan_anti_wall-003

- 对手：kacchan_anti_wall
- failure_class：no_legal_attack
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V1_b_d256_l2_lr1e4_s7/run-477f2bfcad10439db8befe0810b00ecd/traces/kacchan_anti_wall-003.json
- 证据步骤 28：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 127 | 180 | 0.7055555555555556 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 22390 | 180 | 124.38888888888889 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 46 | 180 | 0.25555555555555554 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 46 | 180 | 0.25555555555555554 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 323 | 476 | 0.6785714285714286 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 180 | 0.005555555555555556 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 859 | 180 | 4.772222222222222 |

## Setup and relay

- bridge opportunities: 70
- bridge completions: 0
- ability draws / game: 6.822222222222222
- normal draws / game: 1.9777777777777779

## Attack quality

- attack submissions: 719
- resolved: 719
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 197/719
