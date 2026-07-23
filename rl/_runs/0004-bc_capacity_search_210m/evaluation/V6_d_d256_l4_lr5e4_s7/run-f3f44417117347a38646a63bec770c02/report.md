# 评测报告

运行 ID：run-f3f44417117347a38646a63bec770c02

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
| 胜 / 负 / 平 | 130 / 50 / 0 |
| 完成率 | 100.00% |
| 胜率 | 72.22% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 180 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent_Aluxian | 8 | 2 | 0 | 0 | 0 | 80.00% |
| Agent_Lucario | 5 | 5 | 0 | 0 | 0 | 50.00% |
| crustle_v1 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kiyotah_abomasnow | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_dragapult | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_iono | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_lucario | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kokinn_search | 6 | 4 | 0 | 0 | 0 | 60.00% |
| maktha_1084 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| nursrijan_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| penguin_915 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| pilkwang_v2 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| romanrozen_v9 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| sue_alakazam | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yanxiaohan | 8 | 2 | 0 | 0 | 0 | 80.00% |
| zoli_dragapult | 7 | 3 | 0 | 0 | 0 | 70.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 240 | 777 | 0.30888 |
| correctness | 0 | 180 | 0 |
| length | 23525 | 180 | 130.694 |
| library_pressure | 828 | 180 | 4.6 |
| outcome | 130 | 180 | 0.722222 |
| post_ko_relay | 252 | 474 | 0.531646 |
| powerful_hand | 42 | 180 | 0.233333 |
| rare_candy | 42 | 180 | 0.233333 |
| run_away_draw | 1 | 180 | 0.00555556 |
| setup_relay | 42 | 180 | 0.233333 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-009

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V6_d_d256_l4_lr5e4_s7/run-f3f44417117347a38646a63bec770c02/traces/Agent_Aluxian-009.json
- 证据步骤 23：

### 案例 2：Agent_Lucario-010

- 对手：Agent_Lucario
- failure_class：no_legal_attack
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V6_d_d256_l4_lr5e4_s7/run-f3f44417117347a38646a63bec770c02/traces/Agent_Lucario-010.json
- 证据步骤 34：

### 案例 3：zoli_dragapult-005

- 对手：zoli_dragapult
- failure_class：alakazam_not_active
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V6_d_d256_l4_lr5e4_s7/run-f3f44417117347a38646a63bec770c02/traces/zoli_dragapult-005.json
- 证据步骤 38：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 130 | 180 | 0.7222222222222222 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 23525 | 180 | 130.69444444444446 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 42 | 180 | 0.23333333333333334 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 42 | 180 | 0.23333333333333334 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 252 | 474 | 0.5316455696202531 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 180 | 0.005555555555555556 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 828 | 180 | 4.6 |

## Setup and relay

- bridge opportunities: 64
- bridge completions: 0
- ability draws / game: 7.066666666666666
- normal draws / game: 1.961111111111111

## Attack quality

- attack submissions: 777
- resolved: 777
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 240/777
