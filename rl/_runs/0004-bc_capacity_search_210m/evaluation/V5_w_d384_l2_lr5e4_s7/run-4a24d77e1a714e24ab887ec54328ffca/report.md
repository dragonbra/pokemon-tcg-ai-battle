# 评测报告

运行 ID：run-4a24d77e1a714e24ab887ec54328ffca

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
| Agent_Lucario | 3 | 7 | 0 | 0 | 0 | 30.00% |
| crustle_v1 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_abomasnow | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kiyotah_dragapult | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_iono | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kokinn_search | 9 | 1 | 0 | 0 | 0 | 90.00% |
| maktha_1084 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| nursrijan_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| penguin_915 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| pilkwang_v2 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 9 | 1 | 0 | 0 | 0 | 90.00% |
| yanxiaohan | 10 | 0 | 0 | 0 | 0 | 100.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 218 | 754 | 0.289125 |
| correctness | 0 | 180 | 0 |
| length | 22466 | 180 | 124.811 |
| library_pressure | 864 | 180 | 4.8 |
| outcome | 128 | 180 | 0.711111 |
| post_ko_relay | 270 | 455 | 0.593407 |
| powerful_hand | 62 | 180 | 0.344444 |
| rare_candy | 62 | 180 | 0.344444 |
| run_away_draw | 1 | 180 | 0.00555556 |
| setup_relay | 62 | 180 | 0.344444 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-001

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V5_w_d384_l2_lr5e4_s7/run-4a24d77e1a714e24ab887ec54328ffca/traces/Agent_Aluxian-001.json
- 证据步骤 21：

### 案例 2：kacchan_anti_wall-010

- 对手：kacchan_anti_wall
- failure_class：no_legal_attack
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V5_w_d384_l2_lr5e4_s7/run-4a24d77e1a714e24ab887ec54328ffca/traces/kacchan_anti_wall-010.json
- 证据步骤 44：

### 案例 3：Agent_Lucario-002

- 对手：Agent_Lucario
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V5_w_d384_l2_lr5e4_s7/run-4a24d77e1a714e24ab887ec54328ffca/traces/Agent_Lucario-002.json
- 证据步骤 32：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 128 | 180 | 0.7111111111111111 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 22466 | 180 | 124.81111111111112 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 62 | 180 | 0.34444444444444444 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 62 | 180 | 0.34444444444444444 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 270 | 455 | 0.5934065934065934 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 180 | 0.005555555555555556 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 864 | 180 | 4.8 |

## Setup and relay

- bridge opportunities: 67
- bridge completions: 0
- ability draws / game: 6.711111111111111
- normal draws / game: 2.0

## Attack quality

- attack submissions: 754
- resolved: 754
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 218/754
