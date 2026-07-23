# 评测报告

运行 ID：run-263d3c8f76c34514a00d0a04eb0d8e27

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
| crustle_v1 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_abomasnow | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kiyotah_dragapult | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_iono | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kokinn_search | 7 | 3 | 0 | 0 | 0 | 70.00% |
| maktha_1084 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| nursrijan_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| penguin_915 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| pilkwang_v2 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yanxiaohan | 9 | 1 | 0 | 0 | 0 | 90.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 190 | 707 | 0.268741 |
| correctness | 0 | 180 | 0 |
| length | 23374 | 180 | 129.856 |
| library_pressure | 1078 | 180 | 5.98889 |
| outcome | 134 | 180 | 0.744444 |
| post_ko_relay | 290 | 461 | 0.629067 |
| powerful_hand | 44 | 180 | 0.244444 |
| rare_candy | 44 | 180 | 0.244444 |
| run_away_draw | 4 | 180 | 0.0222222 |
| setup_relay | 44 | 180 | 0.244444 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-006

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V7_l_d384_l4_lr5e4_s7/run-263d3c8f76c34514a00d0a04eb0d8e27/traces/Agent_Aluxian-006.json
- 证据步骤 22：

### 案例 2：Agent_Lucario-009

- 对手：Agent_Lucario
- failure_class：no_legal_attack
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V7_l_d384_l4_lr5e4_s7/run-263d3c8f76c34514a00d0a04eb0d8e27/traces/Agent_Lucario-009.json
- 证据步骤 33：

### 案例 3：crustle_v1-005

- 对手：crustle_v1
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V7_l_d384_l4_lr5e4_s7/run-263d3c8f76c34514a00d0a04eb0d8e27/traces/crustle_v1-005.json
- 证据步骤 16：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 134 | 180 | 0.7444444444444445 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 23374 | 180 | 129.85555555555555 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 44 | 180 | 0.24444444444444444 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 44 | 180 | 0.24444444444444444 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 290 | 461 | 0.6290672451193059 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 4 | 180 | 0.022222222222222223 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1078 | 180 | 5.988888888888889 |

## Setup and relay

- bridge opportunities: 71
- bridge completions: 0
- ability draws / game: 7.316666666666666
- normal draws / game: 2.022222222222222

## Attack quality

- attack submissions: 707
- resolved: 707
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 190/707
