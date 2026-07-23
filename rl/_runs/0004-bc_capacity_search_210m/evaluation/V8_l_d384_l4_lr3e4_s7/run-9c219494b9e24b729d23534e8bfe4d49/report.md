# 评测报告

运行 ID：run-9c219494b9e24b729d23534e8bfe4d49

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
| Agent_Aluxian | 8 | 2 | 0 | 0 | 0 | 80.00% |
| Agent_Lucario | 5 | 5 | 0 | 0 | 0 | 50.00% |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_abomasnow | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kiyotah_dragapult | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_iono | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 7 | 3 | 0 | 0 | 0 | 70.00% |
| maktha_1084 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| nursrijan_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| penguin_915 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| pilkwang_v2 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| romanrozen_v9 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| sue_alakazam | 8 | 2 | 0 | 0 | 0 | 80.00% |
| yanxiaohan | 9 | 1 | 0 | 0 | 0 | 90.00% |
| zoli_dragapult | 7 | 3 | 0 | 0 | 0 | 70.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 207 | 721 | 0.287101 |
| correctness | 0 | 180 | 0 |
| length | 24135 | 180 | 134.083 |
| library_pressure | 975 | 180 | 5.41667 |
| outcome | 134 | 180 | 0.744444 |
| post_ko_relay | 298 | 523 | 0.56979 |
| powerful_hand | 50 | 180 | 0.277778 |
| rare_candy | 50 | 180 | 0.277778 |
| run_away_draw | 1 | 180 | 0.00555556 |
| setup_relay | 50 | 180 | 0.277778 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-008

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V8_l_d384_l4_lr3e4_s7/run-9c219494b9e24b729d23534e8bfe4d49/traces/Agent_Aluxian-008.json
- 证据步骤 35：

### 案例 2：Agent_Lucario-006

- 对手：Agent_Lucario
- failure_class：no_legal_attack
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V8_l_d384_l4_lr3e4_s7/run-9c219494b9e24b729d23534e8bfe4d49/traces/Agent_Lucario-006.json
- 证据步骤 31：

### 案例 3：crustle_wall-002

- 对手：crustle_wall
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V8_l_d384_l4_lr3e4_s7/run-9c219494b9e24b729d23534e8bfe4d49/traces/crustle_wall-002.json
- 证据步骤 17：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 134 | 180 | 0.7444444444444445 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 24135 | 180 | 134.08333333333334 |

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
| 298 | 523 | 0.5697896749521989 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 180 | 0.005555555555555556 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 975 | 180 | 5.416666666666667 |

## Setup and relay

- bridge opportunities: 70
- bridge completions: 0
- ability draws / game: 7.1722222222222225
- normal draws / game: 2.0388888888888888

## Attack quality

- attack submissions: 721
- resolved: 721
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 207/721
