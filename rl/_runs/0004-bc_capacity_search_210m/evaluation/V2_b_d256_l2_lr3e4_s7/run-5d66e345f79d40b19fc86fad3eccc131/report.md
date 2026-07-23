# 评测报告

运行 ID：run-5d66e345f79d40b19fc86fad3eccc131

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
| 胜 / 负 / 平 | 122 / 58 / 0 |
| 完成率 | 100.00% |
| 胜率 | 67.78% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 180 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent_Aluxian | 7 | 3 | 0 | 0 | 0 | 70.00% |
| Agent_Lucario | 1 | 9 | 0 | 0 | 0 | 10.00% |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_abomasnow | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kiyotah_dragapult | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_iono | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kokinn_search | 6 | 4 | 0 | 0 | 0 | 60.00% |
| maktha_1084 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| nursrijan_lucario | 5 | 5 | 0 | 0 | 0 | 50.00% |
| penguin_915 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| pilkwang_v2 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| romanrozen_v9 | 2 | 8 | 0 | 0 | 0 | 20.00% |
| sue_alakazam | 9 | 1 | 0 | 0 | 0 | 90.00% |
| yanxiaohan | 7 | 3 | 0 | 0 | 0 | 70.00% |
| zoli_dragapult | 9 | 1 | 0 | 0 | 0 | 90.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 214 | 717 | 0.298466 |
| correctness | 0 | 180 | 0 |
| length | 22339 | 180 | 124.106 |
| library_pressure | 971 | 180 | 5.39444 |
| outcome | 122 | 180 | 0.677778 |
| post_ko_relay | 273 | 462 | 0.590909 |
| powerful_hand | 52 | 180 | 0.288889 |
| rare_candy | 52 | 180 | 0.288889 |
| run_away_draw | 4 | 180 | 0.0222222 |
| setup_relay | 52 | 180 | 0.288889 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-003

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V2_b_d256_l2_lr3e4_s7/run-5d66e345f79d40b19fc86fad3eccc131/traces/Agent_Aluxian-003.json
- 证据步骤 19：

### 案例 2：Agent_Lucario-001

- 对手：Agent_Lucario
- failure_class：no_legal_attack
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V2_b_d256_l2_lr3e4_s7/run-5d66e345f79d40b19fc86fad3eccc131/traces/Agent_Lucario-001.json
- 证据步骤 25：

### 案例 3：romanrozen_v9-005

- 对手：romanrozen_v9
- failure_class：alakazam_not_active
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V2_b_d256_l2_lr3e4_s7/run-5d66e345f79d40b19fc86fad3eccc131/traces/romanrozen_v9-005.json
- 证据步骤 31：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 122 | 180 | 0.6777777777777778 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 22339 | 180 | 124.10555555555555 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 52 | 180 | 0.28888888888888886 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 52 | 180 | 0.28888888888888886 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 273 | 462 | 0.5909090909090909 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 4 | 180 | 0.022222222222222223 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 971 | 180 | 5.394444444444445 |

## Setup and relay

- bridge opportunities: 67
- bridge completions: 0
- ability draws / game: 6.816666666666666
- normal draws / game: 2.0444444444444443

## Attack quality

- attack submissions: 717
- resolved: 717
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 214/717
