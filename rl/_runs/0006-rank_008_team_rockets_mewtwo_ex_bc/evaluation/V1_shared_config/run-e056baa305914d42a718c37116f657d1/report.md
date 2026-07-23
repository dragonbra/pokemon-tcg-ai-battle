# 评测报告

运行 ID：run-e056baa305914d42a718c37116f657d1

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
| 胜 / 负 / 平 | 81 / 99 / 0 |
| 完成率 | 100.00% |
| 胜率 | 45.00% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 180 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent_Aluxian | 9 | 1 | 0 | 0 | 0 | 90.00% |
| Agent_Lucario | 1 | 9 | 0 | 0 | 0 | 10.00% |
| crustle_v1 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_abomasnow | 1 | 9 | 0 | 0 | 0 | 10.00% |
| kiyotah_dragapult | 2 | 8 | 0 | 0 | 0 | 20.00% |
| kiyotah_iono | 2 | 8 | 0 | 0 | 0 | 20.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 2 | 8 | 0 | 0 | 0 | 20.00% |
| maktha_1084 | 4 | 6 | 0 | 0 | 0 | 40.00% |
| nursrijan_lucario | 1 | 9 | 0 | 0 | 0 | 10.00% |
| penguin_915 | 3 | 7 | 0 | 0 | 0 | 30.00% |
| pilkwang_v2 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| romanrozen_v9 | 2 | 8 | 0 | 0 | 0 | 20.00% |
| sue_alakazam | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yanxiaohan | 2 | 8 | 0 | 0 | 0 | 20.00% |
| zoli_dragapult | 5 | 5 | 0 | 0 | 0 | 50.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 412 | 698 | 0.590258 |
| correctness | 0 | 180 | 0 |
| length | 26982 | 180 | 149.9 |
| library_pressure | 580 | 180 | 3.22222 |
| outcome | 81 | 180 | 0.45 |
| post_ko_relay | 600 | 600 | 1 |
| powerful_hand | 0 | 180 | 0 |
| rare_candy | 0 | 180 | 0 |
| run_away_draw | 0 | 180 | 0 |
| setup_relay | 0 | 180 | 0 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-001

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0006-rank_008_team_rockets_mewtwo_ex_bc/evaluation/V1_shared_config/run-e056baa305914d42a718c37116f657d1/traces/Agent_Aluxian-001.json
- 证据步骤 36：

### 案例 2：Agent_Lucario-001

- 对手：Agent_Lucario
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0006-rank_008_team_rockets_mewtwo_ex_bc/evaluation/V1_shared_config/run-e056baa305914d42a718c37116f657d1/traces/Agent_Lucario-001.json
- 证据步骤 43：

### 案例 3：crustle_v1-001

- 对手：crustle_v1
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0006-rank_008_team_rockets_mewtwo_ex_bc/evaluation/V1_shared_config/run-e056baa305914d42a718c37116f657d1/traces/crustle_v1-001.json
- 证据步骤 19：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 81 | 180 | 0.45 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 26982 | 180 | 149.9 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 600 | 600 | 1.0 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 580 | 180 | 3.2222222222222223 |

## Setup and relay

- bridge opportunities: 0
- bridge completions: 0
- ability draws / game: 4.988888888888889
- normal draws / game: 1.988888888888889

## Attack quality

- attack submissions: 702
- resolved: 702
- unresolved: 0
- unknown prize: 4
- non-prize attacks: 412/698
