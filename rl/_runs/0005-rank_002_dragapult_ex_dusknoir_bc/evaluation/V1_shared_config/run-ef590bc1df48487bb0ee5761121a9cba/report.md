# 评测报告

运行 ID：run-ef590bc1df48487bb0ee5761121a9cba

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
| 胜 / 负 / 平 | 21 / 159 / 0 |
| 完成率 | 100.00% |
| 胜率 | 11.67% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 180 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent_Aluxian | 2 | 8 | 0 | 0 | 0 | 20.00% |
| Agent_Lucario | 0 | 10 | 0 | 0 | 0 | 0.00% |
| crustle_v1 | 0 | 10 | 0 | 0 | 0 | 0.00% |
| crustle_wall | 0 | 10 | 0 | 0 | 0 | 0.00% |
| kacchan_anti_wall | 0 | 10 | 0 | 0 | 0 | 0.00% |
| kiyotah_abomasnow | 0 | 10 | 0 | 0 | 0 | 0.00% |
| kiyotah_dragapult | 2 | 8 | 0 | 0 | 0 | 20.00% |
| kiyotah_iono | 0 | 10 | 0 | 0 | 0 | 0.00% |
| kiyotah_lucario | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kokinn_search | 1 | 9 | 0 | 0 | 0 | 10.00% |
| maktha_1084 | 0 | 10 | 0 | 0 | 0 | 0.00% |
| nursrijan_lucario | 0 | 10 | 0 | 0 | 0 | 0.00% |
| penguin_915 | 2 | 8 | 0 | 0 | 0 | 20.00% |
| pilkwang_v2 | 2 | 8 | 0 | 0 | 0 | 20.00% |
| romanrozen_v9 | 0 | 10 | 0 | 0 | 0 | 0.00% |
| sue_alakazam | 4 | 6 | 0 | 0 | 0 | 40.00% |
| yanxiaohan | 1 | 9 | 0 | 0 | 0 | 10.00% |
| zoli_dragapult | 2 | 8 | 0 | 0 | 0 | 20.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 616 | 668 | 0.922156 |
| correctness | 0 | 180 | 0 |
| length | 27311 | 180 | 151.728 |
| library_pressure | 705 | 180 | 3.91667 |
| outcome | 21 | 180 | 0.116667 |
| post_ko_relay | 731 | 731 | 1 |
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
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0005-rank_002_dragapult_ex_dusknoir_bc/evaluation/V1_shared_config/run-ef590bc1df48487bb0ee5761121a9cba/traces/Agent_Aluxian-001.json
- 证据步骤 25：

### 案例 2：Agent_Lucario-001

- 对手：Agent_Lucario
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0005-rank_002_dragapult_ex_dusknoir_bc/evaluation/V1_shared_config/run-ef590bc1df48487bb0ee5761121a9cba/traces/Agent_Lucario-001.json
- 证据步骤 41：

### 案例 3：crustle_v1-001

- 对手：crustle_v1
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0005-rank_002_dragapult_ex_dusknoir_bc/evaluation/V1_shared_config/run-ef590bc1df48487bb0ee5761121a9cba/traces/crustle_v1-001.json
- 证据步骤 20：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 21 | 180 | 0.11666666666666667 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 27311 | 180 | 151.7277777777778 |

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
| 731 | 731 | 1.0 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 705 | 180 | 3.9166666666666665 |

## Setup and relay

- bridge opportunities: 0
- bridge completions: 0
- ability draws / game: 4.1
- normal draws / game: 1.9944444444444445

## Attack quality

- attack submissions: 671
- resolved: 671
- unresolved: 0
- unknown prize: 3
- non-prize attacks: 616/668
