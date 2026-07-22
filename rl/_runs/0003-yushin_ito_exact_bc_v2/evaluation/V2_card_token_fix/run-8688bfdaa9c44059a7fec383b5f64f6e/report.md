# 评测报告

运行 ID：run-8688bfdaa9c44059a7fec383b5f64f6e

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
| 胜 / 负 / 平 | 136 / 44 / 0 |
| 完成率 | 100.00% |
| 胜率 | 75.56% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 180 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent_Aluxian | 7 | 3 | 0 | 0 | 0 | 70.00% |
| Agent_Lucario | 4 | 6 | 0 | 0 | 0 | 40.00% |
| crustle_v1 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_abomasnow | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_dragapult | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kiyotah_iono | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kokinn_search | 10 | 0 | 0 | 0 | 0 | 100.00% |
| maktha_1084 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| nursrijan_lucario | 7 | 3 | 0 | 0 | 0 | 70.00% |
| penguin_915 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| pilkwang_v2 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yanxiaohan | 8 | 2 | 0 | 0 | 0 | 80.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 194 | 714 | 0.271709 |
| correctness | 0 | 180 | 0 |
| length | 23298 | 180 | 129.433 |
| library_pressure | 957 | 180 | 5.31667 |
| outcome | 136 | 180 | 0.755556 |
| post_ko_relay | 309 | 487 | 0.634497 |
| powerful_hand | 62 | 180 | 0.344444 |
| rare_candy | 62 | 180 | 0.344444 |
| run_away_draw | 3 | 180 | 0.0166667 |
| setup_relay | 62 | 180 | 0.344444 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-005

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0003-yushin_ito_exact_bc_v2/evaluation/V2_card_token_fix/run-8688bfdaa9c44059a7fec383b5f64f6e/traces/Agent_Aluxian-005.json
- 证据步骤 41：

### 案例 2：Agent_Lucario-002

- 对手：Agent_Lucario
- failure_class：no_legal_attack
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0003-yushin_ito_exact_bc_v2/evaluation/V2_card_token_fix/run-8688bfdaa9c44059a7fec383b5f64f6e/traces/Agent_Lucario-002.json
- 证据步骤 42：

### 案例 3：crustle_v1-006

- 对手：crustle_v1
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0003-yushin_ito_exact_bc_v2/evaluation/V2_card_token_fix/run-8688bfdaa9c44059a7fec383b5f64f6e/traces/crustle_v1-006.json
- 证据步骤 20：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 136 | 180 | 0.7555555555555555 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 23298 | 180 | 129.43333333333334 |

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
| 309 | 487 | 0.6344969199178645 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 3 | 180 | 0.016666666666666666 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 957 | 180 | 5.316666666666666 |

## Setup and relay

- bridge opportunities: 65
- bridge completions: 0
- ability draws / game: 7.655555555555556
- normal draws / game: 1.9777777777777779

## Attack quality

- attack submissions: 716
- resolved: 716
- unresolved: 0
- unknown prize: 2
- non-prize attacks: 194/714
