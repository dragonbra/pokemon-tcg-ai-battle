# 评测报告

运行 ID：run-45cc69e099d64087b0573548b2b3007d

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
| 胜 / 负 / 平 | 145 / 35 / 0 |
| 完成率 | 100.00% |
| 胜率 | 80.56% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 180 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent_Aluxian | 6 | 4 | 0 | 0 | 0 | 60.00% |
| Agent_Lucario | 3 | 7 | 0 | 0 | 0 | 30.00% |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kiyotah_abomasnow | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kiyotah_dragapult | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_iono | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_lucario | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kokinn_search | 10 | 0 | 0 | 0 | 0 | 100.00% |
| maktha_1084 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| nursrijan_lucario | 9 | 1 | 0 | 0 | 0 | 90.00% |
| penguin_915 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| pilkwang_v2 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| romanrozen_v9 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| sue_alakazam | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yanxiaohan | 9 | 1 | 0 | 0 | 0 | 90.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 179 | 728 | 0.245879 |
| correctness | 0 | 180 | 0 |
| length | 22468 | 180 | 124.822 |
| library_pressure | 863 | 180 | 4.79444 |
| outcome | 145 | 180 | 0.805556 |
| post_ko_relay | 291 | 443 | 0.656885 |
| powerful_hand | 64 | 180 | 0.355556 |
| rare_candy | 64 | 180 | 0.355556 |
| run_away_draw | 1 | 180 | 0.00555556 |
| setup_relay | 64 | 180 | 0.355556 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-002

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V4_s_d192_l2_lr5e4_s7/run-45cc69e099d64087b0573548b2b3007d/traces/Agent_Aluxian-002.json
- 证据步骤 13：

### 案例 2：Agent_Lucario-001

- 对手：Agent_Lucario
- failure_class：alakazam_not_active
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V4_s_d192_l2_lr5e4_s7/run-45cc69e099d64087b0573548b2b3007d/traces/Agent_Lucario-001.json
- 证据步骤 21：

### 案例 3：kacchan_anti_wall-010

- 对手：kacchan_anti_wall
- failure_class：rare_candy_not_played
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0004-bc_capacity_search_210m/evaluation/V4_s_d192_l2_lr5e4_s7/run-45cc69e099d64087b0573548b2b3007d/traces/kacchan_anti_wall-010.json
- 证据步骤 70：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 145 | 180 | 0.8055555555555556 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 22468 | 180 | 124.82222222222222 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 64 | 180 | 0.35555555555555557 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 64 | 180 | 0.35555555555555557 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 291 | 443 | 0.6568848758465011 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1 | 180 | 0.005555555555555556 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 863 | 180 | 4.794444444444444 |

## Setup and relay

- bridge opportunities: 63
- bridge completions: 0
- ability draws / game: 7.461111111111111
- normal draws / game: 1.9944444444444445

## Attack quality

- attack submissions: 728
- resolved: 728
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 179/728
