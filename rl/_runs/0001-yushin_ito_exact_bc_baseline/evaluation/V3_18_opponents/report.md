# 评测报告

运行 ID：run-b127ca927ef64a20a4af1664817fd512

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
| 胜 / 负 / 平 | 125 / 55 / 0 |
| 完成率 | 100.00% |
| 胜率 | 69.44% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 180 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Agent_Aluxian | 7 | 3 | 0 | 0 | 0 | 70.00% |
| Agent_Lucario | 4 | 6 | 0 | 0 | 0 | 40.00% |
| crustle_v1 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_abomasnow | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_dragapult | 2 | 8 | 0 | 0 | 0 | 20.00% |
| kiyotah_iono | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_lucario | 5 | 5 | 0 | 0 | 0 | 50.00% |
| kokinn_search | 8 | 2 | 0 | 0 | 0 | 80.00% |
| maktha_1084 | 9 | 1 | 0 | 0 | 0 | 90.00% |
| nursrijan_lucario | 5 | 5 | 0 | 0 | 0 | 50.00% |
| penguin_915 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| pilkwang_v2 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yanxiaohan | 7 | 3 | 0 | 0 | 0 | 70.00% |
| zoli_dragapult | 10 | 0 | 0 | 0 | 0 | 100.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 197 | 711 | 0.277075 |
| correctness | 0 | 180 | 0 |
| length | 22604 | 180 | 125.578 |
| library_pressure | 1071 | 180 | 5.95 |
| outcome | 125 | 180 | 0.694444 |
| post_ko_relay | 278 | 464 | 0.599138 |
| powerful_hand | 55 | 180 | 0.305556 |
| rare_candy | 55 | 180 | 0.305556 |
| run_away_draw | 3 | 180 | 0.0166667 |
| setup_relay | 55 | 180 | 0.305556 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：Agent_Aluxian-003

- 对手：Agent_Aluxian
- failure_class：rare_candy_not_played
- trace：/tmp/bceb1-18-opponents-evaluation/run-b127ca927ef64a20a4af1664817fd512/traces/Agent_Aluxian-003.json
- 证据步骤 18：

### 案例 2：crustle_v1-005

- 对手：crustle_v1
- failure_class：no_legal_attack
- trace：/tmp/bceb1-18-opponents-evaluation/run-b127ca927ef64a20a4af1664817fd512/traces/crustle_v1-005.json
- 证据步骤 10：

### 案例 3：Agent_Lucario-001

- 对手：Agent_Lucario
- failure_class：rare_candy_not_played
- trace：/tmp/bceb1-18-opponents-evaluation/run-b127ca927ef64a20a4af1664817fd512/traces/Agent_Lucario-001.json
- 证据步骤 22：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 125 | 180 | 0.6944444444444444 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 22604 | 180 | 125.57777777777778 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 180 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 55 | 180 | 0.3055555555555556 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 55 | 180 | 0.3055555555555556 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 278 | 464 | 0.5991379310344828 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 3 | 180 | 0.016666666666666666 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 1071 | 180 | 5.95 |

## Setup and relay

- bridge opportunities: 74
- bridge completions: 0
- ability draws / game: 7.5
- normal draws / game: 2.0388888888888888

## Attack quality

- attack submissions: 711
- resolved: 711
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 197/711
