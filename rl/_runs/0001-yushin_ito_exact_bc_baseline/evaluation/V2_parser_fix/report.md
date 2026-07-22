# 评测报告

运行 ID：run-873882eef9f04d74a8c76df4f680d1b0

## Metric profile

| 项目 | 数值 |
| --- | --- |
| profile | auto_iteration_v8_setup_relay |
| revision | 2 |
| metrics | outcome, length, correctness, powerful_hand, rare_candy, post_ko_relay, run_away_draw, library_pressure, setup_relay, attack_quality |

## 总体结果

| 项目 | 数值 |
| --- | ---: |
| 总对局 | 170 |
| 胜 / 负 / 平 | 122 / 42 / 0 |
| 完成率 | 96.47% |
| 胜率 | 71.76% |
| 错误数 | 6 |
| 未完成 | 0 |
| 已完成 | 164 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 10 | 0 | 0 | 0 | 0 | 100.00% |
| kacchan_anti_wall | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_abomasnow | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_dragapult | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_iono | 7 | 3 | 0 | 0 | 0 | 70.00% |
| kiyotah_lucario | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kokinn_search | 6 | 4 | 0 | 0 | 0 | 60.00% |
| maktha_1084 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| nursrijan_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| penguin_915 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| pilkwang_v2 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| romanrozen_v9 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| sue_alakazam | 9 | 1 | 0 | 0 | 0 | 90.00% |
| yakitori_raging_bolt | 4 | 0 | 0 | 6 | 0 | 40.00% |
| yanxiaohan | 9 | 1 | 0 | 0 | 0 | 90.00% |
| zoli_dragapult | 9 | 1 | 0 | 0 | 0 | 90.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 161 | 641 | 0.25117 |
| correctness | 6 | 170 | 0.0352941 |
| length | 20653 | 170 | 121.488 |
| library_pressure | 941 | 170 | 5.53529 |
| outcome | 122 | 170 | 0.717647 |
| post_ko_relay | 210 | 391 | 0.537084 |
| powerful_hand | 62 | 170 | 0.364706 |
| rare_candy | 62 | 170 | 0.364706 |
| run_away_draw | 2 | 170 | 0.0117647 |
| setup_relay | 62 | 170 | 0.364706 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 6 |

## 重点案例

### 案例 1：yakitori_raging_bolt-003

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0001-yushin_ito_exact_bc_baseline/evaluation/run-873882eef9f04d74a8c76df4f680d1b0/traces/yakitori_raging_bolt-003.json
- 证据步骤 87：

### 案例 2：yakitori_raging_bolt-004

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0001-yushin_ito_exact_bc_baseline/evaluation/run-873882eef9f04d74a8c76df4f680d1b0/traces/yakitori_raging_bolt-004.json
- 证据步骤 69：

### 案例 3：yakitori_raging_bolt-005

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0001-yushin_ito_exact_bc_baseline/evaluation/run-873882eef9f04d74a8c76df4f680d1b0/traces/yakitori_raging_bolt-005.json
- 证据步骤 59：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 122 | 170 | 0.7176470588235294 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 20653 | 170 | 121.48823529411764 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 6 | 170 | 0.03529411764705882 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 62 | 170 | 0.36470588235294116 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 62 | 170 | 0.36470588235294116 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 210 | 391 | 0.5370843989769821 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 2 | 170 | 0.011764705882352941 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 941 | 170 | 5.535294117647059 |

## Setup and relay

- bridge opportunities: 73
- bridge completions: 0
- ability draws / game: 8.470588235294118
- normal draws / game: 2.052941176470588

## Attack quality

- attack submissions: 641
- resolved: 641
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 161/641
