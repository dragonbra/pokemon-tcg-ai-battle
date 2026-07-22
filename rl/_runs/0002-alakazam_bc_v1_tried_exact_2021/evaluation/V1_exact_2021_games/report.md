# 评测报告

运行 ID：run-d2de0a4992f5490e874804d2901f04aa

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
| 胜 / 负 / 平 | 118 / 47 / 0 |
| 完成率 | 97.06% |
| 胜率 | 69.41% |
| 错误数 | 5 |
| 未完成 | 0 |
| 已完成 | 165 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crustle_v1 | 10 | 0 | 0 | 0 | 0 | 100.00% |
| crustle_wall | 9 | 1 | 0 | 0 | 0 | 90.00% |
| kacchan_anti_wall | 4 | 6 | 0 | 0 | 0 | 40.00% |
| kiyotah_abomasnow | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |
| kiyotah_iono | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kiyotah_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| kokinn_search | 4 | 6 | 0 | 0 | 0 | 40.00% |
| maktha_1084 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| nursrijan_lucario | 6 | 4 | 0 | 0 | 0 | 60.00% |
| penguin_915 | 8 | 2 | 0 | 0 | 0 | 80.00% |
| pilkwang_v2 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| romanrozen_v9 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| sue_alakazam | 10 | 0 | 0 | 0 | 0 | 100.00% |
| yakitori_raging_bolt | 5 | 0 | 0 | 5 | 0 | 50.00% |
| yanxiaohan | 7 | 3 | 0 | 0 | 0 | 70.00% |
| zoli_dragapult | 8 | 2 | 0 | 0 | 0 | 80.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 216 | 684 | 0.315789 |
| correctness | 5 | 170 | 0.0294118 |
| length | 20093 | 170 | 118.194 |
| library_pressure | 616 | 170 | 3.62353 |
| outcome | 118 | 170 | 0.694118 |
| post_ko_relay | 292 | 424 | 0.688679 |
| powerful_hand | 44 | 170 | 0.258824 |
| rare_candy | 44 | 170 | 0.258824 |
| run_away_draw | 0 | 170 | 0 |
| setup_relay | 44 | 170 | 0.258824 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |
| engine_error | 5 |

## 重点案例

### 案例 1：yakitori_raging_bolt-002

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0002-alakazam_bc_v1_tried_exact_2021/evaluation/run-d2de0a4992f5490e874804d2901f04aa/traces/yakitori_raging_bolt-002.json
- 证据步骤 82：

### 案例 2：yakitori_raging_bolt-004

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0002-alakazam_bc_v1_tried_exact_2021/evaluation/run-d2de0a4992f5490e874804d2901f04aa/traces/yakitori_raging_bolt-004.json
- 证据步骤 66：

### 案例 3：yakitori_raging_bolt-006

- 对手：yakitori_raging_bolt
- failure_class：error
- trace：/home/dragon_bra/repos/pokemon-tcg-ai-battle/rl/_runs/0002-alakazam_bc_v1_tried_exact_2021/evaluation/run-d2de0a4992f5490e874804d2901f04aa/traces/yakitori_raging_bolt-006.json
- 证据步骤 66：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 118 | 170 | 0.6941176470588235 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 20093 | 170 | 118.19411764705882 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 5 | 170 | 0.029411764705882353 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 44 | 170 | 0.25882352941176473 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 44 | 170 | 0.25882352941176473 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 292 | 424 | 0.6886792452830188 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 170 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 616 | 170 | 3.623529411764706 |

## Setup and relay

- bridge opportunities: 78
- bridge completions: 0
- ability draws / game: 7.358823529411764
- normal draws / game: 2.0823529411764707

## Attack quality

- attack submissions: 684
- resolved: 684
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 216/684
