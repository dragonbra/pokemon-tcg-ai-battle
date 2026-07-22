# 评测报告

运行 ID：run-1996bf89f89847ab92c4a0a8fa31143e

## Metric profile

| 项目 | 数值 |
| --- | --- |
| profile | auto_iteration_v8_setup_relay |
| revision | 2 |
| metrics | outcome, length, correctness, powerful_hand, rare_candy, post_ko_relay, run_away_draw, library_pressure, setup_relay, attack_quality |

## 总体结果

| 项目 | 数值 |
| --- | ---: |
| 总对局 | 200 |
| 胜 / 负 / 平 | 115 / 85 / 0 |
| 完成率 | 100.00% |
| 胜率 | 57.50% |
| 错误数 | 0 |
| 未完成 | 0 |
| 已完成 | 200 |

## 对局矩阵

| 对手 | 胜 | 负 | 平 | 错误 | 未完成 | 胜率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pokemon-tcg-ai-battle__aristophanivan__2plyexpectimax | 5 | 5 | 0 | 0 | 0 | 50.00% |
| pokemon-tcg-ai-battle__aristophanivan__improved-probabilistic-agent | 4 | 6 | 0 | 0 | 0 | 40.00% |
| pokemon-tcg-ai-battle__aristophanivan__multiply-agent-best-940-lb | 5 | 5 | 0 | 0 | 0 | 50.00% |
| pokemon-tcg-ai-battle__aristophanivan__probability-agent | 5 | 5 | 0 | 0 | 0 | 50.00% |
| pokemon-tcg-ai-battle__aristophanivan__probablity-v2 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| pokemon-tcg-ai-battle__daniilkrasnovvv__pokemon-conservative-probabilistic-agent | 7 | 3 | 0 | 0 | 0 | 70.00% |
| pokemon-tcg-ai-battle__harukiharada__crustle-wall-mirror-ok | 9 | 1 | 0 | 0 | 0 | 90.00% |
| pokemon-tcg-ai-battle__kokinnwakashuu__ptcg-diary-day1 | 7 | 3 | 0 | 0 | 0 | 70.00% |
| pokemon-tcg-ai-battle__kokinnwakashuu__ptcg-lucario-public-lab-anti-crustle-log | 7 | 3 | 0 | 0 | 0 | 70.00% |
| pokemon-tcg-ai-battle__makthanithin__improved-probabilistic-agent | 4 | 6 | 0 | 0 | 0 | 40.00% |
| pokemon-tcg-ai-battle__makthanithin__ptcg-mega-lucario-ex-v62 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| pokemon-tcg-ai-battle__map1e114514__iwapalace-resource-balance-1000-fixed-agent | 10 | 0 | 0 | 0 | 0 | 100.00% |
| pokemon-tcg-ai-battle__pixiux__ptcg-mega-lucario-ex-v62 | 3 | 7 | 0 | 0 | 0 | 30.00% |
| pokemon-tcg-ai-battle__pixiux__ptcg-mega-lucario-ex-v63 | 3 | 7 | 0 | 0 | 0 | 30.00% |
| pokemon-tcg-ai-battle__rahuljiwane__pokemon-tcg-rahul-jiwane7 | 6 | 4 | 0 | 0 | 0 | 60.00% |
| pokemon-tcg-ai-battle__romanrozen__strong-start-baseline-agent-v10-lb-950 | 5 | 5 | 0 | 0 | 0 | 50.00% |
| pokemon-tcg-ai-battle__serariagomes__heurestic-baseline-agent | 9 | 1 | 0 | 0 | 0 | 90.00% |
| pokemon-tcg-ai-battle__stealthtechnologies__pure-dragapult-ex-deck | 4 | 6 | 0 | 0 | 0 | 40.00% |
| pokemon-tcg-ai-battle__yaminh__the-pokemon-company-ai-challenge | 4 | 6 | 0 | 0 | 0 | 40.00% |
| pokemon-tcg-ai-battle__yoikoarmor__pokemon-tcg-gen0-lstm-pipeline-ja | 7 | 3 | 0 | 0 | 0 | 70.00% |

## 指标

| metric_id | 分子 | 分母 | value |
| --- | ---: | ---: | ---: |
| attack_quality | 169 | 680 | 0.248529 |
| correctness | 0 | 200 | 0 |
| length | 24005 | 200 | 120.025 |
| library_pressure | 419 | 200 | 2.095 |
| outcome | 115 | 200 | 0.575 |
| post_ko_relay | 459 | 689 | 0.666183 |
| powerful_hand | 36 | 200 | 0.18 |
| rare_candy | 36 | 200 | 0.18 |
| run_away_draw | 0 | 200 | 0 |
| setup_relay | 36 | 200 | 0.18 |

## failure_class 分布

| failure_class | 数量 |
| --- | ---: |

## 重点案例

### 案例 1：pokemon-tcg-ai-battle__aristophanivan__2plyexpectimax-002

- 对手：pokemon-tcg-ai-battle__aristophanivan__2plyexpectimax
- failure_class：rare_candy_not_played
- trace：/private/tmp/ptcg-arena-top20-alakazam-v9/run-1996bf89f89847ab92c4a0a8fa31143e/traces/pokemon-tcg-ai-battle__aristophanivan__2plyexpectimax-002.json
- 证据步骤 46：

### 案例 2：pokemon-tcg-ai-battle__aristophanivan__improved-probabilistic-agent-005

- 对手：pokemon-tcg-ai-battle__aristophanivan__improved-probabilistic-agent
- failure_class：alakazam_not_active
- trace：/private/tmp/ptcg-arena-top20-alakazam-v9/run-1996bf89f89847ab92c4a0a8fa31143e/traces/pokemon-tcg-ai-battle__aristophanivan__improved-probabilistic-agent-005.json
- 证据步骤 20：

### 案例 3：pokemon-tcg-ai-battle__aristophanivan__probability-agent-002

- 对手：pokemon-tcg-ai-battle__aristophanivan__probability-agent
- failure_class：no_legal_attack
- trace：/private/tmp/ptcg-arena-top20-alakazam-v9/run-1996bf89f89847ab92c4a0a8fa31143e/traces/pokemon-tcg-ai-battle__aristophanivan__probability-agent-002.json
- 证据步骤 28：

### 指标：outcome

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 115 | 200 | 0.575 |

### 指标：length

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 24005 | 200 | 120.025 |

### 指标：correctness

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 200 | 0.0 |

### 指标：powerful_hand

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 36 | 200 | 0.18 |

### 指标：rare_candy

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 36 | 200 | 0.18 |

### 指标：post_ko_relay

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 459 | 689 | 0.6661828737300436 |

### 指标：run_away_draw

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 0 | 200 | 0.0 |

### 指标：library_pressure

| 分子 | 分母 | value |
| ---: | ---: | ---: |
| 419 | 200 | 2.095 |

## Setup and relay

- bridge opportunities: 58
- bridge completions: 0
- ability draws / game: 4.935
- normal draws / game: 2.11

## Attack quality

- attack submissions: 680
- resolved: 680
- unresolved: 0
- unknown prize: 0
- non-prize attacks: 169/680
