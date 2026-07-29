# 0020 零训练跨卡组泛化验收

## 结论

验收通过。0019 选中的 epoch-13 neutral checkpoint 在不更新任何权重、只替换 exact 60-card deck conditioning 的情况下，使用五个不同构筑完成 1,500 局 official-engine Arena 对局：964 胜、536 负，综合胜率 64.27%，完成率 100%，0 error、0 unfinished。

这证明的是可插拔 runtime 与实用零训练策略迁移，不代表五套牌等强，也不证明后续 RL 一定提升。分 deck 结果从 Dragapult 的 42.33% 到 Marnie 的 80.00%，所以后续必须保留 deck-specific rollout、版本、checkpoint、opponent snapshot 与评测边界。

## 固定变量

- checkpoint: `epoch-0013-da9b13d6f82d19d4.pt`
- checkpoint SHA-256: `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`
- deployment persona: `source_id=0`
- ontology SHA-256: `8144c63e512a2a00fabaaf2b19cd002c5b59c25fb0763a112256848460113a4d`
- opponent catalog SHA-256: `92221e5a6b080d46f85d499e491a371a610a94087d7f001a1663bc657cfbe5c0`
- protocol: 30 opponents × 10 games, alternate first/second, 8 workers, 1 CPU thread/worker
- engine: unmodified official runtime; `engine_cuda/` remains research-only and rollout-inadmissible
- training updates: 0

## 正式结果

| 版本 | 胜率 | 胜-负 | 先手 | 后手 | 报告 |
|---|---:|---:|---:|---:|---|
| V2 Dragapult | 42.33% | 127-173 | 43.33% | 41.33% | `evaluation/V2_zero_shot_dragapult.html` |
| V3 Alakazam | 73.00% | 219-81 | 76.67% | 69.33% | `evaluation/V3_zero_shot_alakazam.html` |
| V4 Marnie | 80.00% | 240-60 | 84.67% | 75.33% | `evaluation/V4_zero_shot_marnie.html` |
| V5 Lucario | 54.33% | 163-137 | 61.33% | 47.33% | `evaluation/V5_zero_shot_lucario.html` |
| V6 Kangaskhan | 71.67% | 215-85 | 74.67% | 68.67% | `evaluation/V6_zero_shot_kangaskhan.html` |

## 证据边界

“合法且能完成”由 1,500/1,500 完成、零 error 支持；“具备实用强度”由五套中的四套全池胜率超过 50%、综合胜率 64.27% 支持。Dragapult 低于 50%，说明通用能力仍受构筑和动作语义影响。正式报告包含当前 catalog 的 Team Rocket opponent，因为仓库正式评测必须使用 `--opponents all`；用户排除的是 Team Rocket candidate，本次五套 candidate 中没有它。

下一步可以选择 Dragapult 作为低基线提升分支，或选择 Marnie 作为高基线增益上限分支。该选择属于新的训练实验决策，不能在本次零训练 gate 中自动启动。
