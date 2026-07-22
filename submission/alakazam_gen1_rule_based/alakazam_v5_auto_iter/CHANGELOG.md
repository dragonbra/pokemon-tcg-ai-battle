# Alakazam v5 auto iter 迭代记录

本文件记录每一轮 opponent 池评测、失败复盘和策略改动。胜率以评测框架的
`wins / games` 计算；`yakitori_raging_bolt` 当前会偶发对手侧 `IndexError`，
这类局保留在总局数中，但单独标记为对手错误，不计为我方 agent error。

## v0：基线建立（2026-07-19）

### 评测协议

- 评测仓库：`/Users/hejinyu/Documents/repos/ptcg-agent-kaggle`
- 对手池：16 个注册 opponent
- 正式评测：每个对手 30 局，共 480 局；每隔一局交换先后手
- 引擎：`submission/alakazam_v5_auto_iter/cg/libcg.dylib`，通过
  `eval/alakazam_replay.py` 加载；结果不使用固定随机种子
- 结果：`docs/reports/kaggle/alakazam-v5-auto-iter/baseline-v0-full/summary.json`

### 成绩

| 指标 | 结果 |
|---|---:|
| 胜 / 负 / 未完成 | 249 / 222 / 9 |
| 平均胜率 | 51.9% |
| Meta 加权胜率 | 53.0% |
| 我方 agent error | 0 |
| 对手侧异常 | 9 局，均为 `yakitori_raging_bolt` 的 `IndexError` |

逐 opponent 结果：

| 对手 | 胜率 |
|---|---:|
| `romanrozen_v9` | 63.3% |
| `pilkwang_v2` | 53.3% |
| `kokinn_search` | 33.3% |
| `penguin_915` | 33.3% |
| `crustle_wall` | 80.0% |
| `crustle_v1` | 86.7% |
| `kiyotah_lucario` | 50.0% |
| `kiyotah_dragapult` | 20.0% |
| `kiyotah_iono` | 13.3% |
| `kiyotah_abomasnow` | 70.0% |
| `kacchan_anti_wall` | 40.0% |
| `nursrijan_lucario` | 46.7% |
| `yakitori_raging_bolt` | 53.3%（9 局对手异常） |
| `zoli_dragapult` | 53.3% |
| `sue_alakazam` | 70.0% |
| `maktha_1084` | 63.3% |

### 初步复盘

正式基线的首要风险是 `kiyotah_iono`、`kiyotah_dragapult`、
`kokinn_search` 和 `penguin_915`。对最弱对局额外保存了 5 个 opponent、每个 4 局
的 full trace，路径为：
`docs/reports/kaggle/alakazam-v5-auto-iter/baseline-v0-traces/`。

现阶段证据支持以下第一轮假设：连续攻击线不足是主要问题；仅有 Bench Pokémon
并不代表下一回合有可攻击的 Abra/Kadabra/Alakazam。Iono 对局尤其表现为牌库仍有
资源，但主动位被处理后没有带 Psychic Energy 的接力攻击者。第一轮优先验证
“提前保留并充能第二只攻击线”，暂不改牌表、Mist Energy 判断或抽牌上限。

### 代码变更

- 仅增加 `DECK = read_deck_csv()`，让 evaluator 能读取已有 60 张牌组；不改变
  决策逻辑。
- 未因本轮基线成绩提交策略改动；`v0` 作为后续回归基准。

### 可复现命令

```bash
cd /Users/hejinyu/Documents/repos/ptcg-agent-kaggle
PYTHONDONTWRITEBYTECODE=1 python3 eval/alakazam_replay.py \
  --agent /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/submission/alakazam_v5_auto_iter/main.py \
  --label alakazam_v5_auto_iter_baseline_v0_full \
  --games 30 \
  --output /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/reports/kaggle/alakazam-v5-auto-iter/baseline-v0-full \
  --cg-path /Users/hejinyu/Documents/repos/pokemon-tcg-ai-battle/submission/alakazam_v5_auto_iter
```

## v1：Poffin 连续攻击线保护（2026-07-19）

### 改动

- 新增 `_ready_attack_line_count()`。
- Poffin 在场上攻击线少于 3 只，或已充能的 Abra/Kadabra/Alakazam 少于 2 只时，
  继续优先搜索 Abra；只有攻击线和 ready attacker 都达到门槛后才转向 Dunsparce。
- 这是单变量策略改动，未修改牌表、能量判断、Mist 处理、抽牌上限或 Supporter gate。

### 完整评测

协议保持 v0：16 个对手 × 30 局、交替先后手、同一 `libcg.dylib`。结果文件：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-1-poffin-continuity/summary.json`。

| 指标 | v0 | v1 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 249 / 222 / 9 | 256 / 224 / 0 | +7 胜，未完成局减少 |
| 平均胜率 | 51.9% | 54.4% | +2.5pp |
| Meta 加权胜率 | 53.0% | 53.6% | +0.6pp |
| 我方 agent error | 0 | 0 | 无变化 |

逐 opponent 变化：

| 对手 | v0 | v1 | 变化 |
|---|---:|---:|---:|
| `romanrozen_v9` | 63.3% | 43.3% | -20.0pp |
| `pilkwang_v2` | 53.3% | 46.7% | -6.7pp |
| `kokinn_search` | 33.3% | 53.3% | +20.0pp |
| `penguin_915` | 33.3% | 53.3% | +20.0pp |
| `crustle_wall` | 80.0% | 83.3% | +3.3pp |
| `crustle_v1` | 86.7% | 83.3% | -3.3pp |
| `kiyotah_lucario` | 50.0% | 56.7% | +6.7pp |
| `kiyotah_dragapult` | 20.0% | 23.3% | +3.3pp |
| `kiyotah_iono` | 13.3% | 16.7% | +3.3pp |
| `kacchan_anti_wall` | 40.0% | 46.7% | +6.7pp |
| `nursrijan_lucario` | 46.7% | 50.0% | +3.3pp |
| `sue_alakazam` | 70.0% | 63.3% | -6.7pp |
| `maktha_1084` | 63.3% | 46.7% | -16.7pp |

`yakitori_raging_bolt` v1 为 22W/3L/5D，5 局仍是对手侧异常；`zoli_dragapult`
为 60.0%，`kiyotah_abomasnow` 为 70.0%。完整原始结果中保留全部对手记录。

### 复盘与准入判断

对 v1 的 5 个重点 opponent 另存了每个 4 局 full trace：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-1-poffin-continuity-traces/`。
Iono 仍是主要瓶颈，部分失败在第一次空 Active 时已有 1–2 只 ready attacker，
说明仅放宽 Poffin 目标不足以解决后续两次 KO 的连续接力。

单独使用 100 局直接 head-to-head 对比 `agent/alakazam_v5/main.py`：
v1 为 54W/46L，超过 52% 准入线。由于全矩阵平均和加权分数均上升，v1 暂时保留；
但 `romanrozen_v9`、`maktha_1084` 的单批回退超过 15pp，下一轮必须重点监控，若
定向复测确认则回退本改动。

## v2：Hilda 连续攻击线条件（已回退，2026-07-19）

### 假设与改动

在 Hilda 的第一段搜索中，只有 ready Abra 线达到 2 只时才允许转向
Dudunsparce；否则继续保留进化/能量线路。该改动建立在 v1 的 Poffin 保护之上，
目标是减少只有一只 ready Alakazam 时过早启动 Dunsparce engine。

### 评测结果

完整协议仍为 16 个对手 × 30 局。结果文件：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-2-hilda-continuity/summary.json`。

| 指标 | v1 | v2 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 261 / 214 / 5 | 258 / 217 / 5 | -3 胜 |
| 平均胜率 | 54.4% | 53.8% | -0.6pp |
| Meta 加权胜率 | 53.6% | 53.7% | +0.1pp |
| `kiyotah_iono` | 16.7% | 6.7% | -10.0pp |
| `kiyotah_abomasnow` | 70.0% | 56.7% | -13.3pp |
| `nursrijan_lucario` | 50.0% | 40.0% | -10.0pp |
| `kiyotah_dragapult` | 23.3% | 36.7% | +13.3pp |

### 结论与回退

该改动没有改善总体表现，且直接恶化了当前重点的 Iono 对局；加权分数仅上升
0.1pp 不足以抵消平均胜率和多个 matchup 的回退，判定为失败实验。已主动回退
Hilda 条件，当前代码恢复为 v1；v2 结果和证据保留用于后续避免重复尝试。

## v3：Hilda 能量搜索连续性（已回退，2026-07-19）

### 假设与改动

Hilda 第二段搜索只有在 ready attacker 至少 2 只时才优先拿 Enriching Energy，
否则优先选择 Telepath Energy / Basic Psychic Energy，目标是避免把下一只攻击线的
能量过早转成 Dudunsparce 抽牌资源。

### 评测结果

结果文件：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-3-hilda-energy-continuity/summary.json`。

| 指标 | v1 | v3 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 261 / 214 / 5 | 260 / 216 / 4 | -1 胜 |
| 平均胜率 | 54.4% | 54.2% | -0.2pp |
| Meta 加权胜率 | 53.6% | 55.6% | +2.0pp |
| `kacchan_anti_wall` | 46.7% | 10.0% | -36.7pp |
| `kiyotah_iono` | 16.7% | 6.7% | -10.0pp |
| `zoli_dragapult` | 60.0% | 36.7% | -23.3pp |
| `kiyotah_lucario` | 56.7% | 80.0% | +23.3pp |

### 结论与回退

该改动提升了加权分数，但牺牲了多个对局的稳定性，尤其是 Kacchan anti-wall，
并没有解决 Iono。由于目标是可靠的对手池整体表现，不接受以单一加权指标掩盖
关键 matchup 崩溃，已回退到 v1。后续不再单独用“ready attacker 数量”替换
Hilda 的 Enriching Energy 条件，除非有更细的对手状态证据。

## v4：低阶段攻击线撤退交接（已回退，2026-07-19）

### 假设与改动

允许 Active Fezandipiti/Shaymin 在 Bench 有带 Psychic Energy 的 Abra 或 Kadabra
时撤退，而不只接受 Bench Alakazam。目标是避免被击倒后出现空 Active。

### 评测结果

结果文件：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-4-retreat-handoff/summary.json`。

| 对手 | v1 | v4 | 变化 |
|---|---:|---:|---:|
| `kacchan_anti_wall` | 46.7% | 56.7% | +10.0pp |
| `kiyotah_dragapult` | 23.3% | 10.0% | -13.3pp |
| `pilkwang_v2` | 46.7% | 30.0% | -16.7pp |
| `romanrozen_v9` | 43.3% | 40.0% | -3.3pp |
| `nursrijan_lucario` | 50.0% | 33.3% | -16.7pp |

v4 的 `kiyotah_iono` 为 13.3%，没有解决主要瓶颈；总体结果也低于 v1。
低阶段攻击者虽然能避免空 Active，但会用低伤害攻击替代 Alakazam 的奖赏推进，
因此该改动判定为负收益并已回退。

## v5：Prize-aware 抽牌保护（已回退，2026-07-19）

### 假设与改动

将 `_draw_is_blocked()` 的固定牌库下限从 10 张改为
`max(10, 剩余 Prize + 1)`，希望减少长局 deck-out，并复用 Mixed/F1 中的资源保护
思路。

### 评测结果

结果文件：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-5-prize-reserve/summary.json`。

| 指标 | v1 | v5 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 261 / 214 / 5 | 257 / 217 / 6 | -4 胜 |
| 平均胜率 | 54.4% | 53.5% | -0.9pp |
| Meta 加权胜率 | 53.6% | 54.2% | +0.6pp |
| `kiyotah_iono` | 16.7% | 3.3% | -13.4pp |
| `kiyotah_dragapult` | 23.3% | 23.3% | 0pp |
| `kacchan_anti_wall` | 46.7% | 46.7% | 0pp |

### 结论与回退

当前失败主要不是 deck-out，而是 Iono 等对局中仍需抽牌寻找能量和进化线；
Reserve gate 反而提前阻断了这些动作。由于平均胜率下降且主要瓶颈恶化，已回退
到 v1。未来若重新引入牌库保护，必须先证明具体失败确实由 deck-out 导致。

## v6：Fez recovery（待后续回归确认，2026-07-19）

### 改动

- 上一回合己方 Pokémon 被击倒、当前没有任何带 Psychic Energy 的 Abra/Kadabra/
  Alakazam 攻击线时，允许 Fezandipiti ex 使用 `Flip the Script` 抽 3 张。
- 已有 ready attacker 时继续禁止暴露 Fez；已有直接 Alakazam 线路或当前已能击杀时也不抽。

### 完整评测

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-6-fez-recovery/summary.json`。

| 指标 | v1 | v6 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 261 / 214 / 5 | 274 / 201 / 5 | +13 胜 |
| 平均胜率 | 54.4% | 57.1% | +2.7pp |
| Meta 加权胜率 | 53.6% | 56.7% | +3.1pp |
| 我方 agent error | 0 | 0 | 无变化 |

重点对手：`kiyotah_iono` 13.3%、`pilkwang_v2` 33.3%、`penguin_915` 30.0%、
`kokinn_search` 40.0%、`kacchan_anti_wall` 70.0%。重点 full trace 保存在
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-6-fez-recovery-traces/`。

### 复盘与准入判断

小样本 trace 显示 Fez 并不能单独解决 Iono：有些失败已经有 1–2 个 ready attacker，
仍然输在连续攻击和奖赏竞速。v6 相对 v1 的整体提升暂时保留，但 100 局对旧
`agent/alakazam_v5` 的直接对照为 49W/51L，未达到显著优于旧策略的门槛，后续继续
用完整 opponent 矩阵验证，不能把 v6 的提升视为已证实的单因果收益。

## v7：无收益撤退 guard（保留，2026-07-19）

### 改动

- 无法把带 Psychic Energy 的 Bench Alakazam 交接到 Active 时，撤退选项的优先级
  明确低于 `END`；尤其修复首回合不能攻击时 Active Dunsparce/Fezandipiti 的
  无意义换位。
- 这是单变量改动，未改变牌表、Fez 条件、能量判断或抽牌门槛。

### 完整评测

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-7-retreat-guard/summary.json`。

| 指标 | v6 | v7 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 274 / 201 / 5 | 289 / 188 / 3 | +15 胜、异常局 -2 |
| 平均胜率 | 57.1% | 60.2% | +3.1pp |
| 我方 agent error | 0 | 0 | 无变化 |

重点对手变化：`kiyotah_iono` 13.3%→30.0%、`pilkwang_v2` 33.3%→53.3%、
`kokinn_search` 40.0%→66.7%、`penguin_915` 30.0%→36.7%；
`kacchan_anti_wall` 70.0%→63.3%，记录为需要后续监控的回退。100 局对旧
`agent/alakazam_v5` 为 50W/50L，未证明 direct head-to-head 优势，但完整矩阵
仍显著优于 v6，因此 v7 暂时保留。

## v8：Fez 抽 3 击杀例外（已回退，2026-07-19）

### 假设与改动

尝试把 `_draw_changes_knockout(..., 3)` 放到 Active Alakazam + 手牌 Psychic
Energy 的保护条件之前，允许 Fezandipiti 的三张牌恰好把 Powerful Hand 推到击杀线。
外层仍保留对手只剩两张 Prize 且已有 ready attacker 时不下 Fez 的保护。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-8-fez-lethal/summary.json`。

| 指标 | v7 | v8 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 289 / 188 / 3 | 269 / 209 / 2 | -20 胜 |
| 平均胜率 | 60.2% | 56.0% | -4.2pp |
| 我方 agent error | 0 | 0 | 无变化 |

`kiyotah_iono` 30.0%→13.3%、`kiyotah_dragapult` 30.0%→20.0%、
`kacchan_anti_wall` 63.3%→53.3%，说明局部击杀收益无法抵消暴露两奖 Fez 和节奏
损失。已主动回退到 v7；v8 结果保留用于避免重复引入该条件。

## v9：定向 Xerosic 压手（已回退，2026-07-19）

### 假设与改动

尝试识别 Bellibolt/Kilowattrel 和 Mega Lucario 场面：当对手手牌达到 6 张时，
提前使用 Xerosic's Machinations，而不是等待通用的 8 张手牌门槛；其余对手仍使用
原有 Alakazam/8 张手牌条件。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-9-xerosic-pressure/summary.json`。

| 指标 | v7 | v9 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 289 / 188 / 3 | 244 / 228 / 8 | -45 胜 |
| 平均胜率 | 60.2% | 50.8% | -9.4pp |
| 我方 agent error | 0 | 0 | 无变化 |

虽然 `pilkwang_v2` 由 53.3% 上升到 63.3%，但 `kiyotah_dragapult` 30.0%→10.0%、
`kacchan_anti_wall` 63.3%→36.7%、`maktha_1084` 66.7%→40.0%，说明仅凭场上
Pokémon ID 和手牌数量无法判断 Supporter 是否应让位给进化、抽牌或攻击准备。
已删除该定向条件并回退到 v7。

## v10：低牌库可攻击线 guard（已回退，2026-07-19）

### 假设与改动

当牌库不超过 12 张、当前 Active 已能攻击、当前攻击不是 KO 时，降低 Poffin、
Battle Cage、Night Stretcher、Wondrous Patch、Sacred Ash 和 Lana's Aid 的优先级，
避免继续做可选的牌库消耗；保留可能直接形成下一次进化攻击的搜索卡。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-10-deck-safety/summary.json`。

| 指标 | v7 | v10 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 289 / 188 / 3 | 282 / 195 / 3 | -7 胜 |
| 平均胜率 | 60.2% | 58.8% | -1.4pp |
| 我方 agent error | 0 | 0 | 无变化 |

`kiyotah_lucario` 50.0%→73.3%、`nursrijan_lucario` 50.0%→83.3%、
`sue_alakazam` 83.3%→80.0%（本批样本略升），但 `kiyotah_iono` 30.0%→20.0%、
`kiyotah_dragapult` 30.0%→13.3%、`kacchan_anti_wall` 63.3%→43.3%。
结论是“能攻击”不足以判断应该停止检索，尤其在高 HP/手牌重置对局中可能仍需
寻找下一条完整线路；已回退到 v7。

## v11：低牌库 + 大手牌 guard（已回退，2026-07-19）

### 假设与改动

将 v10 的条件收窄为：牌库不超过 12 张且手牌至少 20 张时，才限制 Poffin、
Battle Cage、恢复卡、Dawn、Hilda、Poké Pad 等会继续消耗牌库的动作；能直接完成
Alakazam 进化的可见线路放行。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-11-deck-hand-safety/summary.json`。

| 指标 | v7 | v11 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 289 / 188 / 3 | 271 / 204 / 5 | -18 胜 |
| 平均胜率 | 60.2% | 56.5% | -3.8pp |
| 我方 agent error | 0 | 0 | 无变化 |

该版本部分改善了 Dragapult，但 `kiyotah_iono` 30.0%→13.3%、
`kacchan_anti_wall` 63.3%→33.3%、`sue_alakazam` 83.3%→53.3%。结合 v10，结论是
牌库安全必须与明确的胜利/攻击计划绑定，不能只依据牌库和手牌数量；已回退到 v7。

## v12：第 4 张 Dunsparce 牌表实验（已回退，2026-07-19）

### 假设与改动

将 1 张 Battle Cage 替换为第 4 张 Dunsparce，希望提高开局第二只 Basic 和后备
攻击线的成功率；策略代码完全不变。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-12-fourth-dunsparce/summary.json`。

| 指标 | v7 | v12 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 289 / 188 / 3 | 262 / 213 / 5 | -27 胜 |
| 平均胜率 | 60.2% | 54.6% | -5.6pp |
| 我方 agent error | 0 | 0 | 无变化 |

`pilkwang_v2` 53.3%→53.3%、`kokinn_search` 66.7%→50.0%虽未完全崩溃，但
`kiyotah_dragapult` 30.0%→13.3%、`kiyotah_iono` 30.0%→20.0%、
`kacchan_anti_wall` 63.3%→36.7%、`sue_alakazam` 83.3%→50.0%，说明 Battle Cage
在当前策略中承担的资源/节奏价值高于增加一张 Dunsparce；已恢复原牌表并回退到 v7。

## v13：空 Bench 紧急后备优先（已回退，2026-07-19）

### 假设与改动

当 Bench 为空且仍有空间时，提高 Dawn/Poffin 建立 Basic 后备的优先级，并降低
非 direct Hilda，试图修复早期 Active 断档；已有 direct Alakazam/Hilda 线路时放行。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-13-emergency-setup/summary.json`。

| 指标 | v7 | v13 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 289 / 188 / 3 | 247 / 228 / 5 | -42 胜 |
| 平均胜率 | 60.2% | 51.5% | -8.8pp |
| 我方 agent error | 0 | 0 | 无变化 |

`pilkwang_v2` 53.3%→33.3%、`kokinn_search` 66.7%→33.3%、
`kacchan_anti_wall` 63.3%→33.3%，证明空 Bench 不是充分的紧急条件；绝对后备优先
会延迟本回合的 Alakazam 攻击。已回退到 v7。

## v14：完整 ex/mega ex 奖赏值（保留，2026-07-19）

### 改动

- 修正 `_prize_value()`：对手池中的普通 Pokémon ex 按 2 Prize、Mega Pokémon ex
  按 3 Prize 处理，而不是把未列入旧集合的 ex 误判为 1 Prize。
- 纳入 `Mega Lucario ex (678)`、`Mega Abomasnow ex (723)`、`Mega Kangaskhan ex
  (756)` 等 3 Prize 目标，以及 Bellibolt/Dragapult/Raging Bolt 等普通 ex。
- 该值同时影响 Boss's Orders 的奖赏交换判断和伤害目标排序；不改变牌表、能量或
  抽牌逻辑。

### 完整评测

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-14-prize-values/summary.json`。

| 指标 | v7 | v14 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 289 / 188 / 3 | 302 / 173 / 5 | +13 胜 |
| 平均胜率 | 60.2% | 62.9% | +2.7pp |
| 我方 agent error | 0 | 0 | 无变化 |

逐 opponent 变化较广：`romanrozen_v9` 50.0%→66.7%、`pilkwang_v2` 53.3%→63.3%、
`kokinn_search` 66.7%→63.3%、`penguin_915` 36.7%→63.3%、
`kiyotah_lucario` 50.0%→66.7%、`sue_alakazam` 83.3%→66.7%。
该改动符合官方卡牌数据中的 `ex/megaEx` 规则，且没有造成我方错误，暂时保留为新基线。
两批各 100 局对旧 `agent/alakazam_v5` 分别为 36W/64L 和 49W/51L，波动较大，
因此不把 direct H2H 当作 v14 的单独胜因；后续仍以固定 16×30 opponent 矩阵为主。

## v15：第 3 张 Basic Psychic 牌表实验（已回退，2026-07-19）

### 假设与改动

将 1 张 Battle Cage 替换为第 3 张 Basic Psychic Energy，测试提高能量密度能否
改善低胜率对局中的 ready attacker 连续性；策略代码保持 v14 不变。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-15-extra-psychic/summary.json`。

| 指标 | v14 | v15 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 302 / 173 / 5 | 287 / 187 / 6 | -15 胜 |
| 平均胜率 | 62.9% | 59.8% | -3.1pp |
| 我方 agent error | 0 | 0 | 无变化 |

`kacchan_anti_wall` 50.0%→70.0%、`maktha_1084` 56.7%→80.0%有所改善，但
`kiyotah_abomasnow` 66.7%→50.0%、`crustle_v1` 96.7%→83.3%、
`penguin_915` 63.3%→46.7%，说明 Battle Cage 的场景价值不能用单张 Basic Energy
替代；已恢复 v14 牌表。

## v16：主行动 Enriching Energy 过量贴附 guard（已回退，2026-07-19）

### 假设与改动

重点 trace 发现，主行动的 `Enriching Energy` 分支没有复用 Abra 线“每只攻击者
优先只保留一张 Psychic Energy”的判断。在 Active Abra/Kadabra 已有 Psychic
Energy、Bench 仍有未充能攻击线时，选项顺序可能导致再次给 Active 贴能量。v16
尝试在该分支中优先未充能的 Abra/Kadabra/Alakazam，同时保留已有非 Psychic
能量时补 Psychic 的可能性；其他能量类型、牌表和抽牌逻辑不变。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-16-enriching-guard/summary.json`。

| 指标 | v14 | v16 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 302 / 173 / 5 | 298 / 178 / 4 | -4 胜 |
| 平均胜率 | 62.9% | 62.1% | -0.8pp |
| 我方 agent error | 0 | 0 | 无变化 |

局部变化包括 `kacchan_anti_wall` 50.0%→60.0%、`kiyotah_lucario` 66.7%→80.0%，
但 `kiyotah_iono` 26.7%→13.3%、`penguin_915` 63.3%→40.0%、
`kiyotah_abomasnow` 66.7%→56.7%。这说明“避免重复贴能量”在部分局面是正确的，
但把 Enriching Energy 从当前 Active 的即时攻击线移走，会牺牲本回合攻击或把
需要特殊能量状态判断的场面误当成可替代的后备建设。总体胜率下降，已回退到 v14。

## v17：低动点手牌的未击倒 Fez 抽牌例外（已回退，2026-07-19）

### 假设与改动

已有人工复盘指出，Active Kadabra 只能造成 30 点伤害、手牌主要是 Xerosic、
Enhanced Hammer 和 Fezandipiti 时，直接攻击的价值很低，应该尝试 Bench
Fezandipiti ex 的 `Flip the Script`。v17 增加了一个窄条件：Active 是 Kadabra、
Bench 有 Fez、对手仍有超过两张 Prize、没有直接 Alakazam 路线，且可见手牌不超过
5 张并全部属于低动点卡/能量时，允许不等待上一回合被击倒而抽 3 张。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-17-fez-low-hand/summary.json`。

| 指标 | v14 | v17 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 302 / 173 / 5 | 301 / 175 / 4 | -1 胜 |
| 平均胜率 | 62.9% | 62.7% | -0.2pp |
| 我方 agent error | 0 | 0 | 无变化 |

重点变化：`kiyotah_iono` 26.7%→6.7%、`kiyotah_abomasnow` 66.7%→50.0%、
`nursrijan_lucario` 56.7%→43.3%，只有 `pilkwang_v2` 53.3%→80.0% 和
`kacchan_anti_wall` 50.0%→63.3% 等局部提升。结论是即便手牌动点低，未被击倒
时放弃 Kadabra 的稳定攻击并暴露两奖 Fez 仍然过于激进；该例外已回退，v14
规则保持不变。

## v18：效果型能量按 Psychic 类型判断（暂时保留，2026-07-19）

### 假设与改动

`_choose_energy_option()` 原先只要发现 Abra/Kadabra/Alakazam 已有任意一张能量，
就把它视为“已满足一张能量上限”。但 Enriching Energy 只提供无色，不能替代
攻击所需的 Psychic Energy；在 Wondrous Patch 等效果选择中，这会阻止 Basic
Psychic 补到只有 Enriching 的攻击线。v18 将该 guard 改为只在目标已有 Psychic
Energy 时触发。主行动的能量选择、牌表和其他策略保持不变。

### 完整评测

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-18-effect-energy-type/summary.json`。

| 指标 | v14 | v18 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 302 / 173 / 5 | 305 / 168 / 7 | +3 胜 |
| 平均胜率 | 62.9% | 63.5% | +0.6pp |
| 我方 agent error | 0 | 0 | 无变化 |

逐对手重点变化：`kiyotah_lucario` 66.7%→86.7%、`nursrijan_lucario` 56.7%→70.0%、
`kiyotah_abomasnow` 66.7%→76.7%，但 `kiyotah_dragapult` 30.0%→20.0%、
`crustle_v1` 96.7%→86.7%。由于整体胜率上升且改动符合能量类型规则，暂时保留
为新基线；`iter-18-focus-traces/` 保存了四个低胜率对手的补充 trace，后续优先
确认 Dragapult 的回退是否与该 guard 有因果关系。

## v19：Dragapult 场面提前使用 Battle Cage（已回退，2026-07-19）

### 假设与改动

官方卡牌文本确认 Battle Cage 可以防止对手攻击/特性把伤害指示物放到 Bench，
理论上能够针对 Dragapult ex 的 Phantom Dive。v19 在对手场上出现 Dreepy、
Drakloak 或 Dragapult ex 时，把 Battle Cage 的主行动优先级提高到通用 setup 之上，
希望在扩散伤害前保护后备攻击线。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-19-dragapult-cage/summary.json`。

| 指标 | v18 | v19 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 305 / 168 / 7 | 299 / 179 / 2 | -6 胜 |
| 平均胜率 | 63.5% | 62.3% | -1.3pp |
| 我方 agent error | 0 | 0 | 无变化 |

`kiyotah_dragapult` 仍为 20.0%，而 `kiyotah_iono` 由 23.3%降至 3.3%、
`kiyotah_abomasnow` 由 76.7%降至 53.3%、`kokinn_search` 由 70.0%降至 40.0%。
结论是 Battle Cage 不能阻止 Active 的主攻击，过早抢下它会延迟进化、能量和抽牌；
该 matchup-specific 优先级已回退，v18 继续作为基线。

## v20：已有攻击线后的 Dragapult Battle Cage 优先级（已回退，2026-07-19）

### 假设与改动

为避免 v19 抢占开局 setup，v20 将条件收窄为：我方已达到 3 只 Abra 线、当前有
可攻击或直接进化攻击路径，且对手场上出现 Dreepy/Drakloak/Dragapult ex 时，才把
Battle Cage 提前到通用行动之前；开局没有攻击线时不触发。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-20-ready-cage/summary.json`。

| 指标 | v18 | v20 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 305 / 168 / 7 | 283 / 194 / 3 | -22 胜 |
| 平均胜率 | 63.5% | 59.0% | -4.6pp |
| 我方 agent error | 0 | 0 | 无变化 |

`kiyotah_dragapult` 20.0%→16.7%，`kiyotah_abomasnow` 76.7%→50.0%、
`nursrijan_lucario` 70.0%→40.0%，说明即使已有攻击线，Cage 优先级仍会抢占
Dudunsparce 抽牌和后续攻击节奏；该条件已回退，v18 保持为最佳基线。

## v21：Dragapult Active 不可击倒时允许 Boss 低奖赏后备（已回退，2026-07-19）

### 假设与改动

v18 的 Dragapult trace 显示，对手 Active Dragapult ex（320 HP）常常无法被当前
手牌一击击倒，但 Bench 的 Dreepy/Drakloak 可以被击倒。原有 Boss's Orders 判断
要求后备目标的 Prize 价值不低于 Active，因此会放弃本回合拿 1 Prize 的机会。v21
尝试仅对 Dragapult ex 放宽为：Active 不能击倒、Bench 存在可击倒目标时允许 Boss。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-21-dragapult-boss/summary.json`。

| 指标 | v18 | v21 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 305 / 168 / 7 | 293 / 180 / 7 | -12 胜 |
| 平均胜率 | 63.5% | 61.0% | -2.5pp |
| 我方 agent error | 0 | 0 | 无变化 |
| `kiyotah_dragapult` | 20.0% | 46.7% | +26.7pp |

虽然 Dragapult 显著改善，但 `kiyotah_lucario` 86.7%→53.3%、`kokinn_search`
70.0%→40.0% 等整体回退，证明单一对手的早期低奖赏交换会污染通用 Boss 资源。
该改动已回退，v18 继续作为整体基线；未来若重试需增加剩余 Prize 和 Boss 资源
条件，而不能只按 Active Pokémon ID 放宽。

## v22：Enhanced Hammer gate 接线校验（无效 control run，2026-07-19）

### 实际状态

本轮原计划测试“只有移除 Mist Energy 后能直接击倒时，才提高 Enhanced Hammer
优先级”。复核评测前后的 `main.py` 发现，v22 只新增了 `_hammer_changes_attack()`
辅助函数，主行动中的优先级仍然调用 v18 的 `active_special_energy` 判断，辅助函数
没有被调用。因此该轮并未实际改变策略，不能把结果归因于 Hammer gate。

### Control run

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-22-hammer-gate/summary.json`。

| 指标 | v18 | v22 control |
|---|---:|---:|
| 胜 / 负 / 未完成 | 305 / 168 / 7 | 265 / 208 / 7 |
| 平均胜率 | 63.5% | 55.2% |
| 我方 agent error | 0 | 0 |

该结果与 v18 的随机洗牌差异较大，但由于代码等价，不能作为策略升降结论；本轮
仅作为评测完整性和结果留档，v18 仍是已验证基线。

## v23：实际接入 Enhanced Hammer gate（已回退，2026-07-19）

### 假设与改动

本轮将 v22 中未被调用的 `_hammer_changes_attack()` 真正接入 Enhanced Hammer
主行动优先级，尝试仅在该辅助判断成立时保留较高优先级，其他情况下让 Hammer
让位于 setup 行动。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-23-hammer-gate/summary.json`；
补充 trace：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-23-focus-traces/`。

| 指标 | v18 | v23 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 305 / 168 / 7 | 273 / 203 / 4 | -32 胜 |
| 平均胜率 | 63.5% | 56.9% | -6.7pp |
| 我方 agent error | 0 | 0 | 无变化 |

`kiyotah_iono` 为 4/30、`kiyotah_dragapult` 为 6/30、`pilkwang_v2` 为 13/30；
补充 trace 中这些失败局仍然在被击倒后只有 0–1 只 ready attacker。复盘代码后
还发现该 gate 的语义不准确：`mist_blocks_attack` 本身已经先于 gate 将 Mist
Energy 场面设为最高 Hammer 优先级，而 `_hammer_changes_attack()` 只识别 Mist，
所以实际改动主要是压低“非 Mist 特殊能量”场面的 Hammer 优先级，并没有实现
预期的“判断当前攻击收益”。该设计既未解决连续攻击问题，又损失了原有资源干预
时机，已回退到 v18 的 `active_special_energy` 判断。

## v24：仅对攻击相关特殊能量提高 Hammer 优先级（已回退，2026-07-19）

### 假设与改动

v18 对手场上任意特殊能量都可能触发 Enhanced Hammer 高优先级。根据卡牌文本，
v24 将该条件收窄为只识别 Mist Energy 与 Telepath Psychic Energy；Enriching
Energy 仅提供无色，预计不应与对手攻击能量同等处理。Bench 的特殊能量判断同步
采用相同类型过滤；牌表、己方能量与攻击逻辑不变。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-24-attack-relevant-hammer/summary.json`。

| 指标 | v18 | v24 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 305 / 168 / 7 | 280 / 193 / 7 | -25 胜 |
| 平均胜率 | 63.5% | 58.3% | -5.2pp |
| 我方 agent error | 0 | 0 | 无变化 |

局部上升包括 `pilkwang_v2` 60.0%→70.0%、`kacchan_anti_wall` 50.0%→56.7%，
但 `kiyotah_dragapult` 20.0%→16.7%、`kiyotah_iono` 23.3%→16.7%、
`nursrijan_lucario` 70.0%→43.3%。整体下降且目标瓶颈未改善，已回退到 v18 的
通用特殊能量判断。

## v25：Active Kadabra 交接给 ready Alakazam（已回退，2026-07-19）

### 假设与改动

当 Active Kadabra 已带 Psychic Energy、Bench 有带 Psychic Energy 的 Alakazam、
手牌至少 3 张，且 Kadabra 当前攻击不能击倒对手时，v25 允许把撤退交接评分提高，
让更强的 Powerful Hand 接管攻击。当前攻击若能 KO，则仍保持立即攻击；其他
Active、牌表和攻击目标逻辑不变。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-25-kadabra-handoff/summary.json`；
补充 trace：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-25-focus-traces/`。

| 指标 | v18 | v25 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 305 / 168 / 7 | 282 / 193 / 5 | -23 胜 |
| 平均胜率 | 63.5% | 58.8% | -4.8pp |
| 我方 agent error | 0 | 0 | 无变化 |

补充 trace 证明该分支确实能触发，但在 `kokinn_search` 等对局中交接会消耗当前
Kadabra 的资源并改变原本的奖赏节奏；目标瓶颈 `kiyotah_iono` 为 7/30、
`kiyotah_dragapult` 为 7/30，未得到连续攻击改善。整体下降，已回退到 v18 的
“只有 Active 无法攻击时才撤退”规则。

## v26：不值得使用的 Xerosic 降到 END 之后（已回退，2026-07-19）

### 假设与改动

Iono 的 30 局 v18 diagnostic 中，Xerosic 使用 11 次且全部来自败局；其中部分
对手手牌只有 4–7 张。代码虽然通过 `_xerosic_is_worth_playing()` 判断“不值得”，
但 fallback 分数 35 仍高于 `END` 的 99，导致只有 Xerosic 可见时依旧会消耗
Supporter。v26 将该 fallback 改到 END 之后，保留值得压手时的优先级。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-26-xerosic-guard/summary.json`；
补充 trace：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-26-focus-traces/`。

| 指标 | v18 | v26 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 305 / 168 / 7 | 280 / 198 / 2 | -25 胜 |
| 平均胜率 | 63.5% | 58.3% | -5.2pp |
| 我方 agent error | 0 | 0 |

Iono 仍为 4/30、Dragapult 仅 2/30，短 trace 也未显示连续攻击改善。该相关性
不足以支持改变 Supporter fallback，已回退到 v18。

## v27：修正 Active Abra/Kadabra 的伪 ready 判断（暂保留，2026-07-19）

### 假设与改动

subagent 复核官方引擎发现，Active Abra/Kadabra 当前使用通用
`_energy_count() >= ATTACK_ENERGY_COUNT`，会把只有 Enriching Energy 的场面误判为
可攻击；但两者攻击成本都是 Psychic。v27 只将 Abra/Kadabra 与 Alakazam 的
ready 判断统一为 `_has_psychic_energy(active)`，Dunsparce 线仍使用通用能量数量。
该改动不改变牌表、搜索顺序或攻击选项，只修正策略层状态估计。

### 完整评测与 control 对比

候选结果：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-27-psychic-ready/summary.json`；
focus trace：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-27-focus-traces/`；同协议的
v18 control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-27-control/summary.json`。

| 指标 | v27 candidate | v27 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 291 / 186 / 3 | 287 / 185 / 8 | +4 胜、少 5 局未完成 |
| 胜率（480 局分母） | 60.6% | 59.8% | +0.8pp |
| 有效胜率（排除未完成） | 61.0% | 60.8% | +0.2pp |
| 我方 agent error | 0 | 0 | 无变化 |

单批仍低于历史 v18 的 63.5%，但同批 control 下没有出现总体下降；该改动还
消除了官方攻击能量规则上的伪 ready 判断，因此暂时保留为新的代码基线，尚未
宣称胜率提升。`kiyotah_iono` candidate/control 为 2/30 与 4/30，未解决该瓶颈；
下一轮必须在 v27 control 之上继续验证，并避免把本轮的 +0.8pp 当作显著收益。

## v28：Poffin 主行动复用 ready attacker gate（已回退，2026-07-19）

### 假设与改动

Poffin 的效果选择已经在 ready attacker 少于 2 只时优先 Abra，但主行动评分仍只
检查场上 Abra 线数量是否达到 3。v28 将主行动条件改为：场上攻击线未满 3 只，
或 ready Abra/Kadabra/Alakazam 少于 2 只且仍有 Bench 空间时，优先 Poffin；不改变
Poffin 的卡牌选择、牌表、能量和 Hilda 逻辑。

### 完整评测与同协议 control

候选结果：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-28-poffin-ready-gate/summary.json`；
control 结果：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-28-control/summary.json`。

| 指标 | v28 candidate | v28 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 302 / 171 / 7 | 295 / 182 / 3 | +7 胜 |
| 胜率（480 局分母） | 62.9% | 61.5% | +1.5pp |
| 有效胜率（排除未完成） | 63.8% | 61.8% | +2.0pp |
| 我方 agent error | 0 | 0 | 无变化 |

候选在 `pilkwang_v2`、`penguin_915` 等对局提升，但 `kiyotah_dragapult` 仍为
4/30，`kiyotah_iono` 为 5/30；首批结果暂作为候选保留，等待独立重复验证。

## v30：v28 策略独立重复验证（回退到 v27，2026-07-19）

### 完整评测

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-30-candidate/summary.json`；
同协议 control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-30-control/summary.json`。

| 指标 | v30 candidate（v28） | v30 control（v27） | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 291 / 187 / 2 | 292 / 185 / 3 | -1 胜 |
| 胜率（480 局分母） | 60.6% | 60.8% | -0.2pp |
| 有效胜率（排除未完成） | 60.9% | 61.2% | -0.3pp |
| 我方 agent error | 0 | 0 | 无变化 |

v28 首批相对 control 为 +7 胜，但独立重复相对 control 为 -1 胜，方向不稳定；
Poffin 主行动 gate 没有稳定改善关键 Dragapult/Iono 对局，已回退。当前代码基线
为 v27 的 Psychic ready 修正，v28 的 focus trace 仍保留作过程证据。

## v29：ready 数为 0 时恢复 Abra 优先（评测中）

### 假设与改动

当前 Night Stretcher/Lana’s Aid 只有在场上完全没有 Abra 时才把 Abra 放到最高
恢复优先级；若场上已有未充能 Abra，恢复选择会偏向 Kadabra/Alakazam，即使场上
没有任何 ready attacker。v29 增加一个窄条件：当 `_ready_attack_line_count()` 为
0 时，恢复卡优先取 Abra；已有 ready attacker 时完全保持 v28 行为。

### 完整评测与回退

结果文件：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-29-recovery-abra/summary.json`。

| 指标 | v28 | v29 | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 302 / 171 / 7 | 285 / 191 / 4 | -17 胜 |
| 平均胜率 | 62.9% | 59.4% | -3.5pp |
| 我方 agent error | 0 | 0 | 无变化 |

`kiyotah_dragapult` 降至 1/30，说明 ready=0 时无条件偏向回收 Abra 会抢占
直接回收 Alakazam/Kadabra 的路线；该条件已回退，v28 保持为当前基线。

## v31：低牌库 Dudunsparce 抽牌 guard 修正（暂保留，2026-07-19）

### 假设与改动

当牌库只剩 `1–3` 张且本回合抽牌不能直接形成击杀时，策略注释要求跳过
Dudunsparce 的 Run Away Draw；但原分数 34 仍高于 `END` 的 99，实际仍会选择
该能力。v31 只把这一 guard 的 fallback 改到 END 之后；正常牌库和能直接改变
击杀线的抽牌保持不变。

### 完整评测与同协议 control

候选结果：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-31-dudunsparce-guard/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-31-control/summary.json`。

| 指标 | v31 candidate | v31 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 298 / 175 / 7 | 294 / 182 / 4 | +4 胜 |
| 胜率（480 局分母） | 62.1% | 61.3% | +0.8pp |
| 有效胜率（排除未完成） | 63.0% | 61.8% | +1.2pp |
| 我方 agent error | 0 | 0 | 无变化 |

candidate 的 `kiyotah_dragapult` 为 10/30、`kiyotah_iono` 为 7/30，较 control
分别 +2 与 -2；`sue_alakazam` 为 25/30，整体 +4 胜但关键瓶颈并未同步改善。
由于 guard 符合低牌库抽牌会把自己抽空的规则且同批 control 非劣，暂保留；后续
需要独立重复验证，不能把 +0.8pp 当作稳定收益。

## v32：不值得的 Boss 降到 END 之后（暂保留，2026-07-19）

### 完整评测与同协议 control

候选结果：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-32-boss-guard/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-32-control/summary.json`。

| 指标 | v32 candidate | v32 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 304 / 174 / 2 | 277 / 201 / 5 | +27 胜 |
| 胜率（480 局分母） | 63.3% | 57.7% | +5.6pp |
| 有效胜率（排除未完成） | 63.6% | 58.0% | +5.6pp |
| 我方 agent error | 0 | 0 | 无变化 |

候选在 `pilkwang_v2`（16/30→12/30）、`penguin_915`（20/30→13/30）、
`kacchan_anti_wall`（18/30→12/30）等多个对手显著优于 control；
`kiyotah_dragapult` 为 8/30、`kiyotah_iono` 为 7/30，未解决首要瓶颈但没有
恶化。由于该 guard 与“无奖赏收益时不应消耗 Boss”一致，且同批差异较大，暂保留；
下一轮先做独立重复验证，再叠加新变量。

## v33：v32 Boss guard 独立重复验证（已完成，2026-07-19）

本轮不新增策略，重复 v32 candidate，并随后运行去掉 Boss fallback guard 的同协议
control，以确认 v32 的 +27 胜是否只是随机批次差异。

### 独立重复结果

候选：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-33-candidate/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-33-control/summary.json`。

| 指标 | v33 candidate | v33 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 303 / 174 / 3 | 300 / 179 / 1 | +3 胜 |
| 胜率（480 局分母） | 63.1% | 62.5% | +0.6pp |
| 有效胜率（排除未完成） | 63.5% | 62.6% | +0.9pp |
| 我方 agent error | 0 | 0 | 无变化 |

两次 candidate 都高于同批 control（+27、+3 胜），虽然提升幅度受洗牌影响较大，
但方向一致；v32 Boss fallback guard 继续保留。它尚未改善到 90%，当前最佳已
验证批次仍约 63%，后续应继续以 v32 candidate 为基线做新的单变量实验。

## v34：不值得的 Xerosic 降到 END 之后（暂保留，2026-07-19）

### 假设与改动

沿用 v32 的 Boss fallback guard，只把 Xerosic 在“不值得压手”时的 fallback 从 35
改为 100，使其低于 END 的优先级；真正值得使用的 Xerosic 分支保持 6。v26 曾在
不同批次中测试过相同方向但没有 control，本轮用 candidate/control 做独立归因。

### 完整评测与同协议 control

候选结果：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-34-xerosic-guard/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-34-control/summary.json`。

| 指标 | v34 candidate | v34 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 295 / 183 / 2 | 281 / 196 / 3 | +14 胜 |
| 胜率（480 局分母） | 61.5% | 58.5% | +2.9pp |
| 有效胜率（排除未完成） | 61.7% | 58.9% | +2.8pp |
| 我方 agent error | 0 | 0 | 无变化 |

candidate 在 `kiyotah_lucario`、`kacchan_anti_wall` 等对手提升，但 `kiyotah_iono`
为 2/30、较 control 下降 3 局；该局部回退尚未重复验证。整体 +14 胜且代码语义
符合 Supporter 保留原则，暂保留 Xerosic guard，但下一轮需独立重复并重点监控
Iono/Dragapult，不能把单批 +2.9pp 视为最终结论。

## v35：Boss + Xerosic 两个 fallback guard 组合验证（评测中）

当前 candidate 同时包含 v32 Boss guard 与 v34 Xerosic guard。本轮将运行组合
candidate，并以同时恢复两个旧 fallback 分数的版本作为 control，检查两个 guard
叠加后是否仍保持总体收益。

### 完整评测与回退

candidate：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-35-candidate/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-35-control/summary.json`。

| 指标 | v35 candidate（双 guard） | v35 control（双回退） | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 276 / 202 / 2 | 304 / 172 / 4 | -28 胜 |
| 胜率（480 局分母） | 57.5% | 63.3% | -5.8pp |
| 有效胜率（排除未完成） | 57.7% | 63.9% | -6.2pp |
| 我方 agent error | 0 | 0 | 无变化 |

双 guard candidate 明显低于 control，无法把 v34 Xerosic guard 与 v32 Boss guard
同时保留；Xerosic guard 已回退，Boss guard 恢复为 v32 的已验证版本。v34 的单批
收益保留为研究证据，不再作为当前代码基线。

## v36：blocked draw ability 降到 END 之后（评测中）

### 假设与改动

当 `_draw_is_blocked()` 为真且抽牌不能直接形成击杀时，Dudunsparce/Kadabra/
Alakazam 的抽牌能力原 fallback 分数为 55，仍高于 `END` 的 99。v36 将该 fallback
改为 100，避免在牌库过低、手牌过多或攻击已经安全时继续抽牌；直接改变击杀线的
抽牌分支不变。候选只改这一处，不叠加 Enriching 或 lethal 预测实验。

### 完整评测与回退

candidate：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-36-blocked-draw-guard/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-36-control/summary.json`。

| 指标 | v36 candidate | v36 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 277 / 196 / 7 | 291 / 182 / 7 | -14 胜 |
| 胜率（480 局分母） | 57.7% | 60.6% | -2.9pp |
| 有效胜率（排除未完成） | 58.6% | 61.5% | -2.9pp |
| 我方 agent error | 0 | 0 | 无变化 |

candidate 的 Dragapult 为 4/30、Iono 为 6/30，均未改善；广泛阻断抽牌会损失
必要的恢复和找线节奏，已回退到 v32 的 `55` fallback。低牌库 Dudunsparce 的
独立 `1–3` 张 guard 保留不变。

## v37：禁止 Enriching Energy 伪装成攻击线能量（评测中）

### 假设与改动

官方卡牌文本规定 Enriching Energy 只提供无色，不能满足 Abra/Kadabra/Alakazam
的 Psychic 攻击成本。当前主行动在目标是未带 Psychic 的攻击线时仍给 Enriching
分数 22，可能在没有更好选项时选择无效贴附。v37 将这种目标的 fallback 降到
END 之后；Enriching→Dudunsparce 的安全抽牌分支和所有 Psychic Energy 选择不变。

### 完整评测与同协议 control

候选结果：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-37-enriching-guard/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-37-control/summary.json`。

| 指标 | v37 candidate | v37 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 293 / 185 / 2 | 291 / 183 / 6 | +2 胜，未完成 -4 |
| 胜率（480 局分母） | 61.0% | 60.6% | +0.4pp |
| 有效胜率（排除未完成） | 61.4% | 61.3% | +0.1pp |
| 我方 agent error | 0 | 0 | 无变化 |

6 局 control 未完成、候选 2 局未完成均为 `yakitori_raging_bolt` 对手侧
`IndexError: -1`，不是我方 agent 崩溃。逐 opponent 方面，candidate 相对 control
在 `kiyotah_dragapult` 为 8/30 对 4/30、`kiyotah_iono` 为 6/30 对 3/30、
`kacchan_anti_wall` 为 20/30 对 13/30；`sue_alakazam` 和 `maktha_1084`
各少 5 胜，说明总体收益仍受洗牌波动影响，不能把 +2 胜解释为稳定的大幅提升。

### Trace 复盘与准入判断

另外保存了 5 个重点对手各 4 局 full trace：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-37-focus-traces/`。
失败样本中 14 局有 10 局最多只有 1 只带 Psychic Energy 的 Abra 线攻击者，
第一只 Active 被击倒时有 11 局 ready 数为 0；这与“场上有 Abra/Kadabra 不等于
下一回合能攻击”的诊断一致。v37 的改动没有引入非法动作，也没有破坏安全的
`Enriching Energy -> Dudunsparce` 抽牌路径；其在三个主要瓶颈对手上的方向改善，
且与官方卡牌文本一致，因此保留 v37 guard，下一轮继续只测试一个新的连续攻击线
变量，不叠加 Xerosic 或 broad draw guard。

## v38：Hilda 只在当前攻击安全时转向引擎（已回退，2026-07-19）

### 假设与改动

候选只在 Hilda 的两个决策点使用更窄的 `_hilda_attack_security()`：Active 当前可
攻击、Active 存在直通 Alakazam 的合法路线，或 Dunsparce 可以用 Trading Places
交接给带 Psychic Energy 的 Alakazam 时，才把攻击线视为安全。该轮没有改变 Poffin、
能量选择、抽牌 guard 或牌表。

### 完整评测与同协议 control

候选：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-38-hilda-guard/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-38-control/summary.json`。

| 指标 | v38 candidate | v38 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 286 / 191 / 3 | 282 / 191 / 7 | +4 胜，未完成 -4 |
| 胜率（480 局分母） | 59.6% | 58.8% | +0.8pp |
| 有效胜率（排除未完成） | 60.0% | 59.6% | +0.4pp |
| 我方 agent error | 0 | 0 | 无变化 |

候选虽然在同批 control 上多 4 胜，但 `kiyotah_iono` 为 4/30、低于 control 的
6/30，`kiyotah_dragapult` 与 control 同为 10/30；`romanrozen_v9`、
`sue_alakazam` 各少 3/2 胜。该变化没有改善首要瓶颈，整体差异不足以抵消强势
对局回退，判定为随机批次下的非稳健收益。

### Trace 复盘与回退

候选重点 trace 位于
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-38-focus-traces/`。16 局定向 trace 中
13 局失败；失败样本仍反复出现 Active 被击倒时 ready attacker 为 0 或最多只有
1 只，说明 Hilda 安全判定不是主要瓶颈。v38 已主动回退，当前代码保持 v37
Enriching guard 与原 `_attack_security()`。

## v39：按 attackId 修正 Dunsparce 伤害估算（评测中，2026-07-19）

### 假设与改动

官方引擎中 Dunsparce 的 `Trading Places`（attack ID `423`）不造成伤害，只把 Active
换到 Bench；普通 `Ram`（attack ID `424`）才造成 20 点伤害。旧策略只按 Pokémon ID
估算，把所有 Dunsparce 攻击都当成 20 点，可能在 Trading Places 可选时误判击杀、
抽牌门槛和 Boss prize race。v39 为当前 legal attack options 增加按 `attackId`
计算的 damage helper，并将主行动的击杀、抽牌 guard 与 Boss 判断改为使用该 helper；
其余策略不变。

### 完整评测与同协议 control

候选：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-39-attack-id/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-38-control/summary.json`。

| 指标 | v39 candidate | v39 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 282 / 196 / 2 | 282 / 191 / 7 | 胜场持平，未完成 -5 |
| 胜率（480 局分母） | 58.8% | 58.8% | 0.0pp |
| 有效胜率（排除未完成） | 59.0% | 59.6% | -0.6pp |
| 我方 agent error | 0 | 0 | 无变化 |

候选在 `kiyotah_dragapult` 为 4/30、`kiyotah_iono` 为 3/30，均未改善主要瓶颈；
定向 trace `docs/reports/kaggle/alakazam-v5-auto-iter/iter-39-focus-traces/` 中未发现
非法动作，Trading Places 的 attack ID 分支也按预期工作，但本批没有转化为胜场。
该规则修正暂不纳入当前策略，已回退到 v37 状态；保留 v39 报告作为后续在更大
样本或出现 Dunsparce 误判样本时重测的依据。

## v40：post-KO 的可验证恢复路线优先（已回退，2026-07-19）

### 假设与改动

候选在 `_fezandipiti_needed()` 中增加 guard：上一回合被击倒后，如果弃牌区有
Abra/Kadabra/Alakazam 且手牌有 `Night Stretcher` 或 `Lana's Aid`，则不优先放下
两奖的 Fezandipiti ex，改走可见的恢复路线；没有同时满足这两个条件时保持原有
Fez 抽牌逻辑。

### 完整评测与 trace

第一次目录 `iter-40-recovery-guard/` 因回退 v39 时遗漏 Boss helper 的局部变量，
出现 384 局 `NameError: target is not defined`，判定为无效运行，不纳入统计。
修复后有效候选结果为：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-40-recovery-guard-rerun/summary.json`；
对照沿用同协议 `iter-38-control/summary.json`。

| 指标 | v40 candidate | v40 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 285 / 192 / 3 | 282 / 191 / 7 | +3 胜，未完成 -4 |
| 胜率（480 局分母） | 59.4% | 58.8% | +0.6pp |
| 有效胜率（排除未完成） | 59.8% | 59.6% | +0.2pp |
| 我方 agent error | 0 | 0 | 无变化 |

重点 trace：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-40-focus-traces/`。定向失败
仍主要表现为第一次 Active 被击倒时 ready attacker 为 0 或只有 1 只；
`kiyotah_dragapult` 为 8/30、`kiyotah_iono` 为 5/30，均低于 control 的 10/30、
6/30。候选的 +3 胜不足以证明恢复 guard 稳健，且若干强势对手回退，已主动回退
v40，当前代码恢复为 v37 基线。

## v41：放宽牌库 2–3 张时的 Dudunsparce 抽牌（暂保留，2026-07-19）

### 假设与改动

官方引擎的 `Run Away Draw` 会先抽出牌库中剩余的牌，再把 Dudunsparce 和所附卡
洗回牌库。因此牌库剩 2–3 张并不会因为抽 3 张而直接牌库耗尽；v31 的
`deck_count <= 3` guard 过度保守。v41 将它收窄为只在 `deck_count <= 1` 时跳过，
不改变手牌过大、已有击杀或抽牌能直接形成击杀时的其他 guard。

规则依据：`engine/source/ptcgProgram 22/CardImpl.h` 的 `Run Away Draw` 文本、
`CardMove.h` 的抽牌后 `ToDeckWithAttach` 实现；当前代码策略说明同步更新在
`STRATEGY.md`。

### 完整评测、control 与独立重复

第一批 candidate：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-41-low-deck-relax/summary.json`；
同协议 control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-41-control/summary.json`。
为排除随机洗牌影响，随后又运行一批独立 candidate：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-41-candidate-repeat/summary.json`，以及独立
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-41-control-repeat/summary.json`。

| 指标 | 第一批 candidate | 第一批 control | 独立 candidate | 独立 control |
|---|---:|---:|---:|---:|
| 胜 / 负 / 未完成 | 298 / 179 / 3 | 283 / 194 / 3 | 292 / 182 / 6 | 279 / 195 / 6 |
| 胜率（480 局分母） | 62.1% | 59.0% | 60.8% | 58.1% |
| 我方 agent error | 0 | 0 | 0 | 0 |
| candidate 胜场差 | +15 | — | +13 | — |

两批 candidate 都高于同协议 control；`kiyotah_dragapult` 两批分别为 9/30、
8/30，Iono 分别为 4/30、6/30，说明 v41 的主要收益不是直接修复这两个最弱
对局，但也没有造成 Dragapult 崩溃。收益集中在 `kokinn_search`、
`nursrijan_lucario`、`kacchan_anti_wall` 等中低胜率对手，且没有我方错误。

### Trace 复盘与准入判断

重点 trace：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-41-focus-traces/`，5 个
对手各 4 局。trace 未发现非法动作、牌库耗尽或我方异常；失败仍主要是第一只
Active 被击倒时没有带 Psychic Energy 的接力攻击者。由于单变量规则假设在两批
对照中方向一致，v41 暂保留为当前基线；后续候选必须以 v41 为 control，不能再以
旧 v37/v38 数据做跨批次结论。

## v42：即时 Alakazam 路线加入进化合法性检查（已回退，2026-07-19）

### 假设与改动

`_has_alakazam_attack_path()` 原先只看 Active、手牌和 Psychic Energy，没有检查
当前回合是否允许进化或 Pokémon 是否 `appearThisTurn`。候选复用已有
`_active_evolution_is_legal()`，并让 `_has_usable_attack_path()`、`_attack_security()`
传入当前状态。该改动只影响优先级判断，不改变返回的合法 option。

### 完整评测与同协议 control

候选：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-42-evolution-legal/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-42-control/summary.json`。

| 指标 | v42 candidate | v42 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 282 / 196 / 2 | 289 / 189 / 2 | -7 胜 |
| 胜率（480 局分母） | 58.8% | 60.2% | -1.5pp |
| 有效胜率（排除未完成） | 58.8% | 60.2% | -1.5pp |
| 我方 agent error | 0 | 0 | 无变化 |

候选的 `kiyotah_dragapult` 为 5/30，与 control 持平；`kiyotah_iono` 虽为 7/30，
但整体在多个对手回退。重点 trace：
`docs/reports/kaggle/alakazam-v5-auto-iter/iter-42-focus-traces/`，失败仍集中于
ready attacker 数量不足，而不是首回合误判本身。v42 已主动回退，当前代码恢复为
v41 的低牌库 Dudunsparce guard。

## v43：无接力目标时禁止 Trading Places（已回退，2026-07-19）

### 假设与改动

`Trading Places`（attack ID `423`）不造成伤害；只有 Bench 有带 Psychic Energy 的
Alakazam 时，换位才有即时收益。候选在主攻击评分中将“没有 ready Alakazam 的
423”降到 `END` 之后，保留有接力目标时的原逻辑。该候选来自
`iter-40-focus-traces` 和 `iter-37-focus-traces` 中无接力却选择 423 的败局。

### 完整评测与同协议 control

候选：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-43-trading-guard/summary.json`；
control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-43-control/summary.json`。

| 指标 | v43 candidate | v43 control | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 未完成 | 295 / 182 / 3 | 299 / 177 / 4 | -4 胜，未完成 -1 |
| 胜率（480 局分母） | 61.5% | 62.3% | -0.8pp |
| 有效胜率（排除未完成） | 61.7% | 62.8% | -1.1pp |
| 我方 agent error | 0 | 0 | 无变化 |

### Trace 复盘与回退

候选 trace：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-43-focus-traces/`。该 guard
确实消除了定向 trace 中无接力的 423 选择，但在部分场面让策略改选普通 20 点攻击
或 `END`，没有形成 ready 接力；`kiyotah_dragapult` 为 6/30，低于 control 的
10/30。规则判断本身正确，但动作收益不足，v43 已回退，当前代码保持 v41。

## v44：Iono 低奖赏后备的定向 Boss（暂保留，2026-07-19）

### 假设与改动

`kiyotah_iono` 经常把两奖的 Bellibolt ex 留在 Active，而 Bench 的 Iono's
Voltorb、Tadbulb、Wattrel 是一奖目标；当当前 Active 无法击倒、但我方当前攻击可以
击倒这些后备时，原有 Boss's Orders prize-race guard 会因为奖赏值不更高而保留 Boss。
本轮新增仅识别 Iono 牌面（265/268/269/270/271），并且只允许拉出一奖的
Voltorb/Tadbulb/Wattrel（265/268/270）且确认当前攻击可击倒；其他 15 个对手仍使用
原有 Boss guard。未修改牌表、能量选择或普通 Boss 逻辑。

### 完整评测、对照与独立重复

每批均为 16 个对手 × 30 局 = 480 局、交替先后手；`yakitori_raging_bolt` 的异常是
对手侧错误，未计为我方 agent error。结果目录：

- candidate：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-44-iono-boss/`
- 同批 control：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-44-control/`
- 独立 candidate repeat：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-44-candidate-repeat/`
- 独立 control repeat：`docs/reports/kaggle/alakazam-v5-auto-iter/iter-44-control-repeat/`

| 指标 | 第一批 candidate | 第一批 control | 独立 candidate | 独立 control |
|---|---:|---:|---:|---:|
| 胜 / 负 / 未完成 | 283 / 194 / 3 | 280 / 195 / 5 | 293 / 182 / 5 | 280 / 197 / 3 |
| 胜率（480 局分母） | 59.0% | 58.3% | 61.0% | 58.3% |
| candidate 胜场差 | +3 | — | +13 | — |
| `kiyotah_iono` | 8/30 | 4/30 | 5/30 | 4/30 |
| 我方 agent error | 0 | 0 | 0 | 0 |

### 复盘与准入判断

定向 candidate 的 Iono 第一批提升了 4 胜，独立重复只提升 1 胜，说明该 matchup
收益有随机波动；但在两批完整对照中 candidate 都高于 control，独立重复多 13 胜，
且没有我方异常。该条件的作用范围明确、可解释且没有改变其他 matchup 的常规
Supporter 优先级，因此暂保留 v44 作为当前基线。它仍只有约 61%，远未达到 90%；
下一轮继续针对 Dragapult、Iono 的连续攻击线，而不是把单一 Iono 小样本提升外推为
全局收益。

## v45：Iono Boss 仅限 Bellibolt ex Active（已回退，2026-07-19）

### 假设与改动

v44 的 Iono 定向 Boss 条件只检查场上是否出现 Iono 牌面，因此在 Active 也是一奖
Voltorb/Wattrel 时，可能把它换成另一只一奖后备。v45 将条件收窄为：Active 必须是
Iono's Bellibolt ex（269），且 Bench 的 Voltorb/Tadbulb/Wattrel 能被本回合当前攻击
击倒；其余策略完全沿用 v44。

### 完整评测、独立重复与回退

两批均为 16 个对手 × 30 局 = 480 局、交替先后手；对手侧异常不计作我方错误。
结果目录：`iter-45-bellibolt-boss/`、`iter-45-control/`、
`iter-45-candidate-repeat/`、`iter-45-control-repeat/`。

| 指标 | 第一批 v45 | 第一批 v44 control | 独立 v45 | 独立 v44 control |
|---|---:|---:|---:|---:|
| 胜 / 负 / 未完成 | 288 / 187 / 5 | 275 / 202 / 3 | 276 / 200 / 4 | 286 / 189 / 5 |
| 胜率（480 局分母） | 60.0% | 57.3% | 57.5% | 59.6% |
| `kiyotah_iono` | 6/30 | 3/30 | 7/30 | 2/30 |
| `kiyotah_dragapult` | 4/30 | 8/30 | 6/30 | 3/30 |
| 我方 agent error | 0 | 0 | 0 | 0 |

合并两批 v45 为 564W/387L/9D，v44 control 为 561W/391L/8D，净差仅 +3 胜且方向
不稳定。Iono 的局部提升没有抵消其他对手的回退，也没有改善 Dragapult；该条件已
主动回退，当前代码恢复 v44。这个实验说明“限制目标范围”本身不等于正确的奖赏
交换判断，后续候选必须同时验证整体对手池和关键 matchup。

## v46：每局开始清空多步效果进度（暂保留，2026-07-19）

### 假设与改动

评测器在同一个 Python module 中连续运行多局，而引擎的 effect serial 会在新局重新
分配。原代码的 `_EFFECT_PROGRESS` 是模块级字典，却只增加进度、从不在新局清空；
trace 证实同一 serial 会在不同 game 中重复出现，并且可能对应不同卡牌或 context。
v46 只在 `agent()` 收到新局的 `select=None` deck 请求时清空该字典，不改变任何卡牌
优先级或返回的合法 action。

### 完整评测、对照与独立重复

每批为 16 个对手 × 30 局 = 480 局、交替先后手；所有 agent error 均为 0，未完成局
来自已知对手侧异常。结果目录：`iter-46-effect-reset/`、`iter-46-control/`、
`iter-46-candidate-repeat/`、`iter-46-control-repeat/`。

| 指标 | 第一批 v46 | 第一批 v44 control | 独立 v46 | 独立 v44 control |
|---|---:|---:|---:|---:|
| 胜 / 负 / 未完成 | 286 / 189 / 5 | 294 / 181 / 5 | 294 / 182 / 4 | 284 / 188 / 8 |
| 胜率（480 局分母） | 59.6% | 61.3% | 61.3% | 59.2% |
| 我方 agent error | 0 | 0 | 0 | 0 |

合并两批为 v46 `580W/371L/9D`、control `578W/369L/13D`，净差只有 +2 胜/960 局，
不构成胜率提升；主要瓶颈 Iono/Dragapult 也没有稳定改善。但跨局状态泄漏是确定的
生命周期错误，修复不会改变单局规则且能使多步搜索选择独立于上一局，因此保留 v46
作为正确性基线。后续策略实验必须在 v46 上比较，不能把这轮的 +2 胜外推为收益。

## 发布版：alakazam_v5_auto_iter（基于 v46，2026-07-19）

暂停继续迭代后，最终保留 v46 作为发布版：它包含 v41 的低牌库
`Dudunsparce` 规则、v44 的 Iono 定向 Boss，以及 v46 的多局 effect 状态隔离修复。
历史候选和回退记录全部保留在本文件中；提交目录只保留这一套当前发布源文件。

发布包：`dist/alakazam_v5_auto_iter.tar.gz`。该包按正常 submission 边界只包含
`main.py`、`deck.csv` 和 `cg/`，并已通过资产、语法、raw-exec 兼容性和归档内容检查。
