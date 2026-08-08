# Frozen 002 胡地强度复测与归因

## 结论

Frozen 002 的接近 60% 胜率可以复现：canonical 为 60.16%，独立新 seed 为 59.18%，相差 0.98%。这支持它稳定强于多数 SP 构筑，而不是单个 epoch/seed 偶然峰值。

最干净的构筑对照是 SP04：两者只有 `Nighttime Mine ×2` 与 `Battle Cage ×2` 的替换。在相同的 2,048 个 engine/search seed、对手与先后手位置下，002 为 59.18%，SP04 为 56.79%，配对净差 2.39%，95% 区间 0.08%–4.71%。

差异主要来自含 Tera 的对局。Frozen 池 55 套中有 20 套含 Tera，按固定频率占 496/2048（24.22%）。该子集 002 比 SP04 高 6.25%（95% 区间 1.72%–10.78%）；非 Tera 子集只高 1.16%（区间 -1.53%–3.85%，跨 0）。因此数据与 Nighttime Mine “Tera 攻击多 1 个无色能量”的机制一致。

这仍然不是对卡牌普适强度的证明。它证明的是：在固定 Policy-0806、Frozen-0806 频率池和当前 official-state CUDA runtime 下，002 的完整 deck-policy system 优于仅将两张 Stadium 换成 Battle Cage 的 SP04。报告没有逐动作 Stadium 上场率，且换牌会改变洗牌和后续策略轨迹，不能声称每一局净胜都由 Stadium 的即时效果直接造成。

## 总表

| 构筑 / 批次 | W-L-D | 胜率 | Wilson 95% CI | 合同备注 |
|---|---:|---:|---:|---|
| Frozen 002 · canonical | 1232-816-0 | 60.16% | 58.02%–62.26% | canonical seed；旧结果未记录 50 回合护栏 |
| Frozen 002 · 独立复测 | 1212-830-6 | 59.18% | 57.04%–61.29% | 新随机 seed 843573811；当前统一护栏 |
| SP01_MAGA | 1133-899-16 | 55.32% | 53.16%–57.46% | 独立 Seeded-2048 |
| SP02_MAGA | 1179-859-10 | 57.57% | 55.42%–59.69% | 独立 Seeded-2048 |
| SP03_MAGA | 1134-901-13 | 55.37% | 53.21%–57.51% | 独立 Seeded-2048 |
| SP04_MAGA | 1177-859-12 | 57.47% | 55.32%–59.60% | 独立 Seeded-2048 |
| SP05_MAGA | 1074-954-20 | 52.44% | 50.28%–54.60% | 独立 Seeded-2048 |
| SP06_MAGA | 1069-959-20 | 52.20% | 50.03%–54.35% | 独立 Seeded-2048 |
| SP07_MAGA | 1063-975-10 | 51.90% | 49.74%–54.06% | 独立 Seeded-2048 |
| SP08_MAGA | 1139-898-11 | 55.62% | 53.45%–57.75% | 独立 Seeded-2048 |
| SP09_MAGA | 965-1074-9 | 47.12% | 44.96%–49.28% | 独立 Seeded-2048 |
| SP04 · 同-seed 受控对照 | 1163-872-13 | 56.79% | 54.63%–58.92% | 与 002 共用逐局 engine/search seed |

## 受控分层

| 分层 | 对局 | 002 胜率 | SP04 胜率 | 配对净差 | 95% CI |
|---|---:|---:|---:|---:|---:|
| 全部 | 2048 | 59.18% | 56.79% | 2.39% | 0.08%–4.71% |
| 含 Tera 构筑 | 496 | 73.99% | 67.74% | 6.25% | 1.72%–10.78% |
| 不含 Tera 构筑 | 1552 | 54.45% | 53.29% | 1.16% | -1.53%–3.85% |

## 为什么 002 看起来特别高

1. **Nighttime Mine 的目标并不少。** 用户直觉中的“环境 Tera 不多”若按构筑种类看是 20/55；按固定评测频率仍有 496/2048 场。约四分之一的权重足以让一个明显的 Tera matchup 增益贡献约 1.5 个总体百分点。
2. **SP04 证明 Stadium 选择贡献约 2–3 点，而非十几点。** 同-seed 对照的 +2.39 点刚刚达到 95% 配对区间下界大于 0；Tera 组贡献了 31 个净胜，约占总净增 49 胜的 63%。
3. **其他 SP 不是单变量对照。** 它们还改变 Alakazam、Dudunsparce、Shaymin、Boss、Xerosic、Enhanced Hammer 或 Wondrous Patch。SP09 相比 SP04 又少 1 张 Alakazam、多 1 张 Battle Cage；其 47.12% 不能归因于 Nighttime Mine 缺失。
4. **这是 deck-policy compatibility。** Policy-0806 是固定策略；卡位会改变起手、检索、动作候选和策略是否落在熟悉分布。某张卡的评测收益可能来自模型更会使用它，而不等同于人类最优策略下的卡牌理论强度。
5. **Battle Cage 并非全面劣势。** 它能阻止对手用攻击或 Ability 向 Bench 放置伤害指示物，理论上会对 Dragapult/Dusknoir 类压力有价值。实际逐对手差异并不单调；这正说明两张 Stadium 是 matchup trade-off，而不是一张严格支配另一张。

## 逐对手同-seed 差分

表按净胜变化绝对值排序。小样本行只用于定位贡献，不单独作显著性结论。

| 编号 | 对手 | 含 Tera | 对局 | 002 W-L-D | SP04 W-L-D | 002 净胜变化 |
|---:|---|:---:|---:|---:|---:|---:|
| 001 | Marnie's Grimmsnarl ex / Froslass | 否 | 408 | 148-260-0 | 198-210-0 | -50 |
| 002 | Alakazam / Dudunsparce | 否 | 272 | 131-139-2 | 107-164-1 | +24 |
| 006 | Teal Mask Ogerpon ex / Hero’s Cape | 是 | 80 | 57-23-0 | 48-32-0 | +9 |
| 004 | Mega Lopunny ex / Mega Froslass ex | 否 | 120 | 79-41-0 | 71-49-0 | +8 |
| 013 | Cynthia's Garchomp ex / Roserade | 否 | 40 | 29-11-0 | 21-19-0 | +8 |
| 038 | Crustle / Cornerstone Mask Ogerpon ex | 是 | 16 | 13-0-3 | 5-7-4 | +8 |
| 003 | Mega Lopunny ex / Mega Froslass ex | 否 | 120 | 74-46-0 | 68-52-0 | +6 |
| 014 | Mega Kangaskhan ex / Crustle | 否 | 40 | 27-13-0 | 22-18-0 | +5 |
| 024 | Crustle / Cornerstone Mask Ogerpon ex | 是 | 16 | 10-5-1 | 5-7-4 | +5 |
| 031 | Marnie's Grimmsnarl ex / Froslass | 否 | 16 | 3-13-0 | 7-9-0 | -4 |
| 012 | Mega Lopunny ex / Mega Froslass ex | 否 | 40 | 37-3-0 | 34-6-0 | +3 |
| 017 | Marnie's Grimmsnarl ex / Froslass | 否 | 24 | 5-19-0 | 8-16-0 | -3 |
| 022 | Mega Lopunny ex / Mega Froslass ex | 否 | 16 | 11-5-0 | 8-8-0 | +3 |
| 027 | Mega Kangaskhan ex / Crustle | 是 | 16 | 13-3-0 | 16-0-0 | -3 |
| 029 | Teal Mask Ogerpon ex / Hero’s Cape | 是 | 16 | 12-4-0 | 9-7-0 | +3 |
| 030 | Mega Kangaskhan ex / Crustle | 否 | 16 | 12-4-0 | 9-7-0 | +3 |
| 040 | Alakazam / Dudunsparce | 否 | 16 | 12-4-0 | 9-7-0 | +3 |
| 007 | Dragapult ex | 是 | 56 | 23-33-0 | 21-35-0 | +2 |
| 020 | Mega Venusaur ex / Meganium | 是 | 24 | 23-1-0 | 21-3-0 | +2 |
| 025 | Mega Lopunny ex / Mega Froslass ex | 是 | 16 | 13-3-0 | 15-1-0 | -2 |
| 026 | Mega Kangaskhan ex / Crustle | 是 | 16 | 15-1-0 | 13-3-0 | +2 |
| 032 | Mega Lopunny ex / Mega Froslass ex | 是 | 16 | 15-1-0 | 13-3-0 | +2 |
| 033 | Mega Kangaskhan ex / Crustle | 否 | 16 | 11-5-0 | 9-7-0 | +2 |
| 039 | Alakazam / Dudunsparce | 否 | 16 | 11-5-0 | 9-7-0 | +2 |
| 005 | Marnie's Grimmsnarl ex / Froslass | 否 | 80 | 41-39-0 | 40-40-0 | +1 |
| 008 | Teal Mask Ogerpon ex / Hero’s Cape | 是 | 56 | 41-15-0 | 42-14-0 | -1 |
| 010 | Festival Lead / Dipplin | 否 | 40 | 15-25-0 | 14-26-0 | +1 |
| 015 | Wellspring Mask Ogerpon ex / Teal Mask Ogerpon ex | 是 | 24 | 17-7-0 | 18-6-0 | -1 |
| 016 | Slowking Toolbox | 否 | 24 | 21-3-0 | 22-2-0 | -1 |
| 018 | Dragapult ex | 是 | 24 | 16-8-0 | 15-9-0 | +1 |
| 019 | Barbaracle / Cornerstone Mask Ogerpon ex | 是 | 24 | 18-6-0 | 19-5-0 | -1 |
| 021 | Team Rocket's Mewtwo ex / Spidops | 否 | 24 | 7-17-0 | 6-16-2 | +1 |
| 023 | Hydrapple ex / Meganium | 是 | 16 | 15-1-0 | 14-2-0 | +1 |
| 028 | Mega Lopunny ex / Mega Froslass ex | 否 | 16 | 15-1-0 | 14-1-1 | +1 |
| 035 | Mega Kangaskhan ex / Crustle | 是 | 16 | 15-1-0 | 14-2-0 | +1 |
| 036 | Teal Mask Ogerpon ex / Hero’s Cape | 是 | 16 | 14-2-0 | 13-3-0 | +1 |
| 041 | Hydrapple ex / Meganium | 是 | 16 | 15-1-0 | 14-2-0 | +1 |
| 042 | Arboliva ex / Meganium / Teal Mask Ogerpon ex | 是 | 16 | 14-2-0 | 13-3-0 | +1 |
| 044 | Mega Starmie ex / Mega Froslass ex | 否 | 8 | 5-3-0 | 4-4-0 | +1 |
| 051 | Archaludon ex / Cinderace | 否 | 8 | 8-0-0 | 7-1-0 | +1 |
| 052 | Archaludon ex / Cinderace | 否 | 8 | 8-0-0 | 7-1-0 | +1 |
| 054 | N's Zoroark ex / Munkidori | 否 | 8 | 7-1-0 | 6-2-0 | +1 |
| 055 | Archaludon ex / Cinderace | 否 | 8 | 8-0-0 | 7-1-0 | +1 |
| 009 | Mega Lucario ex / Solrock | 否 | 40 | 23-17-0 | 23-17-0 | +0 |
| 011 | Mega Kangaskhan ex / Crustle | 否 | 40 | 30-10-0 | 30-10-0 | +0 |
| 034 | Alakazam / Dudunsparce | 否 | 16 | 13-3-0 | 13-2-1 | +0 |
| 037 | Marnie's Grimmsnarl ex / Froslass | 否 | 16 | 9-7-0 | 9-7-0 | +0 |
| 043 | Dragapult ex / Dusknoir | 是 | 16 | 8-8-0 | 8-8-0 | +0 |
| 045 | Mega Starmie ex / Mega Froslass ex | 否 | 8 | 5-3-0 | 5-3-0 | +0 |
| 046 | Mega Starmie ex / Dusknoir | 否 | 8 | 7-1-0 | 7-1-0 | +0 |
| 047 | Erika's Vileplume ex / Cinderace | 否 | 8 | 8-0-0 | 8-0-0 | +0 |
| 048 | Archaludon ex / Cinderace | 否 | 8 | 5-3-0 | 5-3-0 | +0 |
| 049 | N's Zoroark ex / Munkidori | 否 | 8 | 7-1-0 | 7-1-0 | +0 |
| 050 | Mega Starmie ex / Dusknoir | 否 | 8 | 6-2-0 | 6-2-0 | +0 |
| 053 | Mega Starmie ex / Mega Froslass ex | 否 | 8 | 7-1-0 | 7-1-0 | +0 |

## 证据边界

- 通用规则与卡牌级事实：Nighttime Mine 和 Battle Cage 的具体文本来自当前 `data/official/EN_Card_Data.csv`；游戏胜负与回合语义以当前 runtime 为准。
- 实验事实：所有新结果使用 strict FP32、256 resident CUDA lanes、50 完整回合平局、同一方同 Ability 第 20 次判负。
- 策略假设：将“含 Tera 卡的构筑”作为 Nighttime Mine 有潜在直接作用的分层；报告没有证明该 Tera 每局都实际进场或攻击。
- CUDA/official CPU parity 尚未形成完整强度替代合同，因此结论限定于当前 CUDA evaluation runtime。
