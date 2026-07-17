# Alakazam V1 Submission

这是第一版胡地 baseline。策略、卡组和 simulator runtime 都在本目录中；该目录可以独立打包为一份 Kaggle submission。

最终 archive 只包含 `main.py`、`deck.csv` 和 `cg/`。本目录中的 `README.md`、`STRATEGY.md` 和 replay 说明只用于研究、评审和后续迭代，不会被 `scripts/package_submission.sh alakazam_v1` 打包。

## English Deck List

`deck.csv` 保存 simulator Card ID；下面是按官方 `EN_Card_Data.csv` 解析出的英文卡组列表。

### Pokemon

| Count | Card | Card ID |
|---:|---|---:|
| 3 | Alakazam | 743 |
| 3 | Dunsparce | 305 |
| 1 | Fezandipiti ex | 140 |
| 1 | Psyduck | 858 |
| 4 | Abra | 741 |
| 1 | Shaymin | 343 |
| 4 | Kadabra | 742 |
| 3 | Dudunsparce | 66 |

### Trainer

| Count | Card | Card ID |
|---:|---|---:|
| 1 | Eri | 1186 |
| 1 | Night Stretcher | 1097 |
| 4 | Buddy-Buddy Poffin | 1086 |
| 3 | Rare Candy | 1079 |
| 1 | Wondrous Patch | 1146 |
| 1 | Sacred Ash | 1129 |
| 4 | Battle Cage | 1264 |
| 1 | Lana's Aid | 1184 |
| 4 | Poke Pad | 1152 |
| 4 | Hilda | 1225 |
| 4 | Dawn | 1231 |
| 2 | Enhanced Hammer | 1081 |
| 3 | Boss's Orders | 1182 |

### Energy

| Count | Card | Card ID |
|---:|---|---:|
| 1 | Enriching Energy | 13 |
| 4 | Telepath Psychic Energy | 19 |
| 2 | Basic Psychic Energy | 5 |

**Total: 60 cards**

## Strategy

策略设计说明和待补充的策略规范见 [`STRATEGY.md`](STRATEGY.md)。当前 V1 已实现的规则 baseline 是：

- setup 阶段优先把 Abra、Dunsparce 放入场上；
- 主阶段优先使用 Kadabra、Alakazam、Dudunsparce 的抽牌 Ability；
- 优先完成 Kadabra/Alakazam 进化线；
- Psychic Energy 优先附着到当前或下一只攻击手；
- 使用 Dawn、Hilda、Buddy-Buddy Poffin、Poké Pad 维持 setup；
- 只返回 simulator 当前给出的合法 option，遇到未专门处理的效果时保守回退。

这还不是最终策略：目前尚未完整实现 Powerful Hand 的伤害/手牌规划、Prize race、Boss's Orders 的目标价值评估和精细 recovery 时机。后续策略说明应优先补充到 `STRATEGY.md`，再把其中的规则落实到 `main.py`。

## Card Data Mapping Notes

卡组以官方 simulator 的 Card ID 为准。提交表中部分 expansion/collection number 与本地官方 CSV 不一致，因此这里按英文卡名映射，并保留实际 CSV 信息：

- `Dunsparce JTG 120` -> Card ID `305`，与官方 CSV 精确匹配；
- `Fezandipiti ex ASC 142` -> Card ID `140`，官方 CSV 为 `SFA 38`；
- `Rare Candy MEG 125` -> Card ID `1079`，官方 CSV 为 `SVI 191`；
- `Night Stretcher ASC 196` -> Card ID `1097`，官方 CSV 为 `SFA 61`；
- `Boss's Orders MEG 114` -> Card ID `1182`，官方 CSV 为 `PAL 172`；
- `Telepathic {P} Energy POR 88` -> Card ID `19`，官方 CSV 名称为 `Telepath Psychic Energy`，编号为 `POR 87`；
- `Basic {P} Energy MEE 5` -> Card ID `5`，官方 CSV 为 `SVE 5`。

## Local Test Snapshot

这是当前小样本 smoke test，不是 leaderboard 结论。

| Matchup | Battles | Completed | Alakazam V1 result | Steps |
|---|---:|---:|---|---:|
| Alakazam V1 vs Alakazam V1 | 1 | 1 | Same-policy self-play, player 0 won | 202 |
| Alakazam V1 vs Official Water | 1 | 1 | V1 won | 45 |
| Official Water vs Alakazam V1 | 1 | 1 | V1 won | 109 |

Cross-matchup result is currently **2 wins / 2 battles (100%)**, with only two samples. All three battles completed without an agent error.

Replay files:

- [`alakazam_v1_self_battle.json`](../../replays/alakazam_v1_self_battle.json)
- [`alakazam_v1_vs_official_water.json`](../../replays/alakazam_v1_vs_official_water.json)
- [`official_water_vs_alakazam_v1.json`](../../replays/official_water_vs_alakazam_v1.json)

## Packaging

```bash
python3 scripts/check_assets.py
bash scripts/package_submission.sh alakazam_v1
tar -tzf dist/alakazam_v1.tar.gz
```
