# Alakazam V2 Submission

这是基于 V1 卡组和运行时实现的攻击优先版本。目录包含可独立打包的
`main.py`、`deck.csv` 和 `cg/`；`IMPROVEMENT.md` 保留用户提出的改进记录，
`STRATEGY.md` 记录已落地的 V2 规则。

## Deck Changes

V2 只做一项明确的卡组调整：移除 1 张 Psyduck (`858`)，加入第 4 张
Alakazam (`743`)。其余 59 张保留 V1 配置，以便把对局差异归因到策略和这一项
核心攻击手增量。

| Count | Card | Card ID |
|---:|---|---:|
| 4 | Alakazam | 743 |
| 3 | Dunsparce | 305 |
| 1 | Fezandipiti ex | 140 |
| 4 | Abra | 741 |
| 1 | Shaymin | 343 |
| 4 | Kadabra | 742 |
| 3 | Dudunsparce | 66 |
| 3 | Rare Candy | 1079 |
| 4 | Buddy-Buddy Poffin | 1086 |
| 4 | Battle Cage | 1264 |
| 4 | Poké Pad | 1152 |
| 4 | Hilda | 1225 |
| 4 | Dawn | 1231 |
| 3 | Boss's Orders | 1182 |
| 2 | Enhanced Hammer | 1081 |
| 1 | Enriching Energy | 13 |
| 4 | Telepath Psychic Energy | 19 |
| 2 | Basic Psychic Energy | 5 |

The remaining one-of Trainers and recovery cards are unchanged from V1:
Eri `1186`, Night Stretcher `1097`, Wondrous Patch `1146`, Sacred Ash `1129`,
and Lana's Aid `1184`.

## Runtime

The V2 `cg/` directory is copied from the tested V1 simulator runtime. The agent
still returns only indices from the current simulator options and keeps the same
observation-path compatibility for local and Kaggle execution.

## Validation

```bash
python3 scripts/check_assets.py
python3 -m compileall scripts submission
python3 scripts/run_local_battle.py --agent0 alakazam_v2 --agent1 alakazam_v2 \
  --output replays/alakazam_v2_self_battle.json
python3 scripts/run_local_battle.py --agent0 alakazam_v2 --agent1 official_water \
  --output replays/alakazam_v2_vs_official_water.json
```
