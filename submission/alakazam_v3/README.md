# Alakazam V3 Submission

这是基于 V2 卡组和运行时实现的策略改进版本。目录包含可独立打包的
`main.py`、`deck.csv` 和 `cg/`；`IMPROVEMENT.md` 保留用户提出的改进记录，
`STRATEGY.md` 记录已落地的 V3 规则。

## Strategy Changes

V3 沿用 V2 的 60 张卡组，不再改变卡表；主要改动集中在策略行动排序：

- 场上实际存在三只 Abra/Kadabra/Alakazam 后，再扩展 Dunsparce 引擎；
- Telepath Energy 优先服务 Abra，无 Abra 可铺时才 fallback 到 Dunsparce；
- 保护 20 张手牌和 10 张牌库阈值；
- Hilda、Rare Candy、Trading Places 以及恢复牌按 V3 策略选择目标。

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

其余训练家和恢复卡沿用 V2 配置：
Eri `1186`, Night Stretcher `1097`, Wondrous Patch `1146`, Sacred Ash `1129`,
and Lana's Aid `1184`.

## Runtime

The V3 `cg/` directory is copied from the tested V2 simulator runtime. The agent
still returns only indices from the current simulator options and keeps the same
observation-path compatibility for local and Kaggle execution.

## Validation

```bash
python3 scripts/check_assets.py
python3 -m compileall scripts submission
python3 scripts/run_local_battle.py --agent0 alakazam_v3 --agent1 alakazam_v2 \
  --output /tmp/alakazam_v3_vs_v2.json
python3 scripts/run_local_battle.py --agent0 alakazam_v3 --agent1 alakazam_v1 \
  --output /tmp/alakazam_v3_vs_v1.json
```
