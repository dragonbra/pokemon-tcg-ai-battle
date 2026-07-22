# Alakazam V5 Submission

V5 是基于 V4 官方对局 review 修正后的可独立打包 Alakazam agent，包含
`main.py`、60 张卡组和官方 `cg/` runtime。策略保持确定性，只返回 simulator
当前提供的合法选项。

## 卡组

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
| 2 | Battle Cage | 1264 |
| 4 | Poké Pad | 1152 |
| 4 | Hilda | 1225 |
| 4 | Dawn | 1231 |
| 3 | Boss's Orders | 1182 |
| 2 | Enhanced Hammer | 1081 |
| 1 | Enriching Energy | 13 |
| 4 | Telepath Psychic Energy | 19 |
| 2 | Basic Psychic Energy | 5 |
| 1 | Night Stretcher | 1097 |
| 1 | Wondrous Patch | 1146 |
| 1 | Sacred Ash | 1129 |
| 1 | Lana's Aid | 1184 |
| 3 | Xerosic's Machinations | 1197 |

共 60 张。V5 暂不改动 V4 的牌表：review 中的 Hisuian Heavy Ball、Snorlax、
Professor's Research 和 Reversal Energy 不在当前官方引擎的可用卡池中；Eevee
也不是 `Telepath Psychic Energy`（只能搜索 Basic Psychic Pokémon）的合法目标。
现有的 `Enriching Energy` 是可以用于 Dudunsparce 过牌线路的实际特殊能量。

## V5 实现重点

- 首回合不会用 Poké Pad 搜索 Kadabra 或 Alakazam；优先搜索 Dunsparce 等 setup
  Basic。第二回合起才根据当前已有、且不是本回合放下的 Abra 选择 Kadabra。
- Active Abra 保留 `Rare Candy + Alakazam` 直通机会；有多个 Kadabra 进化目标时，
  第二回合优先进化 Bench Abra。第三回合以后，Active Kadabra 有能量且手牌有
  Alakazam 时优先自然进化，不浪费 Rare Candy。
- 只有在上一回合己方被击倒、当前没有 Alakazam 高价值线路且手牌较少时，才放下
  Fezandipiti ex 使用 `Flip the Script`；30 点的 Kadabra 攻击只是最后 fallback。
- Abra、Kadabra、Alakazam 每条进化线默认只保留一张 Psychic Energy。Telepath
  优先贴到未带能量的 Psychic 攻击线，并按规则触发搜索；不能把它当作给
  Dunsparce 搜索 Abra 的合法替代。
- Active Fezandipiti ex 或 Shaymin 在 Bench 有带能量的 Alakazam 时，才用一张能量
  支付撤退并交接攻击；首回合和没有明确攻击收益的 Retreat 都跳过。
- 弃牌区恢复在场上没有 Abra 时优先取回 Abra；Lana's Aid 在合法范围内尽量取回
  多张 Abra 系列。没有其他攻击线能量需求时，才把 Enriching Energy 给
  Dudunsparce 作为额外过牌资源。

## 校验

```bash
python3 scripts/check_assets.py
python3 -m compileall -q scripts submission
bash scripts/package_submission.sh alakazam_v5_auto_iter
```
