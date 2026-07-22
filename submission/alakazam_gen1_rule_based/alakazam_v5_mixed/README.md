# Alakazam V5 Mixed Submission

`v5_mixed` 是固定 V5 牌表上的策略实验版本：保留 V5 的 `deck.csv`、公开 API、
官方 `cg/` runtime、Mist Energy 运行时处理和确定性合法选项，只混合公开
Alakazam agent 中高置信的状态建模与资源 gate。不引入随机性、search 或新的卡组。

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

共 60 张。`v5_mixed/deck.csv` 必须与 `alakazam_v5/deck.csv` 完全一致。review 中的 Hisuian Heavy Ball、Snorlax、
Professor's Research 和 Reversal Energy 不在当前官方引擎的可用卡池中；Eevee
也不是 `Telepath Psychic Energy`（只能搜索 Basic Psychic Pokémon）的合法目标。
现有的 `Enriching Energy` 是可以用于 Dudunsparce 过牌线路的实际特殊能量。

## Mixed 实现重点

- 用 `ready now / next attack path / future line / unenergized line / recovery`
  描述攻击连续性；场上三只 Abra 系列仍是铺场目标，但不再被当成攻击安全的充分条件。
- 将 Dunsparce/Dudunsparce 作为独立抽牌引擎统计。只有当前攻击和下一只打手安全、
  牌库高于硬保护线、且过牌能改善目标时，才使用 Run Away Draw 或 Enriching Energy。
- 抽牌遵循当前 KO → 抽牌后 KO → 下一次攻击连续性 → 额外资源的优先级；手牌 20
  张、牌库 10 张和“牌库不低于剩余 Prize”共同构成保护线。
- Fezandipiti ex 只有在上回合被 KO 且 Flip the Script 对 KO/攻击连续性有必要时才
  下场；对手剩 2 Prize、已有普通 Bench 时不为普通抽牌暴露两奖宝可梦。
- Xerosic 只有在可见状态足以把对手 Alakazam 从 lethal 压到 non-lethal，或明显
  改变 mirror 交换时才使用；Boss/Hammer/Retreat 同样要求可证明的攻击或 Prize 收益。
- 能量和恢复按最短攻击路径选择：没有场上 Abra 系列时先恢复 Abra；每只攻击线
  Pokémon 默认只贴第一张 Psychic Energy，Telepath 不贴重复能量。

### Telepath 的运行时边界

官方引擎中的 Telepath effect 条件是“贴到 Psychic Pokémon”，因此在场上没有
Psychic Pokémon、手里同时有 Abra 和 Telepath 时，策略先放 Abra，再在下一次合法
主行动贴 Telepath。不能把 review 中关于“贴到任意宝可梦也能触发”的描述当作当前
引擎事实；这也是本版本保留的一个明确验证边界。

## 校验

```bash
python3 scripts/check_assets.py
python3 -m compileall -q scripts submission
bash scripts/package_submission.sh alakazam_v5_mixed
```
