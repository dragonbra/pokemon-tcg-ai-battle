# Alakazam V4 Submission

V4 是一个可独立打包的 Alakazam agent，包含 `main.py`、60 张卡组、官方
`cg/` runtime 和策略说明。策略保持确定性，只从 simulator 当前返回的合法
选项中选择。

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

共 60 张。V4 用 3 张 Xerosic's Machinations 替换了 V3 的 Eri 和两张
Battle Cage，保留两张 Battle Cage 作为恢复和场地资源。

## V4 的主要变化

- 第二回合优先识别 `Hilda + Rare Candy + Alakazam` 的 Active Abra 直通线路；
  没有直通条件时，保留 Active Abra 的机会，把 Bench Abra 先进化成 Kadabra。
- Poké Pad 按当前可执行的线路逐张使用：直通线路找 Alakazam，否则找 Kadabra，
  没有可进化线路时才找 Dunsparce。
- Enhanced Hammer 优先处理对手 Active 的特殊能量，尤其是 Mist Energy；
  Alakazam 对带 Mist 的目标按当前官方 runtime 的实际伤害视为 0。
- Retreat 只在换上已有 Psychic Energy 的 Alakazam、或 Active Fezandipiti ex
  贴能量后能完成攻击时执行。
- Fezandipiti ex 只有在避免无 Bench 败局，或上一回合有宝可梦被击倒且确实需要
  Flip the Script 创造机会时才放下。
- 对手场面出现 Abra/Kadabra/Alakazam 且手牌至少 4 张、进入中期后，Xerosic
  提高优先级；前期仍优先保证进化和持续攻击。

## 校验

```bash
python3 scripts/check_assets.py
python3 -m compileall scripts submission
bash scripts/package_submission.sh alakazam_v4
```
