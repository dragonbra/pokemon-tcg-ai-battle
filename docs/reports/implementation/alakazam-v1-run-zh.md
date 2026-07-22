# 胡地 V1 提交与本地验收

日期：2026-07-18

## 提交结构

每一套卡组都是一个完整提交目录：

```text
submission/
├── official_water/
│   ├── main.py
│   ├── deck.csv
│   └── cg/
└── alakazam_v1/
    ├── main.py
    ├── deck.csv
    └── cg/
```

这样可以分别执行 `bash scripts/package_submission.sh official_water` 和 `bash scripts/package_submission.sh alakazam_v1`，不会把两个 agent 或两套卡组混入同一个 archive。

## 胡地 V1 卡表映射

卡组按照 `data/official/EN_Card_Data.csv` 的 Card ID 执行。主要映射如下：

| 卡片 | Card ID |
|---|---:|
| Alakazam MEG 56 | 743 |
| Abra MEG 54 | 741 |
| Kadabra MEG 55 | 742 |
| Dunsparce JTG 120 | 305 |
| Dudunsparce TEF 129 | 66 |
| Fezandipiti ex（官方 CSV 为 SFA 38） | 140 |
| Psyduck ASC 39 | 858 |
| Shaymin DRI 10 | 343 |
| Telepath Psychic Energy（官方 CSV 为 POR 87） | 19 |
| Basic Psychic Energy（官方 CSV 为 SVE 5） | 5 |

`Dunsparce JTG 120` 在官方 CSV 中是 Card ID `305`，V1 已使用这个版本。官方 CSV 中还有若干卡名相同但 expansion/collection number 与提交表不同的情况：`Fezandipiti ex` 只有 `SFA 38`、`Rare Candy` 只有 `SVI 191`、`Night Stretcher` 只有 `SFA 61`、`Boss's Orders` 只有 `PAL 172`。本次按英文 card name 对应到官方 simulator ID，并在表中保留这些差异，不能把它们宣称为同一套印刷版本。

## V1 策略范围

- 起手优先建立 Abra 和 Dunsparce；
- 主动作优先使用 Kadabra、Alakazam 和 Dudunsparce 的抽牌 Ability；
- 优先完成 Kadabra/Alakazam 进化线；
- Telepath Psychic Energy 和 Basic Psychic Energy 优先给攻击线；
- 通过 `Dawn`、`Hilda`、`Buddy-Buddy Poffin`、`Poké Pad` 维持 setup；
- 只在 simulator 给出的合法 option 中选择，未知效果保守回退到合法选项。

这是一版可运行的规则 baseline，不代表已完成伤害计算、Prize race、Boss's Orders 目标价值和最优资源管理。

## 验收

```bash
python3 scripts/check_assets.py
bash scripts/package_submission.sh official_water
bash scripts/package_submission.sh alakazam_v1
python3 scripts/run_local_battle.py \
  --agent0 alakazam_v1 \
  --agent1 alakazam_v1 \
  --max-steps 20000 \
  --output /tmp/alakazam_v1_self_battle.json
python3 scripts/run_local_battle.py \
  --agent0 alakazam_v1 \
  --agent1 official_water \
  --max-steps 20000 \
  --output /tmp/alakazam_v1_vs_official_water.json
```

结果：胡地自战 202 步完成；胡地 V1 先手对官方水系 45 步完成，官方水系先手对胡地 V1 109 步完成，均无异常。
