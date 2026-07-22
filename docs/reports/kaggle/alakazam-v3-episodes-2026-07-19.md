# Alakazam V3 Kaggle 官方 Episode 复盘

## 数据范围

- Submission：`54813215`，文件 `alakazam_v3.tar.gz`。
- 查询时状态：`COMPLETE`，public score `652.4`。
- Kaggle 账号在 replay 中显示为 `炽月殿-ptcg糕手`。
- 原始数据目录：[`replays/kaggle_v3_54813215/`](../../../replays/kaggle_v3_54813215/)。
- 本文只分析 Kaggle 官方 Episode；本地 simulator 输出不纳入结论。
- V3 沿用 V2 的 60 张卡组，本轮主要验证行动排序、Telepath Energy 铺场、Dunsparce
  过牌保护和进化目标选择。

## Episode 总览

| Episode | 类型 | 对手 | V3 位置 | 先手 | 结果 | replay steps | 主要终局/观察 |
|---:|---|---|---:|---:|---|---:|---|
| [86733922](../../../replays/kaggle_v3_54813215/86733922/episode-86733922-replay.json) | public | kyulkyu | 0 | 0 | V3 胜 | 37 | turn 5 Alakazam 以 280 伤害击倒 Crustle |
| [86733250](../../../replays/kaggle_v3_54813215/86733250/episode-86733250-replay.json) | public | wbt | 0 | 0 | V3 负 | 117 | 对带 Mist Energy 的 Crustle 连续造成 0 伤害，turn 19 牌库耗尽 |
| [86732591](../../../replays/kaggle_v3_54813215/86732591/episode-86732591-replay.json) | public | tk | 1 | 0 | V3 负 | 152 | 对手 Boss’s Orders 拉出 Bench 上的 Fezandipiti ex 并完成两奖击倒 |
| [86731938](../../../replays/kaggle_v3_54813215/86731938/episode-86731938-replay.json) | public | Ömer Faruk Yüce | 0 | 0 | V3 胜 | 143 | turn 13 Alakazam 以 600 伤害击倒 Solrock |
| [86731321](../../../replays/kaggle_v3_54813215/86731321/episode-86731321-replay.json) | public | snow_n | 1 | 1 | V3 胜 | 71 | 首只 Alakazam 被击倒后，第二只 Alakazam 在 turn 9 完成反击 |
| [86730455](../../../replays/kaggle_v3_54813215/86730455/episode-86730455-replay.json) | validation self-play | 自己 | 0 | 0 | V3 胜 | 138 | 官方 validation self-play，不计入 public 样本 |

public 样本为 3 胜 2 负。`86730455` 是官方 validation self-play，不能和 public
对手池结果混合计算。

## 关键指标

V3 的 5 局 public 对战中，首只 Alakazam 出现于 simulator turn `5/7/8/5/5`，平均
turn `6.0`，中位数为 `5`。首个攻击发生于 `3/7/2/3/5`。这和 V2 报告中的样本形成
明显对比：V2 有一局在早期没有完成 Kadabra/Alakazam 线，而 V3 的 public 五局都至少
形成了一只 Alakazam。当前 V3 的主要问题已经从“能不能建立攻击线”转移到“建立后能否
识别特殊防守状态，以及如何管理两奖宝可梦”。

V3 的 60 张卡组与 V2 相同；本轮对手构筑包括：

- `86733922`：Ethan’s Typhlosion 与 Crustle/Dwebble。
- `86733250`：Mega Kangaskhan ex 与 Crustle/Dwebble，带 Mist Energy、Spiky Energy、
  Hero’s Cape 和 Handheld Fan。
- `86732591`：另一套 Alakazam/Dudunsparce，带 Nighttime Mine 与 Xerosic’s Machinations。
- `86731938`：Mega Lucario ex、Solrock、Hariyama、Lunatone 的 Fighting 构筑。
- `86731321`：Mega Abomasnow ex、Snover、Kyogre 的 Water 构筑。

## Public 对局复盘

### 86733922：快速建立并完成击倒

这是 V3 最短的 public 胜局。V3 在 turn 1 使用 Buddy-Buddy Poffin、将 Telepath
Psychic Energy 贴给 Abra，并放下 Dunsparce；turn 3 完成 Kadabra、Dudunsparce 和
Enriching Energy 的组合，同时用 Kadabra 对 Dwebble/Crustle 造成 30 伤害。

turn 5 V3 将 Kadabra 进化为 Alakazam，Powerful Hand 造成 280 伤害，直接击倒对手的
Crustle。这里的行动顺序符合 V3 目标：先建立 Abra 系列和过牌引擎，再让 Alakazam
用足够大的手牌兑现击倒，没有因为“可以攻击”而提前浪费关键进化资源。

### 86731321：首只打手被击倒后的恢复

对手是 Mega Abomasnow ex/Kyogre。V3 在 turn 3 通过 Dawn、Kadabra、Dudunsparce 和
多个 Abra 铺场；turn 5 形成 Alakazam，并完成 Hilda、Buddy-Buddy Poffin、Poké Pad
等后续资源准备。

- turn 5：Alakazam 对第一只 Mega Abomasnow ex 造成 260 伤害。
- turn 7：同一只 Alakazam 造成 280 伤害并将其击倒。
- 对手随后用 Mega Abomasnow ex 的攻击造成 600 伤害，击倒 V3 的 Active Alakazam。
- V3 的 Bench 已经有第二只 Alakazam；turn 9 贴上 Telepath Psychic Energy 后造成
  420 伤害，击倒第二只 Mega Abomasnow ex 并获胜。

这是 V3 规则“攻击线达到目标后继续保证下一只打手”的正面案例。首只 Alakazam 被击倒
并没有让 V3 重新从 Abra 开始，而是直接交接给已准备好的第二只。

### 86731938：对高 HP Fighting 场面的有效伤害

对手是 Mega Lucario ex/Solrock/Hariyama。V3 turn 3 用 Kadabra 攻击造成 30 伤害，
turn 5 进化出 Alakazam 并继续补充多只进化线与 Dudunsparce。

之后的有效攻击为：

- turn 7：Alakazam 对 Solrock 造成 400；
- turn 9：Alakazam 对 Hariyama 造成 500；
- turn 11：Alakazam 对 Mega Lucario ex 造成 540；
- turn 13：另一只 Alakazam 对 Solrock 造成 600 并结束对局。

这局说明 V3 的“手牌达到较高数量后停止无意义抽牌”的方向有效：最终获胜前 V3 牌库
剩 3 张、手牌 30 张，已经把手牌资源转化成连续击倒，没有继续拖到牌库耗尽。

### 86732591：对手 Boss’s Orders 暴露两奖目标

这是 V3 的第一场 public 败局，V3 后手。V3 的早期攻击线并非没有启动：

- turn 2：Abra 攻击对手的 Abra，造成 10；
- turn 4：另一只 Abra 攻击对手的 Alakazam，造成 10；
- turn 6：Kadabra 攻击对手的 Alakazam，造成 30；
- turn 8：Alakazam 攻击对手的 Alakazam，造成 220。

问题在于，对手先建立了自己的 Alakazam 交换线，而 V3 在 turn 6 放下了 Fezandipiti
ex。最后一个对手回合的日志显示：

1. 对手将 Kadabra 进化为 Alakazam；
2. 使用 Boss’s Orders；
3. 将 V3 Bench 上的 Fezandipiti ex 拉到 Active；
4. 用 Alakazam 造成 460 伤害将 Fezandipiti ex 击倒。

对手当时只剩 2 张 Prize，因此这个目标被拉出后直接完成两奖终结。V3 的普通进化线
虽然已经存在，但无法抵消一次高价值 Boss 击倒。后续需要把“Fezandipiti ex 是否值得
放下”与对手剩余 Prize、对手手中/场上可能的 Boss’s Orders 风险结合，而不能只看当前
是否需要补手牌。

### 86733250：未识别 Crustle/Mist Energy 的 0 伤害循环

这是本轮最明确的策略缺陷。V3 turn 7 已经完成 Alakazam，并在后续回合持续使用
Powerful Hand；但目标始终是对手 Active Crustle（card ID `345`，serial `72`），
每次 replay 的伤害结果都是 `0`：

`turn 7 / 9 / 11 / 13 / 15: Alakazam -> Crustle, damage=0`

该 Crustle 从 turn 3 起就带有 Mist Energy。仓库中的官方 engine source 对这个交互有
一个需要特别记录的实现细节：

- `CardImpl.h` 将 Alakazam 的 Powerful Hand 实现为 `EffectType::DamageCounter`；
- Mist Energy 设置 `NoEffectEnemyAttack`；
- `EffectInstant.h` 在处理 `DamageCounter` 时先调用 `state.isPreventEffect`，
  结果把伤害改成 `0`。

卡牌文本本身写着 “Damage is not an effect”，但当前 simulator 的具体实现把这条
Alakazam 的 damage-counter 攻击走进了效果免疫判断。replay 的 `damage=0` 与上述源码
路径一致，因此这不是一次普通的伤害计算偏差，而是当前运行时下必须由策略规避的交互。

V3 没有在第一次 0 伤害后改变目标，也没有用 Boss’s Orders 将非 Mist 目标拉到 Active，
而是继续攻击同一只 Crustle、继续消耗牌库。turn 19 V3 牌库变为 0，仍然有 6 张 Prize，
对手场上也没有被击倒的宝可梦，最终以牌库耗尽失败。

这局最优先的后续规则是：攻击后比较目标 HP 是否实际下降；如果 Alakazam 的预期伤害
大于 0 但 replay 状态显示没有伤害，立即标记当前目标为免疫目标，停止重复攻击，并按
以下顺序处理：

1. 若 Bench 有可被当前手牌伤害击倒且没有 Mist Energy 的目标，优先使用 Boss’s Orders；
2. 否则保留攻击手和牌库，等待能改变目标/防守状态的合法行动；
3. 在确认当前攻击不会产生有效伤害前，不再使用 Dudunsparce 等抽牌动作。

## Validation self-play

`86730455` 是 V3 对 V3 的官方 validation self-play。V3 的第一只 Alakazam 到 turn 9
才出现，之后在 turn 9、11、13、15、17 形成连续 Powerful Hand 攻击，V3 视角最终
获胜。这个结果证明提交包可以完成官方 validation，但由于双方使用同一 agent，不能用来
推断对 public 构筑的泛化强度。

## 当前结论与下一轮复盘重点

1. V3 public 为 3 胜 2 负；样本仍然很小，不能直接当作稳定胜率，但相对 V2 的 public
   样本，早期 Alakazam 建立失败已经不再是主要问题。
2. V3 的成功模式是：前期用 Telepath Energy/Poffin 建立 Abra 线，尽快形成第一只
   Alakazam，同时在 Bench 预备第二只可交接的打手。
3. 当前最大的明确缺陷是不会根据实际伤害结果识别特殊防守状态；`86733250` 中对
   Mist Energy + Crustle 的重复 0 伤害直接导致牌库耗尽。
4. 第二个风险是 Fezandipiti ex 的两奖暴露；`86732591` 说明即使普通攻击线已经
   建好，只要对手有 Boss’s Orders，错误放置 Fezandipiti 仍可能立即结束对局。
5. 下一轮建议优先从官方 replay 验证三项指标：攻击前后目标 HP 变化、对手 Active 的
   特殊能量/免疫状态、以及对手剩 2 Prize 时 Fezandipiti ex 是否已经在场。

## 可用 agent logs

V3 自己所在位置的 agent logs 已下载到各 Episode 目录：

- `86733922`：agent 0；
- `86733250`：agent 0；
- `86732591`：agent 1；
- `86731938`：agent 0；
- `86731321`：agent 1；
- `86730455`：agent 0、agent 1 均可用。

public 对手 agent logs 仍按 Kaggle 权限返回 `403 Forbidden`；本报告只使用 replay 中公开
的对手状态、卡牌和行动结果，不把缺失的对手 log 当作策略结论依据。
