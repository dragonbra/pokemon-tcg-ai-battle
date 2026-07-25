# Alakazam V6 最新十场官方 Replay 复盘

> **Historical paths:** The `submission/alakazam_v6/` package referenced below was retired during the 2026-07-26 repository cleanup. Commands and paths are preserved only as historical evidence.


## 结论先行

本报告分析最新 V6 最终提交 `54829503` 的 10 场 public episode，排除 validation
episode `86874383`。提交的 public score 为 `668.3`。

表面结果是 6 胜 4 负，但这 6 场胜局没有一场以我方 Prize 归零结束；多场是在
对手 Active 为空、对手没有继续选择或对局提前终止时结束。因此这批 replay 可以
证明 V6 的合法动作和铺场能力有所改善，但不能把 `60%` 当作正常 Prize race 胜率。

这 4 场败局暴露出四个更具体的问题：

1. **搜索了进化卡，却没有形成合法的 Abra 路线。** Alakazam mirror
   `86881221` 在早期多次拿到 Alakazam/Kadabra，却直到后期才补 Abra，最终只有
   Kadabra 攻击。
2. **非 KO 的低阶段攻击没有价值门槛。** Mega Venusaur 对局 `86877253` 中，
   Abra、Kadabra、Alakazam、Kadabra、Dunsparce 依次攻击，五次都没有拿 Prize，
   对手却推进到只剩 1 张 Prize。
3. **手牌被压低后仍然用 Alakazam 弱攻。** Dragapult 对局 `86880094` 的两个
   后期攻击发生在我方手牌 4 张和 2 张时，对方 Dragapult 仍有 240 和 200 HP，
   这两次攻击没有改变 Prize race。
4. **特殊 matchup 的伤害判断仍不完整。** Team Rocket 对局 `86881791` 中，
   t4 的 Articuno 被处理掉，但 t14/t16 对 Team Rocket's Mewtwo ex 的攻击没有
   继续拿 Prize；replay 日志里的 HP change 为 0。这里需要区分引擎语义和策略
   伤害估计，不能直接把它归咎于普通 setup 失败。

## 数据口径

- Submission：`54829503`，文件为 `alakazam_v6.tar.gz`。
- Public episodes：`86881791`、`86881221`、`86880664`、`86880094`、
  `86879518`、`86878963`、`86878402`、`86877819`、`86877253`、`86876715`。
- 原始官方 replay：[`replays/kaggle_v6_54829503/`](../../../../replays/kaggle_v6_54829503/)。
- 结构化逐局指标：[`games.json`](games.json)。
- 逐事件时间线：[`timeline.json`](timeline.json)。
- 聚合指标：[`summary.json`](summary.json)。

动作事实以 replay 中我方 `playerIndex` 的去重 `LogType.ATTACK`、`PLAY`、
`ATTACH`、`EVOLVE` 和 `SWITCH` 事件为准。旧 analyzer 主要依据 selected action
解析，在 Kaggle replay 的一组 step 同时包含双方状态时会漏记攻击；本报告保留旧
指标作背景，但逐局攻击次数和攻击回合以 `timeline.json` 为准。

## 聚合指标

| 指标 | V6 最新十场 | 解释 |
|---|---:|---|
| 胜 / 负 | 6 / 4 | 不是标准 Prize race 胜率 |
| 首只 Alakazam | 9/10，平均 turn 4.44 | setup 速度明显不差 |
| 官方攻击日志中的首次攻击 | 10/10，平均 turn 5.10 | 包括低阶段攻击 |
| 首次 Alakazam 攻击 | 9/10，平均 turn 5.22 | mirror 败局没有 Alakazam |
| 我方攻击事件 | 30 次，平均 3.0/局 | 攻击数量不等于有效 Prize 推进 |
| 最大 ready attacker | 平均 2.2 | 接力数量已经出现，但路径质量不足 |
| 终局牌库 | 平均 15.5 张 | 仍有 1 场牌库归零的胜局 |
| 终局我方 Prize | 平均 3.5 张 | 10 场没有一场我方 Prize 归零 |

最重要的事实是：V6 的主要问题已经不是“完全做不出 Alakazam”，而是**在做出
Alakazam 后，是否知道这一次攻击能否改变胜负状态**。

## 逐局表

| Episode | 对手构筑 | 结果 | 首只 Alakazam | 我方攻击回合 | 终局我方 Prize / 牌库 | 复盘标签 |
|---:|---|:---:|---:|---|---:|---|
| [86881791](../../../../replays/kaggle_v6_54829503/episode-86881791-replay.json) | Team Rocket's Mewtwo / Spidops | 负 | 3 | 4, 14, 16 | 5 / 10 | 高 HP 特殊 matchup，后续攻击未继续取奖 |
| [86881221](../../../../replays/kaggle_v6_54829503/episode-86881221-replay.json) | Alakazam mirror | 负 | 未完成 | 19, 21 | 6 / 13 | 搜索顺序没有形成 Basic 路线 |
| [86880664](../../../../replays/kaggle_v6_54829503/episode-86880664-replay.json) | Mega Lucario | 胜 | 3 | 4, 6, 7, 9, 11 | 2 / 3 | 多次主攻，但终局对手 Active 为空 |
| [86880094](../../../../replays/kaggle_v6_54829503/episode-86880094-replay.json) | Dragapult | 负 | 7 | 7, 12, 14 | 3 / 24 | Iono/手牌压低后仍弱攻 |
| [86879518](../../../../replays/kaggle_v6_54829503/episode-86879518-replay.json) | Dragapult | 胜 | 7 | 3, 5, 7, 17 | 1 / 13 | Abra 线连续攻击，终局对手 Active 为空 |
| [86878963](../../../../replays/kaggle_v6_54829503/episode-86878963-replay.json) | Mega Starmie | 胜 | 3 | 3 | 4 / 31 | 对手 Bench 清空后提前结束 |
| [86878402](../../../../replays/kaggle_v6_54829503/episode-86878402-replay.json) | Crustle wall | 胜 | 3 | 3 | 6 / 40 | 极早终止，没有形成 Prize 结论 |
| [86877819](../../../../replays/kaggle_v6_54829503/episode-86877819-replay.json) | Mega Kangaskhan / Crustle | 胜 | 6 | 2, 5, 6 | 1 / 0 | 牌库归零但对手终局 Active 为空 |
| [86877253](../../../../replays/kaggle_v6_54829503/episode-86877253-replay.json) | Mega Venusaur / Ogerpon | 负 | 4 | 2, 5, 7, 9, 11 | 6 / 16 | 五次非 KO 攻击，没有 Prize 推进 |
| [86876715](../../../../replays/kaggle_v6_54829503/episode-86876715-replay.json) | Mega Lucario | 胜 | 4 | 4, 9, 10 | 1 / 5 | 有主攻和接力，但对手 Active 为空终局 |

## 重点败局一：86881221，mirror 中路线来源错误

### Replay 事实

这局的关键时间线是：

- t2：使用 Hilda，日志显示把 Alakazam 和 Telepath Psychic Energy 从搜索区拿出；
  随后用 Poké Pad、贴能量并铺 Dunsparce。
- t4：再次使用 Hilda，继续拿 Alakazam/Telepath 路线，进化 Dudunsparce，
  但仍没有形成可用的 Abra -> Kadabra -> Alakazam 线路。
- t14：才通过 Dawn 同时处理 Abra、Kadabra、Alakazam。
- t16/t18：开始把 Abra 放到 Bench 并进化。
- t19、t21：最终只有 Kadabra 使用 `Super Psy Bolt`，没有 Alakazam 攻击。

终局我方 6 Prize、对手 2 Prize，牌库还剩 13 张。这不是牌库保护造成的输局，而是
**前期把 Supporter 和搜索效果转换成了 Stage 1/Stage 2 卡，却没有把 Basic 放到
一条能在未来两回合完成的线路上**。

### 为什么 V6 会这样做

当前 Hilda 的搜索分类在
`main.py:1160-1171`（已退役历史源包）；当 Active
不是 Abra、没有现成直接进化路线、攻击线也没有完成时，默认目标是整个
`EVOLUTION` 集合，而不是一个有来源的 Basic/Stage 1 路线。这样“拿到 Alakazam”在
牌面上看起来像 setup，实际上不能落场。

这与宝可梦 TCG 的进化规则直接冲突：Stage 2 在手上并不等于本回合或下一回合有
攻击者。V6 的 `_recovery_needs()` 在
`main.py:834-851`（已退役历史源包） 也有相同的
抽象问题：只要手牌出现任意 Abra 线卡，就可能被当成已有可用路线；Stage 1/Stage 2
单卡不能替代 Basic 来源。

### V7 方向

搜索不应返回“卡牌类别”，而应返回一个带来源的 `AttackRoute`：

```text
Basic 在场/可检索
  -> 下一回合可进化的 Stage 1
  -> 有 Rare Candy 或自然进化来源的 Stage 2
  -> Psychic Energy 来源
  -> 预计首次攻击回合
```

如果没有 Basic 在场或可从 deck/discard 合法恢复，Hilda/Dawn 不应优先拿单独的
Alakazam。只有 Active Kadabra 已经合法存在时，拿 Alakazam 才是完整路线。

## 重点败局二：86877253，攻击动作没有收益门槛

### Replay 事实

对手是 Mega Venusaur / Teal Mask Ogerpon，Active 是高 HP 目标，Bench 也有
多只高 HP 宝可梦。我们的攻击日志如下：

| 回合 | Active | 目标状态 | 结果 |
|---:|---|---|---|
| t2 | Abra | Ogerpon，约 300 HP | 10 点，未 KO |
| t5 | Kadabra | Ogerpon，约 270 HP | 30 点，未 KO |
| t7 | Alakazam | Ogerpon，约 150 HP | 120 点，未 KO |
| t9 | Kadabra | Mega Venusaur，约 350 HP | 30 点，未 KO |
| t11 | Dunsparce | Mega Venusaur，约 360 HP | 20 点，未 KO |

这 5 次攻击没有让我方 Prize 减少一次；对手推进到 1 Prize，我方仍为 6 Prize。

### 为什么 V6 会这样做

`preparation_is_due()` 会先处理进化和抽牌，但当这些动作都不被判定为“当前必须
做”时，攻击分支在
`main.py:1803-1819`（已退役历史源包） 仍然给普通
攻击一个固定的可接受分数。它没有区分：

- Alakazam 对无法立即 KO 的目标进行一次正常攻击；
- Kadabra/Abra 作为最后手段进行攻击；
- Dunsparce 的 Ram 或 Trading Places 是否真正产生收益。

因此策略把“有合法 attack option”近似成“应该 attack”。这正是用户之前强调的
“第一阶段攻击只是最后手段”没有完整落到全局排序里的地方。

### V7 方向

建议把攻击分为三类，而不是一个 `attack_is_legal`：

1. `guaranteed_prize_attack`：当前攻击确定 KO，或 Boss 后确定 KO，最高优先级。
2. `alakazam_pressure_attack`：Active Alakazam 已准备好、没有更高价值准备动作，
   即使不能 KO 也可以攻击，符合当前卡组特性。
3. `fallback_attack`：Abra/Kadabra 的非 KO 攻击只能在本回合没有合法准备、没有
   抽牌/恢复/接力动作且不攻击就完全失去节奏时使用。

V7 的默认边界应当是：Dunsparce 不因为有一张能量就获得攻击安全性；Trading Places
永不选择，Ram 也只在能确定 KO 时允许。这个规则要同时写进攻击排序和牌库保护的
`attack_security`，不能只在攻击 option 分支里禁用一个 attack id。

## 重点败局三：86880094，手牌压缩后的 Alakazam 弱攻

### Replay 事实

这是 Dragapult 对局。t7 已经形成第一只 Alakazam，但对手的手牌压缩/回合效果后，
我方后续两个攻击节点变成：

- t12：Active Alakazam，手牌 4 张，对方 Dragapult 还剩约 240 HP；
- t14：Active Alakazam，手牌 2 张，对方 Dragapult 还剩约 200 HP。

两次攻击都无法 KO。终局我方 3 Prize、对手 2 Prize，说明“有 Alakazam”没有
转换成领先的 Prize race。

### 为什么 V6 会这样做

V6 的 `attack_security` 主要回答“有没有带 Psychic Energy 的可攻击宝可梦”，
但没有回答“这次攻击是否达到对当前目标有意义的伤害”。在手牌被压到 4/2 张后，
只要准备 gate 没有阻止，Alakazam 仍被当成正常主攻者。

这局也说明“Alakazam 可以在没有合适目标时正常攻击”的规则需要一个边界：正常
攻击是可以的，但如果目标是 320 HP Dragapult、手牌只有 2 张、攻击不会 KO 且
本回合还有恢复/过牌/接力路线，就不应把普通攻击当作唯一优先级。

### V7 方向

保留 Alakazam 的正常攻击倾向，但增加 `attack_value`：

- 预计 KO：Prize value；
- 预计造成可在下回合兑现的伤害：保留一定价值；
- 目标 HP 高、手牌低、不能 KO，且存在恢复或接力动作：攻击降为可选末端；
- 对手下一回合可稳定取两/三 Prize 时，先准备下一只攻击者或恢复手牌。

这不是简单地把 Alakazam 也禁止攻击，而是把“可以攻击”和“现在应该攻击”分开。

## 重点败局四：86881791，Team Rocket matchup 的伤害模型待确认

### Replay 事实

- t3 已经完成第一只 Alakazam，t4 对 Team Rocket's Articuno 发起攻击；
  对手随后失去一张 Prize，说明第一击完成了实际 Prize 推进。
- t14 和 t16 再次使用 Alakazam 攻击 Team Rocket's Mewtwo ex。两次攻击时我方
  手牌分别为 16 和 18，目标仍是 280 HP 的 Mewtwo ex。
- 对应 replay 日志中，攻击后的 `HP_CHANGE` 对 Mewtwo 的 value 是 `0`，且没有
  看到对应的 KO/Prize 推进；终局我方 5 Prize、对手 1 Prize。

### 当前能确认和不能确认的部分

能确认的是：V6 把这两个攻击当成了普通可接受攻击，代码没有从 replay 状态中得到
一个“这次攻击必然无效”的 target-specific guard。不能仅凭这份 observation 断言
具体阻挡原因，因为目标身上的卡是 Team Rocket's Energy，和普通 Mist Energy 的
识别并不相同；需要继续核对官方引擎对该卡/该攻击的日志语义。

当前 `_attack_damage_against()` 只在目标附有 Mist Energy 时返回 0，见
`main.py:1028-1039`（已退役历史源包）。因此这里至少
需要补一项 V7 研究：把“策略估算伤害”和“官方引擎实际伤害/效果”在 replay 中逐击
对齐，尤其是 Team Rocket Energy、特殊能量和防御类 Ability。

## 牌库保护和 Dunsparce 的交叉问题

本批没有牌库耗尽败局，但 `86877819` 以牌库 0 结束仍获得胜利，说明牌库线已经
非常接近终局边界。V6 的静态逻辑还有两个风险：

- `main.py:442-450`（已退役历史源包） 以
  `ATTACK_ENERGY_COUNT[DUNSPARCE] = 1` 把有一张能量的 Dunsparce 算作 ready
  attacker；但 V6 已经禁止 Trading Places，Ram 也不应默认作为攻击安全性。
- `main.py:827-830`（已退役历史源包） 的 15 张 watch
  gate 依赖 `_attack_security()`。如果 Dunsparce 被计入安全攻击者，牌库低于 15
  张时可能继续抽牌，而实际没有能拿 Prize 的路线。

因此 V7 应把 `attack_security` 改成“可在未来有限回合形成有效攻击/Prize 的路线”，
而不是“模拟器当前给了一个 attack option”。

## V7 重构建议

### 1. 路线状态优先于卡牌类别

记录每条攻击路线的来源和时钟：

```text
Basic 来源 -> Stage 1 来源 -> Stage 2 来源 -> Psychic Energy -> Active/Bench -> 首次可攻回合
```

Hilda、Dawn、Night Stretcher、Lana's Aid 和 Poké Pad 都应选择能填补路线缺口的
资源，而不是独立地偏好 Abra/Kadabra/Alakazam 某个 ID。

### 2. 把行动价值和合法性拆开

合法性只回答“模拟器允许不允许”；策略价值还需要回答：

- 是否造成确定 Prize；
- 是否完成下一只打手；
- 是否增加下一回合可用的手牌/能量；
- 是否消耗本回合 Supporter、手填能量或攻击机会；
- 是否让牌库越过 15/10 张保护线。

### 3. 统一低牌库状态机

建议使用三个状态：

- `normal`：牌库大于 15，正常把过牌转成场面资源；
- `watch`：牌库 10-15，只允许能立即形成攻击/恢复/Prize 路线的过牌；
- `terminal`：牌库不超过 10，只允许能在本回合闭合最后 Prize 的过牌，且
  Dunsparce 不作为默认安全出口。

### 4. 把异常终局从胜率中分离

后续报告应同时记录：

- `prize_win`：我方 Prize 归零；
- `opponent_no_active`：对手 Active 为空且对局结束；
- `opponent_early_stop`：对手尚有资源/Bench，但 episode 提前结束；
- `deck_out`：牌库归零；
- `loss_prize_race`：正常对局中 Prize 落后。

本批 6 胜都不是 `prize_win`，所以 V7 比较时不能只看 Kaggle episode reward。

## 需要和用户确认的设计问题

1. Abra/Kadabra 的非 KO 攻击是否只在“本回合没有任何合法准备动作”时允许？本报告
   暂按“最后手段”理解。
2. Dunsparce 的 Ram 是否和 Trading Places 一样默认禁用，只在确定 KO 时例外？
3. `86881791` 中 Team Rocket's Mewtwo ex 的两次攻击为何记录为 `HP_CHANGE=0`？
   这是卡牌/引擎效果，还是 replay step 的观察时序问题？
4. 对手 Active 为空但 Bench 仍有宝可梦的终局，后续统计是否统一视作异常/对手侧
   终止，不计入策略胜率？
5. Hilda/Dawn 在没有合法 Basic 来源时是否应绝对禁止检索单独的 Alakazam/Kadabra？

这些问题确认后，V7 可以直接围绕路线状态、攻击收益和终局分类实现，而不需要
修改固定 `deck.csv`。
