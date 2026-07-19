# Alakazam V5 与 V5 Auto Iteration：最新十场官方 Replay 复盘

## 结论先行

这 20 场官方 public replay 支持用户的第一印象：`v5_auto_iter` 当前样本表现明显
更好，但优势更准确地描述为“攻击线完成率和持续进攻能力更好”，不是“首只 Alakazam
明显更早”。

| 指标 | `v5_auto_iter` | V5 | 观察 |
|---|---:|---:|---|
| public replay | 10 场 | 10 场 | 每个 submission 按 Kaggle 返回的最新优先顺序取 10 场 |
| 胜 / 负 | 5 / 5 | 2 / 8 | 样本胜率 50% vs 20% |
| 首只 Alakazam | 9/10，平均 turn 5.56 | 7/10，平均 turn 5.71 | 落地速度接近，auto 更少完全断线 |
| 首次任意攻击 | 10/10，平均 turn 3.40 | 7/10，平均 turn 6.86 | auto 更常先用 Kadabra/Dunsparce 进行低阶段动作 |
| 首次 Alakazam 攻击 | 5/10，平均 turn 7.60 | 3/10，平均 turn 10.00 | 条件样本很小，但 auto 的主攻兑现更常发生 |
| Abra 线攻击次数 | 平均 2.50 | 平均 1.90 | auto 的连续攻击更充分 |
| 最大 ready attacker 数 | 平均 2.00 | 平均 1.70 | 下一只打手的准备略好 |
| 终局我方牌库 | 平均 14.8 | 平均 22.1 | auto 明显更愿意把资源转换成场面和手牌 |
| 终局我方剩余 Prize | 平均 3.4 | 平均 4.7 | auto 的 Prize 推进更深 |
| 牌库耗尽败局 | 2 | 1 | auto 的进攻节奏以更高牌库风险为代价 |

这里的“首次任意攻击”不能直接当作有效进攻指标：auto 的部分败局在 turn 2–3
就用低阶段攻击，但仍然没有完成 Alakazam 主攻。因此后续评测应同时记录
`first_attack_turn`、`first_alakazam_turn` 和 `first_alakazam_attack_turn`。

当前最可信的策略判断是：auto_iter 已经较好地解决了 V5 的“场上有宝可梦、但没有
下一只真正可攻击打手”的问题；尚未解决的是长局中的牌库预算，以及面对墙/内战时
如何把早期攻击转换成 Prize，而不是只增加攻击次数。

## 数据范围与可比性

- V5 submission：`54820045`，当前 public score `554.2`。
- V5 auto iteration submission：`54820051`，当前 public score `647.7`。
- 两份 submission 均为 `COMPLETE`；public score 是完整 Kaggle 评测结果，不是下面
  10 场样本的胜率。
- 本报告每边选取 Kaggle `competitions episodes` 返回的最新 10 场 public episode，
  排除 validation self-play；因此这是“最新十场”，不是按时间从最早开始的十场，
  也不是两边一一配对的 head-to-head。
- 两边卡组的 60 张卡多重集完全相同，但 `deck.csv` 的卡牌顺序不同。这个顺序会
  影响确定性抽牌，因此本轮结果不能纯粹归因于 `main.py` 策略改动；准确说是
  “策略代码 + 初始 deck 顺序 + 不同对手样本”的综合比较。

原始 replay：

- [`v5_auto_iter` 10 场](../../../replays/kaggle_v5_auto_iter_54820051/)
- [`V5` 10 场](../../../replays/kaggle_v5_54820045/)
- 结构化逐局指标：[`games.json`](games.json)
- 聚合指标：[`summary.json`](summary.json)

Prize 卡牌内容在 Kaggle replay 中被脱敏成 `null`，本报告只使用 Prize 槽位数量，
不尝试推断具体 Prize 卡。

## 聚合结果怎么理解

### 1. auto 的核心提升不是把 Alakazam 做得更快，而是减少“完全没做出来”

两边首只 Alakazam 的条件平均几乎相同：auto 为 turn 5.56，V5 为 turn 5.71；中位数
都为 turn 5。真正的差别是 auto 10 场中有 9 场做出 Alakazam，V5 只有 7 场。

这与 auto 的代码改动相符：

- `Buddy-Buddy Poffin` 在攻击线少于 3 只，或 ready attacker 少于 2 只时继续保留
  Abra，而不是过早转向 Dunsparce；
- 对 Psychic 攻击线的能量目标增加了“已有一张 Psychic Energy 就不要重复贴”的
  限制；
- 增加了多类 Mega/ex 对手的 Prize value，且加了 Iono 场景下的定向 Boss 规则；
- 新局开始时清空 effect serial 状态，避免 evaluator 复用 module 时上一局状态泄漏。

在这 20 场里，auto 的确更常出现第二只可攻击线：最大 ready attacker 平均为 2.0，
V5 为 1.7；Abra 线实际攻击次数平均为 2.5 对 1.9。

### 2. auto 的资源策略更主动，但 draw guard 已经进入需要重新校准的区域

auto 的终局牌库平均少 7.3 张，说明它不是单纯“更幸运地摸到 Alakazam”，而是
更积极地使用 Poffin、Poké Pad、Rare Candy 和攻击线资源。10 场累计的关键卡牌
使用次数如下：

| 卡牌 | auto | V5 | 解释 |
|---|---:|---:|---|
| Buddy-Buddy Poffin | 16 | 14 | auto 更积极铺 Basic 与攻击线 |
| Poké Pad | 20 | 14 | auto 更常把搜索机会转成具体 setup |
| Rare Candy | 8 | 5 | auto 更常兑现快速 Alakazam 路线 |
| Dawn | 16 | 19 | V5 更常依赖 Supporter 过牌/搜索，但未必转换成攻击 |
| Xerosic’s Machinations | 5 | 4 | auto 对内战/大手牌对手略多施加手牌压力 |
| Enhanced Hammer | 6 | 2 | 对特殊能量墙的处理更积极 |

代价也已在 replay 中出现：auto 有 2 场牌库耗尽败局，分别是
`86817907` 和 `86828026`。这不是简单地把 draw guard 再收紧就能解决，因为
`86828026` 终局已经有两只 ready attacker，却在主攻线太晚、Prize 仍为 6 的情况下
把牌库抽空；如果只保护牌库而不提高早期击倒转化率，可能重新回到 V5 的节奏问题。

## V5 Auto Iteration：10 场逐局表

表中 `首攻` 是第一次选择任意攻击，`首 Alakazam 攻` 只统计 Active Alakazam
真正选择攻击的回合；`终局牌库 / Prize` 是我方终局剩余数量。

| Episode | 对手与构筑线索 | 结果 | 首 Alakazam | 首攻 | 首 Alakazam 攻 | Abra 线攻 | 终局牌库 / Prize | 复盘重点 |
|---:|---|:---:|---:|---:|---:|---:|---:|---|
| [86817353](../../../replays/kaggle_v5_auto_iter_54820051/episode-86817353-replay.json) | MBOOK：Mega Lucario / Solrock | 胜 | 4 | 4 | 8 | 2 | 17 / 2 | turn 4 已建立；终局对手 Active 为空，但仍有 Bench，非标准 6 Prize 终结 |
| [86817907](../../../replays/kaggle_v5_auto_iter_54820051/episode-86817907-replay.json) | jam yann：Mist/Spiky Energy 的 Crustle / Mega Kangaskhan | 负 | 5 | 3 | 7 | 4 | 0 / 2 | 有两只 ready attacker，仍在 turn 13 牌库耗尽；是最直接的 draw/Prize 转化失败 |
| [86818454](../../../replays/kaggle_v5_auto_iter_54820051/episode-86818454-replay.json) | Gregers Rygg：Dragapult | 胜 | 5 | 7 | 7 | 1 | 23 / 1 | turn 7 才主攻，但终局对手 Active 为空；说明不是所有胜局都靠高速攻击 |
| [86818982](../../../replays/kaggle_v5_auto_iter_54820051/episode-86818982-replay.json) | PokePoke-777：Alakazam mirror | 负 | 6 | 2 | — | 4 | 25 / 5 | 低阶段攻击早，但最终 Active Shaymin、无 ready attacker，Prize race 5–1 失败 |
| [86819534](../../../replays/kaggle_v5_auto_iter_54820051/episode-86819534-replay.json) | treo_hihihi：Alakazam mirror | 胜 | 6 | 2 | — | 5 | 0 / 4 | turn 18 双方牌库都到 0，属于极限资源局，不应当作普通 Prize race 样本 |
| [86820080](../../../replays/kaggle_v5_auto_iter_54820051/episode-86820080-replay.json) | hoodrichpirobo：Alakazam mirror | 负 | — | 2 | — | 3 | 30 / 6 | 没有完成 Alakazam；最终 Active Kadabra、无 Bench，属于 setup 失败 |
| [86820570](../../../replays/kaggle_v5_auto_iter_54820051/episode-86820570-replay.json) | kunihiro：Marnie’s Grimmsnarl | 负 | 4 | 2 | — | 2 | 28 / 4 | 虽然 turn 4 有 Alakazam，终局退回 Dunsparce、ready=0，对手已到 1 Prize |
| [86821266](../../../replays/kaggle_v5_auto_iter_54820051/episode-86821266-replay.json) | David Tan：Marnie’s Grimmsnarl | 胜 | 6 | 6 | — | 1 | 13 / 2 | 只记录到一次主攻，终局对手 Active 为空；说明 matchup 结果受终局形态影响 |
| [86824177](../../../replays/kaggle_v5_auto_iter_54820051/episode-86824177-replay.json) | madoka1111：Crustle / Mega Kangaskhan | 胜 | 5 | 3 | 5 | 2 | 12 / 2 | turn 3 先低阶段攻击，turn 5 兑现 Alakazam，并保留 2 个 ready attacker |
| [86828026](../../../replays/kaggle_v5_auto_iter_54820051/episode-86828026-replay.json) | Water Xiao：Great Tusk / Crustle / Land Collapse | 负 | 9 | 3 | 11 | 1 | 0 / 6 | Alakazam 太晚；终局虽有两只 ready attacker，但没有拿 Prize，牌库先耗尽 |

### Auto 的三类失败

1. **牌库先耗尽：86817907、86828026。** 前者已经能连续攻击，但 turn 13 仍有
   2 Prize；后者更严重，turn 9 才落地 Alakazam，最终 6 Prize 未动。这里的优先级
   应是缩短“setup 到第一枚 Prize”的时间，而不是单纯再增加抽牌保护。
2. **Prize race 落后：86818982、86820570。** 这两场不是完全没有 Alakazam，
   但终局 ready attacker 断掉，且对手分别到 1 Prize。需要复盘“上一只打手受伤/被
   击倒后，下一只是否已经有能量和合法攻击路径”，而不是只看场上 Alakazam 数量。
3. **完全 setup 失败：86820080。** 这是 auto 仍保留的 V5 型失败：早期可以有
   Kadabra 攻击，但没有把回合转化成 Alakazam。

## V5：10 场对照表

| Episode | 对手与构筑线索 | 结果 | 首 Alakazam | 首攻 | 首 Alakazam 攻 | Abra 线攻 | 终局牌库 / Prize | 复盘重点 |
|---:|---|:---:|---:|---:|---:|---:|---:|---|
| [86816242](../../../replays/kaggle_v5_54820045/episode-86816242-replay.json) | Abhi：Marnie’s Grimmsnarl | 负 | — | — | — | 0 | 43 / 6 | 完全没有攻击线，且对手已拿到 3 Prize |
| [86816792](../../../replays/kaggle_v5_54820045/episode-86816792-replay.json) | PaShm：Crustle / Mega Kangaskhan | 负 | 5 | 7 | — | 1 | 14 / 5 | 建立慢，只有一次 Abra 线攻击，对手到 1 Prize |
| [86817359](../../../replays/kaggle_v5_54820045/episode-86817359-replay.json) | Dino Xie：Duraludon / Archaludon ex | 胜 | 5 | 3 | 5 | 3 | 23 / 6 | 终局对手 Active 为空、Prize 未推进，属于非标准胜局 |
| [86817911](../../../replays/kaggle_v5_54820045/episode-86817911-replay.json) | Hawk：Dragapult | 负 | 3 | 17 | 17 | 2 | 17 / 2 | Alakazam 虽早落地，但首攻拖到 turn 17，主攻兑现严重失败 |
| [86818453](../../../replays/kaggle_v5_54820045/episode-86818453-replay.json) | 中村文彌：Mega Abomasnow / Kyogre | 胜 | 6 | — | — | 0 | 29 / 6 | 对手终局 Active/Bench 为空，非标准 6 Prize 胜局 |
| [86818992](../../../replays/kaggle_v5_54820045/episode-86818992-replay.json) | half pizza：Mega Froslass / Mega Starmie | 负 | 9 | 5 | — | 4 | 13 / 6 | setup 和主攻都晚，对手到 1 Prize |
| [86819545](../../../replays/kaggle_v5_54820045/episode-86819545-replay.json) | Ben Desprets：Mega Lucario / Solrock | 负 | — | 4 | — | 3 | 21 / 5 | 没有形成可验证的 Alakazam 主攻，Prize race 5–1 |
| [86820104](../../../replays/kaggle_v5_54820045/episode-86820104-replay.json) | Climber513：Crustle / Mega Kangaskhan | 负 | 8 | 8 | 8 | 3 | 0 / 2 | 牌库耗尽，仍有 2 Prize；setup 太晚 |
| [86820648](../../../replays/kaggle_v5_54820045/episode-86820648-replay.json) | Mingzhengxuan Wu：Mega Lucario / Fighting | 负 | — | — | — | 0 | 44 / 6 | 早期完全没有 Alakazam 或攻击，属于开局失败 |
| [86830447](../../../replays/kaggle_v5_54820045/episode-86830447-replay.json) | what peter：Mega Abomasnow / Kyogre | 负 | 4 | 4 | — | 3 | 17 / 3 | 有攻击线但只推进到对手 1 Prize，未形成连续击倒 |

## 对 V5 Auto Iteration 的策略判断

### 值得保留

- **Poffin 的攻击线保护。** `field count < 3` 或 `ready attacker < 2` 时继续拿
  Abra，这个方向在样本中与“auto 更少完全 setup 失败、ready attacker 更多”一致。
- **能量不重复贴。** auto 仍然需要继续检查特殊能量和撤退场景，但从策略逻辑上，
  给已经有 Psychic Energy 的 Abra 线重复贴能量的风险已降低。
- **自然进化和 Rare Candy 的混合。** auto 的 Rare Candy 使用次数更多，但并没有
  把所有回合都锁死在 Rare Candy + Alakazam；这比 V5 的某些“等直通路线”更适合
  在真实对局中维持攻击节奏。
- **按 Prize value 处理 Boss。** 对 Mega/ex 和 Iono 场景的显式判断是合理方向，
  但需要更多同类 matchup replay 才能判断是否会误用 Boss。

### 需要重点复盘和可能收紧

- **Dudunsparce draw guard。** auto 当前把额外限制放宽到牌库只剩 1 张才阻止，
  与两场 deck-out 同时出现。不能直接断言这是唯一原因，但应对
  `86817907`、`86828026` 重建每一次抽牌前后的 deck/prize 曲线，确认是否有“本回合
  已经没有可赢线路，却仍继续抽牌”的动作。
- **低阶段攻击的价值门槛。** auto 的败局中首次攻击平均反而很早，因为低阶段攻击
  在 turn 2–3 就被选出；但 `86820080`、`86820570`、`86828026` 说明“攻击过”不等于
  “攻击推进了 Prize”。Kadabra/Dunsparce 攻击应明确回答：是否能击倒、是否保护
  下一只 Alakazam、是否避免被动丢两奖；否则应优先抽到主攻或结束回合。
- **连续攻击需要从“数量”升级为“交接路径”。** `max_ready_attackers=2` 仍可能
  在 `86828026` 中输掉。下一版指标应记录每回合：Active 是否能攻、Bench 是否有
  一张 Psychic Energy、是否有合法进化路径、对手下一回合是否能取两奖。
- **deck 顺序影响必须单独隔离。** 两个 deck 的卡牌计数相同但顺序不同；如果要
  证明是策略代码带来的提升，最好固定同一 `deck.csv` 顺序，只替换 `main.py` 再做
  同一对手矩阵，或者至少做同一对手、同一先后手的 paired replay。

## 下一轮建议

我建议暂时不要因为这 10 场就继续放宽抽牌，也不要立刻回退 auto_iter。下一步按
以下顺序更有价值：

1. 用 `86817907` 和 `86828026` 做逐回合 trace，标出每次 Dawn/Hilda/Dudunsparce/
   Alakazam draw 后的牌库、Prize、ready attacker 和可击倒目标。
2. 对 `86820080`、`86820570`、`86818982` 做“低阶段攻击 vs 保留动作”复盘，确认
   30 点 Kadabra 或 Dunsparce 攻击是否真的改变 Prize race。
3. 固定 V5 的 deck 顺序，分别跑 V5 与 auto_iter 的同一对手、同一先后手对照；只有
   在 paired 结果仍然领先时，才能把本轮提升更有把握地归因到策略。
4. 对 Dragapult、Crustle wall、Alakazam mirror 分别补取 10–20 场，而不是把不同
   对手混成一个小样本胜率。当前 auto 的五场胜局中有多场终局 Active 为空，先确认
   这些终局的规则原因再进行 matchup 结论。

本报告只使用 Kaggle 官方 replay；本地迭代目录中的 480 局矩阵和 self-play 结果
作为实现背景保留，但没有混入上述 20 场 public replay 的胜率计算。
