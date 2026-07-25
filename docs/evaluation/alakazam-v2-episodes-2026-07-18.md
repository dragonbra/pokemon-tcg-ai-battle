# Alakazam V2 Kaggle 官方 Episode 初步复盘

## 数据范围

- Submission：`54810679`，文件 `alakazam_v2.tar.gz`。
- 提交状态：`COMPLETE`，当前 public score `611.8`。
- Kaggle 账号在 replay 中显示为 `炽月殿-ptcg糕手`。
- 原始数据目录：[`replays/kaggle_v2_54810679/`](../../../replays/kaggle_v2_54810679/)。
- 本文只分析 Kaggle 官方 Episode；本地 simulator 生成的 replay 不纳入结论。

## Episode 总览

| Episode | 类型 | 对手 | V2 位置 | 先手 | 结果 | replay steps | 主要终局状态 |
|---:|---|---|---:|---:|---|---:|---|
| [86708975](../../../replays/kaggle_v2_54810679/86708975/episode-86708975-replay.json) | validation self-play | 自己 | 0 | 0 | V2 胜 | 163 | 双方同一提交的验证对局 |
| [86709121](../../../replays/kaggle_v2_54810679/86709121/episode-86709121-replay.json) | public | Đức Nguyễn Minh | 0 | 0 | V2 胜 | 136 | V2 剩 1 Prize，对手 Active 被击倒 |
| [86709641](../../../replays/kaggle_v2_54810679/86709641/episode-86709641-replay.json) | public | princehur | 1 | 0 | V2 负 | 56 | Dunsparce 受到 180 伤害后无 Bench 可替换 |
| [86710197](../../../replays/kaggle_v2_54810679/86710197/episode-86710197-replay.json) | public | Rostislav | 0 | 0 | V2 负 | 144 | V2 牌库为 0，剩 3 Prize；对手剩 1 Prize |

Public 样本为 1 胜 2 负。`86708975` 是官方 validation self-play，不应与 public
对手池结果混为一谈。

## 三局 public 对战的关键观察

### 86709121：对 Marnie's Grimmsnarl ex 的胜局

对手初始牌组是 Marnie's Grimmsnarl ex 核心：`Marnie's Impidimp 4`、`Morgrem 3`、
`Marnie's Grimmsnarl ex 3`，并带 10 张 Basic Darkness Energy、4 张 Lillie's
Determination 和 4 张 Spikemuth Gym。

V2 的有效攻击发生在 simulator turn `5/7/9/11`，全部由 Alakazam 使用
`Powerful Hand`。最终一击击倒对手的 `Morgrem`；V2 当时 Active Alakazam 还剩
110 HP、Bench 有 2 只宝可梦、牌库剩 6 张。这个 replay 说明 V2 的攻击循环在该类
对手上确实启动了，但不能仅凭一局判断对 Marnie 牌组具有稳定优势。

### 86709641：对 Marnie's Grimmsnarl ex 的快速败局

对手同样是 Marnie's Grimmsnarl ex，但构筑更重：`Marnie's Impidimp 4`、`Morgrem 2`、
`Marnie's Grimmsnarl ex 4`，并有 10 张 Basic Darkness Energy。

V2 只在 turn `2` 用 Dunsparce 使用过一次 `Ram`，没有完成 Kadabra/Alakazam 进化。
随后对手的 Grimmsnarl ex 用 180 伤害击倒 Dunsparce；V2 没有 Bench，牌组剩 6 Prize，
因此这是一次明显的开局建立失败，而不是 Alakazam 攻击交换失败。需要重点检查：

- turn 1–2 是否把资源过多花在非进化动作上；
- 没有可用 Alakazam 时是否应保留更多 Basic/Bench 保险；
- 面对先手高能量 Grimmsnarl ex，是否需要更早处理对手 Active。

### 86710197：对 Mega Abomasnow ex / Kyogre 的牌库耗尽型败局

对手牌组非常偏水能量：33 张 Basic Water Energy、4 张 Snover、4 张 Mega Abomasnow ex、
2 张 Kyogre，并带 4 张 Team Rocket's Petrel 和 4 张 Lillie's Determination。

V2 在 turn `8/11` 用 Alakazam 攻击，在 turn `13` 改由 Dunsparce 攻击；最终 V2
牌库为 0、还剩 3 Prize，场上 Active Dudunsparce 和 Bench 仍在，对手只剩 1 Prize。
结合最后一个回合结束后 V2 不再获得行动，这局表现为 simulator 的牌库耗尽失败。

这局的首要问题不是“有没有进化成功”，而是攻击/抽牌引擎把自己的牌库消耗得过快：
需要把 `Psychic Draw`、`Dawn`、`Hilda` 以及 Alakazam 大手牌伤害之间的关系按回合重建，
确认是否存在可以少抽牌、少循环或更早结束比赛的决策。

## 当前阶段结论

1. V2 的 Alakazam 主攻线在胜局中最早于 turn 5 形成，之后可以稳定地每两回合攻击一次。
2. 两场败局分别暴露了两种不同问题：一次是 turn 2 前后的场面建立失败，一次是长局牌库管理失败。
3. 目前 public 样本只有 3 局，不能把 `1/3` 当作策略强度估计；但这 3 局已经足够确定下一轮复盘方向。
4. 后续只从官方 replay 提取这些指标：首只 Alakazam 的 turn、每回合抽牌来源、牌库数量曲线、有效攻击 turn、Prize 曲线、Active/Bench 变化和失败终止原因。

## 可用 agent logs

Kaggle 允许下载 V2 自己所在位置的 logs，已放在对应 Episode 目录：

- `86708975`：agent 0、agent 1 均可用；
- `86709121`：agent 0 可用；
- `86709641`：agent 1 可用；
- `86710197`：agent 0 可用。

对手 agent logs 由 Kaggle API 返回 `403 Forbidden`，当前只保留 replay 中公开的对手状态和行动结果。
