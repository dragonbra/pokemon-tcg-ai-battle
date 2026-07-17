# Alakazam V1 Strategy Specification

这份文件是胡地 agent 的策略说明和后续迭代入口。它不参与提交打包，后续可以直接补充、讨论和修改。

## Current Baseline

V1 的目标是先建立一条可以稳定运行、容易解释的 Alakazam game loop：

```text
建立 Abra / Dunsparce
    -> 通过 Dawn / Hilda / Poffin / Poke Pad 找牌
    -> 进化 Kadabra / Alakazam
    -> 使用 Psychic Draw 和 Dudunsparce 抽牌
    -> 附着 Psychic Energy
    -> 使用 Powerful Hand 攻击
    -> 准备下一只攻击手
```

当前代码采用确定性的 action priority，并且所有动作都必须来自 simulator 的合法选项。

## Strategy Areas To Specify

### 1. Opening Setup

需要明确：

- Active Pokémon 的优先级：Abra、Dunsparce、Psyduck、其他 basic；
- Bench 上第一只和第二只 Pokémon 的优先级；
- 是否保留至少一个 Bench slot；
- Telepath Psychic Energy 触发搜索时应该找哪些 Basic Psychic Pokémon；
- 先用 Buddy-Buddy Poffin、Dawn 还是 Hilda。

### 2. Evolution Timing

需要明确：

- Kadabra Psychic Draw 与 Alakazam Psychic Draw 的使用顺序；
- 什么时候使用 Rare Candy，什么时候保留 Kadabra 路线；
- Dudunsparce 作为抽牌引擎时的进化和回收时机；
- 当前攻击手与下一只攻击手之间的准备阈值。

### 3. Hand Size And Attack

需要明确：

- Powerful Hand 的目标伤害阈值；
- 先抽牌还是先攻击；
- 哪些 Trainer、Energy 和进化卡必须保留在手牌中；
- Enriching Energy 的抽牌收益是否值得牺牲 Psychic Energy 质量。

### 4. Prize Race And Boss's Orders

需要明确：

- 是否存在本回合可完成胜利的攻击；
- Active 与 Bench 目标的 Prize value；
- Boss's Orders 的使用阈值；
- 不能击倒目标时是否改打低血量 Pokémon 或继续铺场。

### 5. Recovery And Resource Management

需要明确：

- Night Stretcher、Lana's Aid、Sacred Ash 的优先级；
- Battle Cage 的铺设时机；
- Enhanced Hammer 的目标价值；
- 牌库剩余数量较低时是否停止 Dudunsparce 抽牌。

## Evaluation Contract

每次策略改动都应至少记录：

- 对战双方 submission 名称；
- 对局数量、完成率、胜负和平均步数；
- agent exception 数量；
- 首次异常的 observation、select context 和合法 options；
- 与上一版本相比的变化。

当前已有的三局 smoke test 只用于验证运行链路，不足以评价策略强弱。
