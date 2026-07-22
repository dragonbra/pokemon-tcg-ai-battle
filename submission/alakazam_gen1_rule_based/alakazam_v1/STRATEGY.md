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

当前代码采用确定性的 action priority，并且所有动作都必须来自 simulator 的合法选项。V1 已经实现了基础的单回合手牌阈值规划：当当前攻击不能击倒 Active 时，会评估抽牌 Ability、进化、抽牌 Trainer 和 `Enriching Energy` 是否能把伤害推过阈值。

搜索效果也已经按 effect 类型处理：`Dawn` 依次找 Basic、Stage 1、Stage 2，`Hilda` 依次找 Evolution、Energy，`Buddy-Buddy Poffin` 和 `Telepath Psychic Energy` 优先找合适的 Basic Pokémon。

当前版本另外补充了三类资源连续性规则：

- `Dudunsparce` 在抽牌不会跨过击倒阈值、且牌库接近耗尽时降低优先级，避免为了短期手牌而制造牌库风险；
- `Night Stretcher`、`Lana's Aid`、`Sacred Ash` 和 `Wondrous Patch` 会结合弃牌区是否真的能重建攻击线来决定时机；
- 伤害目标和 `Boss's Orders` 都优先选择本回合可以击倒、且 Prize value 更高的合法目标。

effect 处理也覆盖了这套卡组中较容易被忽略的路径：`Wondrous Patch` 的弃牌区能量和 Bench 目标、`Enhanced Hammer` 的特殊能量目标，以及攻击效果产生的伤害目标选择。`Psyduck` 的 `Damp` 没有被默认当作抽牌或铺场动作；由于 observation 不暴露对手卡牌的 Ability 文本，V1 只把它作为低优先级的 matchup utility。

## Hand Alakazam Notes

下面把用户提供的「手牌胡地」游玩说明转译成适用于本仓库的策略语言。原说明使用了通用 Pokémon TCG 的示例；具体实现仍以本 simulator 的 `EN_Card_Data.csv` 和 observation 行为为准。

### Core Idea: Glass Cannon With Hand Resources

Alakazam 不是单纯追求“尽快进化并攻击”，而是把手牌当成攻击资源：

```text
手牌数量 -> Powerful Hand 伤害 -> 是否能击倒 Active -> 是否值得本回合攻击
```

因此，打出一张 Trainer 不能只看它当前能产生什么效果，还要考虑它会不会降低本回合的攻击伤害。V1 的基本目标是让手牌同时承担两种角色：

- setup 资源：找到 Basic、Evolution 和 Energy；
- 伤害资源：在攻击前尽可能留在手里。

### Simulator-Specific Damage Formula

外部说明中的 `30 + 20 * hand_size` 是通用示例，不适用于当前 simulator 的这张卡。官方 Card Data 对 Alakazam `743` 的 `Powerful Hand` 描述是：

> Place 2 damage counters on your opponent’s Active Pokémon for each card in your hand.

所以当前实现应使用：

```text
Powerful Hand damage = 20 * current_hand_count
```

例如 8 张手牌对应 160 damage，12 张手牌对应 240 damage。攻击需要 1 个 Psychic Energy；`Enriching Energy` 只提供 `{C}`，不能替代 Psychic Energy，但从手牌附着时会抽 4 张牌，因此它是“抽牌动作”和“能量动作”的折衷资源。

### Match Flow

#### Early Game: Build The Board

- 先建立 Abra、Dunsparce 和后续攻击线，不要为了短期动作过早消耗所有进化卡；
- 用 `Buddy-Buddy Poffin`、`Dawn`、`Hilda`、`Poké Pad` 提高找到线路的概率；
- `Telepath Psychic Energy` 附着到 Psychic Pokémon 后，可以额外从牌库找最多 2 只 Basic Psychic Pokémon，优先用于继续铺 Abra；
- Active 是否应该承担伤害需要结合当前对手攻击力决定，不能固定认为 Alakazam 一定要最晚才上 Active。

#### Mid Game: Evolve And Preserve Hand Size

- Kadabra 的 `Psychic Draw` 进化时抽 2 张，Alakazam 的 `Psychic Draw` 进化时抽 3 张；
- `Dudunsparce` 的 `Run Away Draw` 抽 3 张后会把自己和附着卡洗回牌库，因此它是一次性抽牌引擎，不是永久留在 Bench 的支援 Pokémon；
- `Rare Candy` 可以缩短 Alakazam 线路，但要比较“立即进化并抽 3 张”和“保留资源/继续铺下一只”的价值；
- 非必要时不要在攻击前打光手牌，先判断当前手牌是否已经达到击倒阈值。

#### Late Game: Reach The Knockout Threshold

攻击前按以下顺序判断：

1. 当前手牌伤害是否已经可以击倒对手 Active；
2. 如果不能，`Psychic Draw`、`Run Away Draw`、`Enriching Energy` 或其他合法动作能否把伤害推过阈值；
3. 为了抽牌而消耗的卡是否会让最终手牌反而变少；
4. 如果 Active 无法击倒，是否应该使用 `Boss's Orders` 改变目标，或继续准备下一回合。

这是一种明确的 one-turn threshold policy，适合先实现为 rule-based agent。

### Deck Cards As Strategy Modules

| Module | Cards in this deck | Role |
|---|---|---|
| Attack line | Abra, Kadabra, Alakazam | Build the Psychic attacker and convert hand size into damage |
| Draw engine | Kadabra Psychic Draw, Alakazam Psychic Draw, Dudunsparce | Increase hand size, with Dudunsparce returning to the deck |
| Setup search | Dawn, Hilda, Buddy-Buddy Poffin, Poké Pad | Find the right stage of the evolution line |
| Psychic Energy | Telepath Psychic Energy, Basic Psychic Energy | Pay the attack cost and extend the Basic Psychic board |
| Draw conversion | Enriching Energy | Draw 4 when attached, but does not pay the Psychic attack cost |
| Recovery | Night Stretcher, Lana's Aid, Sacred Ash | Rebuild Pokémon and Basic Energy after trades |
| Board protection | Battle Cage, Shaymin | Reduce Bench damage from effects or attacks under their simulator text |
| Disruption | Enhanced Hammer, Boss's Orders, Eri | Reduce opposing resources or select a better target |
| Comeback draw | Fezandipiti ex | Draw 3 after a Pokémon was Knocked Out on the previous turn |

`Psyduck` 也有 simulator-specific utility：它的 `Damp` Ability 会让场上 Pokémon 失去“需要使用者将自己击倒”的 Ability。它应被视为 matchup-dependent utility card，而不是默认攻击手。

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

## Rule-Based Feasibility

结论：**可以实现，而且很适合作为 rule-based V1；但规则 agent 的目标应是稳定执行“单回合资源规划”，不是声称求得全局最优。**

原因是这套牌的关键变量大多可以从当前 observation 直接得到：

- 自己的手牌数量和具体卡牌；
- Active/Bench 的 Pokémon、HP、Energy 和进化状态；
- 当前合法的 Play、Attach、Evolve、Ability、Attack 选项；
- 对手公开的 Active/Bench、HP、Energy 和 Prize 状态；
- 搜索效果出现时 simulator 暴露的候选卡片。

因此可以把策略拆成可测试的层次：

```text
State extractor
    -> setup / evolution planner
    -> hand-size and knockout threshold calculator
    -> target and Boss's Orders scorer
    -> resource continuity planner
    -> legal option selector
```

核心决策可以先写成如下规则：

```text
if 当前攻击可以赢得比赛:
    攻击
elif 当前 Active 可攻击且 Powerful Hand 已达到击倒阈值:
    攻击
elif 某个合法抽牌/进化动作能达到击倒阈值:
    执行该动作
elif 当前没有下一只攻击手:
    优先铺 Abra、进化线和 Psychic Energy
elif 对手 Active 无法击倒但 Bench 有高价值目标:
    评估 Boss's Orders
else:
    保留关键手牌并结束回合
```

### What Rules Can Handle Well

- 起手和 Bench 铺设优先级；
- Kadabra/Alakazam/Dudunsparce Ability 的时机；
- 精确计算当前手牌对应的 Powerful Hand 伤害；
- 在“立即攻击”和“先抽牌再攻击”之间做阈值决策；
- 根据 Active HP、Prize value 和是否可击倒选择 Boss's Orders；
- 维持下一只 Alakazam 的连续攻击能力。

### What Rules Will Approximate

- 对手手牌是隐藏的，不能准确判断对手下一回合的干扰；
- 牌库洗切后的长期抽牌结果不能完全预测；
- 多回合 Prize race 需要评估未来多个分支；
- 某些随机效果、先后手和对手策略需要概率模型或搜索；
- “现在少打 20 点、换取下一回合更大伤害”的价值不总能由单一阈值表达。

所以合理路线是先实现**分层规则 + 有界候选动作评分**。当规则稳定后，再让搜索只展开少数关键选择：先抽牌还是攻击、是否使用 Boss's Orders、是否消耗 Rare Candy，以及是否使用 Dudunsparce 抽牌。

## Evaluation Contract

每次策略改动都应至少记录：

- 对战双方 submission 名称；
- 对局数量、完成率、胜负和平均步数；
- agent exception 数量；
- 首次异常的 observation、select context 和合法 options；
- 与上一版本相比的变化。

当前已有的三局 smoke test 只用于验证运行链路，不足以评价策略强弱。

最新阈值策略 smoke test：胡地自战 114 步完成；胡地先手对官方水系 24 步完成但落败；官方水系先手对胡地 45 步完成且胡地获胜，均无异常。这里的结果仍然只是运行回归，不代表稳定胜率。

## External Reference

本节内容根据用户提供的「手牌胡地」游玩说明整理。原始分享链接为：

`https://share.google/aimode/PHwZrghJENYyMNhAE`

外部说明中的通用卡名、伤害示例和正式规则描述已按当前 simulator 的 Card Data 做了上面的校正；实现时以后者为准。
