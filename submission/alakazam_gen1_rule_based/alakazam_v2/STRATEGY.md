# Alakazam V2 Strategy Specification

V2 以 V1 的合法选项解析和单回合伤害阈值为基础，把回合目标改为“尽可能保证
本回合攻击，并优先兑现奖赏”。规则顺序如下：

1. 若已有合法击倒，立即攻击；
2. 若攻击合法但不能击倒，攻击优先于可选抽牌；
3. 若 Active Abra/Kadabra 具备 Alakazam 和 Rare Candy，优先直接进化并争取本回合攻击；
4. 只有能把 Powerful Hand 推过击倒阈值的抽牌动作，才会排在攻击/即时进化之前；
5. 没有即时攻击计划时，再做铺场、进化线和下一只攻击手准备。

## Deck And Board Plan

- V2 用第 4 张 Alakazam 替换 Psyduck，降低无关起手目标并提高 Stage 2 命中率。
- 第一回合优先放置 Abra 和 Dunsparce，Telepath Psychic Energy 的搜索优先找 Abra。
- Buddy-Buddy Poffin 不主动找 Psyduck 或 Shaymin；这两个卡位不属于默认铺场核心。
- Alakazam (`743`) 只需要一张 Psychic Energy 就能攻击；已经有一张能量时，
  不再给同一只胡地附第二张能量。Fezandipiti ex (`140`) 仍是原卡组中的辅助
  抽牌位，不受这条限制影响。
- Shaymin 保留在卡组，但不再被搜索目标评分为核心铺场牌。

## Energy Rules

- 前期同时可选时，Telepath Psychic Energy (`19`) 优先于 Basic Psychic Energy (`5`)。
- Alakazam 攻击只需要 Psychic Energy；不为了撤退消耗攻击手的能量。
- Enriching Energy (`13`) 优先给 Dudunsparce (`66`)：它抽牌后会和能量一起洗回牌库，
  形成可循环的过牌资源。
- Alakazam 最多按一张攻击所需能量规划；若 simulator 仍提供第二张能量选项，
  策略将其作为最后合法选项。

## Boss's Orders And Retreat

- Boss's Orders (`1182`) 只有在当前 Active 不能被击倒、而换出目标能在本回合兑现
  奖赏，或换出目标能带来更高奖赏时才提高优先级。否则保留 Supporter。
- 攻击目标先选本回合能击倒的合法目标，再比较奖赏价值和剩余 HP。
- 撤退只在 Active 不能承担攻击且 Bench 有已经附着 Psychic Energy 的 Alakazam
  接力时才进入可用计划；否则选项排在结束回合之后。

## Recovery And Metrics

Night Stretcher、Lana's Aid、Sacred Ash 和 Wondrous Patch 仍按弃牌区是否能重建
攻击线决定时机，牌库接近耗尽时避免无收益的 Dudunsparce 抽牌。

V2 replay 对比应记录：首次攻击回合、每回合攻击比例、攻击击倒比例、撤退次数和
消耗能量数量、Boss's Orders 使用与击倒比例、Rare Candy 直接进化 Alakazam 次数、
首回合 Abra 数量、Enriching Energy 贴到 Dudunsparce 次数，以及 Psyduck/Fezandipiti
进入场区次数。另记录给 Alakazam 附第二张能量的次数，应为 0。
