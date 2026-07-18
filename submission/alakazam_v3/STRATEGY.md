# Alakazam V3 Strategy Specification

> 这是 V3 的完整策略状态与目标行为说明。当前 V3 的 `main.py` 仍与 V2 相同，
> 因此本文件中标为“V3 目标”的规则尚未实现；它们来自官方 replay 复盘和
> `IMPROVEMENT.md` 的讨论结果，待确认后再修改代码。

## 1. 策略目标与状态边界

V3 仍然以合法选项解析和 Alakazam 的手牌伤害为基础，但不再把“能攻击”简单等同于
“现在立刻攻击”。每个回合需要同时判断：

1. 能否在本回合直接取得 Prize；
2. 本回合和下回合的 Abra 系列攻击线是否安全；
3. Dunsparce/Dudunsparce 循环过牌线是否已经建立；
4. 当前手牌和牌库是否已经足够，继续抽牌是否会增加牌库耗尽风险。

V2 已继承的合法选项、确定性和卡组结构保持不变；V3 的目标是修正行动顺序和资源
管理，不是重新引入复杂的隐藏信息推断。

## 2. 卡组与场面结构

- V3 沿用 V2 的卡组：第 4 张 Alakazam (`743`) 替换 Psyduck (`858`)。
- 前期优先让场上实际存在三只 Abra 系列宝可梦，成员可以是 Abra (`741`)、Kadabra (`742`)
  或 Alakazam (`743`)。这个计数只看 Active 和 Bench 合计后的实际数量，不要求特定的
  Active/Bench 分布。
- 场上达到三只 Abra 系列宝可梦后，再主动扩大 Dunsparce (`305`) / Dudunsparce (`66`)
  的数量。Dunsparce 系列原则上多多益善，是主要的循环过牌引擎。
- 不为了刻意留空 Bench 限制普通宝可梦铺场；Fezandipiti ex (`140`) 是例外，默认不
  随意放下。
- Fezandipiti ex 只有在上一回合已经发生昏厥，且根据当前手牌数量和 Alakazam 发动攻击时的手牌数 ×20
  伤害计算，确实需要补充手牌才能达到目标伤害时才考虑使用。“缺三张”不是固定阈值，而是由实际伤害缺口决定。

## 3. 回合优先级

### 3.1 直接兑现 Prize

- 若已有合法击倒，立即攻击。
- 若 Active 无法攻击但一次明确的 Retreat 或 Dunsparce Trading Places 能让准备好的
  Alakazam 在本回合攻击，允许先完成这个换位。
- 若攻击合法但伤害不足以击倒，不再无条件攻击；要比较继续攻击和补充 Dunsparce/抽牌
  后的真实收益。第三局说明了“非致命攻击优先”会错过建立循环过牌的机会。

### 3.2 保住攻击线

- Active Abra/Kadabra 若能通过 Rare Candy 和 Alakazam 在本回合攻击，优先保留这条
  直接进化路线。
- 如果本回合只能进化成 Kadabra，而前场 Abra 仍有机会在本回合通过 Rare Candy 变成
  Alakazam，则优先把 Kadabra 放到 Bench 的 Abra 上，不要锁死前场的直接攻击路线。
- 攻击线没有达到场上实际三只 Abra 系列的目标时，Poffin 和 Telepath Energy 优先补 Abra 系列；判定只看
  场上数量，不区分 Active/Bench。

### 3.3 建立 Dunsparce 循环线

- 当 Abra 系列攻击线达到前期目标后，Poffin 优先补 Dunsparce。
- Hilda 在本回合和下回合的攻击都已有保障时，可以选择 Dudunsparce 加 Enriching
  Energy 的组合，建立可循环的过牌资源。
- Hilda 不能直接寻找 Basic Dunsparce；它的宝可梦目标是 Evolution Pokémon，因此
  “用 Hilda 补 Dunsparce”在实现上指寻找 Dudunsparce。

## 4. Energy Rules

### 4.1 Telepath Psychic Energy

- Telepath Psychic Energy (`19`) 从手牌贴到 Psychic Pokémon 时，才触发从牌库寻找
  最多两只 Basic Psychic Pokémon 并放到 Bench 的效果。
- 手牌有 Abra、场上没有合适的 Psychic 目标时，必须先打出 Abra，再把 Telepath Energy
  贴给它；不能先贴给 Active Dunsparce 后再补 Abra。
- 场上已有 Abra/Kadabra/Alakazam 时，Telepath Energy 优先服务于这些目标，默认用于
  补充 Abra 系列。
- 如果 Active 必须 Retreat，且 Retreat 后有宝可梦能在本回合立即攻击，可以为了这次
  攻击改变 Telepath 的目标。
- 如果没有 Abra 可以铺，Telepath Energy 可以退而贴给 Dunsparce，作为 fallback；只要仍有可铺的 Abra，
  就优先服务于 Abra。

### 4.2 其它能量

- Basic Psychic Energy (`5`) 用于保证攻击线和攻击手的必要能量。
- Enriching Energy (`13`) 优先给 Dudunsparce (`66`)；在安全抽牌条件成立时用于循环
  过牌。
- Alakazam 只需要一张 Psychic Energy 攻击；已经有一张时不主动贴第二张。
- 不为了无明确收益的 Retreat 消耗攻击手的能量。

## 5. 抽牌与牌库安全

- 手牌超过 20 张后，原则上停止非必要抽牌；这通常已经足以让 Powerful Hand 处理大多数
  目标。
- 牌库剩 10 张或更少时，除非抽牌是完成必要伤害、进化或攻击的直接条件，否则停止
  Dudunsparce、Alakazam、Kadabra 及训练家带来的过牌。
- 如果当前手牌已经能让 Alakazam 击倒当前目标，不为了增加伤害继续抽牌。
- 最后一回合只有在“抽牌后能立即击倒对手 Active，并且对局会在牌库耗尽前结束”时才
  允许继续过牌；否则优先保留牌库。

## 6. 搜索、进化与恢复

- Telepath Energy：主要找 Abra。
- Buddy-Buddy Poffin (`1086`)：攻击线不足时找 Abra，攻击线达到目标后找 Dunsparce。
  不主动找 Psyduck 或 Shaymin。
- Hilda (`1225`)：场面不保证本回合/下回合攻击时，优先拿缺少的进化牌和能量；攻击线
  安全时考虑 Dudunsparce 加 Enriching Energy。
- Rare Candy (`1079`)：优先保留前场 Abra 直接进化成 Alakazam 并攻击的机会。
- Night Stretcher (`1097`) 和 Lana’s Aid (`1184`)：都是从弃牌区恢复打手的手段。在合法目标和资源条件
  允许时，优先恢复 Abra→Kadabra→Alakazam 这条进化链，而不是只固定恢复 Abra；具体恢复阶段根据当前手牌、
  场面以及本回合/下回合的攻击需求决定。
- Sacred Ash 和 Wondrous Patch：只有在弃牌区资源能重建攻击线、且当前抽牌安全时使用。

## 7. Retreat、Trading Places 与攻击目标

- 真正的 Retreat 只有两类主要理由：
  1. 不 Retreat 就无法让 Alakazam 在本回合攻击；
  2. Active 是 Fezandipiti ex，撤退能带来明确的保护两奖宝可梦收益。
- 没有上述收益时，即使选项出现，也选择跳过。
- Dunsparce 的 Trading Places 不等同于 Retreat；如果它能把已经准备好的 Alakazam
  换到 Active 并立即攻击，可以使用。
- Boss's Orders (`1182`) 只有在当前 Active 不能被击倒、换出目标能在本回合兑现 Prize，
  或能带来更高 Prize 时才使用。
- 攻击目标先判断是否能本回合击倒，再比较 Prize 价值和剩余 HP。

## 8. 官方 replay 复盘指标

每次官方对局需要记录：

- 场上 Abra 系列总数（Active+Bench 合计）和 Dunsparce 系列数量；
- Telepath Energy 的目标、是否触发搜索、搜索出的 Basic Psychic Pokémon；
- 首次 Alakazam、首次攻击和首次击倒的 turn；
- Poffin、Hilda、Rare Candy、Night Stretcher、Lana’s Aid 的具体目标；
- 每次抽牌前后的手牌数和牌库数；
- 真正 Retreat 与 Trading Places 的次数及其是否带来本回合攻击；
- Alakazam 被击倒后是否成功用 Abra 补回攻击线；
- 牌库耗尽、无 Active 或 Prize 结束等终局原因。

## 9. 实现状态

- 已实现：V2 的合法选项解析、确定性返回、Alakazam 手牌伤害估算、基础攻击目标和
  基础恢复逻辑。
- 尚未实现：Telepath 先铺 Abra 的行动顺序、三只 Abra 系列的场面目标、Dunsparce
  循环线状态、20/10 牌库保护阈值、Hilda/Dudunsparce 组合优先级、Bench Abra 的
  Kadabra 进化规则、Fezandipiti ex 的伤害缺口计算，以及 Night Stretcher/Lana’s Aid
  的进化链恢复优先级。
- 当前 V3 `main.py` 与 V2 相同；本文件是讨论后目标策略，不应被当作已经提交到 Kaggle
  的行为说明。
