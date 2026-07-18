# Alakazam V4 Strategy Specification

## 1. 回合目标

每次主行动按以下顺序判断：

1. 当前 Active 能否实际击倒对手；
2. 如果对手 Active 带 Mist Energy，是否需要先用 Enhanced Hammer，或用 Boss's
   Orders 拉出没有 Mist 的可击倒目标；
3. 是否能在本回合完成 Alakazam 攻击；
4. 是否要保留下一只 Alakazam 的能量和进化线路；
5. 最后才扩展 Dunsparce/Dudunsparce 过牌线。

攻击不能击倒时不是无条件攻击。特别是 Alakazam 对带 Mist Energy 的目标，当前
官方 simulator 的 DamageCounter 实现会产生 0 伤害，因此 agent 会把这种目标视为
不可用的攻击目标，并把结束回合放在重复 0 伤害之前。

## 2. 进化线路

### 2.1 第二回合直通

Active Abra 有 Psychic Energy（或手牌有可附着的 Psychic Energy）、手牌有 Rare
Candy 和 Hilda 时，先用 Hilda 搜索 Alakazam 和 Psychic Energy，再用 Rare Candy
进化并攻击。若 Alakazam 已在手牌，则不需要为了同一线路再次使用 Hilda。

Hilda 的第一项搜索只选 Evolution Pokémon，第二项搜索优先选 Psychic Energy。
如果没有 Rare Candy 直通条件，Active Abra 的普通 Kadabra 进化优先于无意义地
搜索 Alakazam；Hilda 会优先找 Kadabra 为下一回合准备。

### 2.2 Active/Bench 冲突

Active 和 Bench 都有 Abra、且 Active 仍保留 Rare Candy 直通可能时，Kadabra
进化优先落在 Bench Abra。这样不会把 Active 锁成 Kadabra，同时还能通过 Kadabra
的 Psychic Draw 增加找到 Rare Candy 的机会。

若本回合不能直通，才按普通线路进化 Active Abra 为 Kadabra，再在下一回合进化
Alakazam。Alakazam 只需要一张 Psychic Energy，不会主动给已经有能量的 Alakazam
贴第二张。

## 3. 搜索和过牌

- Poké Pad 每次只取一张，并在状态更新后重新判断下一张 Poké Pad 的目标。
- Active Abra 有 Rare Candy 直通线路时，Poké Pad 搜索 Alakazam。
- 没有 Rare Candy 且当前不能进化时，Poké Pad 搜索 Kadabra；若没有可用 Kadabra，
  才搜索 Dunsparce 放 Bench，为下一次 Poké Pad 和 Trading Places 建立入口。
- Buddy-Buddy Poffin 在 Abra 系列不足三只时优先找 Abra，之后再找 Dunsparce。
- Kadabra、Dudunsparce 和 Fezandipiti 的抽牌都受手牌 20 张、牌库 10 张的保护；
  已经能击倒时不为了多一点伤害继续抽牌。

## 4. Energy、Retreat 和下一只打手

- Telepath Psychic Energy 在还有可铺的 Abra 时优先服务 Abra；没有 Abra 可铺时
  才作为 Dunsparce 的 fallback。
- Active Fezandipiti ex、Bench 有带 Psychic Energy 的 Alakazam、且当前能量附着
  后可以 Retreat 时，手填能量优先给 Fezandipiti，然后 Retreat 让 Alakazam 攻击。
- Active 有能量的 Alakazam、Bench 有无能量 Alakazam 时，手填能量优先给 Bench
  打手，避免 Active 被击倒或下一回合受到手牌干扰后断线。
- 没有明确的本回合攻击收益时，不 Retreat；Active 无法进化/攻击时，宁可让它
  被击倒，也不支付没有收益的 Retreat 成本。
- Dunsparce 的 Trading Places 只在能把已带能量的 Alakazam 换到 Active 并立即
  攻击时使用。

## 5. Mist Energy 和目标选择

Enhanced Hammer 的主目标顺序是：

1. 对手 Active 的 Mist Energy；
2. 对手 Active 的其他特殊能量；
3. 对手 Bench 的特殊能量。

如果当前 Alakazam 对 Active 的实际伤害为 0，Hammer 的优先级提升；若手里没有
Hammer，则检查 Boss's Orders 是否能把 Bench 上没有 Mist 且能被击倒的目标拉到
Active。若两者都不能创造有效攻击，避免重复攻击 Mist 目标。

Boss's Orders 只在当前 Active 不能击倒、换出目标能实际击倒，或能带来更高 Prize
价值时使用。攻击目标选择会把 Mist 对 Alakazam 的 0 伤害纳入判断。

## 6. Fezandipiti ex

Fezandipiti ex 是两 Prize 的风险目标，因此不作为普通 Bench 填充。以下两种情况
允许使用：

- 第一回合 Bench 没有任何宝可梦，避免 Active 被击倒后直接输掉；
- 上一回合确实有宝可梦被击倒，当前没有更直接的攻击/进化动作，Flip the Script
  的三张牌能创造进化、攻击或击倒机会。

当对手只剩两张 Prize 且己方 Bench 已经有普通宝可梦时，默认不放 Fezandipiti。

## 7. Xerosic's Machinations

Xerosic's Machinations 只在对手手牌至少 4 张时合法。进入自己第二回合以后，若对手
场面出现 Abra、Kadabra 或 Alakazam，或者对手手牌很多，且当前没有更重要的 Hilda、
Dawn、Rare Candy 或持续攻击动作，就提高 Xerosic 的优先级。它会让对手手牌降到
3 张，尤其针对 Alakazam 内战中的手牌伤害。

当己方被对手使用 Xerosic 时，选择要丢弃的手牌会优先丢重复能量、Battle Cage、
重复的 Poffin/Poké Pad 和其他低即时价值资源，保留 Rare Candy、Hilda、Dawn、
Boss's Orders 以及唯一的 Abra 进化链。
