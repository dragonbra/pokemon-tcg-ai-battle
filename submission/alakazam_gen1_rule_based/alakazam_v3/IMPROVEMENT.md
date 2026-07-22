# Alakazam V3 改进记录与讨论

> 本文分为四层：用户原始观察、Codex 反馈、双方确认的策略约束，以及策略改动设计。
> 用户原文保留在前两部分；A-D 改动已经落地到 V3 `main.py`，E 仍是后续 replay 复盘指标设计。


## 一、用户观察到的问题（原文保留）

第一局对战 Đức Nguyễn Minh 长毛巨魔和雪妖女卡组

1. 我发现你不是很喜欢往备战区放置 dunsparce 和 dudunsparce 这张卡？为什么？
2. 当我们的手牌大于对手场上最大血量的宝可梦，且不缺少这回合关键资源的情况下。我们就不要过牌了，避免卡组卡牌耗尽导致的失败。

第二局对战 pincehur

1. 为什么贴了 telepathic energy 却不往备战区放 Abra？？我不理解？
2. 而且为什么又在进行拔能撤退？
    我们需要明确，撤退是有明确的作用的时候，我们才会选择撤退
    那么第一优先级的事情，是只有通过撤退，才能让这回合使用胡地进行攻击
    第二是，如果前场的是吉雉鸡 ex，那么我们有可能需要进行撤退，保护2奖宝可梦。
    其他情况下，我想不到我们必须撤退的理由，不能因为有这个选项就一定要执行，我们完全可以不执行，直接跳过
3. 稍等，这里我可能看错了，你似乎是使用了 dunsparce 的换位技能
    这个技能，没有太大问题，确实是尽可能放胡地在前场。
第二局输的原因怎么说呢？
整体来说，你确实没有训练家，按理说也确实不会打赢。但是这里面暴露出来一个很大的问题：你不会使用 Telepathic Energy 去往备战区放我们的基础宝可梦。
这是我们备战区没有宝可梦，导致输掉比赛的最重要原因


第三局对战 Rostislav
1. 为什么在第一回合发生了拔能撤退凯西？根本不需要啊
    如果你第一回合没有进行这个操作的话，第二回合你其实是可以使用胡地进行攻击，然后拿到奖赏卡就胜利了的
2. 然后这里我觉得可以稍微讲一下，就是当你使用 Hilda 或者是类似能找到宝可梦的卡牌的时候，你的判断
    首先，我们可以罗列一下能找到宝可梦的卡牌逻辑：
    1. 特殊的超能量（Telepathic Energy）：
    我们需要用这张牌去找凯西（Abra），因为它只能在牌库里面寻找超能属性的宝可梦。
    2. 宝芬（Puffin）：
    在已经能够保证前场和后场各有一个凯西（Abra）的情况下，我们可能就要考虑是不是多做一点土龙弟弟（Dunsparce）。因为土龙弟弟的过牌能力能够循环回到牌库进行轮转，所以它其实是我们在后场要考虑多做一点的宝可梦。当然这还有一个前提，就是确保我们的打手 Abra、Kadabra 以及 Alakazam 的这条进化链是足够的情况下，我们可能就需要用 Hilda 去放 Downspars，这样的话，我们的手牌才能保证变多。当然，手牌也是有上限的，所以在前几回合的 Dunsparce 布局更重要，当然，这也是在满足 Airbar 的前提下。
3. 我看到你有一个回合其实是直接进化胡地就攻击了，但是伤害不够不能昏厥对方前场宝可梦。因为后备区没有 Dunsparce，所以我们手上其实有很多 Dunsparce 都是没有进化的。在没有进化的情况下，相当于我们没办法通过它的特性去使用 Run Away Draw 来过牌，这导致我们后面其实就很绵软无力了。其实你当时的手牌是有机会增加的。如果我们在前面选择 Abra 的时候选择了 Dunsparce，那我们此时就可以通过 Dunsparce 继续过牌了，这一点你要注意。
4. 其实我有一个希望你注意的点：我们的卡组里面有两颗神奇糖果（如果没有记错的话），这张牌确实是非常重要的。
我列举一个场景：
我们前场是一个 Abra，后场也是一个 Abra。我们最终的目标，是希望前场的 Abra 进化成 Alakazam，然后进行攻击。
但是，我看到你在当时的某一个回合做了这么一个操作：给前场进化成了 Kadabra，然后通过它的特性，我们抽到了 Rare Candy。但这时候，Rare Candy 只能对后场的 Abra 进行进化了。
其实，不管是 Kadabra、别的宝可梦还是 Trainer 卡牌在手上的时候，我们确实需要设定一个优先级：
如果一回合内只能进化成 Kadabra 的话，这就意味着被进化成 Kadabra 的这张宝可梦，不可能在这一回合直接进化成 Alakazam 并进行攻击了。
因此，我们的 Kadabra 进化，可能要优先对备战区（后场）的 Abra 进行使用。这样我们在过牌的时候，如果过到了 Rare Candy 以及 Alakazam，我们还是有机会让前场的这只 Abra 在这一回合直接进化成 Alakazam，然后贴上超能量进行攻击。


## 二、Codex 反馈与讨论记录


你在 improvement 里已经修正了自己的观察：那次可能不是拔能撤退，而是使用 Dunsparce 的 Trading Places。

  这两个动作不能混为一谈：

  - 真正的 Retreat：只有 Bench 上已有带能量的 Alakazam，并且撤退能让本回合攻击时才允许；
  - Dunsparce 的 Trading Places：如果它能把已经准备好的 Alakazam 换到 Active 并立即攻击，可以保留这个动
    作。

  我建议我们在实际改代码前，先确认下面几条策略约束：

  - 是否要求至少保证 Active/Bench 各有一只 Abra 后，才开始大量铺 Dunsparce？
  - 至少需要几只 Dunsparce/Dudunsparce 才算抽牌引擎安全？
  - 如果手里有 Abra、Telepath Energy，但场上没有 Psychic Pokémon，是否明确规定“先打 Abra，再贴
    Telepath”？

  - 如果没有 Abra 可铺，Telepath Energy 是否可以退而贴给 Dunsparce？
  - 牌库安全的阈值采用“当前 Active 可击倒”还是“对手场上最高 HP 可击倒”？
  - Hilda 在攻击线安全后是否固定优先 Dudunsparce，而不是继续拿 Alakazam？

  目前最明确、我认为应该优先讨论的是：Telepath 的行动顺序，以及“攻击线已存在但 Dunsparce 引擎未建立”这
  两个条件。其它改动我建议等你确认这些规则后再动。

1 是的，而且尽量保证 active + bench 有三只 abra 系列宝可梦再铺 dunsparce
2 dunsparce 多多益善，我们的宝可梦已经尽可能简化了，所以你暂时不用考虑备战区留空的情况（除了吉雉鸡不
要随意拍下，现在保守的话我们都可以暂时不拍下，除非满足上回合有昏厥宝可梦+我们缺3张手牌伤害的情况下）
3 如果有 Abra在，那么 telepath energy 肯定是优先贴给它的（除非前场需要撤退且撤退后有宝可梦能立刻在这
回合进行攻击的情况下）
4 牌库安全的话，我认为最优先是我们手牌超过20张之后就不要接着弄了（这应该能昏厥绝大部分宝可梦）。然后
的条件是在最后一回合（就是如果我们攻击伤害算好过空牌库就能立刻击倒对方前场宝可梦的情况下，否则不要疯
狂过牌库了）。然后牌库的数量安全线在10张，10张以下就是需要除非满足伤害必要，否则别过牌了。
5 Hilda 仍然是依据场上情况判断，但我的建议是，场上能确保这回合和下回合的进攻的前提下，Hilda 可以多拿
dudunsparce + enriching energy 的组合过牌的。

此外，很重要的一张牌 night stretcher 应该是优先去拿弃牌区里面的 Abra 补充我们的打手的，理论上我们的
进攻手段只有这么4次胡地，如果我们每次被昏厥之后，其实是很难再有其他进攻手段的。

## 三、双方确认的策略约束（归纳）

下面把前面的观察和回答整理成可以交给实现的规则。它们是当前讨论后的目标行为，尚未全部落地到 V3
代码中。

### 1. 前期场面结构

- 前期优先保证场上实际存在三只 Abra 系列宝可梦，即 Abra、Kadabra 或 Alakazam 的组合。
  这里的判定只看 Active 和 Bench 合计后的实际数量，不要求特定的 Active/Bench 分布。
- 场上达到三只 Abra 系列宝可梦后，才主动扩大 Dunsparce/Dudunsparce 数量。
- Dunsparce/Dudunsparce 原则上越多越好，因为它们是主要的循环过牌引擎；不需要为了刻意留空 Bench
  而限制它们。
- Fezandipiti ex 暂时采取保守策略，不因为选项可用就放下；只有在上一回合已经有宝可梦昏厥，且按照
  当前手牌数量和 Alakazam 发动攻击时的手牌数 ×20 伤害计算，确实需要它补充手牌伤害时，才考虑使用。
  “缺三张”不是固定阈值，而是根据当前目标和实际伤害缺口计算。

### 2. Telepath Psychic Energy

- Telepath Psychic Energy 贴到 Psychic Pokémon 才能触发从牌库找 Basic Psychic Pokémon 的效果。
- 手牌中有 Abra 且场上没有合适的 Psychic Pokémon 时，应先把 Abra 放到 Bench，再把 Telepath Energy
  贴给它；不能先把能量贴给 Active Dunsparce，之后才补 Abra。
- 如果场上已有 Abra/Kadabra/Alakazam，Telepath Energy 优先贴给这些 Psychic Pokémon。
- 例外是 Active 必须撤退，且撤退后能让另一只宝可梦在本回合立即攻击；此时允许为了完成攻击而改变
  能量目标。
- 如果没有 Abra 可以铺，Telepath Energy 可以退而贴给 Dunsparce；但只应在没有更优的 Abra 铺场目标时
  作为 fallback。

### 3. 抽牌与牌库安全

- 如果当前手牌已经超过 20 张，原则上停止非必要抽牌；这通常已经足够让 Powerful Hand 击倒大多数目标。
- 牌库剩 10 张或更少时，除非抽牌是完成必要伤害/攻击的直接条件，否则不再继续过牌。
- 如果本回合的攻击伤害已经足够击倒目标，不要为了增加伤害继续使用 Dudunsparce 或其它抽牌效果。
- 最后一轮的判断例外是：只有在计算过抽牌后能立即击倒对手 Active、并在牌库耗尽前结束对局时，才允许
  继续抽牌；否则应停止消耗牌库。

### 4. 搜索与进化资源

- Telepath Energy 主要负责找 Abra。
- Poffin 在场上实际的 Abra 系列数量尚未达到三只时优先补 Abra 系列；达到三只后，优先补 Dunsparce。
- Hilda 不能直接寻找 Basic Dunsparce；它寻找的是 Evolution Pokémon，因此“用 Hilda 补 Dunsparce”在
  实现上应理解为寻找 Dudunsparce。
- 当本回合和下回合的攻击都已经有保障时，Hilda 可以选择 Dudunsparce 加 Enriching Energy，建立循环过牌。
- 如果前场 Abra 本回合有机会通过 Rare Candy 加 Alakazam 直接攻击，而只能先完成一次 Kadabra 进化，
  则优先把 Kadabra 进化放在 Bench 的 Abra 上，保留前场 Abra 的直接进化路线。
- Night Stretcher 和 Lana’s Aid 都是从弃牌区恢复打手的手段。两者在合法目标和资源条件允许时，都应优先
  服务于 Abra→Kadabra→Alakazam 这条进化链，而不是只固定恢复 Abra；具体恢复哪一个阶段，结合当前手牌、
  场面以及本回合/下回合的攻击需求判断。

### 5. Retreat 与 Trading Places

- 真正的 Retreat 只有在它是本回合使用 Alakazam 攻击的必要条件时才优先。
- 如果 Active 是 Fezandipiti ex，可以把保护这只两奖宝可梦作为第二类合理撤退理由，但必须有明确的场面收益。
- Dunsparce 的 Trading Places 不等同于 Retreat；如果它能把已经准备好的 Alakazam 换到 Active 并立即攻击，
  可以保留。
- 不能因为 Retreat 或 Trading Places 选项存在，就默认执行。

## 四、策略改动与实现状态

### A. 把 Telepath 的“铺 Abra”变成行动顺序约束

在主行动排序中增加组合判断：如果手里有 Abra 和 Telepath Energy、场上没有 Psychic 目标，则先打出 Abra，
下一次行动再把 Telepath 贴给它。只有没有 Abra 可以铺时，才允许把 Telepath 贴给 Dunsparce 作为 fallback。

实现状态：已实现。

### B. 将“攻击线安全”和“抽牌线安全”分开判断

不再用“场上有任意 Abra/Kadabra/Alakazam”代表全部 setup 已完成。攻击线目标应判断场上实际存在的数量，
不区分 Active/Bench 的位置；至少分别维护：

- Abra 系列攻击线数量；
- Dunsparce/Dudunsparce 循环线数量；
- 当前 Active 是否能攻击、下回合是否能攻击；
- 当前手牌和牌库是否已经达到停止抽牌阈值。

这样可以在保住三只 Abra 系列的同时继续铺 Dunsparce，而不是二选一。

实现状态：已实现。场上三只的判定按 Active+Bench 实际数量计算，抽牌保护也已独立加入行动排序。

### C. 重新定义 Poffin、Hilda 和 Rare Candy 的条件优先级

- Poffin：场上 Abra 系列少于三只时找 Abra，达到三只后找 Dunsparce。
- Hilda：攻击线或下回合攻击不安全时找缺少的进化牌；攻击线安全时考虑 Dudunsparce 加 Enriching Energy。
- Rare Candy：优先保留能让前场 Abra 直接成为 Alakazam 并攻击的路线，避免先把前场锁成只能继续进化的
  Kadabra。

实现状态：已实现。Hilda 在攻击线安全时优先 Dudunsparce+Enriching Energy，Night Stretcher/Lana’s Aid
优先恢复 Abra→Kadabra→Alakazam 进化链。

### D. 增加显式牌库保护器

把“手牌超过 20”“牌库不超过 10”“当前伤害是否已经足够”放到 Dudunsparce、Kadabra、Alakazam 和训练家
抽牌动作的共同判断中，而不是只在牌库接近 0 时才阻止 Run Away Draw。Fezandipiti ex 是否落下也要使用同一套
基于手牌数量和 Alakazam 手牌数 ×20 伤害缺口的计算。

实现状态：已实现。手牌超过 20 张或牌库剩 10 张及以下时，非必要抽牌会被压低；直接形成击倒的抽牌保留。

### E. 增加恢复与动作指标

后续官方 replay 复盘需要单独记录：

- Telepath Energy 的目标宝可梦及是否触发搜索；
- 场上 Abra 系列总数（Active+Bench 合计）和 Dunsparce 系列数量；
- Hilda、Poffin、Rare Candy、Night Stretcher、Lana’s Aid 的目标；
- 每次抽牌时的手牌数和牌库数；
- 真正 Retreat 与 Trading Places 的次数和结果；
- Alakazam 被击倒后是否成功用 Abra 补回攻击线。

实现状态：暂未加入独立 replay 日志；当前策略选择已经记录恢复目标，后续官方对局复盘时再补充结构化指标。

## 五、Changelog

### 2026-07-19

- 保留用户对三场 Kaggle 官方对局的原始观察和后续回答。
- 将内容整理为“用户观察 → Codex 反馈 → 双方确认规则 → 拟议改动”的结构。
- 明确 Telepath Energy 必须优先服务于 Psychic Pokémon 和 Abra 铺场顺序。
- 增加三只 Abra 系列、20 张手牌、10 张牌库等讨论中的策略约束。
- 增加 Dunsparce/Dudunsparce、Hilda、Rare Candy、Night Stretcher、Lana’s Aid 的目标设计。
- 明确区分真正 Retreat 与 Dunsparce Trading Places。
- 确认 Telepath 在没有 Abra 可铺时可以 fallback 到 Dunsparce；三只 Abra 系列的判定只看场上实际数量，
  不区分 Active/Bench。
- 明确 Fezandipiti ex 的使用条件基于当前手牌数量，以及 Alakazam 按发动攻击时手牌数 ×20 计算出的实际伤害缺口。
- 明确 Night Stretcher 与 Lana’s Aid 都应优先恢复 Abra→Kadabra→Alakazam 进化链。
- V3 `main.py` 已按上述 A-D 方案实现；本次验证通过资产检查、语法检查、raw `exec` 检查，以及 V3 对 V2/V1
  的本地对局。
