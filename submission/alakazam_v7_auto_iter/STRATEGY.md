# Alakazam V7 策略说明

## 主行动顺序

每次收到新的合法 option 都重新规划；攻击是本回合的终止提交。

1. 识别本回合能够完成的终局或确定 Prize，必要时用 Boss 选择可确定 KO 的最高剩余 HP Bench 目标。
2. 确定 primary attacker：Active Kadabra 有能量和 Alakazam 时先自然进化；Active Abra 有能量且具备直通资源时，保留它给 Rare Candy + Alakazam；若 Active 是 Shaymin 等非攻击者，则保留带 Psychic Energy 的 Bench Abra 给 Rare Candy。
3. 在存在带能量的 Bench Abra 主攻击路线时，先让其它无能量 Bench Abra 自然进化成 Kadabra，并在攻击前使用 Kadabra Ability；不能让自然进化覆盖主攻击目标。
   但如果 Active Abra 已有 Psychic、自然进化到 Kadabra 合法且能立刻形成确定 KO，先完成 Active 进化和这条攻击路线；不能为了 Bench 的预备动作错过本回合的确定 Prize。
4. 只使用一次手填能量：普通 Psychic Energy 给需要能量的 Abra 线；Active Kadabra 或
   Alakazam 已经有 Psychic 且 Bench 存在可见接力目标时，先为接力目标准备能量，再考虑
   Enriching Energy 给 Dunsparce/Dudunsparce。
   Bench Abra 即使尚未看到 Kadabra，也可以在非终局时用 Wondrous Patch/Lana's Aid 先准备 Psychic；但直接手动贴 Basic Psychic 只有在当前手牌能验证下一阶段时才优先，下一回合仍依据实际手牌判断自然进化。
5. 需要从弃牌区同时恢复宝可梦和 Psychic Energy 时，优先 Lana's Aid；只有明确补齐现实攻击路线时才使用 Night Stretcher。
6. 在牌库保护线以上优先执行 Fezandipiti、Kadabra、Alakazam 和 Dudunsparce 过牌；Item Lock 时切换到自然进化和过牌。
7. Active Alakazam 已能攻击、对手 Active 已受伤但不能 KO、对手手牌至少 6 张且没有 Boss KO 时，攻击前使用 Xerosic。
8. 非终局确定 KO 前，如果 Active Alakazam 已带 Psychic、手牌有 Alakazam，且 Bench 有已带 Psychic 的 Kadabra 并存在合法进化选项，先完成这只接力 Alakazam；终局闭环不触发该 gate。
9. 如果第二回合同时有 Powerful Hand 与 Boss's Orders，且当前 Active 不能 KO、Boss 能拉出确定 KO 的 Bench 目标，先使用 Boss；否则再按第二回合攻击 gate 判断是否立即 Powerful Hand。
10. 最后才攻击：终局/确定 KO 优先，Active Alakazam 正常攻击其次，Kadabra/Dudunsparce 非 KO 攻击为最后手段；Abra 与 Trading Places 永不攻击。

## Mist / Rock Fighting Energy

Mist Energy（ID 11）和 Rock Fighting Energy（ID 20）都提供特殊能量，并在满足卡面
条件时让宝可梦不受对手宝可梦招式的效果影响。Mist 对附着的任意宝可梦生效；Rock
Fighting Energy 只有附着在 Fighting 属性宝可梦上才生效。官方引擎中 Alakazam 的
Powerful Hand 是通过 `DamageCounter` 放置伤害指示物；因此 Mist 目标，以及满足
Fighting 条件的 Rock 目标，其有效伤害按 0 计算，不能把它们当作可确定 KO 的目标。

遇到这两种能量时，策略按以下顺序处理：

- 如果当前有 Enhanced Hammer 的合法选项，优先移除阻挡 Powerful Hand 的特殊能量；
- 如果没有移除路径，不把 Mist 或满足 Fighting 条件的 Rock 目标当作可 KO 目标，也不
  为了“试一下伤害”直接攻击；
- 对于无法从引擎卡表解析属性的 Rock 目标，按“可能受到保护”保守处理，不宣称确定 KO；
- Boss、终局闭环、过牌后 KO 判断都复用同一个保护能量模型；
- 这条规则只针对已从 observation 看见的附着能量，不预设对手手牌或牌库中的未知牌。

## Item Lock

只把实际观察到的对手攻击日志 `type=15, cardId=235, attackId=323` 解释为 Budew 的
Itchy Pollen。锁定只覆盖紧接着的我方一个回合，并按已观察事件去重，不因为 Budew 仍在场上
或旧日志重复出现而持续。锁定时所有 Item（包括 Rare Candy、Poffin、Poké Pad、Night
Stretcher）都不进入策略候选；Supporter、Energy、自然进化和 Ability 仍按正常预算执行。

## Swap 回合与 Budew Item Lock

- `firstPlayer` 是 shared turn 转换为己方回合序号时的依据，不能固定假设物理 index 0
  先手；否则 swap 对局会错误禁止第二个己方回合的自然进化。
- Budew 的 `Itchy Pollen` 只锁 Item，不影响从手牌自然进化 Kadabra。Active Abra 有
  Psychic、手牌有 Kadabra、进化后能 KO 时，先自然进化再攻击。

## Replay 验收汇报

每个 R1-R5/V7-01-V7-08 case 都报告：局面摘要、V6 实际动作、V7 首选动作、重新规划
后的后续动作链，以及首步和完整动作链各自的 PASS/FAIL 结果。动作使用 `type + cardId`
或 `attackId + target` 描述，不依赖 option 数组下标。
