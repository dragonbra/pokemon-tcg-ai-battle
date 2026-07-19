# Alakazam V7 策略说明

## 主行动顺序

每次收到新的合法 option 都重新规划；攻击是本回合的终止提交。

1. 识别本回合能够完成的终局或确定 Prize，必要时用 Boss 选择可确定 KO 的最高剩余 HP Bench 目标。
2. 确定 primary attacker：Active Kadabra 有能量和 Alakazam 时先自然进化；Active Abra 有能量且具备直通资源时，保留它给 Rare Candy + Alakazam；若 Active 是 Shaymin 等非攻击者，则保留带 Psychic Energy 的 Bench Abra 给 Rare Candy。
3. 在存在带能量的 Bench Abra 主攻击路线时，先让其它无能量 Bench Abra 自然进化成 Kadabra，并在攻击前使用 Kadabra Ability；不能让自然进化覆盖主攻击目标。
4. 只使用一次手填能量：普通 Psychic Energy 给需要能量的 Abra 线；Active Alakazam 已准备好且有 Dudunsparce 过牌路线时，Enriching Energy 给 Dunsparce/Dudunsparce。
5. 需要从弃牌区同时恢复宝可梦和 Psychic Energy 时，优先 Lana's Aid；只有明确补齐现实攻击路线时才使用 Night Stretcher。
6. 在牌库保护线以上优先执行 Fezandipiti、Kadabra、Alakazam 和 Dudunsparce 过牌；Item Lock 时切换到自然进化和过牌。
7. Active Alakazam 已能攻击、对手 Active 已受伤但不能 KO、对手手牌至少 6 张且没有 Boss KO 时，攻击前使用 Xerosic。
8. 最后才攻击：终局/确定 KO 优先，Active Alakazam 正常攻击其次，Kadabra/Dudunsparce 非 KO 攻击为最后手段；Abra 与 Trading Places 永不攻击。

## Item Lock

只把实际观察到的对手攻击日志 `type=15, cardId=235, attackId=323` 解释为 Budew 的
Itchy Pollen。锁定只覆盖紧接着的我方一个回合，并按已观察事件去重，不因为 Budew 仍在场上
或旧日志重复出现而持续。锁定时所有 Item（包括 Rare Candy、Poffin、Poké Pad、Night
Stretcher）都不进入策略候选；Supporter、Energy、自然进化和 Ability 仍按正常预算执行。

## Replay 验收汇报

每个 R1-R5/V7-01-V7-08 case 都报告：局面摘要、V6 实际动作、V7 首选动作、重新规划
后的后续动作链，以及首步和完整动作链各自的 PASS/FAIL 结果。动作使用 `type + cardId`
或 `attackId + target` 描述，不依赖 option 数组下标。
