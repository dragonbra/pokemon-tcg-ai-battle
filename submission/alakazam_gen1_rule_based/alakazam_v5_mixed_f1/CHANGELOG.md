# Alakazam 策略版本变更记录

本文件记录固定 V5 牌表下的策略变化。`deck.csv`、`cg/` runtime、公开 simulator
API、Mist Energy 运行时处理和确定性合法选项在以下版本中均保持不变。

## V5 -> V5 Mixed

### 引入的策略

- 增加可见的攻击连续性状态：场上攻击线数量、未来攻击线数量、当前/下一回合
  攻击路径、未充能攻击线、弃牌区恢复线，以及 Dunsparce/Dudunsparce 引擎状态。
- 将抽牌从固定动作偏好改为目标驱动：优先当前 KO，其次允许能直接形成 KO 的
  抽牌，之后才考虑下一只攻击手和额外资源。
- 在原有手牌 20 张、牌库 10 张保护之外，增加“牌库保留剩余 Prize + 1”的动态
  安全线；只有抽牌能直接形成当前 KO 时允许突破。
- Enriching Energy 只有在 Active Alakazam 已能攻击、攻击线没有未充能缺口、且
  Dudunsparce 的 `Run Away Draw` 可用时才转化为过牌。
- 恢复动作按照最短攻击路径选择：没有场上 Abra 系列时优先恢复 Abra；Lana's Aid
  在合法范围内尽量恢复多张相关资源。
- Enhanced Hammer 只有在移除特殊能量能改变当前攻击结果时才获得最高优先级，
  同时保留 V5 对 Mist Energy 的 0 伤害运行时判断。
- Fezandipiti ex、Boss、Xerosic 和 Retreat 增加 Prize race、攻击收益和暴露风险
  gate；Telepath、能量上限、自然进化和首回合 Poké Pad 规则继续保留。

### 评测暴露的问题

Mixed 的资源保护比 V5 更强，但攻击节奏明显变慢：首次出现 Alakazam 平均回合从
V5 的 `5.22` 延后到 `7.24`，无可用 Active 败局从 `8` 增加到 `13`。Xerosic 使用
次数从 `10` 降到 `4`，Sue Alakazam 内战从 `8/10` 降到 `3/10`，Pilkwang 和
Dragapult 也出现明显回退。

评测结论是：牌库保护本身有效，但不能替代首只 Alakazam 的建立和 ready attacker
接力。因此 Mixed 暂不作为 V5 的直接替换版本。

## V5 Mixed -> V5 Mixed F1

F1 的目标是恢复首攻节奏，同时保留 Mixed 中值得保留的资源保护。

- 首只 Alakazam 建立前，恢复 V5 的 tempo gate。Poffin 依据场上真实攻击线数量
  选择 Abra，不再因为手牌中的攻击线让 `future_count` 提前达标而转向 Dunsparce。
- 首只 Alakazam 建立后，增加 `ready_attacker_count` 作为接力核心状态；已充能的
  Alakazam、Kadabra 和 Abra 都可计入下一只有效攻击手。
- `_attack_continuity()` 不再把“弃牌区有攻击线”直接当成 next attack path，只有
  手牌中真实存在 Night Stretcher、Lana's Aid 或 Sacred Ash 时才算可恢复路径。
- Hilda/Poffin 在攻击线数量已完成但 ready attacker 不足时继续服务于攻击线，只有
  当前攻击与下一只接力都安全后才转向 Dunsparce 引擎。
- Active Fezandipiti ex 或 Shaymin 在 Bench 有已充能 Abra/Kadabra/Alakazam 时，
  可以为了明确的有效攻击交接支付 Retreat Energy，不再只认可 Alakazam。
- Xerosic 保留可见的 lethal-to-non-lethal 判断，同时恢复 V5 的“对手是 Alakazam
  deck 或手牌至少 8 张”的高置信 fallback，不要求知道对手隐藏手牌内容。

### F1 未改变的边界

- 不修改 V5 牌表，不加入公开 notebook 的卡组、随机性或 2-ply search。
- 保留手牌/牌库/Prize reserve、Fezandipiti 暴露限制、Mist Energy 特殊处理、
  Telepath 合法目标和能量重复保护。
- F1 目前是策略修正版，尚未以新的 Kaggle 16×10 结果证明胜率提升；正式评测仍
  需要与 V5 使用相同对手矩阵和交换先后条件进行比较。
