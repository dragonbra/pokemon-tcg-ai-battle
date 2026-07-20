# AutoIter 28 Strategy Advisor

## 规则与卡牌核验

- Buddy-Buddy Poffin（1086）可以从牌库放下 Abra（741）或 Dunsparce（305）；它们都是
  HP 不超过 70 的 Basic Pokémon。
- 一个尚未附 Psychic Energy 的 Bench Kadabra/Alakazam 不是可以立即接班的打手。攻击一旦
  宣告，本回合不能再附能量、进化或使用 Ability，因此必须在攻击前处理可见的接力资源。
- 如果当前没有 Poffin、Telepath、Wondrous Patch、Lana's Aid 或合法的直接附能选项，不能
  因为“理论上应该准备 Bench”而盲目跳过当前攻击。

## 本轮假设

旧逻辑在 `_bench_insurance_due()` 中看到任意 Bench Kadabra/Alakazam 就直接返回 False，
把“存在 Stage 1/2”误当成“ready attacker 已经存在”。本轮移除这个过宽例外，继续由
`_has_visible_bench_handoff_route()` 和当前选项中实际可完成的 anchor 路线负责放行。

## 真实 replay 对照

- `kiyotah_dragapult/game_002.json` turn 7：Active Alakazam 已带 Psychic，但 Bench 两只
  Alakazam 都未附能；旧逻辑直接使用 Powerful Hand，下一回合被击倒后没有 ready attacker。
- `nursrijan_lucario/game_005.json` turn 3：Active Kadabra 已带 Psychic，Bench Kadabra 未
  附能，手牌有 Poffin；旧逻辑因 Stage 1 存在而选择 Dawn，后续攻击线断裂。
- `maktha_1084/game_008.json` turn 8：Bench 的 Alakazam/Kadabra 都未附能，手牌有 Poffin，
  旧逻辑选择 Boss；下一回合 Active 被击倒后仍没有可用打手。

这三个 case 的共同点不是“缺少 Pokémon”，而是“有 Pokémon 但没有 ready continuity”，
因此本轮只修保险资格，不全局抬高 Poffin、Dawn 或 Boss 的固定优先级。

## 验收 fixture

1. 未充能 Bench Kadabra + Poffin + 非终局 KO：选择 Poffin。
2. 未充能 Bench Kadabra、没有可见 anchor：保持攻击。
3. 已充能 Bench Kadabra：仍允许当前 Alakazam 使用 Powerful Hand。

`data/official`/engine 卡表与 `main.py` 的 Mist Energy、Rock Fighting Energy 保护逻辑
保持不变，本轮不修改 `deck.csv`。
