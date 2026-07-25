# AutoIter iter-42 决策：撤回错误 handoff 假设，保持策略观察

## 1. 本轮尝试

- Control：iter-41-final-prize-energy-gate。
- Candidate：策略代码不变；只复核 iter-41 被标记的 `handoff_preparation_missed`，并把错误的验收假设改成两个边界测试：
  - Dudunsparce 抽 3 张后能形成最后 KO 时，保留抽牌优先级。
  - 抽牌不能形成即时 KO 时，具体的 Telepath Energy → Bench Alakazam handoff 优先。
- 工具改动：AutoIter `run` 默认只保存 summary；需要复盘的最新轮次显式传 `--save-traces`，避免每轮意外生成数百 MB 的完整 trace。
- 固定卡组：`deck.csv` 未修改，SHA-256 为 `208edd2df41ae66d11e81b4394762d1fe1cb8f34452265281e22b6bfe4fb744a`。

## 2. 关键 case 复盘

原 advisor 把 `maktha_1084-game-9-turn-13` 的 `trace[139]` 与 analyzer case 状态混淆了：

- `trace[139]` 的实际行动是 Dudunsparce Run Away Draw；手牌从 14 增加到 17，Powerful Hand 从 280 提升到 340，可以击倒 340 HP 的 Active `678`，因此不应被 Bench handoff 抢先。
- analyzer case 对应的攻击状态是 `trace[147]`：手牌 20，攻击造成 400；目标 `678` 是 3 Prize Pokémon，而我方只剩 2 张 Prize，攻击完成最后奖赏，直接攻击正确。
- 其余 11 个 `handoff_preparation_missed` 也逐一复核为终局攻击，或同类立即闭环；没有确认的非终局策略错误。

因此本轮没有修改 `main.py`，也不更新 `BEST_STRATEGY.json`。

## 3. 指标对比

本轮使用 iter-41 的同一批 170 个 full trace 重新生成轻量分析；由于策略代码未变，指标严格相同：

| 指标 | iter-41 control | iter-42 candidate | 变化 |
|---|---:|---:|---:|
| 胜 / 负 / 平 | 102 / 66 / 2 | 102 / 66 / 2 | 0 |
| 胜率 | 60.0% | 60.0% | 0.0pp |
| Meta 加权胜率 | 60.1% | 60.1% | 0.0pp |
| 第二回合 Powerful Hand | 21.8% | 21.8% | 0.0pp |
| post-KO 无 ready attacker | 62.0% | 62.0% | 0.0pp |
| 对局级接力断档 | 31.8% | 31.8% | 0.0pp |
| 空 Bench Run Away Draw | 0 | 0 | 不变 |
| 我方 action error | 0 | 0 | 不变 |

## 4. 采纳状态

**observe，不晋升。** 本轮接受 trace retention 的工具约定，但不把错误 case 当作策略改进，也不宣称胜率提升。下一轮继续优先审计真实的 post-KO 无 ready attacker：必须回溯击倒前一回合的合法选项，确认当时确实存在可执行的 Poffin、Telepath、Lana、Night Stretcher 或 Psychic 附能路线，才能形成策略 candidate。
