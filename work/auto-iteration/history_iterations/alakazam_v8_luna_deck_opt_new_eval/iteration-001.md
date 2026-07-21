# Iteration 001：立即干扰与最大化接力回收

## 变更

- `work/alakazam_v8_current/main.py`
  - 新增 `NIGHTTIME_MINE=1266` 并纳入 Item Lock 过滤集合。
  - 只要对手场上存在合法 Special Energy 目标，Enhanced Hammer 优先于攻击；目标选择仍保留 Active 保护性能量优先。
  - Sacred Ash 在 `maxCount` 内尽量选满，Abra 系列按 Abra → Kadabra → Alakazam 的接力顺序，再选择 Dunsparce 系列。
  - Lana’s Aid 在可见弃牌区存在 Abra 系列时优先恢复尽可能多的攻击线 Pokémon，并把 Basic Psychic 放到 Pokémon 之后。
- `tests/test_alakazam_v8_sol_deck_opt_strategy.py`
  - 新增 6 个最小行为 fixture，覆盖上述四项变化，并保留 Dudunsparce/Fezandipiti 既有路径回归。

## 测试证据

- RED：Hammer、Nighttime Mine、Sacred Ash、Lana’s Aid 四项测试在旧代码失败。
- GREEN：定向测试 `6/6` 通过。
- 回归：`python3 -m unittest discover -s tests -p 'test_*.py'`，`411/411` 通过。

## 评测

- Run：[`run-1fe111daa83241a38a6e36b8d9233f8f`](iteration-001/run-1fe111daa83241a38a6e36b8d9233f8f/)
- 口径：17 个 opponent × 10 局，profile `auto_iteration_v8_setup_relay` revision 2。

| 指标 | Baseline | Iteration 001 | 变化 |
|---|---:|---:|---:|
| 总体胜率 | 103/170 = 60.6% | 111/170 = 65.3% | +8 胜 |
| Powerful Hand | 34/170 = 20.0% | 37/170 = 21.8% | +3 |
| Post-KO relay | 154/523 = 29.4% | 170/469 = 36.2% | +6.8pp |
| recoverable_discard_miss | 130 | 87 | -43 |
| 攻击未拿奖赏 | 137/584 = 23.5% | 128/561 = 22.8% | -0.7pp |
| Dunsparce bridge | 0/44 | 0/42 | 未改善 |
| 空 Bench Run Away Draw | 0 | 0 | 保持 |
| Rare Candy 进化链 | 34/170 = 20.0% | 37/170 = 21.8% | +3 |
| 低牌库消耗审计值 | 2.84 | 2.92 | +0.08 |

## 正确性与结论

- Baseline 有 1 个、Iteration 001 有 2 个 error；两轮均全部来自 `yakitori_raging_bolt` 的 engine error。
- 没有 candidate error、非法动作或未完成对局证据。
- 结果与 Post-KO 指标同步改善，说明最大化恢复目标命中了部分接力断档；但 Dunsparce bridge 仍为零，不能宣称 Dudunsparce 换位路线已经完成。
- 下一轮只继续验证两个具体问题：首回合 Poké Pad 是否过早消耗，以及 Active Dunsparce → Dudunsparce → Run Away Draw 的真实选项链是否被正确保持。
