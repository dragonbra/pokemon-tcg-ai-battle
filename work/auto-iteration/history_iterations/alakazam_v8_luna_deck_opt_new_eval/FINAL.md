# Alakazam V8 Luna Deck Opt New Eval：最终记录

## 交付边界

- 策略源：`work/alakazam_v8_current/main.py`
- 卡组与官方 `cg/` runtime：本轮未修改
- 评测记录：本目录；没有复制第二份策略源
- 发布归档：`submission/dist/alakazam_v8_luna_deck_opt_new_eval.tar.gz`
- 归档 SHA-256：`6bc154fdb95d247c5e83e6312e8bb6834031d6458a7986c6bd9b332d648c6613`
- Kaggle submission：已提交一次；CLI 返回 `Successfully submitted`

## 最终评测

- Run：[final/run-4b116f7130c64dd08f84e0f55e28b943/](final/run-4b116f7130c64dd08f84e0f55e28b943/)
- 口径：17 个 opponent × 10 局，共 170 局
- Profile：`auto_iteration_v8_setup_relay` revision 2
- Candidate correctness：`0/170` error；完成率 `170/170`
- Runtime hash：`8a25d7b3497cf4e7c950f3b5038be5debf0f8157fd24abdfd3a730fe863b60ac`
- Deck hash：`0598646548d081832ec311c15fdc369b32c6f5e63175b0cfd1904d21fd082451`

| 关注指标 | Baseline | Final | 变化 |
| --- | ---: | ---: | ---: |
| 总体结果 | 103W / 66L / 1 error，60.6% | 118W / 52L，69.4% | +15 胜，error 清零 |
| 实际先手 | 56W / 28L / 1 error，65.9% | 64W / 21L，75.3% | +8 胜 |
| 实际后手 | 47W / 38L，55.3% | 54W / 31L，63.5% | +7 胜 |
| Powerful Hand | 34/170，20.0% | 41/170，24.1% | +7 |
| Post-KO relay | 154/523，29.4% | 192/469，40.9% | +11.5pp |
| recoverable_discard_miss | 130 | 79 | -51 |
| 攻击未拿奖赏 | 137/584，23.5% | 136/658，20.7% | -2.8pp |
| Powerful Hand 未拿奖赏 | 71/513，13.8% | 74/589，12.6% | -1.2pp |
| Dunsparce bridge | 0/44 | 0/45 | 未改善 |
| 空 Bench Run Away Draw | 0 | 0 | 保持 |

最终报告的逐 opponent 矩阵、原始 `metrics.json`、`games.jsonl`、`cases.jsonl` 和保留 trace 均在 final run 目录内。

## 已落地的 deck-notes 语义

1. 合法的 Nighttime Mine 优先覆盖对手 Stadium；Enhanced Hammer 只在场上存在 Special Energy 目标时优先处理，并保持 Active 目标优先。
2. Sacred Ash 在 `maxCount` 内尽量选满，按 Abra → Kadabra → Alakazam，再补 Dunsparce 系列；Lana’s Aid 优先恢复攻击线 Pokémon。
3. 上回合己方 Pokémon 被 KO 后，合法的 Fezandipiti ex `Flip the Script` 不再因 Prize 风险被跳过。
4. Active Dunsparce 面对后场已带 Psychic 的 Alakazam 时保留 Dudunsparce → `Run Away Draw` 接力路线；第一回合 Active Dunsparce 使用 Poké Pad 时优先建立 Abra 底座。
5. 第一回合已有足够的 Abra/Dunsparce 基础资源时保留 Poké Pad；Trading Places 和无价值的 Abra 攻击仍被禁止。

## 未解决项与解释

`Dunsparce bridge` 仍为 `0/45`。保留的 trace 表明部分 bridge opportunity 没有合法 Bench 接班者，不能仅凭 opportunity 计数强行执行 Run Away Draw；当前实现优先避免空 Bench Draw 错误，最终该指标为 `0`。最终 run 另有 3 个诊断案例，集中在 `rare_candy_not_played` 和 `no_legal_attack`，没有 candidate error。

## 验证记录

- 定向策略测试：`12/12` 通过
- 全仓库 unittest：`417/417` 通过
- `python3 -m compileall -q work/alakazam_v8_current evaluation scripts tests`：通过
- `python3 scripts/check_assets.py`：通过
- `git diff --check`：通过
