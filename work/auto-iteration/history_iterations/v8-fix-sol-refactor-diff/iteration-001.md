# Iteration 001：Item Lock 铺场边界

## 假设与变更

真实 trace 显示 Item Lock 回合中，旧版会把手里的 Abra 铺到 Bench，当前版直接结束回合。
根因是 `setup.propose()` 在 `facts.item_lock` 时整体返回，错误屏蔽了非 Item 的 Basic
Pokémon。修复后只屏蔽 Poffin 和 Poké Pad，继续允许 Abra/Dunsparce 上场。

## 测试

- 新增 parity 测试：Item Lock 仍允许 Basic Abra 上场。
- RED：旧版 `[0]`，当前版 `[1]`。
- GREEN：旧版和当前版均为 `[0]`。
- V8/parity 测试：53 项通过。

## 17×10 结果

- run id：`run-5ab51a44bd2e429c9d8e9ab3a196e7e4`
- 完整报告：[`iteration-001/run-5ab51a44bd2e429c9d8e9ab3a196e7e4/report.html`](iteration-001/run-5ab51a44bd2e429c9d8e9ab3a196e7e4/report.html)
- W/L/D：`85/84/0`，另有 1 个 `yakitori_raging_bolt` engine error
- 胜率：`85/170 = 50.0%`
- 实际先手：`47/85 = 55.3%`
- 实际后手：`38/85 = 44.7%`
- 第二回合 Powerful Hand：`3/170 = 1.8%`
- 到达第二回合口径：`3/151 = 2.0%`
- Post-KO 成功：`138/442 = 31.2%`
- 空 Bench Run Away Draw：`0`
- 攻击但未拿 Prize：`161/492 = 32.7%`

## 结论

规则语义修复正确，但相对初始当前版仅增加 2 胜和 1 次第二回合 Powerful Hand，不能解释
主要性能下降。保留修复，下一步处理 Active Abra 错误自然进化导致 Rare Candy 路线中断。
