# Iteration 003：恢复开局 Bench Pokémon 选择

## 假设与变更

`crustle_wall-001` 的完整 candidate observation 复放显示，官方引擎在开局使用
`context=2`（`SETUP_BENCH_POKEMON`）提供一个可选的 Dunsparce。旧版选择 `[0]`，重构版
因为 `minCount=0` 且 dispatcher 没有开局 Bench 分支而返回 `[]`，导致所有这类开局 Bench
选择都被跳过。

修复后，开局 Bench 选择按历史 V8 在当前卡组中的 Basic 优先级
`Abra > Dunsparce > Fezandipiti ex > Shaymin` 选择一个；没有顺带修改开局 Active 或其它
effect 选择语义。

## 测试

- 新增 parity 测试：可选开局 Bench 存在 Dunsparce 时必须选择。
- RED：旧版 `[0]`，当前版 `[]`。
- GREEN：旧版和当前版均为 `[0]`。
- V8/parity 测试：55 项通过。
- candidate validate：60 张卡、deck hash 和 cg tree hash 均合法。

## 17×10 结果

- run id：`run-c770941dfd7349f9a5570a54bf6fa19a`
- 完整报告：[`iteration-003/run-c770941dfd7349f9a5570a54bf6fa19a/report.html`](iteration-003/run-c770941dfd7349f9a5570a54bf6fa19a/report.html)
- W/L/D：`78/90/0`，另有 2 个 `yakitori_raging_bolt` 对手侧 engine error
- 胜率：`78/170 = 45.9%`
- candidate error：`0`
- 实际先手：`45/85 = 52.9%`
- 实际后手：`33/85 = 38.8%`，其中 2 局为对手侧 engine error
- 第二回合 Powerful Hand：`9/170 = 5.3%`
- 到达第二回合口径：`9/161 = 5.6%`
- 二回合前额外过牌：`559/170 = 3.29` 张/局
- Post-KO 成功：`129/481 = 26.8%`
- Dunsparce bridge：`0/61`
- 空 Bench Run Away Draw：`0`
- 攻击但未拿 Prize：`152/473 = 32.1%`

## 结论

开局 Bench 语义已与历史 V8 对齐，但本轮只比 Iteration 002 多 1 次第二回合 Powerful Hand，
胜率仍未恢复，Dunsparce bridge 也仍为零。保留这项规则修复；下一步从本轮唯一保留的
candidate 失败 trace `crustle_v1-002` 继续定位更早的启动链分歧。由于引擎不能固定 seed，
不能把本轮与上一轮解释为逐局配对，也不能仅凭 4 胜波动判断该单项修复的因果影响。
