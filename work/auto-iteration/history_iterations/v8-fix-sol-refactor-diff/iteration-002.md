# Iteration 002：保留 Active Abra 的 Rare Candy 路线

## 假设与变更

两份真实 trace 显示，Active Abra 已带 Psychic Energy、Bench 有合法 Abra，手里同时有
Kadabra、Rare Candy 和 Alakazam 时，旧版先把 Bench Abra 自然进化为 Kadabra，保留
Active Abra 在同回合使用 Rare Candy 直达 Alakazam 的路线。重构版错误地先自然进化
Active Abra，导致该回合无法继续进化，只能以 Kadabra 结束启动链。

修复后，只有在 Active Abra 的 Rare Candy 路线完整且 Item 未被锁定时，planner 才先生成
`evolution.bench_abra_kadabra` goal；其它 Active Abra 场景维持原有顺序。

## 测试

- 新增 parity 测试：完整 Rare Candy 路线下先自然进化 Bench Abra。
- RED：旧版 `[1]`，当前版 `[0]`。
- GREEN：旧版和当前版均为 `[1]`。
- V8/parity 测试：54 项通过。

## 17×10 结果

- run id：`run-dfe2afb10770436381cbce95bb9c0f21`
- 完整报告：[`iteration-002/run-dfe2afb10770436381cbce95bb9c0f21/report.html`](iteration-002/run-dfe2afb10770436381cbce95bb9c0f21/report.html)
- W/L/D：`82/88/0`
- 胜率：`82/170 = 48.2%`
- candidate error：`0`
- 第二回合 Powerful Hand：`8/170 = 4.7%`
- 到达第二回合口径：`8/159 = 5.0%`
- 二回合前额外过牌：`571/170 = 3.36` 张/局
- Post-KO 成功：`155/452 = 34.3%`
- Dunsparce bridge：`0/57`
- 空 Bench Run Away Draw：`0`
- 攻击但未拿 Prize：`143/505 = 28.3%`

## 结论

第二回合 Powerful Hand 从 Iteration 001 的 3 次提高到 8 次，说明修复命中了真实启动链；
但胜率仍低于初始当前版的 83 胜，更远低于历史 V8 的 112 胜，不能据此宣称性能恢复。
下一步复放本轮保留的三个 Rare Candy 失败案例，定位第二回合的下一个首动作分歧。
