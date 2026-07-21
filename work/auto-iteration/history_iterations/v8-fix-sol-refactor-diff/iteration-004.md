# Iteration 004：临时 Active 的撤退能量准备

## 假设与变更

Iteration 002 的 `crustle_v1-005` 和 Iteration 003 的 `crustle_v1-002` 出现相同首动作
分歧：Active 分别是 Shaymin 和 Fezandipiti ex、手里有 Telepath Energy 时，历史 V8 会先
把 Energy 附给临时 Active，为其 1 点撤退费用准备后续 Alakazam handoff；重构版没有对应
intent，分别直接结束回合或先使用可选的 Nighttime Mine。

修复只在 Active 为 Fezandipiti ex/Shaymin、尚未附能且可见 Telepath 对准该 Active 时，
通过 continuity intent 提前附能。没有重排全局 policy 阶段，也没有扩大到其它 Pokémon 或
Basic Psychic。

## 测试

- 新增 parity 测试：Fez + Telepath + Nighttime Mine 时先附能。
- RED：旧版 `[0]`，当前版 `[1]`。
- GREEN：旧版和当前版均为 `[0]`。
- V8/parity 测试：56 项通过。

## 17×10 结果

- run id：`run-b3fea2785d53454f9068fe2d439dfe6a`
- 完整报告：[`iteration-004/run-b3fea2785d53454f9068fe2d439dfe6a/report.html`](iteration-004/run-b3fea2785d53454f9068fe2d439dfe6a/report.html)
- W/L/D：`67/103/0`
- 胜率：`67/170 = 39.4%`
- candidate error：`0`
- 实际先手：`37/85 = 43.5%`
- 实际后手：`30/85 = 35.3%`
- 第二回合 Powerful Hand：`5/170 = 2.9%`
- 到达第二回合口径：`5/161 = 3.1%`
- 二回合前额外过牌：`483/170 = 2.84` 张/局
- Post-KO 成功：`116/480 = 24.2%`
- Dunsparce bridge：`0/57`
- 空 Bench Run Away Draw：`0`
- 攻击但未拿 Prize：`159/440 = 36.1%`

## 结论

本轮两个核心指标都未改善。由于引擎不能固定 seed，不能把本轮与上一轮当作逐局配对；
同时 parity 证明这项动作确实属于历史 V8。暂时保留该修复，并继续恢复其后续搜索、撤退和
过牌配套语义，而不是依据单轮随机矩阵立即撤销。下一步复放本轮三个 candidate 失败 trace，
定位最早的配套链路断点。
