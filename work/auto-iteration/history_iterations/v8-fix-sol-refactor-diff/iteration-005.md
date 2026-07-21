# Iteration 005：收紧临时 Active 附能并恢复首回合 Poké Pad 目标

## 假设与变更

Iteration 004 暴露两个配套边界：

1. Shaymin Active 同时可用 Poffin 和 Telepath 时，历史 V8 先用 Poffin 建 Bench，上一轮新增
   的撤退能量 intent 条件过宽，错误抢先附能。
2. 首回合 Poké Pad 同时可搜 Kadabra 与 Dunsparce 时，历史 V8 选择 Dunsparce，重构版固定
   优先 Kadabra。这与连续多轮 `Dunsparce bridge=0` 的异常一致。

本轮把临时 Active 附 Telepath 限制为“没有可见 Poffin”时才触发；并恢复 Poké Pad 首回合
优先 Dunsparce、没有 Dunsparce 时才选 Abra 的效果目标语义。

## 测试

- 新增 parity 边界：Poffin 先于临时 Active 的 Telepath 附能。
- 新增 parity 行为：首回合 Poké Pad 选 Dunsparce 而非 Kadabra。
- 两项 RED 分别为当前版 `[0]`、历史 V8 `[1]`；修复后均 GREEN。
- V8/parity 测试：58 项通过。

## 17×10 结果

- run id：`run-d47961dfcf004b6ab8ff9945d07c562b`
- 完整报告：[`iteration-005/run-d47961dfcf004b6ab8ff9945d07c562b/report.html`](iteration-005/run-d47961dfcf004b6ab8ff9945d07c562b/report.html)
- W/L/D：`75/95/0`
- 胜率：`75/170 = 44.1%`
- candidate error：`0`
- 实际先手：`45/85 = 52.9%`
- 实际后手：`30/85 = 35.3%`
- 第二回合 Powerful Hand：`6/170 = 3.5%`
- 到达第二回合口径：`6/164 = 3.7%`
- 二回合前额外过牌：`516/170 = 3.04` 张/局
- Post-KO 成功：`149/541 = 27.5%`
- Dunsparce bridge：`0/53`
- 空 Bench Run Away Draw：`0`
- 攻击但未拿 Prize：`152/493 = 30.8%`

## 结论

相较 Iteration 004，胜场、Powerful Hand、过牌和 Post-KO 均回升，说明收紧过宽附能条件和
恢复首回合 Poké Pad 目标方向正确；但 Dunsparce bridge 仍未完成，两个核心指标也远低于
历史 V8。下一步继续复放本轮保留的三个 candidate 失败 trace，定位搜到/铺出基础 Pokémon
之后的下一处动作链断点。
