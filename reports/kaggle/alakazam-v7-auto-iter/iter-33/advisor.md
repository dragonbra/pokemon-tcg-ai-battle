# AutoIter iter-33 Advisor：Dudunsparce 的进化堆叠与接力风险

## 复核结论

本轮的规则/卡牌复核确认，Run Away Draw 洗回的对象不是一张孤立的 Dudunsparce：
它包括该实例的进化前卡牌和附着卡。因此牌库净变化必须按实际返回张数计算；当
Dudunsparce、Dunsparce 和 Enriching Energy 都被洗回时，抽 3、放回 3，净变化为 0。

这项修复属于牌库状态模型的正确性修复，不应被解释成“手上有 Dudunsparce 就一定要抽牌”。
空 Bench 时仍不能为了抽牌使用 Run Away Draw；攻击前仍要以真实可执行的 Abra-line 接力
路线为准，不能把孤立的 Stage 1/Stage 2 当成当前回合可用打手。

## 对最新 trace 的建议

- 二回合 `Powerful Hand` 未发生的诊断项里，先排除合法动作不足，再寻找真正被策略顺序
  压过的攻击选项。
- post-KO 无 ready attacker 不能只看击倒后的状态；需要回看击倒前一回合是否已经暴露
  可执行的 Poffin、Telepath、Basic 放置或弃牌区 recovery 路线。
- 下一轮保持单变量，优先分析可在击倒前完成的 Bench 锚点，而不是放宽所有资源 gate。

本 advisor 不修改生产代码；结论仅用于本轮记录和下一轮 case 选择。
