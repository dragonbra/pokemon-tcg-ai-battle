# AutoIter iter-32 Advisor：弃牌区 option 的 `indexInArea` 解析

## Advisor 复核结论

本轮启动了独立的规则/卡牌 advisor 与 replay advisor。两者均未修改生产策略。

- `type=3`、`area=3` 的弃牌区选择中，`index` 是当前 select option 的局部序号，
  `indexInArea` 才是弃牌区中的真实位置；如果只读取 `index`，Night Stretcher 或
  Lana's Aid 可能把错误的牌识别为 Abra、Basic Psychic 或进化宝可梦。
- 该解析修复不应扩大成“手里有资源就禁止攻击”。没有实际 Bench target 或合法
  recovery route 时，孤立的 Basic Psychic、Kadabra、Alakazam 仍不能被假设成接力线。
- Poffin、Telepath、Dudunsparce、Lana's Aid 的既有 gate 保持不变；Mist Energy 与
  满足 Fighting 条件的 Rock Fighting Energy 仍按伤害保护处理。

## 本轮核验范围

- 新增 fixture 覆盖 `index` 与 `indexInArea` 不同的弃牌选项。
- 最新 170 局 trace 中没有出现 `area=3` 选择，因此本轮完整评测不能证明真实
  Night Stretcher 动作链已经被触发；该修复的直接证据来自 fixture 与历史 replay
  option 形状。
- advisor 建议保留当前策略 gate，继续从下一批真实 trace 中寻找可复现的 recovery
  接力 case。
