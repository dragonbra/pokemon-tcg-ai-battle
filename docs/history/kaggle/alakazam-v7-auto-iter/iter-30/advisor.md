# Iter-30 Strategy Advisor

## 独立结论

1. `type=7` 手牌 option 和 `area=2` 附能 option 使用 `indexInArea` 的修复是合理的；历史
   turn 7 case 确实存在局部 `index` 与原始手牌位置分离的情况，修复能让接力 gate 正确识别
   Bench Psychic attachment。
2. 当前解析范围仍不完整：`area=3` 弃牌以及 `area=1/6` 仍使用 `index`。下一轮可重点检查
   弃牌区回收效果的索引语义。
3. focused 20 局结果为 35.0% 胜率、70.4% post-KO 无 ready attacker，不能晋升 iter-30，
   应继续保留 `observe`，`BEST_STRATEGY.json` 仍指向 iter-15。

## 对下一轮的提示

advisor 指出 `sue_alakazam/game_005 turn 15` 的 Night Stretcher option 使用了
`index=0..9`、`indexInArea=7,8,9...` 的形式，当前仍可能按错误的弃牌位置读取 Abra/Psychic。
该问题可能造成接力断线；本轮 V31 先验证已有 replay 明确暴露的 Telepath Active 锚点，弃牌
索引问题作为后续独立 case，不与本轮变量混合。

## 范围声明

advisor 只读分析，没有修改生产代码、测试或卡组文件。
