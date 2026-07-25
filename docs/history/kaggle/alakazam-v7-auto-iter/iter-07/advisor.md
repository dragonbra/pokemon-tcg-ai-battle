# AutoIter iter-07 Advisor：Rock Fighting Energy 属性边界

## 规则核对

官方卡表和引擎均表明：Mist Energy（11）对任意附着宝可梦生效；Rock Fighting
Energy（20）只有附着在 Fighting 属性宝可梦上时，才通过 `NoEffectEnemyAttack`
阻止 Alakazam `Powerful Hand` 的 `DamageCounter`。

当前 observation 的 Pokémon 结构没有直接携带属性，提交包只能通过同一引擎的
`cg.api.all_card_data()` 建立 `card_id -> energyType` 缓存。若轻量 trace loader 无法
加载卡表，对未知 Rock 目标按“可能受保护”处理，避免错误宣称确定 KO。

## 本轮改动

- 将 Rock Fighting Energy 从“无条件保护”改为 Fighting 条件保护。
- Mist 仍对所有附着目标保护。
- 攻击伤害、KO、Boss/终局预测与 Enhanced Hammer 继续共用同一保护判断。
- 新增 Fighting Rock 正例与非 Fighting Rock 反例。

## 结论

规则修正与本地 fixture 一致，但完整评测使用独立随机批次，不能把整体下降归因于
本轮改动；不晋级为稳定 control，继续观察后续策略迭代。
