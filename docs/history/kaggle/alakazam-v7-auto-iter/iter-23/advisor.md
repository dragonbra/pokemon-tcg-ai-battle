# Iter-23 Strategy Advisor

## 核验结论

本轮验证 Lana's Aid 的一个确定性接力闭环：Active Kadabra/Alakazam 已附 Psychic，
Bench 没有 Abra-line 接班者但仍有空位，弃牌区同时可见 Abra 与 Basic Psychic。此时
Lana's Aid 可以一次把两个资源取回；随后下一次主动作把 Abra 放到 Bench，再用合法的
手动填能量提前准备下一回合。这个动作链符合“攻击宣告立即结束回合”和“新下 Basic 本
回合不能进化”的规则。

该路线只在非终局、没有现成 Bench Abra-line 时成立。若已有未充能的 Bench Kadabra
或其它可见接力目标，沿用原有直接附能逻辑；若本次攻击拿最后奖赏，攻击仍优先。

## analyzer 注意事项

本批有 3 个 `bench_insurance_missed=fail`。原始 trace 显示至少一例实际选择 Lana's
Aid 后继续完成铺场，当前 analyzer 的 expected action 集合尚未识别这个恢复链；这类
记录需和真正“有可用资源却没有铺场”的 case 分开，不能直接当作策略回归。
