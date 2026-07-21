# AutoIteration 指标 Profiles

## 作用

指标 profile 是某一轮或某一阶段的可替换观测配置。它回答：

- 这轮具体观察什么；
- 为什么这些指标对应当前长期目标；
- 如何定义事件和分母；
- 哪些情形算不可达、策略漏做或真正资源耗尽；
- 如何展示和解释结果。

它不重新定义 AutoIteration 的生命周期、promotion 状态或通用规则。

## Active profile

当前 active profile 是 [`v8-setup-relay.md`](v8-setup-relay.md)。它服务于当前的
Alakazam 二回合基础能力和 post-KO 接力研究。

下一轮如果关注点变化，应新增一个独立 profile，并在 iteration 记录中引用它。不要
直接覆盖旧 profile，也不要假设不同 profile 的同名指标仍然有相同分母。

## Profile 必填字段

每个指标至少定义：

| 字段 | 内容 |
| --- | --- |
| `name` | 稳定的指标名 |
| `goal` | 对应 G0/G1/G2/G3 的目标 |
| `event` | 机会和成功事件的精确定义 |
| `numerator` | 成功事件的计数方式 |
| `denominator` | 全样本、机会样本、事件样本或对局样本 |
| `unavailable` | 未到达、资源缺失、不可达和真正耗尽的分类 |
| `exclusions` | 排除条件和理由 |
| `direction` | 越高越好、越低越好或目标区间 |
| `priority` | 目标指标、健康指标或结果护栏 |
| `presentation` | 总体、先后手、情形或对手维度 |
| `failure_action` | 变差时的 observe、reject 或复核方式 |

## Profile 变更规则

1. 运行前固定 profile 版本和分母；
2. 修改事件或分母时建立新 profile 版本；
3. 新旧 profile 不得直接做数值 A/B；
4. profile 只描述观察方式，不要求当前 `evaluation/` 立即拥有全部实现；
5. 每个 profile 必须说明哪些指标是主要目标，哪些只是健康检查。
