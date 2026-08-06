# 0031 到 0032 特征迁移

0032 的目标不是扩大 0031，而是把输入收敛为“推理时真实可见、确实进入 forward、
足够支持合法选项判断、且不能由其他保留输入精确重建”的最小合同。

## 张量宽度与容量

| 特征组 | 0031 categorical | 0032 categorical | 0031 numeric | 0032 numeric |
|---|---:|---:|---:|---:|
| 全局决策 | 11 | 7 | 19 | 11 |
| 卡牌实例 | 22 | 6 | 10 | 14 |
| 己方资源账本 | 4 | 3 | 15 | 10 |
| 因果事件 | 16 | 17 | 8 | 4 |
| 合法选项 | 13 | 11 | 10 | 2 |

- 默认参数量：`55,868,802 -> 20,567,042`，减少 `35,301,760`（63.19%）。
- 0032 仍保留全部可变长度卡牌、资源、事件和合法选项 token，不通过截断宽度减少实体。

## 新增或纠正

| 0032 项目 | 变化 | 原因 |
|---|---|---|
| `resolved_energy_histogram[12]` | 新增 | 直接保留 observation 的引擎解析能量类型与重复份数 |
| `appeared_this_turn_state` | 数值改四态分类 | 区分 false、true、未知和不适用 |
| `put_damage_counter`、`coin_head` | 改三态分类 | 区分 false、true 和日志未提供 |
| `result_type`、`reason_type` | 数值改离散分类 | 避免把枚举误当连续大小 |
| event 六种卡牌角色投影 | 新增 | 区分来源/目标、前场/后场、变化前/变化后 |
| event source/target 实例投影 | 新增 | 交换因果方向必须改变模型表示 |
| option 四种卡牌角色投影 | 保留并验证 | 区分来源、目标、上下文和生效卡 |
| Energy 子卡 histogram 状态 | 纠正为不适用 | 有效能量直方图属于其父宝可梦，不属于物理能量卡 |
| 严格 actor projection | 加强 | 未审计 replay 字段立即失败，不进入训练样本 |

## 保留

- `card_parent`：Energy、Tool、进化链卡到场上宝可梦的物理附属关系。
- `option_source`、`option_target`：Attach 等选项的具体来源卡和目标宝可梦。
- `event_source`、`event_target`：公开日志能由 serial 绑定到当前可见实例时的方向关系。
- 卡牌到攻击、卡牌到技能、技能区域、完整静态 Trigger subject。
- 能量卡 prototype 的 9-bit 类型 mask、引擎能量份数、条件标志和 Card-to-Skill 关系。
- 当前已贴附状态的 12 类引擎解析有效能量直方图。

## 删除的可重建字段

- 全局：己方手牌数、已知/未知对手手牌数、己方/对手后场数、合法选项数、最小/最大选择数。
- 卡牌：kind、serial、伤害值、物理能量/Tool/进化数量、有效能量总份数和类型 mask。
- 资源：各公开区域中按卡号重复统计的 active、bench、hand、discard、stadium 数量。
- 选项：重复的 select type/context、上下文剩余数、来源/目标 HP、攻击基础伤害、需求能量数。
- prototype：反向 attack/skill-to-card 关系；保留的 card-to-attack/card-to-skill 可精确重建它们。

## 删除的泄漏或答案字段

- effect node graph、`is_condition` 和 option-effect 关系。
- attach 后供能类型/份数、贴前/贴后能量缺口、newly enabled attack、撤退缺口。
- 伤害后 HP、KO 答案。
- observation 不提供的运行时 `TriggerInfo.subject/object`。
- 从 source/team/persona 推断专家身份的字段。

## 验收依据

- 56/56 单元测试覆盖字段宽度、取值、missingness、关系、角色交换和 forward 影响。
- 完整引擎 prototype：1,267 cards、1,556 attacks、433 skills。
- Episode `89228732`：68 个胜者决策；680 个场上宝可梦能量直方图逐项一致；
  520 个附属子卡全部有 parent；94 个 Attach 全部有 source/target。
- 默认 CUDA forward：8 条记录，teacher logits `[8, 4, 11]`，全部 finite。
- 正式五日数据：2026-07-28 至 2026-08-01，共 2,048,069 条胜者视角决策；
  train 1,851,008，validation 197,061。全部 shard、SHA256、逐记录关系、batch contract
  和精确模型 forward 已通过本地验收。
- 双 T4、随机初始化、无 W&B 的五轮全量训练已作为 Kaggle version 1 启动；训练结果在
  checkpoint 下载并通过完整验收前不作声明。

可重复审计命令：

```powershell
python -m train.0034_clean_recent3_semantic_bc.tools.audit_feature_inputs `
  path/to/episode.json
```
