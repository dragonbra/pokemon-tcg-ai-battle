# Frozen-corpus long-action audit

## 结论

0010 builder 没有再次静默丢记录；它完整继承了 0009 的 238,356 条 source records。
真正的 coverage gap 位于更上游：同一批 3,104 个 winner-player replay group 实际包含
238,439 个该玩家的 visual decision frames，差出的 83 条全部是 expert action length >16，
而且 83/83 都没有进入 0009 dataset。

| 项目 | 数量 |
|---|---:|
| raw winner-player decision frames | 238,439 |
| frozen 0009 / 0010 records | 238,356 |
| missing long-action frames | 83 |
| maximum expert action length | 25 |

split 保持原 episode/date 边界不变时：

| split | 旧 records | 恢复 long action | 新 records |
|---|---:|---:|---:|
| train | 221,289 | 75 | 221,364 |
| validation | 17,067 | 8 | 17,075 |
| total | 238,356 | 83 | 238,439 |

长度分布：

| action length | 17 | 18 | 19 | 20 | 21 | 23 | 25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| records | 37 | 21 | 15 | 4 | 4 | 1 | 1 |

## 触发语义

- `effect_id=1197`，Xerosic’s Machinations：79 条。
- `effect_id=1087`，Hand Trimmer：4 条。
- train：Xerosic 71 + Hand Trimmer 4；validation：Xerosic 8。
- 全部为 `selectType=1` 且 `minCount=maxCount=expert action length`，因此不是可选地少
  丢几张；返回 16 个必然违反 simulator 数量合同。

## 对 V1--V7 的影响

- V1--V7 的内部消融仍使用完全相同的 238,356 条数据，因此相对比较没有因版本间
  数据不同而失去公平性。
- 但这些版本共同没有学习 83 条长弃牌动作，并且旧 inference 把 live min/max 截到
  16；这解释了 official-engine 对局中的 candidate `IndexError`。
- V9 live decoder 允许 GRU 外推到真实 minCount，修复了 candidate legality，但第 17 个
  及以后 choice 尚无训练监督，因此不能视为完整训练修复。

## 重训合同

- 继续使用相同 3,104 个 winner-player group、相同 episode-level split、相同 expert policy。
- 从原始 replay decision frames 恢复全部 83 条，目标总量应为 238,439。
- `max_action_steps` 至少为 25；为保留明确余量，建议设为 32。
- 新 dataset 必须记录 train/validation 新增数、长度直方图、effect 分布和输出 hash。
- 用户选定 V3/V5/V7 结构后只重训该一个结构，不改其他 feature 变量。

## 上游过滤位置

`train/kaggle_bc_top20/training/build_daily_winner_bc_dataset.py` 的 daily winner builder
使用硬编码 `len(action) > 16`，把这些 frame 记入
`reference_codec_limits_or_empty_options` 后继续。0010 builder 对其 source record 是一一编码，
因此没有第二次过滤。修复时应把上游 action 上限和 pointer config 绑定到同一个显式值，
避免 dataset 与 live decoder 再次漂移。
