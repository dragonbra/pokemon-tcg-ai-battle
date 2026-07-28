# 0019 通用胜者行为克隆

**项目 ID：** `0019_universal_winner_bc`
**当前阶段：** 0710–0727 全量 catalog 扫描中；0728 在 09:00 后检查
**目标：** 不针对多龙或任何单一卡组，学习官方对局中所有胜者的通用场面理解与动作知识

## 证据边界

- 通用官方规则：攻击结束回合；进化、Supporter、手填能量、撤退与场地遵守官方时机和次数约束。
- 当前卡牌与 runtime 事实：卡牌 ID、完整合法选项、注册卡组、动作顺序和对局结果均来自当前官方 Episode 与官方引擎合同。
- 项目假设：高覆盖的胜者视角 BC 能预训练更通用的 Encoder；它不保证任意特定卡组在 BC 后立即具备高胜率，最终强度仍需 official-engine Arena 与 deck-specific RL 验证。

## 数据合同

输入为 `data/raw/episodes/archives/` 中 2026-07-10 起的官方每日 ZIP。构建器逐成员读取，禁止完整解压落盘。每个 Episode 只有在以下条件全部成立时进入数据集：

1. 两名玩家的终态均可审计，且存在唯一正奖励胜者；
2. 胜者注册卡组恰为 60 张整数卡牌 ID；
3. Episode ID、ZIP member、TeamNames、先后手与 payload hash 一致；
4. 同一 Episode 多次出现时 payload 必须完全相同，否则 fail closed；
5. 只保留胜者视角，失败者动作不进入 0019。

每条 trajectory 显式保存 exact deck multiset、`deck_sha256`、Unicode NFC 精确 team identity、`source_id`、seat、Episode/player identity 和原始 payload hash。`source_id=0` 永远保留给部署时的中性 persona。不同 deck/team 的标签不是无条件混合；模型同时看到 deck conditioning 与 source conditioning。

split 以完整 Episode 为最小单元，对
`team | deck_sha256 | seat | episode_id` 做固定 SHA-256 后按 90/10 划分。validation 只用于 BC loss、teacher exact、greedy exact 和分组诊断；这些指标不冒充真实对战强度。

## 存储与规模

200 个真实 0727 Episode 的 smoke 中，199 局通过合同，共得到 17,885 decisions：

- 平均 `89.87 decisions / Episode`；
- raw gzip shard 为 `300.60 bytes / decision`；
- 以约 85k Episode 外推，raw 数据约 2.2–2.6 GiB；
- model-ready tensor 为 `7,116.72 bytes / decision`，按当前规模外推约 50.7 GiB；
- raw、tensor、catalog 与 manifest 合计预计约 53–56 GiB。

硬护栏为：0019 dataset 总量 `<100 GiB`、Linux 项目盘至少保留 50 GiB、Windows C 盘至少保留 30 GiB。构建开始前以及每个 shard 原子关闭后复查。中断或越界只留下明确失败状态，不能把 partial 目录伪装成完成数据集。

## 输入与模型

0019 物理固化 0016 V2 R15 实现，不在运行时 import 其他编号项目。主要张量为：

| 张量 | shape | 语义 |
|---|---:|---|
| `global_cat/global_num` | `[B, Cg] / [B, Ng]` | 回合、玩家、资源与全局状态 |
| `entity_cat/entity_num/entity_mask` | `[B, E, 7] / [B, E, 5] / [B, E]` | Active、Bench、手牌、弃牌等实体，`E <= 192` |
| `option_cat/option_mask` | `[B, O, 12] / [B, O]` | 官方引擎给出的合法动作候选，`O <= 128` |
| `registered_card_ids/multiplicity/mask` | `[B, D]` | exact 60-card 构筑的去重卡牌与数量 |
| `ledger_cat/ledger_num/ledger_mask` | `[B, D, 4] / [B, D, 15] / [B, D]` | 因果资源账本 |
| `event_cat/event_num/event_mask` | `[B, T, 8] / [B, T, 4] / [B, T]` | 当前对局已观察事件序列 |
| `known_opponent_hand_*` | `[B, H]` | 已知对手手牌与未知数量边界 |
| `source_id` | `[B] int32` | 精确 expert/team persona；0 为部署中性值 |
| `targets` | `[B, A+1]` | 最长 64 步的 ordered full-action pointer 序列与 STOP |

backbone 宽度 320、8 heads、4 层 board Transformer、1 层 event encoder、4 个 Goal-QKV roles、2 层 scenario Transformer。R15 option ScaleGate 初值为 `0.35`，state ScaleGate 初值为 `1.0`，两者保持可学习且范围为 `(0, 2)`。输出仍是 autoregressive legal-option pointer，不改变 official action contract。

source persona 是额外的可审计 residual embedding。词表大小由最终 catalog 动态决定，模型参数量将在 catalog 完成后精确记录。部署使用 `source_id=0`，因此不会假设某个训练玩家仍在场。

validation 会先做一次共享 backbone encoding，再分别应用真实 `source_id` 与中性
`source_id=0`，因此不会重复执行 Encoder。真实 source 指标使用 `bc/validation/*`，中性部署
诊断使用 `bc/validation_neutral/*`。V1 仍以真实 source validation loss 选点；neutral 指标用于
揭示 persona 与共享 backbone 的差距，不参与 V1 反向传播或 checkpoint 选择。

## 训练目标

训练使用 ordered full-action cross entropy。每个 epoch 对 train split 只做一次更新 pass，随后以 epoch 结束时固定模型完整扫描 validation，记录：

- `bc/optimization/loss`、token accuracy、teacher exact；
- `bc/validation/loss`、token accuracy、teacher exact、greedy exact、legal action；
- `bc/validation_neutral/loss`、token accuracy、teacher exact、greedy exact、legal action；
- 数据吞吐、epoch 时间、参数量、W&B/TensorBoard 同步状态。

初始配置为 AdamW、LR `3e-4`、weight decay `0.02`、batch 256、validation batch 512、BF16、early stopping patience 5、min delta `0.001`。正式选点只看 `bc/validation/loss`；exact-action 是模仿诊断，不是胜率。

按 0016 实测每 epoch 约 420–536 秒 / 760,106 train decisions 外推，0019 单 epoch 初估 70–95 分钟。若 validation loss 在 8–15 epochs 内早停，总训练约 9–24 小时；全量数据完成后以精确 decisions 和 shard shape 重算。

checkpoint 只保存模型权重、epoch、global step 与必要 metadata，不保存 optimizer、RNG 或恢复训练轨迹。任何新训练语义都创建新的 `V<n>_<tag>`，不向旧 run 续写。

W&B 固定使用 private `dragon_bra/pokemon-tcg-policy-learning`，run name 为
`0019 · universal_winner_bc · V<n>_<tag>`。

## 当前与下一阶段

当前工作包是：catalog → raw gzip → model-ready tensor → 全量合同验证 → 时间估算。0710–0727 完成后，09:00 后查询 0728；若存在则作为新增不可变 source 纳入最终 V1 数据 manifest。

下一阶段首先训练一个通用 R15 baseline。若时间充裕，优先做可解释的单变量方案：

1. **中性 persona 蒸馏：** 训练时同时约束 `source_id=0` 输出接近各 expert 条件输出，减少部署时 persona 落差；
2. **deck/source dropout：** 小概率将 source 置 0，但始终保留 exact deck，迫使共享 Encoder 学习跨玩家知识；V1 已记录 neutral validation，但不执行此 dropout；
3. **分层采样：** 只在超大来源明显支配梯度时，对 source/deck 做温和上限，不丢弃长尾数据；
4. **通用 Encoder + deck adapter：** 0019 作为共享预训练，后续冻结大部分 Encoder，只对特定构筑 Decoder/末层做 BC 或 RL。

上述方案在 V1 baseline 完成前不混入同一训练版本。
