# 0019 通用胜者行为克隆

**项目 ID：** `0019_universal_winner_bc`
**当前阶段：** 0710–0728 数据与 V1 通用 BC 已完成；V1 选择 Epoch 13
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

最终 0710–0728 数据已经完整构建并通过全量 hash、dtype、shape 与 aggregate 校验：

- 19 个官方日快照，89,550 个 archive Episode members；
- 89,503 局唯一胜者 Episode，排除 48 局无唯一正奖励胜者的 Episode；
- 509 个 source/team、367 套 exact deck；
- train 80,767 局 / 6,640,006 decisions；validation 8,736 局 / 707,126 decisions；
- raw 为 2,386,657,033 bytes（324.84 bytes / decision）；
- model-ready tensor 为 54,155,429,609 bytes（7,370.96 bytes / decision）；
- catalog + raw + tensor 合计 56,617,027,690 bytes，即 52.73 GiB。

最终 catalog SHA-256 为
`51d0e99309acdc1166f62465dc577f6310c2560d032dd6ec50b567277211a008`，
model-ready content SHA-256 为
`141aa82ac8a162cec5e6e57970d4020024fc0c03ef408b89c4dfa538c870b724`。

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

source persona 是额外的可审计 residual embedding。最终词表含 509 个非中性 source，完整
模型共 17,756,162 个参数。部署使用 `source_id=0`，因此不会假设某个训练玩家仍在场。

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

V1 的 17 个完整 epoch 实测共 54,529.25 秒（15.15 小时）。Epoch 4–17 的稳定期平均
2,803.56 秒，约 46.73 分钟 / epoch；到 Epoch 13 的累计训练时间为 11.99 小时。这些实测值
替代原先按 0016 外推的 70–95 分钟估算。

checkpoint 只保存模型权重、epoch、global step 与必要 metadata，不保存 optimizer、RNG 或恢复训练轨迹。任何新训练语义都创建新的 `V<n>_<tag>`，不向旧 run 续写。

W&B 固定使用 private `dragon_bra/pokemon-tcg-policy-learning`，run name 为
`0019 · universal_winner_bc · V<n>_<tag>`。

## V1 结果与中断边界

`V1_universal_winner_r15` 完成了 17 个完整 epoch。系统在 2026-07-30 05:20:27 重启时，
Epoch 18 已完成 train pass，validation 进行到 480/1,382 batches；因此 V1 被记录为
`interrupted_with_valid_selected_checkpoint`，不伪装为正常 early stopping。

既定选择指标 `bc/validation/loss` 的最佳点是 Epoch 13：

- conditioned validation loss：`0.253080647`；
- conditioned greedy exact-action：`81.2871%`；
- conditioned teacher exact-action：`81.2859%`；
- conditioned legal-action：`100%`；
- checkpoint SHA-256：
  `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`。

Epoch 13 同时是 conditioned loss、teacher exact 和 greedy exact 的最佳点。Epoch 14–17
连续四次未改善；Epoch 18 没有完整 validation，不能形成指标或 checkpoint。

中性 persona 的最低 validation loss 出现在 Epoch 9（`0.365444490`），而 Epoch 13 为
`0.368793374`。这不改变 V1 的既定选点，但证明 conditioned expert imitation 与
`source_id=0` 部署目标之间存在可测差距。上述 BC 指标只证明模仿质量，不构成 official-engine
胜率证据。

## 2026-07-30 source persona 设计复核

source conditioning 最早在 `0015_dragapult_conditioned_bc` 引入。0014 R15 没有 persona；
0015 为避免多个 team 的动作标签无条件冲突，将研究目标显式定义为 `pi(a|s,D,E)`，并在
目标 candidate 固定 THIRD PTCG Club persona。0016 继承该结构，0017 使用 target source 1，
0018 使用 source 0；0019 又将相同结构扩大到 509 个非中性 source。这个设计适合“复刻指定
老师”的 conditioned imitation，但不等于项目现在要求的 source-agnostic `pi(a|s,D)`。

Epoch 13 在完全相同的 707,126 个 validation decisions 上，conditioned greedy exact 为
`81.2871%`，neutral greedy exact 为 `74.8431%`，相差 `6.4440` percentage points；conditioned
loss 为 `0.253080647`，neutral loss 为 `0.368793374`。因此 81.2871% 不能再被解释成最终
neutral deployment 的离线模仿能力。checkpoint 中 persona 分支共有 368,320 个参数；source 98
的 embedding norm 为 `1.2938`（接近全部非中性 source 的中位数 `1.2966`），经过 gate 与投影后
产生 state residual norm `3.8120`、option residual norm `1.0897`。这证明 persona 路径不是仅供
报告的标签，但参数规模和 norm 本身不能替代 official-engine 行为对照。

复核结论是：V1 数据、因果 feature、exact-deck conditioning 和共享权重仍是有效历史资产，
但 conditioned checkpoint selection 高估了与最终 source-0 forward 对齐的 BC 指标，并允许
模型把一部分老师特定决策放入部署时被清零的 residual。后续通用 BC 不再使用 persona embedding、
source dropout 或 conditioned-to-neutral persona distillation；team/source 只保留为 provenance、
Episode split、去重、分层采样和分组评测字段。模型目标改为 `pi(a|s,D)`，并必须按同一个
source-free forward 的 validation 指标选 checkpoint。

## 下一阶段

V1 baseline 已完成并作为 legacy source-conditioned checkpoint 保留。下一项干净对照应从
相同 audited winner 数据训练一个完全没有 source 参数或 source tensor 的通用 Actor：始终保留
exact deck，按 source/deck 分组报告冲突与长尾表现，必要时仅在采样层限制超大来源，最终按
source-free validation 与 deck-specific official-engine 评测共同选点。该工作必须使用新的严格
递增版本，不向 V1 追加训练轨迹，也不得把 0020 的单 deck PPO 曲线当成 persona-free BC 的替代证据。
