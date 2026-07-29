# 0020 可插拔卡组 RL

**项目 ID：** `0020_pluggable_deck_rl`
**当前版本：** `V1_frozen_0019_epoch13`
**当前阶段：** 0019 通用 checkpoint 的零训练跨卡组泛化验收已完成；尚未开始 RL

## 证据边界

- 通用官方规则由 official engine runtime 执行，包括攻击终止回合、进化时机、每回合 Supporter/手填能量/Retreat 限制与胜负条件。
- 当前卡牌与 runtime 事实来自 `data/official/`、候选 package 的 exact `deck.csv` 以及 official engine 产生的真实对局。
- 项目假设是：0019 的共享 Encoder 与合法动作 Decoder 能依靠 exact deck conditioning，在未见卡组上产生有意义的零训练策略。合法动作率或离线 exact-action 不能单独证明该假设，最终以 Arena 对局为准。

## 固定基石

0020 物理复制并冻结 0019 正式选中的 `epoch-0013-da9b13d6f82d19d4.pt`，SHA-256 为 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`。它是 epoch 13、global step 337194 的 model-only checkpoint，只有 `model`、`epoch`、`global_step`、`metadata`，不含 optimizer、scheduler、RNG、rollout buffer 或可精确续跑状态。

运行时实现、feature compiler、causal knowledge state 与 card ontology 均固化在 0020 内，不 import 0019 或其他编号训练项目的可执行代码。原 checkpoint 永不覆盖；未来从该权重开始训练必须建立新的 0020 版本、重建 optimizer 并重新采集 on-policy rollout。

## 输入与模型

模型保持 0019 R15 合同：宽度 320、8 heads、4 层 board Transformer、1 层 event encoder、4 个 Goal-QKV roles、2 层 scenario Transformer，参数量 17,756,162。主要输入包括：

| 输入 | shape | 0020 语义 |
|---|---:|---|
| `global_cat/global_num` | `[B,Cg] / [B,Ng]` | 回合与资源状态 |
| `entity_cat/entity_num/entity_mask` | `[B,E,7] / [B,E,5] / [B,E]` | actor-visible 场面实体，`E<=192` |
| `option_cat/option_mask` | `[B,O,12] / [B,O]` | official engine 合法候选，`O<=128` |
| `registered_card_ids` | `[B,D]` | 当前候选 exact 60-card deck 的去重卡 ID |
| `registered_multiplicity/mask` | `[B,D] / [B,D]` | 每种卡的 exact 投入与有效位 |
| `ledger_cat/ledger_num/ledger_mask` | `[B,D,4] / [B,D,15] / [B,D]` | 与当前构筑一致的因果资源账本 |
| `event_cat/event_num/event_mask` | `[B,T,8] / [B,T,4] / [B,T]` | 当前局 actor-visible 事件序列 |
| `known_opponent_hand_*` | `[B,H]` | 已知/未知对手手牌边界 |
| `source_id` | `[B]` | 部署固定 `source_id=0`，不冒充任何 expert persona |

输出为最长 64 步的 autoregressive legal-option pointer 序列加 STOP。零训练测试只替换 candidate 的 exact deck；checkpoint、ontology、推理参数和 neutral persona 完全相同。

## 零训练验收矩阵

五个 candidate deck 固定取自正式 Arena：0018 最终胡地 `alakazam_dudunsparce_04_sota`、`dragapult_ex_03_v20260729_rl`、`marnies_grimmsnarl_ex_froslass_05_bc`、`mega_lucario_ex_solrock_07_bc`、`mega_kangaskhan_ex_crustle_01_v20260729_bc`。Team Rocket 不作为 candidate。

每个 candidate 独立对 SHA-256 为 `92221e5a6b080d46f85d499e491a371a610a94087d7f001a1663bc657cfbe5c0` 的完整 30-opponent catalog 跑 10 局/对手，使用 `--workers 8 --worker-cpu-threads 1`。Team Rocket 没有作为 candidate，但按正式 `--opponents all` 合同仍属于共同 opponent pool。五个正式报告分别占用 V2–V6，避免将不同 exact deck 的结果混进一个逻辑版本。

| 版本 | exact deck | 胜-负 | 胜率 | 先手 | 后手 | 完成/error |
|---|---|---:|---:|---:|---:|---:|
| `V2_zero_shot_dragapult` | Dragapult ex 03 | 127-173 | 42.33% | 43.33% | 41.33% | 300/300, 0 |
| `V3_zero_shot_alakazam` | Alakazam / Dudunsparce 04 | 219-81 | 73.00% | 76.67% | 69.33% | 300/300, 0 |
| `V4_zero_shot_marnie` | Marnie's Grimmsnarl ex / Froslass 05 | 240-60 | 80.00% | 84.67% | 75.33% | 300/300, 0 |
| `V5_zero_shot_lucario` | Mega Lucario ex / Solrock 07 | 163-137 | 54.33% | 61.33% | 47.33% | 300/300, 0 |
| `V6_zero_shot_kangaskhan` | Mega Kangaskhan ex / Crustle 01 | 215-85 | 71.67% | 74.67% | 68.67% | 300/300, 0 |
| **合计** | 5 个不同构筑 | **964-536** | **64.27%** | **68.13%** | **60.40%** | **1500/1500, 0** |

## 后续 RL 边界

五套牌的零训练 official-engine baseline 已完成。结果支持“共享 checkpoint 具有实用跨卡组泛化能力”，但不同构筑差异很大，尤其 Dragapult 42.33% 与 Marnie 80.00% 不能被一个总均值掩盖；后手综合 60.40% 也低于先手 68.13%。下一阶段可以选择首个 deck-specific RL 分支，但每套牌的 rollout、reward/value calibration、optimizer、opponent pool snapshot、checkpoint 与评测版本必须相互隔离；共享基石不因任何单 deck 更新而改变。V1–V6 均没有训练或 W&B run。

## CUDA rollout 研究边界

根目录 `engine_cuda/` 已从来源 commit
`bf56dfb4b1d58279f0cd1e49a646b46174d8a589` 收编，但当前只是 CUDA vertical
slice，不是 official engine 的完整语义转写。它只有少量通用 smoke opcode；最强的
status parity 证据来自硬编码两张宝可梦与两种攻击的独立 micro-engine，而不是通用
interpreter。它的 codec 容量还是 `128 entities / 80 options / entity_cat width 6`，与
0020 的 `192 / 128 / width 7` 合同不兼容。

因此 CUDA prototype 当前不得产生 0020 on-policy rollout、reward/value target、策略
强度结论或正式评测。即使允许 RNG 实现与官方 `random_device` 不同，legal options、
action transition、public observation、hidden digest、RNG 消耗顺序、reward 与 terminal
result 仍必须逐步 differential parity。完整审计与准入 gate 见
`CUDA_ENGINE_AUDIT.md`；在 gate 通过前，official engine runtime 仍是唯一训练真值与
评测真值。
