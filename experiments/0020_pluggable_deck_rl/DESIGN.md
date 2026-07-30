# 0020 可插拔卡组 RL

**项目 ID：** `0020_pluggable_deck_rl`
**当前版本：** `V10_raging_bolt_source98_persona_poc`
**当前阶段：** Persona PoC 与设计复核已完成；自爆多龙 V9 PPO 继续运行

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
| `source_id` | `[B]` | legacy 0019 checkpoint 输入；正式部署/RL 固定 0，未来通用 Actor 删除该张量 |

Actor 输出为最长 64 步的 autoregressive legal-option pointer 序列加 STOP。零训练测试只替换 candidate 的 exact deck；checkpoint、ontology、推理参数和 neutral persona 完全相同。

RL 分支在 actor 的共享 state `[B,320]` 上增加独立 value head：`LayerNorm(320) -> Linear(320,320) -> GELU -> Linear(320,1) -> Tanh`，输出 `[B]` 的 terminal outcome estimate。value head 有 103,681 个参数，actor + value 合计 17,859,843 个参数。最后一层以零权重和零 bias 初始化，因此 V7 开始时所有状态的 value 都严格为 0；它不改变 actor logits 或动作合同。

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

## Dragapult V7 value calibration

五套牌的零训练结果支持共享 checkpoint 的实用跨卡组泛化，但 Dragapult 的 42.33% 是最需要优化的低基线。V7 固定 Dragapult exact deck 与 SHA-256 为 `92221e5a6b080d46f85d499e491a371a610a94087d7f001a1663bc657cfbe5c0` 的 30-opponent snapshot，使用未修改 official engine、16 个隔离 worker、CUDA 集中 candidate inference 采集 512 个 sampled Episodes。terminal reward 只取胜 `+1`、负 `-1`、平 `0`，`gamma=1`；同一 Episode 的每个决策共享 terminal target，并用 `1 / episode_length` 加权，防止长局主导 loss。无效 Episode 必须丢弃；本次 512/512 有效、0 discarded，共 46,315 个决策。

V7 只训练 value head 4 epochs，batch 1024、AdamW learning rate `1e-4`、max grad norm `0.5`。value loss 从 0.859488 降到 0.604517，explained variance 从 0.062877 升到 0.224882；所有指标有限。验收确认 actor 的 269 个 state tensors 与 V1 源权重逐 tensor 相同，V7 model-only checkpoint 不含 optimizer、scheduler、scaler、RNG、rollout 或 replay。canonical 事实源是 `rl_runs/0020_pluggable_deck_rl/versions/V7_dragapult_value_calibration/artifact/training_metrics.jsonl`，W&B run 已 online synced。

## 猛雷鼓（Raging Bolt）V8 zero-shot

V8 使用 2026-07-30 Kaggle 榜首 `James Cox & Henry Chao` 的 exact 60-card Raging Bolt 构筑。身份链为 submission `54954310`、Episode `88830387`、player index `1`；candidate 只替换 exact deck conditioning，仍使用冻结的 0019 epoch-13 neutral actor，未进行任何训练。这不是常见的单线 Raging Bolt，而是罕见的五色 Area Zero toolbox：3 Mega Kangaskhan ex、3 Meowth ex、3 Teal Mask Ogerpon ex、2 Raging Bolt ex，并混合多种 Basic attacker 与 5 种基础能量。2026-07-30 Top 100 中只有这一份构筑包含 Raging Bolt ex（1/100，投入 2 张）；日报的粗粒度 archetype classifier 将它归入 `Mega Kangaskhan ex / Crustle`，但 exact deck 不含 Crustle。

0019 winner catalog 实际包含同一 exact deck 的 843 个获胜 Episode：James Cox 454 局（source 97）、James Cox & Henry Chao 116 局（source 98）、zoroark190 273 局（source 462）。因此 V8 的历史内部 `zero_shot` 名称只表示“没有为 0020 做额外梯度训练”，不能作为严格的 unseen-deck 泛化证据。V8 仍必须使用 `source_id=0`：这是正确的 neutral 部署合同，不是卡牌匿名化；exact Card ID、投入数量与 deck conditioning 全部保留。任何 source-conditioned 对照只能诊断 0019 将多少策略知识存入了 expert residual，不能作为正式强度、候选权重或 RL 起点。

对相同 30-opponent catalog 跑 10 局/对手，official-engine 结果为 84-216、胜率 28.00%，300/300 完成、0 draw、0 error。95% Wilson interval 为 23.22%–33.33%，因此相对 Dragapult V2 的 42.33% 不是小样本随机波动。相同 exact deck 在原 Kaggle agent 的当前冻结公开样本中为 529-366、59.1%，对 Top 100 submission 为 56.2%（n=192）；跨环境数值不作直接强度比较。结果说明 0019 的 expert-conditioned BC 训练目标与 `source_id=0` 部署目标存在真实缺口，且 neutral actor 未完整保留这套多色资源规划与变长能量弃置策略；它不说明应在部署时恢复 expert source。后续 Raging Bolt value calibration、PPO 与正式评测必须始终使用 `source_id=0`，通过 reward 学习 deck-specific 改进。权威报告为 `evaluation/V8_zero_shot_region_bot.html`；其中 `region_bot` 是已冻结的历史内部标识，不是卡组名称，后续版本统一使用 `raging_bolt`。

## V10 source-98 Persona PoC 与模型复核

V10 保持 epoch-13 checkpoint、Raging Bolt exact deck、30-opponent catalog、10 局/对手和
150/150 先后手日程不变，只把 Actor condition 从 source 0 改为 `James Cox & Henry Chao`
对应的 source 98。官方 runtime 没有 seeded battle ABI，因此 V8/V10 是同协议独立样本，
不是相同发牌的逐局 paired trial。V10 为 79-221、26.33%，先手 25.33%、后手 27.33%，
300/300 完成、0 draw、0 error；Wilson 95% interval 为 21.67%–31.59%。相对 V8 的差值为
-1.67pp，未配对 normal 95% interval 为 -8.78pp–+5.45pp。Persona 没有恢复 Arena 强度，
不能作为候选、RL 起点或 reference policy。

这不代表 Persona 设计没有影响。0015 首次把 source 作为 Actor 输入，目标明确是复刻指定
老师的 `pi(a|s,D,E)`；0016–0019 继承该结构，0019 将它扩大到 509 个 source 并按 conditioned
loss 选择 Epoch 13。同一 Epoch 13 validation split 上，conditioned greedy exact 为 81.2871%，
neutral 为 74.8431%，相差 6.4440pp；loss 为 0.253080647 对 0.368793374。Persona 分支有
368,320 个参数，source 98 的实际 state/option residual norm 为 3.8120/1.0897。离线 conditioned
指标明显包含老师身份信息，但这份信息没有转化为本次 official-engine 强度。

source 0 不是 weight decay：它只令 additive Persona residual 为零，共享 Backbone/Decoder
权重仍然存在。但当前 V9 只训练 Decoder 和 value head，Encoder、deck conditioning 与 Persona
分支全部冻结；PPO 可以从现有 shared feature 中重新组合动作，却没有任何机制保证补回只存于
Persona residual 的表示。因此 V9 继续作为 legacy neutral foundation 的 RL 实验，不冒充干净的
persona-free BC。未来通用 Actor 的目标改为 `pi(a|s,D)`：删除 `source_id` tensor、embedding、
两路 projection 和 gate，在相同拓扑下参数由 17,756,162 降至 17,387,842；source/team 只保留
为 provenance、Episode split、去重、采样和分组评测字段。完整决策见
`decisions/001_source_persona_poc_and_future_contract.md`。

## V9 PPO 合同与项目边界

V9 从 V7 model-only checkpoint warm-start，但必须新建 optimizer 并为每个 update 重新采集 on-policy Episodes。初始合同为 100 updates、512 Episodes/update、GAE lambda `1.0`、2 PPO epochs、batch 1024；actor decoder learning rate `1e-5`、value learning rate `1e-4`、entropy coefficient `0`、reference KL coefficient `0.02`、behavior KL guard `0.01`。仅 autoregressive decoder 与 value head 可训练，board/event/scenario encoder、deck conditioning 与 expert/source residual 均冻结；reference policy 始终是 V1 frozen 0019 actor。

每个 update 都要记录 fresh opponent snapshot、有效/丢弃 Episodes、terminal outcomes、behavior/reference KL、clip fraction、greedy canary flip rate、GPU/rollout 吞吐，并按 update 1、每 10 updates 和最终 update 保存 model-only checkpoint，最多保留 8 个。PPO rollout 胜率是训练诊断，不替代固定 official-engine Arena 的 V9 正式评测。每套牌的 rollout、optimizer、checkpoint 与评测版本必须隔离；共享 V1 基石不因任何单 deck 更新而改变。

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
