# 0022 League Training

**项目 ID：** `0022_league_training`

**当前阶段：** V11 已按用户要求在完整 update 81 后安全停止；该批由 source policy update 80
采集，512/512 finished、0 rollout error，48/48 Live deck 均有 trajectory。V11 证明多 decoder
更新链可运行，但不再把“每批强制 48 套全部更新”作为终极训练合同。下一阶段一次只指定一个
focal deck 作为主要优化对象；新版本 catalog 已加入 `mega_lopunny_ex_001` Limitless 冠军构筑，
其他 48 个 Live decoder 仅利用真实对局作机会式更新；所有 checkpoint 的强度验收统一改走固定的
50-deck Frozen Arena v3。

**目标：** 在 BC 基础模型完成后，验证常驻、批量化 opponent pool 是否能显著提高 RL rollout 与迭代吞吐，同时保持一个可审计、不会随 Live pool 退化的 Frozen League 质量锚点。

## 1. 核心假设

当前 RL 的首要怀疑对象不是 learner actor 的单次推理，而是 opponent 的重复加载、CPU 推理、进程间通信和批量不足。0022 不先假设 CUDA battle engine 已经完成；它首先验证在**未修改 official engine**上，把 opponent policy 常驻并集中批推理后，能否显著提高真实 RL pipeline 的吞吐。

项目的理想实验条件是：

- 0019 Epoch 13 BC Foundation 已冻结，并以固定 `source_id=0` 的 persona-neutral deployment
  合同运行；
- 共享 Encoder 长期冻结；
- 每个 exact deck 拥有独立的 Decoder/Value 分支；
- Frozen pool 保存固定的完整策略，作为质量回归和稳定 opponent；
- Live pool 保存正在进行 Decoder-only RL 的策略，作为可进化 opponent；
- 所有战斗仍由 official engine runtime 执行，`engine_cuda/` 在 parity gate 通过前不能产生正式 rollout 或强度结论。

0022 研究的是“Foundation 表示空间内的可塑性”和“常驻 opponent 服务的吞吐收益”，不是全局最优、完整 self-play equilibrium 或新的官方规则引擎。实测结论记录在 [`decisions/002_throughput_measurement.md`](decisions/002_throughput_measurement.md)：PPO-ready stacked shared GPU League 达到 1.79x end-to-end 和 1.73x opponent decisions；前者通过，后者仍待优化。

## 2. 证据边界

### 官方规则和运行时事实

攻击终止回合、进化时机、Supporter/手填 Energy/Retreat 预算、Prize 和胜负条件均由官方规则与 official engine runtime 定义。策略只能从 engine 返回的合法 option 中选择；合法性不是 PPO reward 学出来的属性。

### 训练事实

终局 reward 默认是：胜 `+1`、负 `-1`、平 `0`，`gamma=1`。同一 Episode 的每个决策使用该局自己的终局 target；actor、value、behavior policy version 和 opponent snapshot 必须逐局记录。rollout 胜率、loss、KL 和吞吐都是训练诊断，不能替代冻结 official-engine 评测。

### 不可声称的结论

- Frozen pool 胜率提升只能证明相对于固定 anchor 的策略改善，不能证明全局最优；
- Live pool 胜率不能单独证明强度，可能受到循环、过拟合或 opponent 分布漂移影响；
- 共享 Encoder 的一次 batched forward 不能自动证明 complete CUDA engine 已经实现；
- 离线 exact-action、合法动作率和 PPO reward 不能替代同一 opponent catalog、先后手和 metric profile 下的 official-engine Arena。

## 3. 模型和策略池

### 3.1 Shared Foundation

0022 严格使用归档资产 `0019-0730-epoch13`：Epoch 13、global step 337194、17,756,162
参数、权重 SHA-256
`da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`。0019 的
`SourceConditionedR15Policy` 结构仍包含历史 persona residual，但 0022 永远提供全零
`source_id: int64[B]`，不允许 deck、team 或玩家身份进入该字段。source/team identity 只保留在
provenance；deck 差异来自既有 exact registered-deck feature 和真实场面输入。

0022 在 `train/0022_league_training/foundation/` 物理固化模型、feature compiler、causal
ledger 和 action contract，在 `assets/card_ontology.json` 固化 ontology；运行时不 import 0019、
0020 或其他编号训练项目。归档 `model.pt` 仅作为按 SHA 校验的不可变数据来源。共享 Encoder、
persona residual、card ontology、feature compiler 和 action contract 在整个 run 中冻结。

Decoder-only 研究假设是：既有表示已经包含足够的卡牌、场面和资源语义，Decoder 可以重新排序合法 option、改进 ordered full-action 序列和 STOP 时机。若多个 deck 在 Frozen anchor 上长期 plateau，才允许创建新的实验版本，开放显式 deck residual、option adapter 或最后 scenario layer；不能在同一版本内悄悄解冻 Encoder。

### 3.2 Frozen pool

Frozen pool 是 49 个**完整策略身份**，不是只有 Encoder 的权重。每个身份固定：

- exact 60-card `deck.csv` 与 deck SHA-256；
- BC checkpoint、模型 schema、ontology 和 feature compiler hash；
- 固定 `source_id=0` 的 neutral-source inference contract；
- decoder/value 权重；
- package 和 catalog 版本；
- official engine runtime / `cg` hash。

第一版 Frozen pool 使用 BC zero-shot 策略和后续明确晋级的 historical best snapshot。逻辑上每项
仍绑定完整 Foundation、deck、decoder 和 feature contract；物理存储上，zero-shot baseline 以
`decoder_ref: foundation` sentinel 引用 Foundation decoder，不重复保存权重。只有晋级后的
historical decoder snapshot 才引用不可变 `.pt` 和 SHA-256。它永远不因 Live update 改写，承担
三项职责：

1. 提供不随训练漂移的高质量 opponent；
2. 给主视角提供固定回归指标；
3. 检测 Live pool 的循环、坍缩和共同退化。

权威资产固定在 `evaluation/arena/frozen/`：`_policy/` 物理保存唯一一份 0019 Epoch 13
Foundation package；49 个 deck identity 目录只保存 exact `deck.csv` 和 manifest。catalog 为
`evaluation/configs/frozen.json`，pool ID 为 `0019_foundation_49_exact_decks_v2`。默认
`python3 -m evaluation ...` 使用该池；历史异构 Kaggle-derived `opponents/` 仍可通过
`--pool opponents` 显式使用，定位为 secondary external-generalization evidence。

### 3.3 Live pool

Live pool 的下一正式版本由 49 个独立的 deck-specific Decoder/Value 资产组成，共享 Frozen Encoder。每个正式
run 只声明一个 focal deck；focal 获得主要采样预算和更新频率，另外 48 个 Live deck 只有在
真实对局中积累到足够 actor trajectory 时才作机会式独立更新。每套 deck 仍使用自己的
trajectory、reference、optimizer 和 checkpoint；对手动作只进入其自身 deck 的 loss，绝不混入
focal loss。side-deck 的低等效更新量不得被描述成与 focal 等速进化。

Live pool 是 curriculum 和探索资产，不是自动晋级的正式 opponent。只有通过固定 Frozen pool、历史 snapshot 和完整 package audit 后，才可以创建新的 Frozen snapshot。

## 4. League rollout 合同

### 4.1 角色

每场 Episode 记录：

```text
actor_deck_id / actor_policy_version
opponent_deck_id / opponent_policy_version
actor_seat / opponent_seat
frozen_or_live_role
opponent_pool_snapshot
official_engine_hash
reward / terminal_reason
```

主视角仍为 `dragapult_ex_001`。V3+ 的 focal 与 Live opponent 都记录 action/log-prob/value，
每条决策携带 exact `policy_deck_id`、`policy_update` 和 actor-relative `reward_sign`；Frozen
opponent 只提供不可训练的质量锚点。每个 deck 只从自己的轨迹过滤出 PPO batch，分别更新自己的
decoder/value 和 optimizer；双方不会共享 loss、optimizer 或 rollout buffer。

### 4.2 初始批次

第一版采用可配置的 focal schedule。默认一个 focal deck 的 iteration 包含：

```text
49 Frozen views + 49 Live views = 98 matchup identities
512 official-engine games / PPO update
严格 256 先手 + 256 后手，确定性轮转覆盖整个 catalog
```

先后手按固定平衡日程分配。`10 games / matchup` 只用于 rollout 和快速诊断，单个 matchup 的胜率不得作为强结论；正式趋势使用跨 iteration 滚动窗口和 Wilson 区间。固定 League probes 为 `alakazam_dudunsparce_001` 与 `marnies_grimmsnarl_ex_froslass_001`，分别报告 focal-vs-Frozen、focal-vs-Live 和 probe-Live-vs-focal。

不能把 focal 的 512 局误称为另外 48 个 deck 获得了 actor 数据；只有被标记为该 deck 的 Live
决策才可进入它的 loss。每 5 个 PPO update 可运行 49 Frozen × 双座位和 49 Live × 双座位，
各 98 局 greedy diagnostic；该 `eval/*` 与 sampled rollout 分开记录，但 98 局只用于快速定位
candidate。正式验收固定为 49 Frozen × 每套 10 局 = 490 局，先后手各 5 局。每次 gate 还记录
`alakazam_dudunsparce_001`、`marnies_grimmsnarl_ex_froslass_001` 的 focal-vs-Frozen、
focal-vs-Live、Live-vs-focal-Live 三组双先后手 probe。

### 4.3 PPO 更新边界

每个 rollout batch 内：

1. 所有 policy version 和 opponent snapshot 在采集期间冻结；
2. 每个 policy 只读取自己作为 actor 的 action/log-prob/advantage；
3. 终局 reward 按 actor 视角取正负；
4. update 完成后才发布新的 Live version；
5. 新 Live version 不回写旧 rollout，也不向旧 `training_metrics.jsonl` 追加不同训练语义。

reference policy、behavior policy、checkpoint/update 和 `rollout/source_policy_update` 分开记录。Live pool 同步刷新可以采用每 `K` 个 update 一次，但必须把 `K` 和 pool snapshot 作为实验变量写入 manifest。

## 5. 常驻 opponent 推理架构

### 5.1 目标拓扑

```text
official-engine workers (CPU state transition only)
    ├── candidate requests -> resident batched GPU service -> candidate action
    └── Frozen requests  -> resident batched GPU service -> Foundation action
                              exact deck routed per session
```

Frozen 的 49 个 exact-deck identity 共用一个 Foundation 模型、一个共享 Encoder 和一个 GPU
服务；每次 request 携带 exact 60-card deck，并按 session 建立因果 feature state。被评测 candidate
使用另一常驻 GPU 服务，避免与 Foundation 权重身份混淆，同时也享受跨 worker batching。
因此一次 Frozen evaluation 只加载两份完整模型，而不是 50 份。禁止每个 opponent worker重复
加载 checkpoint、创建独立 CUDA context，或在每个决策上进行无批量 H2D/D2H 往返。官方 engine
始终留在隔离 CPU worker，GPU 加速不改变状态转移和合法 action 合同。

### 5.2 速度验证 gate

BC 完成后，0022 必须先进行三臂 benchmark，保持 exact deck、checkpoint、engine、seed/先后手日程和 completed-game 合同一致：

1. 当前 tuned CPU opponent baseline；
2. 常驻 CPU batched opponent service；
3. 常驻 GPU batched opponent service。

V1 把主视角与 opponent request 在 0.5 ms 窗口内合批，但 8 workers 只形成平均 6.92 的
batch。V2 改为 128 个 Torch-light engine workers 与 5 ms coalescing，平均 batch 90.28，
同时共享一次 Encoder forward；
主视角路由到可训练 Decoder/Value 并随机采样，opponent 路由到只读 Foundation Decoder 并
greedy decode。未分叉的 47 个 Live head 使用写时复制语义，不在 GPU 上重复相同 tensor。

必须分别记录：

- completed episodes / errors / unfinished；
- official engine selections/s；
- opponent decisions/s；
- learner decisions/s；
- end-to-end episodes/s；
- opponent package-load time；
- feature encode、IPC、H2D/D2H、GPU inference latency；
- CPU/GPU memory、worker 数、worker 内部线程数和 GPU utilization。

0022 的第一阶段只在满足以下条件后才可以声称“opponent throughput 有实质改善”：

- 常驻服务相对当前 tuned baseline 的 opponent decisions/s 至少 `2.0x`；
- end-to-end completed episodes/s 至少 `1.5x`；
- completed/error/unfinished 合同无回归；
- 同一 Frozen pool 的固定 Arena 结果差异不超过预先记录的统计区间，不能用速度换质量；
- profiling 证明收益来自常驻与批量推理，而不是减少局数、改变 engine 合同或降低策略质量。

如果只提高了 policy inference 而 end-to-end rollout 没有提升，不能把结果称为 RL iteration speedup。

当前证据是 partial pass：512 局 D 为 3.568 episodes/s、582.17 engine selections/s、264.44 opponent decisions/s，300 局 A sample 为 1.990、339.14、152.62；完成率均 100%，trajectory scalar 非有限数均为 0。end-to-end 1.79x 已通过，但 opponent-only 为 1.73x，不能绕过 2.0x gate 开始 Live self-evolution。

正式 V1 又提供了 PPO-ready 实测：512/512、0 error、0.6373 games/s，93,950 requests /
13,584 batches。根因不是 GPU/SSD，而是 8-worker 供给不足。相同 256-game workload 在
Torch-light worker 修复后分别达到 64 workers 1.4081、128 workers 2.4506、256 workers
2.7039 games/s；128 已达到最佳值的 90.63%，因此按预先约定选择最小的 10% 带宽内配置。
完整证据见 [`decisions/003_v1_throughput_and_v2_scaling.md`](decisions/003_v1_throughput_and_v2_scaling.md)。

## 6. 质量和准入合同

### 6.1 Frozen pool quality

每个 deck 长期维护三类指针：

- `foundation`：不可变 0019 update-0 起点；
- `latest/challenger`：最新训练产物，只表示训练进度；
- `champion`：最近一次通过固定 Frozen Arena 验收的正式最强版本。

`latest` 不得自动覆盖 `champion`。每次晋级前，对固定 Frozen catalog 执行 490 局、平衡先后手的
official-engine evaluation，并同时保留：

- zero-shot Frozen baseline；
- 当前 Frozen best；
- Live candidate；
- 必要的 historical snapshots。

报告必须按 deck、opponent、先后手和 metric profile 分组。任何单一 matchup 的短期上涨都不能自动晋级。Live candidate 至少要在 focal deck 的 Frozen pool aggregate 指标上达到预先记录的非劣性，并在历史 snapshot 上没有明显崩溃；正式阈值写入该版本 decision record，不能运行后临时改口径。

### 6.2 50-deck catalog 资产

正式 catalog 不是现在预先写死的名单。每个 deck 候选必须满足：

- exact 60-card deck 可审计，来自线上环境或官方可复现 submission/episode；
- 在当前线上环境中真实出现过，且不是只凭卡名或 archetype 猜测；
- 使用频率足够高，能在 BC 数据与环境分析中找到可追溯 evidence；
- 关键卡牌与构筑语义已经被 BC 数据覆盖，而不是完全未学习的新机制；
- 能通过 package、deck hash、card ontology 和 official engine runtime validation；
- 代表不同的资源、攻击、控制、进化、Prize race 或接力模式，避免近似构筑只重复一种梯度；
- 进入 catalog 前记录来源、选择理由、风险、数据覆盖和 exact deck identity。

新增卡牌或新机制的探索属于后续项目：可以通过定向 card embedding/ontology 适配，或让新 deck 加入已进化 Arena 后进行独立适应，但不能把新卡实验无记录地混入 0022 第一版 catalog。

### 6.3 League 对局质量诊断

Frozen League 使用 `league_deck_quality` revision 1。每局 official-engine trace 在清理前提取
通用原子指标：启动与首次攻击、攻击机会与连续性、Prize 转化、多 Prize 回合、KO 后接力、场面
规模与进化/能量、Supporter/手填能量、牌库压力、对手攻击受阻、Ability、伤害事件和主动离场。
50 套 exact deck 由 `evaluation/metrics/league_profiles.py` 完整映射到 22 个策略类别；每类声明
关键卡、重点字段、解释和 reward warning。胡地继续关注 Powerful Hand、Dudunsparce 过桥和
Post-KO relay；多龙关注 Stage 2 成形、铺伤兑现与多 Prize 回合；Dusknoir 组合不得把自我 KO
直接当负奖励；Raging Bolt 的弃能只按 KO 和恢复解释；Crustle/control 的长局只按压制与终局
解释。所有过程指标都是 checkpoint 选择的辅助证据，不得替代 490 局 Frozen 胜率门禁。

## 7. 失败模式和护栏

### 对手共同退化

Live pool 可能共同学会利用当前 pool 的漏洞。Frozen anchors、historical snapshots 和外部/固定 opponent 必须保留，Live-only 胜率不得作为晋级依据。

### 策略循环

49 个 deck 可能出现非传递循环。必须报告 cross-play matrix、按先后手胜率、滚动 Elo/结果和 Frozen anchor 结果；不能只看平均 Live 胜率。

### 终局 reward 稀疏

纯胜负 reward 保持为主合同，但 value calibration、优势归一化、episode-length diagnostics 和充分探索必须记录。不得用未经批准的 shaping reward 改变正式强度目标。

### 数据稀疏和角色混淆

`10 games / matchup` 不是高置信度胜率。每个 policy 的 actor decisions、作为 opponent 的 decisions、seat、pool role 和有效 episode 数必须分开统计。

### 训练吞吐假象

减少游戏数、缩短 episode、改变 opponent 强度或让 worker 忽略错误都不能算加速。所有吞吐比较必须保持完成率和官方 engine 合同一致。

## 8. 项目版本和产物

0022 正式训练版本使用：

```text
experiments/0022_league_training/DESIGN.md
experiments/0022_league_training/DESIGN.html
experiments/0022_league_training/decisions/NNN_<tag>.md
experiments/0022_league_training/evaluation/V<n>_<tag>.html
rl_runs/0022_league_training/versions/V<n>_<tag>/artifact/
rl_runs/0022_league_training/versions/V<n>_<tag>/checkpoint/
rl_runs/0022_league_training/versions/V<n>_<tag>/tensorboard/
rl_runs/0022_league_training/versions/V<n>_<tag>/wandb/
```

每个版本必须记录 BC checkpoint hash、Encoder/Decoder schema、Frozen pool snapshot、Live pool versions、exact deck hashes、official engine hash、catalog manifest、sampling weights、先后手 schedule、W&B run identity 和吞吐 benchmark。版本之间不能覆盖 artifact、checkpoint、metrics 或 evaluation HTML。

### 8.1 已实现的插件与 checkpoint 合同

用户维护的 tracked 插件只放身份和 exact deck：

```text
train/0022_league_training/deck/<deck_id>/
  manifest.json   # role、focal、decoder_ref、provenance
  deck.csv        # 恰好 60 行正整数 card ID
```

`deck_id` 必须是 ASCII snake_case；catalog 拒绝未知文件、未知 manifest 字段、重复 deck ID、
重复 exact-deck hash、Frozen focal 和不完整 provenance。空 staging 可执行 validation，但
`initialize` 必须至少有一个 `role=live, focal=true` 的插件。

Live 的可变参数写入新版本，而不是 `train/`：

```text
rl_runs/0022_league_training/versions/V<n>_<tag>/
  artifact/training_config.json
  artifact/league_catalog.json
  artifact/status.json
  checkpoint/decks/<deck_id>.pt
  checkpoint/decks/<deck_id>.pt.sha256
  checkpoint/live/<deck_id>/update-<n>.pt  # latest 2 + every gate snapshot
```

每个 Live checkpoint 只包含以下 tensor，合计 1,130,883 参数：

| 组件 | shape / 参数 |
|---|---:|
| `pointer_key.weight`, `pointer_query.weight` | 各 `[320,320]` |
| `option_bias.weight/bias` | `[1,320]`, `[1]` |
| `decoder_init.weight/bias` | `[320,320]`, `[320]` |
| GRU `decoder.weight_ih/weight_hh` | 各 `[960,320]` |
| GRU `decoder.bias_ih/bias_hh` | 各 `[960]` |
| `stop` 两个 Linear | `[320,320]`, `[320]`, `[1,320]`, `[1]` |
| scalar value head | LayerNorm 320 + `320→320→1` + Tanh，103,681 参数 |

actor decoder 为 1,027,202 参数。checkpoint 顶层只允许 schema、Foundation SHA、deck
ID/hash、policy role/version、update 和 tensor state；optimizer、scheduler、GradScaler、RNG、
rollout、replay 和未知 tensor key 一律拒绝。写入采用临时文件原子替换并生成 SHA sidecar。

操作入口：

```bash
python3 -m train.0022_league_training verify-foundation
python3 -m train.0022_league_training validate-decks
python3 -m train.0022_league_training initialize --version V1_initial_league
python3 -m train.0022_league_training audit-version --version V1_initial_league
python3 -m train.0022_league_training smoke-rollout --device cuda:0 --workers 4
python3 -m train.0022_league_training canary-ppo --device cuda:0 --workers 4
python3 -m train.0022_league_training benchmark-workers --workers 128 --games 256 --coalesce-ms 5 --output .tmp/evaluation/0022_worker_scaling/workers-128.json
python3 -m train.0022_league_training train-league --version V11_multidecoder_league_20h --workers 128 --coalesce-ms 5 --games-per-update 512 --duration-hours 20
```

`initialize` 只建立不可变 League 状态，不采集对局、不做 backward、不启用 W&B，因此不是
策略强度证据。正式 PPO 开始后必须启用 W&B online，并以 official-engine on-policy Episode、
固定 Frozen evaluation 和逐 policy trajectory 为准。

## 9. 阶段路线

### Gate A：BC 和 catalog 准备

完成 0019 neutral-source Foundation 绑定；建立候选 deck evidence；为每个候选准备 exact
deck/package/hash；不训练 Live。

### Gate B：opponent throughput feasibility

在相同 official engine workload 上比较 tuned CPU、resident CPU、resident GPU 和 unified stacked GPU opponent service。当前 end-to-end gate 已通过，但 opponent 2.0x gate 尚未通过；未完成该 gate 前不开始 Arena self-evolution。

### Gate C：Frozen-only Decoder RL

只训练 `dragapult_ex_001`，下一版本对 49 Frozen + 49 Live views 运行 decoder/value RL。V1 在首个
完整 update 后因低吞吐停止并保留 checkpoint；V2 使用 128-worker 合同继续。确认
Frozen anchor 胜率、吞吐和 checkpoint 版本合同正确。

### Gate D：Frozen + Live league

V11 已在 update 81 后由用户停止，验证了 48 Live 分支各自 trajectory 与独立 backward 的工程
可行性。后续改为 focal-primary：一次集中训练一个 deck，side deck 只作机会式更新；不再因为
某个 side deck 当批缺少 trajectory 而拒绝 focal update。Frozen/Live diagnostic 和两个固定
League identity probe 继续保留，但当前正式强度结论来自 490 局 Frozen Arena v2。

### Gate E：晋级和长期进化

只有 candidate 通过 quality gate、package validation、official-engine evaluation 和 pool snapshot 审计后，才可成为新的 Frozen historical snapshot。新卡牌或开放 Encoder/adapter 是独立后续版本，不修改 0022 第一版解释。

## 10. 成功标准

0022 第一阶段成功，不要求全局最优。它需要同时证明：

1. 常驻 opponent service 在相同完成率和策略质量下显著提升真实 end-to-end RL throughput；
2. Decoder-only RL 能在冻结共享表示中相对 zero-shot/Frozen baseline 获得稳定改进；
3. Frozen + Live pool 不会把 quality regression 隐藏在 Live-only 结果中；
4. 50-deck catalog、policy version、reward、opponent snapshot 和 official-engine evaluation 全部可追溯；
5. 如果 Decoder-only plateau，能从对照实验明确判断瓶颈是表示容量、探索、value calibration 还是 opponent distribution。

0022 的终极形态是“常驻 Arena Training”，但第一步必须先证明它是一个可测的 throughput 和局部策略改进系统，而不是把快速自我对打误认为绝对能力提升。
