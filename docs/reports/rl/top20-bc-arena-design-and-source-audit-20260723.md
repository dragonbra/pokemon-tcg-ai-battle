# Top 20 单专家 BC 替身与 Arena 项目设计及数据初审

日期：2026-07-23（Asia/Shanghai）  
排行榜冻结时间：2026-07-23 04:24:10（2026-07-22T20:24:10Z）  
比赛：`pokemon-tcg-ai-battle`

## 结论摘要

- 已从冻结时点的实时排行榜解析 Top 20 当前 submission，并为每位选手下载、校验一场
  完整官方 replay。20 位选手有一局彼此对战，因此实际保存 19 个唯一 replay，约 77 MB。
- 20 位选手对应 **14 个精确 60 张 deck hash、8 个主牌型**。相同精确构筑仍可能来自不同
  policy；BC 数据必须按 team/submission 分开，不能按 deck hash 合并动作标签。
- 已在本地 `rl/artifact/dataset/top20_arena_20260723/` 建立 `0001`–`0020` 的单专家
  source slot。它们目前只有一局审计样本，均明确标为 `sample_only_not_train_ready`。
- Episode metadata 足以计算“当前排行榜 submission 的已公开对局窗口胜率”。本次共观察
  5,642 局；`junlee789` 和 `Yushin Ito` 都触及 API 最近 1,000 局上限，因此不是生涯总样本。
- Metadata 不含 deck。仅当对手 `submission_id` 也属于本次已审计 Top 20 时，才能把对局
  归到已知 deck；覆盖 709/5,642 = **12.57%**。本文保留这部分局部统计，但不把它称为
  完整 matchup 胜率矩阵。要完成矩阵，需要至少为所有唯一对手 submission 下载一场 replay
  并验证其固定 deck，或下载全部目标 replay 后直接审计双方构筑。

## 1. Repo 中的位置与职责

```text
data/replays/kaggle_top20_bc_20260723/       # Git 忽略；可重建的官方源数据
├── manifest.json                            # 冻结排行榜、样本、deck、窗口统计
├── metadata/<source-id>.json                # 每个 submission 最多最近 1,000 局 metadata
└── episodes/episode-<id>-replay.json        # 一人一局；跨选手按 Episode ID 去重

rl/artifact/dataset/top20_arena_20260723/    # Git 忽略；本地 source staging
├── campaign.json
├── 0001-rank-01-lumenliquidity/
│   ├── source_manifest.json                 # 单一 team/submission policy 边界
│   └── deck.csv                             # 样本中审计出的原版 60 张牌
└── ... 0020-rank-20-guohaoyang/

rl/artifact/dataset/<global-run-id>/         # 后续每个专家的正式 BC dataset
├── dataset.jsonl
├── dataset.jsonl.card_metadata.json
└── dataset.jsonl.summary.json

rl/artifact/checkpoint/<global-run-id>/      # best_validation.pt / latest.pt
rl/_runs/tensorboard/<global-run-id>/        # TensorBoard events
rl/_runs/<global-run-id>/                    # 可提交、可比较的实验事实
├── manifest.json
├── data_manifest.json
├── data_summary.json
├── audit.json
├── training_config.json
├── training_metrics.jsonl
├── training_summary.json
├── commands.md
└── evaluation/<evaluation-run-id>/
```

Source slot 的 `0001`–`0020` 只是本次排行榜快照中的稳定索引，不是全局 RL run ID。
真正开始构建完整 corpus 时，用 `python3 -m rl.core.runs create ...` 顺序分配全局编号；按
当前状态预计从 `0004` 开始，但不能在命令或代码里硬编码，避免与并行实验争用编号。

### 为什么不把 20 个模型直接放进 `evaluation/opponents/`

现有 `evaluation/opponents/` 是固定回归 catalog，职责是提供跨实验不变的验收护栏。Top 20
替身属于会迭代、会冻结多个版本的 Arena league；直接替换固定 catalog 会让前后 run 的
胜率不可比较。

建议保留以下边界：

1. 权重的 canonical copy 永远是 `rl/artifact/checkpoint/<run-id>/best_validation.pt`。
2. 原版 deck、submission ID、模型 schema、源码 revision 和 hash 写入对应 run manifest。
3. 评测或 RL rollout 时，从 checkpoint + deck + 官方 cg 临时 materialize 标准 package；
   只把需要人工验收的候选放进 `work/<name>/`。
4. 一个 RL/Arena run 用自己的 `arena_manifest.json` 固定 opponent run ID、checkpoint hash、
   deck hash、采样权重和 package hash；不修改历史 Arena 快照。
5. 只有明确要把某个替身升级为长期固定回归对手时，才复制成自包含
   `evaluation/opponents/<name>/` package 并更新 catalog。

## 2. 一天项目的执行设计

### 2.1 数据冻结

每位专家固定以下 identity：

```text
competition + captured_at + rank + team_id + submission_id
+ exact 60-card multiset hash + replay Episode IDs
```

同一个 team 的旧 submission 不自动加入当前 corpus；submission 变化可能意味着代码、权重、
deck 或策略都已变化。若要跨 submission 扩容，必须先逐个验证 deck，并把它定义成一个新的
source policy，不能只凭 team name 合并。

完整数据阶段应枚举每个当前 submission 的已公开、已完成 Episode（API 上限 1,000），按
Episode ID 全局去重保存 replay，但在 manifest 中保留 `(episode_id, player_index,
submission_id)` trajectory identity。一个 Top 20 内战 replay 可以为两个独立专家 dataset
各提供一条 trajectory，不需要保存两份 JSON。

### 2.2 Dataset 与 split

每个全局 run 只构建一位专家的数据：

```text
official replay
  -> observation[i] + action[i+1]
  -> ptcg_features_universal
  -> masked legal-candidate targets
  -> chronological episode split 80/10/10
```

硬约束：

- split 以 Episode 为单位，不能把同一局的动作拆到 train/test 两边。
- `source_manifest`、dataset summary、record 和 run manifest 的 team/submission 必须一致。
- 训练前运行 `audit_kaggle_bc_dataset`；任何 deck mismatch、source mismatch、非法 target、
  split 泄漏或 shape mismatch 都直接停止该 run。
- 当前 universal tensor 合同为：`state [64,40]`、`deck [60,18]`、`entity [12,28]`、
  `history [32,8]`、`action [64,22]` 和单个 `expert_id`。详情以
  `rl/model/DESIGN.html` 和 checkpoint metadata 为准。
- 少于 100 局的 submission 仍可训练探索模型，但标为 `provisional_low_data`；本次
  `Majkel1337`（43 局）和 `Eduardo Rocha de Andrade`（41 局）属于这一类，其 held-out
  模仿率置信区间会很宽，不应和 1,000 局来源同等解读。

### 2.3 训练调度

20 个模型共享正式源码和基线超参数，但不共享动作标签或 optimizer state。建议单 GPU
串行训练，CPU 侧下载、构建和审计可以有限并行：

1. 下载阶段最多 2 个并发请求，避免 Kaggle 429；预计完整 5,642 局 replay 是数十 GB。
2. 每个 dataset audit 通过后即进入训练队列，不必等待全部 20 个下载完成。
3. 基线使用 universal 模型、seed 7、AdamW、batch 256、learning rate `3e-4`、最多
   20 epochs；增加 early stopping（例如 validation 5 个 epoch 无提升）以适应一天窗口。
4. 每个 epoch 写 `training_metrics.jsonl` 和 TensorBoard；best checkpoint 只按 validation
   exact-action rate 选择，test 只在训练结束后读取一次。
5. 当前 Alakazam universal checkpoint 可作为接口 smoke/control；不建议直接作为所有牌型
   的必选初始化，因为未出现卡牌的 learned ID embedding 没有被充分训练。是否 warm-start
   必须做独立 ablation，而不是默认混入。

后续若要训练单个多专家通用 policy，只有在显式使用 source conditioning、对冲突标签做
审计并按 source 分开评测后才允许；它不属于本次 20 个独立替身的短项目范围。

### 2.4 标准 Agent 复刻

现有 `rl.train.build_full_action_submission` 已能把 checkpoint、通用推理源码、`main.py` 和
官方 `cg/` 组装为标准 package，但当前参数把 `deck.csv` 与 `--source-package` 耦合。批量
Arena 前应做一个小改造：

```text
--checkpoint <best_validation.pt>
--deck <frozen expert deck.csv>
--cg-source <one audited official-runtime package>
--output <temporary or work package>
--experiment-id <global-run-id>
```

builder 必须交叉校验 checkpoint metadata 中的 deck、source、feature schema 与外部
`deck.csv`，并在 package manifest 保存 checkpoint/deck/source-code/cg tree hash。生成物应：

- 只有 `select=None` 时返回固定 deck；其余 observation 全部由模型确定性推理。
- 只返回 simulator 提供的合法 option index，不允许 teacher/rule fallback。
- 包含 `main.py`、恰好 60 行的 `deck.csv`、物理复制的 `cg/`、通用 strategy 源码和
  `model.bin`，能够通过 `python3 -m evaluation validate <package>`。

### 2.5 评测与 Arena 晋级门槛

每个替身至少经历三层验收：

1. **离线模仿**：validation/test exact action、single/multi exact、selection-count accuracy，
   并按 main/effect selection context 分组。
2. **运行正确性**：package validate；固定评测中 legal action rate 100%、candidate error 0、
   completion 100%。任何引擎/对手错误必须单独归因，不能算作模型胜利。
3. **对战能力**：与当前固定 18-opponent catalog 每个至少 10 局，先后手交替；报告总胜率、
   每对手胜率、错误率及 semantic metrics。结果写回同一
   `rl/_runs/<run-id>/evaluation/`。

离线 action accuracy 衡量“像不像该专家”，本地胜率衡量“这个替身是否足够强”；两者都需要，
不能互相替代。Arena tier 建议：

- `accepted`：全部正确性硬门槛通过，且完成固定 18×10 评测。
- `provisional_low_data`：正确性通过，但来源不足 100 Episode 或 test 不稳定。
- `rejected`：非法动作、candidate error、无法完成 package，或固定池表现明显退化。

## 3. Top 20 一局样本与牌组识别

下表中的胜率是 frozen **当前 submission 的 metadata 窗口胜率**，定义为
`wins / (wins + losses + draws)`；不是 team 生涯胜率。`*` 表示触及 API 最近 1,000 局上限。

| 排名 | 选手 | Elo | 样本 Episode | 识别牌组 | Exact hash | Metadata W-L-D | 胜率 |
|---:|---|---:|---:|---|---|---:|---:|
| 1 | LumenLiquidity | 1167.7 | 87527489 | Dragapult ex / Dusknoir | `cb57ce7f` | 89-38-0 | 70.08% |
| 2 | junlee789 | 1162.3 | 87527472 | Cynthia's Garchomp ex / Roserade | `c04e65de` | 584-416-0* | 58.40% |
| 3 | Majkel1337 | 1158.5 | 87527389 | Alakazam / Dudunsparce | `3f451509` | 32-11-0 | 74.42% |
| 4 | Rmy | 1154.2 | 87527473 | Marnie's Grimmsnarl ex / Munkidori / Froslass | `c20a8a46` | 107-51-0 | 67.72% |
| 5 | tw_shin | 1146.9 | 87527474 | Mega Lopunny ex / Mega Froslass ex | `dd63244c` | 430-282-0 | 60.39% |
| 6 | Yudai Ueno | 1138.8 | 87526929 | Cynthia's Garchomp ex / Roserade | `c04e65de` | 110-70-0 | 61.11% |
| 7 | THIRD PTCG Club | 1130.2 | 87527478 | Team Rocket's Mewtwo ex / Spidops | `71861f2f` | 542-340-2 | 61.31% |
| 8 | Luca | 1124.9 | 87525393 | Archaludon ex / Cinderace | `da1d56e3` | 51-38-0 | 57.30% |
| 9 | kashiwashira | 1115.8 | 87527483 | Team Rocket's Mewtwo ex / Spidops / Mimikyu | `fba1f87c` | 72-48-1 | 59.50% |
| 10 | iwashi | 1115.4 | 87525114 | Alakazam / Dudunsparce（3 Hammer / 3 Mine） | `f1776ee3` | 73-39-0 | 65.18% |
| 11 | Where is my orbit | 1114.5 | 87527481 | Mega Kangaskhan ex / Crustle | `03822cc7` | 72-43-0 | 62.61% |
| 12 | Yushin Ito | 1110.2 | 87527485 | Alakazam / Dudunsparce（4 Hammer / 2 Mine） | `3f451509` | 583-417-0* | 58.30% |
| 13 | youtube.com/@BigBugginnings | 1108.6 | 87527489 | Dragapult ex / Munkidori（无 Dusknoir） | `483f0d89` | 90-63-0 | 58.82% |
| 14 | Eduardo Rocha de Andrade | 1108.0 | 87527488 | Marnie's Grimmsnarl ex / Munkidori / Froslass 变体 | `d40724c0` | 29-12-0 | 70.73% |
| 15 | Lunariz | 1106.0 | 87527110 | Cynthia's Garchomp ex / Roserade | `c04e65de` | 49-23-0 | 68.06% |
| 16 | nasuo445 | 1099.6 | 87527497 | Cynthia's Garchomp ex / Roserade 变体 | `3f8ee4b5` | 232-166-0 | 58.29% |
| 17 | Dries @ Tufa Labs | 1098.5 | 87527492 | Team Rocket's Mewtwo ex / Spidops | `71861f2f` | 76-42-0 | 64.41% |
| 18 | HowardLeeTW | 1097.0 | 87527500 | Marnie's Grimmsnarl ex / Munkidori / Froslass | `c20a8a46` | 58-28-0 | 67.44% |
| 19 | `{{ team_name }}` | 1095.4 | 87527499 | Team Rocket's Mewtwo ex / Spidops 变体 | `7851bf55` | 62-28-0 | 68.89% |
| 20 | GUOHAOYANG | 1094.6 | 87527502 | Marnie's Grimmsnarl ex / Munkidori / Froslass | `c20a8a46` | 89-54-0 | 62.24% |

牌型分布：Cynthia Garchomp 4、Marnie Grimmsnarl 4、Team Rocket Mewtwo/Spidops 4、
Alakazam 3、Dragapult 2、Mega Lopunny/Froslass 1、Archaludon/Cinderace 1、
Mega Kangaskhan/Crustle 1。

需要特别注意：`LumenLiquidity` 和 `youtube.com/@BigBugginnings` 的样本都是 Episode
87527489 的两侧，replay 只保存一次；BC source 仍保持两份独立 player trajectory。

## 4. Metadata matchup 尝试结果

仅依据 metadata 无法识别对手 deck。本次把对手 submission 精确匹配到当前 Top 20 已审计
submission 后，得到 709 局已知牌组对局（总 metadata 的 12.57%）。以下是把这 709 个
player-perspective 样本合并后的**局部**结果：

| 已知对手主牌型 | W-L-D | 局部胜率 |
|---|---:|---:|
| Alakazam | 69-76-0 | 47.59% |
| Cynthia's Garchomp ex | 80-95-0 | 45.71% |
| Dragapult ex | 26-41-0 | 38.81% |
| Marnie's Grimmsnarl ex | 39-37-0 | 51.32% |
| Mega Kangaskhan ex / Crustle | 11-7-0 | 61.11% |
| Mega Lopunny ex / Mega Froslass ex | 31-21-0 | 59.62% |
| Team Rocket's Mewtwo ex / Spidops | 92-84-0 | 52.27% |

这张表不能解释为整个 Top 20 对 meta 的真实 matchup：

- 覆盖率只有 12.57%，并且偏向 Top 20 互相遇到的对局。
- 同一主牌型包含多个 exact hash；这里按人工主牌型聚合。
- 当前 Archaludon submission 没有出现在其他已保存 metadata 的 opponent submission 中，
  所以没有这一列。
- 每位选手的 exact-hash 局部拆分已保存在本地 campaign manifest；由于多数格子样本极小，
  本报告不展示一个容易误导的 20×14 百分比矩阵。

完整 matchup 分析的最低成本方案不是下载全部 5,642 局：先收集 metadata 中所有唯一
opponent `submission_id`，每个下载一场可审计 replay 来建立 `submission_id -> deck hash`
映射；submission 的 package/deck 固定后，即可回填 metadata 矩阵。只有找不到完整初始牌组帧、
submission identity 不稳定或需要逐局策略细节时，才下载该 submission 的全部 JSON。

## 5. 已完成与未完成边界

已完成：

- 可复现 Top-N source 准备器 `rl.train.prepare_top_ladder_bc` 及单元测试。
- 实时 Top 20 快照、20 个 metadata 窗口、19 个唯一完整 replay。
- 20 个单专家 dataset source slot、原版 60 张 deck 和 source manifest。
- 牌组识别、窗口总胜率和 12.57% 覆盖的局部 matchup 尝试。
- `rl/model/DESIGN.html` 已同步 universal schema 与 Top 20 → Arena 阶段边界。

尚未执行：

- 下载 20 个 submission 的完整公开 replay 窗口。
- 构建 20 份 train-ready JSONL、训练 20 个 checkpoint。
- 泛化 submission builder 的 `--deck` / `--cg-source` 参数。
- materialize 20 个标准 package 并完成每个 18×10 本地评测。
- 为 metadata 中所有唯一 opponent submission 建立 deck 映射并生成完整 matchup 矩阵。

本轮只进行了 Kaggle 只读 API 查询与 replay 下载，没有创建 Kaggle submission，也没有修改
官方 engine runtime。
