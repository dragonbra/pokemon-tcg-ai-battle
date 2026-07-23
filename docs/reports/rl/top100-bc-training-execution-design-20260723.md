# Top 100 单专家 BC 训练执行设计

日期：2026-07-23（Asia/Shanghai）  
调研快照：2026-07-23 05:15:27 CST  
适用范围：Top 100 调研、20 个独立单专家 BC Agent、repository evaluation 与候选归档

## 1. 当前结论

- 已冻结实时 Top 100 的静态 Rank、team/submission identity、公开 Episode metadata 和一场
  代表 replay。100 人共 17,156 条 player-episode metadata、96 个唯一 replay、44 个 exact
  deck hash、17 类关键 Pokémon 牌型。
- 当前 submission 达到 500 局的只有 5 人。`LumenLiquidity` 当前为静态 Rank 2，144 局，
  99-45、胜率 68.75%，使用 Dragapult ex / Dusknoir（“自爆多龙”），按用户要求强制入选。
- 用户已授权把普通候选硬底线放宽到 200 局，但样本量仍必须进入选择分、训练验收和风险标记，
  不能降门槛后只按原始胜率排序。
- 本文给出 20 人推荐排程。它是训练 source roster，不是 Arena 晋级名单；只有模型离线模仿、
  package 正确性和本地对战评测均通过后，才交给用户决定是否加入 opponent pool。

调研入口：

- `docs/reports/rl/top100_bc_candidate_research-20260723.html`
- `docs/reports/rl/top100_static_rank_index-20260723.csv`
- `rl/artifact/dataset/top100_research_20260723/campaign.json`
- `data/replays/kaggle_top100_research_20260723/manifest.json`

## 2. 20 人选择规则

### 2.1 两个维度

原始胜率会高估小样本来源，单独按场次又会偏向低胜率老 submission。因此使用：

```text
conservative_win = Wilson 95% lower confidence bound
                   （胜=1，和=0.5，负=0）
sample_sufficiency = min(public_episodes, 500) / 500
selection_score = 0.65 × conservative_win + 0.35 × sample_sufficiency
```

解释：

- 65% 权重给保守胜率，保留“专家本身要强”的主要目标；
- 35% 权重给标签量，反映 BC 学习的样本风险；
- 样本奖励在 500 局封顶，因为 500 是本项目原定的“足够训练”目标，不能让 1,000 局低胜率
  来源仅凭场次无限压过更强来源；
- 普通候选必须至少 200 局；`LumenLiquidity` 是唯一低于 200 局的人工强制例外；
- Lumen 固定为排程 1，其余候选按 `selection_score` 降序，平分时按静态 Rank。

### 2.2 推荐训练 roster

| 排程序 | 静态 Rank | 选手 | 场次 | 胜率 | LCB95 | 选择分 | 核心牌型 | package 名 |
|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | 2 | LumenLiquidity | 144 | 68.75% | 60.78% | 0.4958 | Dragapult ex / Dusknoir | `rank_002_dragapult_ex_dusknoir_bc` |
| 2 | 8 | THIRD PTCG Club | 901 | 61.15% | 58.04% | 0.7273 | Team Rocket's Mewtwo ex / Spidops | `rank_008_team_rockets_mewtwo_ex_bc` |
| 3 | 6 | tw_shin | 735 | 60.27% | 56.69% | 0.7185 | Mega Lopunny ex / Mega Froslass ex | `rank_006_mega_lopunny_ex_bc` |
| 4 | 10 | Yushin Ito | 1,000* | 58.30% | 55.22% | 0.7089 | Alakazam / Dudunsparce | `rank_010_alakazam_bc` |
| 5 | 26 | Kotaro OKUYAMA | 460 | 58.70% | 54.14% | 0.6739 | Marnie's Grimmsnarl ex / Froslass | `rank_026_marnies_grimmsnarl_ex_bc` |
| 6 | 98 | mitomeat823 | 1,000* | 52.70% | 49.60% | 0.6724 | Mega Lucario ex / Solrock | `rank_098_mega_lucario_ex_bc` |
| 7 | 57 | ei ei ei yikuso | 469 | 56.50% | 51.98% | 0.6662 | Alakazam / Dudunsparce | `rank_057_alakazam_bc` |
| 8 | 93 | Michael Long | 570 | 50.88% | 46.78% | 0.6541 | Alakazam / Dudunsparce | `rank_093_alakazam_bc` |
| 9 | 52 | wally0593 | 486 | 52.67% | 48.23% | 0.6537 | Alakazam / Dudunsparce | `rank_052_alakazam_bc` |
| 10 | 58 | palsystem | 459 | 53.81% | 49.35% | 0.6421 | Team Rocket's Mewtwo ex / Spidops | `rank_058_team_rockets_mewtwo_ex_bc` |
| 11 | 21 | nasuo445 | 420 | 58.10% | 53.32% | 0.6406 | Cynthia's Garchomp ex / Roserade | `rank_021_cynthias_garchomp_ex_bc` |
| 12 | 70 | koala_bear “もりたにあん” | 444 | 50.90% | 46.38% | 0.6122 | Mega Kangaskhan ex / Crustle | `rank_070_mega_kangaskhan_ex_bc` |
| 13 | 23 | \_\_Taichicchi\_\_ | 333 | 58.26% | 53.05% | 0.5779 | Marnie's Grimmsnarl ex / Froslass | `rank_023_marnies_grimmsnarl_ex_bc` |
| 14 | 61 | WinDecks | 334 | 53.89% | 48.53% | 0.5493 | Mega Starmie ex / Dusknoir | `rank_061_mega_starmie_ex_bc` |
| 15 | 38 | btk15049 | 311 | 56.27% | 50.71% | 0.5473 | Marnie's Grimmsnarl ex / Froslass | `rank_038_marnies_grimmsnarl_ex_bc` |
| 16 | 34 | Vadim Vasilenko | 323 | 54.49% | 49.04% | 0.5448 | Festival Lead / Dipplin | `rank_034_festival_lead_dipplin_bc` |
| 17 | 82 | ei1333 | 323 | 53.25% | 47.80% | 0.5368 | Marnie's Grimmsnarl ex / Froslass | `rank_082_marnies_grimmsnarl_ex_bc` |
| 18 | 27 | SQUIRTLE (prime) | 284 | 57.04% | 51.41% | 0.5329 | Marnie's Grimmsnarl ex / Froslass | `rank_027_marnies_grimmsnarl_ex_bc` |
| 19 | 50 | rick & shikitora | 290 | 53.45% | 47.70% | 0.5130 | Alakazam / Dudunsparce | `rank_050_alakazam_bc` |
| 20 | 94 | Shardul Gharat | 287 | 53.31% | 47.53% | 0.5099 | Alakazam / Dudunsparce | `rank_094_alakazam_bc` |

`*` 表示 Kaggle API 已触及最近 1,000 局上限。相同牌型或相同 exact deck 的不同 submission
仍是不同专家 policy；数据、模型、optimizer 和评测必须分开。

## 3. 明天训练前的实时刷新

训练启动前必须重新查询这 20 个**精确 submission ID**，利用未来约 5 小时新增对局：

1. 保留本报告的静态 Rank，不根据明天的新榜单改写历史索引；
2. 对每个选中 team 同时查询实时 leaderboard submission ID；
3. 若 submission ID 与本报告一致，重新列举该 submission 的全部公开已完成 Episode；
4. 若 team 已换 submission，禁止把新旧 submission 的动作标签混入同一 dataset：
   - 旧 submission 仍按旧 identity 训练；或
   - 把新 submission 当作新 source，重新下载一局、校验 deck hash、更新 roster revision；
5. 将刷新时间、旧/新场次、增加的 Episode ID、是否换 submission 写入
   `pretrain_refresh.json`；
6. 对 200 局边缘来源再次检查。若某来源消失、deck 改变或不足 200，按静态索引中的
   `selection_score` 选择下一位满足 200 局者替换；Lumen 不替换。

不能按 team name 合并旧 submission。BC manifest、dataset summary 和 run record 必须始终记录
唯一 `team_id + submission_id + deck_sha256`。

## 4. 单个专家的数据事务

每个专家独立执行以下事务，完成后才进入下一个 source 的长期归档：

```text
exact submission metadata refresh
  -> 下载全部公开 completed replay（API 最多最近 1,000 局）
  -> 每局按 submission_id 定位 player_index
  -> 校验每局 60-card multiset hash 与审计 deck 一致
  -> observation[i] + action[i+1]
  -> ptcg_features_universal
  -> episode-level chronological 80/10/10 split
  -> fail-closed dataset audit
  -> BC training / rescue
  -> package / evaluation
  -> cleanup
```

硬约束：

- 不依赖 replay 中可能变化的 display name 定位专家；以 `submission_id` 唯一定位 player index；
- 同一 Episode 的动作不能跨 train/validation/test；
- dataset 只允许一个 source policy，不使用 `--allow-multiple-experts`；
- action 必须存在于 observation 的合法 candidate 中；任何非法 label 直接停止；
- feature schema、tensor shape、card metadata、action contract 必须和最终共享模型配置一致；
- dataset audit 必须检查非零 card/entity/action token，防止再次出现“shape 正确但语义 token 全零”；
- 除 Episode 数外，训练前还要记录 train/validation/test 的 decision records。普通来源建议至少
  `15,000 / 1,500 / 1,500` 条；未达到时标记 `low_decision_density`，不静默当作足量数据。

现有 `download_expert_replays` 在正式批量执行前应收紧为：读取 source manifest、只保留
PUBLIC/COMPLETED、按 submission ID 定位 player，而不是只按 agent display name。该改造不改变
模型语义，但必须有单元测试。

## 5. 共享模型配置与低样本风险

今晚的单 GPU capacity search 继续以 1,000 局 Yushin Ito corpus 测试通用模型容量。20 模型
训练不得在搜索仍进行时猜测 Winner：

1. 等用户根据 capacity search HTML 选择最终结构；
2. 将选定的 `d_model / layers / hidden_dim / heads / dropout / batch size / base learning rate /
   epochs / seed` 写入只读 `shared_model_config.json`；
3. 记录来源 search experiment、trial version、checkpoint/config hash；
4. 20 个专家 V1 都从随机初始化使用同一份配置，不从 Yushin checkpoint warm start；
5. 模型结构保持一致。低样本来源只允许进入预定义的学习率救援，不针对每副卡组任意改网络。

风险判断：200–499 局并不意味着模型必然失败，但相对 1,000 局基线会有更少的稀有 selection
context 和更宽的 held-out 置信区间。报告必须并列展示 Episode、decision records、context 覆盖、
train-validation gap，不能只展示一个总体 Exact Action Rate。

## 6. 离线复现合格线与最多两次救援

当前 1,000 局 Yushin V2 的参考事实是：best validation exact 80.52%、test exact 78.37%、
test legal action 100%。本项目采用下面的统一门槛：

| 指标 | 合格线 | 用途 |
|---|---:|---|
| best validation exact action rate | ≥75% | 用户所说“成功复现操作的概率”的主门槛 |
| validation multi-action exact rate | ≥65% | 防止总体指标只靠简单单选动作 |
| validation selection-count accuracy | ≥98% | 多选数量契约 |
| validation legal action rate | 100% | 硬正确性门槛 |
| test legal action rate | 100% | 冻结模型后的最终审计，不参与调参 |

`75%` 是基于当前有效 1,000 局基线 80.52% 留出的低样本容差，不是假装成理论阈值。HTML 还要
展示各 selection context 的 records 和 exact；低样本 context 不单独触发失败。

每个专家最多 3 个 attempt：V1 基线 + 2 次救援。救援只读取 train/validation，不读取 test、
本地对战或 Kaggle 结果：

1. `V1_shared_config`：共享结构、共享 base LR、seed 7；
2. 若 validation exact <75%：
   - train-validation gap >12 个百分点、validation 后期回落或 loss 振荡：V2 使用相邻较低 LR；
   - train 与 validation 都低且最后 3 epoch 仍同步改善：V2 使用相邻较高 LR；
   - 证据不明确时先用较低 LR；
3. V2 仍不合格时，`V3_lr_other_side` 使用尚未测试的另一侧相邻 LR；
4. 三次中只按 validation exact 选择 checkpoint；差异小于 0.3 个百分点时，选 validation
   policy loss 更低者；
5. V3 仍低于 75% 时标记 `imitation_below_gate`，保留最佳模型和报告，但不得加入 opponent pool。

若 capacity search 选中的 base LR 是 `3e-4`，默认相邻救援点为 `1e-4` 和 `5e-4`。若 Winner
位于边界，执行 Session 必须在 `shared_model_config.json` 中预先写出两个救援 LR，不能看到
某副卡组 evaluation 后临时发明参数。

## 7. 10 小时单 GPU 调度

“至少完成 20 套”优先于把前几个失败模型无限调好：

1. **Phase A：20 个 V1 全覆盖。** Lumen 第一，其余按本表排程序。每个 V1 训练后立刻做离线
   gate，但先保证 20 个 source 都有一个完整 baseline；
2. **Phase B：救援队列。** 只对 V1 未达 75% 的模型执行 V2/V3，顺序为 Lumen 优先，之后按
   `selection_score`；每个最多两次，并受 10 小时总 wall-time 限制；
3. **Phase C：package + evaluation 补齐。** 每个通过 gate 的 best checkpoint 都必须完成
   package validate 和固定评测；未通过者生成失败报告，不用 evaluation 结果继续调 LR。

Yushin V2 当前 20 epochs 约 10–12 分钟。20 个 V1 粗估 4 小时；剩余训练预算用于救援。
下载每个 1,000 局 source 可能约 15–20 分钟，因此允许一个受控的两槽流水线：

```text
GPU:     train/evaluate source N
network: download source N+1（唯一一个预取 source）
```

GPU 训练始终单进程；Kaggle 下载最多 2 workers、全局请求间隔至少 1 秒；磁盘最多同时保留“当前
source + 下一个预取 source”的完整 replay。不得并行启动两个训练。

## 8. Run ID、命名与归档

Capacity search 可能先占用 `0004`，所以不能硬编码编号。每个专家开始时调用：

```bash
python3 -m rl.core.runs create <package-name> \
  --objective "Train one exact Top-100 expert BC surrogate"
```

实际返回值可能是 `0005-...`、`0006-...`，必须原样作为 `<experiment-id>`。目录：

```text
rl/_runs/<experiment-id>/
├── manifest.json
├── source_manifest.json
├── pretrain_refresh.json
├── data_manifest.json
├── audit.json
├── commands.md
├── V1_shared_config/
├── V2_lr_lower/                 # 仅需要时
├── V3_lr_other_side/            # 仅需要时
└── evaluation/<attempt-id>/

rl/artifact/dataset/<experiment-id>/
rl/artifact/checkpoint/<experiment-id>/<attempt-id>/
rl/_runs/tensorboard/<experiment-id>/<attempt-id>/
submission/top_player/<package-name>/
```

`<package-name>` 固定使用报告静态 Rank，不用训练时的新 Rank：

```text
rank_<三位静态Rank>_<核心pokemon_slug>_bc
```

## 9. 标准训练命令

参数从用户最终选定的 `shared_model_config.json` 展开，不能抄本文中的假设值：

```bash
python3 -m rl.train.train_full_action_bc \
  rl/artifact/dataset/<experiment-id>/dataset.jsonl \
  --output rl/_runs/<experiment-id>/V1_shared_config \
  --epochs <shared.epochs> \
  --batch-size <shared.batch_size> \
  --learning-rate <shared.base_learning_rate> \
  --seed <shared.seed> \
  --d-model <shared.d_model> \
  --hidden-dim <shared.hidden_dim> \
  --num-heads <shared.num_heads> \
  --transformer-layers <shared.transformer_layers> \
  --dropout <shared.dropout> \
  --device cuda
```

每个 attempt 保存 config、逐 epoch metrics、summary、checkpoint hash、wall time、峰值显存、
gate 结果和选择理由。不得覆盖失败 attempt。

## 10. 标准 Agent 与 `submission/top_player`

每个通过离线 gate 的 best checkpoint 必须物化为：

```text
submission/top_player/<package-name>/
├── main.py
├── deck.csv                     # 本报告审计的 exact 60 张
├── cg/                          # 官方 runtime 的物理复制
└── strategy/
    ├── model.bin
    ├── manifest.json
    └── 通用 inference/model/features 源码
```

当前 `build_full_action_submission` 把 deck 与 `--source-package` 耦合。批量执行前必须支持独立的
`--deck` 与 `--cg-source`，并交叉校验 checkpoint feature/model metadata、deck hash、源码 hash
和 cg tree hash。不能拿 Yushin 的 source package 误装其他人的 deck。

验证：

```bash
python3 -m evaluation validate submission/top_player/<package-name>
```

Agent 只返回 simulator 提供的合法 option，不含规则 fallback；`select=None` 时返回自己的固定
deck。模型权重的 canonical copy 保留在 checkpoint artifact，package 中保存推理副本和 hash。

## 11. Repository Evaluation

每个通过离线 gate 的 package 运行固定 18 opponent × 10 games：

```bash
python3 -m evaluation run \
  --candidate submission/top_player/<package-name> \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output rl/_runs/<experiment-id>/evaluation/<attempt-id>
```

必须报告：180 局完成性、W-L-D、总胜率、先后手、每对手胜率、errors/unfinished、legal action、
Powerful Hand、二回合 setup、Post-KO relay、attack quality 与 library pressure。晋级硬护栏：

- 180/180 完成；
- candidate errors = 0；
- illegal action = 0；
- 离线 gate 全部通过。

本地胜率不设自动晋级线，由用户晚上查看报告后决定 opponent pool。Evaluation/test 结果不能用于
回头选择 LR，避免把固定评测集调穿。

## 12. 存储清理

每个 source 的 V1/V2/V3、package 和 evaluation 全部完成后：

1. 永久保留 run manifest、source identity、deck.csv、Episode ID 列表、split、dataset SHA-256、
   summary/audit、训练 metrics、best checkpoint、package 和 evaluation；
2. 按用户要求立即删除该 source 的大型派生 `dataset.jsonl`，但保留 90 KB 量级的 card metadata
   和 summary；删除前必须确认不再需要 V2/V3；
3. 完整 raw replay 暂时保留到用户完成 opponent-pool 审核。它们可能在 Kaggle 1,000 局滚动
   窗口中消失，提前删除会使 exact dataset 不可复现；
4. 用户审核后，可删除 full-window raw replay，仅保留本次 Top 100 代表 replay、manifest 和 hash；
5. 清理动作写入 `cleanup.json`，记录删除路径、bytes 和不可恢复边界。禁止删除 checkpoint、
   package、evaluation 或仓库级调研快照。

## 13. 执行前检查清单

- [ ] Capacity search 已结束，用户已选择共享模型结构；
- [ ] `shared_model_config.json` 已冻结 base LR 与两个 rescue LR；
- [ ] 20 个 exact submission 已实时刷新，换 submission 的来源未混入；
- [ ] Lumen 仍为强制 roster 1；
- [ ] downloader 已按 submission ID 定位并只保留 PUBLIC/COMPLETED；
- [ ] builder 已支持独立 deck/cg source；
- [ ] run ID 动态分配，没有和其他 Session 争用；
- [ ] 单 GPU 串行，只有一个下一-source 下载预取；
- [ ] 每个 dataset 单 source、episode split、semantic token audit 通过；
- [ ] 20 个 V1 优先完成，再消费救援预算；
- [ ] 每个成功模型都能从 package 冷启动、validate 并完成 180 局 evaluation；
- [ ] dataset 清理只在所有 rescue 完成后执行，raw replay 延迟到用户审核后删除；
- [ ] 未执行 Kaggle submission，未修改官方 engine runtime。

