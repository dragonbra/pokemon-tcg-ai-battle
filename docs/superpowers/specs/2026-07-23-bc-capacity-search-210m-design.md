# BC 模型容量与超参数搜索（210 分钟训练预算）设计

## 1. 文档用途

本文是交给执行 Session 的完整实验规格。执行者必须继续使用 `0003-yushin_ito_exact_bc_v2`
正在使用并由另一个 Session 修正、最终冻结的 **Yushin Ito 人类专家 BC 数据集**，在单 GPU
上完成一次有明确时间上限的容量/优化搜索。这里的“继续使用 0003 数据”是指每个超参 trial
都从随机初始化出发，继续对同一批人类 replay decision 做 BC；不是从 0003 V2 checkpoint
续训，也不是换成新专家、合成数据、self-play 或 MCTS label。

每个完成的超参 trial 都必须用它自己的 `best_validation.pt` 构建纯模型 Agent，并立即通过
仓库正式 `evaluation` 运行全部 18 个 opponent × 10 games。离线 BC 与真实对局结果随后写入
`rl/_runs/INDEX.html` 和 campaign HTML 报告，供用户明早自行决定最终模型结构。

这不是为了机械地寻找最高的单个 `validation/exact_action_rate` 数字，而是回答三个问题：

1. 对当前已经完备的状态输入而言，什么规模是“足够容量”的模型；
2. 在当前专家数据上，纯 BC 能可靠训练到什么程度，主要瓶颈更像容量、优化还是泛化；
3. 下一轮模型设计最值得投入宽度、深度、优化协议、正则化，还是已经应该转入 rollout/RL。

本实验不修改 feature schema、dataset、label、action contract、loss 定义或 evaluation
opponent。若这些内容在实验过程中变化，所有跨 trial 比较失效，必须停止并重新冻结 campaign。

## 2. 成功定义

实验完成时必须同时具备以下交付物：

- 所有 trial 使用 0003 最终修正版的同一份 Yushin Ito dataset、episode split、特征/动作
  契约和源码 revision；
- 累计训练进程 wall time 不超过 210 分钟（12,600 秒）；
- 至少完成学习率基线、宽度、深度和宽深组合四个方向的比较；
- 每个完成的 trial 都从 `best_validation.pt` 构建 `fallback: None` 的临时纯模型 Agent；
- 每个完成的 trial 都有独立的仓库正式 evaluation（18 opponents × 10 games）和
  `report.html`；
- `rl/_runs/INDEX.html` 逐 trial 展示模型差异，并以 evaluation 核心结果为视觉重点；
- 生成一个独立、可直接打开的 HTML 总报告，既展示 BC loss/validation action accuracy，
  也展示每个 trial 的真实对局结果；
- 同步 `rl/model/DESIGN.html`、实验 decisions/status 和 `rl/_runs/INDEX.html`。

本 campaign 不替用户给出“哪个模型最好”或自动晋级结论。执行 Session 可以按本文预先定义的
validation 规则分配剩余训练预算，但不得用已经看到的 evaluation 或 test 结果决定下一组超参。
最终只提供完整、同口径、可审计的对比证据，由用户明早选择最终结构。

## 3. 实验边界

### 3.1 固定项

下面各项在第一个 trial 启动前冻结，整个 campaign 不允许改变：

- dataset 文件及其 SHA-256；
- dataset card metadata 文件及其 SHA-256；
- train/validation/test episode split；
- feature schema 与所有 tensor shape；
- action contract、candidate mask、BC loss 和 checkpoint 选择指标；
- optimizer 类型（当前 AdamW）及未被搜索的 optimizer 参数；
- batch size `256`；
- dropout `0.0`（除非进入第 9.3 节的过拟合分支）；
- 主搜索 seed `7`；确认 seed `17`；
- epoch 数 `20`；
- evaluation opponent catalog、games 数和 metric profile revision；
- 同一份 source package 的 60 张 deck 与官方 `cg/` runtime。

主矩阵不搜索 batch size、heads、scheduler、warmup 或 loss 权重。它们会同时改变优化动力学或
实验解释，在 210 分钟预算下不应与容量搜索混在一起。所有 `d_model` 均保持 4 heads，保证
`d_model` 可整除且让 heads 不成为额外变量。

### 3.2 禁止项

- 不得读取 test 指标后再决定下一组超参数或形成最终结构建议；
- 不得用正式 evaluation 选择模型后再返回训练；
- 不得并行启动多个 GPU 训练；
- 不得覆盖或复用已有 `V<n>_<tag>`；失败 trial 也要保留编号和状态；
- 不得修改 `engine/source/`、evaluation opponent 或 metric 语义；
- 不得引入规则 fallback 来掩盖模型输出；
- 不得执行 Kaggle submission；
- 未获得用户新的明确授权时，不执行 git commit/push。

## 4. Campaign 冻结与归档

### 4.1 分配新实验编号

不要把搜索结果继续写入正在被其他 Session 修改的 `0003` 目录。执行时通过正式入口动态分配
新的全局 experiment id，不在本文中硬编码 `0004`：

```bash
PYTHONPATH=. python3.11 -m rl.core.runs create bc_capacity_search_210m \
  --objective "Measure BC capacity and evaluation behavior under a 210-minute training budget"
```

将命令返回的真实编号记为 `<experiment-id>`。所有 trial 使用严格递增、不可复用的
`V<n>_<tag>`，并在下面三个位置保持同名：

```text
rl/_runs/<experiment-id>/V<n>_<tag>/
rl/_runs/tensorboard/<experiment-id>/V<n>_<tag>/
rl/artifact/checkpoint/<experiment-id>/V<n>_<tag>/
```

### 4.2 冻结证据

在启动第一个 trial 前，把以下信息写入 experiment 根目录的 `search_manifest.json`：

- dataset 与 card metadata 的绝对路径、bytes、SHA-256；
- data manifest/audit 路径与 SHA-256；
- train/validation/test 的 episodes 和 records 数；
- Git commit、`git status --porcelain` 和影响本实验源码的 diff/patch；
- `rl/model/`、`rl/train/train_full_action_bc.py`、`rl/core/` 的源码 hash；
- Python、PyTorch、CUDA、GPU 型号；
- 训练预算 `12,600` 秒、停止发车阈值 `11,700` 秒；
- 本文档路径和 revision；
- source package、deck hash 和 cg tree hash。

如果工作树是 dirty 的，必须保存能恢复实验源码的 `source_revision.patch`。正在运行的其他
Session 若仍会改动模型或数据，先等待其明确冻结；不得一边训练一边让源码发生变化。

本 campaign 的正式输入必须是 0003 在修复完成后冻结的下列人类专家数据资产：

```text
rl/artifact/dataset/0003-yushin_ito_exact_bc_v2/dataset.jsonl
rl/artifact/dataset/0003-yushin_ito_exact_bc_v2/dataset.jsonl.card_metadata.json
rl/artifact/dataset/0003-yushin_ito_exact_bc_v2/dataset.jsonl.summary.json
```

它继续使用同一个 Yushin Ito 人类 expert corpus 和同一组 episode-level split；不得重新下载、
追加其他专家、改变 split 或把 rollout/self-play 混入。若修复 Session 在冻结时生成了新的明确
revision 路径，执行者可以使用该最终路径，但必须证明它仍属于 0003 的同一人类 corpus，并在
`search_manifest.json` 中记录旧/新路径及 hash。新建 `<experiment-id>` 只是为了隔离超参 trial
的 run/checkpoint/TensorBoard/evaluation 产物，不代表换数据集。

0003 已有 V2 训练结果只能作为背景参考；本 campaign 的 Baseline 也必须从随机初始化重新训练，
保证所有超参 trial 使用同一冻结源码和协议。

## 5. 时间预算

### 5.1 硬约束

- 总训练预算：`210 min = 12,600 s`；
- 停止启动新 trial 的软阈值：累计 `195 min = 11,700 s`；
- 最后 15 分钟是超时安全余量，不是必须花完的额度；
- 预算统计使用每个 `train_full_action_bc` 进程从启动到退出的 wall time，包含数据加载、
  每 epoch train/validation evaluation、checkpoint 和训练器自动执行的 test；
- candidate 打包、repository evaluation 和 HTML 生成不计入 210 分钟训练预算，但必须分别
  记录 wall time，并在明早交付前全部完成；
- 单 GPU 串行执行，不通过并行训练压缩 wall time。

每个 trial 使用完整事务：`train -> package -> validate -> 18×10 evaluation -> 更新 INDEX`。
前一个完成 trial 的 evaluation 和索引落盘前，不启动下一个训练；这样预算结束时不会留下大量
尚未评测的 checkpoint。evaluation wall time 不计入训练秒数，但必须展示在报告中。

每个 trial 完成后立即把实际秒数、每 epoch 中位秒数和 peak GPU memory 写入
`trial_result.json`。启动下一 trial 前，按下式预测耗时：

```text
predicted_seconds = max(同结构已测 wall time, 前两 epoch 中位耗时 × 20) × 1.15
```

只有满足下面条件才允许发车：

```text
cumulative_training_seconds + predicted_seconds <= 11,700
```

若没有同结构历史，使用第 6 节的保守估计。不得为了“刚好用满预算”启动明显无法完整跑完的
trial。意外失败或被中断的 trial 记录实际耗时，并在 INDEX 中显示失败原因；没有完整有效
`best_validation.pt` 的 trial 不运行正式 evaluation。

### 5.2 初始耗时估计

已知 Baseline 20 epochs 约 12 分钟。下面只用于排程，必须被首轮实测覆盖：

| 结构 | 预计耗时 | 保守上限 |
|---|---:|---:|
| Small | 8–10 min | 12 min |
| Baseline | 12 min | 15 min |
| Wide | 20–25 min | 30 min |
| Deep | 18–24 min | 30 min |
| Large | 30–40 min | 48 min |

## 6. 搜索空间

### 6.1 容量结构

`hidden_dim` 固定为 `2 × d_model`，先把宽度和深度作为可解释的主要容量旋钮：

| 逻辑 ID | d_model | layers | hidden_dim | heads | 作用 |
|---|---:|---:|---:|---:|---|
| S | 192 | 2 | 384 | 4 | 下探；判断当前 Baseline 是否已经冗余 |
| B | 256 | 2 | 512 | 4 | 当前 Baseline |
| W | 384 | 2 | 768 | 4 | 只增加宽度 |
| D | 256 | 4 | 512 | 4 | 只增加深度 |
| L | 384 | 4 | 768 | 4 | 宽深组合；容量压力测试 |

所有 trial 必须记录真实 `parameter_count`、checkpoint bytes、peak GPU memory 和 wall time，不能
仅用逻辑名称猜测复杂度。

### 6.2 学习率

搜索集合固定为：

```text
1e-4, 3e-4, 5e-4
```

学习率是容量判断的配套变量：一个更大的模型若只在不合适的学习率上失败，不能据此认定增加
容量无效。

## 7. 自适应执行顺序

执行者必须遵循“先回答最重要问题，再使用剩余预算确认”的顺序，而不是盲目跑满笛卡尔积。
每个条目训练完成后，先完成第 11–13 节规定的 candidate、evaluation 和 INDEX 更新，再进入
下一个条目。搜索控制只消费 train/validation，不消费已生成的 evaluation 结果。

### Phase A：Baseline 学习率标定（必做）

从随机初始化分别训练：

1. `B / lr=1e-4 / seed=7`；
2. `B / lr=3e-4 / seed=7`；
3. `B / lr=5e-4 / seed=7`。

预计训练预算约 36 分钟。只根据 validation 选择 Baseline 最佳学习率 `lr_B*`。若差异小于
0.3 个百分点，优先 validation policy loss 更低、曲线更平稳者；仍相当时保留 `3e-4`。

### Phase B：容量阶梯（必做）

使用 `lr_B*`，从随机初始化完成：

1. `S / seed=7`；
2. `W / seed=7`；
3. `D / seed=7`；
4. `L / seed=7`。

加上 Phase A，预计累计训练约 115–135 分钟。至此必须能够初步回答：

- Small 与 Baseline 的差异有多大；
- 宽度和深度哪一种更有效；
- Large 是否继续提高 train 与 validation；
- 更大模型是否只是提高 train，开始扩大泛化 gap。

### Phase C：优化救援或过拟合验证（按证据选择，最多一次）

在 Phase A+B 预计使用 115–135 分钟后，只为最影响容量解释、且明显可能被学习率误伤的一个
结构增加一次主 seed 救援 trial。优先观察 L，其次是 W/D 中 validation 较高但曲线未收敛者。

- 若 loss 明显振荡、validation 后期反复回落，改用更低的相邻学习率；
- 若 train loss 单调下降且第 20 epoch 仍未平台，改用更高的相邻学习率；
- 若更大模型 train 提高而 validation 下降、gap 扩大，进入第 9.3 节的 dropout 分支，
  不再盲目增加容量；
- 若某结构在 `lr_B*` 已经稳定且明显落后，不为它浪费救援预算。

Phase C 的目标是排除“容量配置被不合适学习率误判”，不是把每个结构补成完整三点网格。

### Phase D：单次 Seed 确认（仅在剩余预算足够时）

若完成 Phase A+B 和至多一次 Phase C 后，按实测耗时预测仍满足 11,700 秒停止发车线，可用
`seed=17` 从随机初始化重跑一个最有信息量的结构：

1. validation 表现领先但只比更小模型略高的结构；或
2. 决定“继续扩容还是已经平台”这一判断的临界结构。

本轮不安排 seed `42`，也不要求同时确认两个结构。Seed 确认仍优先于搜索 batch size、heads、
scheduler。若剩余预算不足就明确标记 seed 不确定性，不挤占 15 分钟安全余量。

### Phase E：停止训练

满足任一条件立即停止发车：

- 累计训练达到 11,700 秒；
- 剩余安全预算不足以完成下一个预测 trial；
- 必做容量阶梯已经完成，额外 trial 无法在停止发车线前完整结束；
- 后续候选只是在重复已经被回答的问题。

停止后不得用剩余零碎时间临时发明新超参数。

## 8. 标准训练命令

每个 trial 都从随机初始化开始，不从其他 checkpoint warm start。示例：

```bash
PYTHONPATH=. python3.11 -m rl.train.train_full_action_bc \
  <frozen-dataset.jsonl> \
  --output rl/_runs/<experiment-id>/<Vn_tag> \
  --epochs 20 \
  --batch-size 256 \
  --learning-rate <lr> \
  --seed <seed> \
  --d-model <d_model> \
  --hidden-dim <hidden_dim> \
  --num-heads 4 \
  --transformer-layers <layers> \
  --dropout <dropout> \
  --device cuda
```

`V<n>_<tag>` 使用清楚的标签，例如：

```text
V1_b_d256_l2_lr1e4_s7
V2_b_d256_l2_lr3e4_s7
V3_b_d256_l2_lr5e4_s7
V4_s_d192_l2_bestlr_s7
```

实际版本号必须顺序递增；以上仅是示意，不得在已有版本发生冲突时复用。

每次训练后检查：

- `training_config.json` 与计划一致；
- `training_metrics.jsonl` 恰有 20 个 epoch；
- `training_summary.json` 存在；
- `best_validation.pt` 与 `latest.pt` 存在；
- TensorBoard event 存在；
- 没有 NaN/Inf、CUDA OOM 或维度错误；
- checkpoint metadata 中 feature schema、model config、dataset 与冻结值一致。

## 9. 判定模型容量与 BC 上限

### 9.1 必须使用的离线指标

主指标：

- `best_validation_exact_action_rate`；
- best epoch；
- best checkpoint 对应的 train exact-action rate；
- validation/train policy loss；
- train-validation exact gap。

辅助指标：

- single-action accuracy；
- multi-action exact rate；
- selection-count accuracy；
- 各 selection context 的 exact-action rate 与样本量；
- context macro average，避免高频简单动作主导总体结果；
- 学习曲线在第 20 epoch 是否仍有明显斜率。

不能只看最后一个 epoch，也不能只看总体 exact。对于样本很少的 context，不根据单次百分比
大幅波动下结论。

当前训练器会在每个 trial 结束后自动生成 `final_test`。搜索控制器不得读取、展示或引用这个
字段来决定后续 trial；INDEX 的 BC 辅助区也不以 test accuracy 排名。若执行者选择增加
`--skip-test` 一类窄接口来彻底隔离 test，必须补测试并保持训练/模型语义不变；否则至少保证
所有自动搜索决策只解析 train/validation。用户明早可以在原始 artifact 中审计 test，但本
campaign 不据此给出最终结构结论。

### 9.2 “足够容量”的操作定义

“足够容量”不是参数最多的模型。报告应把下面条件作为用户决策所需的证据维度，而不是由执行
Session 自动宣布一个胜者：

1. 在调过学习率后，较小模型的 validation exact 是否落在最高观测值 0.3 个百分点以内；
2. validation policy loss 没有材料性劣化；
3. 高频和关键 context 没有一致的明显回退；
4. 增加宽度或深度只能明显提高 train，不能稳定提高 validation；
5. 各规模模型能否通过 package validation 和 evaluation correctness 护栏，以及离线差异是否
   转化为实际胜率/过程指标差异。

0.3 个百分点是近似等价观察区间，不是假装成统计显著性。若 seed 间波动大于该区间，应明确
写“证据不足”，把原始差异完整展示给用户，不替用户选择更小或更大的模型。

### 9.3 诊断规则

| 观察 | 结论 | 下一方向 |
|---|---|---|
| 更大模型同时提高 train 和 validation | 当前容量不足 | 继续扩容或保留更大模型 |
| 更大模型提高 train，validation 基本不动 | 有用容量已平台 | 数据覆盖、目标或 RL，而非继续堆参数 |
| 更大模型提高 train，validation 下降且 gap 扩大 | 过拟合 | dropout/weight decay/更多数据 |
| 所有容量 train 都低，学习率影响大于结构影响 | 优化受限 | scheduler、warmup、optimizer |
| 第 20 epoch train/validation 仍同步提高 | 训练预算不足 | 增加 max epochs/early stopping，而非先扩容 |
| 宽模型有效、深模型无效 | 表示/通道容量不足 | 优先提高 d_model/hidden_dim |
| 深模型有效、宽模型无效 | 组合推理深度不足 | 优先增加 Transformer layers |

若出现明确过拟合，Phase C 可在当前最有潜力的大模型上仅比较 `dropout=0.05` 与现有
`dropout=0.0`；只有预算充足时再试 `0.1`。不要同时修改学习率、dropout 和 weight decay。
当前 AdamW weight decay 没有 CLI 暴露，沿用当前默认值并在报告中明确记录。

### 9.4 “BC 能训练到什么程度”的表述

最终事实摘要必须区分：

- 训练拟合上限：train loss/exact 能到哪里；
- 当前验证上限：调过容量和学习率后 validation 的平台；
- 真实策略效果：固定 opponent evaluation 中的胜率、正确性和策略阶段指标。

Exact Action Rate 不等同于策略价值。完整状态下仍可能存在多个等价合法动作，因此不得把
“Exact 没到 100%”简单解释为模型能力不足。只有容量阶梯、学习率救援和 seed 确认共同支持
时，才可以称为当前训练协议下的“经验 BC ceiling”，不得称为理论上限。

## 10. 对比冻结与用户决策边界

执行 Session 不选择 Winner，不写 promote/reject，也不把某个 checkpoint 自动复制为正式下一
阶段模型。训练停止后生成 `comparison_manifest.json`，逐 trial 固化：

- version、结构超参、optimizer 超参、seed 和源码/dataset hash；
- 参数量、训练/评测 wall time、checkpoint hash；
- train/validation BC 指标和曲线 artifact；
- 独立 evaluation run id、profile/catalog hash、完成性、胜负和核心过程指标；
- complete、failed、OOM、invalid 或 skipped 状态及原因。

表格使用逻辑实验顺序或版本号排序，不按胜率、Exact Action Rate 或人工综合分排序，不添加
“最佳”“推荐”“晋级”徽标。执行 Session 可以客观描述宽度、深度、学习率和容量曲线的现象，
但最终模型结构由用户明早根据 INDEX 和 HTML 证据决定。

Phase A–D 的自适应分支只允许使用 train/validation 曲线；虽然每个 trial 会立即产生
evaluation，搜索控制不得用 evaluation 选择后续 trial。test 也不得用于分支或排序。

## 11. 每个 trial 构建真实 Agent package

每个成功完成 20 epochs 并生成有效 `best_validation.pt` 的 trial，都使用该 checkpoint 构建
自己的纯模型 candidate。source package 必须与冻结的 0003 expert deck 一致，并通过 deck/cg
检查；不要无条件依赖已经移动或删除的旧 `work/alakazam_v9`。当前可用路径应在执行时验证，
例如 `work/alakazam_bc_v1`。

在用户尚未选择最终结构前，不把每个搜索 candidate 污染到长期 `work/`。使用唯一临时目录：

```bash
PYTHONPATH=. python3.11 -m rl.train.build_full_action_candidate \
  --checkpoint rl/artifact/checkpoint/<experiment-id>/<trial-version>/best_validation.pt \
  --source-package <validated-source-package> \
  --output /tmp/<experiment-id>-candidates/<trial-version>

python3 -m evaluation validate /tmp/<experiment-id>-candidates/<trial-version>
```

每个 candidate manifest 必须满足：

- `schema_version = ptcg_pure_model_candidate_v1`；
- `fallback = null`；
- checkpoint 路径和 SHA-256 指向对应 trial，而不是其他版本；
- 60 张 deck 与 0003 人类专家数据的 deck 相同；
- `cg/` 与 evaluation catalog 兼容；
- 实际 agent 从 checkpoint 加载模型，不调用规则策略作为后备。

临时 candidate 可以在全部报告完成后删除；checkpoint、candidate hash、source package hash 和
evaluation manifest 必须长期保留，保证用户选定结构后能够重建正式 `work/<name>`。

## 12. 每个 trial 的 Repository evaluation

### 12.1 一 trial 一份正式报告

当前 evaluation CLI 的 `--control` 只展示 control package 信息，并不会实际执行 control 对局。
因此本 campaign 不使用 `--control` 伪造比较；每一个成功训练的 trial 都作为独立 candidate
完成一轮正式 evaluation：固定全部 18 个 opponent、每个 10 局、相同 profile、max steps 和
catalog revision，共 180 局。

一个 completed trial 如果没有对应 evaluation `report.html`，就不算交付完成，也不能在 INDEX
中显示为绿色/正常。OOM、NaN、metadata 不一致或未满 20 epochs 的 trial 在 INDEX 显示失败，
不使用残缺 checkpoint 跑正式 evaluation。

### 12.2 标准命令

```bash
python3 -m evaluation validate /tmp/<experiment-id>-candidates/<trial-version>
python3 -m evaluation run \
  --candidate /tmp/<experiment-id>-candidates/<trial-version> \
  --opponents all \
  --games 10 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output rl/_runs/<experiment-id>/evaluation/<trial-version>
```

每轮 evaluation 完成后立即读取实际生成目录中的 `manifest.json`、`summary.json`、
`metrics.json` 和 `report.html`，写回该 trial 的 `trial_result.json`，然后更新
`rl/_runs/INDEX.html`。不得等所有训练结束后才补一批无法核对来源的手工数字。

### 12.3 每个 trial 必须提取的指标

- 计划 games、完成 games、errors、unfinished 和 correctness/illegal action；
- wins、losses、draws、win rate；
- 实际先手和后手的 games、win rate；
- 按 18 个 opponent 的 W/L/D、win rate、errors；
- Powerful Hand 与二回合 setup；
- Post-KO relay；
- attack quality；
- Rare Candy、Run Away Draw、library pressure 等健康指标；
- metric profile id/revision、catalog/hash、candidate/checkpoint hash；
- evaluation wall time 和原始 `report.html` 相对链接。

若某个 trial correctness 护栏失败，INDEX 与总报告用显眼的错误状态展示事实，但不自动得出
最终结构结论。若失败来自公共 inference/runtime bug，必须停止后续训练，修复后重新冻结源码；
不得把修复前后 trial 放在同一可比矩阵中。

## 13. HTML 展示与 INDEX 重构

### 13.1 `rl/_runs/INDEX.html` 是首要入口

用户首先从 `rl/_runs/INDEX.html` 查看实验，因此每个 completed trial 的 evaluation 完成后都要
立即刷新 INDEX，而不是只在 campaign 结束时生成一个孤立报告。

现有 INDEX 以“模型/离线指标”为主、evaluation 只占一列，并带有 keep/reject 结论。本次应
优化为 **evaluation-first** 结构，同时保留历史 run 和原始链接。搜索 campaign 在顶层 run 下
按 `V<n>` 展开，每一个超参 trial 都有独立行。

建议列顺序如下：

| 区域 | INDEX 必须展示的字段 |
|---|---|
| Trial | experiment/version、状态、模型差异简述 |
| Evaluation 结果 | completed/180、W-L-D、win rate、errors、unfinished |
| Evaluation 分解 | 先手/后手胜率、关键过程指标摘要、原始 report 链接 |
| 模型结构 | d_model、layers、hidden、heads、parameter count |
| 训练配置 | learning rate、seed、dropout、20 epochs、wall time |
| BC 辅助 | train loss、validation loss、validation exact、best epoch |
| Artifacts | config、metrics、checkpoint metadata、evaluation、campaign report |

视觉层级要求：

- W-L-D、win rate、errors/unfinished 和 evaluation report 链接放在最醒目位置；
- validation exact 与 loss 作为 BC 阶段辅助信息，字体/配色次于真实对局指标；
- 不在主 INDEX 突出 test exact；
- correctness failure/error 使用明显状态，但不自动写 keep/reject；
- 不显示“Winner”“最佳”“推荐”或综合排行榜；
- 同一 campaign 的 trial 采用固定逻辑/版本顺序，确保用户能直接看到每次超参变化；
- 详细的 18-opponent 分解留在各自 `report.html` 和 campaign report，INDEX 保持可扫描。

INDEX 中的数字必须从 `training_summary.json`、`trial_result.json`、evaluation `summary.json` 和
`metrics.json` 生成，不能手抄。历史条目若缺少同口径字段，显示 `not available`，不能伪造。

### 13.2 Campaign 独立 HTML 报告

同时生成：

```text
rl/_runs/<experiment-id>/bc_capacity_search_report.html
```

报告必须是独立 HTML，可直接从文件系统打开；CSS 和图表使用内联 CSS/SVG，不依赖 CDN、
网络 JavaScript 或外部字体。数字从 run JSON、checkpoint metadata 和 evaluation artifacts
读取。如果需要新增生成器，应放在 `rl/train/` 或合适的 RL 模块中，不放回通用 `scripts/`。

### 13.3 Campaign 报告必须展示的内容

1. **实验范围**
   - 明确写出所有 trial 继续使用 0003 Yushin Ito 人类专家 dataset；
   - dataset/source/git hashes 与固定 split；
   - 不混入新数据、不续训旧 checkpoint 的边界。

2. **预算与完成性**
   - 210 分钟训练预算、实际使用分钟、剩余分钟；
   - 每个 trial 的训练/evaluation wall time 与峰值显存；
   - completed、failed、OOM、invalid、skipped 及原因；
   - 每个 completed trial 是否拥有 180-game evaluation。

3. **完整 trial 对比表**
   - version、结构、learning rate、seed、parameter count、checkpoint bytes；
   - best epoch、train/validation loss、validation exact、gap；
   - evaluation completed、W-L-D、win rate、errors/unfinished；
   - 先后手与关键过程指标；
   - 全部原始 artifact/report 相对链接。

4. **学习曲线与容量图**
   - train/validation exact vs epoch；
   - train/validation policy loss vs epoch；
   - validation exact vs parameter count；
   - evaluation win rate vs parameter count；
   - evaluation win rate vs validation exact；
   - runtime/显存 vs 参数量；
   - 图例明确，不截断坐标制造夸大差异。

5. **按 context 的 BC 表现**
   - 每个 trial 的样本量、micro/macro exact 和差值；
   - 对低样本 context 标注不确定；
   - 动作准确率只能作为 BC 拟合证据，不能替代 evaluation。

6. **每个 trial 的真实 Agent evaluation**
   - 各自 180 局的完成性、correctness、W-L-D 与胜率；
   - 18-opponent 分解；
   - 先手/后手；
   - Setup/Powerful Hand/Post-KO relay/attack quality/library pressure；
   - 每个原始 `report.html` 的相对链接；
   - 将 Exact Action Rate 与真实对局指标并排展示，不强行声称二者必然相关。

7. **中立观察与待用户决策项**
   - 宽度、深度、学习率变化分别产生了什么可观察差异；
   - 是否存在欠拟合、平台、优化受限或过拟合信号；
   - seed/样本不确定性；
   - 后续可能方向，但不指定最终模型结构，不写 Winner/keep/reject。

HTML 中不得只贴总胜率；必须同时展示错误/未完成、opponent 分解和过程指标。也不得只展示
少数领先 trial，所有完整 trial 和失败 trial 都必须有可审计记录。

## 14. 项目文档同步

实验结束后同步：

- `rl/model/DESIGN.html`：当前输入/shape 未变化的事实、仍使用 0003 人类数据的边界、搜索过的
  结构、真实参数量、BC 训练目标、各 trial evaluation、当前处于“等待用户选择最终结构”的
  阶段；用户未决定前不得把任一 trial 写成正式最终结构；
- `rl/_runs/<experiment-id>/decisions.md`：自适应分支、训练停止原因、未执行 trial 及原因；
- 每个 trial 的 `status.json`；
- `rl/_runs/INDEX.html`：按第 13.1 节改成 evaluation-first，并逐 trial 展示核心指标、BC 辅助
  指标和 HTML 链接；
- 必要时补 `commands.md`，保证每次训练、临时打包和每轮 evaluation 可复现。

更新 `rl/model/DESIGN.html` 时必须用训练 config、dataset audit、checkpoint metadata 和
evaluation manifest 交叉核对，不能根据旧报告推测 shape、参数量或阶段结论。

## 15. 最终执行清单

### Freeze

- [ ] 新 experiment id 已通过 `rl.core.runs create` 分配；
- [ ] 其他 Session 已停止修改本 campaign 的 dataset/model/train 源码；
- [ ] 已确认继续使用 0003 Yushin Ito 人类专家 dataset，不续训旧 checkpoint、不混新数据；
- [ ] 0003 dataset、metadata、split、source、deck、cg、Git/diff 已 hash 并归档；
- [ ] GPU 独占且无其他训练进程竞争；
- [ ] `search_manifest.json` 已记录 12,600/11,700 秒预算。

### Train

- [ ] Phase A 三个 Baseline LR 完成；
- [ ] Phase B 的 S/W/D/L 完成；
- [ ] Phase C 至多一次，且只根据预先定义的 validation 证据进行；
- [ ] Phase D 仅在剩余预算允许时确认一个结构的 seed 17；
- [ ] 每个 trial 有 config、20 epoch metrics、summary、status、wall time 和参数量；
- [ ] 累计训练 wall time 不超过 12,600 秒；
- [ ] 自适应搜索没有消费 test 或 evaluation 结果。

### 每个 completed trial 的真实 Agent 闭环

- [ ] `best_validation.pt` SHA-256 已记录；
- [ ] 临时纯模型 package 已构建且 `fallback=null`；
- [ ] `evaluation validate` 通过；
- [ ] 18 opponents × 10 games 正式 evaluation 完成；
- [ ] evaluation manifest/summary/metrics/report 路径已写回 `trial_result.json`；
- [ ] INDEX 已新增或刷新该 trial 行。

### Evaluate and report

- [ ] 每个 completed trial 都有独立 18×10 正式 evaluation；
- [ ] 所有 trial 的 profile revision/catalog/hash 一致；
- [ ] INDEX 以 evaluation W-L-D/胜率/错误为重点，BC loss/validation exact 为辅助；
- [ ] INDEX 展示每个 trial 的超参差异和原始 report 链接；
- [ ] `bc_capacity_search_report.html` 包含完整离线和真实对局结果；
- [ ] `rl/model/DESIGN.html` 与 `rl/_runs/INDEX.html` 已同步；
- [ ] 没有替用户选择 Winner、promote 或 reject；
- [ ] 没有自动 Kaggle submission、commit 或 push。

## 16. 收口标准

在 210 分钟训练预算结束时，训练必须处于以下状态之一：

- **Complete**：必做矩阵完成，每个 completed trial 都已有 evaluation 和 INDEX 行；
- **Budget-complete with uncertainty**：必做矩阵完成但救援/seed 证据不足；完整展示不确定性，
  不替用户选择模型；
- **Invalid**：冻结项发生变化、核心 trial 不完整或结果不可比较。此时不得宣称找到容量上限。

无论属于哪一种，预算结束后都不再追加训练。执行者应立即补齐最后一个 completed trial 的
candidate/evaluation、evaluation-first INDEX、campaign HTML 和项目文档同步。最终状态是
“等待用户明早选择结构”，不是由执行 Session 自动指定下一阶段模型。
