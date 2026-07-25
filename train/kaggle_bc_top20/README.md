# Kaggle 单专家 BC Worker

这套入口把 Top-20 campaign 中尚未完成的 18 个单专家 BC job 放到私有 Kaggle
Notebook 执行。它复用仓库现有 downloader、dataset builder/audit、训练器和 package
builder，不另造 feature、action、loss 或模型契约。

## 不变约束

- 每个 job 只绑定一个明确 `team_id + submission_id + exact deck hash`，不混专家。
- 当前 Kaggle Episode API 只能提供该 submission 可访问的最新至多 1,000 局，不等于玩家
  全职业生涯历史。
- 20 个 job 中 order 1、2 已在本地完成，输入生成器默认排除；云端只生成 order 3–20。
- 共享 `ptcg_features_universal`、`full_action_set_v1`、20 epochs 和冻结的 S 模型配置；
  每个专家仍从 seed 7 的随机初始化开始，拥有独立 optimizer、checkpoint 和版本目录。
- LR rescue 只看 train/validation；选定 checkpoint 后才读取 test。离线 gate 不参与
  package 强度宣称。
- Notebook 只做数据、训练和标准 package 结构校验，不在云端伪造 official simulator
  对局。正式强度证据仍由本地 `python3 -m evaluation run` 产生。

## 1. 生成冻结输入包

生成目录不到数 MiB：代码、20 个冻结 source manifest 中尚未完成的 18 个、精确 deck 与
共享配置。它不包含 replay、模型、凭据、官方卡表或 Competition Use Only `cg/` 二进制；
Notebook 只从挂载的官方 competition input 读取卡表，并从 `sample_submission` 物理复制
runtime。

首个 order 3 pilot 使用嵌入式 archive，避免 private Dataset 挂载路径和版本漂移：

```bash
python3 -m train.kaggle_bc_top20.prepare_input \
  --output /tmp/ptcg-bc-cloud-input-order03 \
  --owner tommycyd \
  --dataset-slug pokemon-tcg-bc-cloud-input \
  --orders 3
```

Dataset 必须保持 private；命令中不要加入 `--public`。当前本机 CLI 的 OAuth 会话需要先
重新登录时，执行 `kaggle auth login`。Notebook 下载 Episode 还需要一个 private
Notebook secret：优先使用 `KAGGLE_API_TOKEN`，也兼容旧的 `KAGGLE_USERNAME` 和
`KAGGLE_KEY`。任何 secret 都不能放入 Dataset 或 notebook cell。

## 2. 生成 CPU prepare 与 GPU train Kernel

模板位于 `notebooks/kaggle_bc_worker/`。渲染器会给每个 order 创建独立 Kernel slug，
避免不同玩家覆盖同一 Kernel 的 version/output。

大约 20 GiB 的公开日包在 T4 Kernel 中没有稳定挂载，因此生产流程必须拆成两段。CPU
prepare 读取日包、API fallback、构建并审计 JSONL；GPU train 只挂 prepare output，不再挂
日包。

```bash
python3 -m train.kaggle_bc_top20.render_kernel \
  --output /tmp/ptcg-bc-prepare-03 \
  --owner tommycyd \
  --input-archive /tmp/ptcg-bc-cloud-input-order03/ptcg_kaggle_bc_input.tar.gz \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22 \
  --job-order 3 \
  --mode prepare \
  --retention dataset

kaggle kernels push -p /tmp/ptcg-bc-prepare-03
```

`PREPARE_RESULT.json.status == dataset_ready` 后，把完成的 prepare Kernel 作为只读 source：

```bash
python3 -m train.kaggle_bc_top20.render_kernel \
  --output /tmp/ptcg-bc-train-03 \
  --owner tommycyd \
  --input-archive /tmp/ptcg-bc-cloud-input-order03/ptcg_kaggle_bc_input.tar.gz \
  --kernel-source tommycyd/<completed-prepare-kernel> \
  --job-order 3 \
  --mode train \
  --retention package \
  --accelerator NvidiaTeslaT4

kaggle kernels push -p /tmp/ptcg-bc-train-03 --accelerator NvidiaTeslaT4
```

`retention=dataset` 会保留处理后的 JSONL，便于以后不重新下载 replay 就重训；
`retention=package` 只保留 package、选中 checkpoint 和审计证据。原始 replay 无论哪种
模式都在运行结束前删除，以控制 Kernel output 大小。

跨阶段会重算 JSONL 与所有 sidecar SHA-256，并再次校验 source identity、dataset audit 和
单专家 data manifest。prepare output 不满足任一条件时，GPU 不会开始训练。

首个 pilot 完成前不要批量推送。若账号 UI 明确允许两个并行 GPU Kernel，可以为不同
order 各生成不同 slug 后并行推送；否则串行执行。并发不会扩大 Episode 的 1,000 局窗口，
还会按实际占用同时消耗 GPU quota。

### Yushin 十日日包同数据对比

要与 `ptcg-yushin-id-only-bc-v2-my-model` 保持训练 corpus 一致，使用独立的 order 1
对比工作包，并把 corpus mode 切为 `daily_team_winners`。该模式完全沿用参考 Notebook
的十个日期、规范化后的 `TeamNames == Yushin Ito`、winner-only、最长 `visualize` trace、
动作合法性过滤以及最大日期 validation；只把通过筛选的帧换成
`ptcg_features_universal + full_action_set_v1` 编码。它不会调用 Episode API，也不会加入
submission 最近 1,000 局中的额外对局。

```bash
python3 -m train.kaggle_bc_top20.prepare_input \
  --output /tmp/ptcg-yushin-daily-input \
  --owner tommycyd \
  --dataset-slug pokemon-tcg-yushin-daily-our-model-input \
  --daily-winner-team "Yushin Ito"

python3 -m train.kaggle_bc_top20.render_kernel \
  --output /tmp/ptcg-yushin-daily-our-model \
  --owner tommycyd \
  --input-archive /tmp/ptcg-yushin-daily-input/ptcg_kaggle_bc_input.tar.gz \
  --job-order 1 \
  --corpus-mode daily_team_winners \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-13 \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-14 \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-15 \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-16 \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-17 \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-18 \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-19 \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-20 \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-21 \
  --replay-dataset-source kaggle/pokemon-tcg-ai-battle-episodes-2026-07-22 \
  --retention dataset
```

渲染结果默认关闭 Internet。`dataset.jsonl.summary.json` 会记录命中的 Episode 数、决策数、
训练/验证分布、不同 deck 数、数据 SHA-256 和筛选条件；这些字段用于与参考 Notebook 的
`preflight.json` 对账。模型结构、超参数、loss 和 feature adapter 是实验变量，不应拿两边
不同的离线 accuracy 单独作为实战强度结论。

## 3. 拉回 package，而不是整份数据集

```bash
kaggle kernels output tommycyd/pokemon-tcg-bc-train-03 \
  -p /tmp/ptcg-bc-train-03-output \
  --file-pattern '.*(RESULT.json|candidate\.tar\.gz)$'
```

候选 tar 顶层直接是 `main.py`、`deck.csv`、`cg/` 和 `strategy/`，没有额外套目录。用本地
importer 核对 `RESULT.json` SHA-256、拒绝路径穿越/套目录、解压到全新目录，并立即跑
repository package validation：

```bash
python3 -m train.kaggle_bc_top20.import_candidate \
  --archive /tmp/ptcg-bc-train-03-output/<package_name>-candidate.tar.gz \
  --result /tmp/ptcg-bc-train-03-output/RESULT.json \
  --output work/<package_name> \
  --evaluation-cg-source evaluation/opponents/romanrozen_v9/cg
```

Kaggle 当前 competition input 与本地固定 catalog 可能是不同的官方 runtime revision。
importer 先验证原始 cloud tar hash，再只对本地工作副本物理替换 `cg/`，并同时记录 cloud 与
evaluation runtime hash；deck、model 与 strategy 不会改变。原始 tar 仍是云端原样证据。

然后执行 official-engine evaluation：

```bash
python3 -m evaluation run \
  --candidate work/<package_name> \
  --opponents all \
  --games 10 \
  --workers 8 \
  --worker-cpu-threads 1 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output rl_runs/evaluation/<000N-experiment>/V<n>_<tag>.html
```

只有 package validation 和 18×10 official-engine evaluation 都有完整证据后，才物理复制
到 `evaluation/opponents/<name>/` 并修改 catalog。Notebook 不自动把自身输出晋升为
opponent，也不自动决定最终模型。

## Pilot 停止条件

第一个 job 建议使用 order 3。满足以下条件后才扩到 3 个、再扩到剩余 15 个：

- API secret 可以无交互认证，冻结 Episode 数与 source manifest 一致；
- 下载过程中没有 401、429 重试风暴或单局反复超时；
- 峰值工作盘低于账号当次显示上限并保留至少 5 GiB；
- `dataset_audit.json.status == passed`；
- `RESULT.json.status == candidate_ready` 且 candidate archive hash 可复核；
- 拉回后的本地 `evaluation validate` 通过；
- 实测总 wall time 和 GPU quota 消耗足以在本周预算内完成剩余 job。

任何一项失败都停在该 job，修复模板/认证/配额问题后新建 Kernel version；不要把失败
output 当作候选，也不要重复 Kaggle competition submission。

## Order 3 pilot 实测

- CPU prepare：848 局完整通过；07-22 日包命中 257，API fallback 591；下载 699 秒，
  dataset build 181 秒，audit 52 秒，worker 总计 937 秒。
- JSONL：70,695 records，1,423,888,919 bytes，SHA-256
  `043bd9888411ac18584895e20baf01d3b909951a311d4de24d00e030e5500611`；单一来源
  `54809294:tw_shin`，audit 无 violation。
- GPU train/package：Tesla T4，V1 共 1,241 秒；candidate tar 15,117,550 bytes，
  SHA-256 `02d2c3e51287066b7f4d5700842674e6e74942723a009c421f2cf795d8eeb00f`。
- V1 validation exact 82.07%，但 multi-action exact 62.59% 未过 65% gate，故
  `offline_gate_passed=false`。这不阻止保留候选，但禁止从离线指标宣称晋级。
- 本地固定 18×10 官方评测：44 胜、136 负、0 error、0 unfinished，胜率 24.44%。流水线
  和运行时已验证可用；该候选强度偏弱，未自动加入 opponent catalog。
