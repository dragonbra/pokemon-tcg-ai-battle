# Kaggle Notebook 承载多专家 BC 的可行性调研

> **Historical design:** The `train/kaggle_bc_top20/` implementation referenced below was retired on 2026-07-26. The document remains as feasibility evidence, not a current execution guide.


日期：2026-07-23
范围：Top-20 单专家行为克隆的数据下载、构建、GPU 训练、标准 package 输出；不包含
Kaggle competition submission。

## 结论

方案可行，适合把剩余 18 个 job 从本地迁到 Kaggle，但需要修正两个预期：

1. “玩家历史所有对局”目前做不到。比赛 Episode API 对一个 submission 暴露的是最新
   至多 1,000 局可访问 public completed Episodes；同一玩家的不同 submission 也不能在
   没有 source conditioning 的情况下混成一个 BC policy。
2. “每人固定 30 GPU 小时”不是当前公开官方文档中的硬保证。Kaggle 提供免费 GPU，
   但具体剩余 quota、可用 GPU、单次上限和并发能力由账号 UI、供给和调度决定。campaign
   只能读取当时账号显示值后排程，不能把社区常见的 30 h/week 写成保证。

order 3 实测证明方案可行，但生产单位必须是“一位专家一组 CPU prepare + GPU train
Kernel/version”。T4 无法稳定挂载约 20 GiB 日包；CPU 先输出经哈希审计的 JSONL，GPU 再通过
`kernel_source` 训练和打包。训练结果只输出候选，不自动晋升。

## 官方能力核实

截至调研日，Kaggle 官方资料确认：

- Notebooks 是运行 competition/dataset 代码的云环境，并提供 GPU image；competition 和
  private Dataset 可以作为输入源。
- `kernel-metadata.json` 支持 `enable_gpu`、`enable_internet`、`machine_shape`、
  `dataset_sources`、`competition_sources` 和 private Kernel。
- 官方 CLI 支持 `kernels push` 后运行、`kernels status` 查询和 `kernels output` 按正则
  拉取产物，因此本地可以只下载 candidate tar 与 `RESULT.json`。
- 2026-02 版官方 CLI 文档列出 P100、T4、T4 Highmem、L4、A100、H100 等 accelerator
  ID，同时明确部分高端规格只对特定比赛参与者或管理员开放。普通账号不应预设拿到
  A100/H100。
- API 允许创建多个不同 Kernel，但公开文档没有承诺它们会同时获得 GPU。是否能并行取决
  于账号限制与调度；代码层面的“可提交多个”不等于算力层面的“保证并行”。

官方来源：

- <https://www.kaggle.com/docs/notebooks>
- <https://github.com/Kaggle/docker-python>
- <https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md>
- <https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md>
- <https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/code>

本次已使用账号 `tommycyd` 的 OAuth 会话实际推送 private Kernel；没有执行 Kaggle
competition submission。账号 UI 的剩余 quota 与可并行数仍未形成可审计快照，因此不能写成
固定承诺。

## 为什么值得迁移

本地已完成样本给出了足够清晰的量级：

| Expert | Episodes | replay | JSONL | 下载 | V1 训练 |
|---|---:|---:|---:|---:|---:|
| order 1 · LumenLiquidity | 242 | 1.00 GiB | 418 MiB | 263 s | 166 s |
| order 2 · THIRD PTCG Club | 982 | 4.15 GiB | 1.3 GiB | 625 s | 522 s |

模型只有 3,539,395 参数，实测 peak GPU memory 约 470 MiB。训练不是显存瓶颈；真正的
成本是逐局 replay 下载、代理/连接稳定性和本地工作盘。后续本地失败日志出现了 2,797 s
只完成 50/848、connection reset/proxy OAuth refresh 等问题。把请求和计算放到 Kaggle
云端有望规避本地 WSL/代理链路，同时释放本机，但 Episode 仍是逐局 API 请求，并不会因为
在 Kaggle 内部运行就变成 `/kaggle/input` 的本地挂载文件。加速幅度必须由 pilot 实测，
不能先验保证。

## 单 Kernel 资源估算

基于已经完成的 982 局样本，一位接近 1,000 Episodes 的专家约需要：

- 原始 replay 峰值 4–6 GiB；
- universal JSONL 约 1–2 GiB；
- 单 attempt checkpoint/log 约百 MiB；
- V1 约 3–10 分钟，最多两个 rescue 后训练约 10–30 分钟；
- 下载在健康链路约 5–15 分钟，但失败链路可能远高于此。

所以单 job 适合 Notebook；18 个原始 corpus 同时留盘则不适合。Worker 每次只处理一位，
数据集构建成功后在结束阶段删除 raw replay。`retention=dataset` 保留处理后的 JSONL，之后
若要重训不必重新拉 replay；本地只按文件模式下载 candidate。

实测已把流程拆成 CPU fetch/build Kernel 和 GPU train/package Kernel。跨 Kernel 合同会
重算 JSONL、sidecar 和 prepare result 的 SHA-256，并校验同一
`team_id + submission_id + deck_sha256`；GPU 阶段不再挂日包，也不需要 Episode API secret。

## Order 3 实测结果

冻结来源为 `tw_shin`、submission `54809294`、team `16387915`、848 Episodes、deck hash
`dd63244cb42c5002bb2c7e415e8224e3dc8440ee02743f0db22d9d44594a72cc`。

| 阶段 | 实测 |
|---|---:|
| 07-22 日包直接命中 | 257 Episodes |
| API fallback | 591 Episodes |
| replay 获取 | 699 s |
| dataset build / audit | 181 s / 52 s |
| CPU worker 总时长 | 937 s |
| dataset | 70,695 records / 1,423,888,919 bytes |
| GPU worker 总时长 | 1,241 s |
| peak GPU memory | 470,373,888 bytes |
| candidate tar | 15,117,550 bytes |

Dataset SHA-256 为
`043bd9888411ac18584895e20baf01d3b909951a311d4de24d00e030e5500611`，audit 为 passed、
848 unique Episodes、单一 `54809294:tw_shin` source、零 violation。candidate tar SHA-256 为
`02d2c3e51287066b7f4d5700842674e6e74942723a009c421f2cf795d8eeb00f`。

训练只需要 V1；validation exact-action 82.07%，但 multi-action exact 62.59% 未过 65%
gate，所以 `offline_gate_passed=false`。本地用固定 catalog runtime 运行 18×10 官方评测，
180/180 完成、0 error、0 unfinished，44 胜 136 负，胜率 24.44%。这验证了流水线与 package
运行时，不支持把该候选自动晋升为 opponent。

另一个实测边界是 runtime revision：Kaggle 当前 competition `cg/` tree hash 为
`d6ed81f7...`，本地固定 catalog 为 `8a25d7b3...`。importer 必须先验证原始 tar，再只对本地
工作副本物理归一化 `cg/` 并记录两边 hash；不得关闭 evaluation 的 cg compatibility 护栏。

## 并行策略

并行分三层：

- 单 job 下载：仓库 downloader 硬限制最多 2 workers、每次请求至少间隔 1 秒，以降低
  429 风险；不提高到更大并发。
- 单 job 训练：一个模型使用一个 CUDA device，串行 V1/V2/V3；不做多卡/DDP。Kernel
  环境枚举到两张 Tesla T4 不代表训练器进行了双卡并行。
- 多 job：每个 order 使用独立 Kernel slug。只有账号 UI 明确允许时才同时运行两个；否则
  用 `push -> status -> output` 串行。即使允许并行，建议上限 2，先观察 API 429 和 quota
  消耗。

不能把 18 位专家放入同一个模型批量训练；那会破坏独立 optimizer/checkpoint/source
审计，也会让一个 12 小时级别或存储故障吞掉整批进度。

## 长期执行计划

### P0：当前已落地

- private input Dataset 生成器：默认排除已完成 order 1、2，且不重新上传官方卡表或
  Competition Use Only `cg/`；Notebook 从官方 competition input 复制两者；
- 参数化 Kernel renderer：每个 job 独立 slug；
- 两阶段 Notebook：CPU secret preflight、Dataset-first 下载与 dataset audit；GPU preflight、
  V1 + 最多两个 rescue、validation-only selection、frozen test、标准 package、结构 validation；
- 输出 `RESULT.json`、candidate tar、选中 checkpoint、数据/训练审计；
- 原始 replay 清理与输入/输出 SHA-256。
- 本地安全 importer：先校验 archive hash 与顶层 package 契约，再解压到新目录；如 cloud 与
  catalog runtime revision 不同，只归一化本地 `cg/`，最后调用 repository validation。

原实现位于已退役的 `train/kaggle_bc_top20/`；本报告仅保留当时的可行性结论，不再提供可执行入口。

### P1：一个 pilot（已完成）

order 3 已完成 CPU prepare、GPU train/package、local import/validation 和 18×10 官方评测。
配额前后值没有可靠 UI 快照，仍不得从运行时长反推剩余 quota。

### P2：三位小批次

选择 orders 3–5，默认串行。若账号明确支持并发，最多同时两个不同 Kernel。拉回三份
candidate，在本地各跑独立 18×10 official evaluation，检查 Kaggle 与本地 package hash
一致、没有 CUDA/CPU inference 或 cg runtime 差异。

### P3：剩余 15 位

按 roster order 继续。每个 job 保持独立 output/version；失败 job 新建版本恢复，不覆盖成功
output。GPU quota 不足时跨周继续，不切回本地训练，也不为了赶进度混专家或降低审计标准。

### P4：本地吸收

只拉 `RESULT.json` 和 candidate tar，先放 `evaluation/arena/candidates/<name>/`；完成 package validate
和 official evaluation 后，先物理复制到 `evaluation/arena/candidates/<name>/`；再由用户
根据证据决定是否迁入 `evaluation/arena/opponents/<name>/` 并添加 catalog。本机 8 核执行
BC candidate 的 18×10 时使用已校准的
`--workers 8 --worker-cpu-threads 1`，其他硬件先 benchmark。不得从 offline accuracy 自动
选 Winner，也不得自动发 Kaggle submission。
