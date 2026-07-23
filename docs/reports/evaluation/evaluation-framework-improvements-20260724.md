# Evaluation 框架近期改造与展示设计

- 日期：2026-07-24
- 状态：今晚可执行设计；性能数字已在本机真实官方引擎对局中验证
- 适用范围：仓库内 `evaluation/` CLI、批量调度、manifest 和 Markdown/HTML 报告
- 非目标：修改 `engine/source/`、改变候选策略、用 mock 结果替代真实对局

配套文档：今晚训练实现见
[`../rl/online-ppo-training-design-20260724.md`](../rl/online-ppo-training-design-20260724.md)，
未来训练吞吐方案见
[`../rl/rl-throughput-optimization-roadmap-20260724.md`](../rl/rl-throughput-optimization-roadmap-20260724.md)。

## 1. 结论

正式 Evaluation 继续保持“每局独立 engine 和策略进程”的真实性边界，同时把本机默认
运行方式从串行切换为经过验证的 CPU 并行调用：

```bash
python3 -m evaluation validate work/<candidate>

python3 -m evaluation run \
  --candidate work/<candidate> \
  --opponents all \
  --games 10 \
  --workers 8 \
  --worker-cpu-threads 1 \
  --metric-profile auto_iteration_v8_setup_relay \
  --no-visualize \
  --output rl/_runs/<experiment>/evaluation/<Vn_tag>
```

这条命令不修改官方 engine、候选模型或 opponent package。它只改变不同对局之间的调度，
并限制每个 worker 内部的 PyTorch/OpenMP/MKL/OpenBLAS 线程数。

需要精确区分：旧的串行入口不会在“不改变调用参数”的情况下自动变快；只有代码包含
`--workers` 与 `--worker-cpu-threads` 支持，并在运行时显式启用上述参数，才会获得本次
验证的加速。

## 2. 已验证基线

测试对象为 `work/yushin_ito_bc_capacity_v4`，协议为固定 18 个 opponent、每个 10 局，
共 180 局真实官方 engine 对战。本机为 8 个物理 CPU 核心。

| 路径 | 完成情况 | 总耗时 | 吞吐 | 相对串行 |
|---|---:|---:|---:|---:|
| 串行 | 180/180 | 234.41 秒 | 0.77 games/s | 1.00x |
| 4 workers，未限制内部线程 | 出现 worker error | 488.41 秒 | 不采用 | 0.48x |
| 4 workers × 1 CPU thread | 180/180 | 60.29 秒 | 2.99 games/s | 3.89x |
| 8 workers × 1 CPU thread | 180/180 | 40.23 秒 | 4.47 games/s | 5.83x |
| GPU 动态 batch 研究原型 | 180/180 | 24.01 秒 | 7.50 games/s | 9.76x |

当前正式推荐值是 `8 workers × 1 CPU thread`，而不是 GPU 原型。GPU 路径使用了候选专用
常驻推理服务和代理 package，目前没有正式 CLI、独立单元测试、通用 checkpoint 合同和
正式评测等价性验收，只能作为后续实现证据。

这些结果只证明吞吐差异，不证明策略强度差异。不同 run 的随机胜负不能用于判断并行
执行是否改变策略语义；语义等价性要靠输入、动作、状态隔离、错误率和长期 soak test
验证。

## 3. 近期代码改造范围

### 3.1 批量调度

`evaluation run` 应正式支持并记录：

- `--workers N`：最多同时执行的独立对局数；
- `--worker-cpu-threads N`：注入每个 worker 的 `OMP_NUM_THREADS`、
  `MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 和 `NUMEXPR_NUM_THREADS`；
- manifest 中的实际 workers、内部线程限制、总 wall time 和运行环境；
- 无论完成顺序如何，`games.jsonl`、指标聚合和报告仍按 catalog/game 固定顺序写入；
- 单局超时、异常、trace 缺失与进程退出必须独立归类，不能让一个 worker 破坏整个报告。

默认 CLI 不硬编码所有机器都使用 8 workers。硬件、候选模型和 opponent 计算量不同，
仓库文档应把 `8 × 1` 标记为本机已校准 preset；其他机器先在 `/tmp` 做小规模 benchmark。

### 3.2 性能事实采集

每次正式 run 的 manifest 和 summary 应增加或统一以下字段：

- `started_at`、`finished_at`、`wall_time_seconds`；
- `games_per_second`、总 engine selections、`selections_per_second`；
- `workers`、`worker_cpu_threads`；
- candidate/opponent/runtime hash；
- finished、unfinished、engine error、worker error；
- 单局耗时的 p50、p90、p95、最大值；
- 可选的主机 CPU 数、RAM、GPU 型号、PyTorch/CUDA 版本；
- 若启用 batch inference，则增加 inference requests、GPU calls、平均/p50/p95 batch size、
  encode time、queue time 和 GPU forward time。

硬件信息只用于解释吞吐，不参与策略晋级判断。

## 4. HTML/UI 展示改造

报告首页应先回答“结果是否可信、强度如何、运行是否健康”，再展示细粒度过程指标。

### 4.1 顶部摘要

首屏固定展示四组卡片：

1. **结果护栏**：完成局数、胜负平、胜率、error、unfinished；
2. **运行身份**：candidate、checkpoint/package hash、run id、metric profile revision；
3. **性能**：总耗时、games/s、selections/s、workers × threads；
4. **协议**：opponent 数、每 opponent 局数、先后手交换规则、官方 runtime hash。

error 或 unfinished 非零时，结果护栏应使用显眼的警告状态，并在胜率旁明确写出“本次
结果不能作为无错误正式晋级证据”。性能变快不能覆盖正确性失败。

### 4.2 Opponent 表格

Opponent 明细表增加：

- opponent、W-L-D、胜率、先手/后手分解；
- errors、unfinished、平均和 p95 单局耗时；
- selections 数量和对局长度；
- 只看失败、只看错误、只看慢局的筛选按钮；
- 按胜率、错误、p95 耗时排序；
- 对保留案例提供 trace/case 的直接链接。

所有排序和筛选在独立 HTML 内用本地 JavaScript 完成，不依赖外部服务，也不改变原始
JSON 产物顺序。

### 4.3 指标与案例

- 默认展示语义化指标组，原始 metric payload 折叠在“审计详情”中；
- 每个指标同时展示 numerator、denominator、value、方向和有效样本说明；
- 零机会显示“未定义/无有效样本”，不能显示成 0%；
- 失败案例按 engine error、合法性、开局、Post-KO、攻击质量和牌库风险分组；
- trace 未被保留时显示明确原因，不生成失效链接；
- 页面顶部提供 manifest、summary、metrics、games 和 cases 的下载/打开入口。

### 4.4 性能面板

性能面板只展示可测量事实：

- wall time 与吞吐；
- 单局耗时分布；
- worker 并发和内部线程限制；
- CPU/GPU inference 路径；
- 若存在动态 batch，展示实际 batch size 分布，而不是只展示配置上限。

报告必须明确标注“性能指标不构成策略强度证据”。

## 5. 今晚应完成的 Evaluation 工作

今晚的 Evaluation 范围只包括：

1. 保留并测试 CPU 并行调度与 worker 线程限制；
2. 在 `evaluation/HANDOFF.md` 中把 `8 × 1` 写为本机正式评测首选命令；
3. 在 manifest/report 中展示实际并行参数和 wall time；
4. 确保串行和并行都能生成相同 schema 的完整报告；
5. 使用 BC V4 做一次 180 局回归，要求 180/180 完成且 0 error；
6. 不在今晚把 GPU 推理原型集成到正式 Evaluation。

## 6. 后续 Evaluation 优化 backlog

按风险从低到高排列：

1. 补齐单局耗时分布和 selections/s，不改变执行语义；
2. HTML 的排序、筛选、性能卡片和慢局定位；
3. 自动做 `workers=4/8` 的短 benchmark，并给出本机建议值；
4. 把常驻 GPU policy server 做成通用 checkpoint 服务，增加 session reset 和 batch audit；
5. 对 CPU package 与 GPU server 逐 observation 重放，验证动作和 session history 等价；
6. 只有完成单元测试、长时间 soak、180 局正式回归和报告审计后，才允许 GPU 路径成为
   可选正式入口。

Evaluation 不采用训练专用的 native vector engine 或共享对手状态。训练可以追求更高
吞吐，正式评测继续保留更强的隔离和可审计性。

## 7. 验收标准

- 不修改 `engine/source/`；
- `python3 -m unittest -v tests.test_evaluation_assets` 通过；
- Evaluation 相关 CLI/batch/report tests 通过；
- `python3 -m evaluation validate work/yushin_ito_bc_capacity_v4` 通过；
- 8 × 1 正式 run 为 180/180 completed、0 error；
- manifest 记录实际并行度、内部线程限制和 wall time；
- HTML 首屏能同时看见结果护栏、运行身份、协议和性能；
- 报告不把吞吐、离线指标或过程指标写成策略提升结论。
