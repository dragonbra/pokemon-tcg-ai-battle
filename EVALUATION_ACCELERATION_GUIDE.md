# Evaluation 加速启用说明（2026-08-08）

本文用于交接本轮 feature compiler 与 resident inference 吞吐优化。它只描述工程吞吐，不构成策略强度证据；官方 Engine、actor schema、39 个输入 tensors、checkpoint、logits 和 action contract 均未改变。

## 最终推荐配置

当前准入的组合是：

- GPU model-static prototype embedding cache：0035-derived policy 在模型 `.to(device, dtype).eval()` 后自动启用，无需 CLI flag。
- Engine pool：多个独立 official-engine battle pointer 由 OS worker 并发维护。
- Worker-local feature compiler：compiler 跟随持有 Engine observation 的 worker。
- Compiler backend：必须使用 `policy_stateless`。
- Collate、H2D、FP16 model forward 和 decode 仍统一位于 resident GPU server。

不要把 `0035_incremental`、`--async-h2d` 或 `--resident-tensor-cache` 当作推荐配置；这些路径保持显式 opt-in，但本机实测没有通过吞吐准入。

## 直接运行吞吐 profile

推荐先运行一组相邻、等合同对照。输出必须放在仓库 `.tmp/evaluation/` 下。

中央 stateless 基线：

```bash
python3 -m evaluation.performance_profile \
  --mode policy \
  --games 128 \
  --workers 16 \
  --engine-pool-size 16 \
  --batch-size 64 \
  --batch-wait-ms 2 \
  --inference-dtype fp16 \
  --output .tmp/evaluation/feature_compiler/central_stateless.json
```

推荐的 worker-local stateless：

```bash
python3 -m evaluation.performance_profile \
  --mode policy \
  --games 128 \
  --workers 16 \
  --engine-pool-size 16 \
  --batch-size 64 \
  --batch-wait-ms 2 \
  --inference-dtype fp16 \
  --worker-local-compiler \
  --worker-compiler-backend policy_stateless \
  --output .tmp/evaluation/feature_compiler/worker_local_stateless.json
```

若要测试另一个自包含 policy package，可额外传入：

```text
--policy-root /absolute/path/to/package
```

该 package 必须是 persona-free canonical policy，并暴露 `POLICY` 以及可产生 raw canonical record 的 `strategy.online_runtime.OnlineCausalEncoder`。Worker-local 模式目前要求对战双方使用兼容且相同的 policy runtime；不满足合同时会 fail closed。

## 如何判断运行成功

检查输出 JSON：

- `outcome.completed_games == workload.games`
- `outcome.errors == 0`
- `outcome.unfinished == 0`
- `workload.worker_local_compiler == true`
- `workload.worker_compiler_backend == "policy_stateless"`
- `inference` 与 `worker` 中存在 compiler/IPC/collate/model 分段指标

比较性能时必须保持 games、workers、engine pool、batch、wait、dtype、checkpoint、exact-deck schedule 和机器负载一致。至少做三次相邻重复并比较中位数；不要用不同策略产生的胜负或不同长度 trajectory 证明调度语义一致。

本轮同机相邻 N16E16/B64/2ms 结果：

| 模式 | selections/s | wall | 完成 |
|---|---:|---:|---:|
| central stateless | 342.22 | 65.72 s | 128/128，0 error |
| worker-local stateless | 528.77 | 42.52 s | 128/128，0 error |

吞吐提升 54.5%，wall 降低 35.3%；server prepare-record 时间从 65.51 降到 0.20 ms/batch。Worker-local incremental 为 525.77 selections/s，比 stateless 低 0.57%，因此未选用。

## 普通 evaluation CLI 的边界

`python3 -m evaluation run` 当前暴露 `--engine-pool-size` 和 resident inference 参数，但尚未暴露 `--worker-local-compiler`。因此：

- 吞吐校准和本轮复现使用 `python3 -m evaluation.performance_profile`。
- 程序化调用可在 `BatchConfig` 中设置 `worker_local_compiler=True`、`worker_compiler_backend="policy_stateless"`，同时保证双方 resident inference、`engine_pool_size > 1`、`compiler_workers == 1`。
- 在正式通用 CLI 暴露该能力前，不要声称普通 `evaluation run` 已自动使用 worker-local compiler。

## RL 使用边界

RL rollout 当前不会自动继承 worker-local compiler。0034 full-semantic collector 仍在主进程串行执行 feature encode，并对每个 observation 先构造 singleton tensors；若要在 RL 中获得收益，需要在对应编号项目内物理移植以下拓扑，不能运行时 import 0035：

1. Engine worker 按 session 维护 stateless causal encoder。
2. worker 返回 canonical record，learner 进程对整批 record 只 collate 一次。
3. prototype encoder 冻结时，在 rollout model 上启用 GPU-resident prototype cache。
4. trajectory 使用 columnar tensor storage，避免 singleton collate、拆回 tensor dict 和 PPO 再次重组。

最近 0034 V6 的真实记录中，feature encode 约占 rollout wall 的 29%，model forward 约占 38%。据此，移植 worker-local stateless、prototype cache 和一次性 batch collate 后，合理目标是 rollout 约 1.3×–1.4×、完整 PPO update 约 1.2×–1.3×；在正式 official-engine canary 前不要把该估算写成实测结论。

## 验证与版本

本轮门禁：

- 0035 tests：115/115 passed。
- evaluation tests：295/295 passed。
- 0031 tests：77/77 passed；benchmark tests：4/4 passed。
- 旧/新 stateless compiler 在 35-step chronology 上 canonical record 与所有 tensors 完全一致。
- `engine/source/` 未修改。

相关提交：

- `3633218 perf: 缓存推理阶段的原型嵌入`
- `b23b23d perf: 下沉无状态特征编译并消除中央漏斗`
- `136c886 perf: 固化无状态特征编译基准与分层接口`

更详细的设计与负结果见 `experiments/0035_lifetime_aware_feature_compiler/DESIGN.md` 和 `evaluation/README.md`。
