# 0038 Update 230 Kaggle Package

## 结论

U125/U145 的 exporter 存在两个打包问题：

1. 错误继承了旧 0034 candidate 的 `deck.csv`。旧包 deck multiset hash 为
   `f0314f...`，与 0038 训练及 Frozen 使用的 007 exact deck
   `07bedf...` 不同一张卡（旧包多 1 张 `1071`、少 1 张 `1213`）。
2. 错误继承了旧 semantic runtime，遗漏 frozen prototype embedding cache。

U230 已修正：deck 直接取自 0038 权威 Frozen-007 资产，semantic runtime 直接取自
0038 项目本身；exporter 对 exact deck hash fail closed。

## 语义验收

- Checkpoint：update 230，SHA-256 `a70b41d8...f9f8ab8`。
- Frozen-0806：`1250-798 / 2048 = 61.0352%`，0 error。
- Live official-engine A/B：训练侧未合并 LoRA/Allocation/Meta 权重与 Kaggle 包内合并权重，
  72 个 candidate callback 全部返回相同 primitive action；包含 1 次 Phantom macro 和
  6 次内部 target callback，0 diff。
- Phantom exhaustive parity：`n=1..5` 共 330 allocations，最终权威状态与 primitive
  展开 0 failure（`experiments/0038_action_boundary_rl/OFFICIAL_PARITY_FINAL_V3.json`）。
- 官方 worker smoke：171 primitive selects，finished，0 error。
- 单元/合同测试：38/38 通过。

本机常规多进程 evaluation 入口仍会遇到已知的 WSL `libcg`/PyTorch 导入顺序导致的
step-0 worker crash；在同一 official runtime 中预先加载 PyTorch 后，以上 171-select
worker smoke 正常完成。该本机 launcher 问题不来自候选策略异常。

Value head 不进入 Kaggle 推理包；它只用于训练 PPO/GAE。Actor、Last Option Q/V LoRA
合并结果、Action Decoder、Opponent Meta conditioning 与 Allocation Head 均已部署。

## 推理速度

固定输入、单线程 CPU、20 次完整 actor forward：

| 包 | prototype cache | ms/forward | forward/s |
|---|---:|---:|---:|
| U145 旧包 | 无 | 628.88 | 1.59 |
| U230 新包 | 一次构建后复用 | 7.77 | 128.68 |

该专项 microbenchmark 为约 **80.9x**。这不是 CUDA/FP16/engine batching 的收益，而是
静态 prototype embeddings 从“每次 forward 重算”改为“每进程一次、跨局复用”。权重、
device 或 dtype 改变时缓存失效；训练态可训练 prototypes 绕过缓存。

## 产物

- Payload：`archive/submission/0038_dragapult_ex_rl_update230/`
- Archive：`archive/submission/dist/0038_dragapult_ex_rl_update230.tar.gz`
- Archive SHA-256：`a40fe101023f6e936f69587cb96c5c1cc7455b5e3ad075dfe3683eb98fdd7535`

旧 U125/U145 包不应继续提交或作为与 Frozen 对齐的部署证据。
