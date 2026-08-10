# CUDA Search scratch-lane 使用说明

## 定位与边界

这组接口为 CUDA official POD runtime 增加了批量搜索所需的底层原语：从正在运行的主 rollout 中复制指定 lane 到隔离的 scratch arena，再对 scratch lane 使用不同隐藏信息粒子和未来 RNG 继续模拟。

它不是一个完整 MCTS 实现。本分支没有内置候选动作生成、UCT/PUCT、节点选择、价值回传、停止条件或最终动作选择；这些仍由上层 search coordinator 实现。不能仅凭 scratch fork 的存在宣称搜索提高了胜率。

主 rollout 与 scratch 必须使用同一 CUDA device 和同一 rule pack。正式强度结论仍应使用官方 CPU engine、固定对手、全新 seeds 和平衡先后手验证。

## 构建

先按 `engine_cuda/docs/usage_guide_zh.md` 生成私有 `official_rules.bin`，再构建 PyTorch 扩展：

```bash
cmake -S engine_cuda -B engine_cuda/build/torch_official -G Ninja \
  -DPTCG_CUDA_BUILD_TORCH=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build engine_cuda/build/torch_official --parallel
export PYTHONPATH="$PWD/engine_cuda/python:$PWD/engine_cuda/build/torch_official"
```

`official_rules.bin` 来自 competition-use-only 官方源码，不应提交到 Git、W&B 或模型包。

## API

### 1. 批量复制主 lane

```python
import torch
from pathlib import Path
from ptcg_cuda_engine.native import create_official_engine

rule_pack = Path("engine_cuda/generated/private/<id>/official_rules.bin").read_bytes()
main = create_official_engine(rule_pack, batch_size=512)
scratch = create_official_engine(rule_pack, batch_size=64)

# scratch 第 i 个 lane 来自 main 的 source_indices[i]。
source_indices = torch.tensor(
    [3, 17, 17, 81] * 16,
    dtype=torch.int64,
    device="cuda",
)
main.fork_lanes_to(scratch, source_indices)
```

`source_indices` 必须是与 engine 同设备的 contiguous CUDA `int32` 或 `int64` 一维 tensor，长度必须等于 scratch batch。允许重复索引，因此同一 root 可以复制成多个 future particles。

复制内容包括完整 `OfficialStatePod`、MT19937 状态、resident action/status 和 causal semantic history。内部 semantic-history 指针会重新绑定到 scratch arena；之后推进 scratch 不应修改 main。

### 2. Privileged hidden-order particles

```python
hidden_order_seeds = torch.arange(64, device="cuda", dtype=torch.int64) + 1000
future_rng_seeds = torch.arange(64, device="cuda", dtype=torch.int64) + 2000
scratch.redeterminize_hidden_order(hidden_order_seeds, future_rng_seeds)
```

这个接口只重新排列当前已经存在的双方 Deck/Prize 顺序，并替换未来 RNG。它不改变 Hand/Deck/Prize 的成员身份，因此应称为 hidden-order particle，而不是公共信息 belief sampling。若上层知道真实隐藏区成员，这是一种训练期 privileged search；不得把它描述为部署时可获得的信息。

### 3. Actor-valid clean-root belief particles

```python
exact_decks = torch.empty((64, 2, 60), dtype=torch.int32, device="cuda")
observer_seats = torch.zeros(64, dtype=torch.int64, device="cuda")
membership_seeds = torch.arange(64, dtype=torch.int64, device="cuda") + 3000
future_rng_seeds = torch.arange(64, dtype=torch.int64, device="cuda") + 4000

codes = scratch.redeterminize_public_belief_clean(
    exact_decks,
    observer_seats,
    membership_seeds,
    future_rng_seeds,
)
accepted = codes == 1
```

`exact_decks` 必须是 CUDA contiguous `int32[batch, 2, 60]`；另外三个参数必须是 CUDA contiguous `int64[batch]`。该接口只应在刚 fork 出来的 scratch arena 上调用。返回码含义：

- `1`：接受并完成重新确定化；
- `2`：observer/root 无效，例如不是该 observer 的正常决策状态；
- `3`：root 含当前实现不能安全重采样的可见隐藏身份或已知 Deck/Prize 信息；
- `4`：exact deck 与固定的公开卡牌不一致。

接受的 lane 会保留公开区域和固定卡牌，并在 focal Deck/Prize 与 opponent Hand/Deck/Prize 之间重采样剩余卡牌身份，再安装独立未来 RNG。拒绝码必须 fail closed，不能把拒绝 lane 当作有效 particle。

### 4. 在 scratch 上继续模拟

fork 或重新确定化后，scratch 使用与主 rollout 相同的 resident API：

```python
codec = scratch.encode_policy_v1()
# 上层 actor/search coordinator 生成每个 lane 的完整合法动作 token。
scratch.pack_actions(option_indices, action_counts)
scratch.apply_packed_actions()
scratch.advance_to_decision()
```

为公平比较 candidate plans，同一个公共 root 的各候选应使用完全相同的一组 particle IDs；随机种子、候选、结果和 provenance 也应保存。不要只保留赢的 rollout。

## 隔离 smoke

在有 CUDA、已构建 `_ptcg_cuda` 且已有 exact 60-card deck 的环境运行：

```bash
python engine_cuda/tools/run_official_scratch_fork_smoke.py \
  --rules engine_cuda/generated/private/<id>/official_rules.bin \
  --deck path/to/deck.csv \
  --output .tmp/cuda_search/scratch_fork_smoke.json
```

门禁固定拆成三组证据：

1. **主状态 RNG 未被污染**：单独保存并比较完整 `OfficialMt19937`（624 words、index 和 draw count），分别检查 fork 后及 scratch 推进一步后都 bitwise 不变；只比较 draw count 不够。
2. **主状态未变化**：分别比较完整 state bytes、status、resident action bytes 和 causal semantic history；semantic-history pointer 只在 scratch 副本中重绑定，不允许写回 source。
3. **无 learner 信息泄露**：把同一个公共 root 复制成两个 sibling，以不同 hidden-membership/future-RNG seeds 重新确定化；先屏蔽 RNG 字节并确认两个完整 privileged state 确实不同，避免做成空测试。只有两个 lane 都返回 code `1`，且 `encode_policy_v1` 与 `encode_semantic0031_v2_lanes` 在重采样前后及 sibling 之间都 bitwise 一致，才通过。

`state_bytes()`、`semantic_history_raw()` 和 scratch 的完整 engine state 是 privileged search internals，不是 learner 输入。这里的“无信息泄露”严格指已登记的 learner-visible encoder 接口不会随隐藏 particle 改变；如果上层把 raw state、particle seed、opponent exact deck 或 search outcome 拼进 actor forward，这个门禁不能保护它，实验应直接判无效。

## 与官方 Search 的区别

官方 CPU SDK 的 `SearchBegin` / `SearchStep` 位于 `engine/source/ptcgProgram 22/Export.cpp`、`Search.h`：`SearchBegin` 从一个 serialized state 建立 search state，并由调用方补齐 `myDeck`、`myPrize`、`enemyDeck`、`enemyPrize`、`enemyHand`、必要时 `enemyActive` 和 `manualCoin`；`SearchStep(searchId, selected)` 每次复制父 state、执行一个完整合法选择，然后返回新的 `searchId`。官方实现的内部池以 128 个 `State` 为一组分配。

| 维度 | 官方 CPU Search | 本分支 CUDA scratch search primitives |
| --- | --- | --- |
| 执行方式 | 单个 state、按 `searchId` 逐步派生 | 一次从主 batch 复制多个 lane，在 GPU 上批量推进 |
| 隐藏信息入口 | 调用方显式提供各隐藏区卡 ID | 可保留真实成员只改顺序，或在 clean root 按 exact deck 重采样 belief |
| 随机性 | state 自带 RNG；`manualCoin` 可由调用方控制 | 每个 particle 可安装独立 hidden-order/membership seed 与 future RNG seed |
| 状态历史 | 复制官方 `State` | 复制 POD、resident action/status 和 causal semantic history，并重绑定指针 |
| 隔离模型 | 新 `searchId` 持有新的 CPU state | 独立 scratch arena；kernel 不应写回 main rollout lane |
| 批量吞吐 | API 本身不是 GPU batch 接口 | 针对大批 root/candidate/particle 的 CUDA resident 执行 |
| 树策略 | 不提供 UCT/PUCT 或自动规划 | 同样不提供；只提供 fork、重新确定化和推进原语 |
| 权威性 | 官方规则语义来源 | 加速器复刻；必须持续做 CPU/CUDA parity 与官方评测 |

两者不是“官方有完整 MCTS、CUDA 重写 MCTS”的关系。官方 Search 也主要是可复制、可分支的规则执行 API；本分支改变的是执行位置、批处理方式、粒子构造和隔离存储合同。上层算法仍需明确候选宽度、搜索深度、rollout policy、价值估计、置信门槛和预算。

## 已知限制

- `redeterminize_public_belief_clean` 只接受语义历史足够完整、没有冲突隐藏知识的 clean decision root；它不是任意中间状态通用的 RIS-MCTS 重确定化。
- scratch fork 会复制 rule pack 到目标 arena，适合复用 scratch batch，不应为每个节点反复构造 engine。
- 当前接口不自动去重相同 state、不自动生成 diverse actions，也不提供 transposition table。
- GPU smoke 证明隔离合同，不证明 CPU/CUDA 全规则等价，更不证明搜索策略强度。
