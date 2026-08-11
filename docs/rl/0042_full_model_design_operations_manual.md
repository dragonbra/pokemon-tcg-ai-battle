# 0042 Full Model Design 使用与运维手册

**面向读者：**刚 clone 仓库、对 0042 没有上下文的开发 Agent / 运维 Agent  
**项目 ID：**`0042_full_model_design`  
**当前正式版本名：**`V1_ppo_protocol_v2_baseline`  
**当前状态（2026-08-11）：**PPO Protocol V2 preflight 已通过；16-update non-candidate smoke 尚在进行；正式训练和正式评测尚未启动  
**本手册性质：**操作说明，不取代任何强制合同

## 0. 先读结论

0042 是一个以 exact Dragapult deck 为 focal、以完整且不可变的 `Policy-0809` 为全部对手权重、使用 55 套 exact 60-card deck 构成训练环境的 FP32 PPO 项目。

最短的正确执行顺序是：

```text
clone / 同步仓库
  -> 阅读强制协议
  -> 检查 Python、CUDA、W&B 和不可变权重
  -> 准备并验证官方 CPU runtime 与 CUDA Engine 本地产物
  -> 跑 focused tests
  -> 跑 identity / architecture / mixed-deck / Protocol V2 preflight
  -> 跑隔离的 non-candidate smoke
  -> 人工确认后，以全新版本目录启动 formal PPO
  -> update 0、10、20... 自动执行 Frozen-0809 CUDA-2048
  -> 人工审查证据，再决定 PROMOTE / HOLD / REJECT
```

不得从“训练命令能启动”推断“项目语义正确”。0042 的正式启动条件是 policy identity、per-lane exact-deck routing、CPU/CUDA parity、candidate deployment identity 和版本存储合同同时通过。

## 1. 权威文件与阅读顺序

Agent 在执行任何 0042 RL、CUDA routing、opponent pool、Frozen evaluation、candidate deployment 或性能优化前，必须完整阅读：

1. `AGENTS.md`：仓库级工作约定；
2. `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`：全仓唯一 canonical policy identity / Frozen / Promote 协议；
3. `0042_POLICY_0809_RL_FROZEN_CONTRACT.md`：0042 的 Policy-0809-only 项目合同；
4. `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`：涉及规则、observation、action、reward 或模型语义时的规则证据；
5. `experiments/0042_full_model_design/DESIGN.md`：0042 当前模型和训练语义的权威设计；
6. `experiments/0042_full_model_design/DECISIONS.md`：已作出的项目决策；
7. `experiments/0042_full_model_design/manifest.json`：当前状态、固定 hash 和 preflight 证据索引。

若本手册与上述合同冲突，以更严格的强制合同为准。本手册中的状态会随项目推进过期；模型、schema、训练阶段发生变化时，必须同步更新 `DESIGN.md` 和 `DESIGN.html`，不能只改本手册。

## 2. 项目心智模型

### 2.1 三个不能混淆的对象

```text
focal candidate
  0042 当前 update 的完整模型；被 PPO 更新

opponent policy
  完整、独立、immutable Policy-0809；不被 PPO 更新

opponent exact deck
  每条 lane 实际使用的 60 张牌；决定合法且 actor-visible 的 deck static fields
```

`opponent_policy_id` 决定模型权重，`exact_deck_sha256` 决定这条 lane 的卡组静态语义。所有 opponent lane 的权重相同，不表示它们可以共用第一条 lane 的 deck features。

### 2.2 模型数据流

```text
public causal state
  -> frozen PrototypeEncoder / StateEncoder
  -> frozen OptionEncoder
  -> trainable Value latent decoder
       q0 -> Value Adapter -> Value
       q1 -> frozen MetaHead -> 15-class logits

raw decoder hidden
  + detached(q1, MetaProb, Value)
  + public seat
  + registered own-deck archetype embedding
  -> trainable Policy Strategy Adapter
  -> Query / STOP logits
```

PrototypeEncoder、StateEncoder 和 OptionEncoder 是冻结的语义空间。ActionDecoder、Value latent blocks、Value Adapter、Policy Strategy Adapter、allocation head 和 Prize auxiliary 按设计进入训练。MetaHead 参数冻结，但 Meta CE 可以通过它更新 q1/Value latent trunk。

`q1`、Meta probability 和 Value 在送入 Policy Strategy Adapter 前 detach，因此 policy loss 不会穿过 context 回传到 Value 侧。opponent hidden hand/deck/Prize、opponent exact-deck ID 和 Meta target 都不是 actor-visible 输入。

### 2.3 CPU Engine 与 CUDA Engine 的职责

`engine/source/` 是未修改的官方 C++ engine 源码，也是规则语义 oracle。任何开发、构建、测试或文档工作都不得修改它。派生的 seeded CPU runtime 由 `python3 -m evaluation.runtime.seeded` 构建到 `engine/build/seeded_official/`。

`engine_cuda/` 是 GPU-resident 的高吞吐执行实现。它的用途是让 256 个 heterogeneous lanes 常驻 GPU，并批量执行官方规则状态和 Policy-0809 推理。它不是新的规则权威；当 CPU/CUDA 不一致时，先检查 policy identity 和 deck routing，然后以官方 CPU engine 为 oracle 定位首次 divergence。

正确的 CUDA opponent 路径是：

```text
256 lanes
  -> 每条 lane 绑定 Policy-0809
  -> 每条 lane gather 自己的 exact-deck static fields
  -> 一份 immutable Policy-0809 weights 做 batched forward
  -> action scatter 回原 lane
```

允许共享的是已经审计为同一 `Policy-0809` 的 immutable opponent weights 和 batched compute。禁止 focal/opponent 共享 mutable module、Parameter、tensor storage、prototype cache 或会被 focal update 改写的 resident state。

## 3. 从 clone 开始准备环境

所有命令默认从仓库根目录执行。

```bash
git clone https://github.com/dragonbra/pokemon-tcg-ai-battle.git
cd pokemon-tcg-ai-battle
git status --short
```

新 clone 应先保持 clean。不要在准备环境时改写 `engine/source/`，不要把本地 `.tmp/`、checkpoint、W&B staging 或 private rule pack 加入 Git。

### 3.1 Python 环境

项目要求 Python 3.11+。安装 RL 与 W&B 依赖：

```bash
python3 --version
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[rl,tracking]'
python3 -c 'import numpy, torch, wandb; print(torch.__version__, torch.cuda.is_available())'
```

GPU 机器必须安装与 driver/CUDA toolchain 兼容的 PyTorch wheel。不要只看 `torch.cuda.is_available()`；同时检查实际 GPU、显存、driver、NVCC 和 CMake：

```bash
nvidia-smi
nvcc --version
cmake --version
ninja --version
```

0042 当前默认 `cuda:0`、256 CUDA lanes、256-game rollout batch 和 2,048-decision logical PPO minibatch。当前 16 GB WSL 主机用 1,024-row physical graphs 累积出每个 logical step；不同显卡不能照抄另一台机器的编译 architecture 或吞吐结论。先做本机 smoke 和 peak-memory 检查。

### 3.2 W&B

正式训练默认写入 private project `dragon_bra/pokemon-tcg-policy-learning`：

```bash
python3 -m wandb login
python3 -c 'import wandb; print(bool(wandb.Api().api_key))'
```

正式 run 使用 `--wandb-mode online`。离线 smoke 使用本地 JSON/JSONL，不得伪装为正式 online run。训练指标的 canonical 顺序是：

```text
training_metrics.jsonl flush
  -> TensorBoard
  -> W&B mirror
```

W&B 失败不能回滚本地指标或 checkpoint，但必须留在 `status.json` 中。

## 4. 不可变模型与本地运行资产

项目环境默认提供模型权重；Agent 不需要重新训练 0031/0036。仍必须在启动前核对文件和 SHA-256。缺失或 hash 不一致时应停止并重新同步权重，不能下载一个“看起来相同”的 checkpoint 或修改 registry 绕过检查。

| 资产 | 路径 | 预期 SHA-256 |
|---|---|---|
| Policy-0809 actor | `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt` | `926321955b6f3144b62e65899b5041ca3a47b17f305202ba9dffc0c92aaa7c7f` |
| paired V9 Value | `archive/pretrained/0031_friend_0809_gsb_v5_value_v9/value_head.pt` | `f8cb92f45f625e6519b6f103deb6d4ef68323fca93037ebd4c25db9da122a486` |
| pre-PPO allocation sidecar | `rl_runs/0038_action_boundary_rl/versions/V3_update0_chance_boundary_fallback/checkpoint/update-000000.pt` | `d572c8673fcf16c39fb17d8d261bec6578885a69c227e579bc4e6f616bb22820` |
| FP16 portable base package | `archive/submission/0031_zero_shot_0809_007_dragapult_ex_fp16_storage_fp32_runtime/strategy/model.bin` | `3f6683b0d916c72a31f3f487edb597c0cec4aedf529a236a4effacb177de89db` |

核验：

```bash
sha256sum \
  archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt \
  archive/pretrained/0031_friend_0809_gsb_v5_value_v9/value_head.pt \
  rl_runs/0038_action_boundary_rl/versions/V3_update0_chance_boundary_fallback/checkpoint/update-000000.pt \
  archive/submission/0031_zero_shot_0809_007_dragapult_ex_fp16_storage_fp32_runtime/strategy/model.bin

python3 archive/pretrained/0031_friend_0809_gsb_v5_value_v9/verify_archive.py
```

allocation sidecar 只提供 update-0 的 `allocation_head.*`。0042 会重新从 immutable actor/value 建模，严格只覆盖 allocation head；它不会加载 0038 的 PPO decoder、optimizer、rollout 或 RNG 状态。

### 4.1 Policy registry 自检

`train/0042_full_model_design/policy_registry.json` 当前只注册完整 `Policy-0809`。其 effective policy hash 必须是：

```text
0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96
```

执行只读 identity audit：

```bash
python3 - <<'PY'
from train.0042_full_model_design.policy_identity import resolve_policy_identity

audit = resolve_policy_identity("Policy-0809", purpose="operator_preflight")
print(audit.to_manifest())
assert audit.status == "PASS"
assert audit.effective_policy_sha256 == "0d0091140d72e78f1070c549b8367583a9d4f5537d0cb67decab40ac3bb9da96"
PY
```

该 resolver 会核对 checkpoint file hash、完整 component inventory 和 component content hash。requested/materialized identity 不一致会 `FATAL`；不得降级成 warning。

## 5. 准备官方 CPU runtime

seeded runtime 会从只读官方源码派生，不需要改动 source：

```bash
python3 -m evaluation.runtime.seeded
find engine/build/seeded_official -maxdepth 3 -type f -print
```

预期至少得到 versioned `libcg.so` 和 `manifest.json`。manifest 记录 official source、adapter 和 compiler identity。0042 的 CPU collector 和 Frozen schedule 通过统一入口取得该 runtime，不应静默回退到任意 candidate 内的未 seeded library。

若构建失败，先检查 C++ compiler 和官方源码是否完整；不要编辑 `engine/source/` 来“修到能编译”。

## 6. 准备 CUDA Engine

### 6.1 0042 当前固定路径

0042 runner 当前读取：

```text
.tmp/cuda_0032_rules/official_rules.bin
.tmp/engine_cuda_benchmark/build_sm120_staged/_ptcg_cuda.so
```

这两个是 machine-local 产物，Git ignore。当前主机的已验证实例 SHA-256 是：

```text
official_rules.bin  e8bb8537ccadc5d77b75c4198ffb1467d9b73d119e1e8c6325a9e5cd503a80e6
_ptcg_cuda.so       d2410f8f9a8b6171355deba0feda82b9ab2166845e1b516b3de69ffe20dc188d
```

这些 file hash 是当前机器构建的 provenance，不是跨 toolchain 的通用等价证明。新机器自行编译的 `.so` 可以有不同 file hash，但必须通过同一 ABI、rule-pack、CPU/CUDA parity 和 0042 regression gates。

### 6.2 official rule pack

`official_rules.bin` 从 competition-use-only 官方源码派生，禁止提交到 Git、W&B 或公开 package。完整提取/验证流程见 `engine_cuda/docs/usage_guide_zh.md`。

如项目环境已提供已验证 rule pack，将它放到 0042 固定路径并保留来源 manifest。若需重建，使用 `engine_cuda/tools/extract_official_rules.py` 与 `engine_cuda/tools/compile_official_rule_pack.py`；提取过程需要官方 seeded oracle 和 card data。不要用 `engine_cuda/rules/smoke_rules.json` 代替正式 official rule pack。

核验当前固定路径：

```bash
test -s .tmp/cuda_0032_rules/official_rules.bin
sha256sum .tmp/cuda_0032_rules/official_rules.bin
```

### 6.3 编译 PyTorch CUDA extension

先查询 GPU compute capability，再把 `CMAKE_CUDA_ARCHITECTURES` 设置为对应整数。例如当前 RTX 5080 使用 `120`；RTX 3060 使用 `86`。不要在未知硬件上照抄 `120`。

```bash
python3 -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
python3 -m pip install cmake ninja

TORCH_CMAKE_DIR="$(python3 -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")')"

cmake -S engine_cuda \
  -B .tmp/engine_cuda_benchmark/build_sm120_staged \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DPTCG_CUDA_BUILD_TORCH=ON \
  -DTorch_DIR="$TORCH_CMAKE_DIR"

cmake --build .tmp/engine_cuda_benchmark/build_sm120_staged \
  --target _ptcg_cuda --parallel
```

在非-SM120 机器上，build directory 也可以换成本机 architecture 专用目录，但 0042 runner 当前硬编码 `build_sm120_staged`。正式运行前要么把通过验证的本机构建 materialize 到该固定路径，要么通过一个经过测试和设计记录的项目改动修改路径；禁止用 symlink/错误二进制冒充已验证产物。

验证 import 和基本 ABI：

```bash
test -s .tmp/engine_cuda_benchmark/build_sm120_staged/_ptcg_cuda.so

PYTHONPATH=.tmp/engine_cuda_benchmark/build_sm120_staged:engine_cuda/python \
python3 - <<'PY'
import _ptcg_cuda
print(_ptcg_cuda.__file__)
print(_ptcg_cuda.OFFICIAL_STATE_ABI_VERSION)
PY
```

CUDA extension 必须由运行训练的同一 Python/PyTorch/CUDA ABI 环境加载。出现 undefined symbol 时，优先检查 PyTorch、CUDA toolkit、compiler ABI 和旧 `.so`；不要改 policy code 绕开。

### 6.4 CUDA Engine 的验收边界

基础 smoke：

```bash
PYTHONPATH=.tmp/engine_cuda_benchmark/build_sm120_staged:engine_cuda/python \
python3 engine_cuda/tools/smoke_torch_extension.py \
  --extension-dir .tmp/engine_cuda_benchmark/build_sm120_staged \
  --batch 256 --steps 10 --policies 1
```

这个 smoke 只证明 extension 可以运行，不证明 0042 的 mixed-deck semantics。正式训练前还必须通过第 8 节的 per-lane routing 和 parity tests。

## 7. Gate C 本地 attestation 资产

0042 formal update-0 会在第一次 Frozen-0809 CUDA-2048 后执行 training checkpoint 与导出 package 的 283-decision fixed-snapshot parity。当前 runner 读取：

```text
.tmp/evaluation/0038_semantic_parity_audit/gate_c_final/fixed_283_primitive_trace.jsonl
.tmp/evaluation/0038_semantic_parity_audit/gate_c/semantic_trace_manifest.json
```

项目环境应默认同步这套本地 attestation bundle。trace 的预期 SHA-256 是：

```text
18c1684a3dc8158494fd9820278a85e60e351b2b274b520bb9fb413b1fa056ca
```

仓库还保存了不可变的 gzip regression fixture：

```text
train/0042_full_model_design/tests/fixtures/semantic_parity_v1/fixed_283_primitive_trace.jsonl.gz
```

它可用于核对/恢复 trace 内容，但 formal runner 还需要与本机 official rule pack 和 deck case 对应的 `semantic_trace_manifest.json`。不要捏造 manifest 或删除 update-0 package parity gate。

```bash
test -s .tmp/evaluation/0038_semantic_parity_audit/gate_c_final/fixed_283_primitive_trace.jsonl
test -s .tmp/evaluation/0038_semantic_parity_audit/gate_c/semantic_trace_manifest.json
sha256sum .tmp/evaluation/0038_semantic_parity_audit/gate_c_final/fixed_283_primitive_trace.jsonl
```

## 8. 启动前测试与 preflight

### 8.1 快速静态检查

```bash
python3 -m compileall -q train/0042_full_model_design

python3 -m unittest -v \
  train.0042_full_model_design.tests.test_project_identity \
  train.0042_full_model_design.tests.test_policy_identity \
  train.0042_full_model_design.tests.test_policy0809_frozen_contract \
  train.0042_full_model_design.tests.test_candidate_deployment \
  train.0042_full_model_design.tests.test_ppo_protocol_v2 \
  train.0042_full_model_design.tests.test_smoke_ppo_diagnostics
```

全量 0042 tests：

```bash
python3 -m unittest discover \
  -s train/0042_full_model_design/tests \
  -p 'test_*.py' -v
```

测试、mock 和 static checks 只是合同证据，不是策略强度证据。

### 8.2 architecture / gradient audit

```bash
python3 -m train.0042_full_model_design.diagnostics.strategy_adapter_v2_audit \
  --output .tmp/strategy_adapter_v2_audit/operator-runtime-audit.json
```

它应确认：zero gates exact、frozen/trainable boundary 正确、Policy/Value/Meta gradient 路径正确、own/opponent semantic types 分离、hidden-zone counterfactual 不改变 actor-visible forward。

### 8.3 candidate deployment gate

使用 update-0 model-only checkpoint 时，候选必须经过：

```text
完整 effective candidate merge
  -> FP16 storage artifact
  -> strict-load 为 FP32 runtime tensors
  -> deployment-effective hash audit PASS
```

raw FP32 checkpoint 只可用于诊断/PPO，不能作为 Frozen/Kaggle strength evidence。

### 8.4 mixed-deck CUDA parity

用实际 update-0 checkpoint 运行 256 heterogeneous lanes 与 grouped reference：

```bash
python3 -m train.0042_full_model_design.diagnostics.mixed_deck_cuda_parity \
  --checkpoint PATH_TO_UPDATE_000000_PT \
  --output .tmp/evaluation/0042_mixed_deck_cuda_parity/operator-u0.json
```

输出必须 `status=PASS`、`unique_exact_decks=55`、`first_divergence=null`，并且 mixed/grouped 两条路径的 lane routing audit failure 都为 0。

### 8.5 Protocol V2 real preflight

这一步会真实执行一个完整 256-game frequency unit，不是轻量单元测试：

```bash
python3 -m train.0042_full_model_design.diagnostics.protocol_v2_preflight \
  --output .tmp/0042_protocol_v2_preflight/run-operator-001
```

`--output` 必须是 `.tmp/0042_protocol_v2_preflight/` 下不存在的新目录。PASS 至少表示：

- 256/256 games 有效；
- 55 套 exact decks 按固定频次完整出现；
- bulk rollout features 留在 CUDA，D2H 为 0；
- per-lane exact-deck routing PASS；
- 三个 PPO data epochs 每个 decision 恰好使用一次，保留最后 short minibatch；
- old logprob、old Value、normalized advantage 和 return targets 在 epoch 间不刷新。

### 8.6 isolated non-candidate smoke

先用 1–2 updates 做本机验收，再按当前项目决定运行 16-update smoke：

```bash
python3 -m train.0042_full_model_design.diagnostics.smoke_ppo \
  --output .tmp/0042_smoke_ppo/run-operator-001 \
  --updates 2
```

完整 smoke：

```bash
python3 -m train.0042_full_model_design.diagnostics.smoke_ppo \
  --output .tmp/0042_smoke_ppo/run-operator-16u \
  --updates 16
```

smoke artifact 明确标记 `NON-CANDIDATE_SMOKE_ONLY`，不能进入 Promote、正式 Frozen 或 Kaggle package 证据。它必须不创建/污染 `rl_runs/0042_full_model_design/` 正式版本，不修改 policy registry，不创建 W&B formal run。

## 9. 我们如何冻结评测策略

“冻结”在 0042 有三层，不能只理解为 `requires_grad=False`。

### 9.1 冻结 opponent identity

唯一 opponent policy ID 是 `Policy-0809`。resolver 从 registry 读取 immutable manifest，核对完整模型的：

```text
prototype encoder
state encoder
option input encoder
option transformer layer 0
option transformer layer 1
final norm
LoRA/delta inventory
action decoder
schema / checkpoint metadata
```

所有 component hash 组成 effective identity。只冻结 decoder、只复用 focal trunk 或 `0809 trunk + other head` 都不是 `Policy-0809`。

训练与 Frozen evaluation 各自 materialize 独立 opponent instance；focal optimizer step、`train()`、checkpoint load、rollback、CUDA graph 或 cache 都不能改变 opponent。

### 9.2 冻结 candidate deployment identity

每个需证明 Kaggle strength 的 checkpoint 先形成完整 effective candidate，合并 decoder、Value 条件路径、allocation head、adapters 和所有部署 delta。随后：

```text
source FP32 checkpoint
  -> complete effective candidate
  -> FP16 storage
  -> exact FP16 artifact strict-load into FP32 runtime
  -> greedy official-engine evaluation
```

FP16 转回 FP32 不会恢复被舍弃的精度。因此 source FP32 hash、portable FP16 file hash 和 deployment-effective content hash 是三个不同的 provenance/identity 字段。deployment-effective hash 只覆盖部署 tensor 内容和 runtime-semantic schema；version、update、source checkpoint/file hash 等 provenance 单独记录，不能让完全相同的 effective policy 被重新命名。正式 package 必须重算并匹配 qualifying Frozen report 的 deployment-effective hash。

### 9.3 冻结 schedule

0042 Frozen contract ID 是：

```text
0042_frozen_0809_seeded_agent_first_player_v1
```

固定项：

- master seed：`341512806`；
- base schedule：55 exact decks、256 slots；
- base schedule SHA-256：`74ee1527686c8b2344b15a1fd2c49a7dc2386da5c8164fb92fbd969cba8900ec`；
- CPU-256：replica 0；
- CUDA-2048：replicas 0..7；
- seeded toss 只固定 toss winner；winner Agent 必须真实处理 official context 41 并选择先/后手；
- 每局最多 50 个完整回合；
- 同一 physical actor 在同一 official turn 第 20 次提交同一完整选择时判负；
- 正式结果必须 0 error、0 unfinished、0 semantic fallback。

seed derivation 包含 focal deployment identity、opponent effective identity、exact-deck slot、replica 和 namespace。candidate identity 改变时 schedule hash 会相应改变；不能拿另一个 candidate 的 seeds 冒充同一 identity-bound evidence。

## 10. 对手如何分配权重与频次

### 10.1 模型权重分配

0042 不为 55 套 deck 训练 55 份 opponent 权重。所有 opponent lanes 都请求同一个：

```text
policy_id = Policy-0809
effective_policy_sha256 = 0d009114...9da96
```

runtime 可以让一份 immutable Policy-0809 weights resident 在 GPU 上并批量推理。每条 lane 仍需独立绑定：

```text
(policy_id, effective_policy_sha256, exact_deck_sha256,
 observation_schema_version, feature_preprocessing_version,
 device, runtime_dtype)
```

这个 tuple 是 deck-dependent static cache 的最低 key 合同。只用 deck number、display name、array position 或第一条 lane 作为 key 都是错误。

### 10.2 环境采样频次

`train/0042_full_model_design/league/frozen_catalog.json` 固定：

- pool ID：`0042_policy_0809_neutral_55_v1`；
- 55 套 unique exact decks；
- 每个 frequency unit 恰好 256 games；
- Top-100 分配 240 slots，使用 largest remainder；
- potential rank 101–500 固定选择 16 slots；
- 每个 entry 的 `games` 是它在每个 256-game unit 中的次数。

权威频次不要复制到 config 或手工表格；从 catalog 读取：

```bash
python3 - <<'PY'
import json

catalog = json.load(open("train/0042_full_model_design/league/frozen_catalog.json"))
assert len(catalog["entries"]) == 55
assert sum(row["games"] for row in catalog["entries"]) == 256
for row in catalog["entries"]:
    print(row["games"], row["deck_id"], row["exact_deck_sha256"])
PY
```

训练时每个 256-slot unit 会 shuffle slot 顺序并使用完整频次；构成、采样权重或 curriculum 改变必须成为新的 version/config identity。Frozen evaluation 使用相同 256-slot base frequency，但由 identity-bound schedule 生成固定 seeds 和 replicas。

### 10.3 exact-deck lane audit

CUDA codec 会将每个 ready lane 的 `resource_cat/resource_num/resource_mask` 与这条 job 的实际 60-card multiset 比较。mismatch 会报告 job、role、expected deck hash 和实际 ledger，然后 `FATAL`，该 Episode 不得进入 PPO batch。

## 11. 正式训练的特殊配置和行为

### 11.1 Formal V1 固定配置

| 配置 | 值 |
|---|---:|
| focal deck | `007_dragapult_ex` / `dragapult_ex_07bedfffbfad` |
| opponent policy | full immutable `Policy-0809` |
| games/update | 256 |
| retained trajectories/update | 256（全部） |
| CUDA lanes / rollout batch | 256 / 256 |
| PPO minibatch | 2,048 decisions |
| PPO physical forward microbatch | 1,024 decisions（每个 logical step 最多 2 次 forward/backward） |
| behavior probe chunk | 512 decisions |
| PPO epochs | 最多 3 |
| actor LR | `5e-6` |
| Value-side LR | `1e-4` |
| PPO clip | `0.10` |
| entropy coefficient | `0.003` |
| Value coefficient | `0.5` |
| max grad norm | `0.5` |
| GAE | `gamma=1`, `lambda=0.95`, turn clock |
| Meta anchor | `0.10` |
| weight decay / scheduler | `0` / none |
| training dtype | FP32 |
| Frozen evaluation | update 0 和每 10 updates，CUDA-2048 |
| checkpoint | model-only，每 update 全量保留 |
| formal update limit | none；人工 stop |

`V1_ppo_protocol_v2_baseline` 的 runner 会拒绝 `--updates`、非 `FULL_MODEL`、非 256 games、非 2,048 minibatch、非 3 epochs 或非 eval-every 10。不要通过换一个 version string 偷渡另一套 V1 语义。

### 11.2 完整 data epochs 与 KL guard

每个 update 先用 source policy update `k-1` 收集 on-policy trajectories，再生成 checkpoint update `k`。PPO 对所有有效 decisions 做 fresh permutation、without replacement 的完整遍历，保留最后 short minibatch。

epoch 2/3 只有在固定 rollout-wide behavior KL guard 通过时才进入。guard 包含 4,096 个确定性 shuffle rows 加全部 compound/macro rows；target/hard KL 是 `0.015/0.025`。old logprob、old Value、advantage 和 return 在 rollout 后一次计算，不能跨 epoch 刷新。

update 1 和每第 10 个 update 额外执行 all-decision pre-update old-logprob audit；其他 update 对 guard set audit。probe rows 不计入 optimizer samples。

### 11.3 chance-boundary replacement 仅属于训练 rollout

Phantom allocation 前若 Confusion 等真实随机结果形成 pre-allocation chance boundary，训练 collector 可在同 opponent/seat slot 中用新的可审计 seed 替换该 Episode，并将来源写入 `artifact/schedules/chance_boundary_replacements/`。这是训练数据接纳合同。

Frozen evaluation 不允许失败/unfinished 后换 seed 补跑；其 2,048 个 jobs 必须全部按原 schedule terminal。

### 11.4 sampled rollout 不是 frozen greedy strength

rollout 指标描述生成该批数据的 stochastic behavior policy。`rollout/source_policy_update=k-1` 不等于 `checkpoint/update=k` 的 greedy strength。正式强度只能来自同合同的 Frozen greedy evaluation，并记录 `eval/checkpoint_update`。

### 11.5 Update 0 的特殊工作

正式 run 创建 update-0 model-only checkpoint 后，会先：

1. 完整 materialize FP16-storage/FP32-runtime candidate；
2. 执行 Frozen-0809 CUDA-2048 baseline；
3. 保存 `artifact/schedules/eval_frozen_2048.json`；
4. 保存 `artifact/frozen_results/core-update-000000.json`；
5. 执行 283-decision training-to-package attested parity；
6. 初始化 `candidate_leader.json`，但状态仍是 `NOT_PROMOTED`。

因此首次 training metric 可能明显晚于进程启动。watchdog 的 first-metrics grace 必须覆盖 update-0 Frozen-2048 和 package parity 时间，不能看到几分钟没有 JSONL 就手工判死。

## 12. 启动 formal PPO

### 12.1 启动前人工确认

正式启动会创建 immutable version、W&B run、每 update checkpoint，并执行昂贵的 CUDA-2048 evaluation。必须先取得用户对“启动这个 formal run”的明确授权。阅读本手册或请求写文档不构成启动授权。

确认以下路径全都不存在或为空：

```text
rl_runs/0042_full_model_design/versions/V1_ppo_protocol_v2_baseline/artifact/
rl_runs/0042_full_model_design/versions/V1_ppo_protocol_v2_baseline/checkpoint/
rl_runs/0042_full_model_design/versions/V1_ppo_protocol_v2_baseline/tensorboard/
rl_runs/0042_full_model_design/versions/V1_ppo_protocol_v2_baseline/wandb/
experiments/0042_full_model_design/evaluation/V1_ppo_protocol_v2_baseline.html
```

任一路径已使用时，禁止覆盖或复用。分配下一个严格递增 `V<n>_<ascii_snake_case>`，并先把新实验语义同步进 DESIGN/DECISIONS/config；不要仅改 CLI version 名称。

### 12.2 推荐通过 watchdog 前台启动

```bash
python3 -m train.0042_full_model_design.monitor_training \
  --version V1_ppo_protocol_v2_baseline \
  --first-metrics-grace-seconds 7200 \
  --metrics-stale-seconds 7200 \
  --minimum-free-gib 100 \
  -- \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -m train.0042_full_model_design.training.run_full_semantic \
    --version V1_ppo_protocol_v2_baseline \
    --preset FULL_MODEL \
    --engine-backend accelerated:cuda_resident \
    --games-per-update 256 \
    --trajectory-games-per-update 256 \
    --cuda-lane-count 256 \
    --rollout-batch-size 256 \
    --ppo-minibatch-size 2048 \
    --ppo-forward-microbatch-size 1024 \
    --ppo-epochs 3 \
    --eval-every 10 \
    --wandb-mode online \
    --launch-formal
```

V1 formal run 不传 `--updates`。`--launch-formal` 是显式授权 token；缺少它，runner 会拒绝创建 run。
formal runner 同时会拒绝缺少 `expandable_segments:True` 的 CUDA allocator 配置。

watchdog 输出位于：

```text
.tmp/training_monitor/0042_full_model_design/V1_ppo_protocol_v2_baseline/
  training.log
  heartbeat.json
  alert.json（只有告警时）
```

### 12.3 观察训练

```bash
tail -f .tmp/training_monitor/0042_full_model_design/V1_ppo_protocol_v2_baseline/training.log

python3 -m train.0042_full_model_design.monitor_training --help

./scripts/start_tensorboard.sh
```

重点查看：

- `status.json` 的 state、checkpoint update、rollout source update、W&B sync；
- `training_metrics.jsonl` 是否按 update 单调追加；
- `rollout/lane_routing_audit_*` 是否始终 PASS/0 failures；
- `rollout/cuda_feature_d2h_bytes` 是否保持 0；
- behavior KL、ratio、entropy、grad norm 是否 finite；
- adapter gate、residual ratio 和 Value calibration；
- `eval/*` 与 `rollout/*` 是否被正确区分；
- GPU peak allocated/reserved、host memory、disk free；
- Frozen results 是否 2,048 terminal、0 error、0 unfinished、0 semantic fallback。

### 12.4 安全停止

正式 run 在每个完整 update 结束后检查 stop sentinel。请求安全停止：

```bash
touch rl_runs/0042_full_model_design/versions/V1_ppo_protocol_v2_baseline/artifact/STOP_REQUESTED
```

不要用 `kill -9` 作为正常停止方式。等待当前 update 完成并确认：

```text
training_summary.json: state = stopped_by_request
status.json: state = stopped_by_request
最后一个 update checkpoint 已原子写入
training_metrics.jsonl 最后一行完整
```

### 12.5 唯一允许的 update-0 resume

`--resume-update0` 只允许一种非常窄的情况：formal version 已在 pre-PPO update 0 中断，`status.state=failed`，唯一 checkpoint 是 `update-000000.pt`，且 metrics 为空。它仍需要 `--launch-formal`。

任何已经做过 PPO 的 checkpoint 都不能在同一 version 精确续跑。必须创建新版本、重新初始化 optimizer，并重新采集 on-policy 数据。

## 13. Frozen CPU-256、CUDA-2048 与候选导出

### 13.1 正式 CUDA-2048

formal runner 在 update 0 和每 10 updates 自动使用同一个 candidate materializer、Policy-0809 resolver、hard gate 和 Frozen schedule。Update 0 的 package parity 必须执行真实 0042 Value/Meta/Strategy Adapter 路径；旧 0038 `meta_head + meta_conditioner` 接口不能作为替代。权威逐局结果位于：

```text
rl_runs/0042_full_model_design/versions/<version>/artifact/frozen_results/
  core-update-000000.json
  core-update-000010.json
  core-update-000020.json
  ...
```

`candidate_leader.json` 只是候选定位记录，不是自动 Promote 决策。

### 13.2 CPU-256 复核

CPU-256 是独立 frequency unit（replica 0），用于较小规模官方 CPU 路径复核：

```bash
python3 -m train.0042_full_model_design.diagnostics.cpu256_policy0809_preflight \
  --checkpoint rl_runs/0042_full_model_design/versions/<version>/checkpoint/update-XXXXXX.pt \
  --output .tmp/evaluation/0042_cpu256/<version>-update-XXXXXX.json
```

该入口仍在 `cuda:0` 上 materialize candidate/opponent 并通过 official CPU engine workers 运行；输出必须同时包含 candidate deployment 和 opponent policy identity `PASS`。

### 13.3 导出自包含 candidate package

对已经选定、且有 qualifying Frozen identity evidence 的 checkpoint：

```bash
python3 -m train.0042_full_model_design.export_full_semantic_candidate \
  --source archive/submission/0031_zero_shot_0809_007_dragapult_ex_fp16_storage_fp32_runtime \
  --checkpoint rl_runs/0042_full_model_design/versions/<version>/checkpoint/update-XXXXXX.pt \
  --output evaluation/arena/candidates/<candidate_name>

python3 -m evaluation validate evaluation/arena/candidates/<candidate_name>
```

`<version>`、`XXXXXX` 和 `<candidate_name>` 是操作时必须替换的明确目标，不可直接复制尖括号文本执行。output 必须是不存在的新目录。exporter 会在失败时删除不完整 output，并拒绝 symlink、checkpoint/schema mismatch 或 deployment-effective hash mismatch。

将 candidate 放入 `evaluation/arena/candidates/` 不等于进入正式 opponent pool，也不等于 Kaggle submission。收编到 `evaluation/arena/opponents/`、PROMOTE 和 Kaggle upload 都需要用户另行明确确认。

## 14. 产物地图

```text
train/0042_full_model_design/
  实现、configs、policy registry、55-deck league、tests、diagnostics

experiments/0042_full_model_design/
  DESIGN.md / DESIGN.html、DECISIONS.md、manifest、正式 evaluation HTML

rl_runs/0042_full_model_design/versions/<Vn_tag>/
  artifact/
    training_config.json
    status.json
    training_metrics.jsonl
    opponent_snapshot.json
    schedules/
    frozen_results/
    candidate_leader.json
    training_summary.json
  checkpoint/
    update-000000.pt
    update-000001.pt
    ...
  tensorboard/
  wandb/

.tmp/
  可删除的 preflight、smoke、candidate materialization、monitor 和本机 build 产物
```

model-only checkpoint 禁止包含 optimizer、scheduler、GradScaler、RNG、DataLoader 位置、rollout buffer 或 replay。默认保留每个 update；没有用户明确授权不得 prune/rotate/delete。

## 15. 禁用或容易误用的入口

不要把以下历史/诊断入口当 0042 正式 Frozen evidence：

- `train.0042_full_model_design.evaluation.run_update0_frozen`：代码已 fail-closed 禁用；
- `train.0042_full_model_design.evaluate_frozen_checkpoint`：保留旧 seeded-512/default V2 路径，不能替代当前 0042 Frozen-0809 contract；
- raw FP32 checkpoint 的 greedy 对局：只能诊断；
- fixed-seat、balanced-seat、旧 512、Frozen-0806 或旧 contract report：不可与当前证据比较；
- `diagnostics.smoke_ppo`：明确是 non-candidate smoke；
- sampled training rollout win rate：不是 checkpoint greedy strength。

如果一个入口绕过 project-local candidate materializer、Policy-0809 resolver、deployment audit、identity-bound schedule 或 official engine，它就不是 0042 的正式评测路径。

## 16. 常见故障

### `checkpoint is missing` / SHA mismatch

重新同步默认模型资产，按第 4 节逐一核对。不要编辑 `policy_registry.json` 让错误 hash 通过。

### `validated CUDA rules/extension artifacts are unavailable`

检查第 6 节的两个固定路径、文件大小、rule-pack provenance 和 extension import。新 build 必须重新跑 parity gates。

### `_ptcg_cuda` undefined symbol / import failure

清理的只能是明确的本机构建目录，不要删除仓库或 checkpoint。使用当前 Python/PyTorch/CUDA toolchain 重新 CMake configure/build；确认 `Torch_DIR` 来自执行训练的同一解释器。

### `FATAL: Opponent policy identity violation`

读取错误中的 requested ID、component 和 actual hash。检查 actor checkpoint 是否被替换、focal module 是否误接入 opponent、registry 是否 drift。不得 warning-and-continue。

### `FATAL: exact-deck routing mismatch`

检查 lane job、actor role、deck hash、resource ledger 和 deck-static cache key。最常见错误是 mixed batch 复用 `jobs[0].opponent_deck`。用 grouped-by-deck reference 找到首次 divergence。

### `candidate deployment ... FAIL`

确认 merge 顺序、完整 trained heads/adapters、FP16 storage、strict FP32 runtime 和 schema metadata。不要用 source FP32 成绩替代失败的 deployment artifact。

### W&B auth failure

先执行 `python3 -m wandb login` 和 API-key probe。若经批准改用 offline，必须记录原因；不能把 offline smoke 混进正式 online run。

### CUDA OOM

保存错误、GPU peak metrics 和当前 version status。当前合同固定 logical minibatch 2,048、physical microbatch 1,024、expandable segments。不要静默改变 lane count、rollout batch 或这些 memory-execution 参数后继续写同一 version。WSL 下禁止 `gpustat -i 1`；使用训练内 telemetry 或至少 10 秒采样。出现 `dxgvmb_send_*` D-state、`make_resident -12` 或 `device not ready` 时先停止训练并从 Windows 执行 `wsl --shutdown`。

### metrics 长时间不出现

update 0 会先执行 CUDA-2048 与 package parity。检查 heartbeat、GPU utilization 和 log，而不是立刻重启。确认 watchdog grace 覆盖实际 baseline 时间。

### run 已存在

这是 immutable version guard 正常工作。不得删除旧失败记录或复用旧目录；分配更高 `V<n>` 并记录新假设/修复。

## 17. Agent 交付前检查表

### 环境与资产

- [ ] Python 3.11+、PyTorch CUDA、W&B 可用；
- [ ] actor、Value、allocation sidecar、portable base package hash 正确；
- [ ] official seeded CPU runtime 可构建；
- [ ] official rule pack 与 `_ptcg_cuda.so` 存在且 provenance 可审计；
- [ ] Gate C fixed trace/manifest 完整。

### 语义与测试

- [ ] 完整阅读 canonical protocol 与 0042 contract；
- [ ] `Policy-0809` identity audit PASS；
- [ ] focal/opponent 无 Parameter/tensor storage alias；
- [ ] 55-deck/256-slot catalog audit PASS；
- [ ] mixed-deck CUDA vs grouped reference PASS；
- [ ] CPU/CUDA lockstep parity PASS；
- [ ] candidate FP16-storage/FP32-runtime audit PASS；
- [ ] focused 与 full 0042 tests PASS；
- [ ] real Protocol V2 preflight PASS；
- [ ] non-candidate smoke PASS。

### 正式运行

- [ ] 已取得用户对本次 formal launch 的明确授权；
- [ ] version 和四个 version subdirectories 全新；
- [ ] authoritative evaluation HTML 不存在；
- [ ] formal command 不带 `--updates`，配置保持 V1 contract；
- [ ] watchdog、disk threshold、W&B online 已配置；
- [ ] update 0 Frozen-2048 和 attested package parity PASS；
- [ ] 每 update model-only checkpoint 全量保留；
- [ ] rollout source update、checkpoint update 与 frozen eval update 没有混称；
- [ ] 没有自动 Promote、自动收编 opponent 或自动 Kaggle submission。

## 18. 一句话原则

0042 的性能优化只能发生在身份语义之后：先证明 focal 与完整 immutable `Policy-0809` opponent 相互独立，再证明每条 lane 使用自己的 exact 60-card static fields，再证明 CPU/CUDA parity，最后才讨论 resident batching、cache、CUDA graph 和吞吐。
