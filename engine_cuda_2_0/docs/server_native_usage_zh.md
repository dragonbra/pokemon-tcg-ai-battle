# Engine CUDA 2.0：Linux GPU 服务器原生使用指南

本文说明如何在 Linux GPU 服务器或 RTX 5080 + WSL2 中直接构建和运行
`engine_cuda_2_0/`，全程不使用 Docker。仓库可以放在任意目录；下文通过 Git 自动寻找
仓库根目录，不依赖 `/root`、用户主目录或某台机器的固定路径。

本引擎位于独立目录 `engine_cuda_2_0/`，不会覆盖原有的 `engine_cuda/`。当前对应的
Git 分支是 `codex/cortex-cpu-test-checkpoint-engine-cuda-2.0`。

## 1. 当前支持范围

- Frozen Pool：`0812_top100_top500_stadiums_mill_plus_limitless_dragapult_v3`
- 牌组数量：65 套，固定 schedule 为 256 slots
- 新增牌组包括自爆多龙、Top 100/Top 500 补充牌组、Great Tusk / Mill
- Top 500 出现的 20 种场地牌已全部在牌池中覆盖
- 草系 Teal Mask Ogerpon ex / Hero's Cape 是 006 号牌组
- 0809 模型以 FP16 portable artifact 保存，并在运行时恢复为 FP32
- 官方 CPU 引擎始终是规则语义 oracle；CUDA 结果不能只比较最终胜负

2026-08-12 的服务器验收使用 RTX 4080 SUPER（SM 89），完成全部 2,080 个无序牌组
组合、每个组合 50 局，共 104,000 局和 20,558,750 次 decision。CPU/POD/CUDA 的
state、status、canonical bytes 和 outcome mismatch 均为 0。详细结果见
[`frozen65_server_parity_20260812.md`](frozen65_server_parity_20260812.md)。

## 2. 拉取正确分支

```bash
git fetch dragonbra
git switch codex/cortex-cpu-test-checkpoint-engine-cuda-2.0
git pull --ff-only dragonbra codex/cortex-cpu-test-checkpoint-engine-cuda-2.0
```

进入仓库中的任意目录后，初始化本文后续命令使用的变量：

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

export CUDA_ARCH="$(python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see a CUDA GPU")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}{minor}")
PY
)"

printf 'repo=%s\nbranch=%s\ncuda_arch=%s\n' \
  "$REPO_ROOT" "$(git branch --show-current)" "$CUDA_ARCH"
test -d "$REPO_ROOT/engine_cuda_2_0"
```

不要在 `main`、`dev/cyd_main` 或旧的 `engine_cuda/` 目录中执行本指南的构建命令。

## 3. Linux 服务器与 WSL2 环境

建议环境：

- Linux x86_64
- NVIDIA driver 可正常加载
- CUDA Toolkit（需要 `nvcc`）
- Python 3.11+
- 与本机 CUDA 兼容的 PyTorch
- CMake 3.24+、Ninja、C++20 编译器

先检查环境：

```bash
nvidia-smi
nvcc --version
python3 --version
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))
PY
```

常见值是：RTX 4080 SUPER 为 `89`/`sm_89`，RTX 3060 为 `86`/`sm_86`，RTX 5080
为 `120`/`sm_120`。上面的命令会自动读取当前 GPU，因此无需把架构写死。RTX 5080
属于 Blackwell，必须使用支持 `sm_120` 的 CUDA Toolkit 和 PyTorch build；建议 CUDA
12.8 或更新版本。如果 `nvcc` 报 `Unsupported gpu architecture 'compute_120'`，说明本机
Toolkit 太旧，应升级工具链，不能把目标伪装成 `sm_89`。

### RTX 5080 + WSL2 特别注意

1. 在 Windows 宿主机安装支持 RTX 5080/WSL2 的 NVIDIA Windows 驱动。
2. 在 WSL 中运行 `nvidia-smi` 确认 GPU 已透传。
3. WSL 内只安装 Linux CUDA Toolkit，不安装 Linux NVIDIA display driver，也不要使用会
   连带安装驱动的 `cuda` 或 `cuda-drivers` 元包。
4. 建议把仓库放在 WSL 的 Linux 文件系统（如任意 `~/work/...` 目录），避免放在
   `/mnt/c` 或 `/mnt/d` 上造成大量小文件编译明显变慢。
5. PyTorch 的 CUDA build、`nvcc` 和 driver 必须互相兼容；以本节检测脚本实际输出为准。

安装缺少的 Python/构建依赖时，可使用服务器现有环境或虚拟环境：

```bash
python3 -m pip install cmake ninja
```

## 4. 准备本地私有资产

以下文件不会随 Git 分支上传，必须从服务器已有资产或可信备份放回相应位置：

```text
engine_cuda_2_0/generated/private/official_3aaeaa92/official_rules.bin
engine_cuda_2_0/generated/private/official_3aaeaa92/official_rules.bin.manifest.json
archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt
archive/submission/0031_mega_lopunny_froslass_dd632_0809/strategy/model.bin
```

其中 `official_rules.bin` 是私有规则包；两个模型文件也只保留在服务器，不得 `git add -f`
或上传到公开 artifact。

核对 0809 模型哈希：

```bash
sha256sum \
  archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt \
  archive/submission/0031_mega_lopunny_froslass_dd632_0809/strategy/model.bin
```

预期值：

```text
11ac5c9a40ab46536a1bc651c388617edac4d7c5c4e1e2de1476d3317e7ad7f3  model.pt
3f6683b0d916c72a31f3f487edb597c0cec4aedf529a236a4effacb177de89db  model.bin
```

若哈希不一致，立即停止；不要跳过 identity audit。

## 5. 原生构建 CUDA runtime 与 PyTorch extension

先构建普通 CUDA runtime：

```bash
cmake -S engine_cuda_2_0 -B engine_cuda_2_0/build/native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build engine_cuda_2_0/build/native --parallel
```

运行最小 smoke：

```bash
./engine_cuda_2_0/build/native/ptcg_cuda_smoke \
  --envs 4096 --steps 1000 --policies 12
```

再构建供 Python/0809 模型使用的 PyTorch extension：

```bash
TORCH_DIR="$(python3 - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")
PY
)"

cmake -S engine_cuda_2_0 -B engine_cuda_2_0/build/torch_official -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DPTCG_CUDA_BUILD_TORCH=ON \
  -DTorch_DIR="$TORCH_DIR"
cmake --build engine_cuda_2_0/build/torch_official --parallel
```

让 Python 找到 extension 和引擎包：

```bash
export PYTHONPATH="$REPO_ROOT/engine_cuda_2_0/build/torch_official:$REPO_ROOT/engine_cuda_2_0/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 engine_cuda_2_0/tools/smoke_torch_extension.py \
  --extension-dir engine_cuda_2_0/build/torch_official \
  --batch 4096 --steps 100 --policies 12
```

修改任何 CUDA header、`.cu` 文件或 ABI 后都必须重新编译，不能继续复用旧 extension。

## 6. 先做快速回归

不依赖 GPU 的合同检查：

```bash
python3 -m unittest discover -s engine_cuda_2_0/tests -p 'test_*.py' -v
python3 engine_cuda_2_0/tools/compile_rule_pack.py \
  engine_cuda_2_0/rules/smoke_rules.json --check
```

原生 GPU extension smoke：

```bash
PYTHONPATH="$REPO_ROOT/engine_cuda_2_0/build/torch_official:$REPO_ROOT/engine_cuda_2_0/python:$REPO_ROOT" \
python3 engine_cuda_2_0/tools/smoke_torch_extension.py \
  --extension-dir engine_cuda_2_0/build/torch_official \
  --batch 256 --steps 20 --policies 12
```

任何 `state_mismatches`、`status_mismatches`、`outcome_mismatches` 或 engine error 非 0
都属于失败，不能因为最终胜负相同而忽略。

## 7. 用 0809 模型跑 Frozen65

单套牌先跑少量 smoke。以下示例测试自爆多龙（057）32 局：

```bash
export PYTHONPATH="$REPO_ROOT/engine_cuda_2_0/build/torch_official:$REPO_ROOT/engine_cuda_2_0/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 engine_cuda_2_0/tools/evaluate_policy_0809_cuda.py \
  --deck-number 057 \
  --smoke-games 32
```

其他常用编号：

```text
006  Teal Mask Ogerpon ex / Hero's Cape
057  Dragapult ex / Dusknoir（自爆多龙）
063  Mega Lucario ex / Solrock（Gravity Mountain）
064  Mega Lopunny ex / Mega Froslass ex（Lumiose City）
065  Crustle / Great Tusk Mill
```

分别 smoke 新增的三套牌：

```bash
for deck in 063 064 065; do
  python3 engine_cuda_2_0/tools/evaluate_policy_0809_cuda.py \
    --deck-number "$deck" --smoke-games 32
done
```

正式 2,048 局单牌组评估时去掉 `--smoke-games`：

```bash
python3 engine_cuda_2_0/tools/evaluate_policy_0809_cuda.py --deck-number 057
```

也可以执行目录中的前十套：

```bash
python3 engine_cuda_2_0/tools/evaluate_policy_0809_cuda.py --top-ten
```

成功输出必须包含：

```text
candidate_audit=PASS
opponent_audit=PASS
```

同时检查生成的 JSON 中 inference fallback、collector error 和 engine error 均为 0。

## 8. Frozen65 全组合 CPU/CUDA parity

这项测试不是模型强度测试。它用同一 seed 和动作策略逐 decision 比较官方 CPU、host
POD 与 CUDA POD 的状态和结果。

先生成 65 套牌的 2,080 个无序组合。服务器有多张 GPU 时，可用 `--shards` 拆分：

```bash
POOL=evaluation/arena/frozen_pools/0812_top100_top500_stadiums_mill_plus_limitless_dragapult_v3
CASE_DIR=.tmp/evaluation/frozen65_native_cases
python3 engine_cuda_2_0/tools/generate_frozen_pair_cases.py \
  "$POOL" "$CASE_DIR" --shards 1
```

原生编译矩阵执行器：

```bash
mkdir -p engine_cuda_2_0/build/native_matrix
nvcc --threads 4 --split-compile 1 -std=c++20 -O0 -arch="sm_${CUDA_ARCH}" \
  -I"engine/source/ptcgProgram 22" \
  -Iengine_cuda_2_0/include \
  -Iengine_cuda_2_0/extractor \
  engine_cuda_2_0/benchmarks/official_battle_ordered_matrix_cuda_paired.cu \
  engine_cuda_2_0/src/official_engine_kernels.cu \
  -o engine_cuda_2_0/build/native_matrix/official_battle_ordered_matrix_cuda_paired
```

运行全部组合，每个组合 50 个 seed：

```bash
engine_cuda_2_0/build/native_matrix/official_battle_ordered_matrix_cuda_paired \
  engine_cuda_2_0/generated/private/official_3aaeaa92/official_rules.bin \
  "$CASE_DIR/all_cases.tsv" \
  1 50 512 coverage-first-legal \
  | tee .tmp/evaluation/frozen65_native_matrix.log
```

完整矩阵会运行较长时间。第一次使用时建议先截取两个组合做 smoke：

```bash
head -n 2 "$CASE_DIR/all_cases.tsv" > "$CASE_DIR/smoke_cases.tsv"
engine_cuda_2_0/build/native_matrix/official_battle_ordered_matrix_cuda_paired \
  engine_cuda_2_0/generated/private/official_3aaeaa92/official_rules.bin \
  "$CASE_DIR/smoke_cases.tsv" \
  1 2 512 coverage-first-legal
```

每个 `CASE` JSON 都必须满足：

```text
state_mismatches = 0
status_mismatches = 0
outcome_mismatches = 0
```

`unfinished_battles` 可以非 0：coverage policy 可能在可逆动作间循环，达到 512 decision
上限会记为 unfinished；上限之前的每个 decision 仍然参与 CPU/CUDA 对比。

## 9. 新牌组和场地牌的接入流程

新增牌组时不要直接修改已有牌组：

1. 在 Frozen Pool 的 `decks/` 下建立新编号目录，保存 exact 60-card `deck.csv`。
2. 更新该牌组 `manifest.json`、池 `manifest.json` 与固定 `schedule.json`。
3. 更新 exact-deck SHA-256；哈希校验失败时禁止继续。
4. 先跑普通 Python 合同测试和该牌组的 0809 32 局 smoke。
5. 用新牌组对支持该场地牌的旧牌组做定向 parity，确认特效实际触发。
6. 最后把新牌组加入全组合，每个新旧组合跑 50 局。

“牌表中包含某张场地牌”不等于其特效已验证。验收报告还应记录该牌被打出、替换、Ability
或相关 attack 实际触发的次数。

## 10. 常见问题

### 找不到 0809 checkpoint

模型被 Git ignore 是设计行为。将模型放回第 4 节的固定路径，并核对 SHA-256；不要改代码
绕过检查，也不要把模型提交到分支。

### Frozen source snapshot hash 校验失败

通常表示当前 checkout、牌池 manifest 和本地缓存来自不同版本。先确认分支和 commit，再清理
对应的本地临时结果并重新生成；不要篡改预期哈希来让测试变绿。

### `no kernel image is available`

构建时使用了错误的 compute capability。重新查询 GPU，并用正确的
`CMAKE_CUDA_ARCHITECTURES` 和 `-arch=sm_XX` 全量重编译。

### CUDA OOM

先降低 `--batch`、lane count 或并行 shard 数。显存同时被 state POD、device stack、模型、
activation 和 rollout buffer 使用，不能只按模型文件大小估算。

### CPU/CUDA 胜负相同但 state mismatch

仍然判定失败。CUDA 2.0 的目标是逐 decision 的状态、合法动作语义、continuation/RNG、
flow status 与终局结果同时一致。
