# Engine CUDA 2.0：RTX 5080 + WSL2 快速上手

这份文档给在 RTX 5080 的 WSL2 Ubuntu 环境中运行 CUDA 2.0 的使用者。全程原生运行，
不需要 Docker，仓库也可以放在任意路径。

更完整的模型身份、Frozen65 全矩阵和故障排查说明见
[`server_native_usage_zh.md`](server_native_usage_zh.md)。

## 1. WSL2 前置条件

- Windows 宿主机安装支持 RTX 5080 和 WSL2 的 NVIDIA Windows 驱动。
- WSL 内不要安装 Linux NVIDIA display driver。GPU driver 由 Windows 映射进 WSL。
- WSL 内需要 CUDA Toolkit、Python、PyTorch、CMake、Ninja 和 C++ 编译器。
- RTX 5080 的 compute capability 是 12.0，编译目标为 `sm_120`；建议使用 CUDA 12.8+
  和支持 Blackwell 的 PyTorch build。
- 建议把仓库放在 WSL 的 Linux 文件系统，而非 `/mnt/c` 或 `/mnt/d`。

检查 GPU 和编译器：

```bash
nvidia-smi
nvcc --version
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

预期 GPU 名称包含 `RTX 5080`，capability 为 `(12, 0)`。若 `nvidia-smi` 在 WSL 内不可用，
先修复 Windows 驱动/WSL GPU 透传；若 `nvcc` 不认识 `sm_120`，先升级 CUDA Toolkit。

## 2. 获取分支并初始化动态路径

```bash
git fetch dragonbra
git switch codex/cortex-cpu-test-checkpoint-engine-cuda-2.0
git pull --ff-only dragonbra codex/cortex-cpu-test-checkpoint-engine-cuda-2.0

export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

export CUDA_ARCH="$(python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the RTX 5080")
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}{minor}")
PY
)"

printf 'repo=%s\nbranch=%s\ncuda_arch=%s\n' \
  "$REPO_ROOT" "$(git branch --show-current)" "$CUDA_ARCH"
test "$CUDA_ARCH" = 120
```

`REPO_ROOT` 来自当前 Git checkout，不要求仓库位于任何固定目录。

## 3. 放入不随 Git 上传的本地资产

以下文件必须从可信备份复制到当前 checkout；它们被 Git ignore，不在分支里：

```text
engine_cuda_2_0/generated/private/official_3aaeaa92/official_rules.bin
engine_cuda_2_0/generated/private/official_3aaeaa92/official_rules.bin.manifest.json
archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt
archive/submission/0031_mega_lopunny_froslass_dd632_0809/strategy/model.bin
```

检查模型：

```bash
cd "$REPO_ROOT"
sha256sum \
  archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt \
  archive/submission/0031_mega_lopunny_froslass_dd632_0809/strategy/model.bin
```

预期 SHA-256：

```text
11ac5c9a40ab46536a1bc651c388617edac4d7c5c4e1e2de1476d3317e7ad7f3  model.pt
3f6683b0d916c72a31f3f487edb597c0cec4aedf529a236a4effacb177de89db  model.bin
```

哈希不一致时停止运行，不要关闭 identity audit，也不要提交这两个模型。

## 4. 原生编译

```bash
cd "$REPO_ROOT"
python3 -m pip install cmake ninja

cmake -S engine_cuda_2_0 -B engine_cuda_2_0/build/native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build engine_cuda_2_0/build/native --parallel
```

构建 PyTorch extension：

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

export PYTHONPATH="$REPO_ROOT/engine_cuda_2_0/build/torch_official:$REPO_ROOT/engine_cuda_2_0/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

## 5. 最小验证

```bash
cd "$REPO_ROOT"
./engine_cuda_2_0/build/native/ptcg_cuda_smoke \
  --envs 4096 --steps 1000 --policies 12

python3 engine_cuda_2_0/tools/smoke_torch_extension.py \
  --extension-dir engine_cuda_2_0/build/torch_official \
  --batch 256 --steps 20 --policies 12
```

再用 0809 模型测试自爆多龙 32 局：

```bash
python3 engine_cuda_2_0/tools/evaluate_policy_0809_cuda.py \
  --deck-number 057 --smoke-games 32
```

测试草系 Ogerpon 与三套最新牌组：

```bash
for deck in 006 063 064 065; do
  python3 engine_cuda_2_0/tools/evaluate_policy_0809_cuda.py \
    --deck-number "$deck" --smoke-games 32
done
```

成功条件：candidate/opponent identity audit 均为 `PASS`，inference fallback、collector
error 和 engine error 均为 0。

## 6. 运行 65 套牌的全组合 parity

```bash
cd "$REPO_ROOT"
export POOL="evaluation/arena/frozen_pools/0812_top100_top500_stadiums_mill_plus_limitless_dragapult_v3"
export CASE_DIR=".tmp/evaluation/frozen65_native_cases"

python3 engine_cuda_2_0/tools/generate_frozen_pair_cases.py \
  "$POOL" "$CASE_DIR" --shards 1

mkdir -p engine_cuda_2_0/build/native_matrix
nvcc --threads 4 --split-compile 1 -std=c++20 -O0 \
  -arch="sm_${CUDA_ARCH}" \
  -I"engine/source/ptcgProgram 22" \
  -Iengine_cuda_2_0/include \
  -Iengine_cuda_2_0/extractor \
  engine_cuda_2_0/benchmarks/official_battle_ordered_matrix_cuda_paired.cu \
  engine_cuda_2_0/src/official_engine_kernels.cu \
  -o engine_cuda_2_0/build/native_matrix/official_battle_ordered_matrix_cuda_paired
```

先跑两个组合，每个组合 2 局：

```bash
head -n 2 "$CASE_DIR/all_cases.tsv" > "$CASE_DIR/smoke_cases.tsv"
engine_cuda_2_0/build/native_matrix/official_battle_ordered_matrix_cuda_paired \
  engine_cuda_2_0/generated/private/official_3aaeaa92/official_rules.bin \
  "$CASE_DIR/smoke_cases.tsv" 1 2 512 coverage-first-legal
```

smoke 通过后运行完整 2,080 组合 × 50 局：

```bash
engine_cuda_2_0/build/native_matrix/official_battle_ordered_matrix_cuda_paired \
  engine_cuda_2_0/generated/private/official_3aaeaa92/official_rules.bin \
  "$CASE_DIR/all_cases.tsv" 1 50 512 coverage-first-legal \
  | tee .tmp/evaluation/frozen65_native_matrix.log
```

每个 `CASE` 的 `state_mismatches`、`status_mismatches` 和 `outcome_mismatches` 必须全为
0。`unfinished_battles` 可以非 0，因为 coverage policy 可能在 512 decision 上限前未结束。

## 7. 5080 常见错误

- `Unsupported gpu architecture 'compute_120'`：CUDA Toolkit 太旧，升级到支持 Blackwell
  的版本。
- `no kernel image is available`：复用了为旧 GPU 编译的产物。删除对应 `build/native` 或
  `build/torch_official` 后用自动检测出的 `CUDA_ARCH=120` 重编译。
- PyTorch 看不到 GPU：先确认 WSL 内 `nvidia-smi`，再安装支持 Blackwell 的 PyTorch
  CUDA build；不要在 WSL 内安装 Linux display driver。
- 编译非常慢：确认 checkout 位于 WSL Linux 文件系统，而不是 `/mnt/c` 或 `/mnt/d`。
- checkpoint 缺失：模型本来就不随 Git 分发，应从可信备份恢复并核对 SHA-256。

