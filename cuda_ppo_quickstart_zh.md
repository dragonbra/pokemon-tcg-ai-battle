# CUDA 引擎 PPO 快速使用说明

本文用于复现当前 GPU-resident official engine + Semantic0031 模型的 rollout，
并确认运行的确是 CUDA 引擎，而不是 CPU official engine worker 配合 GPU 推理。
所有命令均从仓库根目录执行。

## 1. 先确认使用正确分支

```bash
git fetch origin
git switch codex/enginecuda-resident-rollout
git pull --ff-only
```

该版本对应提交：

```text
b1dff2d perf: accelerate CUDA resident PPO rollout
```

不得修改 `engine/source/`。CUDA runtime 位于 `engine_cuda/`，官方源码只作为
只读 oracle 和规则包来源。

## 2. 不要混淆两套运行路径

下面两种路径不是同一个引擎：

| 入口 | 对局引擎 | 模型推理 | 是否为本文的 CUDA PPO 路径 |
|---|---|---|---|
| `python -m evaluation.performance_profile` | CPU official engine workers | GPU resident server | 否 |
| `engine_cuda/tools/benchmark_0035_cuda_full_pool_rollout.py` | CUDA `OfficialCudaEngine` | CUDA | 是 |

Evaluation 加速指南中的 `--batch-size 64` 是 resident inference server 的推理
batch，不是 CUDA 引擎的 lane 数。用该入口测得 `1.77 games/s`，不能据此判断
CUDA 引擎速度。

## 3. 必需资产

运行前检查以下文件存在：

```bash
test -f engine_cuda/generated/private/official_3aaeaa92/official_rules.bin
test -f bc_models/semantic0031_0806_shared_prototype_fp32.pt
test -f rl_runs/0035_lucario_semantic0031_cpu_ppo/versions/V4_lucario_shared0031_cpu_512u_first50_20260807/artifact/opponent_snapshot.json
test -f train/0034_dragapult_third_large_model_rl/league/decks/mega_lucario_ex_solrock_77a53ffc32f8/deck.csv
```

`official_rules.bin`、私有 checkpoint 和运行资产不会随公开 Git 自动补齐。
缺失时应从已验证机器安全复制，不能用其他同名文件替代，也不能跳过 hash/manifest
检查。

## 4. 编译 PyTorch CUDA extension

要求 Python 能导入 CUDA 版 PyTorch，并且 `torch.cuda.is_available()` 为 `True`：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY
```

根据 capability 构建。例如 RTX 4080 的 capability 通常为 `(8, 9)`，对应 `89`：

```bash
python -m pip install cmake ninja

CUDA_ARCH="$(python -c 'import torch; major, minor = torch.cuda.get_device_capability(0); print(f"{major}{minor}")')"
TORCH_CMAKE_DIR="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")')"

cmake -S engine_cuda -B engine_cuda/build/torch_official -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DPTCG_CUDA_BUILD_TORCH=ON \
  -DTorch_DIR="$TORCH_CMAKE_DIR"
cmake --build engine_cuda/build/torch_official --parallel
```

每次更新 CUDA header、kernel、ABI 或切换提交后都必须重新构建，不能继续加载旧
`_ptcg_cuda.so`。

## 5. 确认实际加载的是 CUDA extension

```bash
PYTHONPATH="$PWD/engine_cuda/build/torch_official:$PWD/engine_cuda/python" \
python - <<'PY'
import _ptcg_cuda
import torch

print("extension:", _ptcg_cuda.__file__)
print("ABI:", _ptcg_cuda.ABI_VERSION)
print("official state ABI:", _ptcg_cuda.OFFICIAL_STATE_ABI_VERSION)
print("device:", torch.cuda.get_device_name(0))
PY
```

`extension` 必须指向刚才构建目录中的 `_ptcg_cuda.so`。若指向另一个 build 目录，
应先修正 `PYTHONPATH`，否则测到的是旧引擎。

## 6. 复现完整 CUDA 对局吞吐

先用已验证的 50 对手池、152 条并发 lane 运行：

```bash
mkdir -p .tmp/evaluation/enginecuda_friend_check

PYTHONPATH="$PWD/engine_cuda/build/torch_official:$PWD/engine_cuda/python" \
python engine_cuda/tools/benchmark_0035_cuda_full_pool_rollout.py \
  --games 152 \
  --opponent-limit 50 \
  --max-decisions 256 \
  --check-interval 8 \
  --compact-semantic-prefixes \
  --output .tmp/evaluation/enginecuda_friend_check/full_pool_152.json
```

结果至少检查：

```bash
python - <<'PY'
import json

path = ".tmp/evaluation/enginecuda_friend_check/full_pool_152.json"
report = json.load(open(path, encoding="utf-8"))
print("passed:", report["passed"])
print("extension:", report["extension"])
print("device:", report["device"])
print("completed:", report["collector"]["completed_games"])
print("errors:", report["collector"]["errors"])
print("games/s:", report["collector"]["games_per_second_wall"])
print("CUDA resident:", report["cuda_resident"])
PY
```

正确门禁应满足：

- `passed == true`；
- `completed_games == 152`；
- `errors == 0`；
- `extension` 为当前 build 的 `_ptcg_cuda.so`；
- `cuda_resident` 中 engine、codec、模型、动作路径均为 CUDA；
- 运行过程中 `nvidia-smi` 能看到该 Python 进程和显存占用。

本次 RTX 4080 参考实测为 `152/152`、`0 error`、约 `20.38 games/s`。
不同 GPU、功耗、时钟和软件版本可以不同，但若只有约 `1.77 games/s`，优先按第 8 节
排查，而不是直接归因于 batch。

## 7. batch / lane 应怎样设置

这个 benchmark 用 `--games` 同时决定静态并发对局数，也就是 CUDA engine lanes。
它没有 Evaluation CLI 的 `--batch-size` 参数。

- `64 lanes`：可以运行，但可能不能充分摊薄模型和 kernel launch 开销；
- `152 lanes`：适合做完整、可比的零错误吞吐门禁；
- `512 lanes`：当前约 32 GiB GPU 的推荐训练并发规模；
- `1200 lanes`：能够运行，但 PyTorch reserved memory 实测约 `28.61 GiB`，吞吐仅约
  `20.22 games/s`，没有超过 152/512 档，OOM 风险更高，不建议作为默认配置。

PPO 的 `minibatch_size` 是优化器更新时一次训练多少 decision，与 engine lanes 是两个
不同概念。调大 PPO minibatch 不会自动增加 rollout 的并发对局数。

静态 benchmark 达到 `max-decisions` 时，极少数超长对局可能仍未结束；只要
`errors == 0`，这属于长尾而非 engine error。正式 resident collector 会在 lane 终局后
原地补充新 Episode，不应等待整批最慢对局。

## 8. 只有约 1.77 games/s 时的检查顺序

1. 确认运行的是 `benchmark_0035_cuda_full_pool_rollout.py`，而不是
   `evaluation.performance_profile` 或普通 `evaluation run`。
2. 检查 JSON 的 `extension`，确认没有加载旧 `_ptcg_cuda.so`。
3. 切换/拉取分支后重新编译 extension。
4. 必须加 `--compact-semantic-prefixes`。
5. 检查 `cuda_resident` 字段，不能出现 CPU inference fallback。
6. 确认 checkpoint、50 对手 snapshot、focal deck 和 seed 合同一致。
7. 用 `nvidia-smi` 检查 GPU 是否被其他进程占用、是否处于低功耗或显存不足状态。
8. 分别测试 `--games 64`、`152`、`512`，保持其他参数完全一致，比较相邻三次运行的
   中位数。
9. 不要把包含 PPO backward/update、checkpoint 写盘或 W&B 同步的端到端速度，与纯
   rollout `games/s` 直接比较。

仅将 lanes 从 64 提高通常会有帮助，但从约 `1.77` 到参考值的巨大差距不应先归因于
batch。最常见原因是运行入口不同、旧 extension、未启用 compact semantic prefixes，
或把完整 PPO update 的速度当成纯 rollout 速度。

## 9. CUDA PPO 合同 smoke

下面的入口验证 CUDA rollout buffer、GAE 和 PPO update 能闭环，但它是诊断 smoke，
不是正式训练版本：

```bash
PYTHONPATH="$PWD/engine_cuda/build/torch_official:$PWD/engine_cuda/python" \
python engine_cuda/tools/run_0035_cuda_ppo_collector_probe.py \
  --steps 4 \
  --deck-limit 50 \
  --minibatch-size 256 \
  --output .tmp/evaluation/enginecuda_friend_check/ppo_contract_probe.json
```

报告必须满足 `passed == true`、`collector.error_count == 0`，并确认
`cuda_resident` 中 official engine、semantic observation、rollout buffer、GAE 和 PPO
update 均为 `true`。正式训练仍需使用对应编号项目的 version/config、固定 opponent
snapshot、model-only checkpoint 和 W&B 记录合同。

## 10. 结果边界

`games/s` 只证明工程吞吐，不证明策略强度。任何策略强度结论仍必须使用同一 official
engine runtime、opponent snapshot、seed/先后手合同和冻结 greedy evaluation。
