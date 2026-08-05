# CUDA 引擎使用说明

本文说明 `dev/cyd_main` 分支下 `engine_cuda/` 的构建、Docker CUDA 运行、
0022 的 40 牌组与 Frozen51 parity 验证，以及 PyTorch/GPU 常驻接口。所有命令都从仓库根目录执行。

## 1. 当前状态与证据边界

截至 2026-08-06，在 Docker + RTX 3060 Laptop GPU 上完成了以下验证：

| 验证集 | 对局/检查量 | CPU 官方引擎与 CUDA 结果 |
|---|---:|---:|
| production turn-flow runtime | 3 个运行时场景 | state/status mismatch 均为 0 |
| 40 副牌镜像对局 | 800 局，165,623 decisions | mismatch 0，官方平局 13 |
| 40 副牌全部非镜像有序 matchup | 1,560 局，319,574 decisions | mismatch 0，官方平局 9 |
| 3 个高风险牌组定向回归 | 1,500 局，312,530 decisions | mismatch 0 |
| Frozen51 全部非镜像有序 matchup | 2,550 局，545,094 decisions | mismatch 0，官方平局 13 |
| semantic micro fixture | 163 checks | 全部通过 |

这里的 parity 不只比较最终胜负。每个 decision 都比较合法动作语义、CPU POD/CUDA
原始状态、flow status，终局再比较 win/loss/draw。官方引擎产生的 draw 保留为 draw，
不会改写为任意一方胜负。

这些结果证明当前 40 牌组与 Frozen51 验收语料在已运行 seeds/策略下语义一致，不等于穷举了所有
随机策略、镜像对局、卡牌组合和未来牌组。`engine/source/` 中未修改的官方 CPU 引擎仍是 oracle。

## 2. 目录

```text
engine_cuda/
  benchmarks/               CUDA smoke 和 paired benchmark
  extractor/                只读官方源码到 POD/rule IR 的桥接器
  fixtures/0022_deck40/     已验证的 40 副 exact 60-card 牌组
  include/ptcg_cuda/        POD、规则解释器、continuation、runtime ABI
  python/ptcg_cuda_engine/  Python/PyTorch 绑定层
  src/                      CUDA kernel 和 torch extension
  tests/                    CPU 合同与回归测试
  tools/                    Docker、parity、PPO/runtime 工具
  generated/private/        本地私有规则包，Git ignore
  build/                    本地编译产物，Git ignore
  artifacts/                本地报告，Git ignore
```

`fixtures/0022_deck40/manifest.json` 固定 40 副牌的 ID、SHA-256、分组和 focal deck。
每个子目录仅包含 `deck.csv` 与经过清理的公开 provenance；修改任何牌表都会触发 hash
校验失败。

## 3. 环境要求

- Python 3.11 或更高版本；
- Docker Desktop + WSL2（Windows）或 Docker Engine（Linux）；
- NVIDIA driver 和可用的 Docker GPU runtime；
- RTX 3060 使用 SM 86；其他 GPU 构建时显式修改 CUDA architecture；
- PyTorch extension 需要与 PyTorch 匹配的 CUDA devel image。

先验证容器能看到 GPU：

```powershell
docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

预期输出中包含显卡名称，并且命令退出码为 0。若 Docker 报
`could not select device driver`，先修复 NVIDIA Container Toolkit/WSL GPU 转发，
不要继续做 parity 测试。

## 4. 不需要 GPU 的检查

```powershell
python -m unittest discover -s engine_cuda/tests -p "test_*.py" -v

python engine_cuda/tools/compile_rule_pack.py `
  engine_cuda/rules/smoke_rules.json `
  --check
```

PyTorch/CUDA 专用测试在没有 CUDA extension 时可以 skip；普通 schema、rule pack、
deck manifest 和 parity case generation 测试必须通过。

## 5. 准备 private official rule pack

`official_rules.bin` 来自 competition-use-only 官方源码，禁止提交到 Git、W&B、模型包
或公开 artifact。仓库的 `.gitignore` 已屏蔽 `engine_cuda/generated/private/`。

工具默认只读使用：

```text
engine/source/ptcgProgram 22/
```

如果本机已有通过验证的 snapshot，放到：

```text
engine_cuda/generated/private/official_3aaeaa92/official_rules.bin
engine_cuda/generated/private/official_3aaeaa92/official_rules.bin.manifest.json
```

需要从源码重建时，先准备冻结的 seeded oracle 和英文/日文 card data，再执行：

```powershell
python engine_cuda/tools/extract_official_rules.py `
  --source "engine/source/ptcgProgram 22" `
  --oracle "<path-to-frozen-libcg_seeded.so>" `
  --card-data data/simulation/EN_Card_Data.csv data/simulation/JP_Card_Data.csv `
  --output engine_cuda/generated/private/official_3aaeaa92

python engine_cuda/tools/compile_official_rule_pack.py `
  engine_cuda/generated/private/official_3aaeaa92/official_rules.json `
  engine_cuda/generated/private/official_3aaeaa92/official_rules.bin

python engine_cuda/tools/verify_rule_pack.py `
  --manifest engine_cuda/generated/private/official_3aaeaa92/official_rules.bin.manifest.json `
  --source "engine/source/ptcgProgram 22" `
  --oracle "<path-to-frozen-libcg_seeded.so>" `
  --card-data data/simulation/EN_Card_Data.csv data/simulation/JP_Card_Data.csv
```

提取器会检查 competition-only license、source commit、文件 hash、card/attack 数量和
Docker image ID；任何不一致都应 fail closed，不能跳过后继续训练。

## 6. Docker 中编译 CUDA runtime

在 Windows PowerShell 中进入一个临时 devel 容器：

```powershell
docker run --rm -it --gpus all `
  -v "${PWD}:/workspace" `
  -w /workspace `
  pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel bash
```

在容器内安装构建工具并编译 RTX 3060 目标：

```bash
python -m pip install cmake ninja

cmake -S engine_cuda -B engine_cuda/build -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build engine_cuda/build --parallel

./engine_cuda/build/ptcg_cuda_smoke \
  --envs 4096 --steps 1000 --policies 12
```

编译带 PyTorch 的官方 runtime extension：

```bash
TORCH_DIR="$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")')"

cmake -S engine_cuda -B engine_cuda/build/torch_official -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPTCG_CUDA_BUILD_TORCH=ON \
  -DTorch_DIR="$TORCH_DIR"
cmake --build engine_cuda/build/torch_official --parallel

PYTHONPATH=engine_cuda/build/torch_official:engine_cuda/python \
python engine_cuda/tools/smoke_torch_extension.py \
  --extension-dir engine_cuda/build/torch_official \
  --batch 4096 --steps 100 --policies 12
```

`OfficialStatePod` ABI v6 每个 state 为 119,936 bytes。runtime 至少需要 32 KiB
device stack；RTX 3060/WDDM 上 stack reservation 会带来额外显存占用。扩大 batch 前应按
512、1,024、2,048、4,096 逐级观察 `nvidia-smi`，不要直接假定服务器上的最佳 batch
适用于 6 GiB 3060。

## 7. Docker CUDA smoke

以下 runner 会自行启动 Docker，规则包和仓库都以只读方式挂载，build/artifact 目录单独
可写：

```powershell
python engine_cuda/tools/run_official_turn_flow_paired.py

python engine_cuda/tools/run_official_semantic_micro_fixture.py
```

turn-flow 结果至少应满足：

```text
passed = true
state_abi = 6
state_bytes = 119936
state_mismatches = 0
status_mismatches = 0
cpu_digest = gpu_digest
```

默认 runner 为 RTX 3060 编译 `sm_86`。在其他 architecture 上运行前，应同步调整工具的
nvcc architecture 或使用对应 CMake 构建，不要依赖 JIT 猜测。

## 8. 0022 与 Frozen51 牌组验证

### 8.1 0022 的 40 牌组

先跑两个 case 的快速检查：

```powershell
python engine_cuda/tools/run_0022_deck40_official_parity.py `
  --pair-mode focal `
  --case-limit 2 `
  --seed-count 1
```

复现 800 局 mirror gate：

```powershell
python engine_cuda/tools/run_0022_deck40_official_parity.py `
  --pair-mode mirrors `
  --seed-start 1 `
  --seed-count 20
```

复现 1,560 个非镜像 ordered matchup：

```powershell
python engine_cuda/tools/run_0022_deck40_official_parity.py `
  --pair-mode all `
  --seed-start 1 `
  --seed-count 1
```

若需要审计 effect branch 覆盖率，加 `--branch-coverage`。若使用自定义牌组目录，显式传
`--deck-root <path>`；目录必须遵守同一 40-deck manifest/hash 合同。

默认输出为：

```text
engine_cuda/artifacts/0022_deck40_official_parity.json
```

验收时检查顶层 `passed=true`，并确认以下字段全部为 0：

- `state_mismatches`：CPU POD 与 CUDA raw state 不同；
- `status_mismatches`：flow/yield/terminal status 不同；
- `outcome_mismatches`：最终 player0/player1/draw 结果不同；
- `unfinished_battles`：达到 decision limit 仍未终局。

`draws` 可以非 0。它是官方规则结果，不是 parity 失败。

### 8.2 Frozen51 的 Docker CUDA 全矩阵

先生成冻结牌组的私有 matchup manifest，不运行对局：

```powershell
python engine_cuda/tools/run_0022_deck40_official_parity.py `
  --deck-root evaluation/arena/frozen `
  --pair-mode all `
  --seed-start 1 `
  --seed-count 1 `
  --decision-limit 512 `
  --policy coverage-first-legal `
  --skip-setup `
  --skip-battle `
  --output engine_cuda/artifacts/frozen51_manifest_summary.json
```

然后在一个 persistent Docker GPU 容器中编译并运行全部 51 x 50 个非镜像有序 matchup：

```powershell
python engine_cuda/tools/run_official_battle_ordered_matrix_cuda.py `
  --manifest engine_cuda/generated/private/official_3aaeaa92/frozen51_all_s1_1/manifest.json `
  --seed-start 1 `
  --seed-count 1 `
  --decision-limit 512 `
  --policy coverage-first-legal `
  --persistent-container `
  --nvcc-threads 1 `
  --output engine_cuda/artifacts/frozen51_all_ordered_s1.json
```

RTX 3060/Windows Docker 建议使用 `--nvcc-threads 1` 控制主机内存峰值。代码未变化时可加
`--skip-persistent-build` 复用已有二进制；修改任何 CUDA header 后必须重新编译，不能复用旧 runner。

最终报告必须满足：`completed_cases=2550`、`decisions_compared=545094`、
`state_mismatches=0`、`status_mismatches=0`、`outcome_mismatches=0`、
`unfinished_battles=0`。本次报告包含 `draws=13`，与官方 CPU 引擎逐场一致。

生成可提交的 Frozen51 支持摘要（原始 matrix artifact 仍保持 Git ignore）：

```powershell
python engine_cuda/tools/audit_frozen51_support.py `
  --evidence engine_cuda/artifacts/frozen51_all_ordered_s1.json
```

## 9. Python/PyTorch runtime 接口

构建 extension 后：

```python
from pathlib import Path

from ptcg_cuda_engine.native import create_official_engine

rules = Path(
    "engine_cuda/generated/private/official_3aaeaa92/official_rules.bin"
).read_bytes()

engine = create_official_engine(
    rules,
    batch_size=4096,
    device_index=0,
    device_stack_bytes=32 * 1024,
)
```

生产热路径的顺序是：

```text
reset_states/reset_seeded_interactive
  -> classify/advance_to_decision
  -> encode_policy_v1
  -> shared-foundation super-batch forward
  -> pack_actions/apply_packed_actions
  -> repeat until terminal
```

`state_bytes()`、`statuses()`、`game_results()`、`action_bytes()` 和
`encode_policy_v1()` 返回的都是由 engine 持有的 CUDA view。使用这些 view 时必须保持
`engine` 存活。正式 rollout 热路径中不要调用 `.cpu()`、`.item()` 或按 lane 读取 route
count；host 读取只应发生在 reset、checkpoint、训练 batch 导出和显式 metrics/parity 点。

PPO 可以只优化 learner decoder/value head，并让所有 ready lanes 共享一次 foundation
encoder super-batch forward。对手与 learner 通过 `policy_id`/static deck fields 分行注入；
不要为每个 opponent 复制完整 foundation model。

## 10. 常见问题

### 找不到 private rule pack

runner 会报告 `private rule pack does not exist`。按第 5 节生成或放入
`engine_cuda/generated/private/official_3aaeaa92/`；不要把该文件 `git add -f`。

### Docker 中找不到源码

命令必须从仓库根目录运行。目标目录名是 `engine_cuda`，工具中的容器路径也是
`/workspace/engine_cuda`；不要用早期原型分支的目录名。

### CUDA OOM

先减小 environment batch 和 PPO rollout retention。`OfficialStatePod`、device stack、
模型 activation、optimizer 和 rollout buffer 都占显存，只看模型参数量会低估总占用。

### CPU/CUDA 最终胜负相同但 state mismatch

仍然视为失败。目标是逐 decision 的语义、状态、RNG/continuation 和最终 outcome 同时一致，
不能只用相同胜负掩盖中间状态分叉。
