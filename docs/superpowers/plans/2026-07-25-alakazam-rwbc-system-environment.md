# Alakazam RWBC System Training Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 直接向 `/usr/local/bin/python` 的真实系统环境安装并验证 `train/project_0011_alakazam_sota_reward_weighted_bc` 所需依赖，使 RTX 5080 上的 PyTorch CUDA 训练链路和项目 smoke test 可运行。

**Architecture:** 不创建 venv、Conda 或 Docker 环境；所有包均由 `/usr/local/bin/python -m pip` 安装到该解释器的系统级 site-packages。实施过程先审计共享环境，再安装支持 Blackwell 的官方 PyTorch CUDA wheel 和仓库 RL extra，最后通过依赖一致性、真实 GPU forward/backward/AMP、项目单测与编译检查四层验证。

**Tech Stack:** Python 3.11.15、pip 26.1.2、PyTorch 官方 CUDA 12.8 wheel、NumPy `<2`、TensorBoard `>=2.14`、unittest、NVIDIA RTX 5080 / WSL2。

## Global Constraints

- 不创建 `.venv`、Conda 环境、Docker 环境或任何额外 Python 环境。
- 所有安装和验证命令必须使用 `/usr/local/bin/python`；pip 必须通过 `/usr/local/bin/python -m pip` 调用。
- PyTorch 必须来自官方索引 `https://download.pytorch.org/whl/cu128`，以获得 RTX 5080 / Blackwell 所需的 CUDA 12.8 wheel 支持。
- 不安装完整 CUDA Toolkit 或 `nvcc`；本项目没有自定义 CUDA 扩展。
- 安装前后必须保存系统 Python 的版本、site-packages、`pip freeze` 和 `pip check` 证据。
- 不自动卸载安装前存在的未知包；若依赖冲突，先报告具体冲突和版本变化。
- 不修改历史 V1–V5 配置、artifact、checkpoint、TensorBoard 或 submission 资产。
- 不启动全量训练，不构建缺失的真实 0011 dataset，不执行 Kaggle 提交、上传或其他外部可见操作。
- 不安装 Anthropic/Claude SDK；训练项目没有 Claude 运行时依赖。
- 设计文档 `docs/superpowers/specs/2026-07-25-alakazam-rwbc-training-environment-design.md` 是本计划的需求来源。

## File and State Map

**Created runtime state:**
- System package metadata and modules under `/usr/local/lib/python3.11/dist-packages/`.
- Editable-install metadata in system site-packages pointing to `/home/cyd/repos/pokemon-tcg-ai-battle`.
- pip download cache under the current user's standard cache directory.

**Created audit files:**
- `/tmp/pokemon-tcg-ai-battle-python-env-before.txt` — pre-install interpreter, site paths, package snapshot, dependency check, and GPU driver summary.
- `/tmp/pokemon-tcg-ai-battle-python-env-after.txt` — post-install versions, package snapshot, dependency check, and CUDA capability summary.
- `/tmp/pokemon-tcg-ai-battle-gpu-smoke.py` — temporary deterministic CUDA/AMP/backward smoke program.

**Repository files:**
- No production or test source files are planned for modification.
- The approved spec and this plan are documentation-only repository additions.

---

### Task 1: Capture and Validate the Shared System Python Baseline

**Files:**
- Create: `/tmp/pokemon-tcg-ai-battle-python-env-before.txt`
- Inspect: `/usr/local/bin/python`
- Inspect: `/usr/local/lib/python3.11/dist-packages/`

**Interfaces:**
- Consumes: Current machine state and `/usr/local/bin/python`.
- Produces: A complete pre-install snapshot used to identify package changes or conflicts after installation.

- [ ] **Step 1: Confirm that no virtual environment is active and that the target interpreter is exact**

Run:

```bash
/usr/local/bin/python - <<'PY'
import os
import site
import sys

assert sys.executable == "/usr/local/bin/python", sys.executable
assert not os.environ.get("VIRTUAL_ENV"), os.environ.get("VIRTUAL_ENV")
assert not os.environ.get("CONDA_PREFIX"), os.environ.get("CONDA_PREFIX")
print("executable:", sys.executable)
print("version:", sys.version.replace("\n", " "))
print("prefix:", sys.prefix)
print("base_prefix:", sys.base_prefix)
print("site_packages:", site.getsitepackages())
print("user_site:", site.getusersitepackages())
PY
```

Expected: executable is `/usr/local/bin/python`; `VIRTUAL_ENV` and `CONDA_PREFIX` assertions pass; system site-packages includes `/usr/local/lib/python3.11/dist-packages`.

- [ ] **Step 2: Verify pip targets the same interpreter and inspect write access**

Run:

```bash
/usr/local/bin/python -m pip --version
/usr/local/bin/python - <<'PY'
from pathlib import Path
import site

for value in site.getsitepackages():
    path = Path(value)
    print(path, "exists=", path.exists(), "writable=", path.exists() and path.is_dir())
PY
```

Expected: pip reports Python 3.11 under `/usr/local/lib/python3.11/dist-packages`. If the later install returns `Permission denied`, stop and report that `/usr/local` requires elevated OS permission; do not redirect into user-site because that would violate the selected target environment.

- [ ] **Step 3: Write the pre-install audit snapshot**

Run:

```bash
{
  printf '%s\n' '=== timestamp ==='
  date --iso-8601=seconds
  printf '%s\n' '=== python ==='
  /usr/local/bin/python -VV
  /usr/local/bin/python -c 'import sys, site; print(sys.executable); print(sys.prefix); print(site.getsitepackages()); print(site.getusersitepackages())'
  printf '%s\n' '=== pip ==='
  /usr/local/bin/python -m pip --version
  printf '%s\n' '=== pip freeze ==='
  /usr/local/bin/python -m pip freeze --all
  printf '%s\n' '=== pip check ==='
  /usr/local/bin/python -m pip check
  printf '%s\n' '=== nvidia ==='
  /usr/lib/wsl/lib/nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
} > /tmp/pokemon-tcg-ai-battle-python-env-before.txt
```

Expected: file exists and includes Python, pip, installed packages, dependency status, and RTX 5080 driver evidence. A pre-existing `pip check` failure is not caused by this task; record it before deciding whether installation can proceed.

- [ ] **Step 4: Confirm no prior torch installation is silently present**

Run:

```bash
/usr/local/bin/python - <<'PY'
from importlib.util import find_spec

for name in ("torch", "numpy", "tensorboard"):
    spec = find_spec(name)
    print(name, spec.origin if spec else "NOT_INSTALLED")
PY
```

Expected from machine inventory: `torch NOT_INSTALLED`. Record any different result before replacing it.

---

### Task 2: Install the Official Blackwell-Compatible PyTorch CUDA Wheel

**Files:**
- Modify state: `/usr/local/lib/python3.11/dist-packages/`
- Read: `/tmp/pokemon-tcg-ai-battle-python-env-before.txt`

**Interfaces:**
- Consumes: Writable `/usr/local` system Python and NVIDIA driver 596.49.
- Produces: Importable CUDA-enabled `torch` from the official CUDA 12.8 wheel index.

- [ ] **Step 1: Inspect the exact official CUDA 12.8 candidate selected by pip without installing**

Run:

```bash
/usr/local/bin/python -m pip install \
  --dry-run \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch
```

Expected: resolver selects a Linux CPython 3.11 CUDA 12.8 build of `torch`, not a `+cpu` build. Save the selected version in the session log. If no compatible CPython 3.11 wheel exists, stop instead of switching to an unofficial index.

- [ ] **Step 2: Install torch from the official CUDA 12.8 index**

Run:

```bash
/usr/local/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch
```

Expected: installation succeeds in the `/usr/local/bin/python` environment and installs the corresponding NVIDIA CUDA runtime wheels. Do not add `--user`, `--break-system-packages`, or a venv.

- [ ] **Step 3: Verify package provenance and CUDA build metadata before project installation**

Run:

```bash
/usr/local/bin/python - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch_file:", torch.__file__)
print("torch_cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
assert sys.executable == "/usr/local/bin/python"
assert torch.version.cuda is not None, "CPU-only torch wheel was installed"
assert torch.cuda.is_available(), "CUDA runtime is installed but GPU is unavailable"
PY
```

Expected: torch resides under system site-packages, reports a CUDA runtime, and sees the GPU.

---

### Task 3: Install the Repository and RL Dependencies into System Python

**Files:**
- Read: `pyproject.toml:1-19`
- Modify state: `/usr/local/lib/python3.11/dist-packages/`
- Create state: editable package reference to `/home/cyd/repos/pokemon-tcg-ai-battle`

**Interfaces:**
- Consumes: Installed CUDA-enabled torch and repository `pyproject.toml`.
- Produces: Editable `pokemon-tcg-ai-battle` package plus `numpy<2`, `tensorboard>=2.14`, and declared Kaggle/runtime packages in the same system interpreter.

- [ ] **Step 1: Preview dependency resolution against the already installed torch**

Run from `/home/cyd/repos/pokemon-tcg-ai-battle`:

```bash
/usr/local/bin/python -m pip install --dry-run -e '.[rl]'
```

Expected: editable project plus missing dependencies are listed; installed CUDA torch satisfies `torch>=2.2`. If the resolver proposes replacing torch with a CPU build, stop and install the project with the next step's `--extra-index-url` so CUDA candidates remain visible.

- [ ] **Step 2: Install the editable project and RL extra with the official CUDA index available**

Run from `/home/cyd/repos/pokemon-tcg-ai-battle`:

```bash
/usr/local/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e '.[rl]'
```

Expected: editable install succeeds, `numpy` resolves below 2.0, TensorBoard is at least 2.14, and the CUDA-enabled torch remains installed.

- [ ] **Step 3: Verify dependency consistency**

Run:

```bash
/usr/local/bin/python -m pip check
```

Expected: `No broken requirements found.` If any conflict appears, compare it with `/tmp/pokemon-tcg-ai-battle-python-env-before.txt`; do not remove unrelated packages automatically.

- [ ] **Step 4: Verify imports, paths, and declared version constraints**

Run from `/home/cyd/repos/pokemon-tcg-ai-battle`:

```bash
/usr/local/bin/python - <<'PY'
import sys
import numpy
import tensorboard
import torch
import rl_environment
import train.alakazam_sota_reward_weighted_bc

print("python:", sys.executable)
print("numpy:", numpy.__version__, numpy.__file__)
print("tensorboard:", tensorboard.__version__, tensorboard.__file__)
print("torch:", torch.__version__, torch.__file__)
print("torch_cuda_runtime:", torch.version.cuda)
print("rl_environment:", rl_environment.__file__)
print("rwbc:", train.alakazam_sota_reward_weighted_bc.__file__)
assert sys.executable == "/usr/local/bin/python"
assert int(numpy.__version__.split(".", 1)[0]) < 2
assert torch.cuda.is_available()
PY
```

Expected: all imports succeed from the target interpreter; project modules resolve to this repository through the editable install.

---

### Task 4: Execute a Real RTX 5080 CUDA, AMP, Backward, and Optimizer Smoke Test

**Files:**
- Create: `/tmp/pokemon-tcg-ai-battle-gpu-smoke.py`
- Modify state: GPU memory only for the duration of the test

**Interfaces:**
- Consumes: System-installed CUDA-enabled PyTorch.
- Produces: Evidence that RTX 5080 kernels, autocast, gradients, and AdamW updates execute successfully and synchronously.

- [ ] **Step 1: Create the deterministic GPU smoke program**

Write `/tmp/pokemon-tcg-ai-battle-gpu-smoke.py` with exactly:

```python
import sys

import torch


def main() -> None:
    assert sys.executable == "/usr/local/bin/python", sys.executable
    assert torch.cuda.is_available(), "CUDA is unavailable"

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    properties = torch.cuda.get_device_properties(device)

    print("torch", torch.__version__)
    print("torch_cuda_runtime", torch.version.cuda)
    print("device", name)
    print("compute_capability", capability)
    print("total_memory_bytes", properties.total_memory)

    torch.manual_seed(20260725)
    torch.cuda.manual_seed_all(20260725)

    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 32),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(128, 256, device=device)
    targets = torch.randn(128, 32, device=device)

    before = model[0].weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()

    product = torch.randn(512, 512, device=device) @ torch.randn(512, 512, device=device)
    torch.cuda.synchronize()

    assert torch.isfinite(loss).item(), loss
    assert torch.isfinite(product).all().item()
    assert not torch.equal(before, model[0].weight.detach())
    print("loss", float(loss.detach().cpu()))
    print("gpu_smoke", "PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the GPU smoke program**

Run:

```bash
CUDA_LAUNCH_BLOCKING=1 /usr/local/bin/python /tmp/pokemon-tcg-ai-battle-gpu-smoke.py
```

Expected: device is NVIDIA GeForce RTX 5080, compute capability is printed, finite loss is printed, and final line is `gpu_smoke PASS`. `CUDA_LAUNCH_BLOCKING=1` ensures kernel failures surface at the responsible operation.

- [ ] **Step 3: Classify any failure before changing packages**

Use this decision table:

| Failure | Classification | Next action |
|---|---|---|
| `torch.cuda.is_available()` false | Driver/runtime discovery | Record `torch.version.cuda`, `nvidia-smi`, and library error; do not install Toolkit blindly |
| `no kernel image is available` | Wheel lacks Blackwell SASS/PTX | Confirm wheel came from cu128 official index; replace only with a newer official CUDA wheel |
| `libcuda.so` or driver error | WSL driver bridge | Verify `/usr/lib/wsl/lib` and Windows NVIDIA driver before touching Python packages |
| CUDA OOM in this small test | External GPU usage or runtime fault | Record `nvidia-smi`; do not lower the smoke workload as a first response |
| AMP dtype/operator error | PyTorch build/operator compatibility | Reproduce without AMP only to isolate; AMP remains required for final PASS |

Expected: no package change occurs until the failure class is supported by evidence.

---

### Task 5: Run Project-Level CPU Smoke and Static Compilation Checks

**Files:**
- Test: `tests/test_alakazam_sota_reward_weighted_bc.py`
- Read: `train/project_0011_alakazam_sota_reward_weighted_bc/`
- Read: `rl_environment/`
- No planned repository modifications

**Interfaces:**
- Consumes: Fully installed system Python environment.
- Produces: Project-specific evidence for dataset join, reward weighting, small-model CPU training, checkpoint creation, and syntax/import compilation.

- [ ] **Step 1: Run the dedicated reward-weighted BC test module**

Run from `/home/cyd/repos/pokemon-tcg-ai-battle`:

```bash
/usr/local/bin/python -m unittest -v tests.test_alakazam_sota_reward_weighted_bc
```

Expected: all tests pass, including temporary dataset join, weighted loss, three-epoch CPU training, and best checkpoint generation. If the test fails, invoke systematic-debugging before editing code or changing dependency versions.

- [ ] **Step 2: Compile the training and shared RL packages**

Run from `/home/cyd/repos/pokemon-tcg-ai-battle`:

```bash
/usr/local/bin/python -m compileall -q rl_environment train
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Verify the real training CLI loads without starting training**

Run from `/home/cyd/repos/pokemon-tcg-ai-battle`:

```bash
/usr/local/bin/python -m train.alakazam_sota_reward_weighted_bc --help
```

Expected: help lists `build-dataset`, `train`, and `export-candidate`; no training or data mutation occurs.

- [ ] **Step 4: Confirm real datasets are still an explicit external handoff**

Run from `/home/cyd/repos/pokemon-tcg-ai-battle`:

```bash
/usr/local/bin/python - <<'PY'
from pathlib import Path

paths = [
    Path("rl_runs/dataset/0010-alakazam_sota_model"),
    Path("rl_runs/dataset/0009-reward_weighted_bc"),
]
for path in paths:
    print(path, "READY" if path.exists() else "PENDING_EXTERNAL_HANDOFF")
PY
```

Expected for this session: missing paths are reported as `PENDING_EXTERNAL_HANDOFF`. Do not synthesize or download replacements in this implementation.

---

### Task 6: Capture Final Environment Evidence and Report the Boundary

**Files:**
- Read: `/tmp/pokemon-tcg-ai-battle-python-env-before.txt`
- Create: `/tmp/pokemon-tcg-ai-battle-python-env-after.txt`
- Read: `/tmp/pokemon-tcg-ai-battle-gpu-smoke.py`

**Interfaces:**
- Consumes: Completed installation and all validation results.
- Produces: Auditable before/after package state and a truthful completion report that separates environment readiness from missing real data.

- [ ] **Step 1: Write the post-install audit snapshot**

Run:

```bash
{
  printf '%s\n' '=== timestamp ==='
  date --iso-8601=seconds
  printf '%s\n' '=== python ==='
  /usr/local/bin/python -VV
  /usr/local/bin/python -c 'import sys, site; print(sys.executable); print(sys.prefix); print(site.getsitepackages()); print(site.getusersitepackages())'
  printf '%s\n' '=== core versions ==='
  /usr/local/bin/python -c 'import numpy, tensorboard, torch; print("numpy", numpy.__version__); print("tensorboard", tensorboard.__version__); print("torch", torch.__version__); print("torch_cuda_runtime", torch.version.cuda); print("cuda_available", torch.cuda.is_available()); print("device", torch.cuda.get_device_name(0)); print("capability", torch.cuda.get_device_capability(0))'
  printf '%s\n' '=== pip freeze ==='
  /usr/local/bin/python -m pip freeze --all
  printf '%s\n' '=== pip check ==='
  /usr/local/bin/python -m pip check
} > /tmp/pokemon-tcg-ai-battle-python-env-after.txt
```

Expected: snapshot contains exact installed versions, CUDA runtime, RTX 5080 capability, complete package list, and `No broken requirements found.`

- [ ] **Step 2: Compare package snapshots without mutating the environment**

Run:

```bash
/usr/local/bin/python - <<'PY'
from pathlib import Path

before = Path("/tmp/pokemon-tcg-ai-battle-python-env-before.txt").read_text()
after = Path("/tmp/pokemon-tcg-ai-battle-python-env-after.txt").read_text()
print("before_snapshot_bytes", len(before.encode()))
print("after_snapshot_bytes", len(after.encode()))
print("before_has_torch", "torch==" in before.lower())
print("after_has_torch", "torch==" in after.lower())
PY
```

Expected: both snapshots are readable; before lacks torch according to inventory and after contains torch. The final report should list key added or replaced versions rather than dumping all transitive NVIDIA packages.

- [ ] **Step 3: Check repository status only for unexpected source changes**

Run from `/home/cyd/repos/pokemon-tcg-ai-battle`:

```bash
git status --short
```

Expected: only the approved documentation files created in this session appear. Python installation state is outside Git. If tests generated tracked-file modifications, inspect and report them rather than deleting or reverting without review.

- [ ] **Step 4: Report completion with evidence and explicit exclusions**

The final report must state:

- `/usr/local/bin/python` and exact Python version used.
- Exact torch version and `torch.version.cuda`.
- RTX 5080 device name and compute capability.
- GPU smoke outcome, dedicated unittest outcome, `compileall` outcome, and `pip check` outcome.
- Paths to `/tmp/pokemon-tcg-ai-battle-python-env-before.txt` and `/tmp/pokemon-tcg-ai-battle-python-env-after.txt`.
- Whether 0010 and 0009 real data paths are ready or pending.
- That no venv was created, no full training ran, and no Kaggle external action occurred.
- Any package conflicts, skipped step, or failed command verbatim; never call the environment ready if a required validation failed.

Expected: environment readiness is claimed only if Tasks 2–5 pass completely; real-training readiness remains pending until the external data handoff is complete.
