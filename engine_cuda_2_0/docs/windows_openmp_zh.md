# Windows OpenMP 运行时修复

部分 Windows Anaconda 环境会同时包含两份不同版本的 Intel OpenMP：

- Conda/MKL：`<conda-root>\Library\bin\libiomp5md.dll`
- pip PyTorch：`<python-site-packages>\torch\lib\libiomp5md.dll`

Conda NumPy 会先加载 MKL 的 DLL，PyTorch 随后加载自己打包的 DLL，因此仅执行
`import torch` 也可能报 `OMP Error #15`。`KMP_DUPLICATE_LIB_OK=TRUE` 会掩盖错误，
但可能造成错误结果或崩溃，不能作为修复。

项目采用隔离环境解决：复用现有 CUDA PyTorch，同时在 `.venv-cuda` 中覆盖安装
PyPI NumPy wheel。该 wheel 使用 OpenBLAS，不再加载 Conda 的 Intel OpenMP；进程中
只保留 PyTorch 自带的一份 `libiomp5md.dll`。

在仓库根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File engine_cuda_2_0/tools/setup_windows_cuda_env.ps1 -RequireCuda
.\.venv-cuda\Scripts\Activate.ps1
python -m pytest engine_cuda_2_0/tests -q
```

验证器会分别测试 `numpy -> torch` 和 `torch -> numpy` 两种导入顺序，并检查：

- NumPy 来自 `.venv-cuda` 且使用 OpenBLAS；
- 没有设置 `KMP_DUPLICATE_LIB_OK`；
- 进程只加载一份 `libiomp5md.dll`；
- 使用 `-RequireCuda` 时，PyTorch 能识别 CUDA GPU。

这个修复不修改 Anaconda 全局 DLL，也不会把 MKL 强制为单线程。Docker/Linux CUDA
构建不需要使用该 Windows 环境。
