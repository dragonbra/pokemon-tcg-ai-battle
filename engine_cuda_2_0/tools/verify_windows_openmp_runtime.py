from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that the Windows CUDA Python environment loads one OpenMP runtime."
    )
    parser.add_argument(
        "--order",
        choices=("numpy-first", "torch-first"),
        default="numpy-first",
    )
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def loaded_module_paths() -> list[Path]:
    if sys.platform != "win32":
        return []

    from ctypes import wintypes

    process = ctypes.windll.kernel32.GetCurrentProcess()
    modules = (wintypes.HMODULE * 4096)()
    required_bytes = wintypes.DWORD()
    enum_modules = ctypes.windll.psapi.EnumProcessModulesEx
    enum_modules.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    )
    enum_modules.restype = wintypes.BOOL
    if not enum_modules(
        process,
        modules,
        ctypes.sizeof(modules),
        ctypes.byref(required_bytes),
        0x03,
    ):
        raise ctypes.WinError()

    module_count = min(
        required_bytes.value // ctypes.sizeof(wintypes.HMODULE),
        len(modules),
    )
    get_name = ctypes.windll.psapi.GetModuleFileNameExW
    get_name.argtypes = (
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    )
    get_name.restype = wintypes.DWORD
    paths: list[Path] = []
    for module in modules[:module_count]:
        buffer = ctypes.create_unicode_buffer(32768)
        if get_name(process, module, buffer, len(buffer)):
            paths.append(Path(buffer.value).resolve())
    return paths


def main() -> int:
    args = parse_args()
    if sys.platform != "win32":
        raise SystemExit("this verifier is for native Windows Python")
    if os.environ.get("KMP_DUPLICATE_LIB_OK"):
        raise RuntimeError("KMP_DUPLICATE_LIB_OK must not be used as an OpenMP fix")

    if args.order == "numpy-first":
        import numpy
        import torch
    else:
        import torch
        import numpy

    numpy_path = Path(numpy.__file__).resolve()
    environment = Path(sys.prefix).resolve()
    if not numpy_path.is_relative_to(environment):
        raise RuntimeError(
            f"NumPy is not installed inside the project environment: {numpy_path}"
        )

    config = numpy.show_config(mode="dicts")
    blas_name = str(config["Build Dependencies"]["blas"]["name"])
    if "openblas" not in blas_name.lower():
        raise RuntimeError(f"expected the PyPI OpenBLAS NumPy wheel, got {blas_name}")

    iomp_paths = sorted(
        {
            str(path)
            for path in loaded_module_paths()
            if path.name.lower() == "libiomp5md.dll"
        }
    )
    if len(iomp_paths) != 1:
        raise RuntimeError(
            f"expected exactly one loaded libiomp5md.dll, got {iomp_paths}"
        )
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    result = {
        "passed": True,
        "import_order": args.order,
        "python": sys.version.split()[0],
        "environment": str(environment),
        "numpy": numpy.__version__,
        "numpy_blas": blas_name,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "loaded_libiomp5md": iomp_paths,
        "unsafe_duplicate_override": False,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
