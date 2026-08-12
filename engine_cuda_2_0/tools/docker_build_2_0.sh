#!/usr/bin/env bash
set -euo pipefail

torch_dir="$(python -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")')"
cmake -S engine_cuda_2_0 -B engine_cuda_2_0/build/torch_official -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPTCG_CUDA_BUILD_TORCH=ON \
  -DTorch_DIR="$torch_dir"
cmake --build engine_cuda_2_0/build/torch_official --parallel

PYTHONPATH=engine_cuda_2_0/build/torch_official:engine_cuda_2_0/python \
python engine_cuda_2_0/tools/smoke_torch_extension.py \
  --extension-dir engine_cuda_2_0/build/torch_official \
  --batch 64 --steps 8 --policies 4
