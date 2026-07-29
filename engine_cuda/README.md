# PTCG CUDA-only engine prototype

> **CUDA-specific experimental subtree.** Everything under `engine_cuda/` is
> intended for the GPU-resident training path. It is not a drop-in replacement
> for the frozen CPU/official engine, and generated builds, benchmark artifacts,
> model checkpoints, and competition-private generated rules are not committed.

This directory is an independent, deck-agnostic prototype for a GPU-resident
Pokemon TCG training environment. It is intentionally separate from the frozen
official/local C++ engine.

The imported manifests under `configs/` are historical source-repository
examples. Paths such as `bc_models/`, `arena_agents/`, and `/root/autodl-tmp`
are not authoritative in this repository and must not be used as the 0020
opponent snapshot. A 0020 manifest must be regenerated from versioned packages
under `evaluation/arena/opponents/` and record their hashes.

The target hot loop is:

```text
device battle state
  -> fixed-capacity routing by policy_id and codec_id
  -> device codec adapter into a reusable cohort buffer
  -> 10+ resident frozen BC models (plus one trainable learner)
  -> device action normalization
  -> device rule interpreter
```

No per-decision JSON, Python observation lists, `.cpu()`, `.item()`, ctypes
`Select`, or host read of route counts is part of the target contract. Host
traffic is allowed at reset, checkpoint load, training batch export, metrics,
and explicit debug/parity checkpoints.

## What works now

- A fixed-layout batched CUDA state with one environment per CUDA thread.
- A numeric opcode interpreter whose behavior is independent of deck identity.
- Device-side PolicyCodecV1-shaped smoke buffers and policy routing.
- Per-player `policy_id`, with a configurable capacity up to 32 policies.
- A standalone CUDA smoke benchmark.
- A deterministic Python reference interpreter for tests without CUDA.
- A heterogeneous policy-pool interface that keeps tensors on the device.
- A model-pool and engine VRAM estimator.
- A versioned JSON rule-pack compiler/validator.

This is a vertical slice, not a claim of full official-engine parity. The smoke
rule pack implements a small generic opcode subset. Full card coverage must be
earned through differential replay and opcode coverage gates described in
[`docs/implementation_plan.md`](docs/implementation_plan.md).

The current strongest local BC packages are heterogeneous in both model and
codec. The example manifest contains legacy ID-only and Marnie prize/compact
codec IDs. It deliberately excludes the standalone Alakazam strategy runtime:
opponents and the learner are BC models only. Those exact device codecs are not
implemented by this smoke slice yet. Production should route first and encode
one fixed cohort at a time so every codec layout is not materialized per
environment.

## Known divergence kept explicit

The frozen local engine can crash when Ninetales card `660`, Supernatural
Shapeshifter, borrows Amarys card `1207` and executes its delayed effect. This
prototype does not silently choose new semantics. That exact path becomes
`KNOWN_DIVERGENCE_660_1207` and terminates the affected environment. It can be
changed only after a fixed official engine is obtained and paired regression is
completed. The frozen `libcg_seeded.so` must not be overwritten.

## Quick checks

CPU-only validation uses only the Python standard library:

```powershell
python -m unittest discover -s engine_cuda/tests -v
python engine_cuda/tools/compile_rule_pack.py `
  engine_cuda/rules/smoke_rules.json `
  --check
python engine_cuda/tools/estimate_memory.py `
  --envs 4096 16384 `
  --manifest engine_cuda/configs/policy_pool.example.json
```

CUDA build (the default architecture is the local RTX 3060 Laptop, SM 86):

```bash
cmake -S engine_cuda -B engine_cuda/build -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build engine_cuda/build
./engine_cuda/build/ptcg_cuda_smoke --envs 16384 --steps 1000 --policies 12
```

Use `-DPTCG_CUDA_BUILD_TORCH=ON` only in an environment where PyTorch and its
matching CUDA development toolkit are both installed.

Some PyTorch images require an explicit config path, for example:

```bash
cmake -S engine_cuda -B engine_cuda/build/torch -G Ninja \
  -DPTCG_CUDA_BUILD_TORCH=ON \
  -DTorch_DIR=/path/to/site-packages/torch/share/cmake/Torch
cmake --build engine_cuda/build/torch --parallel
```

After building the extension, validate zero-copy tensor views and the named
divergence path with:

```bash
PYTHONPATH=engine_cuda/build/torch \
python engine_cuda/tools/smoke_torch_extension.py --batch 4096 --steps 100
```

To load the example pool's 10 actual frozen BC checkpoints plus one actual BC
learner and measure local CUDA residency (including a conservative learner
optimizer-state reserve), run:

```bash
PYTHONPATH=engine_cuda/build/torch \
python engine_cuda/tools/probe_actual_policy_residency.py \
  --batch 4096 --engine-steps 100 --dtype bf16
```

See `docs/acceptance_status_20260729.md` for the distinction between completed
prototype checks and outstanding formal promotion gates.

The reproducible six-BC transition smoke uses
`tools/run_policy_codec_v1_ppo_smoke.sh`. It deliberately runs the seeded CPU
engine with a CUDA FP32 learner and therefore validates PPO/checkpoint plumbing,
not completion of the CUDA battle engine. Validate its resumable checkpoint
with `tools/verify_policy_codec_v1_ppo_smoke.py`.

The `CudaEngine` Python object owns the allocations and must remain alive while
any zero-copy tensor returned by `encode_policy_v1`, `route_ready`, or `digest`
is in use.

## Private engine boundary

Official engine source is competition-use-only. Do not copy `CardImpl.h`,
effect headers, or other official source into this directory or a public
dataset. A future extractor may consume the private source locally and emit a
numeric rule pack under `generated/private/`; that path is ignored here and
must also be excluded by every packaging script. Generated provenance must
record source hashes, extractor version, card/opcode counts, and unsupported
rules.
