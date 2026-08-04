# Pokemon TCG CUDA Engine

`engine_cuda/` is the GPU-resident Pokemon TCG battle engine and differential
validation suite used by this repository. It includes the CUDA/POD runtime,
PyTorch bindings, Docker runners, the audited 0022 40-deck fixture, and CPU vs
CUDA semantic/outcome parity tests.

For installation, private rule-pack preparation, Docker commands, Python API,
and 40-deck validation, read [docs/usage_guide_zh.md](docs/usage_guide_zh.md).

## Current Validation Snapshot

Validated on 2026-08-04 in Docker with an RTX 3060 Laptop GPU:

- production `OfficialDeviceArena` turn-flow smoke: passed, zero state/status mismatch;
- 40 mirror matchups: 800 battles, 165,623 decisions, zero mismatch, 13 official draws;
- all 1,560 non-mirror ordered matchups: 319,574 decisions, zero mismatch, 9 official draws;
- targeted high-risk decks: 1,500 battles, 312,530 decisions, zero mismatch;
- semantic micro fixture: 154 checks passed;
- focused Python regression tests: 3/3 passed.

These results cover the audited 0022 deck catalog and deterministic parity
policies. They do not prove every possible card interaction or stochastic
policy trajectory. The unmodified official CPU engine remains the oracle.

## Repository Boundary

The following are source-controlled:

- CUDA kernels, POD state and rule interpreter;
- host/PyTorch runtime bindings;
- public deck fixtures and tests;
- Docker build and parity runners.

The following must remain local and are ignored by Git:

- `engine_cuda/generated/private/` official rule packs and extraction output;
- `engine_cuda/build/` compiled binaries/extensions;
- `engine_cuda/artifacts/` benchmark and parity reports;
- checkpoints, W&B staging, traces, and official generated fixtures.

Do not modify `engine/source/`. The tools mount it read-only and compare CUDA
behavior against that official implementation.

## Minimal Checks

```bash
python -m unittest discover -s engine_cuda/tests -p "test_*.py" -v
python engine_cuda/tools/compile_rule_pack.py \
  engine_cuda/rules/smoke_rules.json --check
```

After preparing the private rule pack and confirming Docker GPU access:

```bash
python engine_cuda/tools/run_official_turn_flow_paired.py

python engine_cuda/tools/run_0022_deck40_official_parity.py \
  --pair-mode mirrors --seed-count 20
```

The default CUDA architecture is SM 86 for RTX 3060. Set
`CMAKE_CUDA_ARCHITECTURES` explicitly when building for another GPU.
