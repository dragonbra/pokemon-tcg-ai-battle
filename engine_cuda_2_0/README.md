# Pokemon TCG CUDA Engine

`engine_cuda_2_0/` is the GPU-resident Pokemon TCG battle engine and differential
validation suite used by this repository. It includes the CUDA/POD runtime,
PyTorch bindings, Docker runners, the audited 0022 40-deck fixture, and CPU vs
CUDA semantic/outcome parity tests.

For installation, private rule-pack preparation, Docker commands, Python API,
and Frozen51 validation, read [docs/usage_guide_zh.md](docs/usage_guide_zh.md).

## Engine CUDA 2.0 checkpoint

This directory is the isolated CUDA 2.0 copy for branch
`codex/cortex-cpu-test-checkpoint-engine-cuda-2.0`; the original
`engine_cuda/` tree is not used or modified by its build and evaluation paths.
The 2.0 Frozen evaluator uses the 65-deck/256-slot
`0812_top100_top500_stadiums_mill_plus_limitless_dragapult_v3` pool and the
branch-provided compact 0809 CPU checkpoint.  Its matching portable artifact
is FP16 on disk and materialized as FP32 at runtime.  The pool includes deck
056, Limitless self-destruct Dragapult (057), eight Top-500 additions
(058-065), Gravity Mountain, Lumiose City, Great Tusk / Mill, and Teal Mask
Ogerpon deck 006.  CPU/POD/CUDA semantic validation is run independently from
the checkpoint evaluation.

The 2026-08-12 Frozen65 server matrix covers every unordered deck pair with
50 seeds: 104,000 battles and 20,558,750 compared decisions with zero
CPU/POD/CUDA mismatch. See
[`docs/frozen65_server_parity_20260812.md`](docs/frozen65_server_parity_20260812.md).

## Current Validation Snapshot

Validated through 2026-08-07 in Docker with an RTX 3060 Laptop GPU:

- production `OfficialDeviceArena` turn-flow smoke: passed, zero state/status mismatch;
- 40 mirror matchups: 800 battles, 165,623 decisions, zero mismatch, 13 official draws;
- all 1,560 non-mirror ordered matchups: 319,574 decisions, zero mismatch, 9 official draws;
- targeted high-risk decks: 1,500 battles, 312,530 decisions, zero mismatch;
- Frozen51 non-mirror ordered matrix: 2,550 battles, 545,094 decisions,
  zero state/status/outcome mismatch, 13 official draws;
- Frozen51 mirror matrix: 153 battles across 51 decks and 3 additional seeds
  under `coverage-random-legal`, 31,350 decisions, zero state/status/outcome
  mismatch, 2 official draws and zero unfinished battles;
- semantic micro fixture: 163 checks passed;
- focused Python regression tests: 3/3 passed;
- compact 0031 semantic bridge: exact archive/checkpoint loaded on the RTX 3060,
  31 real battles finished with 1,186 model decisions, zero inference fallback
  and zero engine error; TF32 throughput was 99.29 total decisions/s at batch 8.
- shared-prototype 0806 FP32 checkpoint: 22 real battles finished in a batch-8,
  256-step smoke with 2,048 decisions, zero inference fallback and zero engine
  error; throughput was 1.14 finished games/s on the RTX 3060 Laptop GPU.
- fixed-seed/action CPU `CausalKnowledge` vs CUDA semantic lockstep: 180/180
  decisions exact for integer, float, mask and relation tensors; zero CUDA state
  errors, zero logits tolerance failures and zero greedy-action divergences;
  maximum FP32 logits error was `7.152557373046875e-06`.
- causal-history wrap gate: 156 compared decisions after the 64-event ring had
  wrapped, with a maximum cumulative event count of 344 and zero mismatch.
- semantic bridge and policy-pool focused CUDA regression: 22/22 tests passed;
  shared and single-policy semantic routes materialize only actual ready lanes.

These results cover the audited 0022 and Frozen51 deck catalogs under the
recorded deterministic and random-legal parity policies, including Frozen51
mirrors and additional seeds. They do not prove every possible card interaction,
future deck, seed, or stochastic policy trajectory. The unmodified official CPU
engine remains the oracle.

## Frozen51 Acceptance

The 2026-08-06 fail-closed audit of `evaluation/arena/frozen/` admits all 51
exact decks for the tested engine-semantic contract. The complete non-mirror
ordered matrix covers all 51 x 50 seat-ordered pairs and compares 545,094
decisions over 2,550 battles with zero CPU/POD/CUDA state, status, or outcome
mismatch. The 13 draws are official engine results and remain draws on CUDA.

The durable report is
`docs/evaluation/combat_mat/0031_latest_frozen_test/cuda_support.html`, with
machine-readable evidence in the adjacent `cuda_support.json`. This admission
is finite trajectory evidence for engine semantics; model observation and
training compatibility are separate contracts.

## Compact 0031 Semantic Model

`python/ptcg_cuda_engine/semantic0031_bridge.py` loads the committed compact
0031 archive, verifies its archive/checkpoint/deck commitments, expands FP16
storage into the required FP32 runtime model, and keeps model/prototype tensors
resident on CUDA. `tools/run_0031_semantic_cuda_smoke.py` is the end-to-end
official-engine gate.

The independent `semantic0031_codec_v2` path materializes only routed ready lanes
directly from `OfficialDeviceArena`; padding and empty cohorts do not enter the
codec or shared-foundation forward. It carries per-card serial identity, option
context/effect-card relations, and a 64-event arena-owned SoA causal ring. All
24 official log types have audited CUDA hook and tensor mappings. The legacy
`policy_codec_v1_to_semantic0031_v2` compatibility path still represents facts
that its source codec does not carry as `UNKNOWN` or a false sequence mask.

The fixed-seed/action CPU canonical lockstep gate now passes across the complete
180-decision trace. Integer, float, mask and relation tensors are exact; the
64-event ring is compared after wrap; FP32 logits remain within tolerance and
greedy actions agree. This is finite trajectory evidence rather than an
exhaustive proof of every future deck, seed or stochastic policy trajectory.
The official engine transition and outcome contract is unchanged.

The loader also accepts the compact
`0031_compact_shared_prototype_fp32_v1` checkpoint schema. One canonical
prototype encoder copy is restored as the state/option aliases without FP16
conversion. `tools/run_0031_semantic_cuda_smoke.py --checkpoint PATH` runs that
checkpoint through complete official CUDA battles.

## Repository Boundary

The following are source-controlled:

- CUDA kernels, POD state and rule interpreter;
- host/PyTorch runtime bindings;
- public deck fixtures and tests;
- Docker build and parity runners.

The following must remain local and are ignored by Git:

- `engine_cuda_2_0/generated/private/` official rule packs and extraction output;
- `engine_cuda_2_0/build/` compiled binaries/extensions;
- `engine_cuda_2_0/artifacts/` benchmark and parity reports;
- checkpoints, W&B staging, traces, and official generated fixtures.

Do not modify `engine/source/`. The tools mount it read-only and compare CUDA
behavior against that official implementation.

## Minimal Checks

```bash
python -m unittest discover -s engine_cuda_2_0/tests -p "test_*.py" -v
python engine_cuda_2_0/tools/compile_rule_pack.py \
  engine_cuda_2_0/rules/smoke_rules.json --check
```

After preparing the private rule pack and confirming Docker GPU access:

```bash
python engine_cuda_2_0/tools/run_official_turn_flow_paired.py

python engine_cuda_2_0/tools/run_0022_deck40_official_parity.py \
  --pair-mode mirrors --seed-count 20
```

The default CUDA architecture is SM 86 for RTX 3060. Set
`CMAKE_CUDA_ARCHITECTURES` explicitly when building for another GPU.
