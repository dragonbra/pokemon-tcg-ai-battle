# Prototype validation results - 2026-07-29

## Host and toolchain

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB, compute capability 8.6.
- Build: NVIDIA CUDA 13.0.0 devel container, `nvcc -std=c++17 -arch=sm_86`.
- The existing project CUDA runtime image did not contain `nvcc`; it was not
  modified. A separate pinned devel image was used.

## Correctness checks

- 25 CPU/device-aware unit tests passed on the Linux server: corpus archive,
  rule-pack ABI, deterministic interpreter, fixed routing, BC-only manifest,
  memory bands, codec/action adapters, terminal behavior, and the named
  Ninetales/Amarys divergence.
- CUDA compilation completed with host warnings enabled.
- The optional PyTorch extension built and loaded against Torch 2.10.0 + CUDA
  13.0. It exposed engine-owned CUDA tensors without a copy.
- Three CUDA policy-pool tests passed: fixed 12-policy routing, device
  multi-select normalization, and 12-adapter gather/dispatch/scatter.
- The first environment's CPU-reference digest and CUDA digest were identical
  after 1,010 actions: `727461752255114756`.
- The explicit Ninetales 660 / Amarys 1207 probe returned status `ERROR` and
  code `6601207`; it did not crash and did not invent fixed semantics.
- The last device route contained every ready environment at both tested batch
  sizes, with no route or state error.

Compute Sanitizer could not attach on this Windows WDDM host. CUDA reported
that the WDDM debugger interface must be enabled by an administrator and that
the device was unsupported in the current mode. The instrumented program still
ran, but that is not a sanitizer pass. A Linux/TCC CI runner remains required
for the sanitizer acceptance gate.

This was reproduced from the CUDA 13.0 devel container with Compute Sanitizer
2025.3.0. The tool exists inside the container, but Docker Desktop shares the
Windows WDDM driver rather than providing a native Linux GPU driver. The run
ended with two attachment errors:

```text
Failed to initialize WDDM debugger interface.
Please run EnableDebuggerInterface.bat as an administrator.
Device not supported.
```

Therefore, moving the same command into a Linux container on this Windows host
does not bypass WDDM. A Docker container on a native Linux NVIDIA host is a
valid route for the formal sanitizer gate.

## Measured smoke throughput

The timed loop contains action interpretation, fixed PolicyCodecV1-shaped
encoding, and 12-policy device routing. It contains no host copy or host route
count read. It does not contain neural-network inference and is not a full game
throughput claim.

| environments | iterations | env-steps/s | prototype arena |
|---:|---:|---:|---:|
| 4,096 | 1,000 | 2,552,590 | 108.219 MiB |
| 16,384 | 1,000 | 3,364,469 | 432.875 MiB |

The PyTorch-bound path was also timed for 100 iterations:

| environments | env-steps/s | route count | known divergence |
|---:|---:|---:|---:|
| 4,096 | 1,931,742 | 4,096 | `6601207` |
| 16,384 | 3,132,360 | 16,384 | `6601207` |

The engine object must outlive its zero-copy tensor views. Reset from a CPU
spec intentionally synchronizes once so the temporary host buffer can be
released; the decision loop uses CUDA tensors and does not synchronize.

Compiled prototype layout:

- `BattleState`: 5,376 bytes per environment.
- Codec buffers: 22,240 bytes per environment.

The planning estimator is intentionally higher than this incomplete slice:

| environments | engine operational estimate | 10 frozen BC + 1 BC learner total |
|---:|---:|---:|
| 4,096 | 0.411 GiB | 0.793 GiB |
| 16,384 | 1.307 GiB | 2.030 GiB |

The total uses BF16 frozen weights, a four-times-FP32 learner training budget,
sequentially reused inference workspace, and a reset-time quota for fixed
cohorts. It excludes long-horizon rollout storage and full training
activations, so it is not yet a reservation value for a production job.

## Historical local policy-residency probe

The earlier capacity manifest's 11 frozen packages and a separate learner were
actually instantiated and moved to BF16 on the local 6 GiB RTX 3060. That
historical manifest included the standalone Alakazam strategy runtime. It is no
longer in scope: the current manifest contains ten frozen BCs and one BC
learner. The learner additionally reserved one BF16 gradient-sized tensor plus
an FP32 master weight and two FP32 Adam moment tensors. The smoke CUDA engine
was then created and stepped while all models remained alive.

| environments | engine steps | model + learner reserve | engine arena | final device used | final device free |
|---:|---:|---:|---:|---:|---:|
| 512 | 10 | 309.477 MiB allocated | 13.528 MiB | 1,389.5 MiB | 4,754.0 MiB |
| 4,096 | 100 | 309.477 MiB allocated | 108.219 MiB | 1,483.5 MiB | 4,660.0 MiB |

The CUDA context/container baseline already used 1,025.5 MiB, so the 4,096
environment test added about 458 MiB of device usage over that baseline. All
12 historical policy slots loaded, all 4,096 environments were present in the
last route,
and no OOM occurred. This proves that the local GPU has ample room for
development-scale residency and engine smoke tests. It does not include real
heterogeneous BC forwards, their activations, long-horizon rollout tensors, or
the RL loss/update graph, so it is not proof that a complete training profile
fits in 6 GiB.

The reproducible probe is:

```bash
python engine_cuda/tools/probe_actual_policy_residency.py \
  --batch 4096 --engine-steps 100 --dtype bf16
```

## Interpretation

The experiment establishes that a compact, batched, deck-agnostic opcode
interpreter plus device routing is technically viable and inexpensive relative
to a 32 GiB target GPU. A later server run established zero mismatch on a
704-decision sample for the complete BC-only target pool. A separate full
127,205-row PolicyCodecV1 policy replay then found 12 CPU/GPU raw action flips,
including eight across different option-equivalence groups, so exact neural
action parity is not yet established. It also does not establish official
engine parity or end-to-end CUDA-engine/BC parity. The largest remaining work
is semantic: numeric extraction,
all effect/selection opcodes, RNG ordering, exact legacy codec implementations,
and differential replay. Neural inference will probably dominate once real BCs
replace the smoke actions, which is why route-first reusable cohort buffers and
resident heterogeneous adapters are part of the production design.
