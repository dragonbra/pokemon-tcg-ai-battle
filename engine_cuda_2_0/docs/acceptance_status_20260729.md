# CUDA engine acceptance status - 2026-07-29

> Historical snapshot. The current status, official POD ABI v5 measurements,
> and corrected VRAM estimates are in
> [`acceptance_status_20260730.md`](acceptance_status_20260730.md).

This file distinguishes prototype evidence from the formal promotion gates in
`implementation_plan.md`. A partial result is not a pass. At this snapshot,
zero of the six combined formal gates below are complete.

| Formal gate | Status | Evidence and missing work |
|---|---|---|
| CPU POD: at least 100,000 decisions, zero difference | PARTIAL | The smoke Python reference and CUDA interpreter produced the same digest after 1,010 actions for one environment. A separate two-deck status-core CUDA kernel matched 18,485 official decisions and 504,738 compared fields, but this is targeted primitive coverage, not the generic CPU POD mirror or the required 100,000-decision corpus. |
| PolicyCodecV1 127,205 decisions plus frozen corpora for every legacy codec | PARTIAL | The CPU native PolicyCodecV1 gate was rerun on this server: 1,024/1,024 games, 127,205 decisions, 25 contexts, 11 action families, and zero mismatch against JSON+Python (`docs/105_champion_v1_phase1b_native_codec_20260715_zh.md`). All 127,205 validated rows are frozen in 29 hashed NPZ shards (content SHA256 `b23d4921be877ce5805ae4676cc5d073798a9dd674b15b92a35e5f17071b1888`) and were replayed through a real 39M-parameter BC. The first 4,096 packaged-CPU/CPU-adapter actions matched exactly, but GPU inference had 12 raw-index flips; four were in the same `option_equiv` group and eight were not. This is a failed exact policy-action gate on CPU-encoded tensors, not a device-codec replay. Device codec generation and frozen corpora for the BC ID-only and Marnie prize/compact codecs are still missing. |
| Complete seeded CPU/CUDA paired replay for every promoted deck | PARTIAL | Marnie versus Alakazam setup draw order matched for seeds 1 through 1,000 (2,000 opening hands) under both recorded oracle builds. The CUDA directory is still a generic opcode vertical slice and cannot execute their complete games; search, evolution, Item/Supporter/Stadium effects, arbitrary damage-counter movement, and several special selections remain. |
| 10+ frozen BCs and one learner resident; Nsight shows no hot-path transfer or host synchronization | PARTIAL | The BC-only pool now contains ten frozen BCs plus one BC learner; the standalone Alakazam strategy runtime is excluded. Eleven simultaneous real FP32 model instances (87,596,188 parameters total) passed 704 same-state CPU/GPU action comparisons with zero mismatch. A post-fix Nsight range over all adapters had zero H2D, zero D2H, and zero host synchronization; only GPU-internal D2D copies remained. Peak Torch memory was 2,545.943 MiB allocated and 3,982 MiB reserved. The full CUDA engine and device codecs are not yet wired into this same range, so the combined formal gate remains open. |
| Compute Sanitizer on Linux/TCC and 24-hour soak | PARTIAL | Native Linux Compute Sanitizer 2025.3.1 passed memcheck, racecheck, initcheck, and synccheck for the status-core CUDA kernel with zero errors/hazards. This bypasses the Windows WDDM limitation correctly. The full engine does not exist and no qualifying 24-hour full-pipeline soak has completed. |
| End-to-end throughput at least 3x the tuned CPU pipeline on identical hardware | NOT RUN | The current 15-core CPU reference is 17.916 games/s over 2,048 games. The measured CUDA millions of environment-steps/s cover only smoke opcodes, codec-shaped writes, and routing. They omit full rules, real BC inference, rollout storage, and learning, so no CPU/CUDA speedup number exists yet. |

## Local 6 GiB result

Host: NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB, driver 610.62, WDDM.

An earlier 4,096-environment capacity probe loaded 11 frozen packages (including
the now-excluded standalone Alakazam runtime) and one learner, reserved BF16
gradients plus FP32 master/Adam state for the learner, and ran 100 smoke engine
steps. It ended with 4,660 MiB reported free. The current target manifest is ten
frozen BCs plus one BC learner. This is enough for
local compilation, differential micro-tests, policy residency, and modest
batched correctness tests. Complete RL training memory remains unknown until
real inference activations and rollout/update tensors are implemented and
measured.

## Native Linux RTX 4080 SUPER result

Host: NVIDIA GeForce RTX 4080 SUPER, 32,228.812 MiB CUDA-visible memory,
CUDA 13.0, PyTorch 2.12.1+cu130, and Nsight Systems/Compute Sanitizer 2025.3.1.

The earlier broad capacity probe loaded 11 frozen packages plus one learner in
BF16; that historical probe included the standalone Alakazam runtime which is
no longer part of the target opponent pool. It also reserved one BF16 gradient
mirror and three FP32 mirrors for learner master weights and Adam moments, then
created 4,096 environments and ran 100 engine/codec/router steps. All 4,096
environments were present in the final device route. These were vector
environments inside one process and one CUDA context, not 4,096 host workers.
The 16,384-environment figures below are estimates only.

| Measurement | Result |
|---|---:|
| device baseline used | 256.188 MiB |
| after models and learner reserve | 584.188 MiB |
| final with engine | 736.188 MiB |
| net device increase over baseline | 480.000 MiB |
| Torch model + learner-reserve allocation | 309.492 MiB |
| learner training reserve | 106.859 MiB |
| engine allocation | 108.219 MiB |
| device free at end | 31,492.625 MiB |

The current BC-only pool was then exercised with actual heterogeneous model
forwards, fixed 192-entity/128-option buffers, and capacity 512 per policy. The
learner is a trainable instance initialized from a BC checkpoint; it is not a
handwritten strategy runtime.

| Real BC pool measurement | Result |
|---|---:|
| frozen BCs + BC learner | 10 + 1 |
| instantiated parameters | 87,596,188 |
| fixed resident decision capacity | 5,632 |
| CPU/GPU action parity | 704 decisions, zero mismatch |
| post-fix Nsight H2D / D2H / host sync | 0 / 0 / 0 |
| Nsight-run policy throughput | 4,737.948 decisions/s |
| peak Torch allocated / reserved | 2,545.943 / 3,982.000 MiB |
| sustained policy workload | 281,600 decisions in 59.514 s |
| sustained policy throughput | 4,731.637 decisions/s |
| active-sample GPU utilization | 92.53% average; 122 samples at 100% |
| active-sample power | 267.26 W average; 288.15 W maximum |

The sustained utilization CSV contains 144 samples at 500 ms cadence. Twelve
setup/teardown samples were idle; 132 samples were active. This explains why a
single dashboard refresh can show zero even though the steady inference range
keeps the GPU busy. These are policy-only measurements and are not the formal
end-to-end engine speedup.

Primary evidence is archived in
`artifacts/bc_only_10frozen_1learner_hotpath_4080s.json`,
`artifacts/bc_only_10frozen_1learner_hotpath_4080s.nsys-rep`,
`artifacts/bc_only_10frozen_1learner_hotpath_4080s.sqlite`,
`artifacts/bc_only_10frozen_1learner_hotpath_4080s_audit.json`, and
`artifacts/idonly_10frozen_1learner_sustained_gpu_samples_postfix_20260729.csv`.

The standalone opcode/codec/router slice then ran 4,096 environments for
1,000,000 steps per environment: 4.096 billion environment-steps in 517.324
seconds (7.918 million environment-steps/s). The last route contained all
4,096 environments and the sampled state ended with error zero. This is a
high-volume prototype stress test, not the required 24-hour full-engine soak.

## CPU reference under the actual 15-core allocation

The host exposes 128 logical CPUs and allows affinity on CPUs 0-127, but its
cgroup reports `cpu.max = 1500000 100000`. The container therefore has 15 CPU
cores of aggregate quota, not 128 usable cores. Opponent actors are separate
single-threaded processes, while `engine_threads` controls the OpenMP team used
during engine calls; the two counts must not be interpreted as continuously
active cores and simply added together.

A 512-game scan kept 10 fixed opponent workers, 512 concurrent environments,
the same seed, and identical inputs. No run incurred a throttled cgroup period.
All six runs produced identical action, terminal, and training digests.

| Engine threads | Rollout seconds | Games/s | Decisions/s |
|---:|---:|---:|---:|
| 1 | 40.929 | 12.509 | 836.082 |
| 2 | 33.319 | 15.367 | 1,027.042 |
| 4 | 34.456 | 14.860 | 993.151 |
| 8 | 32.938 | 15.544 | 1,038.922 |
| 12 | 34.195 | 14.973 | 1,000.731 |
| 15 | 33.471 | 15.297 | 1,022.378 |

The best scan point, 8 engine threads and 10 opponent workers, was then run for
2,048 games with 512 concurrent environments. All 2,048 games were usable with
zero errors. The rollout took 114.313 seconds for 17.916 games/s and 1,216.878
decisions/s over 139,105 decisions. Across 3,060 cgroup periods it recorded
zero throttled periods and zero throttled microseconds. The full PPO iteration
peaked at 17.425 GiB allocated and 23.402 GiB reserved CUDA memory. This is the
current CPU-engine/GPU-policy reference, not a CUDA-engine result.

The earlier 16- and 32-engine-thread configurations are excluded from the
candidate baseline because they exceed the 15-core allocation. The 512-game
scan and longer result are recorded in
`artifacts/cpu15_pipeline_baseline_20260729.json`.

The existing CPU PolicyCodecV1 parity program was also rerun rather than merely
citing its historical report. All 1,024 games finished; 127,205 decisions over
25 selection contexts and 11 action families matched the JSON+Python reference
with no mismatch. Native encoding took 3.837 seconds versus 70.816 seconds for
the reference (18.46x codec-only speedup). This validates the CPU native codec,
not the unfinished device codec.

## Full PolicyCodecV1 policy replay

The complete 127,205-row frozen corpus was passed through a real
39M-parameter EntityPointerPolicyV1 checkpoint on CPU and GPU in FP32. This
tests policy inference over frozen CPU-encoded tensors; it still does not test
a CUDA implementation of PolicyCodecV1.

| Measurement | Result |
|---|---:|
| packaged CPU vs CPU device-adapter sample | 4,096 decisions, zero mismatch |
| packaged CPU vs GPU raw action indexes | 12 / 127,205 mismatches (0.00943%) |
| mismatches in the same `option_equiv` group | 4 |
| inequivalent action mismatches | 8 / 127,205 (0.00629%) |
| full parity-loop elapsed time | 2,224.326 seconds |
| post-loop CPU policy throughput | 62.936 decisions/s |
| post-loop GPU policy throughput | 1,613.390 decisions/s |
| post-loop policy-only speedup | 25.635x |

All 12 raw differences were reproduced at their original batch boundaries:
`30397`, `33581`, `36687`, `48354`, `48402`, `65563`, `66034`, `76027`,
`79489`, `94125`, `99587`, and `106530`. Singleton batch shape did not change
either backend's decision. CPU one-thread probes retained the CPU choice, and
GPU FP64 probes retained the GPU choice for two representative failures. The
first divergent top-two margins were small (approximately `2.9e-6` through
`1.31e-2`), so this is a cross-backend neural numeric-boundary issue rather
than a device-adapter mask/decode bug. Arbitrary logit rounding would not
reproduce the packaged CPU choices consistently and is not accepted as a fix.

The exact action gate therefore remains failed. Production must either promote
a GPU-frozen policy version after battle A/B validation or retrain/export with
a decision-margin requirement, then freeze that version's corpus. The failure
and diagnostics are archived in
`artifacts/policy_codec_v1_full_127205_device_replay_4080s_20260729.json`,
`artifacts/policy_codec_v1_suffix_75540_127205_device_replay_4080s_20260729.json`,
`artifacts/policy_codec_v1_device_mismatch_diagnostic_8rows_v2_20260729.json`,
and `artifacts/policy_codec_v1_device_mismatch_diagnostic_4rows_20260729.json`.
The compact machine-readable conclusion is
`artifacts/policy_codec_v1_full_device_replay_diagnosis_20260729.json`.

## Frozen-GPU policy battle A/B

The first transition option was exercised without training against the final
six-opponent pure-BC pool: Pure Lucario, Cynthia, Kangaskhan-Crustle, Marnie
Prize Control v4, Yushin Alakazam, and the Dragapult large-data two-layer BC.
Rule Lucario, rule Crustal, standalone search/rule Alakazam, and the older
Marnie probability ensemble were excluded.

| Measurement | Result |
|---|---:|
| paired battles completed | 600 / 600 |
| comparable learner decisions | 39,904 |
| raw CPU/GPU action flips | 1 |
| terminal winner flips | 1 |
| CPU wins / GPU wins | 280 / 279 |
| GPU minus CPU win rate | -0.167 percentage points |
| engine/select errors | 0 |
| pairing-invariant errors | 0 |
| Marnie Prize Control subset | 100 pairs, 6,077 decisions, 0 action flips |

The only flip was the Yushin Alakazam game at seed `2030133553`, step 42:
CPU chose raw index `[10]` and GPU chose `[9]`. Both indexes shared one
`option_equiv`, but the final winner still changed. This was not an engine coin
flip: setup/continuation RNG was paired and unchanged. It was a deterministic
CPU/CUDA neural numeric-boundary choice. Consequently, `option_equiv` is
accepted as a training-label equivalence only, not as proof of complete battle
semantic equivalence. The GPU policy semantics are frozen with this measured
risk instead of claiming exact raw-action parity.

## PPO transition smoke and checkpoint resume

The same final six-opponent pool passed a short PPO plumbing test on the RTX
4080 SUPER. This is explicitly a **seeded CPU official engine + CUDA FP32
learner** transition test using the native CPU rollout codec. It is not a test
of a complete CUDA battle engine.

Each iteration used 12 games with stratified scheduling: exactly two games per
opponent with one learner game in each seat, 12 rollout environments, six
single-threaded persistent opponent actors, eight engine threads, one PPO
epoch, and no BC update loss.

| Measurement | Fresh iteration | Resumed iteration |
|---|---:|---:|
| usable games / scheduled games | 12 / 12 | 12 / 12 |
| learner decisions | 915 | 843 |
| rollout errors / max-step games | 0 / 0 | 0 / 0 |
| optimizer step range | 0 -> 8 | 8 -> 15 |
| peak CUDA allocated | 2.682 GiB | 2.740 GiB |
| learner wins | 4 | 5 |

The resumed process loaded `checkpoint_last.pt` and recorded
`optimizer_rng_state_restored=true`, `behavior_provenance_mismatch=false`, and
continuous optimizer steps. The final checkpoint retained every original model
key and changed 33,508,552 shared floating-point elements relative to the
starting BC. All six frozen opponent checkpoint SHA-256 values, including
Marnie Prize Control v4, were identical before and after both iterations.

The executable commands are recorded in
`tools/run_policy_codec_v1_ppo_smoke.sh`; machine verification is implemented
by `tools/verify_policy_codec_v1_ppo_smoke.py`. Generated reports and
checkpoints remain server artifacts and are intentionally excluded from Git.

Per the project owner's instruction, the 24-hour soak is skipped for this
transition milestone. That does not convert the formal soak gate into a pass.

The planning estimate is dominated by learner rollout retention once model and
engine memory are this small:

| Learner rollout retained per environment | 4,096 environments | 16,384 environments |
|---:|---:|---:|
| 0 KiB | 0.793 GiB | 2.030 GiB |
| 256 KiB | 1.793 GiB | 6.030 GiB |
| 1,024 KiB | 4.793 GiB | 18.030 GiB |
| 2,048 KiB | 8.793 GiB | 34.030 GiB |

These planning rows include operational engine headroom, BF16 frozen weights,
an Adam-like learner budget, and a sequentially reused inference workspace.
They do not replace a measurement of real activations and rollout tensors. In
particular, a 16,384-environment job must keep retained rollout data well below
2 MiB per environment on a 32 GiB card.

The Nsight NVTX range
`decision_hot_path_engine_codec_route` contained 300 kernel launches and 200
device memsets. It contained zero correlated memcpy operations and zero
synchronization or blocking CUDA runtime calls. Model weights remained alive,
but model forwards were intentionally outside this range because production
adapter wiring is not implemented; this result must not be quoted as a full
policy-decision trace.

The exact local frozen oracle requires GLIBC 2.38 and was therefore captured in
an Ubuntu 24.04 container, then compared on the native Linux CUDA host. It was
not used to replace the server oracle.

| Oracle | SHA256 | Status-core result | Marnie/Alakazam setup |
|---|---|---:|---:|
| local frozen | `1ff4c83190ed9512798de528ce5180c543e7f4c3a7e059b175a379b148911df5` | 18,485 decisions, 504,738 fields, zero mismatch | seeds 1-1,000, 2,000 hands, zero mismatch |
| pre-existing server | `f951877d83576098973decd64b0748bf7791d9390b06624146384c179b39e064` | 18,485 decisions, 504,738 fields, zero mismatch | seeds 1-1,000, 2,000 hands, zero mismatch |

The status-core decks deliberately exercise official seed folding and shuffle,
mulligan/setup, turn draw, energy attachment and requirements, direct damage,
poison/checkup, confusion coin flip and self-damage, knockout/discard ordering,
Prize selection, and terminal result. This is useful independent primitive
evidence, but it is not evidence that arbitrary promoted decks are supported.

WandB 0.28.1 is installed on the server and completed an offline logging smoke
run. Cloud monitoring still requires an API key and the intended entity/project.

## Known engine divergence

Ninetales `660` borrowing Amarys `1207` remains a named, quarantined
divergence. The CUDA prototype returns `KNOWN_DIVERGENCE_660_1207`; it does not
invent replacement semantics. Formal paired corpora must exclude the case
until a fixed official engine is available, then add it as a dedicated
old-engine/new-engine regression fixture. The frozen CPU oracle must not be
modified in place.
