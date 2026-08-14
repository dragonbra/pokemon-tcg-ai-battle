# 0044 G2 Dragapult Policy Option LoRA

0044 is the self-contained Champion-G2 Dragapult PPO continuation defined by:

- `docs/rl/0044_Benchmark_V2_Protocol.md`
- `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`

The latest completed formal training run is
`V10_g3_v9_u50_generalist_001_067_core16_benchmark_v2`, stopped at durable U110.
It is retained with its historical fixed 128/128 training-seat semantics. Its
official Benchmark V2 path already used seeded toss winners and the winning Agent's
real context-41 choice, so that evaluation evidence remains valid.

V9 has an exact-U50 handoff contract. After its durable U50 checkpoint, matching
PASS Benchmark V2 CUDA-2048 report, and canonical `eval/*` row all exist, the
supervisor stops V9 and launches
`V10_g3_v9_u50_generalist_001_067_core16_benchmark_v2`. V10/G3 uses a fresh
optimizer and local U0, keeps Champion-G2 opponents uniformly sampled over 001–067
with PFSP disabled, and runs indefinitely. Its focal lanes cover all 67 exact decks
on every update (3–4 lanes each, seeded random permutation) while retaining one
resident focal policy. Deck066 remains the fixed Benchmark V2 longitudinal sentinel;
it is not presented as an all-deck G3 aggregate.

Every training successor after V10 must use
`seeded_toss_winner_agent_context_41_choice_v1`: the schedule supplies only a
coin-winner seed and toss-winner identity, the complete winning focal/Champion
policy chooses first or second at official context 41, and actual seat counts are
observed telemetry rather than balanced harness quotas. Missing choice evidence is
fatal before PPO. V10 opponent decks were exact-deck IID uniform over 001–067;
V12 changes this explicitly to the Meta-first schedule described below.

Champion-G3 is now the immutable promoted V10 U110 policy. The next formal training
version is `V12_g4_g3_meta_balanced_512_meta_residual` because V11 is already the
formal Full-67 evaluation version. V12 uses G3 for focal initialization, Reference
KL, and the singleton frozen opponent policy. Both focal and opponent decks use a
512-lane Meta-first schedule with identical Meta marginals, independent deck/lane
shuffles, and balanced member-deck frequency. PFSP remains disabled. It adds a
zero-initialized 29-way rank-4 policy-only Meta Actor Residual (74,240 parameters),
uses logical minibatch 4096 / physical 1024 / 3 epochs, actor LRs 5e-6, Option LoRA
1e-5, and Value/Prize LR 2e-5.

## Current launch and verification commands

```bash
python3 -m train.0044_g2_dragapult_policy_option_lora.training.run_v12 \
  --readiness-output .tmp/evaluation/0044_v12_g3_to_g4/readiness.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m train.0044_g2_dragapult_policy_option_lora.training.run_v12 \
  --launch-formal --wandb-mode online

python3 -m train.0044_g2_dragapult_policy_option_lora.training.run_v9 \
  --readiness-output .tmp/evaluation/0044_v9_deck066_readiness/readiness.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m train.0044_g2_dragapult_policy_option_lora.training.run_v9 \
  --launch-formal --wandb-mode online

python3 -m train.0044_g2_dragapult_policy_option_lora.training.handoff_v9_u50_to_v10 \
  --v9-pid 198707 --poll-seconds 2

python3 -m train.0044_g2_dragapult_policy_option_lora.import_assets --all
PYTHONPATH=. /home/cyd/.cache/uv/archive-v0/uIcEF-JYHod8LomN/bin/pytest -q train/0044_g2_dragapult_policy_option_lora/tests
PYTHONPATH=. python3 -m train.0044_g2_dragapult_policy_option_lora.preflight --gpu-smoke --output .tmp/evaluation/0044_preflight/report.json
python3 -m train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.build print
python3 -m train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.build build
python3 -m train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.build smoke \
  --rule-pack .tmp/cuda_0032_rules/official_rules.bin \
  --output .tmp/evaluation/0044_cuda_engine_2/build_smoke.json
python3 -m train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.smoke \
  --rule-pack .tmp/cuda_0032_rules/official_rules.bin \
  --output .tmp/evaluation/0044_cuda_engine_2/policy_transition_smoke.json
python3 -m train.0044_g2_dragapult_policy_option_lora.cuda_engine_2.performance \
  --old-binary engine_cuda/build/ptcg_cuda_smoke \
  --new-binary engine_cuda_2_0/build/native/ptcg_cuda_smoke \
  --extension-dir engine_cuda_2_0/build/native \
  --output .tmp/evaluation/0044_cuda_engine_2/performance.json
```

The import command is the only code allowed to read the approved historical sources. It freezes both immutable assets and the semantic inference source into 0044. Runtime consumers use only project-local registries, semantic policy directories and `semantic_runtime/`.

## Asset domains

- `assets/decks/`: 67 immutable exact-deck definitions with canonical IDs and directories `001`–`067`.
- `assets/policies/`: complete Policy-0809 plus immutable Champion-G2 and Champion-G3 under semantic policy directories.
- `assets/evaluation/`: frozen schedule assets plus versioned Benchmark contracts.

The authoritative current design and audit are under `experiments/0044_g2_dragapult_policy_option_lora/`.

The 0044 build entrypoint intentionally builds only `ptcg_cuda_smoke` and `_ptcg_cuda`, always with `--parallel 1`. CUDA 12.8 compilation of one optional continuation diagnostic translation unit was observed at roughly 30 GiB RSS, so the CMake `all` target is not part of the safe 0044 build path. This restriction does not remove a production runtime target.

The production performance contract keeps `official_engine_kernels.cu`, `semantic0031_bridge.py`, and `semantic0031_resident.py` byte-identical to the audited 0042 path. Features enter through the device-resident `encode_semantic0031_v2_lanes` API and are consumed directly as CUDA `DecisionBatch` tensors; importing the CPU canonical feature compiler from the CUDA integration is a hard failure.

SHA-256 values remain mandatory registry integrity fields. They are not deck IDs, policy IDs, or meaningful folder names.
