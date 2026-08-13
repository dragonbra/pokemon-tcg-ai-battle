# 0044 G2 Dragapult Policy Option LoRA

0044 is the self-contained Champion-G2 Dragapult PPO continuation defined by:

- `docs/rl/0044_Benchmark_V2_Protocol.md`
- `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`

The current formal run is `V6_u4_core16_benchmark_v2`. It continues from the
durable V5 U4 model-only checkpoint with a fresh optimizer. The focal deck is
`007`; rollout opponents use the immutable complete Champion-G2 policy and sample
decks `001`–`067` uniformly. PFSP is disabled for V6. CUDA Engine 2.0, effective
policy isolation, policy-only Option LoRA, own-deck 29-row conditioning, PPO and
W&B gates pass. Periodic Benchmark V2 uses the independent complete Policy-0809
opponent over the fixed Core-16 Meta domain.

## Current launch and verification commands

```bash
python3 -m train.0044_g2_dragapult_policy_option_lora.training.run_v6 \
  --readiness-output .tmp/evaluation/0044_v6_readiness/readiness.json

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m train.0044_g2_dragapult_policy_option_lora.training.run_v6 \
  --launch-formal --wandb-mode online

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
- `assets/policies/`: complete Policy-0809, Champion-G1 provenance assets, and immutable Champion-G2 under semantic policy directories.
- `assets/evaluation/`: frozen schedule assets plus versioned Benchmark contracts.

The authoritative current design and audit are under `experiments/0044_g2_dragapult_policy_option_lora/`.

The 0044 build entrypoint intentionally builds only `ptcg_cuda_smoke` and `_ptcg_cuda`, always with `--parallel 1`. CUDA 12.8 compilation of one optional continuation diagnostic translation unit was observed at roughly 30 GiB RSS, so the CMake `all` target is not part of the safe 0044 build path. This restriction does not remove a production runtime target.

The production performance contract keeps `official_engine_kernels.cu`, `semantic0031_bridge.py`, and `semantic0031_resident.py` byte-identical to the audited 0042 path. Features enter through the device-resident `encode_semantic0031_v2_lanes` API and are consumed directly as CUDA `DecisionBatch` tensors; importing the CPU canonical feature compiler from the CUDA integration is a hard failure.

SHA-256 values remain mandatory registry integrity fields. They are not deck IDs, policy IDs, or meaningful folder names.
