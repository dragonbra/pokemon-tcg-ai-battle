# 0044 V24 Deck 023 Standard Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start a new immutable 0044 PPO version from exact durable V23 U11, using exact focal deck 023 with the project-standard learning-rate profile and entropy coefficient.

**Architecture:** Add one self-contained version launcher that strict-binds the parent checkpoint and stopped-run boundary, validates deck 023's registry and Own Archetype V2 identity, and delegates to the existing 0044 PPO runtime. Preserve the V23 rollout/opponent/execution contract except for the explicitly requested focal deck, learning-rate profile, and entropy coefficient changes. Launch with a fresh optimizer, fresh on-policy data, and online W&B logging.

**Tech Stack:** Python, PyTorch PPO, official-engine CUDA rollout collector, pytest, TensorBoard, W&B.

## Global Constraints

- Version is `V24_v23_u11_deck023_standard_lr_entropy`; no V23 artifacts are overwritten or appended.
- Parent is exact model-only `V23_v22_u9_final_entropy_0015@update-000011`, SHA-256 `9bcb8d0c948d40fc7756fbf204ce2946150692d29605c5e5b0114c71472738de`.
- Focal exact deck is `023` (`Hydrapple ex / Meganium`), Own Archetype V2 class `27`.
- Learning rate is exactly `0044_standard_lr_v1`; entropy coefficient is exactly `0.003`.
- Preserve 512 terminal rollout games, Champion-G3 full immutable opponent policy, Meta `00/01/02/03/05/27` weights at 3.0×, three PPO epochs, logical minibatch 4096, physical/probe microbatch 256, reference-after-cache offload, PFSP disabled, and periodic evaluation disabled.
- Optimizer is fresh and all on-policy data is freshly collected; checkpoint retention remains model-only/all.
- Training and rollout remain FP32. No Kaggle FP16 deployment conversion enters PPO.
- Official engine source is read-only. Policy identity violations hard-fail before rollout.

---

### Task 1: Version contract regression

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/tests/test_v24_deck023_standard_training.py`
- Create: `train/0044_g2_dragapult_policy_option_lora/training/run_v24.py`

**Interfaces:**
- Consumes: V23 U11 model-only checkpoint and the existing `run_v1.run(...)` entry point.
- Produces: `readiness(version_root: Path) -> dict[str, object]` and `launch(wandb_mode: str, updates: int | None) -> None`.

- [ ] Write assertions for exact parent identity, focal deck/archetype, standard LR rates, entropy `0.003`, preserved rollout contract, and launch forwarding.
- [ ] Run the new test and confirm it fails because `run_v24` does not exist.
- [ ] Implement `run_v24.py` with fail-closed parent/status/deck/policy checks.
- [ ] Run the new test and confirm it passes.

### Task 2: Authoritative design synchronization

**Files:**
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md`
- Modify: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html`

**Interfaces:**
- Consumes: the constants and semantic delta declared by `run_v24.py`.
- Produces: matching human-readable Markdown and HTML descriptions of the active training stage.

- [ ] Record the exact parent, focal deck/archetype, restored LR/entropy values, preserved opponent schedule, and fresh optimizer/on-policy boundary in both documents.
- [ ] Add document assertions to the V24 regression test.
- [ ] Run the V24 and LR-profile regression tests.

### Task 3: Readiness and formal launch

**Files:**
- Create at runtime: `rl_runs/0044_g2_dragapult_policy_option_lora/versions/V24_v23_u11_deck023_standard_lr_entropy/{artifact,checkpoint,tensorboard,wandb}/`

**Interfaces:**
- Consumes: `python3 -m train.0044_g2_dragapult_policy_option_lora.training.run_v24` readiness and `--launch-formal` entry points.
- Produces: canonical `training_config.json`, `status.json`, model-only U0 checkpoint, training metrics, TensorBoard events, and one stable W&B run.

- [ ] Run readiness against the unused formal version path and require `READY_AWAITING_USER_LAUNCH`.
- [ ] Verify CUDA health, W&B import/login context, available disk, and absence of another 0044 trainer.
- [ ] Launch online training with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and no finite update cap.
- [ ] Verify the process remains live, U0 checkpoint/config/status identities match V24, and rollout begins without a policy-identity or engine error.

## Self-review

- Spec coverage: exact deck 023, latest durable checkpoint, standard LR, standard entropy, fresh run semantics, documentation, and launch verification are all assigned above.
- Placeholder scan: no deferred implementation placeholders remain.
- Type consistency: the test and launcher use the existing `readiness(...)`, `launch(...)`, and `run(...)` interfaces.
- Execution choice: the user already authorized implementation and launch, so execution proceeds inline; no sub-agent is used.
