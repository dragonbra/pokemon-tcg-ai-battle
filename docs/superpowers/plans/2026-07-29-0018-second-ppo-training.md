# 0018 Second PPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a reproducible Alakazam decoder-only PPO run that can make measurable greedy action-ranking changes without losing the strong BC representation.

**Architecture:** Keep the R15 Encoder frozen and update only the existing Decoder plus Value Head. Use undiscounted terminal credit (`gae_lambda=1.0`), zero entropy bonus, a higher actor learning rate under PPO/KL guards, 512 official-engine Episodes per update, and model-only checkpoints after every update. Measure both sampled rollout strength and deterministic policy movement against the frozen BC reference.

**Tech Stack:** Python 3.11, PyTorch, official PTCG engine runtime, repository PPO infrastructure, TensorBoard, W&B online.

## Global Constraints

- Reward remains terminal-only: win `+1`, loss `-1`, draw `0`.
- Do not modify `engine/source/`.
- Use project/version W&B naming in `dragon_bra/pokemon-tcg-policy-learning`.
- Save model weights and metadata only; never save optimizer, scheduler, RNG, rollout, or replay state.
- Allocate a fresh immutable `V<n>_<tag>` directory and never overwrite V1/V2.
- Preserve unrelated user changes in evaluation tests and the arena plan.
- Synchronize both `experiments/0018_alakazam_terminal_rl/DESIGN.md` and `DESIGN.html`.

---

### Task 1: Reproducible PPO configuration

**Files:**
- Modify: `train/0018_alakazam_terminal_rl/run.py`
- Test: `train/0018_alakazam_terminal_rl/tests/test_training.py`

**Interfaces:**
- Produces: `seed_training_rng(seed: int) -> None`
- Produces CLI fields `--entropy-coefficient`, `--reference-kl-coefficient`, and `--target-behavior-kl`.

- [ ] Add a failing test that patches `torch.manual_seed` and verifies the main process receives the configured seed.
- [ ] Run `python3 -m unittest train.0018_alakazam_terminal_rl.tests.test_training.TrainingContractTest.test_training_rng_seeds_torch` and verify it fails.
- [ ] Implement `seed_training_rng`, call it before model creation, and pass the three CLI coefficients into `PPOConfig`.
- [ ] Add parser tests proving `0`, `0.02`, and `0.01` survive parsing and configuration.
- [ ] Run the focused training contract tests and verify they pass.

### Task 2: Policy-movement diagnostics and dense checkpoints

**Files:**
- Modify: `train/0018_alakazam_terminal_rl/training/ppo.py`
- Modify: `train/0018_alakazam_terminal_rl/run.py`
- Test: `train/0018_alakazam_terminal_rl/tests/test_training.py`

**Interfaces:**
- Produces scalar metrics under `ppo/reference_*` and `ppo/behavior_*` for W&B/TensorBoard.
- Produces one model-only `checkpoint/update-XXXXXX.pt` per PPO update when `--checkpoint-every 1` is selected.

- [ ] Add tests that the parser accepts checkpoint cadence 1 and rejects non-positive cadence.
- [ ] Record parameter displacement and the existing reference/behavior KL diagnostics at each update.
- [ ] Keep PPO clipping, gradient clipping, and behavior-KL early stop active at the higher LR.
- [ ] Verify every saved payload excludes optimizer, scheduler, scaler, RNG, and rollout state.
- [ ] Run the 0018 unit suite.

### Task 3: V2 disposition and authoritative design

**Files:**
- Modify: `rl_runs/0018_alakazam_terminal_rl/versions/V2_ppo_decoder_terminal/artifact/status.json`
- Create: `rl_runs/0018_alakazam_terminal_rl/versions/V2_ppo_decoder_terminal/artifact/training_summary.json`
- Modify: `experiments/0018_alakazam_terminal_rl/DESIGN.md`
- Modify: `experiments/0018_alakazam_terminal_rl/DESIGN.html`

**Interfaces:**
- Records V2 as a planned stop after retained update 10, with no proven greedy gain.
- Records the V4 hypothesis, exact tensor/parameter boundary, loss coefficients, observability, and checkpoint policy.

- [x] Replace the misleading V2 `failed: KeyboardInterrupt` disposition with a planned-stop record while preserving the interrupt fact.
- [x] Document the 300-game BC/U1/U10 diagnostic and V2 parameter displacement.
- [x] Add the V4 configuration and acceptance/stop criteria to both DESIGN formats.
- [x] Cross-check both documents against `training_config.json` and code defaults.

### Task 4: Smoke, commit, and formal V4 launch

**Files:**
- Create at runtime: `rl_runs/0018_alakazam_terminal_rl/versions/V3_ppo_lambda1_lr1e5_smoke/`
- Create at runtime: `rl_runs/0018_alakazam_terminal_rl/versions/V4_ppo_decoder_lambda1_lr1e5/`

**Interfaces:**
- Formal command uses 512 Episodes/update, lambda 1.0, two PPO epochs, actor LR `1e-5`, value LR `1e-4`, entropy 0, reference KL coefficient `0.02`, target behavior KL `0.01`, and checkpoint cadence 1.

- [x] Run compile checks and the 0018 unit suite.
- [x] Run `V3_ppo_lambda1_lr1e5_smoke` with a small official-engine rollout and verify finite loss, legal Episodes, W&B disabled, and model-only checkpoint format.
- [ ] Commit only the 0018 implementation, documentation, plan, and V2 tracked records with a Chinese Git-style message; immediately push `dev/cyd_main`.
- [ ] Confirm GPU availability and launch V4 with W&B online and project/version run naming.
- [ ] Inspect update 1 metrics for rollout validity, KL, clip fraction, gradient norm, throughput, and checkpoint creation before leaving the run unattended.
- [ ] Continue only while behavior KL remains below `0.01`, checkpoints are valid, and fixed evaluation evidence does not show material regression.
