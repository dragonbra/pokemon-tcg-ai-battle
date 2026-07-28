# 0018 Alakazam Terminal RL Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained 0018 Alakazam terminal-reward RL project and make resident GPU inference for compatible Arena BC opponents a required gate before PPO training.

**Architecture:** Copy and freeze the verified 0017 official-engine PPO implementation into the 0018 namespace, replace its actor/deck provenance with the 0016 R15 epoch-10 Alakazam policy, and preserve terminal-only reward. Engine workers remain isolated CPU processes; both candidate requests and compatible BC-opponent requests are relayed to the parent process, where models remain resident on one CUDA device and inference is dynamically batched by policy ID/model family.

**Tech Stack:** Python 3.11, PyTorch, multiprocessing official-engine workers, Unix/local IPC, W&B, TensorBoard, standard-library `unittest`.

## Global Constraints

- Never modify `engine/source/`.
- `train/0018_alakazam_terminal_rl/` must not import executable code from another numbered training project.
- The target deck is the exact 60-card 0016 target deck with SHA-256 `267ce842b45f960afef843e68f3aad86573bf7d19329ea23e7469ec8265c017b`.
- The source checkpoint is 0016 R15 epoch 10 with SHA-256 `9db3b4d37a4d4f5cc483df1ac78ad4c5d39fed320c5325ebaa5fa61fd1114ebf`.
- Reward is terminal-only: win `+1`, loss `-1`, draw `0`; engine errors and truncations are invalid Episodes.
- Formal W&B runs use `dragon_bra/pokemon-tcg-policy-learning` and the project + version run-name contract.
- Formal PPO may start only after resident-GPU opponent inference passes correctness and throughput gates.
- Model-only checkpoints exclude optimizer, scheduler, replay buffer, raw observations and traces.

---

### Task 1: Freeze source policy and target deck

**Files:**
- Create: `train/0018_alakazam_terminal_rl/`
- Create: `train/0018_alakazam_terminal_rl/deck.csv`
- Create: `train/0018_alakazam_terminal_rl/deck_manifest.json`
- Modify: `train/0018_alakazam_terminal_rl/constants.py`
- Modify: `train/0018_alakazam_terminal_rl/policy/actor_critic.py`
- Test: `train/0018_alakazam_terminal_rl/tests/test_policy.py`

**Interfaces:**
- Consumes: 0016 checkpoint metadata and exact target deck as immutable provenance.
- Produces: `load_source_actor_critic(device: torch.device) -> AlakazamActorCritic` and `TARGET_DECK: tuple[int, ...]`.

- [ ] Copy the 0017 implementation tree into the new 0018 namespace without runtime imports to 0017.
- [ ] Replace project, checkpoint, deck, source-ID and schema constants with the audited 0016 values.
- [ ] Rename actor-critic and diagnostics from Dragapult to Alakazam while preserving the R15 tensor/action contract.
- [ ] Add tests for checkpoint SHA, exact 60-card multiset, R15 metadata, source ID `0`, and actor state equality.
- [ ] Run `python3 -m unittest -v train.0018_alakazam_terminal_rl.tests.test_policy` and require PASS.

### Task 2: Establish the authoritative 0018 design

**Files:**
- Create: `experiments/0018_alakazam_terminal_rl/DESIGN.md`
- Create: `experiments/0018_alakazam_terminal_rl/DESIGN.html`
- Create: `experiments/0018_alakazam_terminal_rl/manifest.json`

**Interfaces:**
- Consumes: source/deck hashes, the 22-group actor input contract and the 0017 PPO contract.
- Produces: synchronized Markdown/HTML descriptions of inputs, model, opponent service, reward, metrics, storage and stage gates.

- [ ] Document official-rule, runtime-fact and project-hypothesis boundaries.
- [ ] Record the exact target construction and distinguish it from the mirror Xerosic ablation.
- [ ] Document CPU-engine/GPU-policy request flow, supported/fallback opponent classes and benchmark gates.
- [ ] Document Decoder-only L0 PPO and the L1-L5 unfreezing ladder.
- [ ] Cross-check all hashes, dimensions and parameter counts against code/checkpoint metadata.

### Task 3: Add resident GPU opponent inference

**Files:**
- Create: `train/0018_alakazam_terminal_rl/opponent_inference/__init__.py`
- Create: `train/0018_alakazam_terminal_rl/opponent_inference/catalog.py`
- Create: `train/0018_alakazam_terminal_rl/opponent_inference/resident_pool.py`
- Modify: `train/0018_alakazam_terminal_rl/rollout/protocol.py`
- Modify: `train/0018_alakazam_terminal_rl/rollout/worker.py`
- Modify: `train/0018_alakazam_terminal_rl/rollout/collector.py`
- Test: `train/0018_alakazam_terminal_rl/tests/test_opponent_inference.py`

**Interfaces:**
- Consumes: `SubmissionPackage`, session ID, observation and CUDA device.
- Produces: `ResidentOpponentPool.select(requests: list[OpponentRequest]) -> list[OpponentResponse]`, eligibility audit and latency/batch metrics.

- [ ] Classify enabled Arena opponents as rule-based CPU or supported BC GPU family from auditable package files/manifests.
- [ ] Load each supported policy exactly once, move model weights to CUDA, call `eval()` and use `torch.inference_mode()`.
- [ ] Keep causal history/encoder state isolated by game session and route requests by policy ID.
- [ ] Relay eligible opponent decisions from engine workers to the parent collector; retain local CPU calls for ineligible rule policies.
- [ ] Batch simultaneous requests for the same compatible policy and report resident bytes, request count, batch sizes and latency.
- [ ] Test eligibility, session isolation, device placement, response ordering and CPU fallback without modifying Arena packages.

### Task 4: Validate end-to-end and benchmark the gate

**Files:**
- Modify: `train/0018_alakazam_terminal_rl/benchmark_rollout.py`
- Modify: `train/0018_alakazam_terminal_rl/smoke.py`
- Modify: `train/0018_alakazam_terminal_rl/training_smoke.py`
- Create: `.tmp/evaluation/0018_gpu_opponent_benchmark/<run-id>/report.json`

**Interfaces:**
- Consumes: identical official-engine jobs under CPU-opponent and resident-GPU modes.
- Produces: correctness/throughput evidence and an explicit `formal_rl_ready` boolean.

- [ ] Run policy-only parity checks on fixed observations and require identical greedy full actions.
- [ ] Run official-engine smoke games with balanced seats and require 100% valid, 100% legal completion.
- [ ] Benchmark identical opponent/job schedules in CPU and resident-GPU modes, recording Episodes/sec, decisions/sec, p50/p95 latency, load time, resident VRAM and GPU utilization.
- [ ] Require no policy/model reload per Episode and no cross-session state leakage.
- [ ] Run all focused 0018 tests, `compileall`, `git diff --check`, and verify that no 0018 file imports 0016/0017 executable modules.

### Task 5: Prepare but do not start formal PPO

**Files:**
- Modify: `train/0018_alakazam_terminal_rl/run.py`
- Create: `rl_runs/0018_alakazam_terminal_rl/opponent_pool_snapshot.json`

**Interfaces:**
- Consumes: benchmark gate and frozen 26-opponent catalog snapshot.
- Produces: validated `value` and `ppo` CLI contracts that refuse formal training if the gate is absent or mismatched.

- [ ] Set the initial PPO defaults to 256 complete Episodes/update, 12 workers, actor LR `3e-6`, value LR `1e-4`, GAE lambda `0.97`, four PPO epochs and minibatch 1024.
- [ ] Add a preflight that verifies source/deck/pool hashes, CUDA availability, free disk and the resident-inference benchmark gate.
- [ ] Preserve canonical JSONL -> TensorBoard -> W&B metric ordering and project/version naming.
- [ ] Run the CUDA training smoke only; do not launch a formal W&B PPO version in this task.
