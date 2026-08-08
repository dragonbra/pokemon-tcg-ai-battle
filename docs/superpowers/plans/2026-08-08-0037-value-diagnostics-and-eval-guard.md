# 0037 Value Diagnostics and Frozen Evaluation Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run two fair 50-update 0037 PPO experiments with focal-only Encoder LoRA, optional LayerNorm tuning, fixed Frozen-0806 greedy evaluation every five updates, and Value diagnostics by official-engine game phase and terminal outcome.

**Architecture:** Keep the official-engine rollout, PPO target, and N16E8 runtime unchanged. Add low-rank weight parametrizations to focal board-Encoder layers 0, 1 and 3 while the independently loaded Frozen opponent remains untouched; the second arm additionally tunes focal state/option LayerNorm affine parameters. Separate schedule provenance identity from the environment-only frozen-evaluation commitment, then compute read-only diagnostics from the already prepared on-policy batch so monitoring adds no extra Encoder pass.

**Tech Stack:** Python 3.11, PyTorch, unittest, canonical JSONL, TensorBoard, W&B online, official seeded engine runtime.

## Global Constraints

- Do not modify `engine/source/`.
- Create a new strictly increasing `V3_*` run; never append to failed V2.
- Continue saving model-only checkpoints for every update with retention `all`.
- Use official engine turn metadata, not action counts, for phase diagnostics.
- Keep terminal reward and PPO/GAE optimization semantics unchanged.
- Mirror all new scalar diagnostics through canonical JSONL, TensorBoard, then W&B.
- Synchronize both `experiments/0037_dragapult_value_initialized_rl/DESIGN.md` and `DESIGN.html`.

---

### Task 1: Environment-only frozen schedule guard

**Files:**
- Modify: `train/0037_dragapult_value_initialized_rl/training/run_full_semantic.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_rollout_batch.py`

**Interfaces:**
- Consumes: `list[RolloutJob]` from `build_jobs`.
- Produces: `_environment_schedule_sha256(jobs: list[RolloutJob]) -> str`, stored as `environment_sha256`.

- [ ] **Step 1: Write failing tests**

Add tests proving that changing `game_id` or `source_policy_update` leaves the environment commitment unchanged, while changing opponent, seat, engine seed, search seed, or policy seed changes it.

- [ ] **Step 2: Run the focused tests and observe failure**

Run: `python3 -m unittest -v train.0037_dragapult_value_initialized_rl.tests.test_rollout_batch`

- [ ] **Step 3: Implement the environment commitment**

Hash only `opponent_id`, `focal_first`, `engine_seed`, `search_seed`, and `policy_seed`; retain the existing full schedule hash as provenance. Compare `environment_sha256` at every frozen checkpoint evaluation.

- [ ] **Step 4: Run the focused tests**

Run the command from Step 2 and require PASS.

### Task 2: Turn- and outcome-stratified Value diagnostics

**Files:**
- Modify: `train/0037_dragapult_value_initialized_rl/training/batch_full_semantic.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_full_terminal_credit.py`

**Interfaces:**
- Consumes: `PreparedBatch.old_value`, `gae_return`, `terminal_return`, `turns`, `episode_index`, and episode final turns.
- Produces: flat scalar keys under `value_diag/*` from `value_diagnostic_metrics(batch)`.

- [ ] **Step 1: Write failing tests**

Construct deterministic winning and losing trajectories spanning official turns and assert counts, mean Value, GAE explained variance, terminal explained variance, and opening/late/remaining-turn group keys.

- [ ] **Step 2: Run the focused tests and observe failure**

Run: `python3 -m unittest -v train.0037_dragapult_value_initialized_rl.tests.test_full_terminal_credit`

- [ ] **Step 3: Preserve final-turn metadata**

Add a per-decision `terminal_turns: Tensor` field to `PreparedBatch`, populated from `EpisodeTrajectory.turns`, and validate that decision turns do not exceed the final official engine turn.

- [ ] **Step 4: Implement weighted diagnostics**

Use episode-equal decision weights and report group decision count, effective episode mass, Value mean/std, Value MAE/bias to GAE and terminal targets, plus both explained variances. Emit absolute official-turn bins `turn_00_03`, `turn_04_07`, `turn_08_11`, `turn_12_plus`; terminal-distance bins `remaining_00_01`, `remaining_02_03`, `remaining_04_plus`; and terminal-outcome bins `focal_win`, `focal_loss`, `draw`.

- [ ] **Step 5: Run the focused tests**

Run the command from Step 2 and require PASS.

### Task 3: Focal-only Encoder LoRA and optional LayerNorm tuning

**Files:**
- Create: `train/0037_dragapult_value_initialized_rl/policy/adaptation.py`
- Modify: `train/0037_dragapult_value_initialized_rl/policy/actor_critic.py`
- Modify: `train/0037_dragapult_value_initialized_rl/training/ppo_full_semantic.py`
- Modify: `train/0037_dragapult_value_initialized_rl/training/storage_full_semantic.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_frozen_heads.py`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_storage.py`

**Interfaces:**
- Consumes: the focal `SemanticActorCritic` returned by `load_actor_critic`; never the separate Frozen opponent actor.
- Produces: `AdaptationConfig`, adapter injection/inventory helpers, optimizer parameter groups, and model-only adapter/LN state.

- [ ] **Step 1: Write failing trainability and isolation tests**

Assert LoRA-only trainable names are decoder, critic, and adapter parameters; LayerNorm arm additionally exposes only state/option LayerNorm affine tensors. Assert original Encoder weights and an independently loaded opponent state hash remain unchanged.

- [ ] **Step 2: Implement low-rank parametrizations**

For `state_encoder.board_encoder.layers[0,1,3]`, add rank-8, alpha-16 additive low-rank parametrizations to `self_attn.in_proj_weight`, `self_attn.out_proj.weight`, `linear1.weight`, and `linear2.weight`. Initialize the output factor to zero so both arms exactly reproduce source logits at update 0.

- [ ] **Step 3: Implement optional LayerNorm tuning**

When enabled, mark weight and bias trainable for every `nn.LayerNorm` below `state_encoder` and `option_encoder`, excluding `prototype_encoder`, decoder, and Value modules.

- [ ] **Step 4: Extend optimizer and checkpoint contracts**

Add a dedicated adapter group at LR `3e-5` and LayerNorm group at LR `1e-5`. Save adapter and optional LayerNorm tensors with decoder/value tensors in a new model-only schema; reject optimizer and base Encoder weights.

- [ ] **Step 5: Run focused tests**

Run: `python3 -m unittest -v train.0037_dragapult_value_initialized_rl.tests.test_frozen_heads train.0037_dragapult_value_initialized_rl.tests.test_storage`

### Task 4: New V3/V4 run contracts and monitoring cadence

**Files:**
- Modify: `train/0037_dragapult_value_initialized_rl/training/run_full_semantic.py`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DESIGN.md`
- Modify: `experiments/0037_dragapult_value_initialized_rl/DESIGN.html`
- Test: `train/0037_dragapult_value_initialized_rl/tests/test_contract.py`

**Interfaces:**
- Consumes: Task 1 guard and Task 2 metrics.
- Produces: immutable `V4_lora_r8_eval5_50u` and `V5_lora_r8_layernorm_eval5_50u` artifacts with distinct W&B runs; V3 is retained as a zero-update launcher failure.

- [ ] **Step 1: Write/update contract assertions**

Assert default `updates == 50`, `eval_every == 5`, distinct arm identifiers, and the training config diagnostics/adaptation schema.

- [ ] **Step 2: Configure V3**

Expose an explicit experiment arm argument. Use versions `V4_lora_r8_eval5_50u` and `V5_lora_r8_layernorm_eval5_50u`, W&B display prefixes beginning with `0037`, fixed greedy evaluation cadence five, budget 50 updates, and exact diagnostic/adaptation contracts in `training_config.json`.

- [ ] **Step 3: Update both authoritative design documents**

Record V2's schedule-guard failure and neutral update-10 frozen result; document the environment-only fix, eval-every-5 cadence, rollout-source timing, phase/outcome metric definitions, and V3 status.

- [ ] **Step 4: Run project tests**

Run: `python3 -m unittest discover -v train/0037_dragapult_value_initialized_rl/tests`

### Task 5: Smoke and sequential formal launch

**Files:**
- Create at runtime: `rl_runs/0037_dragapult_value_initialized_rl/versions/V4_lora_r8_eval5_50u/*`
- Create at runtime: `rl_runs/0037_dragapult_value_initialized_rl/versions/V5_lora_r8_layernorm_eval5_50u/*`

**Interfaces:**
- Consumes: tested V3 code and unchanged source checkpoints.
- Produces: two fresh sequential W&B online runs and their checkpoint-0 official frozen evaluations.

- [ ] **Step 1: Run a non-W&B diagnostic smoke**

Use a disposable `.tmp/evaluation/0037_v3_diagnostics_smoke/` output and a small official-engine rollout to verify that every new metric is finite and the environment guard accepts provenance-only update changes.

- [ ] **Step 2: Verify fresh version paths**

Confirm V3 and V4 artifact, checkpoint, TensorBoard, W&B, and formal evaluation paths are unused.

- [ ] **Step 3: Launch the formal V3 process**

Start a fail-closed sequential launcher: V3 LoRA-only first, then V4 LoRA+LayerNorm only if V3 exits successfully. Both use W&B online, N16E8/I8, 5 ms coalescing, and independent logs/PIDs.

- [ ] **Step 4: Inspect initialization evidence**

For each arm, confirm checkpoint 0, schedule commitment, W&B run URL, and the first fixed 512-game evaluation. Do not auto-select or continue to 200 without comparable update-50 frozen evidence and an explicit selection criterion.
