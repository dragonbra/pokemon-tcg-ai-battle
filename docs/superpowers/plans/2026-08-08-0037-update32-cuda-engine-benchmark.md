# 0037 Update32 CUDA Engine Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the exported deck-007 Update32 actor and immutable 0806 opponent policy through complete GPU-resident CUDA-engine games, then compare its rollout throughput with the existing 0037 official CPU-engine baseline without treating CUDA outcomes as formal strength evidence.

**Architecture:** Add one focused benchmark CLI that loads the two portable semantic actors, materializes the fixed 0037 seeded-512 deck/seat schedule on CUDA, and keeps engine state, semantic-v2 observations, greedy decoding, and action application device-resident until periodic terminal checks. Pure schedule/model-loading helpers are unit-tested without requiring CUDA; runtime output records hashes, completion/errors, timing, component scope, and the official-oracle evidence boundary.

**Tech Stack:** Python 3.11, PyTorch 2.11 CUDA 12.8, `_ptcg_cuda` PyTorch extension, official CUDA rule pack, `unittest`.

## Global Constraints

- Do not modify `engine/source/`; the unmodified official CPU engine remains the oracle.
- Use exact focal deck `dragapult_ex_07bedfffbfad` and 0037 V5 Update32 exported actor.
- Use immutable friend-0806 epoch-11 opponent weights with SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8` provenance.
- Preserve the fixed 512-job opponent/seat/engine-seed schedule; CUDA currently has one engine seed input, so do not claim exact equivalence to the separate official Search-seed ABI.
- CUDA throughput is training/collector feasibility evidence; policy-strength conclusions still require the official CPU engine.
- Benchmark artifacts stay under `.tmp/engine_cuda_benchmark/` and are not committed.

---

### Task 1: Fixed-schedule and portable-model contracts

**Files:**
- Create: `engine_cuda/tools/benchmark_0037_update32_cuda_rollout.py`
- Create: `engine_cuda/tests/test_benchmark_0037_update32_cuda_rollout.py`

**Interfaces:**
- Consumes: 0037 `eval_fixed_seeded512.json`, frozen-0806 deck root, portable actor directories.
- Produces: `load_schedule(path, deck_root, focal_deck) -> ScheduleBatch` and `expanded_portable_state(payload) -> dict[str, Tensor]`.

- [ ] **Step 1: Write failing schedule and state-expansion tests**

```python
def test_load_schedule_preserves_job_order_seat_and_seed(self):
    batch = module.load_schedule(schedule, decks, focal)
    self.assertEqual(batch.engine_seeds, (101, 101))
    self.assertEqual(batch.focal_players, (0, 1))
    self.assertEqual(batch.deck_rows[0], (focal, opponent))
    self.assertEqual(batch.deck_rows[1], (opponent, focal))

def test_expanded_portable_state_restores_both_prototype_aliases(self):
    expanded = module.expanded_portable_state({"prototype_encoder.x": tensor})
    self.assertIs(expanded["state_encoder.prototypes.x"], tensor)
    self.assertIs(expanded["option_encoder.prototypes.x"], tensor)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python3 -m unittest -v engine_cuda.tests.test_benchmark_0037_update32_cuda_rollout`

Expected: import/file failure because the benchmark module does not exist.

- [ ] **Step 3: Implement strict pure helpers**

```python
@dataclass(frozen=True)
class ScheduleBatch:
    engine_seeds: tuple[int, ...]
    focal_players: tuple[int, ...]
    deck_rows: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    opponent_ids: tuple[str, ...]

def expanded_portable_state(state):
    canonical = {k.removeprefix("prototype_encoder."): v for k, v in state.items()
                 if k.startswith("prototype_encoder.")}
    if not canonical:
        raise ValueError("portable checkpoint has no canonical prototype encoder")
    output = dict(state)
    for prefix in ("state_encoder.prototypes.", "option_encoder.prototypes."):
        output.update({prefix + suffix: value for suffix, value in canonical.items()})
    return output
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run: `python3 -m unittest -v engine_cuda.tests.test_benchmark_0037_update32_cuda_rollout`

Expected: all helper-contract tests pass without CUDA.

### Task 2: Complete CUDA-resident rollout CLI

**Files:**
- Modify: `engine_cuda/tools/benchmark_0037_update32_cuda_rollout.py`
- Modify: `engine_cuda/tests/test_benchmark_0037_update32_cuda_rollout.py`

**Interfaces:**
- Consumes: `ScheduleBatch`, `_ptcg_cuda.OfficialCudaEngine`, two portable `SemanticPolicy` models.
- Produces: JSON schema `0037_update32_cuda_rollout_benchmark_v1` with completion, error, timing, memory, hashes, and evidence-boundary fields.

- [ ] **Step 1: Add failing validation tests**

```python
def test_validate_result_rejects_error_or_unfinished_lane(self):
    with self.assertRaisesRegex(RuntimeError, "did not finish cleanly"):
        module.validate_completion(total=8, completed=7, errors=1)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `python3 -m unittest -v engine_cuda.tests.test_benchmark_0037_update32_cuda_rollout`

Expected: `validate_completion` is missing.

- [ ] **Step 3: Implement model loading and the resident loop**

```python
with torch.inference_mode():
    while decisions < max_decisions:
        statuses = engine.statuses()
        semantic = semantic0031_v2_ready_batch(
            engine.encode_semantic0031_v2_lanes(lanes), max_action_steps=64
        )
        actor = semantic["global_cat"][:, 3].long() - 1
        learner_turn = statuses.eq(NEEDS_ACTION) & actor.eq(focal_players)
        learner_actions, learner_lengths = learner.act_device_semantic0031_v2(
            semantic, route_mask=learner_turn
        )
        opponent_actions, opponent_lengths = opponent.act_device_semantic0031_v2(
            semantic, route_mask=statuses.eq(NEEDS_ACTION) & ~learner_turn
        )
        engine.pack_actions(
            torch.where(learner_turn[:, None], learner_actions, opponent_actions),
            torch.where(learner_turn, learner_lengths, opponent_lengths),
        )
        engine.apply_packed_actions()
        engine.advance_to_decision()
```

- [ ] **Step 4: Run 8-game smoke then 64/512-game benchmarks**

Run the CLI with `--game-limit 8`, then `64`, then `512`; require every run to report all lanes terminal, zero engine errors, and zero host action/observation copies.

- [ ] **Step 5: Run focused regression tests**

Run: `python3 -m unittest -v engine_cuda.tests.test_benchmark_0037_update32_cuda_rollout engine_cuda.tests.test_semantic0031_bridge engine_cuda.tests.test_policy_pool_torch`

Expected: all tests pass (CUDA-dependent tests may skip only when their documented extension precondition is absent).

### Task 3: CPU/CUDA throughput interpretation

**Files:**
- Modify when evidence is complete: `experiments/0037_dragapult_value_initialized_rl/DESIGN.md`
- Modify when evidence is complete: `experiments/0037_dragapult_value_initialized_rl/DESIGN.html`

**Interfaces:**
- Consumes: CUDA benchmark JSON and existing 0037 N16E8/I8 official CPU-engine timing.
- Produces: an explicit throughput ratio and a go/no-go statement for collector integration.

- [ ] **Step 1: Compare like-for-like scope**

Compute both raw complete-games/s and a scope note: CUDA includes both actor forwards and engine transitions but does not yet reproduce the official separate Search-seed ABI or PPO optimization/backward.

- [ ] **Step 2: Record the evidence boundary**

Update both DESIGN formats only if the complete CUDA benchmark passes. State that official CPU-engine frozen evaluation remains authoritative until a paired fixed-action CPU/CUDA gate covers the exact 007 schedule and runtime ABI.

- [ ] **Step 3: Verify documentation consistency**

Run: `git diff --check` and compare all hashes/timings in DESIGN against the emitted JSON.

