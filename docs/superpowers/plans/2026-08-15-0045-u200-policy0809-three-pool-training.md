# 0045 U200 Policy-0809 Three-Pool Training Plan

**Goal:** Continue the deck-007 specialist from V3 U200 in a new immutable version, with targeted rollout Meta weights and three separately logged Policy-0809 CUDA-512 evaluation pools every five updates.

**Architecture:** Keep the 0045 minimal Actor/Critic topology and PPO objective unchanged. Start a fresh optimizer from the exact V3 U200 model-only checkpoint and preserve Frozen-0045-Init U0 as the reference anchor. Rollout remains against immutable Champion-G2; only its exact-deck schedule changes. Periodic evaluation materializes one deployment-effective candidate per checkpoint and evaluates three disjoint, Meta-balanced pools against complete immutable Policy-0809.

**Tech Stack:** Python, PyTorch, CUDA Engine 2.0, official engine runtime, pytest, TensorBoard, W&B.

---

### Task 1: Freeze the new schedule contracts in tests

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/tests/test_minimal_optimizer_and_tiny_v2.py`
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/benchmark_tiny_v2_three_pool_schedule.py`

1. Add tests requiring three disjoint active-Meta pools: `02/03/05`, `00/01/04/27`, and all remaining non-empty Meta classes.
2. Require exactly 512 jobs per pool, near-uniform Meta counts, complete immutable `Policy-0809`, unique seeds, and common random schedules independent of focal deployment identity.
3. Run the focused tests and confirm the missing implementation fails before adding it.
4. Implement the deterministic schedule and rerun the focused tests.

### Task 2: Add one-materialization three-pool evaluation

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/run_benchmark_tiny_v2_three_pool.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/training/periodic_evaluation.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_minimal_optimizer_and_tiny_v2.py`

1. Add metric-contract tests for separate `eval/low_score/*`, `eval/priority/*`, and `eval/remaining/*` curves plus a clearly named 1,536-game macro/aggregate diagnostic.
2. Materialize the focal candidate once, strict-load FP16 storage to FP32 runtime, verify candidate/opponent storage independence, and run all 1,536 jobs against Policy-0809.
3. Produce one parent report with three 512-game pool reports/summaries, per-Meta rows, deployment hashes, CUDA identity, and hard PASS gates.
4. Extract separately stepped W&B metrics at the exact checkpoint update.

### Task 3: Add the V4 continuation entry point

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/training/run_v4_u200_policy0809_three_pool.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/training/run_v1.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_minimal_optimizer_and_tiny_v2.py`

1. Parameterize the generic loop with the new periodic evaluation profile while retaining the old Tiny V2 path for historical versions.
2. Bind the new version to exact V3 U200 and Frozen-0045-Init hashes, fresh optimizer, fresh on-policy data, fixed deck 007, unchanged cold-start LR/PPO settings, and unlimited updates by default.
3. Set rollout Meta weights to `03=6`, `05=6`, `00/01/02/04=3`, `27=1`, and every other active Meta implicitly `1`.
4. Keep rollout opponent policy `Champion-G2`; bind all three evaluation pools to `Policy-0809`.
5. Emit a readiness audit that fails on used paths or identity/config drift.

### Task 4: Synchronize authoritative documentation

**Files:**
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/decisions.md`

1. Document the V4 parent/reference split, fresh optimizer boundary, rollout weights, opponent identity distinction, three pool definitions, cadence, and W&B namespaces.
2. State that the model architecture, PPO objective, LR profile, rollout size, and Frozen-0045-Init reference are unchanged.

### Task 5: Verify and launch

1. Run the focused tests, then the complete 0045 test suite.
2. Run readiness and a short schedule/report smoke without mutating historical versions.
3. Execute the U200 three-pool baseline evaluation; require 3 x 512 terminal games, zero error/unfinished, identity PASS, and one focal materialization.
4. Launch V4 online with no update cap, verify its process, W&B identity, canonical config/status, and first fresh rollout source update.

