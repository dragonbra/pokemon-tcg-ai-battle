# U270 CPU CUDA Seed Parity Diagnostic Plan

> **For agentic workers:** Execute inline and keep all diagnostic artifacts under `.tmp/evaluation/0040_cpu_cuda_seed_parity/`.

**Goal:** Determine whether U270 official CPU and resident CUDA runs are individually deterministic and whether the canonical Frozen-0806 seed identifies the same initial game and trajectory.

**Architecture:** Reuse the project Frozen-0806 job builder, official seeded CPU runtime, resident CUDA collector, U270 model-only checkpoint, and existing lockstep parity utilities. Add only compact diagnostic capture under `.tmp`; do not alter engine, model, evaluation distribution, or training semantics.

**Tech Stack:** Python 3.11, PyTorch, official seeded engine runtime, `ptcg_cuda_engine`, project 0040 rollout collectors.

## Global Constraints

- Checkpoint: `rl_runs/0040_dragapult_0809_action_boundary_rl/versions/V2_snapshot_loader_fix_long_run/checkpoint/update-000270.pt`.
- Master evaluation seed: `341512806`; report derived per-game engine, search, and policy seeds.
- Four canonical Frozen-0806 jobs, greedy action selection, CUDA batch size one.
- Stop cross-backend trajectory comparison if initial game identity differs.
- No engine semantics, model weights, sampling distribution, or official source changes.

---

### Task 1: Verify Repeatability

**Files:**
- Use: `.tmp/evaluation/0040_cpu_cuda_seed_parity/run.py`
- Produce: `.tmp/evaluation/0040_cpu_cuda_seed_parity/report.json`

- [ ] Run the four U270 CPU jobs twice and compare compact per-game signatures.
- [ ] Run the same four U270 CUDA batch-one jobs twice and compare compact per-game signatures.
- [ ] Confirm checkpoint and canonical schedule hashes before accepting results.

### Task 2: Verify Initial Game Identity

**Files:**
- Use: `train/0040_dragapult_0809_action_boundary_rl/semantic_parity/run_gate_d_lockstep.py`
- Produce: `.tmp/evaluation/0040_cpu_cuda_seed_parity/lockstep.json`

- [ ] Run CPU and CUDA engines from identical focal/opponent decks and each derived per-game engine seed.
- [ ] Compare actor/priority, initial canonical observation tensors, and legal-action tensors before applying a policy action.
- [ ] Record opponent, toss/seat metadata, initial hashes, and compact mismatch fields.

### Task 3: Locate First Divergence

**Files:**
- Produce: `.tmp/evaluation/0040_cpu_cuda_seed_parity/lockstep.json`

- [ ] Continue decision-by-decision only for games whose initialization passes.
- [ ] Compare observation/features, legal actions, greedy chosen action, and post-action engine state.
- [ ] Classify the earliest mismatch as A/B/C/D/E, or F when the full game matches.

### Task 4: Report Evidence

- [ ] Cross-check generated JSON, exact commands, checkpoint hash, schedule hash, seed derivation, batch size, and git diff.
- [ ] Deliver explicit PASS/FAIL results, first divergence evidence, likely cause bounded by observed evidence, and the next smallest experiment.
