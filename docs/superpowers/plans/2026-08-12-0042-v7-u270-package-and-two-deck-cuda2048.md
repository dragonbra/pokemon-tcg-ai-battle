# 0042 V7 U270 Package and Two-Deck CUDA-2048 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop V7 cleanly, export its U270 checkpoint as a local Kaggle-ready 023 package, and run separate Frozen-0809 CUDA-2048 evaluations of the same U270 weights with the canonical 007 deck and the V6 exact deck.

**Architecture:** Treat checkpoint weights, deployment deck identity, and opponent identity as separate immutable inputs. The 023 package must reproduce U270's already-qualified periodic deployment identity exactly; the 007 and V6 evaluations each rematerialize U270 under their own exact deck and receive distinct effective candidate/schedule hashes. Keep all outputs separate and never submit to Kaggle.

**Tech Stack:** Python 3, PyTorch, 0042 project-local exporter/materializer, resident official CUDA engine, tar/gzip, unittest.

## Global Constraints

- Stop V7 only through `artifact/STOP_REQUESTED`; do not signal or kill a partial update.
- Package source checkpoint is exactly `V7_hydrapple_ex_meganium_023/checkpoint/update-000270.pt`, regardless of the later stop boundary.
- The Kaggle-ready package focal deck remains V7's 023 exact deck and must match U270 periodic Frozen deployment-effective hash `5762bd9483afa89dfb0a6d3d829a26bec6944ce5431a3f7e16900775770c5d8c`.
- Package output is self-contained under `archive/submission/`; compressed archive goes under `archive/submission/dist/`; no root `submission/` directory is created.
- 007 evaluation focal deck is `dragapult_ex_07bedfffbfad`, exact hash `07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`.
- V6 evaluation focal deck is `dragapult_ex_0042_v6`, exact hash `7bdb3bb183008d9204efc68ad77b6df1039ad760c98476aa82d5809f5c446ca3`.
- Both CUDA evaluations use eight 256-game units, independent full immutable Policy-0809 opponents, seeded toss with Agent choice, 0 error/unfinished/semantic fallback, FP16 storage, and strict FP32 runtime.
- Results are two independent identity-bound benchmarks and are not merged or described as paired-game evidence.
- No automatic promotion and no Kaggle submission are authorized.
- Official `engine/source/` remains read-only.

---

### Task 1: Close V7 at a complete update boundary

**Files:**
- Create at runtime: `rl_runs/0042_full_model_design/versions/V7_hydrapple_ex_meganium_023/artifact/STOP_REQUESTED`
- Verify: `rl_runs/0042_full_model_design/versions/V7_hydrapple_ex_meganium_023/artifact/training_summary.json`

**Interfaces:**
- Consumes: the runner's after-complete-update sentinel boundary.
- Produces: a clean immutable terminal status without changing U270.

- [ ] Create the stop sentinel and wait for both trainer/watchdog to exit.
- [ ] Require `state=stopped_by_request`, no error, matching final checkpoint sidecar, and W&B final sync evidence.

### Task 2: Validate U270 qualifying evidence

**Files:**
- Read: `rl_runs/0042_full_model_design/versions/V7_hydrapple_ex_meganium_023/checkpoint/update-000270.pt`
- Read: `rl_runs/0042_full_model_design/versions/V7_hydrapple_ex_meganium_023/artifact/frozen_results/core-update-000270.json`

**Interfaces:**
- Consumes: U270 source checkpoint and periodic Frozen report.
- Produces: fail-closed source/portable/effective identity tuple for packaging.

- [ ] Match checkpoint SHA-256 to its sidecar and Frozen candidate audit source hash.
- [ ] Require 2,048 terminal games, zero errors/unfinished/semantic fallback, candidate audit PASS, and full Policy-0809 opponent audit PASS.
- [ ] Record U270 023 W-L-D, schedule hash, portable hash, and deployment-effective hash without treating it as a new evaluation.

### Task 3: Export and validate the 023 Kaggle-ready package

**Files:**
- Create: `archive/submission/0042_hydrapple_ex_meganium_023_policy0809_rl_v7_u270_fp16_storage_fp32_runtime/`
- Create: `archive/submission/dist/0042_hydrapple_ex_meganium_023_policy0809_rl_v7_u270_fp16_storage_fp32_runtime.tar.gz`

**Interfaces:**
- Consumes: `export_candidate(... deployment_deck=023 ..., require_frozen_selection=True)` and U270 qualifying evidence.
- Produces: self-contained exact-60 Kaggle package plus deterministic archive inventory.

- [ ] Export the package with exact deck 023 and require manifest/source checkpoint/Frozen evidence agreement.
- [ ] Strict-load the packaged FP16 model into FP32 runtime and require own-archetype class 6, exact deck hash, and effective hash equal to U270's periodic Frozen audit.
- [ ] Run package validator and package-focused tests; reject symlinks, forbidden recovery state, missing `cg/`, or non-60 deck.
- [ ] Create the tarball with the package directory as its single top-level member; verify extraction inventory and record SHA-256.

### Task 4: Run U270 with canonical 007 CUDA-2048

**Files:**
- Create: `.tmp/evaluation/0042_v7_u270_two_deck_cuda2048/007_dragapult_ex/`

**Interfaces:**
- Consumes: U270 checkpoint plus canonical 007 exact deck.
- Produces: one independent custom-deck Frozen-0809 CUDA-2048 report.

- [ ] Materialize U270 with 007 under `kaggle_fp16_storage_fp32_runtime_v1` and require candidate/opponent independence.
- [ ] Execute eight units and require 2,048 terminal games, zero errors/unsupported/semantic fallback, lane-routing PASS, and complete toss/seat evidence.
- [ ] Persist W-L-D, first/second rates, Wilson interval, throughput, candidate identity, opponent identity, and schedule hash.

### Task 5: Run U270 with V6 exact deck CUDA-2048

**Files:**
- Create: `.tmp/evaluation/0042_v7_u270_two_deck_cuda2048/v6_dragapult_ex/`

**Interfaces:**
- Consumes: U270 checkpoint plus immutable `dragapult_ex_0042_v6` exact deck.
- Produces: a second independent custom-deck Frozen-0809 CUDA-2048 report.

- [ ] Materialize U270 with V6 exact deck and require a deck-specific effective identity distinct from 007 and 023.
- [ ] Execute eight units under the same Frozen contract and health gates.
- [ ] Persist the same complete audit fields and explicitly retain the separate schedule identity.

### Task 6: Final cross-artifact audit

**Files:**
- Create: `.tmp/evaluation/0042_v7_u270_two_deck_cuda2048/summary.json`

**Interfaces:**
- Consumes: package manifest, tarball hash, and both CUDA reports.
- Produces: one concise inventory without combining benchmark scores.

- [ ] Verify all three deployments share U270 source checkpoint SHA but have three distinct exact-deck identities and deployment-effective hashes.
- [ ] Verify both CUDA reports contain 2,048 games and independent schedule hashes.
- [ ] Record package path/hash and the two report paths/results; state `promotion_status=NONE` and `kaggle_submission_performed=false`.
