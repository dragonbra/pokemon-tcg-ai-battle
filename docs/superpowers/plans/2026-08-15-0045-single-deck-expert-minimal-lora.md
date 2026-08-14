# 0045 Single-Deck Expert Minimal-LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Dragapult deck-007 PPO specialist whose Actor has no Critic-derived inference inputs, while preserving the 0044 semantic backbone, Q/V LoRA, decoder, allocation head, Critic, PPO objective, and immutable generalist opponent.

**Architecture:** Copy the permitted 0044 implementation and immutable assets into a new `0045_single_deck_expert_minimal_lora` sibling, then replace the strategy-conditioned Actor API with a minimal policy-only path. An explicit fail-closed migrator maps compatible 0044 tensors into the new schema, creates `Frozen-0045-Init`, and emits tensor equality and identity evidence. CUDA Benchmark Tiny V2 uses a deterministic 512-game subset with deployment-effective FP16-storage/FP32-runtime materialization every five updates; multiples of ten also satisfy the formal periodic-evaluate cadence without a duplicate run.

**Tech Stack:** Python 3.11, PyTorch, pytest, repository semantic runtime, official CUDA engine runtime, JSON/JSONL, TensorBoard, W&B online.

## Global Constraints

- Do not mutate `train/0044_g2_dragapult_policy_option_lora`, its run assets, or `engine/source/`.
- Read and enforce `docs/rl/RL_PROMOTE_CHAMPION_FROZEN_POLICY_PROTOCOL_V1.md`; requested/materialized policy identity mismatches are fatal.
- 0045 may not import executable code from another numbered `train/<project_id>/`.
- The fixed focal deck is exact deck `007`; exact 60-card/resource features remain in the frozen semantic observation.
- Actor trainables are only the existing Action Decoder, Allocation Head, and final Option-block Q/V LoRA with rank 4 and alpha 8.
- Actor inference must never consume Critic output; `detach()` is not an acceptable substitute.
- Critic keeps the 0044 latent-query trunk, `V_win`, `V_prize`, opponent-Meta head, and Value adapter, warm-started from the selected source checkpoint.
- Use a fresh optimizer and preserve the source 0044 PPO settings, including prize advantage, for V1.
- PPO/model-only checkpoints remain FP32 with retention `all`; candidate evaluation uses `kaggle_fp16_storage_fp32_runtime_v1`.
- W&B formal runs use private project `dragon_bra/pokemon-tcg-policy-learning`; `training_metrics.jsonl` is canonical.
- Do not start the long 0045 run until the exact source checkpoint and exact generalist opponent identity are unambiguous and all hard gates pass.
- No commits or external publication are made unless the user separately requests them.

---

### Task 1: Freeze the audited 0044 source facts

**Files:**
- Create: `experiments/0045_single_deck_expert_minimal_lora/source_audit.json`
- Create: `experiments/0045_single_deck_expert_minimal_lora/decisions.md`

**Interfaces:**
- Consumes: live V24 `status.json`, checkpoint metadata, 0044 model source, policy registry, deck registry.
- Produces: immutable source checkpoint locator/hash, exact deck-007 identity, exact opponent policy ID/hash, pre-change forward graph, and preserved PPO config.

- [ ] Record the currently running process, version, latest complete checkpoint, source-policy update, checkpoint SHA-256, and model schema without stopping it.
- [ ] Record the exact 0044 Actor and Critic call graph from runtime code, including every `StrategyContext` tensor consumed by Actor.
- [ ] Resolve deck `007` to its exact 60-card content hash and Own-Archetype mapping used only by the Critic.
- [ ] Resolve both the prose-requested G2 identity and active V24 opponent identity; write a fatal ambiguity gate if they differ.
- [ ] Record the complete source PPO config and Benchmark V2 schedule identity.
- [ ] Validate the JSON against exact on-disk hashes and fail if any source changes during the audit.

### Task 2: Create the self-contained 0045 project and authority documents

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/`
- Create: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- Create: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`
- Create: `experiments/0045_single_deck_expert_minimal_lora/manifest.json`
- Create: `experiments/0045_single_deck_expert_minimal_lora/evaluation/index.html`

**Interfaces:**
- Consumes: Task 1 source facts and copied immutable 0044 assets.
- Produces: importable 0045 package with no `train.0044_*` executable dependency and synchronized design authorities.

- [ ] Copy the required semantic runtime, policy, rollout, CUDA, evaluation, training support, exact decks, and immutable policy assets while excluding caches and old run outputs.
- [ ] Rename project/schema/contract identifiers to 0045 and scan source/bytecode for forbidden cross-numbered imports.
- [ ] Write `DESIGN.md` with exact tensor shapes, pre/post graphs, trainable topology, losses, deployment identity, Tiny V2 cadence, and expansion ladder disabled.
- [ ] Render equivalent `DESIGN.html` and create the project/evaluation manifests.
- [ ] Run an import test and the self-containment scan.

### Task 3: Implement the minimal Actor and absolute Critic separation

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/policy/actor_critic.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/policy/strategy_adapters.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/policy/__init__.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/rollout/cuda_collector.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/cuda_engine_2/inference.py`

**Interfaces:**
- Consumes: `SemanticActorCritic.encode_dual_options(batch)` and frozen semantic state.
- Produces: `encode_policy(batch) -> (validated, state, policy_options)` and `encode_critic(batch) -> (value, auxiliary)` with no shared trainable modules or Actor consumption of auxiliary tensors.

- [ ] Add failing tests showing Critic mutation changes 0044 Actor logits and defining the required 0045 invariance.
- [ ] Replace `DecoderPolicyHead` with a decoder-only head whose `logits(batch, options, state, **kwargs)` directly calls the Action Decoder using the stable pre-adapter policy readout.
- [ ] Remove `PolicyStrategyAdapter`, `MetaActorResidual`, `StrategyContext`, policy-side Own-Archetype lookup, `V_win`, `z_meta`, and opponent-Meta tensors from every Actor call signature.
- [ ] Keep the Critic-only Value adapter/archetype path and dual Option fork; confirm the Allocation Head receives only policy-owned state/options/features.
- [ ] Update CUDA collection and inference call sites to invoke the policy path independently from value computation.
- [ ] Run focused Actor/Critic forward and separation tests.

### Task 4: Implement strict 0044-to-0045 migration and Frozen-0045-Init

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/migration/__init__.py`
- Create: `train/0045_single_deck_expert_minimal_lora/migration/from_0044.py`
- Create: `train/0045_single_deck_expert_minimal_lora/policy/policy_only_export.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_migration.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_policy_only_export.py`

**Interfaces:**
- Produces: `migrate_0044_checkpoint(source, destination, report_path, deck_id='007') -> MigrationReport` and `export_policy_only(model, path) -> DeploymentIdentity`.

- [ ] Enumerate explicit inherited prefixes for backbone, base Option Encoder, policy Q/V LoRA, decoder, allocation head, and complete Critic.
- [ ] Enumerate retired 0044-only policy prefixes and reject all unclassified source tensors.
- [ ] Strictly reject inherited shape mismatches, missing keys, unexpected keys, non-equal copied tensors, and silent initialization of compatible tensors.
- [ ] Emit counts/names for copied, dropped, newly initialized, missing, unexpected, and mismatched tensors plus before/after equality hashes.
- [ ] Serialize a model-only U0 checkpoint and immutable same-architecture `Frozen-0045-Init` manifest.
- [ ] Export a Critic-free policy payload and verify identical logits and greedy/sampled actions under fixed RNG.

### Task 5: Rebuild PPO ownership, optimizer, references, and logging

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/training/ppo_full_semantic.py`
- Create: `train/0045_single_deck_expert_minimal_lora/training/run_v1.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/training/lr_profiles.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_gradient_ownership.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_optimizer_groups.py`

**Interfaces:**
- Consumes: migrated U0 model and immutable `Frozen-0045-Init` reference.
- Produces: fresh optimizer groups named `action_decoder`, `allocation_head`, `policy_option_lora`, `value_head`, `value_adapter`, and `prize_aux` only.

- [ ] Remove policy-adapter/meta-residual config, optimizer groups, metrics, and checkpoint fields.
- [ ] Preserve decoder/allocation LR `5e-6`, LoRA LR `1e-5`, Critic LR `2e-5`, clip/entropy/KL/value/prize settings, rollout size 512, PPO epochs 3, and minibatch size 4096 from V24.
- [ ] Make specialist reference KL target `Frozen-0045-Init` and label any historical KL separately.
- [ ] Log required rollout/eval/KL/value/entropy/gradient/update-norm fields and the guarded improvement-per-reference-KL diagnostic.
- [ ] Backpropagate policy-only and critic-only losses separately in tests; assert exact allowed gradient owners and no frozen-backbone gradients.
- [ ] Print and persist actual module/tensor/parameter counts and every optimizer group member.

### Task 6: Implement Benchmark Tiny V2 and the five-update cadence

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/benchmark_tiny_v2_schedule.py`
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/run_benchmark_tiny_v2.py`
- Create: `train/0045_single_deck_expert_minimal_lora/evaluation/render_benchmark_tiny_v2.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_benchmark_tiny_v2.py`

**Interfaces:**
- Produces: deterministic 512-game common-random-number schedule and periodic report metrics keyed by `eval/checkpoint_update`.

- [ ] Derive exact balanced counts over Benchmark V2’s frozen 16 selected Meta classes and decks for 512 jobs.
- [ ] Use independent deterministic engine/search/policy/coin seeds and preserve Agent handling of context 41.
- [ ] Require deployment-effective identity PASS before CUDA evaluation; raw FP32 results remain diagnostics only.
- [ ] Trigger at U5/U10/U15/...; mark U10/U20/... as both Tiny and periodic evaluate without running the same 512 jobs twice.
- [ ] Write total/first/second win rate, completion/error counts, per-meta/per-deck results, and report hashes.
- [ ] Test exact counts, identity independence, unique seeds, deterministic schedule hash, and cadence semantics.

### Task 7: Run all CPU hard gates and synchronize evidence

**Files:**
- Create: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V1_minimal_lora_dragapult_007/artifact/architecture_audit.json`
- Create: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V1_minimal_lora_dragapult_007/artifact/test_results.json`
- Create: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V1_minimal_lora_dragapult_007/artifact/status.json`

**Interfaces:**
- Produces: machine-readable PASS/FAIL evidence for all CPU gates before GPU ownership changes.

- [ ] Run model construction, migration on an isolated copy, Critic→Actor invariance, LoRA→Critic invariance, gradient ownership, export parity, opponent immutability, and self-containment tests.
- [ ] Persist pre/post forward graphs, removed dependencies, migration equality, actual parameter counts, optimizer groups, and opponent before/after effective hashes.
- [ ] Cross-check `DESIGN.md` and `DESIGN.html` against code, checkpoint metadata, and manifests.
- [ ] Mark status `gpu_validation_pending` only if every CPU hard gate passes.

### Task 8: Handoff the GPU and validate migrated U0

**Files:**
- Create: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V1_minimal_lora_dragapult_007/artifact/gpu_smoke.json`
- Create: `experiments/0045_single_deck_expert_minimal_lora/evaluation/V1_minimal_lora_dragapult_007_u0_tiny_v2.html`
- Create: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V1_minimal_lora_dragapult_007/artifact/evaluation.json`

**Interfaces:**
- Consumes: all Task 7 PASS evidence and the user-authorized live 0044 process.
- Produces: exact stopped source checkpoint identity, CUDA smoke evidence, formal U0 Tiny V2 evaluation, and a ready-but-not-started long-run command.

- [ ] Wait for a durable complete 0044 checkpoint boundary, verify process/version/PID, send SIGINT, and confirm final status/checkpoint without deleting any 0044 asset.
- [ ] Freeze that exact source path/hash in `source_audit.json`; refuse migration if it differs from the selected checkpoint.
- [ ] Materialize 0045 U0 and `Frozen-0045-Init`, then verify the immutable opponent hash before and after one smoke update.
- [ ] Run a short CUDA rollout/optimization smoke and assert finite losses, low behavior KL, separate Actor/Critic gradients, and no opponent mutation.
- [ ] Run the full 512-game U0 Benchmark Tiny V2 with 0 errors/unfinished and render the formal report/index.
- [ ] Persist the exact long-run command/config, but do not launch it until the source/opponent ambiguity is explicitly resolved and all deliverables are reviewed.

## Self-Review

- Spec coverage: all 20 sections map to Tasks 1–8; expansion ranks 8/16 and earlier-block/state LoRA are documented only and never enabled.
- Placeholder scan: no deferred implementation placeholders are present; runtime-resolved hashes/counts are required outputs rather than guessed constants.
- Type consistency: migration, policy-only export, policy/critic encode APIs, optimizer group names, and Tiny V2 cadence are defined once and reused consistently.
- Execution choice: the user requested continuous implementation in this session, so execution proceeds inline with explicit CPU and GPU checkpoints; no subagent dispatch is used.
