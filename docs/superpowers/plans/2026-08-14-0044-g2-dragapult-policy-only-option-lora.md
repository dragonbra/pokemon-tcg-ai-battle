# 0044 G2 Dragapult Policy-Only Option LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained 0044 RL project that initializes exact focal deck `007` from immutable Champion-G2, adds a fresh rank-4/alpha-8 Last Option Q/V LoRA exclusively to the policy branch, preserves the LoRA-free G2 Value branch, and trains with stabilized post-G2 learning rates.

**Architecture:** Freeze the complete 0043 Champion-G2 effective policy as the 0044 update-0 source. Share prototype, state, Option input, and Option block 0 computation; execute Option block 1 twice from the same pre-block tensor and state memory: the frozen original branch feeds Value, while a frozen-base copy carrying zero-delta Q/V LoRA feeds the Action Decoder. Keep opponent policies independently resolved and immutable, make the start-of-run complete G2 policy the reference snapshot, and version all new checkpoint/export identities rather than weakening 0043 `no_option_lora` contracts.

**Tech Stack:** Python 3, PyTorch, AdamW/PPO Protocol V2, CUDA Engine 2.0 resident rollout, official engine evaluation, W&B online logging, JSON manifests, unittest/pytest.

## Global Constraints

- Project ID is `0044_g2_dragapult_policy_option_lora`; runtime code must not import executable modules from another numbered project.
- Existing `train/0043_champion_league_rl`, Champion-G2, and 0043 run/evaluation assets remain unmodified.
- Copy only executable source and required immutable runtime assets; never copy `rl_runs/0043_champion_league_rl`, caches, generated reports, optimizer state, replay, or rollout buffers.
- Focal exact deck identity is `007`; deck IDs remain the canonical zero-padded `001`–`067` identities.
- V1 has exactly one opponent policy: immutable complete `Champion-G2`. `Policy-0809` and Champion-G1 are not part of this run. PFSP remains enabled on the opponent-deck dimension.
- Preserve the audited 256-lane `128 PFSP + 64 uniform + 64 latest-champion` branch contract. Because the active policy pool is the singleton `{Champion-G2}`, `uniform` and `latest-champion` are policy-equivalent and both draw decks uniformly from exact IDs `001`–`067`; the effective deck mixture is therefore 50% deck-level PFSP plus 50% uniform exploration. Every lane binds its sampled deck to Champion-G2's matching own-deck embedding row and exact static deck fields.
- PFSP uses the focal policy's result against each opponent deck (equivalently the `(deck_id, Champion-G2)` joint key in this singleton-policy run), starts from the Beta(1,1) uniform prior, and freezes a reproducible curriculum for ten updates before recalibration. Low focal win-rate decks receive greater PFSP mass, subject to the recorded floor/cap.
- After each successfully saved checkpoint whose update is divisible by 10 (`U10`, `U20`, ...), run the formal Benchmark V1 CUDA-2048 contract with focal deck `007`, the deployment-effective candidate materialized from that exact checkpoint, and immutable Champion-G2 opponents. Benchmark V1 allocates games as evenly as possible across non-empty Own Archetype V2 classes and then across their member decks.
- Periodic Benchmark V1 is greedy, official-engine, identity-gated evaluation. It writes `eval/*` with `eval/checkpoint_update`; it never updates PFSP, never enters PPO targets, and is never combined with sampled `rollout/*` strength claims.
- Initialization source is immutable `Champion-G2`, source update 407, with all effective decoder, strategy/value adapters, allocation, prize, and 29-row own-deck embedding tensors preserved.
- Last Option LoRA is exactly block index `1`, rank `4`, alpha `8`, Q/V rows only in self-attention and state cross-attention, with Kaiming A and zero B for exact zero delta; trainable inventory is exactly 8 tensors and 10,240 parameters.
- Value, Meta anchor, and prize-Value losses must have zero gradient into Option LoRA. Policy, entropy, reference-KL, and Prize actor advantage may update Option LoRA.
- Value consumes the frozen, LoRA-free Option block-1 output. Policy/decoder and allocation policy consume the LoRA branch where semantically applicable.
- Update-0 focal logits, values, macro logits, legal actions, and greedy actions must match Champion-G2 exactly on fixed CPU and CUDA fixtures.
- Reference KL is relative to a frozen complete Champion-G2 snapshot materialized at run start, not Policy-0809 or Champion-G1.
- Initial LR contract: Action Decoder `1e-5`, Allocation Head `1e-5`, Last Option Q/V LoRA `2e-5`, Policy Strategy Adapter `1e-5`, Value Win/Value Adapter/Value Prize `1e-4`, weight decay `0`. This is a distinct experiment contract and may only change in a new version.
- PPO remains Protocol V2: 256 on-policy games/update, three no-replacement epochs, logical minibatch 2048, forward microbatch 1024, entropy coefficient `0.003`, target behavior KL `0.015`, hard behavior-KL guard `0.025`, reference-KL coefficient `0.02`.
- PPO behavior/model checkpoints remain FP32 model-only with retention `all`. Kaggle-facing evidence remains `kaggle_fp16_storage_fp32_runtime_v1`.
- All opponent policy IDs must resolve and pass full effective-identity audits before CUDA routing; no focal/opponent tensor or mutable cache sharing.
- Do not modify `engine/source/`.
- Any model/input/loss/checkpoint change must be reflected in both `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md` and `DESIGN.html`.

---

### Task 1: Freeze the 0044 project boundary and G2 source identity

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/`
- Create: `experiments/0044_g2_dragapult_policy_option_lora/manifest.json`
- Create: `experiments/0044_g2_dragapult_policy_option_lora/DECISIONS.md`
- Create: `train/0044_g2_dragapult_policy_option_lora/assets/source_snapshot.json`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_project_identity.py`

**Interfaces:**
- Consumes: current 0043 executable tree and immutable `assets/policies/definitions/champion_g002/{model.bin,source_update_000407.pt}`.
- Produces: `PROJECT_ID`, project-local paths, a content-hashed source snapshot, and a test that rejects imports containing `train.0043_` or filesystem paths into numbered projects.

- [ ] Copy the current 0043 executable/runtime tree while excluding caches and generated run/evaluation outputs.
- [ ] Mechanically rename package references, project IDs, display names, and default version paths to 0044.
- [ ] Record every copied source/asset SHA-256 and the exact dirty-source commit/worktree provenance in `source_snapshot.json`.
- [ ] Add a source scan test that rejects cross-numbered runtime imports and non-0044 numbered runtime paths.
- [ ] Run `python3 -m unittest train.0044_g2_dragapult_policy_option_lora.tests.test_project_identity -v`; expect PASS.
- [ ] Commit the isolated project scaffold.

### Task 2: Add the dual Last Option block and zero-delta LoRA

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/policy/option_policy_lora.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/policy/adaptation.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/policy/actor_critic.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/semantic_runtime/model/option_encoder.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_policy_only_option_lora.py`

**Interfaces:**
- Consumes: `OptionEncoder` input construction and two-block TransformerDecoder.
- Produces: `DualOptionEncoding(value_options, policy_options)`, `PolicyOnlyOptionLoRA`, `encode_dual_options(batch, state, prototype_memory)`, and `policy_option_lora_parameters()`.

- [ ] Write failing tests asserting the adapter targets only block-1 self/cross `in_proj_weight` Q/V rows, has 8 tensors/10,240 parameters, and begins at exact zero delta.
- [ ] Split OptionEncoder into reusable `encode_input`, `encode_prefix`, and `encode_final` operations without changing the legacy forward result.
- [ ] Deep-copy only block 1 for the policy branch, freeze its base tensors, and install rank-4/alpha-8 Q/V parametrizations with Kaiming A and zero B.
- [ ] Return LoRA-free final options to Value and LoRA-adapted final options to decoder/policy callers.
- [ ] Assert no copied base tensor is trainable or serialized as an RL delta.
- [ ] Run the focused test; expect exact U0 equality and inventory PASS.
- [ ] Commit the dual-option architecture.

### Task 3: Prove policy/value gradient isolation

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/policy/actor_critic.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/policy/action_distribution.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/ppo_full_semantic.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/rollout/cuda_collector.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_option_lora_gradient_isolation.py`

**Interfaces:**
- Consumes: `DualOptionEncoding`.
- Produces: policy evaluation APIs that always receive `policy_options`, Value APIs that accept only `value_options`, and gradient diagnostics by loss family.

- [ ] Write independent backward tests for policy, terminal Value, Meta anchor, and prize-Value losses.
- [ ] Route Action Decoder and root policy through `policy_options`; route Value memory and Meta anchor through `value_options`.
- [ ] Route macro/allocation actor log-prob through the intended policy branch while retaining its existing state input contract.
- [ ] Assert policy loss yields nonzero LoRA gradient after the zero-output factors receive their first update; assert every Value-family loss yields exactly no LoRA gradient.
- [ ] Add W&B scalars for LoRA gradient norm, delta norm, relative output drift, and forbidden Value-to-LoRA gradient norm (required zero).
- [ ] Run the isolation test suite; expect PASS.
- [ ] Commit the gradient boundary.

### Task 4: Load immutable G2 as exact 007 update-0 and snapshot the reference

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/initialize_g2_dragapult.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/policy/actor_critic.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/policy_identity.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_g2_initialization.py`

**Interfaces:**
- Consumes: project-local Champion-G2 portable artifact, its source U407 checkpoint, deck `007`, and 29-way own-archetype mapping.
- Produces: `build_update0_model(deck_id="007")`, immutable complete-G2 `DecoderPolicyHead` reference, and a signed initialization manifest.

- [ ] Write tests comparing G2 and 0044 U0 tensor identities for all inherited components and exact forward outputs.
- [ ] Strict-load G2 into the expanded 0044 architecture, allowing only the eight fresh LoRA tensors to be absent.
- [ ] Bind deck `007` to its exact deck content hash and own-archetype row.
- [ ] Snapshot complete G2 decoder plus strategy adapter as the reference policy before any optimizer step.
- [ ] Assert the reference has no shared Parameter/storage with the focal trainable modules and remains byte-stable after a synthetic focal update.
- [ ] Run CPU and CUDA zero-step parity; expect exact masks/actions and documented FP32 numerical equality.
- [ ] Commit the initialization/reference contract.

### Task 5: Install stabilized optimizer and PPO configuration

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/config.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/training/ppo_full_semantic.py`
- Create: `train/0044_g2_dragapult_policy_option_lora/configs/G2_007_LORA_STABLE.json`
- Create: `train/0044_g2_dragapult_policy_option_lora/training/run_v1.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_optimizer_contract.py`

**Interfaces:**
- Consumes: isolated trainable groups from Tasks 2–4.
- Produces: seven disjoint AdamW groups and the formal `V1_g2_007_policy_option_lora_stable_lr` launcher.

- [ ] Write a failing optimizer inventory test for exact group names, counts, learning rates, no duplicates, no frozen parameters, and zero weight decay.
- [ ] Configure Decoder/Allocation/Strategy/LoRA LRs as `1e-5/1e-5/1e-5/2e-5`; preserve all Value-family LRs at `1e-4`.
- [ ] Preserve PPO Protocol V2, the audited `128/64/64` PFSP/uniform/latest schedule, and CUDA Engine 2.0 settings inherited from the approved 0043 line; collapse only the active policy pool to singleton Champion-G2 while keeping deck-level PFSP over 001–067.
- [ ] Add synchronous every-10-update Benchmark V1 CUDA-2048 evaluation, fail-closed candidate deployment materialization, resumable per-checkpoint evidence, and W&B `eval/*` telemetry without feeding results into PFSP/PPO.
- [ ] Record `rollout/source_policy_update`, `checkpoint/update`, current and base LR per group, reference KL to G2, behavior KL, adapter/LoRA drift, all 67 per-deck game/win/prize counts, and sampled-deck coverage/entropy.
- [ ] Add manual-stop semantics and model-only retention `all` under the new 0044 version directory.
- [ ] Run optimizer/config regression tests; expect PASS.
- [ ] Commit the formal training configuration.

### Task 6: Version checkpoint, export, and deployment identity

**Files:**
- Create: `train/0044_g2_dragapult_policy_option_lora/checkpoint.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/evaluation/candidate.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/evaluation/export_kaggle.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/policy_identity.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_checkpoint_and_export.py`

**Interfaces:**
- Consumes: G2 base identity plus 0044 trainable deltas.
- Produces: `0044_g2_policy_option_lora_model_only_v1` checkpoints and portable artifacts with LoRA mathematically merged into policy block-1 Q/V weights.

- [ ] Write strict save/load tests accepting exactly the declared trainable inventory and rejecting optimizer/RNG/replay fields.
- [ ] Store G2 parent hashes, deck/taxonomy hashes, LoRA target/rank/alpha, LR contract, and policy/value branch semantics.
- [ ] During candidate materialization, merge LoRA only into the deployed actor's final Option block and preserve all G2 inherited tensors.
- [ ] Convert the complete effective candidate to FP16 storage, strict-load FP32 runtime, and compute deployment-effective content hash.
- [ ] Prove exported logits/actions match the qualifying portable evaluation artifact and hard-fail missing/mismatched LoRA identity.
- [ ] Run checkpoint/export tests; expect PASS.
- [ ] Commit the storage/deployment contract.

### Task 7: Integrate CUDA Engine 2.0 without policy routing regression

**Files:**
- Modify: `train/0044_g2_dragapult_policy_option_lora/cuda_engine_2/inference.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/cuda_engine_2/resident.py`
- Modify: `train/0044_g2_dragapult_policy_option_lora/rollout/cuda_collector.py`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_cuda_dual_option_routing.py`

**Interfaces:**
- Consumes: dual Option branch focal model and independently materialized opponent policies.
- Produces: resident focal dual-branch inference with unchanged opponent policy isolation and lane/deck bindings.

- [ ] Add lockstep tests for focal policy/value option branches and complete opponent identities.
- [ ] Reuse shared Option prefix tensors only within the focal policy identity; never share mutable modules/caches with opponent policies.
- [ ] Ensure G2/Champion opponents execute their own single effective Option path while focal executes the dual final block.
- [ ] Record dual-final-block timing, peak allocation/reservation, throughput, policy loads, lane routing audits, and zero CPU feature fallback.
- [ ] Run a small CUDA rollout and PPO smoke with bounded concurrency; expect zero error/unfinished and identity PASS.
- [ ] Commit CUDA routing support.

### Task 8: Publish the authoritative design and readiness evidence

**Files:**
- Create: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.md`
- Create: `experiments/0044_g2_dragapult_policy_option_lora/DESIGN.html`
- Create: `experiments/0044_g2_dragapult_policy_option_lora/active_training_config.json`
- Create: `experiments/0044_g2_dragapult_policy_option_lora/readiness.json`
- Modify: `docs/rl/0044_Project_Charter.md`
- Test: `train/0044_g2_dragapult_policy_option_lora/tests/test_design_contract.py`

**Interfaces:**
- Consumes: all implemented identities, tensor shapes, optimizer groups, tests, and performance measurements.
- Produces: authoritative human-readable design and a machine-readable READY/NOT_READY gate.

- [ ] Document exact tensor flow, two final-block branches, shapes, gradient ownership, G2 inheritance, deck-007 binding, reference semantics, LR rationale, checkpoint/export boundary, and current/next stage.
- [ ] Render equivalent DESIGN.html content and verify required facts appear in both documents.
- [ ] Run all 0044 unit tests, project isolation scan, model-only checkpoint test, CPU/CUDA zero-step parity, CUDA rollout smoke, PPO one-update smoke, and export/reload parity.
- [ ] Benchmark against the inherited 0043 CUDA path and report the incremental second-final-block cost without claiming strength from throughput tests.
- [ ] Set readiness to READY only when every hard gate passes; otherwise record exact blockers.
- [ ] Commit the completed pre-training work package.

## Self-review

- Spec coverage: G2 initialization, exact 007 focal, LoRA r4/a8/10,240, dual final block, Value detach, stabilized LR, complete G2 reference, CUDA 2.0, checkpoint/export identity, W&B, and design synchronization each map to a task.
- Placeholder scan: no deferred implementation placeholders are permitted in the tasks; each gate has an exact expected invariant.
- Type consistency: `DualOptionEncoding.value_options` is consumed only by Value/Meta; `.policy_options` is consumed by root policy and actor-side action evaluation throughout Tasks 2–7.
