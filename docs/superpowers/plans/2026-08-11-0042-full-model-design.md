# 0042 Full Model Design Implementation Plan

> Execute this plan in the current workspace. Preserve all existing numbered projects and do not modify `engine/source/`.

**Goal:** Create a self-contained 0042 RL project whose frozen semantic encoders feed a trainable Value trunk and action decoder, with zero-gated Value and readout-only Policy Strategy Adapters, strict Value-to-Policy stop-gradient, pretrained 15-class Meta retention, and auditable checkpoint/runtime contracts.

**Architecture:** Mechanically fork the current 0040 full-semantic PPO implementation into `train/0042_full_model_design/`, then remove Option LoRA and dormant 23-class Meta conditioning. Add one canonical `StrategyContext` built once per root decision from public seat identity, detached q1/meta/value signals, and an independent policy-side own-archetype embedding. Keep the legacy GRU state transition unchanged and adapt only the hidden tensor used by Query/STOP scoring.

**Tech stack:** Python 3.11, PyTorch, existing 0031 semantic policy/checkpoint format, existing 0040 PPO/rollout/export infrastructure, pytest/unittest.

---

### Task 1: Freeze the implementation baseline and project identity

**Files:**
- Create: `train/0042_full_model_design/**`
- Create: `experiments/0042_full_model_design/manifest.json`
- Create: `experiments/0042_full_model_design/DECISIONS.md`
- Create: `train/0042_full_model_design/configs/FULL_MODEL.json`
- Modify: `train/0042_full_model_design/source.py`

1. Copy the executable 0040 project into the new numbered project, excluding caches and audit-only generated output.
2. Replace project IDs, formal preset identity, and output roots with 0042 identities while retaining immutable paired-0809 actor/value hashes.
3. Make the canonical preset explicitly declare `no_option_lora`, 15 own archetypes, 16-wide own embeddings, scalar zero gates, and a non-default `meta_anchor_coef`.
4. Add startup validation that rejects all Option LoRA switches and legacy adapter keys.
5. Run import and preset validation tests.

### Task 2: Implement semantic own-archetype and strategy context contracts

**Files:**
- Create: `train/0042_full_model_design/policy/strategy_context.py`
- Create: `train/0042_full_model_design/policy/strategy_adapters.py`
- Modify: `train/0042_full_model_design/policy/__init__.py`
- Modify: `train/0042_full_model_design/rollout/collector.py`
- Modify: `train/0042_full_model_design/training/batch_full_semantic.py`

1. Define separate `OwnArchetypeId` and opponent-label contracts, taxonomy version/hash, `Others`, and exact-deck-to-own-archetype resolution.
2. Define validated first/second one-hot input sourced from the registered match/observation identity, never from opponent hidden state.
3. Implement immutable-per-root `StrategyContext` carrying detached q1, meta probabilities, adapted Value, public side, and policy own embedding.
4. Ensure rollout transitions preserve only the public strategy inputs required to deterministically reconstruct PPO replay context; keep opponent Meta targets training-only.
5. Add unit tests for taxonomy failure, unsupported deck fallback, target/input separation, and fixed context reuse.

### Task 3: Implement the 0042 Value side

**Files:**
- Modify: `train/0042_full_model_design/policy/actor_critic.py`
- Modify: `train/0042_full_model_design/policy/value_network.py`
- Test: `train/0042_full_model_design/tests/test_strategy_model.py`

1. Expose decoded latents explicitly as q0 `z_v` and q1 `z_m`.
2. Freeze the pretrained 15-class `value_head.heads.archetype` parameters while retaining its autograd graph.
3. Add independent value-owned LayerNorm, `E_V_own`, normally initialized MLP(336,320,320), and scalar `g_V=0` on q0 only.
4. Feed adapted q0 through the existing scalar Value head and preserve `2*sigmoid(logit)-1` semantics.
5. Remove dormant 23-class `OpponentMetaHead(state.summary)` and conditioner from canonical construction.
6. Assert Prototype/State/Option encoders are frozen and that no parametrized LoRA wrapper exists.

### Task 4: Implement readout-only Policy Strategy Adapter

**Files:**
- Modify: `train/0042_full_model_design/semantic_policy/model/action_decoder.py`
- Modify: `train/0042_full_model_design/policy/action_distribution.py`
- Modify: `train/0042_full_model_design/policy/compound_evaluation.py`
- Modify: `train/0042_full_model_design/policy/actor_critic.py`
- Modify: relevant copied Kaggle/runtime decoder files under `train/0042_full_model_design/kaggle_runtime/`

1. Add policy-owned hidden and Meta-context LayerNorms, independent `E_pi_own`, MLP(674,320,320), and scalar `g_pi=0`.
2. Build context once after Value decode and pass it through sampling, greedy, replay, and compound-action evaluation.
3. Adapt `state.hidden` only inside Query/STOP logit evaluation; keep `consume(..., state.hidden)` byte-for-byte on the legacy recurrent hidden.
4. Share a single readout helper across all policy execution paths.
5. Port the identical contract to export/inference runtime and reject packages missing strategy metadata or weights.

### Task 5: Replace optimizer and Meta-retention loss contracts

**Files:**
- Modify: `train/0042_full_model_design/training/ppo_full_semantic.py`
- Modify: `train/0042_full_model_design/integrated/config.py`
- Modify: `train/0042_full_model_design/integrated/loss_registry.py`
- Modify: `train/0042_full_model_design/training/run_full_semantic.py`

1. Delete all Option LoRA and option LayerNorm optimizer groups.
2. Add explicit `value_adapter` and `policy_strategy_adapter` groups using their parent branch LRs, with separately auditable gates/embeddings in the manifests.
3. Add weighted pretrained 15-class Meta CE using `meta_anchor_coef`; preserve frozen MetaHead parameters while allowing gradients into q1 and Value latent blocks.
4. Preserve existing PPO, GAE, reward, clipping, opponent, and episode semantics.
5. Log gate values, effective residual ratios, Meta loss/accuracy/entropy/turn buckets, and requested gradient norms.

### Task 6: Implement strict 0042 checkpoint and initialization

**Files:**
- Modify: `train/0042_full_model_design/training/storage_full_semantic.py`
- Modify: `train/0042_full_model_design/initialization.py`
- Modify: `train/0042_full_model_design/export_full_semantic_candidate.py`
- Test: `train/0042_full_model_design/tests/test_strategy_checkpoint.py`

1. Define a model-only 0042 schema containing complete trainable Value/Policy/adapters plus base hashes, taxonomy version/hash, dimensions, and `no_option_lora=true`.
2. Implement an explicit base initialization path that strict-loads the paired pretrained actor/value and initializes only enumerated 0042 keys with zero gates.
3. Make formal 0042 round-trip strict: missing adapter keys, unexpected adapter keys, taxonomy mismatch, and Option LoRA keys are fatal.
4. Preserve the repository ban on optimizer/scheduler/RNG/rollout state in model-only checkpoints.
5. Extend candidate materialization without changing the canonical deployment precision protocol.

### Task 7: Add architecture, gradient, parity, and diagnostics regressions

**Files:**
- Create: `train/0042_full_model_design/tests/test_strategy_architecture.py`
- Create: `train/0042_full_model_design/tests/test_strategy_gradients.py`
- Create: `train/0042_full_model_design/tests/test_strategy_policy_paths.py`
- Create: `train/0042_full_model_design/diagnostics/strategy_sensitivity.py`

1. Prove OptionEncoder trainable count is zero, Option LoRA is absent, optimizer groups exclude Option adaptation, and Value/Policy embeddings/norms share no Parameter objects.
2. Run independent policy/value/meta backward tests and record tensor-level nonzero gradient matrices.
3. Preserve two-step gate startup regressions for both adapters.
4. Compare real-checkpoint CPU FP32 base vs zero-gate model for logits, Value, Meta logits, masks, and greedy action with exact equality.
5. Prove adapter-enabled scoring can alter logits but never the GRU transition, and prove sampling/replay/greedy/compound/export use the same fixed context/readout helper.
6. Add V and Meta counterfactual sensitivity diagnostics reporting KL, entropy, and action rankings without changing training semantics.

### Task 8: Synchronize authoritative design and onboarding documentation

**Files:**
- Create: `experiments/0042_full_model_design/DESIGN.md`
- Create: `experiments/0042_full_model_design/DESIGN.html`
- Create: `docs/rl/0042_full_model_design.md`

1. Document the verified 0031 architecture and checkpoint contract from current code.
2. Document 0031-to-0042 changes, tensor shapes, q0/q1 roles, readout-only adapter, detach boundary, own/opponent archetype distinction, and removal of Option LoRA.
3. Document frozen/trainable modules, optimizer groups, gradient paths, initialization, strict checkpoints, runtime parity, diagnostics, and policy identity boundary.
4. Cross-check all counts/shapes/paths against tests and current checkpoint metadata.

### Task 9: End-to-end verification and factual report

**Files:**
- Modify only 0042 files if failures reveal 0042 defects.

1. Run focused 0042 unit tests.
2. Run copied baseline contract tests affected by project isolation and execution paths.
3. Run model construction against the real paired actor/value checkpoints.
4. Run the diagnostic gradient inventory and zero-gate parity fixture.
5. Run `git diff --check` and inspect the final diff for accidental changes outside 0042/docs.
6. Report exact parameter inventory, gradient matrix, zero-gate results, execution-path parity, checkpoint schema, and any remaining blocker without starting a training run or formal evaluation.
