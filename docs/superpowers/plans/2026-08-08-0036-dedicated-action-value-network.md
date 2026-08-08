# 0036 Dedicated Action Value Network Implementation Plan

> Superseded before formal training by `2026-08-08-0036-dragapult007-focal-value-network.md`. The generic 10k implementation and dataset remain pipeline/audit evidence; no generic ablation version was launched.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a frozen-0031-encoder action-decision Value Network on a recent, audited, dual-perspective 10,000-Episode dataset, then make the selected critic loadable by later PPO runs.

**Architecture:** Freeze the self-contained friend-0806 0031 semantic policy, expose its complete encoded state and legal-option memories, and compare two MLP baselines with an eight-query latent cross-attention critic. Train terminal win probability as the primary objective; opponent archetype and signed terminal Prize difference are optional auxiliary objectives, while hidden-hand prediction remains disabled.

**Tech Stack:** Python 3.11, PyTorch, official Kaggle Episode ZIPs, gzip JSONL shards, TensorBoard, W&B online project `dragon_bra/pokemon-tcg-policy-learning`.

## Global Constraints

- Project ID is `0036_dedicated_action_value_network`; executable code is self-contained under `train/0036_dedicated_action_value_network/` and imports no numbered training project.
- `engine/source/` is read-only and is not modified.
- One training example is one non-registration `agent(observation)` callback before its full ordered action, not one turn and not one decoder token.
- Train/validation isolation is by whole Episode; both player perspectives from one Episode always share one split.
- Source/team/exact opponent identity is audit-only and never enters the Value forward path.
- The 0031 friend-0806 epoch-11 model is loaded strictly by SHA-256; its encoder and prototype parameters remain frozen.
- Formal run artifacts use strictly increasing `rl_runs/0036_dedicated_action_value_network/versions/V<n>_<tag>/` paths, model-only checkpoints, all-epoch retention, TensorBoard, canonical JSONL metrics, and W&B online mirroring.
- No Kaggle submission, dataset upload, or public action is authorized by this plan.

---

### Task 1: Freeze project and source identities

**Files:**
- Create: `train/0036_dedicated_action_value_network/{contracts,domain,features,knowledge,model}/**`
- Create: `train/0036_dedicated_action_value_network/assets/*.json`
- Create: `experiments/0036_dedicated_action_value_network/manifest.json`
- Create: `experiments/0036_dedicated_action_value_network/DESIGN.md`
- Create: `experiments/0036_dedicated_action_value_network/DESIGN.html`
- Test: `train/0036_dedicated_action_value_network/tests/test_project_contract.py`

**Interfaces:**
- Consumes: archive asset `0031_friend_0806_epoch11_best_validation_loss`, checkpoint SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`.
- Produces: self-contained `SemanticPolicy`, `DecisionBatch`, compiler, collator, prototype and causal-knowledge modules.

- [ ] Write a failing contract test that rejects imports matching `train.00` and verifies all snapshotted source hashes.
- [ ] Copy the immutable archive model source and prototype assets into 0036 without changing their semantics.
- [ ] Add a strict checkpoint loader that checks model-only payload keys, source metadata, tensor count, parameter count and SHA-256.
- [ ] Run `python3 -m unittest -v train.0036_dedicated_action_value_network.tests.test_project_contract` and require PASS.

### Task 2: Build the recent dual-perspective Episode catalog

**Files:**
- Create: `train/0036_dedicated_action_value_network/data/archetypes.py`
- Create: `train/0036_dedicated_action_value_network/data/catalog.py`
- Create: `train/0036_dedicated_action_value_network/assets/archetypes_v1.json`
- Test: `train/0036_dedicated_action_value_network/tests/test_catalog.py`

**Interfaces:**
- Consumes: immutable official daily ZIPs for 2026-08-03 through 2026-08-06.
- Produces: exactly 10,000 Episode rows, exactly 20,000 player trajectories, deterministic 9,000/1,000 whole-Episode train/validation split, deck hashes, seats, outcomes, final Prize counts, 15-class opponent labels and signed final-diff labels.

- [ ] Write tests for deterministic recent-date quotas, whole-Episode split isolation, balanced win/loss perspectives and final-diff sign reversal.
- [ ] Define 14 main-axis classes plus Other from the frozen 55-deck 0806 environment; store names, required card IDs and priority explicitly.
- [ ] Stream ZIP members, reject nonterminal/draw/malformed Episodes, validate both exact 60-card registrations and terminal fields, and hash payloads.
- [ ] Select 2,500 Episodes from each of 0803/0804/0805/0806 using deterministic hash rank, with fail-closed quota checks. The original newest-heavy quota is retained in decisions as a failed audit because 0806 had only 2,886 fully qualified Episodes.
- [ ] Persist catalog and audit under `experiments/0036_dedicated_action_value_network/data_audit/` and run its verifier.

### Task 3: Materialize action-decision features and labels

**Files:**
- Create: `train/0036_dedicated_action_value_network/data/materialize.py`
- Create: `train/0036_dedicated_action_value_network/data/replay_contract.py`
- Test: `train/0036_dedicated_action_value_network/tests/test_materialize.py`

**Interfaces:**
- Consumes: Task 2 catalog, chronological replay frames, frozen canonical compiler and prototypes.
- Produces: immutable gzip JSONL shards containing the 39 actor tensors plus `value_target`, `archetype_target`, `final_diff_target`, `episode_weight` and audit-only identity.

- [ ] Test that registration frames, automatic engine steps and decoder sub-tokens do not become examples.
- [ ] Test that every callback uses the pre-action observation and exact subsequent full ordered action only for audit.
- [ ] Compile both perspectives with independent `CausalKnowledge`; never share state across players or Episodes.
- [ ] Assign each decision weight `1 / actor_decisions_in_episode`; verify each player trajectory sums to one.
- [ ] Write shards atomically, hash every shard, record maxima/counts/class distributions and run a full decompression/schema/hash verification pass.

### Task 4: Implement Value models, losses and metrics

**Files:**
- Create: `train/0036_dedicated_action_value_network/model/value_network.py`
- Create: `train/0036_dedicated_action_value_network/training/objective.py`
- Create: `train/0036_dedicated_action_value_network/training/metrics.py`
- Test: `train/0036_dedicated_action_value_network/tests/test_value_network.py`
- Test: `train/0036_dedicated_action_value_network/tests/test_objective.py`

**Interfaces:**
- Consumes: frozen `EncodedState.tokens/mask/summary`, encoded legal-option tokens/mask and integer labels.
- Produces: `value_logit [B]`, `archetype_logits [B,15]`, `final_diff_logits [B,13]`; hidden-hand head is absent from V1 forward unless explicitly enabled later.

- [ ] Test two baselines: pooled raw memory MLP and pretrained summary MLP.
- [ ] Test the eight-query, two-layer pre-LN latent decoder with masked state/option cross-attention and latent self-attention.
- [ ] Assert encoder parameters receive no gradients and Value parameters do.
- [ ] Implement weighted BCE-with-logits plus optional multiclass archetype CE and 13-class final-diff CE.
- [ ] Implement weighted log loss, Brier score, ECE, accuracy, AUROC, diff MAE/exact accuracy and archetype macro accuracy without using audit fields in forward.

### Task 5: Run five immutable ablations

**Files:**
- Create: `train/0036_dedicated_action_value_network/run_ablation.py`
- Create: `train/0036_dedicated_action_value_network/training/{dataset,trainer,checkpoints,logging}.py`
- Create: `rl_runs/0036_dedicated_action_value_network/versions/V1_*/artifact/training_config.json` through separate strictly increasing version directories.
- Test: `train/0036_dedicated_action_value_network/tests/test_training.py`

**Interfaces:**
- Consumes: immutable Task 3 dataset and source checkpoint.
- Produces: one model-only checkpoint per epoch and validation predictions/metrics for each ablation.

- [ ] Implement configurations for `raw_pool_mlp`, `summary_mlp`, `latent_value_only`, `latent_value_archetype`, and `latent_value_archetype_diff`.
- [ ] Ensure every epoch has one train optimization pass and one full fixed-model validation pass.
- [ ] Write canonical metrics before TensorBoard and W&B; persist mirror status without rolling back local facts.
- [ ] Select by episode-weighted validation Value BCE, using Brier/ECE and per-archetype strata as guardrails rather than auxiliary accuracy.
- [ ] Retain every model-only epoch checkpoint atomically and reject optimizer/RNG/replay fields.

### Task 6: Admit the selected critic for RL initialization

**Files:**
- Create: `train/0036_dedicated_action_value_network/export_value.py`
- Create: `experiments/0036_dedicated_action_value_network/DECISIONS.md`
- Modify: `experiments/0036_dedicated_action_value_network/DESIGN.md`
- Modify: `experiments/0036_dedicated_action_value_network/DESIGN.html`
- Test: `train/0036_dedicated_action_value_network/tests/test_export.py`

**Interfaces:**
- Consumes: selected ablation checkpoint and immutable encoder identity.
- Produces: a critic-only model package with architecture/config/schema/checkpoint hashes and an explicit PPO initialization API.

- [ ] Verify held-out Value BCE improves on the constant-rate and existing summary-MLP baselines.
- [ ] Verify logit parity before/after export and fail on encoder identity mismatch.
- [ ] Document that offline prediction estimates behavior-policy value and must continue on-policy calibration after actor updates.
- [ ] Expose Value as PPO baseline/GAE input; do not turn `V(next)-V(current)` into an unreviewed environment reward.
- [ ] Run a small official-engine rollout calibration smoke before any policy-strength claim.
