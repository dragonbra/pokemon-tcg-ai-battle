# 0038 Integrated Update-0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a V5-weight-free 0038 update-0 from Policy-0806 + pretrained Value, warm-start the Phantom allocation head from Zero-Shot sequential behavior, and prepare independently switchable Prize, Meta, Tempo and enlarged evaluation infrastructure without starting formal PPO.

**Architecture:** Keep the exact 0037 actor/value/adaptation interfaces while replacing the action and trajectory contract with 0038 compound boundaries. Auxiliary heads consume only the existing deployable visible representation, publish separate losses/optimizer groups, and are inert when disabled. Data collection and official parity are separate fail-closed gates before an update-0 checkpoint can be released.

**Tech Stack:** Python 3.11, PyTorch, official seeded engine runtime, unittest, JSON/JSONL model-only manifests.

## Global Constraints

- Never modify `engine/source/`, official ABI, observation schema, or `select -> list[int]` protocol.
- Never load V5 or any RL-updated checkpoint into the new model.
- The only core sources are Policy-0806 SHA `0ca395...df8` and pretrained Value SHA `e88b2...360`.
- LoRA is last Option block Q/V only, rank/alpha `4/8`, zero-delta at initialization.
- Allocation data teacher is the same Zero-Shot policy under the legacy sequential contract.
- Old trajectory logprobs are never valid for new PPO ratios.
- No formal 50/200-update run; all formal launch paths remain fail closed.

---

### Task 1: Initialization contract and strict loader

**Files:**
- Create: `train/0038_action_boundary_rl/initialization.py`
- Create: `experiments/0038_action_boundary_rl/INITIALIZATION_MANIFEST.json`
- Modify: `train/0038_action_boundary_rl/policy/actor_critic.py`
- Test: `train/0038_action_boundary_rl/tests/test_initialization_contract.py`

**Interfaces:**
- Produces `InitializationConfig`, `InitializationManifest`, `build_update0_model(...)`, and `assert_no_rl_checkpoint_source(...)`.
- The loader accepts only the two exact source hashes and explicitly allowlists new head prefixes.

- [ ] Add tests that reject V5 paths/hashes and unexpected missing core tensors.
- [ ] Build the core from Policy-0806 and pretrained Value with fresh zero-delta LoRA.
- [ ] Compare disabled-head logits, Value, legal mask and greedy action against a no-LoRA Zero-Shot reference.
- [ ] Write the immutable initialization manifest including optimizer/scheduler/RNG reset semantics.

### Task 2: Modular flags, presets and loss registry

**Files:**
- Create: `train/0038_action_boundary_rl/integrated/config.py`
- Create: `train/0038_action_boundary_rl/integrated/loss_registry.py`
- Create: `train/0038_action_boundary_rl/integrated/presets.py`
- Test: `train/0038_action_boundary_rl/tests/test_integrated_config.py`

**Interfaces:**
- Produces `IntegratedFlags`, `LossRegistry`, `LossTerm`, and immutable BASE/PRIZE/META/PRIZE_META/INTEGRATED presets.

- [ ] Validate all requested feature flags and reject incompatible legacy/action-boundary mixing.
- [ ] Register policy/value/prize/meta/entropy/tempo terms with independent weights and optimizer groups.
- [ ] Prove disabled modules do not consume RNG or change core outputs.

### Task 3: Prize auxiliary

**Files:**
- Create: `train/0038_action_boundary_rl/integrated/prize.py`
- Modify: `train/0038_action_boundary_rl/policy/value_network.py`
- Test: `train/0038_action_boundary_rl/tests/test_prize_aux.py`

**Interfaces:**
- Produces `PrizeAuxHead`, `PrizeAuxConfig`, `prize_rewards`, `prize_gae`, `combine_actor_advantages`, and `gradient_alignment`.

- [ ] Preserve Query 0 and `V_win`; source `V_prize` from an unused query/sidecar.
- [ ] Implement off/directional/terminal_neutral reward modes with scale `1/24`.
- [ ] Compute and normalize win/prize advantages separately on an own-turn clock.
- [ ] Add fixed-minibatch gradient norm/cosine diagnostics without per-minibatch double backward.

### Task 4: Opponent meta classification

**Files:**
- Create: `train/0038_action_boundary_rl/integrated/opponent_meta.py`
- Create: `experiments/0038_action_boundary_rl/opponent_meta_taxonomy_v1.json`
- Test: `train/0038_action_boundary_rl/tests/test_opponent_meta.py`

**Interfaces:**
- Produces `OpponentMetaHead`, `OpponentMetaTaxonomy`, detached `z_opp`, accuracy/F1/ECE/Brier metrics, and public-view fingerprints.

- [ ] Map Frozen deck IDs to versioned archetypes with unknown/ambiguous labels.
- [ ] Condition the action query only on predicted detached embeddings.
- [ ] Prove true deck labels and hidden hand/deck/Prize fields never enter inference tensors.
- [ ] Add counterfactual hidden-information invariance tests.

### Task 5: Tempo metrics and curriculum

**Files:**
- Create: `train/0038_action_boundary_rl/integrated/tempo.py`
- Test: `train/0038_action_boundary_rl/tests/test_tempo.py`

**Interfaces:**
- Produces visible-only `AttackOpportunity`, `TempoEpisodeMetrics`, and provenance-preserving curriculum weights.

- [ ] Detect second-own-turn and legal Attack root options using only current observation/history.
- [ ] Track attack conversion, streaks, interruption reasons, damage and Prize progress.
- [ ] Keep tempo loss disabled and prevent future deck order access.

### Task 6: Zero-Shot sequential allocation dataset

**Files:**
- Create: `train/0038_action_boundary_rl/data/allocation_bc.py`
- Modify: `train/0038_action_boundary_rl/rollout/collector.py`
- Modify: `train/0038_action_boundary_rl/rollout/protocol.py`
- Test: `train/0038_action_boundary_rl/tests/test_allocation_bc_data.py`

**Interfaces:**
- Produces compact `AllocationBCSample`, battle-level split manifests, dataset hashes and a shadow collector that never uses allocation-head actions.

- [ ] Record root features and stable targets when Zero-Shot chooses Phantom Dive.
- [ ] Reconstruct the six subsequent sequential target choices into one canonical label.
- [ ] Split by battle and report target count, allocation, seat, archetype, KO/Prize and phase distributions.
- [ ] Persist only approved visible features/labels; never persist PPO old logprobs as reusable training data.

### Task 7: Allocation BC trainer and update-0 checkpoint

**Files:**
- Create: `train/0038_action_boundary_rl/training/allocation_bc.py`
- Modify: `train/0038_action_boundary_rl/checkpoint.py`
- Test: `train/0038_action_boundary_rl/tests/test_allocation_bc_training.py`

**Interfaces:**
- Produces early-stopped allocation-only BC, validation metrics, and `0038_update0_model_only_v1` checkpoint metadata.

- [ ] Freeze all core/LoRA/value tensors and optimize only allocation head.
- [ ] Report NLL, top-1, count MAE, immediate Prize agreement, entropy and unseen coverage.
- [ ] Save a common update-0 with fresh optimizer/scheduler/RNG/rollout-state declarations and no recovery state.

### Task 8: Official sequential/macro parity gate

**Files:**
- Create: `train/0038_action_boundary_rl/parity_action_boundary.py`
- Test: `train/0038_action_boundary_rl/tests/test_action_boundary_official_parity.py`

**Interfaces:**
- Produces per-allocation authority-state, successor-mask, reward/terminal/winner/RNG/observation comparisons and invalid/fallback counters.

- [ ] Enumerate `1/7/28/84/210` allocations for `n=1..5`.
- [ ] Compare legacy and macro primitive expansions under identical official state/RNG.
- [ ] Reject any hidden-information, stable-identity or event-span mismatch.
- [ ] Require zero failures before update-0 is marked releasable.

### Task 9: Enlarged Frozen evaluation design and smoke gates

**Files:**
- Create: `train/0038_action_boundary_rl/evaluation_design.py`
- Create: `experiments/0038_action_boundary_rl/EVALUATION_DESIGN.md`
- Modify: `experiments/0038_action_boundary_rl/DESIGN.md`
- Modify: `experiments/0038_action_boundary_rl/DESIGN.html`
- Test: `train/0038_action_boundary_rl/tests/test_evaluation_design.py`

**Interfaces:**
- Produces one versioned 2,048-game Frozen panel as eight disjoint 256-game shards, with persisted paired per-game results and power/MDE caveats. There is no separate 4,096-game Confirmation panel.
- Adds three metric-frequency tiers: cheap update aggregates, offline Frozen-panel aggregates, and sparse fixed-minibatch diagnostics at updates 0, 5, 10, then every 10 updates.
- Configures rollout/backend capacity without a fixed 512-game assumption and supports explicit `fixed_epochs` versus preferred `fixed_optimizer_budget` scaling.

- [ ] Generate disjoint paired seed panels stratified by seat/opponent archetype.
- [ ] Implement Wilson CI, paired flips, McNemar and paired bootstrap summaries.
- [ ] Estimate sample sizes for +1pp/+2pp using observed discordance assumptions.
- [ ] Run only tests, disabled parity, leakage checks, BC validation, official smoke, fixed-minibatch gradients and at most five stability updates.
- [ ] Keep formal INTEGRATED launch blocked pending human confirmation.
