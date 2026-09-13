# 0045 U125 FFN LoRA and LayerNorm Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand exact V17 U125 with zero-effective Option/State FFN LoRA and a separately optimized final Option LayerNorm, prove behavior parity, then continue PPO in a new version.

**Architecture:** Reconstruct U125 with its original r16 QVO/shared-State adaptation before installing any new module. Add policy-only Option final-block FFN deltas, shared State final-board-block FFN parametrizations, and the actual final Option norm weight/bias; create expanded parent/reference checkpoints with complete-delta identity and start a fresh optimizer/on-policy run.

**Tech Stack:** PyTorch parametrizations/functional_call, project-local PPO, CUDA official-engine evaluation, pytest, W&B.

## Global Constraints

- Exact parent is V17 U125 SHA-256 `cad14c2dcb9c3dffed8fa739969cfb0952f664607854dda16fdc03f1d127dc2b`.
- Existing Option attention LoRA and State attention LoRA both use LR `2e-5`; every other existing group keeps its V17 LR.
- New Option FFN LoRA is r16, zero-effective, LR `3e-5`, weight decay 0.
- New State FFN LoRA is r16, zero-effective, LR `1.5e-5`, weight decay 0.
- Final Option LayerNorm weight+bias is a separate group at LR `5e-6`, weight decay 0.
- Checkpoints remain complete model-only; optimizer/scheduler/RNG/rollout state is forbidden.
- Opponent stays independently materialized immutable Policy-0814.
- No formal run starts until logits, greedy action, trainable inventory, optimizer partition, and checkpoint reconstruction gates pass.

---

### Task 1: Expanded Adaptation Modules

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/policy/ffn_lora_expansion.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/policy/adaptation.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/policy/option_policy_lora.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_v19_u125_ffn_lora_expansion.py`

**Interfaces:**
- Produces `install_state_ffn_lora`, `state_ffn_lora_parameters`, Option FFN delta tensors, and explicit expansion flags in `AdaptationConfig`.
- Option FFN overrides `linear1.weight` and `linear2.weight` only in the policy functional call.

- [ ] Write tests asserting exact target names, r16 shapes, zero B matrices, and exact before/after logits.
- [ ] Run the focused test and confirm it fails before implementation.
- [ ] Implement the zero-delta modules and strict inventory assertions.
- [ ] Run the focused test and confirm the module/parity cases pass.

### Task 2: Trainable and Optimizer Contracts

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/policy/actor_critic.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/training/ppo_full_semantic.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/training/lr_profiles.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/training/trainable_audit.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_v19_u125_ffn_lora_expansion.py`

**Interfaces:**
- Adds optimizer groups `policy_option_ffn_lora`, `shared_state_ffn_lora`, and `option_final_norm`.
- Extends `LearningRateProfile`/`PPOConfig` with the three exact rates while leaving defaults disabled for old models.

- [ ] Write failing optimizer partition tests for names, tensor counts, disjoint ownership, rates, and weight decay 0.
- [ ] Integrate trainable ownership and new groups without changing old-model manifests.
- [ ] Run old V13/V16 optimizer tests plus the new focused test.

### Task 3: Exact U125 Migration and Parity Gate

**Files:**
- Create: `train/0045_single_deck_expert_minimal_lora/training/run_v19_u125_ffn_lora_expansion.py`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_v19_u125_ffn_lora_expansion.py`

**Interfaces:**
- Reconstructs legacy U125 first, installs the expansion, emits expanded parent/reference complete-delta checkpoints, and records a parity audit.
- Launches from logical update 125 with a fresh optimizer and baseline U125 eval.

- [ ] Write failing readiness/migration tests binding exact parent/reference hashes.
- [ ] Implement deterministic expansion seeds and tensor/logits/greedy parity audit.
- [ ] Validate both expanded checkpoints independently reconstruct to their recorded full-state hashes.
- [ ] Verify launch kwargs preserve Policy-0814 rollout/eval identity and exact LR profile.

### Task 4: Deployment and Documentation

**Files:**
- Modify: `train/0045_single_deck_expert_minimal_lora/semantic_runtime/deployment/compound_inference.py`
- Modify: `train/0045_single_deck_expert_minimal_lora/evaluation/candidate.py`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.md`
- Modify: `experiments/0045_single_deck_expert_minimal_lora/DESIGN.html`
- Test: `train/0045_single_deck_expert_minimal_lora/tests/test_v19_u125_ffn_lora_expansion.py`

**Interfaces:**
- Portable FP16 artifacts reconstruct the expanded policy and preserve deployment-effective hashing.
- Design docs record tensor targets, optimizer groups, parity boundary, parent identity, and stage.

- [ ] Add portable loader support with strict expanded metadata and state inventory.
- [ ] Test FP16 storage to FP32 runtime parity for the expanded candidate.
- [ ] Update both authoritative design documents from code/audit facts.

### Task 5: Formal Verification and Launch

**Files:**
- Runtime outputs: `rl_runs/0045_single_deck_expert_minimal_lora/versions/V19_policy0814_u125_ffn_lora_norm_expansion/`

**Interfaces:**
- Consumes all prior gates and starts the formal online W&B run only after PASS.

- [ ] Run focused and relevant regression suites.
- [ ] Run readiness/migration and inspect the machine-readable parity/trainable audits.
- [ ] Launch with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and confirm baseline eval begins before PPO.
