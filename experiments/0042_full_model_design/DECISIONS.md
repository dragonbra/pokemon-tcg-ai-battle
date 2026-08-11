# 0042 Decisions

## 2026-08-11: canonical model boundary

- Freeze PrototypeEncoder, StateEncoder, and OptionEncoder.
- Remove 0040 Option Q/V LoRA and local Option LayerNorm tuning from the 0042 runtime.
- Keep the pretrained q1 15-class Meta head frozen while retaining its forward autograd graph.
- Add a q0 Value Adapter and a decoder readout-only Policy Strategy Adapter, both scalar
  zero-gated with normally initialized residual MLPs.
- Detach q1, Meta logits, and adapted V at the Value-to-Policy boundary.
- Use separate `E_V_own` and `E_pi_own` embedding tables.
- Treat registered own archetype and opponent Meta target as different semantic types even while
  both use the initial 15-class taxonomy.
- Replace the legacy fixed optimizer budget with PPO Protocol V2: 256 games, complete shuffled
  traversal without replacement, up to three KL-guarded data epochs, and 2,048-decision minibatches.
- Use the exact 55 numbered decks from `0806_kaggle_top100_plus_v1`; do not retain the copied 0040
  named deck directory layout.
- Resolve every training and evaluation opponent as an independent, complete immutable
  `Policy-0809`; focal updates must never change opponent paths, tensors, or effective identity.
- Use FP32 training. Every formal candidate evaluation must export FP16 storage and rematerialize
  FP32 runtime before an identity-bound Frozen-0809 CUDA-2048 evaluation.
- Use exact focal deck `007_dragapult_ex`; after smoke, formal training is unbounded and performs
  Frozen-0809 CUDA-2048 at update 0 and every 10 updates.
- Treat bulk feature D2H or any per-lane exact-deck routing mismatch as a fatal pre-PPO error.
- Keep rollout features in one contiguous CUDA-resident store and gather PPO minibatches on
  device. Batch Phantom allocation-head evaluation by allocation/target shape instead of issuing
  one small GPU launch per macro.
- Use one deterministic rollout-wide behavior-KL guard sample per update: 4,096 shuffled
  decisions plus every compound/macro decision. Reuse the same sample after each data epoch.
  Run the all-decision pre-update old-logprob audit at update 1 and every tenth update; all other
  updates use the guard set. This changes diagnostic cost only, not PPO data traversal, frozen
  targets, or optimizer coverage.
- Compact heterogeneous ready lanes by current actor role before complete-policy forward. Focal
  and opponent models evaluate only their assigned rows, then scatter results back to lane order.
  This optimization must preserve independent Full0809 opponent materialization, exact-deck static
  fields, and per-decision output parity; it is not permission for cross-policy weight sharing.
- Define the CUDA residency contract precisely: semantic rollout features remain on device and
  bulk feature D2H must be zero. Compact action/control/trajectory scalars may cross to the host for
  official-engine stepping and Episode/GAE construction; they must never trigger CPU feature
  recompilation followed by feature H2D.
- Keep the Protocol V2 logical minibatch at 2,048 decisions, but execute it as 1,024-row physical
  graphs with a shared logical loss denominator and one accumulated optimizer step. Use 512-row
  behavior-probe chunks and expandable CUDA allocator segments. This is a memory-execution detail,
  not a change to data epochs, sample coverage, optimizer-step count, or PPO target semantics.

## Evidence

- `train/0042_full_model_design/tests/test_strategy_architecture.py`
- `.tmp/strategy_adapter_v2_audit/0042_runtime_audit.json`
- `.tmp/strategy_adapter_v2_audit/0042_value_meta_baseline.json`
- `.tmp/evaluation/0042_policy0809_contract/mixed_deck_role_compacted_parity.json`
- `.tmp/0042_protocol_v2_preflight/role_compacted_retry_20260811/report.json`
- `docs/rl/0042_full_model_design.md`
