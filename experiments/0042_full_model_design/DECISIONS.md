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
- Preserve candidate deployment's stale-module hard gate. The first formal update-0 attempt exposed
  that the preceding Policy0809 parity loader left its temporary `strategy.*` namespace cached.
  Fix the parity loader to reject preexisting namespaces and remove only its temporary package
  modules/path immediately after importing the required runtime types. The failed V1 status remains
  recorded; resume uses the dedicated update-0 path and reruns the deployment/Frozen baseline.
- Define `deployment-effective` as the hash of deployed FP16 tensors plus runtime-semantic schema
  fields only. Source checkpoint SHA, portable file SHA, repository version, checkpoint update,
  and training provenance remain separately audited fields; changing those fields cannot rename an
  otherwise identical effective policy. The third update-0 attempt exposed the old conflation when
  the same deterministic U0 tensors acquired a new source checkpoint hash. U0 now hard-pins the
  corrected effective identity and its derived 007/Full0809 CUDA-2048 schedule; later checkpoints
  derive their own identity-bound schedule instead of being compared to the U0 schedule.
- Make the immutable-trace CUDA/package parity scaffold execute the real 0042 path
  (`q0/q1 -> frozen MetaHead -> adapted V -> Strategy Context -> Policy Adapter`) when the package
  declares Strategy Adapter modules. The earlier scaffold only knew the historical 0038
  `meta_head + meta_conditioner` interface and failed before comparing logits. The fixed 283-record
  replay reports zero greedy divergence, package root tolerance failures, and Value sign divergence;
  subprocess failures now preserve diagnostic stdout/stderr in the formal log.
- Seal V1 at model-only U250 and branch with a fresh optimizer from that immutable checkpoint.
  Preserve Policy-0809/U0—not U250—as the reference-KL snapshot by constructing the trainer before
  strict-loading focal U250 weights. Never load V1 optimizer, RNG, replay, or rollout state.
- Select explicit actor-group LRs from one identical 256-game, 22,719-decision diagnostic:
  ActionDecoder `2e-5`, Policy Strategy Adapter `4e-5`, allocation head `2e-5`; keep Value groups
  `1e-4`. The selected arm completed 3x coverage with behavior KL `4.30e-4` and clip fraction
  `0.253%`; its artifact is diagnostic-only.
- Split behavior telemetry into root and compound/macro KL, and record each optimizer group's
  pre/post-clip gradient norm and effective LR. A macro KL larger than aggregate KL must remain
  visible even when macro rows have lower total weight.
- Exclude pre-seat context-41 rows (`relative_first_player=0`) from sparse gradient diagnostics;
  this does not filter PPO training data because stochastic training rollout does not delegate
  context 41. Keep `0` legal only in the explicit pre-seat runtime path.
- 2026-08-12: remove numeric-closeness gates from training launch entirely, by explicit user
  decision. CPU/CUDA Value and FP32/FP16 comparisons remain recorded diagnostics, but their
  tolerance/pass fields cannot block PPO. Exact tensor/mask semantics, greedy-action identity,
  checkpoint/deployment identity, exact-deck routing, and independent Policy-0809 identity remain
  hard gates. V15 is retained as a failed U0 historical version; recovery starts V16/010 from the
  immutable V14 U50 model-only checkpoint with fresh optimizer and on-policy data, then shifts
  011 and the five remaining focal-deck versions through V17–V22.

## Evidence

- `train/0042_full_model_design/tests/test_strategy_architecture.py`
- `.tmp/strategy_adapter_v2_audit/0042_runtime_audit.json`
- `.tmp/strategy_adapter_v2_audit/0042_value_meta_baseline.json`
- `.tmp/evaluation/0042_policy0809_contract/mixed_deck_role_compacted_parity.json`
- `.tmp/0042_protocol_v2_preflight/role_compacted_retry_20260811/report.json`
- `docs/rl/0042_full_model_design.md`
