# RL Promote Champion & Frozen Policy Protocol V1

**Status:** Mandatory / Non-negotiable

**Scope:** RL training, frozen evaluation, opponent-pool sampling, CUDA/CPU evaluation, checkpoint promotion, policy routing and batching

**Version:** 1.1
**Date:** 2026-08-11

---

## 0. Purpose

This document defines the policy-identity and Promote Champion rules for the RL stage.

The highest-priority invariant is:

> **First guarantee policy semantic correctness; only then optimize execution performance.**

Routing, batching, encoder sharing, caching, resident CUDA inference, deduplication, and any other performance optimization are subordinate to policy identity. An optimization is valid only when it is proven to preserve the exact effective policy that was requested.

A focal/main-view model change must never silently change any opponent already registered in the opponent pool.

## 0.1 Canonical Entry Point and Project Conformance

This file is the repository-wide canonical entry point for RL policy identity,
Frozen evaluation, Promote Champion, and Kaggle-facing candidate deployment. It is
not an implementation note for one numbered experiment. Before creating or changing
any RL project, an Agent must read this document and adapt that project's training,
evaluation, promotion, and package-export boundaries to this protocol.

Every current and future RL project that produces a Kaggle-capable policy must provide
a project-local conformance adapter with all of the following behavior:

1. resolve every focal and opponent `policy_id` to an auditable immutable effective
   identity before routing or batching;
2. materialize the project's complete effective candidate under
   `kaggle_fp16_storage_fp32_runtime_v1`, including its base, trained heads, LoRA, and
   all other project-specific inference weights in the declared merge order;
3. use the same candidate materialization and effective-content hashing semantics for
   periodic RL evaluation, formal Frozen CPU/CUDA evaluation, Promote evidence, and
   final Kaggle package export/reload;
4. place a hard preflight identity gate at the narrowest shared evaluation boundary,
   before official games begin;
5. persist the required candidate/opponent identity audits and reproducibility fields
   in the formal evaluation manifest;
6. prove with regression tests that the qualifying evaluation artifact and exported
   package reconstruct the same deployment-effective policy;
7. keep PPO optimization and behavior-policy rollout FP32 unless a later versioned
   protocol explicitly changes that rule.

The implementation under `train/0040_dragapult_0809_action_boundary_rl/` is the
current reference implementation, not shared runtime infrastructure and not the
definition of this protocol. Repository project-isolation rules still apply: a future
numbered project must not import executable code from 0040. It must implement or
promote an appropriate self-contained adapter while preserving the identities,
ordering, audits, and failure semantics defined here.

Architecture-specific field names and module composition may differ. Semantic
conformance may not: an Agent must not claim compatibility merely because it copied
an 0040 class name, manifest label, or checkpoint shape.

---

# 1. Core Definitions

## 1.1 Policy Identity

A `Policy` is an immutable inference entity identified by a unique `policy_id`.

A policy is not defined only by its decoder, checkpoint filename, architecture shape, or nominal model family. It is defined by the **complete effective inference weights and the model/schema configuration required to reproduce its inference function**.

Examples:

- `Policy-0806`
- `Policy-0809`
- `Policy-update00250`
- future promoted snapshots

Given a `policy_id`, the system must be able to deterministically reconstruct the exact effective policy that the ID represents.

---

## 1.2 Focal Policy

The `focal` policy is the model currently being trained or evaluated from the main-player perspective.

Examples:

- current focal base may be `Policy-0809`
- a future experiment may switch focal base to `Policy-0812`
- an RL candidate may be `update00250`

Changing the focal policy **must not change the identity, weights, representation network, decoder, LoRA, or behavior of any frozen opponent policy**.

---

## 1.3 Frozen Policy

A frozen policy is a policy whose effective inference weights are immutable during the run.

`Frozen` means the entire effective inference function is frozen, not merely the last decoder layer.

A policy must not be called frozen if any part of its effective inference path is substituted by modules from another policy.

---

## 1.4 Snapshot Policy

When an RL checkpoint is manually promoted, it becomes a new immutable policy snapshot.

Example:

```text
update00250
    ↓ manual promote
Policy-update00250
```

The snapshot must reproduce exactly the effective model that existed at promotion time.

If the snapshot consists of a frozen base plus trained deltas such as decoder weights and LoRA, it is acceptable to store the policy compositionally rather than physically duplicate all base tensors, but the composition must be immutable and exactly reconstructable.

---

# 2. Non-Negotiable Policy Identity Invariants

## 2.1 Focal/Opponent Independence

The focal policy and opponent policies are independent identities.

If the focal model changes:

```text
0809 → 0812
```

the opponent pool must remain unchanged unless the experiment configuration explicitly changes an opponent policy ID.

Forbidden:

```text
focal = 0812
opponent requested = Policy-0806

actual execution =
0812 encoder/trunk + 0806 decoder
```

This is a policy identity violation.

---

## 2.2 Full Policy Requirement

When a run requests `Policy-0806`, the opponent must execute the full effective `Policy-0806`.

For the current architecture, this means all policy-specific inference modules must resolve to the 0806 policy, including where applicable:

```text
0806 prototype encoder
→ 0806 state encoder
→ 0806 option input encoder
→ 0806 option transformer layer 0
→ 0806 option transformer layer 1
→ 0806 final norm
→ 0806 action decoder
```

Likewise, `Policy-0809` must execute the full effective 0809 policy.

---

## 2.3 Hybrid Policies Are Forbidden Unless Explicitly Registered

An accidental composition such as:

```text
0809 prototype/state/option trunk
+
0806 final option block/decoder
```

must never be treated as `Policy-0806`.

If a hybrid policy is ever intentionally desired for an experiment, it must receive a separate explicit policy ID and manifest, for example:

```text
Experimental-Hybrid-0809Trunk-0806Head
```

It must never silently inherit the identity of either source policy.

---

## 2.4 Architecture Equality Does Not Imply Shareability

The following are **not** sufficient reasons to share modules across policies:

- same tensor shape
- same architecture
- same model family
- same pretraining pipeline
- same card/deck
- same encoder interface
- same hidden size
- historical assumption that the encoder was once shared

Cross-policy module sharing is allowed only when the effective module weights are proven identical, preferably by immutable content hash.

---

# 3. Current Frozen Evaluation Policies

## 3.1 Frozen Policy-0806

`Frozen-0806` means:

> All opponents use the complete effective pretrained `Policy-0806` weights.

The focal model may be 0809, 0812, an RL checkpoint, or any other policy. This must have no effect on the opponent.

---

## 3.2 Frozen Policy-0809

`Frozen-0809` means:

> All opponents use the complete effective pretrained `Policy-0809` weights.

It is a distinct benchmark from Frozen-0806.

---

## 3.3 Benchmark Separation

The following two evaluations are separate benchmark tracks:

```text
CUDA-2048 vs Frozen-0806
CUDA-2048 vs Frozen-0809
```

Their results must be reported separately.

They must not be merged into one score unless a future protocol explicitly defines such an aggregate.

---

# 4. RL Opponent-Pool Semantics

## 4.1 Policy Sampling

When RL uses an opponent pool, each sampled opponent must first resolve to a concrete `policy_id`.

Example:

```text
lane 0  → Policy-0806
lane 1  → Policy-0809
lane 2  → Policy-update00250
...
```

Once a policy is selected for an episode, that opponent must use the complete effective weights of that policy for the full episode unless a future protocol explicitly defines otherwise.

---

## 4.2 Distribution Preservation

If a 256-lane minibatch is required to preserve a specific deck/opponent-policy frequency distribution, routing optimizations must preserve that distribution exactly.

Performance optimization must not alter:

- sampled policy identity
- matchup identity
- deck distribution
- first/second-player assignment
- sampling probabilities
- policy action function

---

## 4.3 Batched Inference Is Allowed

If a batch contains multiple known policies, grouping by policy for inference is allowed.

Example:

```text
256 lanes
├─ 120 × Policy-0806
├─  80 × Policy-0809
└─  56 × Policy-update00250
```

Valid optimization:

```text
group lanes by policy_id
→ run each policy with its own exact effective weights
→ scatter actions back to original lanes
```

This is encouraged if it improves throughput while preserving exact semantics.

---

# 5. Routing and Encoder-Sharing Rules

## 5.1 Required Order of Operations

All routing systems must follow this order:

```text
1. Resolve requested policy_id
2. Resolve immutable policy manifest
3. Materialize/verify effective weights
4. Bind lanes to policy identity
5. Only then perform batching/sharing/caching optimization
```

Never start from "which encoder can we reuse?" and infer policy identity afterward.

---

## 5.2 Legal Sharing

Sharing computation between policies is allowed only if the shared computation is semantically identical.

Preferred proof:

```text
module effective-weight hash A == module effective-weight hash B
```

If two policies truly reference the same immutable base tensor set, the runtime may deduplicate those computations.

---

## 5.3 Illegal Sharing

The runtime must not substitute a module from the focal policy merely because:

- it is already resident on GPU
- it has the same architecture
- it is cheaper to batch
- the opponent only has a different decoder
- earlier experiments historically shared a base
- the current RL optimizer does not update that module

A fixed but wrong hybrid policy is still wrong.

---

# 6. Snapshot and Promote Champion Protocol V1

## 6.1 Evaluation Before Promote

A candidate may be evaluated independently on:

```text
CUDA-2048 vs Frozen-0806
CUDA-2048 vs Frozen-0809
```

Both benchmark results must remain separate.

Additional benchmark tracks may be added later without redefining the policy identity rules in this document.

## 6.1.1 Kaggle Candidate Deployment Identity Contract

Every evaluation whose purpose is to estimate the strength of an agent that may be
submitted to Kaggle must evaluate the candidate under contract:

```text
kaggle_fp16_storage_fp32_runtime_v1
```

The required candidate materialization order is:

```text
immutable base FP32
+ checkpoint decoder / trained heads
+ checkpoint LoRA merged into its declared base modules
→ complete effective candidate converted to FP16 storage
→ that exact FP16 artifact strict-loaded into FP32 runtime tensors
→ greedy official-engine evaluation
```

Converting the FP16 artifact back to FP32 changes its runtime dtype but does not
recover the discarded FP32 precision. Therefore the deployment-effective candidate
is not interchangeable with the source FP32 training checkpoint. Both identities
must be retained:

- source FP32 checkpoint path and SHA-256;
- portable FP16 checkpoint/artifact SHA-256;
- deployment-effective candidate SHA-256;
- checkpoint update and model/action/observation schema versions;
- `storage_dtype = fp16`;
- `runtime_dtype = fp32`;
- conversion-order/merge contract;
- explicit candidate deployment identity audit `PASS`.

The final promoted/submission package must recompute this deployment-effective
content hash and match the hash recorded by the qualifying Frozen evaluation. A
matching source FP32 checkpoint is necessary but not sufficient; if the final FP16
content differs, export/promotion must hard fail.

This contract is mandatory for:

1. every RL Frozen evaluation scheduled at the configured cadence, currently every
   five updates;
2. every formal Frozen CUDA-2048 candidate evaluation;
3. every formal Frozen CPU-256 candidate evaluation;
4. any other CPU or CUDA result presented as evidence of expected Kaggle submission
   strength.

CPU and CUDA implementations should load the same immutable portable FP16 candidate
artifact. If they independently materialize it from the same immutable source, they
must prove the same deployment-effective semantic content hash under the same
conversion contract and record each concrete artifact file SHA-256. Container-level
serialization bytes may differ without changing tensor/metadata content, so file SHA
alone is provenance and is not a substitute for the effective content hash. Both paths
then run FP32 inference. A raw FP32 checkpoint may
be used for diagnostics, PPO, ablation, or parity investigation, but its result must
not be reported as Kaggle-strength, Frozen Promote Champion, or package-strength
evidence.

This candidate rule does not weaken or replace opponent policy identity. The Frozen
opponent must still resolve independently to its registered immutable full policy.
Changing an opponent's storage/runtime representation in a way that changes its
effective weights requires a new registered effective identity; candidate conversion
must never silently quantize, rebuild, or otherwise mutate the opponent.

Missing, failed, mismatched, or incomplete candidate deployment audit is fatal. A
run must abort before games start or be rejected as invalid evidence; warning and
continue is forbidden. Historical raw-FP32 or Hybrid results remain preserved, but
must not be retroactively relabeled as compliant evidence under this contract.

### PPO Exception

PPO behavior-policy rollout, loss computation, optimizer master weights, and
model-only training checkpoints remain FP32. The Kaggle conversion is an immutable
evaluation/deployment materialization after a checkpoint is saved. It must not be
inserted into the PPO optimization path or used to silently change the behavior
policy that generated on-policy trajectories.

## 6.1.2 Frozen CPU256 / CUDA2048 Seed Contract

Promote Champion V1 uses Frozen contract
`frozen_0806_seeded_agent_first_player_v3` with master evaluation seed
`341512806`. The immutable exact-deck/opponent schedule contains 256 slots.

- Frozen CPU256 executes exactly replica `0` of those 256 slots for each candidate.
- Frozen CUDA2048 executes replicas `0..7` of the same 256 slots for each candidate.
- CUDA replica `0` must therefore use the same focal identity, opponent identity,
  slot, engine seed, Search seed, and toss-winner seed as CPU replica `0`.
- Replicas `1..7` are deterministic, mutually distinct additions; they must not
  replace, reroll, or modify replica `0`.
- Engine, Search, and toss-winner seeds are derived from the master seed, focal
  identity, opponent identity, slot, replica, and namespace. Changing any of these
  inputs defines different evidence and must not be silently compared as the same run.
- The seeded toss fixes only the winner of the toss. That Agent must process official
  context 41 and choose first or second; the harness must not assign, alternate, or
  balance seats.
- Failed or unfinished games must not be replaced with new seeds. A formal result is
  valid only when every scheduled game is terminal with zero error and zero unfinished.
- Reports must record the master seed, base schedule hash, evaluation schedule hash,
  every per-game engine/Search seed, replica and slot, toss winner, Agent choice, and
  actual seat.

Changing the master seed, seed derivation, replica set, schedule, or seat-selection
rule requires a new versioned Frozen contract ID. Frozen-0806 and Frozen-0809 retain
separate reports even when they use the same seed contract.

---

## 6.2 Manual Promote

Promotion is manual.

Even if a candidate improves on one or more benchmark pools, it does not automatically become champion.

The human operator decides whether to promote.

---

## 6.3 Promotion Creates an Immutable Snapshot

When a checkpoint such as:

```text
update00250
```

is promoted, it becomes a new registered policy:

```text
Policy-update00250
```

This policy must thereafter be immutable.

Future training may produce descendants, but must not mutate the promoted snapshot.

---

## 6.4 Snapshot Reconstruction

If `Policy-update00250` was produced from:

```text
base = Policy-0809
trained = decoder + selected LoRA
```

then its policy manifest must be sufficient to reconstruct exactly:

```text
Policy-0809 immutable base
+ update00250 decoder
+ update00250 LoRA
= exact Policy-update00250 effective inference model
```

Using a later focal encoder, later decoder, later LoRA, or another base checkpoint is forbidden.

---

## 6.5 Snapshot as Future Opponent

If a future run requests:

```text
opponent_policy_id = Policy-update00250
```

the runtime must load/reconstruct the exact promoted snapshot.

It must not substitute any part of the current focal model.

---

# 7. Required Policy Manifest

Every registered frozen/promoted policy should have an immutable manifest.

Recommended fields:

```yaml
policy_id: Policy-update00250
policy_kind: rl_snapshot

parent_policy_id: Policy-0809

model_schema_version: ...
observation_schema_version: ...
action_schema_version: ...

base_checkpoint:
  path_or_registry_id: ...
  sha256: ...

trained_checkpoint:
  path_or_registry_id: ...
  sha256: ...

components:
  prototype_encoder:
    source_policy: Policy-0809
    effective_sha256: ...
  state_encoder:
    source_policy: Policy-0809
    effective_sha256: ...
  option_input_encoder:
    source_policy: Policy-0809
    effective_sha256: ...
  option_transformer_layer_0:
    source_policy: Policy-0809
    effective_sha256: ...
  option_transformer_layer_1:
    source_policy: Policy-update00250
    effective_sha256: ...
  action_decoder:
    source_policy: Policy-update00250
    effective_sha256: ...

effective_policy_sha256: ...
created_from_update: 250
promoted_at: ...
```

Exact field names may differ in implementation. The invariant is that policy identity must be auditable and exactly reconstructable.

---

# 8. Runtime Hard-Fail Requirements

Policy identity violations must be fatal.

Do not silently warn and continue.

Before RL rollout or frozen evaluation starts, the runtime must be able to validate the requested opponent policy.

Example:

```text
requested opponent = Policy-0806

prototype encoder source = 0806
state encoder source     = 0806
option layer 0 source    = 0806
option layer 1 source    = 0806
decoder source           = 0806
```

Expected:

```text
PASS
```

If instead:

```text
prototype encoder source = focal-0812
decoder source           = 0806
```

the run must abort with a clear policy identity violation.

Suggested behavior:

```text
FATAL: Opponent policy identity violation.
Requested: Policy-0806
Observed effective composition: ...
```

---

# 9. Training/Evaluation Parity Requirements

Training rollout and frozen evaluation must use the same policy-resolution semantics.

Differences such as:

```text
training opponent = hybrid
evaluation opponent = full policy
```

or the reverse are forbidden unless an experiment explicitly defines them as distinct named policies.

Sampling mode may differ when intentionally configured, but the underlying effective policy weights must remain the requested policy.

---

# 10. CPU/CUDA Semantic Parity

CPU and CUDA engines may use different implementations, but they must represent the same game and the same requested policy.

For any new or materially modified CUDA routing path, perform a minimal lockstep parity gate before trusting aggregate win-rate results.

Recommended gate:

```text
same per-game seed
same focal policy
same full opponent policy
batch/lane = 1
greedy selection where practical
```

Compare:

```text
actor
state
observation
legal actions
chosen action
next state
```

When mismatch occurs, report the first divergence.

Before diagnosing engine semantics, verify policy identity first. A policy mismatch can create trajectory divergence even when engine state, observation, and legal actions are identical.

---

# 11. Evaluation Run Manifest

Every formal frozen evaluation should record enough information to identify exactly what was run.

Recommended fields:

```text
candidate policy ID
candidate checkpoint hash
candidate source FP32 checkpoint hash
candidate portable FP16 artifact hash
candidate deployment-effective identity/hash
candidate storage/runtime dtype (`fp16` / `fp32`)
candidate deployment identity audit (`PASS` required)
opponent policy ID
opponent effective-policy hash
engine implementation/version
CPU/CUDA path
lane count
selection mode
master seed
per-game seeds
matchup/opponent distribution
first/second-player assignment rule
repo commit
evaluation config hash
```

A benchmark result without a verified opponent policy identity must not be treated as valid Promote Champion evidence.

For formal CPU or CUDA evidence, the manifest must embed the runtime audit result,
including the requested policy ID, every materialized component hash, the effective
policy hash, and an explicit `PASS` status. Missing audit data and every status other
than `PASS` are fail-closed: the result is invalid for Promote Champion, even if all
games completed successfully.

---

# 12. Promote Champion V1 Decision Record

For each candidate, retain separate records such as:

```text
Candidate: update00250

Frozen-0806 CUDA-2048
  baseline: ...
  candidate: ...
  delta: ...
  policy identity audit: PASS

Frozen-0809 CUDA-2048
  baseline: ...
  candidate: ...
  delta: ...
  policy identity audit: PASS

Human decision:
  PROMOTE / REJECT / HOLD

Notes:
  ...
```

Promotion remains a human decision.

---

# 13. Mandatory Regression Tests

At minimum, maintain tests for:

1. `Policy-0806` resolves only to 0806 effective components.
2. `Policy-0809` resolves only to 0809 effective components.
3. A promoted snapshot resolves to its immutable original base + promoted deltas.
4. Changing focal policy does not change the effective hash of any existing opponent policy.
5. Grouped/batched routing returns the same policy outputs as independent per-policy inference for fixed inputs.
6. A deliberate focal/opponent hybrid causes a hard failure unless registered as its own explicit policy.
7. Training and evaluation resolve the same `policy_id` to the same effective-policy hash.
8. A Kaggle-facing candidate with FP32 stored weights hard-fails the deployment gate.
9. A candidate with FP16 stored weights but non-FP32 runtime tensors hard-fails.
10. Every-five-update and formal Frozen candidate evaluation persist the same
    deployment-effective identity that the Kaggle package loader materializes.
11. PPO sampling continues to use the live FP32 policy and does not inherit the
    candidate deployment conversion.

---

# 14. Anti-Regression Example

The historical failure mode that this protocol explicitly forbids is:

```text
Policy-0809 focal prototype/state/option input/layer 0
+ Policy-0806 option layer 1/final norm/action decoder
= Hybrid(0809 trunk, 0806 head)
```

This composition is stationary, but it is not `Policy-0806`. Every historical run
known to have used this route must be retained and labeled:

```text
Hybrid-opponent / invalid-as-Frozen0806
```

It must not be deleted, rewritten, or used as Full-0806 benchmark evidence. A future
intentional hybrid requires its own experimental policy ID and immutable manifest.

---

# 15. Authority and Change Control

This file is the only canonical V1 protocol. Project design notes, run manifests,
and implementation comments may link to it but must not create an independent copy
of these rules. A future protocol revision must be explicit, versioned, and must not
retroactively relabel old Hybrid evidence as a full frozen policy.

```text
requested:
Frozen Policy-0806

actual CUDA execution:
Policy-0809 focal prototype/state/option trunk
+
Policy-0806 final option block/decoder
```

Even if:

- all reused focal modules are frozen,
- the hybrid opponent is stationary,
- throughput is higher,
- training appears stable,
- win rate improves,

the run is semantically invalid as a `Frozen Policy-0806` run.

A stationary hybrid is still not the requested policy.

---

# 16. Priority Rule

Whenever policy correctness and performance optimization conflict:

> **Correctness wins.**

The required engineering order is:

```text
Policy identity correctness
        ↓
CPU/CUDA semantic parity
        ↓
evaluation/training reproducibility
        ↓
routing/batching optimization
        ↓
throughput
```

No throughput gain justifies silently changing the opponent policy.

---

# 17. Short Form: Rules That Must Never Be Violated

1. A `policy_id` uniquely identifies immutable effective inference weights.
2. Focal-model changes must not alter any existing opponent policy.
3. Frozen-0806 means full 0806; Frozen-0809 means full 0809.
4. Promoted snapshots must be exactly reconstructable forever.
5. Hybrid policies are forbidden unless explicitly registered under a distinct policy ID.
6. Training and evaluation use the same policy-resolution rules.
7. Opponent-pool sampling binds each episode/lane to a concrete full policy.
8. Batching/routing optimizations may group policies but may not change them.
9. Cross-policy sharing requires proof of effective-weight identity.
10. Policy identity violations are fatal errors.
11. Formal evaluation results require a passing opponent-policy identity audit.
12. Promote Champion remains a human decision.
13. Kaggle-facing candidate evidence requires FP16 storage, FP32 runtime, and a
    passing candidate deployment identity audit; raw FP32 candidate results are
    diagnostics only.

---

## Appendix A — Conceptual Example

Correct:

```text
focal = Policy-0812

opponent lane A = Policy-0806
→ full 0806 effective policy

opponent lane B = Policy-0809
→ full 0809 effective policy

opponent lane C = Policy-update00250
→ exact immutable update00250 snapshot
```

Allowed performance optimization:

```text
group A lanes → batch full 0806 inference
group B lanes → batch full 0809 inference
group C lanes → batch full update00250 inference
```

Forbidden optimization:

```text
all lanes
→ reuse focal 0812 encoder
→ attach each opponent decoder
```

unless those shared modules are proven byte/effective-weight identical to the requested policy components.

---

## Appendix B — Policy Identity Gate

Before a formal RL training or frozen evaluation run begins:

```text
resolve opponent policy IDs
        ↓
load manifests
        ↓
verify component sources/hashes
        ↓
verify no focal substitution
        ↓
verify effective policy hash
        ↓
PASS → run
FAIL → abort
```

This gate is mandatory for Promote Champion evidence.
