# Full implementation and acceptance plan

## Scope and invariant

The deliverable is a general battle engine, not an environment specialized for
Marnie, Alakazam, or another deck. A deck changes reset data and which rules are
reachable; it must not change engine code. One learner policy may be fine-tuned
while at least ten frozen BC policies remain resident and act as opponents.

The official/local CPU engine remains the semantic oracle until every promoted
CUDA rule has passed differential tests. Unsupported behavior must return a
named error; it must never fall back to the CPU in the middle of a game because
that would reintroduce synchronization and hide coverage gaps.

## Phase 0: freeze contracts and baselines

1. Record the exact CPU engine binary SHA256, source snapshot hash, compiler,
   card database hash, PolicyCodecV1 version, deck hashes, and RNG contract.
2. Freeze replay fixtures covering setup, every selection context, status and
   knockout ordering, delayed effects, copied attacks, and game termination.
3. Measure end-to-end games/s, all engine selections/s, learner decisions/s,
   JSON/codec/H2D/inference/D2H/Select time, peak RSS, and GPU peak allocation.
4. Inventory every pool checkpoint: architecture adapter, input codec, output
   decoder, parameter count, dtype, exact deck hash, and evaluation provenance.

Acceptance: the baseline can be reproduced on a clean host and every artifact
has a recorded hash. The Ninetales 660 / Amarys 1207 case is a separately named
known divergence and is excluded from ordinary win-rate accounting.

## Phase 1: CPU POD mirror and numeric rule pack

1. Replace pointers, STL containers, RTTI, exceptions, virtual calls, and
   recursive effect execution with fixed-width IDs, bounded arrays, and an
   explicit effect stack.
2. Define numeric card, attack, skill, target, condition, modifier, and effect
   tables. Keep the generator private; publish neither official source nor a
   reversible source dump.
3. Implement deterministic RNG with one state per environment and document the
   exact consumption order.
4. Run the POD mirror on CPU first. Compare legal options, normalized action,
   public observation, hidden-state digest, RNG state, reward, and terminal
   result after every decision.

Acceptance: zero mismatch on at least 100,000 seeded decisions and on all
targeted fixtures. Every reachable rule is either supported or maps to a named
unsupported opcode. No heap allocation occurs after reset.

## Phase 2: CUDA state and interpreter

1. Port the validated POD structs unchanged to device memory. Start with one
   CUDA thread per environment; use bounded interpreter budgets to contain
   divergent or cyclic effects.
2. Use fixed-capacity state, option, effect-stack, delayed-effect, log, and
   scratch buffers. Add canaries and high-water counters in debug builds.
3. Split kernels by synchronization boundary: reset, advance-to-decision,
   encode, route, apply action, reward/terminal extraction, and selective debug
   digest.
4. Keep rule tables read-only and shared across environments. Evaluate constant
   memory only for genuinely hot small tables; large rule packs stay in global
   read-only memory.
5. Compact or regroup environments only when profiling proves divergence is a
   bottleneck. Preserve stable environment IDs for replay and learner storage.

Acceptance: CPU POD and CUDA have zero step-level mismatch on the Phase 1
corpus. Compute Sanitizer reports no race, out-of-bounds, uninitialized read,
or leak. At least 4,096 simultaneous environments run for one million actions
without an engine error outside the declared unsupported list.

## Phase 3: device codec registry and action decoder

1. Route by `policy_id` and `codec_id` before encoding. Emit each codec into a
   reusable fixed-capacity cohort buffer so heterogeneous policies do not force
   every environment to carry every codec layout.
2. Implement the exact legacy ID-only codecs used by the packaged pointer,
   mean-pool, recurrent, and goal-query BCs; then the Marnie prize/compact
   extensions and PolicyCodecV1. The standalone Alakazam strategy runtime is
   outside the opponent-pool scope; a BC Alakazam model uses the normal ID-only
   path. A deliberate distillation to one codec is an alternative only after
   action-parity and battle A/B gates; it is not an implicit conversion.
3. Differentially compare every tensor element with each validated CPU codec,
   including entity references, option equivalence, min/max selection,
   hidden-information boundaries, actor-relative ownership, deck-registration
   features, and auxiliary inputs.
4. Implement greedy/sampled multi-select normalization on device: legal mask,
   stop token, uniqueness, equivalence handling, min/max cardinality, and stable
   option-to-engine mapping.
5. Remove all hot-loop `.cpu()`, `.numpy()`, `.tolist()`, `.item()`, JSON, and
   Python action-list construction. Add a test that profiles and fails when a
   device synchronization appears inside rollout.

Acceptance: zero mismatch over at least the existing 127,205-decision
PolicyCodecV1 corpus plus a frozen corpus for every legacy codec. Illegal
decoded actions are zero. Hidden-state features never appear in either
player's policy input.

## Phase 4: heterogeneous 10+ BC policy pool

1. Give each acting seat a `policy_id`. Group policies by adapter family, codec
   version, dtype, and static tensor shape.
2. Keep all frozen weights resident. Route ready environments into fixed padded
   device cohorts; do not read cohort counts on the host. Initially launch each
   configured policy adapter over its fixed cohort. Then capture stable paths
   with CUDA Graphs or compile compatible models into grouped execution.
3. Support standard pointer, mean-pool, recurrent decoder, and auxiliary-head
   BC adapters behind one `act_device` contract. Auxiliary outputs must not
   alter action semantics unless explicitly configured.
4. Report raw option-index parity and `option_equiv`-canonicalized parity
   separately. An inequivalent CPU/GPU flip is not a codec success. Either
   retrain/export the checkpoint with a measured decision-margin requirement,
   or promote a separately versioned GPU policy after seeded battle A/B tests;
   never hide flips with arbitrary logit rounding.
5. Separate frozen inference mode from the learner: frozen parameters have no
   gradients or optimizer state; the learner retains BC initialization and is
   the only trainable policy in rollout.
6. Sample opponent IDs and seats on device or once at reset. Record opponent
   policy/deck IDs in rollout tensors for stratified metrics and curriculum.

Acceptance: at least ten frozen policies plus one learner load concurrently;
each receives only its assigned states; same-state actions match its packaged
runtime; no per-decision D2H/H2D transfer appears in an Nsight trace. Pool
composition can change through a manifest without rebuilding engine kernels.

## Phase 5: RL integration

1. Expose device tensors through a PyTorch custom op without copying. Keep
   rollout observations, legal masks, actions, log probabilities, values,
   rewards, done flags, opponent IDs, and recurrent state on GPU.
2. Use asynchronous reset queues and fixed rollout horizons. Export only the
   learner's transitions to PPO/other RL loss; frozen opponent decisions do not
   enter the learner optimizer.
3. Use separate CUDA streams only after correctness: engine/codec, opponent
   inference, learner inference, and update. Add events for explicit ownership
   rather than global synchronization.
4. Add checkpoint/resume state for learner, optimizer, scheduler, RNG streams,
   league manifest, deck/rule hashes, and environment episode counters.

Acceptance: resumed training reproduces the next rollout digest; learner-only
gradients are confirmed; training survives 24 hours without memory growth;
throughput and policy latency meet the gate below.

## Phase 6: promotion and continuous parity

1. Run exact seeded CPU-vs-CUDA paired games for every supported matchup and
   random legal-action fuzzing for rare branches.
2. Gate every rule-pack or engine change on option/state/RNG/terminal parity,
   card/opcode coverage, sanitizer, deterministic replay, and performance.
3. Report unsupported-opcode frequency by card and matchup. A deck enters the
   RL opponent pool only when all rules reachable from both decks are covered.
4. When an official fixed engine becomes available, add it beside the frozen
   old engine, reproduce the 660/1207 case, and migrate only after paired
   regression. Never rewrite historical baseline results.

Acceptance: zero unexplained mismatch, zero silent fallback, zero unsupported
event in the promoted league corpus, and identical outcome/action traces for
the deterministic acceptance set.

## Performance and memory gates

- 4,096 environments are the first correctness/performance tier; 16,384 is the
  target throughput tier on a 32 GiB training GPU.
- Engine-only operational memory targets are 0.3-0.5 GiB at 4,096 environments
  and 1.1-1.7 GiB at 16,384, including state, codec tensors, routes, scratch,
  and safety headroom but excluding model weights and learner optimizer state.
- Ten frozen BC models must remain resident. Frozen weights should use BF16 or
  FP16 after action-parity validation. Models are executed sequentially by
  routed cohort initially so activation workspaces can be reused.
- The learner memory budget includes training weights, gradients, FP32 master
  weights when used, optimizer moments, activations, and rollout storage.
- Promotion requires at least 3x end-to-end games/s over the best tuned CPU
  pipeline on the same host, p99 decision latency below the CPU path, and no
  host synchronization in the steady-state rollout range. The final target is
  set from an Nsight trace, not kernel-only microbenchmarks.

## User acceptance checklist

Run and archive the following evidence:

1. Build manifest and hashes for CPU oracle, CUDA build, rule pack, card data,
   all decks, all policy checkpoints, and PolicyCodecV1.
2. Unit tests for every opcode and selection context.
3. At least 100,000 decision-level CPU POD parity cases.
4. At least 127,205 device codec parity cases.
5. Full seeded CPU-vs-CUDA paired games across all promoted deck pairs.
6. Random legal-action fuzzing and malformed-action rejection.
7. Compute Sanitizer and 24-hour soak reports.
8. An Nsight Systems trace proving the hot loop has no per-decision host copy
   or synchronization.
9. VRAM measurements at 4,096 and 16,384 environments with 10+ frozen BCs and
   one learner, compared with the estimator.
10. End-to-end throughput, p50/p95/p99 policy latency, GPU utilization, engine
    error counts, unsupported-opcode counts, and per-opponent game statistics.
