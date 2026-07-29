# 0020 CUDA Engine Parity Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `engine_cuda/` from a smoke prototype to a reproducible,
step-parity rollout engine for explicitly approved 0020 deck/opponent packages.

**Architecture:** Keep the unmodified official runtime as the oracle. First
build a fixed-layout CPU POD engine and a private numeric rule-pack extractor,
then run the exact same POD transition code on CUDA. Promotion is deck-reachable
and fail-closed: a complete official-engine trace corpus must match after every
decision before a package can enter CUDA rollout collection.

**Tech Stack:** C++20, CUDA 12.8+, CMake/Ninja, Python 3.11, PyTorch, standard
library `unittest`, official engine runtime.

## Global Constraints

- Never modify `engine/source/`.
- Never copy official card/effect source into public or packaged artifacts.
- Keep generated private rule packs under ignored `engine_cuda/generated/private/`.
- The allowed RNG difference covers generator/seed control only; RNG consumption
  sites, branch semantics, selections, state, reward, and terminal parity remain required.
- No mid-game CPU fallback is allowed in a promoted CUDA environment.
- Unsupported rules must terminate with stable named errors.
- The 0020 policy contract is 192 entities, 128 options, entity categorical width 7.
- Formal gameplay evaluation continues to use the official engine even after
  CUDA rollout promotion.
- Do not commit or push unless the user explicitly requests it.

---

### Task 1: Freeze the Oracle and Differential Corpus Contract

**Files:**
- Create: `engine_cuda/python/ptcg_cuda_engine/parity_contract.py`
- Create: `engine_cuda/tools/capture_official_parity.py`
- Create: `engine_cuda/tests/test_parity_contract.py`
- Create: `experiments/0020_pluggable_deck_rl/cuda_parity/oracle_manifest.json`

**Interfaces:**
- Produces: `ParityFrame`, `ParityTrace`, and `OracleManifest` JSON records.
- Consumes: an external seeded oracle library path, exact deck files, seeds,
  deterministic legal-action schedules, and source/binary hashes.

- [ ] **Step 1: Write the parity schema test**

```python
def test_frame_requires_all_semantic_surfaces() -> None:
    required = {
        "legal_options", "selected_action", "public_observation",
        "hidden_digest", "rng_consumption", "reward", "error",
        "terminal", "winner", "finish_reason",
    }
    assert required <= set(ParityFrame.required_fields())
```

- [ ] **Step 2: Run the test and confirm it fails before implementation**

Run: `python3 -m unittest -v engine_cuda.tests.test_parity_contract`

- [ ] **Step 3: Implement immutable schema and canonical hashing**

```python
@dataclass(frozen=True)
class ParityFrame:
    decision: int
    legal_options: tuple[tuple[int, ...], ...]
    selected_action: tuple[int, ...]
    public_observation: dict[str, object]
    hidden_digest: str
    rng_consumption: int
    reward: tuple[float, float]
    error: int
    terminal: bool
    winner: int
    finish_reason: int
```

- [ ] **Step 4: Implement fail-closed oracle capture**

Require seeded/digest ABI symbols, reject the standard unseeded ABI, record
SHA-256 for oracle, source tree, decks, action schedule, and every trace, and
write atomically under `experiments/0020_pluggable_deck_rl/cuda_parity/`.

- [ ] **Step 5: Run schema tests and a two-seed capture smoke**

Expected: two complete canonical traces or an explicit missing-seeded-oracle
failure; never a partial success record.

### Task 2: Build the CPU POD Semantic Mirror

**Files:**
- Create: `engine_cuda/include/ptcg_cuda/pod_state.h`
- Create: `engine_cuda/include/ptcg_cuda/pod_transition.h`
- Create: `engine_cuda/src/pod_transition.cpp`
- Create: `engine_cuda/tests/pod_transition_test.cpp`
- Modify: `engine_cuda/CMakeLists.txt`

**Interfaces:**
- Produces: `advance_to_decision(PodBattleState&, RulePackView)` and
  `apply_normalized_action(PodBattleState&, ActionSpan, RulePackView)`.
- Consumes: fixed-layout state, numeric rule pack, explicit RNG state, and a
  normalized multi-select action.

- [ ] **Step 1: Add failing turn/KO/deck-out tests**

Cover checkup before turn draw, empty-deck loss only at turn draw, per-turn flag
reset, Active KO with Bench replacement, Prize-taking, last-Prize win, and
no-Active win.

- [ ] **Step 2: Add failing evolution and action-budget tests**

Cover present-at-turn-start, one evolution per turn, Rare Candy timing,
Supporter, manual Energy, Retreat, Stadium, and attack-as-terminal-commit.

- [ ] **Step 3: Implement bounded POD state without heap allocation**

```cpp
struct PodBattleState {
    RngState rng;
    TurnState turn;
    PlayerState players[2];
    CardInstance cards[kMaxBattleCards];
    EffectFrame effects[kMaxEffectFrames];
    SelectionFrame selection;
    TerminalState terminal;
};
```

- [ ] **Step 4: Implement generic transition ordering**

Use explicit phases for setup, main, attack resolution, KO/Prize/replacement,
checkup, and next-turn draw. Remove the smoke behavior where any Active KO is
an immediate win.

- [ ] **Step 5: Run C++ POD tests and compare 1,000 targeted oracle frames**

Expected: zero mismatch and zero allocation after reset.

### Task 3: Generate and Audit the Numeric Rule Pack

**Files:**
- Create: `engine_cuda/tools/private_rule_pack_extractor.cpp`
- Create: `engine_cuda/python/ptcg_cuda_engine/rule_coverage.py`
- Create: `engine_cuda/tests/test_rule_coverage.py`
- Create: `experiments/0020_pluggable_deck_rl/cuda_parity/reachable_rules.json`

**Interfaces:**
- Produces: ignored numeric tables plus a tracked non-reversible coverage manifest.
- Consumes: private official source locally, exact promoted decks, and official traces.

- [ ] **Step 1: Write coverage tests for every promoted deck card**

```python
def test_every_reachable_rule_is_supported_or_named() -> None:
    report = audit_reachable_rules(manifest, rule_pack)
    assert report.silent_missing == ()
    assert report.duplicate_numeric_ids == ()
```

- [ ] **Step 2: Implement a private extractor with provenance hashes**

Emit card, attack, ability, condition, target, modifier, selection, and effect
tables. Do not emit card names/text or source fragments in tracked output.

- [ ] **Step 3: Add opcode implementations by semantic family**

Implement movement/search/shuffle, evolution, attachment, damage/counters,
conditions, continuous modifiers, triggers/delays, Trainer/Ability/Stadium,
copied attacks, and terminal effects. Each family receives oracle fixtures
before it is marked supported.

- [ ] **Step 4: Produce per-deck reachable coverage**

Record supported, named-unsupported, and observed counts by card/effect and
matchup. A deck is promotable only with zero observed unsupported events.

### Task 4: Implement the Exact 0020 Device Codec

**Files:**
- Modify: `engine_cuda/include/ptcg_cuda/opcodes.h`
- Modify: `engine_cuda/include/ptcg_cuda/state_layout.cuh`
- Modify: `engine_cuda/src/engine_kernels.cu`
- Create: `engine_cuda/tests/test_0020_codec_contract.py`
- Create: `engine_cuda/tools/compare_0020_codec.py`

**Interfaces:**
- Produces: device tensors matching `train/0020_pluggable_deck_rl/base_model.py`.
- Consumes: actor-visible POD state and exact registered deck identity.

- [ ] **Step 1: Write shape and overflow tests**

Require 192 entities, 128 options, entity categorical width 7, exact option
order/equivalence, and named overflow with no truncation.

- [ ] **Step 2: Replace prototype codec constants and layouts**

Keep engine state capacity separate from per-policy codec capacity. Route first,
then encode only the 0020 cohort into its exact fixed buffers.

- [ ] **Step 3: Implement elementwise frozen-corpus comparison**

Compare every categorical, numeric, mask, parent, registered-deck, ledger,
event, known-hand, min/max, and source tensor against the canonical CPU 0020
feature compiler.

- [ ] **Step 4: Run the complete frozen corpus**

Expected: zero element mismatch, zero illegal decode, and zero capacity fallback.

### Task 5: Port the Validated POD Transition to CUDA

**Files:**
- Modify: `engine_cuda/src/engine_kernels.cu`
- Modify: `engine_cuda/include/ptcg_cuda/runtime.h`
- Modify: `engine_cuda/src/torch_binding.cpp`
- Create: `engine_cuda/tools/run_step_parity.py`
- Create: `engine_cuda/tests/test_step_parity.py`

**Interfaces:**
- Produces: reset, advance, encode, route, apply, reward, terminal, and digest
  device operations with stable environment IDs.
- Consumes: the same POD structs and rule pack validated on CPU.

- [ ] **Step 1: Compile one shared host/device transition implementation**

Avoid a second hand-written semantic implementation. Isolate only allocation,
launch, and address-space differences behind host/device adapters.

- [ ] **Step 2: Add decision-by-decision differential execution**

For every action, compare options, observations, hidden digest, RNG consumption,
reward, errors, terminal state, winner, and finish reason.

- [ ] **Step 3: Run targeted fixtures and random legal-action fuzzing**

Require zero unexplained mismatch for all reachable rule families and at least
100,000 seeded decisions across promoted matchups.

- [ ] **Step 4: Run supported-host safety checks**

Run memcheck, racecheck, initcheck, and synccheck on native Linux, then a 24-hour
full pipeline soak. WSL/WDDM results do not satisfy this step.

### Task 6: Integrate and Promote 0020 CUDA Rollouts

**Files:**
- Create: `train/0020_pluggable_deck_rl/cuda_rollout.py`
- Create: `train/0020_pluggable_deck_rl/tests/test_cuda_rollout.py`
- Create: `rl_runs/0020_pluggable_deck_rl/versions/<new_version>/artifact/engine_snapshot.json`
- Modify: `experiments/0020_pluggable_deck_rl/DESIGN.md`
- Modify: `experiments/0020_pluggable_deck_rl/DESIGN.html`

**Interfaces:**
- Produces: learner-only GPU rollout tensors and immutable engine/opponent snapshots.
- Consumes: promoted CUDA engine hash, rule-pack hash, exact deck hash, opponent
  catalog snapshot, seeds, and a frozen 0020 policy checkpoint.

- [ ] **Step 1: Add a fail-closed promotion manifest loader**

Reject missing hashes, incomplete parity gates, unsupported deck/opponent pairs,
codec mismatches, and unapproved RNG contracts before allocating environments.

- [ ] **Step 2: Keep the full decision hot path on device**

No JSON, `.cpu()`, `.item()`, host route-count reads, or per-decision Python
actions. Host access is limited to reset scheduling, metrics, checkpoints, and
explicit parity/debug barriers.

- [ ] **Step 3: Benchmark identical hardware and workload**

Compare complete games/s, decisions/s, rollout export, learner inference/update,
peak CPU, RSS, VRAM, transfers, and synchronization against the tuned official
CPU pipeline. Require at least 3x end-to-end throughput.

- [ ] **Step 4: Create a new immutable 0020 version and sync design docs**

Record engine/rule/deck/opponent hashes, parity corpus hash, RNG exception,
throughput, soak result, and remaining evaluation boundary. Never append CUDA
training to V1 or an existing zero-shot version.
