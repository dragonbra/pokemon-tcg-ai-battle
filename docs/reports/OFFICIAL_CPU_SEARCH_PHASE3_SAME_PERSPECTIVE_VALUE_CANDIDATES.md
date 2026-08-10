# OFFICIAL CPU SEARCH - PHASE 3
# SAME-PERSPECTIVE DETERMINISTIC VALUE-SEARCH CANDIDATE AUDIT

Date: 2026-08-11

Scope: the unmodified official CPU engine in `engine/source/ptcgProgram 22/`, plus the current paired 0031 featurizer / 0036 Value V9 inference assets. No CUDA semantics, production inference changes, Value override, multi-step search, or card-level whitelist is used.

## 1. SAME_PERSPECTIVE_STEP ground truth

For this V0, a branch is a `SAME_PERSPECTIVE_STEP` only when all of the following are proved for one accepted `selected` vector:

1. Root is nonterminal, `root.selectPlayer == focalPlayer`, `root.activePlayerIndex() == focalPlayer`, and `root.phase == GamePhase::Main`.
2. Exactly one public `SearchStep(rootSearchId, selected)` is submitted. No research wrapper advances through a returned forced decision.
3. Branch is nonterminal, `branch.selectPlayer == focalPlayer`, `branch.activePlayerIndex() == focalPlayer`, and `branch.phase == GamePhase::Main`.
4. `branch.turn == root.turn` and `branch.firstPlayer == root.firstPlayer`. The active player is derived as `((turn + 1) ^ firstPlayer) & 1` ([State.h](../../engine/source/ptcgProgram%2022/State.h#L1217)).
5. The search-side `Game::rng` serialization is identical immediately before and after this `SearchStep`.
6. The branch's Value-visible features are invariant under relevant hidden-zone determinizations, and match an equivalent live transition from the same root with a branch-local inference context.

`Search::step` clones the source `State`, validates the selection, executes it, and automatically calls `State::step()` while `selectMax == 0` ([Search.h](../../engine/source/ptcgProgram%2022/Search.h#L166)). The returned boundary is therefore the first later selection node or terminal state, not necessarily a complete human-level action.

Perspective is selected by code, not inferred from context: `ToJsonApi` sets `playerIndex = state.selectPlayer`, then uses it for logs and `Current` ([ToJson.h](../../engine/source/ptcgProgram%2022/ToJson.h#L285)); `Current` writes that value as `yourIndex` ([ToJson.h](../../engine/source/ptcgProgram%2022/ToJson.h#L139)). Thus, if root and branch `selectPlayer` both equal the same focal player, the serializer uses the same player index.

Important qualification: same player index does **not** make Search JSON byte-identical to live agent JSON. `Search::start` fills hidden deck, Prize, and opponent-hand placeholders from caller hypotheses ([Search.h](../../engine/source/ptcgProgram%2022/Search.h#L89)). In the probe, reconstructed Prize identities appeared in Search JSON while equivalent live JSON retained `null`. All six accepted cases differed at the twelve `players[*].prize[*]` paths. Current 0031 features nevertheless matched because `_visible_counts()` deliberately counts own active/bench/hand/discard and public fields, not Prize identities ([knowledge/state.py](../../train/0040_dragapult_0809_action_boundary_rl/semantic_policy/knowledge/state.py#L193)); exact hidden membership is established only from a legitimate full-deck view ([knowledge/state.py](../../train/0040_dragapult_0809_action_boundary_rl/semantic_policy/knowledge/state.py#L326)). This behavior must remain pinned by parity tests.

## 2. Decision-node rule

Eligibility is attached to the current official decision node and its concrete handler path, not to a human label or an entire `selectType`.

- A Basic Bench choice is a `Main` node whose `Play` option encodes a hand index. `SelectedPlay` moves that Pokémon from Hand to Bench ([GameProc.h](../../engine/source/ptcgProgram%2022/GameProc.h#L128)).
- A manual Energy target is a `Main` node whose `Attach` option encodes both source hand index and target area/index ([AddOption.h](../../engine/source/ptcgProgram%2022/AddOption.h#L63)); `SelectedMain` dispatches it to `SelectedAttach` ([GameProc.h](../../engine/source/ptcgProgram%2022/GameProc.h#L680)).
- Retreat payment and Retreat switch are two separate nodes. Retreat pushes the switch handler and then the Energy-payment handler ([GameProc.h](../../engine/source/ptcgProgram%2022/GameProc.h#L189)). One payment `SearchStep` stops at `Energy/DiscardEnergy -> Card/Switch`; one target `SearchStep` then returns to `Main`.
- Boss target selection is `Card/Switch` owned by the focal use player even though each option points to an opponent Pokémon. Generic effect ownership starts from `usePlayerIndex` and changes only when `enemySelect` is set ([EffectProc.h](../../engine/source/ptcgProgram%2022/EffectProc.h#L450)); `effectSwitchEnemyBench()` targets the enemy board without setting `enemySelect` ([CreateCard.h](../../engine/source/ptcgProgram%2022/CreateCard.h#L1560)).
- Energy Switch source and destination are separate nodes. The audited destination was `Card/AttachFrom`; the source effect and destination effect are chained in [CreateCard.h](../../engine/source/ptcgProgram%2022/CreateCard.h#L1573), and the destination applies `SwitchEnergyProc` ([EffectInstant.h](../../engine/source/ptcgProgram%2022/EffectInstant.h#L1303)).

Consequently, `Card/Switch` can mean a focal choice over either player's board, while another `Card` node can expose a deck search followed by shuffle. `selectType` or `selectContext` alone is not an eligibility proof.

## 3. Conservative V0 eligibility contract

A candidate is V0-eligible only if every item is `PASS`; `UNKNOWN` means reject.

| Gate | Required evidence |
|---|---|
| Focal ownership | Root `selectPlayer == focalPlayer` |
| Own-turn Main phase | Root and branch internal `phase == Main`; root and branch active player equal focal |
| Actual choice | At least two accepted `selected` vectors **and** at least two distinct Value-visible afterstates; `len(options)` alone is insufficient |
| One-step branch | One public `SearchStep`; no forced-node auto-advance by the research wrapper |
| Nonterminal / same perspective | Branch nonterminal and `branch.selectPlayer == focalPlayer` |
| Same turn | Exact `turn` and `firstPlayer` equality; derived active player unchanged |
| RNG-free | Search-side RNG state byte/serialization equality before and after the call, plus no source-level RNG site in the handler path |
| No opponent boundary | Returned `selectPlayer` is focal; any opponent-owned return is rejected |
| Information-set safe | Value-visible output invariant under relevant own/enemy deck, Prize, and enemy-hand hypotheses; source handler must not read hidden identity/order |
| Feature compatible | A clone of root causal feature state consumes branch logs/observation; every `DecisionBatch` tensor equals the equivalent live path |
| Value compatible | Paired 0036 Value V9 produces exact equal output for live and Search batches |

Two additions to the preliminary contract are material:

- `root/branch.phase == Main` and `activePlayerIndex == focal` are needed to mean “own turn internal,” not merely “turn did not change.” Public Search JSON does not serialize `phase`; a fully reliable checker therefore needs internal Search `State` access or an additional trusted bridge.
- Multiple syntactically accepted selections are not enough. Ordered aliases or semantically duplicate options must produce at least two distinct Value-visible afterstates.

Post-hoc RNG comparison alone is not a safe production guard because branches share search-side `Game::rng`: discovering a failure means RNG was already advanced for sibling branches. V0 needs a source-audited RNG-free path or isolated search-side RNG per candidate before it can call the transition safely.

## 4. Same-perspective decision taxonomy

This is an engine-primitive matrix, not a card whitelist. “PASS (traced)” applies only to the exact handler shape tested; it does not authorize every caller sharing that type/context.

| Decision family | Type/context | Multi-choice | One step | Same focal | Same turn/Main | RNG-free | Hidden-safe | Feature-compatible | V0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Basic board placement | `Main/Main`, `Play` Basic | PASS | PASS -> `Main` | PASS | PASS | PASS | PASS | PASS | PASS (traced) |
| Manual Energy target | `Main/Main`, `Attach` | PASS | PASS -> `Main` | PASS | PASS | PASS | PASS | PASS | PASS (traced) |
| Retreat Energy payment | `Energy/DiscardEnergy` | PASS | PASS -> `Card/Switch` | PASS | PASS | PASS | PASS | PASS | PASS (traced) |
| Own switch target | `Card/Switch` | PASS | PASS -> `Main` | PASS | PASS | PASS | PASS | PASS | PASS (traced) |
| Opponent Pokémon target chosen by focal | `Card/Switch`, enemy-board payload | PASS | PASS -> `Main` | PASS | PASS | PASS | PASS | PASS | PASS (traced) |
| Visible attached-resource destination | `Card/AttachFrom` in Energy Switch chain | PASS | PASS -> `Main` | PASS | PASS | PASS | PASS | PASS | PASS (traced) |
| Tool attachment target | `Main/Main`, `Attach` | possible | possible | likely | likely | UNKNOWN | UNKNOWN | UNKNOWN | REJECT pending trace |
| Evolve target | `Main/Evolve` or `Evolve/Evolve` | possible | handler-dependent | handler-dependent | handler-dependent | UNKNOWN | UNKNOWN; evolution triggers may run | UNKNOWN | REJECT |
| Damage/heal/counter allocation | `Card` plus damage/heal contexts | possible | handler-dependent | handler-dependent | often | UNKNOWN | KO/triggers can cross boundaries | UNKNOWN | REJECT |
| Generic Ability/effect target | `Card/EffectTarget` or `Main/Ability` | possible | arbitrary chain | arbitrary | arbitrary | arbitrary | arbitrary | UNKNOWN | REJECT |
| Deck search target | `Card/ToHand`, `ToBench`, etc.; often `selectDeck` | possible | may complete | often focal | often same | frequently shuffle | FAIL: hidden deck identity/order | not audited | REJECT |
| Draw-producing action | often `Main/Play` or ability chain | possible | may return `Main` | can PASS | can PASS | can PASS | FAIL: fixed hidden deck order | FAIL under perturbation | REJECT |
| Random selection / coin | several types/contexts | any | arbitrary | arbitrary | arbitrary | FAIL | FAIL | not relevant | REJECT |
| `Attack` / `End` | `Main`/`Attack` or `End` option | possible | crosses resolution | perspective may flip | FAIL | handler-dependent | handler-dependent | not audited | REJECT by scope |
| `Skill`, `Count`, `YesNo`, `SpecialCondition`, `CardOrAttachedCard` | context-dependent | possible | arbitrary | arbitrary | arbitrary | UNKNOWN | UNKNOWN | UNKNOWN | REJECT without handler proof |
| Forced singleton | any type/context | FAIL actual-choice gate | yes | any | any | any | any | any | REJECT (no argmax value) |

Draw is the decisive counterexample. Official `Draw` simply moves cards from the end of the already ordered deck and does not access RNG ([CardMove.h](../../engine/source/ptcgProgram%2022/CardMove.h#L285)). Billy & O'Nare calls that draw effect ([CardImpl.h](../../engine/source/ptcgProgram%2022/CardImpl.h#L3801)). Its SearchStep was same focal, same turn, Main-phase, and RNG-free, yet swapping hidden top-deck identities changed returned hand/logs and Value inputs. Therefore `RNG-free != information-set deterministic`.

## 5. Runtime testcase results

Probe: [official_cpu_search_same_perspective_probe.cpp](../../tests/official_cpu_search_same_perspective_probe.cpp) through the public `SearchBegin/SearchStep` API, with internal metadata read only for audit.

| Test | Root -> returned boundary | Actor/turn | RNG | Hidden perturbation at Value input | Result |
|---|---|---|---|---|---|
| A Bench placement | `Main/Play Basic -> Main` | `0 -> 0`, turn `1 -> 1`, Main -> Main | unchanged | invariant | PASS |
| B Energy attach target | `Main/Attach -> Main` | `0 -> 0`, turn `1 -> 1`, Main -> Main | unchanged | invariant | PASS |
| C Retreat switch target | `Card/Switch -> Main` | `0 -> 0`, turn `3 -> 3`, Main -> Main | unchanged | invariant | PASS |
| D Retreat payment | `Energy/DiscardEnergy -> Card/Switch` | `0 -> 0`, turn `3 -> 3`, Main -> Main | unchanged | invariant | PASS |
| E Opponent board target | focal `Card/Switch -> Main` | `0 -> 0`, turn `3 -> 3`, Main -> Main | unchanged | invariant | PASS |
| F Trainer effect destination | `Card/AttachFrom -> Main` | `0 -> 0`, turn `3 -> 3`, Main -> Main | unchanged | invariant | PASS |
| G Hidden draw counterexample | `Main/Play -> Main` | `0 -> 0`, turn `3 -> 3`, Main -> Main | unchanged | **changed** | REJECT as designed |

Every positive root had two legal candidate selections. Candidate A and B produced different serialized internal State hashes and visibly different board/resource allocations. The RNG hash was `6fc53fd01bb1bca8` before and after every traced branch, including the rejected draw counterexample.

The hidden probe jointly permuted every hidden vector for which a distinct permutation existed: focal deck order, focal Prize, enemy deck order, enemy Prize, and enemy hand. Raw Search observations changed because reconstructed Prize identities are serialized; all six positive cases nevertheless retained exact Value-input equality. The draw counterexample changed `card_cat`, `event_cat`, `option_cat`, and `option_source`.

## 6. Equivalent live-state and Value-input parity

Current input path:

```text
official observation
  -> OnlineCausalEncoder.encode
  -> mutable CausalKnowledge.consume(logs/current)
  -> CausalSnapshot
  -> compile_canonical_row
  -> collate_canonical_records
  -> DecisionBatch tensors
  -> frozen 0031 state + option encoder
  -> 0036 latent-query Value head
```

`OnlineCausalEncoder` requires the fixed actor, exact registered 60-card deck, official select/options, and a chronological mutable `CausalKnowledge` ([online_runtime.py](../../train/0040_dragapult_0809_action_boundary_rl/semantic_policy/deployment/online_runtime.py#L38)). `CausalKnowledge` persists exact/bounded resource ledger state, known/possible/remembered opponent cards, exact deck knowledge when legitimately revealed, decision/event indices, and a recent event window ([knowledge/state.py](../../train/0040_dragapult_0809_action_boundary_rl/semantic_policy/knowledge/state.py#L71)). The Value model consumes both encoded state tokens and the returned decision's option embeddings ([value_network.py](../../archive/pretrained/0031_friend_0809_gsb_v5_value_v9/value_source/value_network.py#L114)). It is not an observation-only scalar function.

The parity probe follows the required fork point:

```text
consume live root observation once
          |
          +-- clone causal context -> consume real successor OA -> FA -> Value A
          |
          +-- clone causal context -> consume Search successor OS -> FS -> Value S
```

For Bench, Attach, Retreat payment, Retreat target, Boss target, and Energy Switch destination:

- every `DecisionBatch` key, shape, dtype, and element matched exactly (`FA == FS`);
- paired 0036 Value V9 outputs matched exactly (`Value A == Value S`);
- candidate hidden perturbations left Search `DecisionBatch` unchanged;
- the root causal context's decision index remained unchanged after all branch encodes.

The official incremental branch logs were sufficient for these six transitions. They are not proof that logs suffice for every handler.

A simple call of `production_featurizer(search_observation)` on the live encoder is unsafe: `consume()` appends events, updates hidden-card memory/ledger state, and increments its decision index ([knowledge/state.py](../../train/0040_dragapult_0809_action_boundary_rl/semantic_policy/knowledge/state.py#L452)). The research probe uses an explicit clone of every mutable `CausalKnowledge` container while sharing immutable prototypes. Production currently exposes no supported clone/fork API, so integration is not ready even though the six parity cases pass.

## 7. Hidden-information safety boundary

Finite perturbation is a bug detector, not a proof over all determinizations. V0 hidden safety needs both:

1. a source-level handler proof that the one-step path reads/writes only actor-visible board/resource state and does not consult hidden identity/order; and
2. adversarial reconstruction probes showing invariant Value inputs across own/enemy deck, Prize, and opponent-hand hypotheses.

The six positive fixtures satisfy both for the traced simple-card path. This cannot be generalized to all instances of those contexts: a switch, attachment, evolution, or damage operation may run card-specific triggers, prevention effects, KO work, or later decisions. Dynamic boundary checks detect the returned actor/turn/RNG outcome, but source/effect identity remains necessary to establish hidden safety before execution.

## 8. Research-only eligibility checker

The fail-closed prototype is [official_cpu_search_v0_eligibility.py](../../tests/research/official_cpu_search_v0_eligibility.py), tested by [test_official_cpu_search_v0_eligibility.py](../../tests/test_official_cpu_search_v0_eligibility.py). It returns:

```text
ELIGIBLE
REJECT_NOT_ACTUAL_CHOICE
REJECT_PERSPECTIVE_FLIP
REJECT_TURN_CHANGE
REJECT_TERMINAL
REJECT_RNG
REJECT_HIDDEN_SENSITIVE
REJECT_FEATURE_INCOMPATIBLE
UNKNOWN
```

It accepts only explicit evidence and intentionally has no `selectType -> safe` mapping. Missing legal-selection count, distinct afterstate count, internal Main-phase evidence, actor/turn boundary, RNG equality, hidden invariance, or feature parity returns `UNKNOWN`.

Public Search observation/metadata alone is **not sufficient** for the complete classifier:

- it exposes `yourIndex`, `turn`, and `firstPlayer`, so focal ownership and derived active player are available;
- it exposes terminal result and next decision options;
- it does not expose internal `GamePhase`;
- it does not expose RNG before/after;
- it cannot prove that no hidden identities were read;
- it cannot fork the mutable inference-side causal context;
- it cannot prove that multiple accepted selections produce distinct Value-visible afterstates without executing/encoding them.

The proposed loop

```text
while uniquely forced AND focal-owned AND deterministic:
    SearchStep(unique_selection)
```

remains outside V0. Phase 3 evaluates only one node, and forced singletons fail the “actual choice” gate.

## 9. Reproduction

Commands:

```bash
python3 tests/official_cpu_search_same_perspective_probe.py
python3 tests/official_cpu_search_value_feature_probe.py --with-value-forward
python3 -m unittest tests.test_official_cpu_search_v0_eligibility -v
```

Generated local evidence:

- `.tmp/official_cpu_search_same_perspective/probe_output.jsonl`
- `.tmp/official_cpu_search_same_perspective/feature_summary.json`
- `.tmp/official_cpu_search_same_perspective/value_forward_run.log`

Results: official CPU probe `PASS`; feature/actual Value forward probe `PASS`; eligibility unit tests `4/4 PASS`. `engine/source/` was not modified.

## 10. Conclusion

Official CPU Search supports a narrow `SAME_PERSPECTIVE_STEP` when a focal-owned, own-turn Main-phase decision is advanced by exactly one `SearchStep` and returns nonterminal to the same focal player without changing turn, active player, phase, or RNG. In traced Basic placement, manual Energy target, Retreat payment, Retreat target, focal-owned opponent target, and visible attached-resource destination nodes, branch-local 0031 features and actual 0036 Value V9 outputs exactly matched equivalent live transitions and were invariant to joint hidden-zone perturbations. This is not derivable from `selectType/context` alone: draw proves same-focal + same-turn + RNG-free can still be hidden-sensitive, and raw Search JSON carries reconstructed hidden Prize identities. A research V0 checker is feasible only with internal phase/RNG evidence, handler-level hidden/RNG audit, and a forkable causal feature context; the current production pipeline lacks that supported fork and therefore should not yet enable Value override.
