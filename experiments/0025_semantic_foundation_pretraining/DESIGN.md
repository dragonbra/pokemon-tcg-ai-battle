# 0025 Semantic Foundation Pretraining

Status: **V1 semantic contract complete; V2 James Cox Raging Bolt paired BC dataset and trainer ready; formal training not started.**

## Objective

0025 is the self-contained successor to the 0019 universal-winner BC lineage. It keeps the audited full-action imitation contract, exact-deck conditioning, causal ledger, visible board, events, and legal-option ordering, while repairing two representational failures:

1. A card ID was expected to carry most card semantics. Individual attacks, skills, effect chains, typed costs, targets, and numeric rules were collapsed or absent.
2. Several heterogeneous inputs were summarized before an option could ask a specific question of them. The decoder could not independently retrieve board, prototype, resource, and event evidence.

This project does not add source/team identity to the actor. Source remains provenance for split, audit, sampling, and evaluation only.

## Evidence Layers

| Layer | Authority | Use in 0025 |
|---|---|---|
| General game rules | Official rulebook audit | Turn budgets, attack termination, zones, evolution/attachment timing |
| Current card implementation | Read-only official engine tables | Card, Skill, Attack, Effect, Target, TargetCondition, Trigger prototypes |
| Public runtime state | Observation and legal options | Dynamic board, exact visible values, option identity, `attackId`, selections |
| Project modelling | 0025 code | Causal ledger, typed deficit features, tokenization, memory routing, training objectives |

The official engine source is never modified. `full_engine_prototype_export.cpp` includes official headers and writes a derived JSON sidecar; its binary is built only under `engine/build/`.

The extraction evidence records the complete `engine/source` path+content tree digest `a81742957f29a23259b0253b71c7e63831c67b6641440b941e5c4c0d921e753e` and extractor source digest `6587eae6c01f2a3dd69067e9f12169000fb305815e425f33b8c57d6f9d769e8e`.

## Frozen 0019 Lineage

The following implementations were copied into 0025 and are runtime-independent of 0019:

| Frozen file | 0019 source SHA-256 |
|---|---|
| `legacy/base_model.py` | `8ee6f5ba84aa5a5a18919887f499598e03fbfedb6acd4b2d8050427dead2f551` |
| `knowledge/ledger.py` | `ab69d0601441b9c604b1ed0b7e3e4c4aee754b27760f781425a82060ddf768af` |
| `knowledge/state.py` | `89fe7bffdb11fa3acc3ee8ccd91959191e1545922d43daff1bc4d3b1a09cf09c` |
| `data/replay_catalog.py` | `5ae9afc6c7a577b292ed3e08a0639635fccd25d1e8311cfffdf2c13ee9d6723c` |
| `data/replay_contract.py` | `fadb14b9c5459691c165eb6450cd4ac82810d972f7c75d74471a01913029f63a` |
| `data/raw_dataset.py` | `869f47efe45dfa4631dc373531bb5ba180a4ebfa2756dc73186fb156d3459d1a` |
| `storage.py` | `17e1f1a836234cbd83469cdbc783b94907301e8b25caa03d6ade32e861218425` |
| `config.py` | `0cc94bcfa5f297040131b061827111cb851d87251a7ccc004afd41762d939f3c` |

The preserved actor-visible legacy tensors are:

- `global_cat [4]`, `global_num [12]`
- `entity_cat [E, 7]`, `entity_num [E, 5]`
- `option_cat [O, 12]`
- ordered option-index action, `min_count`, `max_count`

Exact-deck, ledger, recent events, and known/unknown opponent-hand information are retained rather than replaced.

## Static Prototype Contract

Two immutable sidecars are stored once per dataset/package:

| Sidecar | Coverage | Bytes | SHA-256 |
|---|---:|---:|---|
| Public CardData/Attack | 1,267 cards, 1,556 attacks | 1,082,972 | `5b45041cd14beed8a847f2e99ce955a105fbeb63d09256769d41f8cc88bdbc9d` |
| Full official engine | 1,267 cards, 433 skills, 1,556 attacks, 3,067 effects | 3,386,641 | `5ab0b28e21d40a17a7b153744332424f08d4e32cde4e2ab20f11cb262e799845` |

The full sidecar preserves:

- card type, Pokémon/evolution type, HP, retreat, weakness, resistance, energy type, official ability/play/delay skill IDs, ordered attack IDs;
- skill type, areas, once-per-turn and activation flags, triggers, ordered effects;
- attack ID, owner card ID, base damage, attack flags, ordered typed energy cost, pre-effects, post-effects;
- effect type, selection type/count/context, values, condition/comparator, loop/priority, target player/areas/conditions, and behavior flags.

Static prototypes are referenced by ID. They are not repeated in every decision.

## Dynamic Decision Schema

All sequence dimensions are ragged and accompanied by masks after collation.

| Memory/input | Per-token shape | Contents |
|---|---:|---|
| Legacy/global | `[4] + [12]` | Phase, select context, turn, counts, old budgets |
| Board entities | `[E,7] + [E,5]` | Card, owner, zone, slot, attachment parent, damage/attachment counts |
| Prototype refs | `[Pc], [Pa], [Ps], [Pe]` | Referenced card, attack, skill, and effect IDs |
| Exact deck | `[D] + [D]` | Unique registered card IDs and multiplicity |
| Causal ledger | `[L,4] + [L,15]` | Visible zones, bounded/exact deck and Prize counts, evidence age |
| Recent events | `[T,8] + [T,4]` | Actor-relative event identity, areas, card and numeric payload |
| Known hand | `[H] + [1]` | Known opponent card IDs and unknown count |
| Turn budget | `[5]` | Supporter, Stadium, hand attachment, Retreat, turn-end flags |
| Semantic option | `[O,16] + [O,14] + [O,14]` | Identity, `attackId`, skill/card context, direct numeric facts, field states |

`FieldState` is categorical and explicit: `PAD=0`, `PRESENT=1`, `UNKNOWN=2`, `NOT_APPLICABLE=3`. A real numeric zero is therefore not confused with missingness or padding.

The v1 direct option numbers include base damage, ordered-cost count, attached count, exact typed matches, typed/total energy deficit, target current/max/remaining HP, base-damage KO indicator, attack termination, and current engine remainder counters.

Important boundary: base damage and conservative Basic/Rainbow matching are implemented. A complete current-state effect interpreter, Special Energy text evaluation, weakness/resistance/tool modifiers, and final prize delta are not yet implemented. Such values remain unknown rather than being guessed. The structured effect tokens make that next implementation finite and auditable.

## Model Data Flow

```text
raw observation + legal options + exact deck + causal history
    |
    +--> legacy global/board tensors --------------------> board memory
    +--> card/attack/skill/effect prototype refs --------> prototype memory
    +--> exact deck + causal ledger ---------------------> resource memory
    +--> events + known/unknown hand --------------------> event memory
    +--> turn budgets -----------------------------------> global state
    +--> option identity + direct typed facts -----------> option queries

option queries
    -> gated cross-attention(board memory)
    -> gated cross-attention(prototype memory)
    -> gated cross-attention(resource memory)
    -> gated cross-attention(event memory)
    -> autoregressive GRU pointer
    -> legal option indices + STOP
```

Each memory has its own Transformer encoder. Every option queries each memory separately. Each new context branch is a zero-initialized gated residual, so the semantic paths begin as controlled additions rather than immediately overwriting option identity.

Prototype sequence length is genuinely variable: a simple Item may contribute one card token and its play skill/effects; a Pokémon may contribute a card token, one or more attack tokens, ability/play/delay skill tokens, and all referenced effect tokens. Padding exists only in batch collation and is excluded by attention masks.

The smoke model uses `D=64`. The current formal default is `D=320`, 8 heads, two blocks per memory, and zero dropout in the framework. Capacity, dropout, and depth are pretraining experiment variables, not settled strength claims.

## Forward Pseudocode

```python
def forward(raw_decision, selected_prefix):
    legacy = frozen_0019_codec(raw_decision.observation)
    causal = ledger.consume(raw_decision.visible_logs)

    option_facts = bind_options(
        legal_options=raw_decision.legal_options,
        attack_id=True,
        prototype_refs=True,
        typed_energy_gap=True,
        base_damage_and_target_hp=True,
        explicit_field_states=True,
    )

    board_memory = board_transformer(legacy.entities)
    prototype_memory = prototype_transformer(
        lookup(card_refs, attack_refs, skill_refs, effect_refs)
    )
    resource_memory = resource_transformer(exact_deck, causal.ledger)
    event_memory = event_transformer(causal.events, causal.known_hand)

    q = encode_option_identity_and_direct_facts(legacy.options, option_facts)
    q = gated_residual(q, attend(q, board_memory))
    q = gated_residual(q, attend(q, prototype_memory))
    q = gated_residual(q, attend(q, resource_memory))
    q = gated_residual(q, attend(q, event_memory))

    hidden = initialize_decoder(raw_decision.global_state, raw_decision.turn_budget)
    hidden = consume_selected_prefix(hidden, q, selected_prefix)
    mask_already_selected_options()
    enforce_min_count_and_max_count_for_stop()
    return pointer_logits(q, hidden) + stop_logit(hidden)
```

## Dataset Build Contract

`data.materialize` streams committed gzip JSONL raw shards, verifies each shard SHA-256, keeps each episode-player causal state chronological, and writes atomic gzip decision shards. Every record separates:

- `actor`: the only fields accepted by model collation;
- `target`: ordered legal-option indices and termination;
- `audit`: identity, split, source ID, and source payload commitment.

Source/team provenance is therefore retained for audit without entering actor forward.

0025 also contains its own frozen archive-to-catalog and catalog-to-raw entrypoints. A later full run can therefore rebuild the complete raw contract without importing 0019 executable code:

```bash
python3 -m train.0025_semantic_foundation_pretraining.data.replay_catalog \
  --archives-dir data/raw/episodes/archives \
  --start-date 2026-07-10 --end-date 2026-08-02 \
  --output <new-catalog.json> --workers <calibrated-workers>

python3 -m train.0025_semantic_foundation_pretraining.data.raw_dataset \
  --catalog <new-catalog.json> --output <new-raw-root> \
  --workers <calibrated-workers>
```

Then the semantic materialization command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 \
  -m train.0025_semantic_foundation_pretraining.data.materialize \
  --raw-root <audited-raw-root> \
  --output <new-unused-versioned-dataset-root> \
  --prototypes train/0025_semantic_foundation_pretraining/assets/official_public_prototypes_v1.json \
  --start-date 2026-07-10 --end-date 2026-08-02
```

The current implementation is intentionally single-process. A full production build should add group-preserving workers and storage guards before execution. Tonight's run used `--max-decisions 8` only.

## V2 First Controlled Experiment: James Cox Raging Bolt

The first strength experiment is intentionally narrow. It combines two exact Kaggle
`TeamNames` as one human expert corpus while retaining them as separate provenance groups:

| Exact source | Winning Episodes | Train decisions | Validation decisions |
|---|---:|---:|---:|
| `James Cox` | 454 | 30,087 | 2,386 |
| `James Cox & Henry Chao` | 619 | 40,484 | 4,217 |
| **Total** | **1,073** | **70,571** | **6,603** |

All 1,073 Episodes use the same exact registered 60-card deck:
`f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a`.
The parent catalog split is preserved whole-Episode: 981 train and 92 validation Episodes.
There is no Episode overlap. The formal evidence is
`data_audit/james_cox_raging_bolt_2026-07-20_2026-08-01.json`; its paired raw-builder
catalog is stored alongside it.

Source identity remains absent from `actor`. The materializer additionally removes the
legacy codec's copied `action` field from `actor.legacy`; the label exists only under
`target.ordered_action`. Split-specific gzip shards are content-hashed and loaded one
shard at a time. A train epoch covers each train decision exactly once per active arm.

### Paired model arms

| Arm | Actor-visible representation | D | Parameters |
|---|---|---:|---:|
| `legacy_default` | frozen ID/location representation | 320 | 7,154,562 |
| `legacy_budget_matched` | same legacy representation, capacity control | 576 | 21,725,570 |
| `semantic` | legacy + typed multi-memory representation | 320 | 23,760,966 |

The first comparison answers both capacity questions: semantic versus the actual old default,
and semantic versus a legacy-only model within about 9% parameter count. Formal semantic and
legacy arms use dropout 0.10. They share dataset, Episode split, batch order, optimizer family,
learning rate, ordered option-pointer + STOP targets, and validation metrics.

The BC objective is token cross-entropy over the full ordered action, including STOP. Every
epoch records online optimization loss/token accuracy/teacher exact from the one update pass,
then full validation loss, token accuracy, teacher-forced exact action, free-greedy exact action,
legal action rate, and action-length accuracy. Offline metrics are imitation evidence only.
Gameplay strength still requires matched official-engine evaluation.

The formal version is `V2_james_cox_raging_bolt_ablation`. It uses W&B online, TensorBoard,
canonical JSONL metrics, a foreground watchdog, and four finite model-only checkpoint slots per
arm (`latest`, best validation loss, best teacher exact, best greedy exact). Optimizer, RNG,
DataLoader position, replay, and other exact-resume state are not saved.

## V1 Smoke And Cost

The 32-step benchmark is from one real 0019 trajectory, episode steps 3 through 62. It includes 5 legal Attack options.

| Measure | Result |
|---|---:|
| Legacy codec (10-run pipeline median) | 0.0754 ms/decision |
| Complete 0025 semantic pipeline (10-run median) | 0.3857 ms/decision |
| 0019 observed padded model-ready cache | 7,370.96 bytes/decision |
| 0025 dynamic gzip stream sample | 378.25 bytes/decision |
| Static public + full-engine sidecars | 4,470,966 bytes once |
| 8-step materializer smoke | 3,826-byte decision shard, reload + forward passed |

The storage formats are not identical: 0019 is the observed padded training cache, while 0025 V1 is compact gzip JSONL. The ratio is useful for capacity planning, not a training-throughput claim. A packed/Arrow training cache remains a later benchmark.

Using the observed 0019 rate of `7,347,132 decisions / 19 days`:

| Range | Status | Decisions | Projected compact storage | 1-thread encoding | Ideal 8-worker encoding |
|---|---|---:|---:|---:|---:|
| 2026-07-10..07-29 | Projection; archives locally available | 7,733,823 | 2.73 GiB | 49.7 min | 6.2 min |
| 2026-07-10..08-02 | Projection; 07-30..08-02 unavailable locally | 9,280,588 | 3.27 GiB | 59.7 min | 7.5 min |

Only 07-10..07-28 has observed 0019 decision counts. Local episode archives currently end at 07-29. No claim is made that the unobserved days have the same volume, and no full build was started.

## Verification

Nineteen tests pass:

- attacks 934 and 935 retain distinct identity, cost, and damage;
- numeric zero, unknown, not-applicable, and padding remain distinct;
- Teal Mask Ogerpon ex attack 120 sees a 3-Grass typed deficit with Lightning attached and 2 with Grass attached;
- source/team fields cannot enter actor payload;
- real-row forward produces finite logits and is padding invariant;
- selected options and STOP obey the autoregressive `minCount/maxCount` contract;
- 0025 imports no executable code from another numbered project.
- the self-contained archive catalog and raw-build entrypoints import successfully.
- exact expert/deck slicing fails closed on duplicate Episodes and non-60-card decks;
- actor payload physically excludes both provenance and the copied target action;
- split shards, hashes, deterministic coverage, shared STOP target, paired metrics, batched
  legal greedy decode, and model-only checkpoint reload are verified;
- a real three-arm GPU smoke completed finite forward/backward, validation, TensorBoard, status,
  and checkpoint publication; all four semantic memory gates moved away from zero.

## Training Boundary And Next Versions

V1 is a research framework, not a trained candidate package. There is no W&B run because no formal BC/value/RL optimization occurred.

Before broader foundation pretraining:

1. Extend/audit the raw corpus through the chosen freeze date.
2. Implement and test the deterministic current-state effect resolver, especially damage modifiers, typed Special Energy behavior, protection, weakness/resistance, tool-modified HP, KO/prize delta, and zone transitions.
3. Benchmark packed/Arrow collation and add group-preserving parallel materialization plus storage/RSS guards.
4. Complete V2 paired BC training and select each arm only by its frozen validation contract.
5. Export both selected policies and run matched official-engine opponents, seeds, and first/second-player balance before making strength claims.
