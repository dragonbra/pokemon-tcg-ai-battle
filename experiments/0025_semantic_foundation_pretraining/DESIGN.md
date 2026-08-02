# 0025 Semantic Foundation Pretraining

Status: **V4 canonical semantic foundation is implemented, trained, exported, and locally evaluated in the unmodified official engine. Best-greedy is the current research selection; formal V4 publication remains the next gate. V3 remains immutable historical evidence.**

## Objective

0025 is the self-contained successor to the 0019 universal-winner BC lineage. V4 keeps the audited full-action imitation contract, exact-deck conditioning, causal ledger, visible board, events, and legal-option ordering, but **does not keep the legacy actor tensor interface**. It repairs two representational failures:

1. A card ID was expected to carry most card semantics. Individual attacks, skills, effect chains, typed costs, targets, and numeric rules were collapsed or absent.
2. Several heterogeneous inputs were summarized before an option could ask a specific question of them. The decoder could not independently retrieve board, prototype, resource, and event evidence.

This project does not add source/team identity to the actor. Source remains provenance for split, audit, sampling, and evaluation only.

## Current V4 Canonical Contract

V4 is a clean actor/model boundary, not a compatibility adapter around V3. Raw observations and
official prototypes compile directly to named typed records. The exact accepted actor keys are
declared once in `features/canonical/schema.py`; model forward rejects every missing or extra key.

| Family | Cat width | Numeric width | Variable axis | Meaning |
|---|---:|---:|---:|---|
| Global | 11 | 17 | one token | Select context, turn budgets/status, zone counts, deck/hand/Prize pressure, known-hand boundary |
| Card instance | 5 | 7 | `C` | Visible card identity/owner/zone/kind/status, HP/damage, attachments, evolution parent |
| Resource ledger | 4 | 15 | `R` | Exact registered-deck multiset, visible zones, bounded/exact deck and Prize counts, evidence age |
| Recent event | 8 | 4 | `E` | Actor-relative log type, card/areas/visibility and finite numeric payload |
| Legal option | 14 | 16 + 16 states | `O` | Action/source/target/attack/energy identity, direct attack and HP facts, explicit value state |
| Option-skill relation | ID + role + parent | - | `S` | Actual official skill ID and the exact option it belongs to |
| Option-effect relation | ID + role + parent | - | `F` | Official effect-chain node and the exact option it belongs to |

Every categorical column owns a separate embedding. Numeric columns use numeric MLP projections.
`PAD=0`, `PRESENT=1`, `UNKNOWN=2`, and `NOT_APPLICABLE=3` are distinct. Padding is introduced only
by collation and is excluded with masks. No source/team identity, copied target action, or `legacy`
payload can enter actor forward.

```text
global + card instances + exact-deck ledger + recent events
    -> typed token projections + Card prototypes
    -> 4-layer state Transformer
    -> full state memory + attention-pooled state summary

legal options
    + source/target card-instance gathers
    + Card/Attack prototype lookup
    + explicitly parented Skill/Effect relations
    -> typed option projection
    -> 3-layer Transformer decoder cross-attending full state memory
    -> semantic option tokens

state summary + semantic option tokens
    -> autoregressive GRU pointer
    -> unique ordered legal-option indices + STOP
```

The production configuration is `D=320`, 8 heads, FFN multiplier 3, dropout 0.10, four state
layers and three option cross-attention layers. It has 21,837,082 trainable parameters. Current
official ID capacities are finite and fail closed: Card 2,048, Attack 2,048, Skill 512, Effect
4,096; the read-only prototype snapshot maxima are 1,267 / 1,556 / 434 / 3,067 respectively.
Live inference additionally fails closed above 128 legal options or 64 selected action steps; the
current canonical corpus maxima are 67 and below 64 respectively.

Deterministic current-state features presently include base damage, typed and total Energy deficit,
attached/required Energy counts, target/source current and maximum HP, HP after base damage, and a
base-damage KO flag. Dynamic effect resolution, Special Energy text, weakness/resistance, tool HP
modifiers and final prize delta remain explicit unresolved work; V4 supplies structured Effect
tokens instead of guessing these values.

## Current V4 Dataset

`V2_james_cox_raging_bolt_canonical` was compiled directly from the committed raw corpus in 92.57
seconds using one CPU core and 267 MiB peak RSS. It contains 77,174 decisions in 28,139,349 bytes
of compressed shards plus prototype sidecars:

- train: 70,571 decisions / 981 complete Episodes;
- validation: 6,603 decisions / 92 complete Episodes;
- exact deck SHA-256: `f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a`;
- canonical manifest SHA-256: `619311eb3df8c6f1ebef874670363988c26dd7665325b9d6fef8adc4bef4f8cf`;
- observed maxima: 110 card tokens, 24 ledger tokens, 64 events, 67 options, 36 skill relations, 166 effect relations.

All 77,174 rows passed categorical-vocabulary, finite-numeric, relation-index and actor-leakage
audits. Dataset shards and the two prototype sidecars are content-hashed. Source IDs 73/74 are
stored only under audit and manifest provenance.

## V4 Formal BC Outcome

`V4_canonical_semantic_foundation` trained from random initialization on CUDA with batch 384,
validation batch 512, seed `20260802`, AdamW and patience 6. The foreground watchdog completed
normally after 18 epochs / 3,312 optimizer updates; W&B run
`0025_semantic_foundation_pretraining--v4_canonical_se-455df0e583` is synced. All four retained
files are model-only checkpoints with no optimizer, scheduler, scaler, RNG, DataLoader or replay
state.

| Selection | Epoch | Validation loss | Greedy exact | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| best validation loss | 12 | **0.321322** | 81.01% | `a4e5b07b...fde1cf21` |
| best greedy exact | 14 | 0.324919 | **82.02%** | `adc4eaec...1865aa3` |

V4 exceeded the V3 semantic arm's 73.42% and the V3 legacy-default arm's 79.62% best greedy exact.
This is stronger full-action imitation evidence, not official-engine strength by itself.

## V4 Local Official-Engine Checkpoint Comparison

The canonical online runtime uses the same `CausalKnowledge`, `compile_canonical_row`, prototype
tables and `collate_canonical_records` as materialization. It does not reconstruct legacy tensors
or inject source identity. Both checkpoint packages strict-load their model-only payload and pass
the standard exact-60-card package validator.

Both strict local research runs used the unmodified official engine, Frozen pool
`0019_foundation_51_exact_decks_v4`, 51 opponents x 10 games, alternating five first/five second,
base seed 22022, eight isolated workers, one CPU thread per worker, shared `cuda:0` inference, and
metric profile `auto_iteration_v8_setup_relay`. Canonical inference is fail closed: online encoder
exceptions or illegal decoded actions become evaluation errors instead of silently selecting the
minimum legal action. The runtime seeds Python, NumPy, and Torch, but the official engine's internal
RNG is not exposed. The two checkpoint results are therefore same-contract independent stochastic
samples, not paired trials, even when their nominal per-game seeds match.

| Local checkpoint | W-L-D | Win rate | First / second | Complete / errors | Wall time |
|---|---:|---:|---:|---:|---:|
| best validation loss | 149-361-0 | 29.22% | 31.37% / 27.06% | 510/510 / 0 | 674.91 s |
| best greedy exact | **153-357-0** | **30.00%** | 31.76% / 28.24% | 510/510 / 0 | 655.19 s |

Best-greedy gains four wins / 0.78 percentage points in these independent samples. That difference
is too small to claim a stable strength advantage, but it is directionally consistent with the
offline selection metric. The attack-quality proxy is more favorable for best-greedy: attacks that
did not take a Prize fall from 58.95% (1,192/2,022) to 48.75% (861/1,766). Best-greedy is therefore
the research selection for the next formal V4 gate, with replay-level semantic error analysis still
required.

Strict local reports are retained under
`.tmp/evaluation/0025_v4_best_loss_frozen_all_fail_closed/` (run
`run-068755f72ad84b84bc0a5716848f1bb9`, SHA-256 `f05e22d7...16738ab`) and
`.tmp/evaluation/0025_v4_best_exact_frozen_all_fail_closed/` (run
`run-d8ab4e9f95824195b2b2031fc0287a79`, SHA-256 `82d07665...8966c39`). Earlier non-strict runs
remain preliminary diagnostic evidence only because their zero-error summaries could not expose
fallback. Local reports are not substitutes for `experiments/0025.../evaluation/V4_*.html` formal
publication.

## Evidence Layers

| Layer | Authority | Use in 0025 |
|---|---|---|
| General game rules | Official rulebook audit | Turn budgets, attack termination, zones, evolution/attachment timing |
| Current card implementation | Read-only official engine tables | Card, Skill, Attack, Effect, Target, TargetCondition, Trigger prototypes |
| Public runtime state | Observation and legal options | Dynamic board, exact visible values, option identity, `attackId`, selections |
| Project modelling | 0025 code | Causal ledger, typed deficit features, tokenization, memory routing, training objectives |

The official engine source is never modified. `full_engine_prototype_export.cpp` includes official headers and writes a derived JSON sidecar; its binary is built only under `engine/build/`.

The extraction evidence records the complete `engine/source` path+content tree digest `a81742957f29a23259b0253b71c7e63831c67b6641440b941e5c4c0d921e753e` and extractor source digest `6587eae6c01f2a3dd69067e9f12169000fb305815e425f33b8c57d6f9d769e8e`.

## Historical Frozen 0019/V3 Lineage

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

The following actor-visible legacy tensors describe V2/V3 only and are not accepted by V4:

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

## Historical V3 Dynamic Decision Schema

This section records the superseded V3 experiment contract. All sequence dimensions were ragged and accompanied by masks after collation.

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

## Historical V3 Model Data Flow

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

The current canonical materialization command is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 \
  -m train.0025_semantic_foundation_pretraining.data.materialize_canonical \
  --raw-root <audited-raw-root> \
  --output <new-unused-versioned-dataset-root> \
  --prototypes train/0025_semantic_foundation_pretraining/assets/official_public_prototypes_v1.json
```

The current implementation is intentionally single-process, bounded by 2,048-record shards, and
checks storage after every closed shard. The complete James Cox build is finished; no projected
or partial count is used for V4 training.

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

The first formal attempt, `V2_james_cox_raging_bolt_ablation`, was interrupted by a
user-requested host reboot after two complete epochs. Its original metrics, twelve model-only
checkpoint slots, TensorBoard event, and W&B identity are retained as incomplete evidence. The
reboot left 2,159 NUL bytes after the two valid epoch records in the canonical JSONL; the file is
preserved rather than rewritten. V2 is not resumed and is not a completed comparison.

The replacement formal version is `V3_james_cox_raging_bolt_restart`. It constructs all three
models and AdamW optimizers from scratch with seed `20260802`; no V2 checkpoint, optimizer, RNG,
DataLoader, or W&B state is loaded. It uses W&B online run
`0025_semantic_foundation_pretraining--v3_james_cox_ra-126d724e53`, TensorBoard, canonical JSONL
metrics, a foreground watchdog, and four finite model-only checkpoint slots per arm (`latest`,
best validation loss, best teacher exact, best greedy exact). The frozen dataset, split, schema,
batch order, training budget, and validation contract remain unchanged so V3 is a clean rerun.

### V3 outcome

V3 ran from 13:17:05 to 14:14:51 Asia/Shanghai (57.78 minutes), produced 21 canonical
epoch records and 12,972 optimizer updates, and ended through `TRAINING_MONITOR_COMPLETE` with
no watchdog alert. Each arm retained four model-only checkpoint slots and no optimizer,
scheduler, scaler, RNG, DataLoader, or replay state.

| Arm | Early-stop epoch | Best loss (epoch) | Exact at best loss | Best greedy exact (epoch) |
|---|---:|---:|---:|---:|
| `legacy_default` | 12 | 0.307583 (6) | 78.45% | **79.62% (12)** |
| `legacy_budget_matched` | 14 | 0.317917 (8) | 78.34% | **78.77% (14)** |
| `semantic` | 21 | 0.422923 (15) | 72.07% | **73.42% (21)** |

The semantic arm learned much more slowly, improving from 49.28% exact after epoch 1 to 73.42%,
but it did not beat either legacy control. On this frozen offline imitation contract, the strong
claim that the current semantic representation alone improves BC is not supported. Parameter
count also does not explain the outcome: the parameter-matched legacy arm remains 5.35 percentage
points ahead of semantic on best greedy exact. This result does not show that semantic features
are useless in official-engine play; it shows that the current encoder/routing/optimization
contract does not turn them into better exact-action imitation on this corpus.

### V3 lowest-loss official-engine evaluation

The user-selected deployment point is `legacy_default/best_validation_loss.pt`, epoch 6:

- validation loss `0.3075830024137147`, greedy exact action `78.4492%`;
- checkpoint SHA-256 `759518ffa90f10121aa3441052db50931341c26309ea6dab128370e2f1c39223`;
- self-contained package `evaluation/arena/candidates/0025_v3_legacy_default_best_loss`;
- exact James Cox deck SHA-256 `f50fa3a23cdf21be7cf7d3f558b8ff0b82e8d4e7ba8f61b7b4cacc1a0080c16a`.

The formal run used the unmodified official engine and Frozen pool
`0019_foundation_51_exact_decks_v4` (catalog SHA-256
`f6e4ccb18a55c0e28410349b1299f3e9d5a61a2ca5cc7828c1b805fadf4cfa66`). It ran 10 games
against each of 51 enabled exact-deck opponents, alternating five games as first player and five
as second player. Both policy sides used the shared `cuda:0` inference service; the runner used
eight isolated workers with one CPU thread each and deterministic seed contract `22022`.

| Official-engine result | Value |
|---|---:|
| Completed | 510 / 510 |
| Wins / losses / draws | 105 / 405 / 0 |
| Win rate | **20.59%** |
| Errors / unfinished | 0 / 0 |
| Wall time | 227.46 s |
| Throughput | 2.24 games/s |

Against the exact Frozen James Cox identity, this checkpoint scored 2/10. Its strongest sampled
matchups were `ionos_bellibolt_ex_kilowattrel_01` and `ns_zoroark_ex_001` at 9/10 each; ten
opponent identities were 0/10. The immutable report is
[`evaluation/V3_james_cox_raging_bolt_restart.html`](evaluation/V3_james_cox_raging_bolt_restart.html),
run `run-f5303161ddfa4838a7815635b78b7446`, report SHA-256
`d5a0034f9299aa2d5a376916020b7119dfeba6ec3a51702c85d4bb0dd20a8ee8`.

This closes the deployability question but exposes a large offline-to-gameplay gap: 78.45%
validation exact action does not by itself produce a strong Frozen policy. It does **not** show
that the semantic arm is weaker in official-engine play, because semantic and the best-greedy
legacy checkpoints have not yet been evaluated under this same contract. The next controlled
experiment is a matched checkpoint comparison; action-level replay analysis should then separate
feature omissions from compounding imitation and state-distribution errors.

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

Twenty-one tests pass:

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

## V4 Training Contract And Next Gate

`V4_canonical_semantic_foundation` trains one `canonical_semantic` arm from random initialization;
it does not load V2/V3 weights. AdamW uses learning rate `3e-4`, weight decay `0.02`, bfloat16
autocast, gradient clipping at 1.0, seed `20260802`, batch size 384 and validation batch size 512.
Each epoch performs exactly one shuffled train pass and one complete fixed validation pass.
Patience is 6 epochs at minimum loss improvement 0.0005. Formal metrics are written in canonical
JSONL order, then TensorBoard, then W&B online under private project
`dragon_bra/pokemon-tcg-policy-learning`.

Only four replaceable model-only checkpoint slots are retained: latest, best validation loss,
best teacher exact and best greedy exact. Checkpoints contain no optimizer, scheduler, scaler,
RNG, DataLoader position or replay state. Their metadata commits the dataset manifest, training
config, model contract, implementation source digest and the absence of an initialization
checkpoint.

GPU calibration on the production model measured batch 256 at 4.58 GiB peak allocated and about
150 decisions/s; batch 512 reached 8.79 GiB allocated / 9.76 GiB reserved and about 279
decisions/s. Batch 384 is selected to retain headroom above the desktop GPU baseline and for
longer-sequence batches.

V1 established the research framework; V2 was interrupted; V3 is completed historical evidence
with a 510-game official-engine report. The V4 offline result is not a strength claim. After
training, the next gate is a self-contained canonical online exporter followed by matched
official-engine Frozen evaluation. The unresolved deterministic effect resolver remains the next
feature-semantic extension; MoE or broader capacity is not justified before this canonical input
contract is measured.
5. Evaluate the semantic and best-greedy controls under this exact Frozen contract, then audit matched replay failures before making representation-level strength claims.
