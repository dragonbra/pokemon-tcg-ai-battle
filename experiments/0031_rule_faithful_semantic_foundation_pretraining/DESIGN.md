# 0031 Rule-Faithful Semantic Foundation Pretraining

Status: **Formal training is active in `V4_lr5e4_no_early_stop_b512`. `torch.compile` is not admitted for formal training.**

## Dataset boundary

The audited source interval is now 2026-07-10 through 2026-08-02. The immutable winner catalog contains 111,697 unique positive-terminal winner Episodes: 100,559 train and 11,138 validation, covering 423 exact decks and 596 provenance sources. Its catalog SHA-256 is `f34f48392929cdc6adb351296c72511c1448c96859f649ae6e5eba8d4475b86d`.

The immutable canonical dataset is `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/dataset/V1_rule_faithful_winners_20260710_20260802`. It contains **9,385,376** decisions: 8,448,060 train and 937,316 validation. Materialization used eight CPU workers and gzip level 3, completed in 4,729.74 seconds (78 minutes 49.74 seconds), and produced 9,756,163,489 payload bytes. Its manifest SHA-256 is `6a7ceab6938a456d7d0ae1e45c38086af54fb01af563a3feb69223f2bde89a7b`.

All 2,292 shards were independently hash-checked and then decompressed row by row. The second pass verified all row counts, schema and split contracts, global maxima, and fixed-bucket collation of the seven boundary representatives. Source/team identity remains audit-only and is absent from actor forward.

## Evidence boundary

Official general rules follow `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`. Concrete card and event semantics come from the current unmodified official engine runtime and `official_full_engine_prototypes_v2.json`. Feature selection, network capacity, bucketing, and BC optimization are project choices, not official rules.

Team Rocket's Energy (card 15) is the canonical regression. The engine prototype says `Psychic | Darkness = 80`, `energy_count = 2`, `only_team_rocket = true`, and skill 8. The actor receives the physical Energy card ID, serial, owner, attachment edge, static engine mask/count/restriction/skill facts, and one independent categorical child token for every resolved unit in the attached Pokemon's official `energies` multiset. The public observation does not identify which physical card produced each flattened unit, so no false per-card mapping is asserted. It never replaces this card with Colorless or zero Energy.

## V2 actor schema

Schema: `0031_rule_faithful_semantic_decision_v2`.

| Family | categorical width | numeric width | numeric field-state width | principal relations |
|---|---:|---:|---:|---|
| global | 12 | 24 | 24 | - |
| card / resolved Energy unit | 9 | 7 | 7 | `card_parent` |
| own resource ledger | 4 | 15 | 15 | - |
| bounded event | 31 | 4 | 4 | source, target, before, after |
| legal option | 19 | 2 | 2 | source, target, context, effect card, skill, effect |

These are separate named tensors, not one flat first-layer tensor. Categorical columns retain disjoint parameter rows but use one packed lookup per family. Numeric scalars retain independent two-layer projection parameters and independent state embeddings but execute as batched tensor contractions. These are exact operator fusions, not shared field semantics. Each numeric family has an explicit applicability tensor; padding is masked and does not create actor facts.

V2 preserves:

- physical Pokemon, attached Energy, Tool, pre-evolution, visible/remembered cards, stadium, context card, and effect card as distinct instance tokens;
- every resolved Energy unit with multiplicity, parented to its Pokemon, independently of physical attached Energy-card children;
- exact option ordinal and serial-bound skill identity, without fabricating Active source/target edges for Attack options that do not name an instance;
- official Switch, Change, and MoveAttached participant identities and before/after relations;
- exact enum/boolean applicability, `Effect.failSkip` magnitude, ordered Target Areas and Target Conditions;
- full self-deck order while exact, invalidated by shuffle or unknown insertion, plus conservative opponent-hand candidate sets and persistent revealed-card memory.

Unknown, not-applicable, false, and present zero remain distinct. Discrete identities and enums use embeddings or typed relations; exact counts and magnitudes use numeric projections.

## Static prototypes

The v2 asset contains 1,267 cards, 433 skills, 1,556 attacks, and 3,067 flattened ordered effects. Card, attack, skill, trigger, effect, target, and condition coverage is audited by `contracts/prototype_field_inventory.json`. Prototype collection limits are validated before encoding; overflow raises instead of truncating.

Attack inputs include ordered required Energy symbols and base damage. Effect constants, coefficients, comparators, targets, ordered areas, and conditions are facts. The actor must learn matching, arithmetic, damage, enablement, and interaction rules.

## Explicit exclusions

The actor does not receive typed/total Energy deficit, surplus Energy, preferred removal, attach/remove counterfactuals, `newly_enabled`, post-damage HP, KO labels, source/team/persona identity, future replay information, or target leakage. This keeps the feature layer a truthful observation/prototype representation rather than a partial rules engine.

## Model and cost

Default configuration is `d_model=320`, 8 heads, a four-layer `global + cards` board encoder, a one-layer event encoder, a tokenwise normalized resource memory, a three-family summary MLP, two option cross-attention layers, and an ordered legal-option pointer decoder with STOP. Current capacity is **56,352,322 parameters**. The 0028 model had 21,837,082 parameters, so optimized V2 is 2.58x its capacity, primarily because the lossless engine prototype encoder and typed dynamic fields retain substantially more independent categorical semantics.

The state memory still contains exactly `1 + C + R + E` tokens. Card attachment/parent relations enter the board encoder; event participant edges retrieve the encoded card instances; every resource and event instance remains directly available to option cross-attention. Only the expensive interaction topology changed: four deep layers no longer apply quadratic self-attention to the full combined sequence. Resource/event/global family summaries initialize the ordered decoder, while option-specific retrieval performs the required fine-grained cross-family interaction.

On 512 chronological real 0025 raw decisions, 0031 V2 averages 55.85 card-family tokens versus 45.11 in 0025/0028-compatible encoding (+23.8%). Of these, 5.87 on average are resolved Energy-unit tokens. Event, resource, and option counts are unchanged on the aligned sample; skill/effect reference means are 4.84/20.86 versus 5.17/21.90. Seven warm CPU repetitions measured median feature compilation at 0.573 ms/decision versus 0.301 ms for 0025 (1.90x), still below 1 ms/decision.

A new 2,098-decision real V2 sample was compiled from 32 audited Episodes. While 0030 RL concurrently occupied the RTX 5080, a deliberately coarse production-width batch-32 A/B measured the accepted hierarchical state path plus fused AdamW at 150.6 decisions/s versus 128.6 for the same packed-field joint-state reference, about **1.17x**. On the accepted state path, fused AdamW measured 150.6 versus 134.7 for standard AdamW, about **1.12x**. These short contaminated runs establish direction, not isolated formal throughput. The rejected first hierarchy (`board + resource Transformer + event Transformer + fusion Transformer`) was slower at small batch because extra launches outweighed its attention savings.

Canonical materialization performs feature compilation and canonical JSON serialization in the same Episode worker. The parent receives only final serialized rows plus compact audit counters, avoiding round trips of full raw observations and compiled dictionaries. Decompressed payload hashes match the former serializer byte for byte on the 2,098-row sample. Shards use recorded gzip level 3 to trade modest additional disk for faster one-time dataset construction.

## Variable lengths and padding

`C/R/E/O/S/F` vary because a decision can contain different numbers of card instances, known resources, recent events, legal options, related skills, and effects. The semantic records stay variable-length. Collation creates dense `[batch, length, width]` tensors and boolean masks; only the batch tensor shape is padded.

Dynamic eager collation remains the formal default. Optional fixed upper-bound buckets are:

```text
card:   16, 32, 64, 96, 128, 160
event:  8, 16, 32, 64
option: 4, 8, 16, 32, 64, 96
effect: 8, 16, 32, 64, 96, 128, 192, 256, 384
skill:  8, 16, 32, 64, 96
action steps including STOP: 2, 4, 8, 16, 32, 64
```

The complete dataset maxima are card 134, resource 30, event 64, option 78, effect 317, skill 82, and action including STOP 32. The earlier sample-derived maxima were insufficient, so the finite defaults now cover these audited values and still fail closed above their final bounds. The collator never truncates. When fixed padding is selected, training records are ordered by the exact joint bucket signature to improve shape reuse.

For the 512-row sample at batch 32, fixed buckets increased padded cells versus dynamic padding by 17.8% for cards, 54.2% for options, 19.4% for effects, 49.3% for skills, and 1.5% for events. An RTX 5080 d64/layer-2 diagnostic measured eager forward+backward+AdamW at 243.8 decisions/s dynamic and 245.1 fixed; the 0.5% difference is noise, not evidence of a speedup.

## `torch.compile` status

The teacher-forced path now enters `forward(..., teacher_forcing=True)`, and the decoder availability update uses fixed-shape masks instead of data-dependent `nonzero`; `backend="eager", fullgraph=True` passes a real compiled-row regression. Model-only checkpoint saving also unwraps compiled modules, so eager inference can load ordinary keys without `_orig_mod.` prefixes.

Inductor is **not** ready for formal use. Even d32/batch-1 full-graph compilation expands the trainable full-prototype computation into a very large graph and starts 16 compile workers with multi-GB host-memory use without completing inside the diagnostic window. Independent bucket dimensions also produced 124 joint signatures across 2,048 decisions, so fixed buckets alone do not guarantee a small practical graph cache. Formal `run_bc` therefore has no compile switch. Future work must benchmark a bounded prototype/core compilation boundary or a validated dynamic-shape strategy before admission.

## Training and checkpoint contract

BC performs exactly one optimization pass over train and a complete teacher-forced plus greedy validation pass each epoch. CUDA training uses fused AdamW and prefetch depth 4. Formal runs use private W&B project `dragon_bra/pokemon-tcg-policy-learning`; source provenance remains in manifests/audits and never reaches actor forward.

0031 has an explicit exact-resume exception to the repository's model-only default:

- four finite model-only slots (`latest` plus three best criteria) remain the inference/export artifacts;
- one atomically replaced `checkpoint/resume/latest_resume.pt` stores all arm weights, AdamW state, optional scheduler/GradScaler state, completed epoch, global update, early-stop/best/history state, Python/NumPy/Torch CPU/CUDA RNG, and compatibility commitments;
- resume is permitted only for the same version and exact dataset, training config, model contract, and implementation hashes; canonical metrics and resume epoch must agree;
- the exact boundary is a completed epoch. A crash inside an epoch replays that incomplete epoch; batch-level sampler/prefetch state is not claimed;
- current bfloat16 autocast uses no GradScaler and current trainer has no scheduler, but the checkpoint format supports both.

With 56.35M fp32 parameters, one model-only file is roughly 215 MiB and a populated AdamW resume file is roughly 645 MiB before serialization overhead. Four model-only slots plus one resume slot therefore require roughly 1.5-1.8 GiB at peak. Retention is finite and resume state is never exported to a candidate.

## Readiness and verification

`V1_rule_faithful_foundation` was stopped before completing epoch 1 while investigating low startup throughput; it has no checkpoint and remains an interrupted audit record. Controlled full-data tests rejected multi-process shard loading: serial loading at batch 256 reached 456.0 decisions/s with only 5.5% data wait, whereas four-process whole-shard IPC reached 295.0 decisions/s. The experimental loader was fully reverted.

`V2_no_early_stop_b384` was stopped before epoch 1 completed so the user-requested learning rate could change from 0.0003 to 0.0005. `V3_lr5e4_no_early_stop_b384` was then stopped after 20 batches so batch 512 could receive a same-contract full-data benchmark. Neither version has a checkpoint or completed epoch.

The active formal version is `V4_lr5e4_no_early_stop_b512`: learning rate 0.0005, batch and validation batch 512, bfloat16, fused AdamW, dynamic length bucketing, prefetch depth 4, and `early_stopping_patience=0`. A 100-batch full-data preflight measured **629.8 decisions/s**, 7.5% data wait, and 8.11/9.89 GB peak CUDA allocated/reserved memory, 11.2% faster than batch 384. The configured epoch limit is intentionally nonbinding at 1,000,000; training stops only by user decision or an operational fault. W&B run ID is `0031_rule_faithful_semantic_foundation_pretraining--v-cbffa699bd`.

The implementation passes 58 project semantic, model, fixed-bucket, compile-fullgraph, checkpoint, and training tests. The chronological benchmark artifact is `.tmp/0031_feature_benchmark/run-20260804-v2/feature_benchmark.json`; full-data batch preflights are `.tmp/0031_training_optimization/full_serial_b256.json`, `full_serial_b384.json`, and `full_serial_b512.json`. Full dataset verification is `.tmp/0031_training_optimization/full_dataset_verification.json`; the completed dataset CUDA/fixed-bucket smoke is `.tmp/0031_rule_faithful_bc_smoke/run-c81d419894`.
