# 0031 Rule-Faithful Semantic Foundation Pretraining

Status: **V2 semantic implementation is training-ready; no formal 0031 dataset or run has been created. `torch.compile` is not admitted for formal training.**

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

These are separate named tensors, not one flat first-layer tensor. Each categorical column has its own embedding; each numeric family has an explicit applicability tensor and numeric projection. Padding is masked and does not create actor facts.

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

Default configuration is `d_model=320`, 8 heads, four state layers, three option cross-attention layers, and an ordered legal-option pointer decoder with STOP. Current capacity is **56,868,802 parameters**. The 0028 model had 21,837,082 parameters, so V2 adds 35,031,720 parameters (2.60x total), primarily because the lossless engine prototype encoder and typed dynamic fields contain substantially more independent categorical semantics.

On 512 chronological real 0025 raw decisions, 0031 V2 averages 55.85 card-family tokens versus 45.11 in 0025/0028-compatible encoding (+23.8%). Of these, 5.87 on average are resolved Energy-unit tokens. Event, resource, and option counts are unchanged on the aligned sample; skill/effect reference means are 4.84/20.86 versus 5.17/21.90. Seven warm CPU repetitions measured median feature compilation at 0.573 ms/decision versus 0.301 ms for 0025 (1.90x), still below 1 ms/decision. The larger model and longer card memory are expected to reduce formal training throughput, but no production-size apples-to-apples 0031 epoch has run, so a precise slowdown is not claimed.

## Variable lengths and padding

`C/R/E/O/S/F` vary because a decision can contain different numbers of card instances, known resources, recent events, legal options, related skills, and effects. The semantic records stay variable-length. Collation creates dense `[batch, length, width]` tensors and boolean masks; only the batch tensor shape is padded.

Dynamic eager collation remains the formal default. Optional fixed upper-bound buckets are:

```text
card:   16, 32, 64, 96, 128
event:  8, 16, 32, 64
option: 4, 8, 16, 32, 64
effect: 8, 16, 32, 64, 96, 128
skill:  8, 16, 32, 64
action steps including STOP: 2, 4, 8, 16, 32, 64
```

The originally proposed effect maximum 64 is invalid: the real sample reaches 91. V2 therefore includes 96/128 buckets and fails closed above 128. The collator never truncates. When fixed padding is selected, training records are ordered by the exact joint bucket signature to improve shape reuse.

For the 512-row sample at batch 32, fixed buckets increased padded cells versus dynamic padding by 17.8% for cards, 54.2% for options, 19.4% for effects, 49.3% for skills, and 1.5% for events. An RTX 5080 d64/layer-2 diagnostic measured eager forward+backward+AdamW at 243.8 decisions/s dynamic and 245.1 fixed; the 0.5% difference is noise, not evidence of a speedup.

## `torch.compile` status

The teacher-forced path now enters `forward(..., teacher_forcing=True)`, and the decoder availability update uses fixed-shape masks instead of data-dependent `nonzero`; `backend="eager", fullgraph=True` passes a real compiled-row regression. Model-only checkpoint saving also unwraps compiled modules, so eager inference can load ordinary keys without `_orig_mod.` prefixes.

Inductor is **not** ready for formal use. Even d32/batch-1 full-graph compilation expands the trainable full-prototype computation into a very large graph and starts 16 compile workers with multi-GB host-memory use without completing inside the diagnostic window. Independent bucket dimensions also produced 124 joint signatures across 2,048 decisions, so fixed buckets alone do not guarantee a small practical graph cache. Formal `run_bc` therefore has no compile switch. Future work must benchmark a bounded prototype/core compilation boundary or a validated dynamic-shape strategy before admission.

## Training and checkpoint contract

BC performs exactly one optimization pass over train and a complete teacher-forced plus greedy validation pass each epoch. Formal runs use private W&B project `dragon_bra/pokemon-tcg-policy-learning`; source provenance remains in manifests/audits and never reaches actor forward.

0031 has an explicit exact-resume exception to the repository's model-only default:

- four finite model-only slots (`latest` plus three best criteria) remain the inference/export artifacts;
- one atomically replaced `checkpoint/resume/latest_resume.pt` stores all arm weights, AdamW state, optional scheduler/GradScaler state, completed epoch, global update, early-stop/best/history state, Python/NumPy/Torch CPU/CUDA RNG, and compatibility commitments;
- resume is permitted only for the same version and exact dataset, training config, model contract, and implementation hashes; canonical metrics and resume epoch must agree;
- the exact boundary is a completed epoch. A crash inside an epoch replays that incomplete epoch; batch-level sampler/prefetch state is not claimed;
- current bfloat16 autocast uses no GradScaler and current trainer has no scheduler, but the checkpoint format supports both.

With 56.9M fp32 parameters, one model-only file is roughly 217 MiB and a populated AdamW resume file is roughly 650 MiB before serialization overhead. Four model-only slots plus one resume slot therefore require roughly 1.5-1.8 GiB at peak. Retention is finite and resume state is never exported to a candidate.

## Readiness and verification

No formal 0031 version exists. Formal start must materialize a new immutable V2 dataset and allocate `V1_rule_faithful_foundation`; 0025/0028 shards cannot be reused because the actor schema changed.

The implementation passes the project semantic, model, fixed-bucket, compile-fullgraph, checkpoint, and training tests. The chronological benchmark artifact is `.tmp/0031_feature_benchmark/run-20260804-v2/feature_benchmark.json`; the eager GPU padding diagnostic is beside it as `gpu_eager_padding_benchmark.json`. The earlier noncanonical smoke remains historical V1 evidence and does not validate the new V2 dataset or formal run.
