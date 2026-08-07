# 0035 Lifetime-Aware Feature Compiler

Status: **V2 semantic audit and paired benchmark complete; median and p95 both improve, but the optimized compiler remains opt-in because the absolute admission target is not met.**

## Scope and evidence boundary

0035 is a self-contained runtime fork of 0031. It changes how canonical features are materialized, not what the actor observes. Official general rules remain those summarized in `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`; concrete observation and log fields come from the current unmodified official Engine runtime; battle/turn/action cache ownership is a 0035 engineering strategy.

The Engine returns an in-memory JSON byte buffer through `GetBattleData`; the Python wrapper decodes it to a mapping before the compiler runs. There is no per-step observation file. 0035 does not change that ABI or JSON format and does not modify `engine/source/`.

## Lineage and compatibility

The source project is `0031_rule_faithful_semantic_foundation_pretraining` at commit `473f6e5bd83468c70fd977a841803aea19d854c5`, augmented by the exact source-file commitments in `manifest.json` because the compiler factoring work existed in the source worktree. All executable runtime files and both prototype assets are physically copied into 0035; no numbered project is imported at runtime.

Schema remains `0031_rule_faithful_semantic_decision_v2`. Existing 0031 model-only and portable checkpoints remain the source-weight contract. Model architecture, 56,352,322-parameter default, loss, reward, value interface, option logits, ordered pointer decoder, STOP behavior, and legal-action semantics are unchanged. No training is performed in V1.

## Unchanged actor fields and shapes

| Family | categorical width | numeric width | field-state width | relations |
|---|---:|---:|---:|---|
| global | 12 | 24 | 24 | none |
| card / resolved Energy | 9 | 7 | 7 | parent |
| own resource ledger | 4 | 15 | 15 | none |
| bounded event | 31 | 4 | 4 | source, target, before, after |
| legal option | 19 | 2 | 2 | source, target, context, effect card, skill, effect |

The batch contract still contains exactly 39 actor tensor/mask keys. Variable card, resource, event, option, skill, effect, and action axes retain their existing order and padding semantics. Unknown, inapplicable, false, present zero, and padding remain distinct.

## Runtime data flow

```text
official Engine JSON bytes
  -> Python observation mapping
  -> CausalKnowledge.consume (same facts and chronology)
  -> FeatureCompileSession
       battle constants
       turn epoch
       action/entity caches
       relation rebasing
       fail-closed full compiler
  -> unchanged canonical record
  -> unchanged 39-key collator
  -> unchanged SemanticPolicy and decoder
```

## Lifetime ownership

| Lifetime | Owned facts | Invalidation |
|---|---|---|
| process/model | schema field order, vocabularies, zones, prototype card/skill/attack/effect tables and static semantic expansions | prototype/schema/config change |
| battle/session | actor/opponent mapping, exact deck, sorted deck counts, resource identity order and initial counts, first player after first-observation verification | new session, actor/deck/schema/prototype mismatch |
| turn epoch | current turn key and optional turn-start instance snapshot | turn key change or chronology reset |
| action/entity | board zones, per-card dynamic state, causal resources, event append/age, selection/options, globals and all current relation endpoints | every affected observation transition |

Turn-scoped does not mean all once-per-turn flags are frozen. `turnActionCount`, supporter/stadium/energy/retreat flags, status, `appearThisTurn`, board state, and selection can change inside a turn and are action-owned.

## Fine-grained cache design

Cards use stable `(kind, serial)` entity keys and placement overlays. Unattached leaf rows have a guarded raw-equality fast path; Pokémon attachment trees use a complete consumed-field projection. Same-object mutation, primitive type changes, `cardId` aliases, child ownership and slot bounds retain stateless semantics. Absolute canonical indexes are not reused across layouts: parent and cross-family relations are rebased after the current layout is known. Duplicate positive serials fail closed to the stateless compiler.

V2 tested moving versions to a decoded-observation session boundary. Exact whole-layer versions were semantically valid, but they did not reduce work: only 10 of 525 decisions could reuse a complete CardLayer, and zone hits were dominated by empty/small zones while dirty zones still required entity traversal and a second assembly pass. The connected V2 path therefore does not retain that duplicate scan. The prototype and lifecycle tests remain as audit evidence in `features/observation_versions.py`; online inference still uses the simpler fragment path.

The admitted V2 implementation improvement replaces Python-recursive card projections with a protocol-5 binary commitment produced by the standard-library C implementation. It is type exact (`True`, `1`, and `1.0` remain distinct), covers the complete raw card tree conservatively, and therefore may cause harmless extra misses but cannot hide a consumed-field change. This reduces change-detection cost without adding a second observation traversal or changing the Engine representation.

Events use the existing bounded 64-row causal window. Immutable type/payload/state columns are encoded only when a strictly increasing source event is appended; age is updated from the current newest event. Event source/target/before/after relations are resolved against the current card index map and patched when a referenced serial moves.

An independently tested resource prototype preserves battle-static card-ID order and initial counts, with age separated from dynamic cores. It is not connected to V1: its median improved 23.3%, but p95 regressed 36.1%, so the admission rule rejected it.

Options remain action-owned, but prototype skill/effect expansion is cached by normalized source/context/effect/attack semantics. Option ordinal, numeric values, availability, bounds and card relations are always rebuilt against the current selection and layout. Globals are small and action-dynamic, so they are rebuilt directly.

## Dirty graph and safety

```text
zone/layout change -> affected card fragments -> card index map -> event/option relation rebase
card scalar change -> one dynamic card fragment
event append/evict -> new/removed immutable rows + age vector + new relations
resource revision -> affected card-ID rows
selection change -> context/effect synthetic cards + options + globals
turn change -> turn header/global metadata, not a blanket battle-cache clear
unknown or ambiguous transition -> full rebuild -> cache reseed
```

Logs are positive dirty hints, not sole correctness authority. Cheap primitive signatures enumerate only fields read by the compiler. Whole-observation JSON hashing, recursive freezing and whole-layer equality are excluded from the hot path. Duplicate/reused serials, non-monotonic decisions, actor/deck mismatch, unknown log structure, invalid selection bounds or invalid relation endpoints trigger a full rebuild.

The stateless compiler remains the explicit fallback authority. A periodic production shadow mode is a next-stage requirement; V1 does not claim that it is implemented.

## Baseline and admission

The golden real trajectory has one actor-local 35-decision sequence covering setup, ordinary main, nested selection, deck view, evolution, Ability, attack, switch/KO replacement and terminal-near observations. Its SHA-256 is `0db50f9991849143aba40053f2c5751f2063f76f4f636bb10e0ad0215b0c254e`. A second audited fixture contains 10 complete actor-local trajectories, 525 decisions, both seats and seven exact decks; its SHA-256 is `87aa39bb7dd8023db0bd05afa954416519bc4abf64552d4a4eff270288d96485`.

The original full-path baseline was 0.127 ms knowledge, 0.580 ms compiler, 0.454 ms collate and 1.190 ms total. The rejected whole-layer incremental path measured 1.237 ms compiler versus 0.593 ms full.

The final V1 paired/interleaved run used seven repetitions of 5,000 chronological decisions after 1,000 warmup decisions. Across-repetition medians were: compiler `0.574 -> 0.430 ms` (-25.1%), total feature pipeline `1.166 -> 1.069 ms` (-8.3%), and wall throughput `813.9 -> 826.7 decisions/s` (+1.6%). Compiler p95 regressed from `0.820` to `0.934 ms` (+14.0%). The artifact is `.tmp/0035_feature_compiler/paired_v2_option_7x5000.json` and is intentionally untracked diagnostic evidence.

The final V2 paired/interleaved run used seven repetitions of 10,000 chronological decisions after 1,000 warmup decisions. Across-repetition medians were: compiler `0.591 -> 0.437 ms` (-26.0%), compiler p95 `0.950 -> 0.910 ms` (-4.3%), total feature pipeline `1.219 -> 1.069 ms` (-12.3%), and wall throughput `734.2 -> 769.9 decisions/s` (+4.9%). Thus V2 fixes the V1 tail-regression failure, but still misses the absolute `<=0.25 ms`, relative `>=50%`, and end-to-end `>=10%` admission targets. The artifact is `.tmp/0035_feature_compiler/type_exact_commitment_v2_7x10000.json`.

Admission requires exact Python-record and all-39-tensor parity, zero shadow/logit/action mismatches, compiler median no more than 0.25 ms and at least 50% below paired full rebuild, no p95 feature regression, 128/128 official-engine completion with zero errors, at least 10% N16E8 throughput gain, and no more than 3% N16E1 regression. Until those gates pass, deployment defaults to the stateless path.

## Current and next stage

V2 retains the V1 self-contained lineage, 35-decision golden commitments, 525-decision multi-trajectory record/tensor parity, card/event fragment caching, option semantic caching and fail-closed fallback. The optimized compiler is deliberately not exported or enabled by default because it still misses the absolute median and end-to-end admission gates, even though its p95 now improves.

The next implementation stage must eliminate whole-card-layer traversal/materialization rather than add more recursive signatures. It should introduce observation-boundary structural sharing or zone-version tokens, then add periodic shadow parity and run checkpoint logit/action plus equal-contract N16E1/N16E8 official-engine profiles. V1 creates no training checkpoint or W&B run.
