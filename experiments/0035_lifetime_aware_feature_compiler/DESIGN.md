# 0035 Lifetime-Aware Feature Compiler

Status: **V5 worker-local stateless compilation admitted for evaluation throughput; equal-contract official-engine N16E16 throughput improves 54.5% over central stateless compilation, while the compiler and actor semantics remain unchanged.**

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
  -> reference 39-key collator, or session-owned PersistentTensorBank
       stable CPU tensor storage
       unchanged slots reused without allocation/conversion
       changed ragged rows patched in place
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

V3 moves the reuse boundary through canonical collation. Each battle/session may own a `PersistentTensorBank` containing capacity-managed CPU tensors for all 39 keys. A slot that is unchanged returns the same underlying storage without calling `torch.tensor`; a changed ragged slot keeps its allocation and updates only its logical rows, while empty fields explicitly zero the canonical one-row padding placeholder. Masks and logical views change only when their lengths change. The returned views intentionally live only until the next call on that session, matching synchronous online inference; historical callers must clone them. Session reset discards every buffer. The reference collator remains the independent authority and the default path.

This is tensor persistence, not merely Python-record memoization. It still has a Python dirty-detection cost because V3 receives an ordinary canonical record and must compare its rows. Passing compiler-owned dirty ranges directly to the bank is the next optimization boundary; it can remove that second scan without changing the Engine ABI.

## Model-static prototype GPU cache

V4 closes a larger boundary after model loading. The default policy previously called `OfficialPrototypeEncoder.encode_all()` in every forward, re-encoding all configured card, attack, skill and effect identities even though frozen inference weights never change. `SemanticPolicy` now owns a derived, non-checkpoint `PrototypeEmbeddings` cache. Eval mode lazily builds it once on the model's current device/dtype; subsequent state and option encoding reuse the same GPU-resident tensors.

The default 2,049 card + 2,049 attack + 513 skill + 4,097 effect rows at width 320 occupy 5,573,120 bytes in FP16. Initial construction costs about 2.12 ms on the RTX 5080 and is excluded from steady-state forward timing because it occurs once per loaded model. `_apply` device/dtype mutations and `load_state_dict` invalidate the derived cache. The cache is a plain runtime attribute: it is not an `nn.Parameter`, persistent buffer, or checkpoint key.

Training semantics are explicit. If any prototype-encoder parameter requires gradients, train mode bypasses the cache and executes an autograd-connected `encode_all()` on every forward. If all prototype parameters are frozen, train mode may reuse the ordinary detached cache while downstream Transformer, LoRA and decoder parameters continue receiving gradients. This matches the intended RL boundary without silently freezing a trainable prototype path.

## Worker-local stateless compiler topology

V5 changes evaluation scheduling, not feature semantics. With `worker_local_compiler=true` and `worker_compiler_backend=policy_stateless`, each OS process that owns official-engine battles also owns one ordinary causal encoder per battle side. It consumes the same decoded observation and invokes the unchanged stateless canonical compiler locally. Only the canonical record and minimal `current/select` control fields cross the resident-inference socket; PyTorch import, collation, H2D, the model and decoding remain centralized in the GPU server.

This removes the single central Python compiler stage between parallel Engine workers and batched GPU inference. Session IDs, actor/deck identity and chronology remain isolated, compiler failures fail closed, and the server never silently falls back to the old serial topology. The incremental backend and session/event tensor experiments are not selected: the admitted V5 contract is explicitly `policy_stateless`, so disabling every optional incremental/tensor-cache flag yields the same canonical records, 39 tensors, logits and actions as the original stateless path.

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

The final V3 full-vs-persistent paired/interleaved run used seven repetitions of 10,000 chronological decisions after 1,000 warmup decisions. Full rebuild plus ordinary collation measured `0.579 ms` compiler, `0.448 ms` collate, `1.178 ms` total and `792.0 decisions/s`; incremental compilation plus the persistent tensor bank measured `0.422 ms`, `0.411 ms`, `0.993 ms` and `900.4 decisions/s`. Total median fell 15.7%, total p95 fell from `1.831` to `1.672 ms` (-8.7%), and throughput rose 13.7%. The artifact is `.tmp/0035_lifetime_aware_feature_compiler/persistent_tensor_compare_7x10k.json`.

A second paired run isolates the tensor boundary by holding the incremental compiler constant. Ordinary collation versus persistent tensors measured collate median `0.457 -> 0.420 ms` (-8.1%), collate p95 `1.033 -> 0.743 ms` (-28.0%), total median `1.076 -> 1.015 ms` (-5.7%), and total p95 `2.152 -> 1.738 ms` (-19.2%). The artifact is `.tmp/0035_lifetime_aware_feature_compiler/tensor_only_compare_7x10k.json`. These results show direct tensor persistence contributes independently; it does not account for the compiler-fragment gain.

The V4 CUDA component benchmark used the real 56,352,322-parameter checkpoint, FP16, batch 64, seven alternating repetitions and 100 measured forwards after 20 warmups. Uncached versus cached forward median was `13.111 -> 10.801 ms` (-17.6%), p95 `33.660 -> 29.380 ms` (-12.7%), and modeled decisions/s `4,881 -> 5,925` (+21.4%). Cached and uncached deterministic action tensor commitments were identical. The artifact is `.tmp/0035_lifetime_aware_feature_compiler/prototype_cache_cuda_7x100_b64.json`.

The strict official-engine ablation used two packages exported from the same 0035 source/checkpoint; the uncached package changes only `prototype_memory()` to recompute every forward. Each arm ran three 128-game N16E16 profiles with batch 64, 2 ms wait and FP16. All 768 games completed with zero errors. Across-run medians were: GPU model `20.572 -> 17.411 ms/batch` (-15.4%), model wall `22.383 -> 18.961 ms/batch` (-15.3%), total wall `67.068 -> 63.279 s` (-5.6%), and official-engine selections/s `335.35 -> 355.47` (+6.0%). The six artifacts are `.tmp/0035_lifetime_aware_feature_compiler/prototype_ablation_uncached_n16e16_r{1,2,3}.json` and `prototype_cache_official_cached_n16e16{,_r2,_r3}.json`.

The V5 adjacent official-engine comparison used the same Policy-0806 checkpoint/deck, N16E16, batch 64, 2 ms wait and FP16. Central stateless compilation completed 128/128 games with zero errors at `342.22 selections/s` and `65.72 s` wall; worker-local stateless compilation completed 128/128 with zero errors at `528.77 selections/s` and `42.52 s` wall. This is `+54.5%` throughput and `-35.3%` wall time. Server `prepare_records` fell from `65.51` to `0.20 ms/batch`, directly confirming removal of the central compiler funnel. A worker-local incremental arm reached `525.77 selections/s`, 0.57% below stateless, so V5 deliberately selects stateless. The artifacts are `.tmp/0035_lifetime_aware_feature_compiler/central_stateless_n16e16_b64_r2.json`, `worker_local_stateless_n16e16_b64_r1.json`, and `worker_local_incremental_n16e16_b64_r1.json`.

Admission requires exact Python-record and all-39-tensor parity, zero shadow/logit/action mismatches, compiler median no more than 0.25 ms and at least 50% below paired full rebuild, no p95 feature regression, 128/128 official-engine completion with zero errors, at least 10% N16E8 throughput gain, and no more than 3% N16E1 regression. Until those gates pass, deployment defaults to the stateless path.

## Current and next stage

V5 retains the self-contained lineage and unchanged actor/model/action contracts. Prototype caching remains automatic whenever its weights are semantically frozen. Evaluation throughput now uses worker-local stateless compilation when explicitly requested; ordinary evaluation defaults remain unchanged, and no incremental or resident tensor cache is required. The full evaluation test suite passes 295/295 tests.

The next scheduling stage can address central collation/IPC handoff without changing compiler semantics. Deck-static GPU resource bases and compiler-owned dirty ranges remain research directions rather than admitted runtime defaults. V5 creates no training checkpoint or W&B run.
