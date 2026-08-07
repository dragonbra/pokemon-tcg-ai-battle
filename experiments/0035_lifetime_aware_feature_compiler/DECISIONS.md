# 0035 Decisions

## 2026-08-08: create a semantics-preserving runtime fork

- Allocate 0035 instead of continuing to mutate 0031. The new project is a physical, self-contained runtime fork and has no executable dependency on 0031.
- Preserve schema `0031_rule_faithful_semantic_decision_v2`, all 39 actor tensors, model/checkpoint structure, logits, legal-action decoder, and action contract. This project performs no training and introduces no actor-visible field.
- Keep the official Engine, `GetBattleData` JSON representation, and ctypes action ABI unchanged. Lifetime ownership and dirty tracking are compiler implementation strategies, not official game rules or new engine facts.
- Reject whole-layer dependency freezing. On the real 35-decision trajectory every coarse layer missed, and compile time regressed from 0.593 ms to 1.237 ms (2.09x slower).
- Adopt entity/segment caching: card fragments keyed by serial, append-only event rows, resource fragments keyed by card ID with age separated, and option prototype expansions keyed by normalized semantics. Real-trajectory audits found 85.2% reusable complete card rows, 76.3% reusable old event rows, 73.4% reusable resource rows when age is separated, and 85.2% option semantic-key reuse.
- Treat turn only as an epoch/invalidation boundary. `supporterPlayed`, `stadiumPlayed`, `energyAttached`, `retreated`, status, `appearThisTurn`, selection, and board state remain action-mutable.
- Retain a stateless full compiler as the semantic authority. Unknown structure, serial ambiguity, chronology mismatch, actor/deck mismatch, or relation inconsistency falls back to full rebuild and reseeds the cache.
- Do not enable the optimized path by default until exact record/tensor/logit/action parity and paired microbenchmark plus N16E1/N16E8 official-engine gates pass.

## 2026-08-08: keep V1 opt-in after paired admission benchmark

- Preserve the card/event fragment implementation and normalized option-semantic cache as a semantics-safe research path. Exact canonical records and all 39 collated tensors match on the 35-decision golden trajectory and 10 additional complete trajectories totaling 525 decisions; the full 0035 suite passes 83/83.
- Reject the resource fragment from the connected V1 path. Its component median improved from 39.748 to 30.486 microseconds, but p95 regressed from 78.294 to 106.548 microseconds.
- Reject whole EventLayer reuse. The audited chronology produced zero segment hits because either the event window or card-relation layout changed every decision, so its key construction was pure overhead.
- Record the final seven-by-5,000 paired result: compiler median 0.574 to 0.430 ms (-25.1%), total feature median 1.166 to 1.069 ms (-8.3%), decisions/s 813.9 to 826.7 (+1.6%), while compiler p95 regressed 0.820 to 0.934 ms (+14.0%).
- Mark V1 performance admission as failed and retain `incremental=False` in the online runtime/export path. Do not spend the next iteration adding more per-entity recursive signatures; the next useful boundary is observation/zone structural sharing that avoids whole-card-layer traversal and assembly.

## 2026-08-08: V2 version-boundary experiment and type-exact commitment

- Implement and test an `ObservationVersionTracker` prototype after JSON decoding. Whole CardLayer reuse occurred on only 10/525 audited decisions. Zone-level reuse reported many hits, but profiling showed they were dominated by empty/small zones; dirty zones still performed entity traversal and a second assembly pass. The connected version/zone path measured compiler median/p95 `0.549/1.240 ms` against full `0.588/0.952 ms`, so it was rejected and removed from online/benchmark execution.
- Reject a second `serial -> raw entity` dirty scan for the same reason: it duplicated the card traversal already performed by the compiler and measured `0.479/1.022 ms`, still worse than the simpler V1 incremental path.
- Replace Python-recursive card projections with the standard-library protocol-5 binary commitment. It preserves primitive types, includes the complete raw card tree conservatively, requires no additional dependency, and fails closed for unsupported mappings.
- Final V2 7x10,000 paired result: compiler median `0.591 -> 0.437 ms` (-26.0%), compiler p95 `0.950 -> 0.910 ms` (-4.3%), total feature median `1.219 -> 1.069 ms` (-12.3%), throughput `734.2 -> 769.9 decisions/s` (+4.9%). This fixes V1's tail regression but does not satisfy the absolute/relative admission gates.
- Keep `incremental=False`. Do not run N16E1/N16E8 admission for a compiler that still fails its prerequisite microbenchmark. A future step-change requires native producer dirty versions or direct tensor-segment persistence, not another Python observation scan.

## 2026-08-08: V3 persist canonical tensors at session lifetime

- Add a session-owned `PersistentTensorBank` after canonical compilation. It retains capacity-managed storage for all 39 actor tensors, reuses unchanged storage without allocation or Python-to-tensor conversion, patches changed ragged rows, and resets all buffers at session end. The official Engine and its JSON observation ABI remain untouched.
- Preserve the ordinary collator as the independent semantic authority. Exact dtype, shape and value parity passes over 35 + 525 chronological decisions. Explicitly clear the one-row zero placeholder when a ragged family transitions from nonempty to empty, preventing stale values from a prior decision.
- Record the 7x10,000 full-vs-persistent result: total median `1.178 -> 0.993 ms` (-15.7%), total p95 `1.831 -> 1.672 ms` (-8.7%), and throughput `792.0 -> 900.4 decisions/s` (+13.7%). Record the compiler-held-constant tensor-only result: collate median `0.457 -> 0.420 ms` (-8.1%), collate p95 `1.033 -> 0.743 ms` (-28.0%), and total median `1.076 -> 1.015 ms` (-5.7%).
- Keep production and evaluation defaults unchanged because official-engine N16E1/N16E8 admission has not yet been run. Expose the optimized research path explicitly as `OnlineCausalEncoder(..., incremental=True, persistent_tensors=True)` and copy it into exported candidates.
- The next optimization is a compiler-to-bank dirty-range contract. V3 still compares canonical rows in Python to discover changed tensor regions; eliminating that second scan should create the remaining direct benefit without modifying the Engine.

## 2026-08-08: V4 cache model-static prototype embeddings on GPU

- Cache the derived card/attack/skill/effect `PrototypeEmbeddings` once per frozen model device/dtype/weight state. The default FP16 cache is 5,573,120 bytes and costs about 2.12 ms to build once; it is neither a parameter nor checkpoint state.
- Invalidate on `_apply` device/dtype mutation and `load_state_dict`. Train mode bypasses the cache whenever any prototype parameter requires gradients. When all prototype parameters are frozen, reuse the detached cache while preserving gradients into downstream Transformer/LoRA/decoder parameters.
- Record exact cached/uncached deterministic action commitments and the 7x100 batch-64 CUDA result: forward median `13.111 -> 10.801 ms` (-17.6%), p95 `33.660 -> 29.380 ms` (-12.7%), modeled throughput `4,881 -> 5,925 decisions/s` (+21.4%).
- Reject the first official comparison as a formal ablation because its uncached arm used the older 0031 package and therefore changed the compiler together with the cache. Re-run with two packages exported from the same 0035 source/checkpoint, changing only `prototype_memory()`.
- In the strict three-run-per-arm 128-game N16E16 ablation, all 768 games finished with zero errors. Median GPU model time fell `20.572 -> 17.411 ms/batch` (-15.4%); wall fell `67.068 -> 63.279 s` (-5.6%); selections/s rose `335.35 -> 355.47` (+6.0%). Admit automatic prototype caching for semantically frozen inference/prototype parameters.
