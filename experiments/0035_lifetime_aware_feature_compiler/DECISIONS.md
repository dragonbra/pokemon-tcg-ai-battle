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
