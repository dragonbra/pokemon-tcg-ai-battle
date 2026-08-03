# 0031 Decisions

## 2026-08-04: V2 semantic completeness

- Freeze 0025 and 0028 as historical provenance; 0031 has no executable dependency on another numbered project.
- Preserve every official resolved Energy unit as a Pokemon-parented categorical token while retaining each physical Energy card as a separate attachment child. Never invent a per-card partition of the flattened unit multiset.
- Bind options to exact serial/ordinal facts, retain official event participants, exact applicability, ordered Target Areas/Conditions, deck-order knowledge, opponent candidate sets, and persistent revealed identities.
- Fail closed on prototype or batch bucket overflow. Never silently truncate.
- Do not expose Energy gaps/surplus/removal advice, attach counterfactuals, newly enabled attacks, final damage, KO answers, or source persona.
- Use schema `0031_rule_faithful_semantic_decision_v2`; current capacity is 56,868,802 parameters.

## 2026-08-04: batching and compile boundary

- Keep dynamic padding as the formal eager default.
- Provide optional fixed upper-bound padding for card/event/option/effect/skill/action families and order fixed-bucket training rows by joint signature.
- Extend effect buckets through 96 and 128 because chronological real data reaches 91; the proposed maximum 64 rejected valid decisions.
- Route teacher forcing through module `forward` and remove decoder data-dependent boolean indexing so a real `fullgraph=True`, eager-backend compile regression passes.
- Do not admit Inductor for formal training. Full prototype+policy compilation did not complete inside the diagnostic window and used 16 compile workers with multi-GB host memory; 2,048 real decisions also yielded 124 joint bucket signatures.

## 2026-08-04: exact epoch resume exception

- Retain four model-only inference/export slots and add one atomically replaced exact-resume slot.
- Resume state includes weights, optimizer, optional scheduler/GradScaler, completed-epoch trainer state, and Python/NumPy/Torch RNG.
- Resume only under identical dataset/config/model-contract/implementation commitments and matching canonical metrics epoch.
- The exact boundary is the end of a completed epoch. An interrupted partial epoch is rerun; batch-level continuation is not claimed.
- Resume files are finite-retention training assets and never enter candidate packages.

Verification evidence: 52 project tests and 19 experiment-lifecycle tests pass; the chronological 512-decision semantic/padding benchmark is under `.tmp/0031_feature_benchmark/run-20260804-v2/`. No formal dataset or training version was created.
