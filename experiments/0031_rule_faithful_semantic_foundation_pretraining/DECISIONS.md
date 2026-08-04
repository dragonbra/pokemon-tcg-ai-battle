# 0031 Decisions

## 2026-08-04: training throughput architecture

- Pack categorical fields into disjoint-offset embedding tables and batch independent numeric MLP/state parameters. Forward facts and gradients match independent-field references; no semantic columns are shared or removed.
- Replace four layers over the full `1 + C + R + E` sequence with four board layers over `global + cards`, one event layer with encoded participant relations, tokenwise resource memory, and a three-family summary MLP. Options still cross-attend the complete `1 + C + R + E` memory.
- Reduce option cross-attention from three layers to two and use fused AdamW on CUDA with default prefetch depth 4.
- Reject the heavier first hierarchy containing separate resource and fusion Transformers: at small batches its extra launches erased the theoretical attention benefit.
- Keep `torch.compile` disabled. The optimization does not depend on compiled wrappers, and ordinary eager models/checkpoints remain the inference contract.
- On a 2,098-decision real V2 sample under concurrent 0030 RL GPU load, the accepted production-width batch-32 path measured 150.6 decisions/s versus 128.6 for the same packed-field joint reference. Fused AdamW measured 150.6 versus 134.7 for standard AdamW. Treat these as directional A/B evidence, not isolated throughput.
- The optimized default has 56,352,322 parameters; the joint-only summary attention is not instantiated in formal hierarchical models.
- For full materialization, serialize canonical rows inside compile workers and return only the final JSON plus compact split/episode/deck/length metadata. A 2,098-row regression produced byte-identical decompressed train and validation payload hashes relative to the former main-process serializer.
- Use gzip level 3 for canonical shards and record it in the manifest. This changes storage compression only, not decompressed records or training tensors.

## 2026-08-04: extend audited source through 2026-08-02

- Add and fully CRC-validate the official 2026-08-02 daily Episode archive; retain its immutable SHA-256 and source inventory.
- Merge the audited 0028 prefix with the 2026-08-02 delta under the 0031 split contract. The resulting catalog has 111,697 winner Episodes, split 100,559/11,138, with 423 exact decks and 596 provenance sources.
- Keep decision counts explicitly provisional until V2 materialization. The current 9.37 million estimate uses the 0028 observed mean decisions per Episode and is not a training manifest fact.

## 2026-08-04: V2 semantic completeness

- Freeze 0025 and 0028 as historical provenance; 0031 has no executable dependency on another numbered project.
- Preserve every official resolved Energy unit as a Pokemon-parented categorical token while retaining each physical Energy card as a separate attachment child. Never invent a per-card partition of the flattened unit multiset.
- Bind options to exact serial/ordinal facts, retain official event participants, exact applicability, ordered Target Areas/Conditions, deck-order knowledge, opponent candidate sets, and persistent revealed identities.
- Fail closed on prototype or batch bucket overflow. Never silently truncate.
- Do not expose Energy gaps/surplus/removal advice, attach counterfactuals, newly enabled attacks, final damage, KO answers, or source persona.
- Use schema `0031_rule_faithful_semantic_decision_v2`; the semantic-completeness implementation initially had 56,868,802 parameters before the later throughput architecture decision.

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

Verification evidence: 53 project tests and 19 experiment-lifecycle tests pass; the chronological 512-decision semantic/padding benchmark is under `.tmp/0031_feature_benchmark/run-20260804-v2/`. No formal dataset or training version was created.
