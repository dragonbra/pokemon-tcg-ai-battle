# 0031 Decisions

## 2026-08-04: formal training start

- Preserve `V1_rule_faithful_foundation` as interrupted before its first completed epoch and without a checkpoint. It was stopped to investigate low startup throughput; subsequent controlled evidence rejected the initial loader-bottleneck hypothesis.
- Reject whole-shard ProcessPool loading. On the complete dataset at batch 256, the unchanged serial path reached 456.0 decisions/s with 5.5% data wait; four spawn workers reached only 295.0 decisions/s because startup and large Python-object IPC dominated. The experiment was fully reverted.
- Start `V2_no_early_stop_b384` with batch and validation batch 384, BF16, fused AdamW, dynamic length bucketing, prefetch depth 4, and no `torch.compile`. A 100-batch full-data preflight reached 566.6 decisions/s with 6.3% data wait and 6.20/8.09 GB peak allocated/reserved CUDA memory.
- Disable early stopping with patience 0 and use a nonbinding 1,000,000-epoch limit. The user decides when to stop. Persist one atomic exact-resume checkpoint after every complete epoch; an incomplete epoch is deliberately replayed after interruption.
- Stop V2 before its first completed epoch at the user's request and raise the learning rate from 0.0003 to 0.0005 in a new version. Stop the resulting V3 after 20 batches to compare batch 512 before final launch; neither interrupted version produced a checkpoint.
- Admit batch 512 for the final active `V4_lr5e4_no_early_stop_b512` run. On the same complete dataset and 100-batch contract it reached 629.8 decisions/s with 7.5% data wait and 8.11/9.89 GB peak allocated/reserved CUDA memory, versus 566.6 decisions/s for batch 384. Retain watchdog OOM monitoring because the larger batch has less headroom.

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
- The complete 2026-07-10 through 2026-08-02 materialization is authoritative at 111,697 Episodes and 9,385,376 decisions (8,448,060 train; 937,316 validation). Eight workers completed it in 4,729.74 seconds with a 9,756,163,489-byte payload.
- Independently verify every one of the 2,292 compressed shard commitments, then decompress and parse every row to check physical counts, schema, split, and global maxima before declaring the dataset training-ready.

## 2026-08-04: extend audited source through 2026-08-02

- Add and fully CRC-validate the official 2026-08-02 daily Episode archive; retain its immutable SHA-256 and source inventory.
- Merge the audited 0028 prefix with the 2026-08-02 delta under the 0031 split contract. The resulting catalog has 111,697 winner Episodes, split 100,559/11,138, with 423 exact decks and 596 provenance sources.
- Replace the provisional 9.37 million estimate with the materialized manifest fact: 9,385,376 decisions.

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
- Derive final fixed-bucket coverage from the complete dataset, not the chronological sample. Audited maxima are card 134, event 64, option 78, effect 317, skill 82, and action including STOP 32; defaults extend to 160/64/96/384/96/64 respectively and continue to fail closed above those limits.
- Route teacher forcing through module `forward` and remove decoder data-dependent boolean indexing so a real `fullgraph=True`, eager-backend compile regression passes.
- Do not admit Inductor for formal training. Full prototype+policy compilation did not complete inside the diagnostic window and used 16 compile workers with multi-GB host memory; 2,048 real decisions also yielded 124 joint bucket signatures.

## 2026-08-04: exact epoch resume exception

- Retain four model-only inference/export slots and add one atomically replaced exact-resume slot.
- Resume state includes weights, optimizer, optional scheduler/GradScaler, completed-epoch trainer state, and Python/NumPy/Torch RNG.
- Resume only under identical dataset/config/model-contract/implementation commitments and matching canonical metrics epoch.
- The exact boundary is the end of a completed epoch. An interrupted partial epoch is rerun; batch-level continuation is not claimed.
- Resume files are finite-retention training assets and never enter candidate packages.

Verification evidence includes 58/58 passing project tests, the chronological semantic/padding benchmark under `.tmp/0031_feature_benchmark/run-20260804-v2/`, the full row audit under `.tmp/0031_training_optimization/full_dataset_verification.json`, and the completed-dataset CUDA fixed-bucket smoke under `.tmp/0031_rule_faithful_bc_smoke/run-c81d419894`. No formal training version was created.
