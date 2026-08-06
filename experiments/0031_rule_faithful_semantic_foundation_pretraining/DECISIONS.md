# 0031 Decisions

## 2026-08-06: friend-0806 pretrained release and Third PTCG Club zero-shot evaluation

- Publish `large-model-0806.tar.gz` as the self-contained pretrained asset `archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/`. Preserve the supplied `0031_model_only_checkpoint_v1` payload byte-for-byte as `model.pt` with SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`; it is epoch 11 / global step 141878 from metadata version `V2_full_winners_bs1024_20260616_20260803` and contains no resumable optimizer state.
- Bind the release to its own checkpoint metadata and external dataset/config/implementation hashes. This release is an additional pretrained asset; it does not replace or append to the active local `V4_lr5e4_no_early_stop_b512` training run.
- Evaluate the exact 0033 Third PTCG Club Dragapult deck (`deck.csv` SHA-256 `5db1e0d52fc723e8b2f688d76f780715fd726171f4d804f1d4d55673ba9b0ac2`) using an fp16-storage/fp16-runtime zero-shot export against `0019_foundation_51_exact_decks_v4`. The official-engine run `run-8f9bef885a85483db2103d2cf291501d` completed 510/510 games with 265 wins, 245 losses, 0 draws, 0 errors, and 51.96% win rate.
- Store the authoritative result at `evaluation/V9_friend0806_epoch11_dragapult_third_frozen0019.html`. Under the same deck/pool/runtime-precision contract, the two friend-0805 reference runs were 47.65% and 47.45%; the 0806 point estimate is respectively +4.31 and +4.51 percentage points. Treat these finite official-engine samples as comparable strength evidence, not an automatic promotion or statistical-certainty claim.

## 2026-08-05: epoch-2 continuation throughput optimization

- Profile the exact epoch-2 batch-512 path before resuming. Representative post-warmup work attributed about 401 ms/batch to forward, 270 ms/batch to backward, 23 ms/batch to host-to-device transfer, 1.3 ms/batch to gradient clipping, and 2.7 ms/batch to fused AdamW. The dominant GPU autograd cost was repeated generic `IndexBackward` from shared prototype-table advanced indexing; gzip/JSON/collation remained a secondary producer cost.
- Replace differentiable prototype-table advanced indexing with `torch.nn.functional.embedding`. Table values, shapes, and trainability are unchanged, while the backward path uses the dedicated embedding operator.
- Replace masked-logit materialization in teacher loss with dense cross entropy over transposed logits and `ignore_index=-100`. Targets, valid-token mean reduction, teacher forcing, and gradients retain the same contract. Parse canonical JSON rows with `orjson` 3.11.5 while retaining the same decoded Python values and gzip shards.
- Admit the combined eager path after an exact epoch-2 equivalence audit: loss delta `5.96e-8`, gradient relative L2 `9.65e-4`, maximum absolute gradient delta `7.63e-6`, finite gradients for all 197 trainable parameter tensors, and maximum BF16 logit delta `9.77e-4` (one BF16 quantization unit).
- On same-period batch-512 measurements, the unoptimized reference reached 620.3 decisions/s. The admitted path reached 721.4 decisions/s over 150 measured batches (+16.3%) and 706.5 decisions/s over 300 measured batches (+13.9%), with peak CUDA memory approximately unchanged at 8.32/9.90 GB allocated/reserved. Treat this as the local RTX 5080 execution envelope, not a cross-hardware guarantee.
- Reject asynchronous finite guards and reduced progress scalar synchronization because they did not improve throughput. Reject decoder-only Inductor because GRUCell FakeTensor support and the local Triton compiler toolchain did not satisfy the formal path. Reject a manually expanded GRU because BF16 numerical drift exceeded the equivalence boundary. Reject threaded/process collation, pinned-memory CUDA-stream prefetch, and prefetch depth 8 because each was slower; retain prefetch depth 4.
- Continue the same `V4_lr5e4_no_early_stop_b512` semantic run and stable W&B identity from exact epoch 2/update 33002. Because the admitted execution implementation changes the implementation commitment without changing model/optimizer/RNG/trainer state, preserve the original epoch-2 resume file and publish an audited compatibility-only migrated resume file before launch.

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

Verification evidence includes 61/61 passing project tests, 41/41 relevant repository experiment/W&B tests, the chronological semantic/padding benchmark under `.tmp/0031_feature_benchmark/run-20260804-v2/`, the full row audit under `.tmp/0031_training_optimization/full_dataset_verification.json`, the completed-dataset CUDA fixed-bucket smoke under `.tmp/0031_rule_faithful_bc_smoke/run-c81d419894`, and the epoch-2 optimization/equivalence artifacts under `.tmp/0031_training_optimization/`.
