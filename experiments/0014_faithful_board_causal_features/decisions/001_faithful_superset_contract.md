# 001 - Faithful superset, one cache, A0 then AC

Date: 2026-07-27  
Status: accepted

0014 reuses the immutable 0013 `decision_record_v3` raw dataset and its existing train/validation
split through 2026-07-25. It does not rebuild labels, mix experts, or create one dataset per model
variant.

One `V1_full_feature_superset` model-ready cache stores the faithful 0010 tensors plus every audited
new feature family. A0 selects only the faithful 0010 view and uses the original 7,154,562-parameter
ID-only pointer architecture. It must not receive zero-valued auxiliary tokens or changed masks.

After A0 establishes the new-source baseline, the primary next run is AC with all audited feature
families enabled through typed lightweight readers and redesigned board-conditioned resource
retrieval. Individual A1-A5 ablations are diagnostic fallbacks, not prerequisites for AC.

The training loop reuses 0013's optimized contract: one optimization pass over train per epoch,
online teacher-forced optimization metrics, no static train evaluation or train greedy decode,
BF16 CUDA AMP, complete fixed-model validation every epoch, and local-first W&B mirroring.
