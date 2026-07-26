# 007 - Make checkpoint selection pointers atomically replaceable

Date: 2026-07-26
Status: accepted

Formal `V2_m0_model_ready_input` completed two full optimization and train/validation evaluation
epochs. It wrote canonical metrics and immutable content-addressed checkpoints for both epochs. After
epoch 2, checkpoint selection failed because `criteria/latest.json` had been created with `O_EXCL`
at epoch 1 and was incorrectly treated as an immutable artifact.

V2 is preserved failed and will not resume. Its W&B run is complete and synced. Epoch 1 validation
loss/exact action were 1.13807/0.54475; epoch 2 were 1.17102/0.54110. Legality remained 1.0.

Content-addressed epoch `.pt` and `.json` files remain immutable and exclusive. Selection files
`latest.json`, `best_validation_loss.json`, and `best_validation_exact.json` are pointers by design;
they now update through a same-directory temporary file, fsync, and atomic replace. A regression
test writes two distinct epochs, verifies `latest` advances, verifies an unselected best pointer
remains unchanged, and confirms duplicate epoch content is still rejected.

The next formal M0 attempt must use a strictly increasing V3 version and train all three epochs from
scratch on the same accepted V3 model-ready dataset and unchanged optimization contract.
