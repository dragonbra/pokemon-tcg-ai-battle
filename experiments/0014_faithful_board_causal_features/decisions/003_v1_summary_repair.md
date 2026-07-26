# 003 - Reconstruct zero-byte V1 derived summaries

Date: 2026-07-27  
Status: repaired

`V1_0010_faithful_control` completed all six epochs and retained its canonical seven-row
`training_metrics.jsonl`, six content-addressed checkpoint payloads/manifests, TensorBoard events,
synced W&B run, and terminal `status.json`. At process completion, the derived
`training_summary.json` and `checkpoint_selection.json` were observed as zero-byte files even
though the terminal status had already been written as complete.

The two derived files were reconstructed from canonical JSONL row 6, the three immutable criterion
manifests, and W&B status. No metric, checkpoint, model weight, dataset, optimizer state, or W&B
history was altered. The writer is hardened separately to fsync and reject an empty publication.
