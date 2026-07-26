# 005 - Reject V2 model-ready publication after a source-digest race

Date: 2026-07-26
Status: accepted

The 8-worker V2 build produced all 159 expected shards and 162,128 records. During the build, a
format-only edit removed one trailing blank line from `training/batching.py`. Worker processes had
already imported the preceding bytes, while the publication manifest computed the compiler digest
again after the edit. The process-start digest was
`839bcf7273115db8d0caf654c5a194f3cf6c396d33fd6a36c769e0f070a4cbb4`; the manifest recorded
`60064dffb6d2780959589a83d57596bcc41f5834cd9146c24259a55c6503cf1d`.

Although Python behavior did not change, exact source provenance did. V2 is therefore invalid and
must never be used for training. It remains a local audit artifact and is not overwritten.

The materializer now captures compiler and materializer SHA-256 commitments before reading source
data, verifies the same commitments immediately before publication, and aborts/removes staging if
either changes. After the current implementation is committed and pushed, a strictly new
`V3_model_ready_semantic_v2` dataset will be built without source edits. Formal M0 allocation must
bind V3 and reject V2.
