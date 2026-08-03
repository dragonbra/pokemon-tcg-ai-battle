# V1 Throughput Stop

`V1_mid_universal_semantic_foundation` was stopped by the user before epoch 1 completed. It
produced zero checkpoints and is not a policy candidate. The canonical metrics file contains only
the run-start record.

Boundary timing isolated the bottleneck: representative gzip/JSON plus collation took about 0.175
seconds per batch while cached-batch CUDA forward/backward/update took about 0.234 seconds. V1 ran
these stages serially, causing the GPU to wait for CPU preparation. The immutable V1 directory is
retained as a stopped diagnostic and will not be resumed or reused.
