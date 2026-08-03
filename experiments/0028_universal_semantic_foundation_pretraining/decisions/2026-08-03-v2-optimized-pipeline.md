# V2 Optimized Training Pipeline

V2 changes training scheduling, not actor or model semantics. Canonical JSONL remains the
hash-audited source. Training uses deterministic bounded-window length buckets and a producer
thread with prefetch depth 2; terminal rendering is rate-limited. Producer errors propagate and
early close joins the thread.

The 500-batch formal-model benchmark processed 128,000 decisions at 465.84 decisions/s. Producer
wait was 2.01 seconds out of 274.77 seconds (0.73%), consumer-active time was 99.27%, peak CUDA
allocation was 5.46 GB, and peak CUDA reservation was 7.88 GB. This passes the 15% maximum
data-wait gate, so mmap tensor rematerialization is not justified for V2.

Evidence: `../data_audit/throughput_2026-08-03_optimized_500.json`, SHA-256
`fb0abb7d2d74db264aca8161a33a647404ab2bc265bf8711dd74aafbb6744f30`.
