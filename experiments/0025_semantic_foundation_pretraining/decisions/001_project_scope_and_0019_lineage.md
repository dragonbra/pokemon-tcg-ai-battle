# Decision 001: project scope and 0019 lineage

Date: 2026-08-02

0025 is a new self-contained project rather than an in-place 0019 modification. The raw row, full-action, split, exact-deck, and causal knowledge contracts are inherited. Required executable files are physically frozen inside 0025 with source SHA-256 commitments; runtime imports from numbered projects are prohibited and tested.

The actor representation changes materially, so this is not checkpoint-compatible with 0019. The official output remains an ordered sequence of legal option indices plus STOP.

V1 is limited to prototype export, semantic compilation, a multi-memory model skeleton, streaming materialization, tests, and cost measurement. It deliberately does not launch a complete data build or formal training while 0024 is active. Consequently V1 has no W&B run, checkpoint, TensorBoard event, or arena candidate.

The official engine remains read-only. Full prototype extraction compiles an external utility into `engine/build/`; the official headers and sources are not patched.

