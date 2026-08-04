# 0031 Training Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 0031 materially faster on the RTX 5080 without dropping, falsifying, or pre-solving any rule-faithful actor-visible field, then materialize the complete audited 2026-07-10 through 2026-08-02 winner dataset.

**Architecture:** First fuse mathematically independent typed-field operators while preserving exact parameters and outputs. Then replace the single long four-layer state sequence with complete family-specific memories: global plus cards use the deep board encoder, resources use a lightweight encoder, events use a shallow temporal encoder, and a small fusion bank exposes all encoded families to option cross-attention and state summary. Benchmark each change independently on the same real sample; only after the GPU path is accepted, compile the full dataset with bounded CPU workers.

**Tech Stack:** Python 3.11, PyTorch 2.11, CUDA 12.8, BF16 AMP, SDPA Transformer layers, gzip JSONL canonical datasets, `unittest`.

## Global Constraints

- Do not modify `engine/source/`.
- Preserve every 0031 semantic field, exact card instance relation, event participant relation, target area/condition, official prototype fact, and known-opponent identity.
- Do not expose source/team identity to the actor.
- Do not add derived advice such as energy surplus, attack enablement, or preferred retreat payment.
- Categorical and numeric field fusion must be forward- and gradient-equivalent to the legacy independent-field implementation.
- Architectural optimization must fail closed on invalid masks and retain actor-visible access to full card, resource, and event memories.
- GPU benchmarks must use the same real rows, seed, batch policy, AMP mode, warmup, and optimizer settings for each A/B comparison.
- Do not start formal BC training in this task.
- Full materialization must target a new immutable 0031 dataset directory and require complete catalog coverage.

---

### Task 1: Reproducible Real-Sample Baseline

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/benchmark_training.py`
- Create: `.tmp/0031_training_optimization/<run-id>/baseline.json` (untracked evidence)

**Interfaces:**
- Consumes: `CanonicalDecisionDataset`, `SemanticPolicy`, fixed seed and real canonical rows.
- Produces: a JSON benchmark with synchronized step latency, decisions/s, data wait, peak CUDA memory, parameter count, model architecture label, and environment/load metadata.

- [ ] Add benchmark arguments for model configuration, warmup steps, timed repeats, optional profiler trace, and fixed-bucket padding while keeping real dataset rows as the only input.
- [ ] Record `nvidia-smi`-equivalent device facts and whether another GPU process is active, so contaminated measurements cannot be mistaken for isolated throughput.
- [ ] Materialize a bounded canonical sample from the audited raw sources if no suitable 0031 sample exists.
- [ ] Run the unchanged model on the sample with eager dynamic padding and save the baseline JSON.

### Task 2: Packed Categorical Fields

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/model/typed_fields.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_model.py`

**Interfaces:**
- Consumes: `values: Tensor[..., field_count]`, per-field vocabulary sizes, independent padding row zero.
- Produces: `CategoricalFields.forward(values) -> Tensor[..., d_model]` using one disjoint-offset embedding lookup and reduction.

- [ ] Add a test-only legacy reference that owns one `nn.Embedding` per field and copy identical weights into the packed implementation.
- [ ] Assert exact/equivalent CPU forward output, CUDA BF16-tolerant forward output, input-independent embedding gradients, zero padding gradients, vocabulary validation, and state-dict round trip.
- [ ] Run the focused test and verify it fails before the packed implementation exists.
- [ ] Store all field rows in one embedding table, reserve a distinct zero padding row per field through offsets, perform one lookup, mask local zero values, and sum along the field axis.
- [ ] Run focused tests and the real-sample A/B benchmark; retain the change only if semantics pass and dispatch/step time improves or is neutral within benchmark noise.

### Task 3: Batched Independent Numeric Fields

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/model/typed_fields.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_model.py`

**Interfaces:**
- Consumes: `values: Tensor[..., width]`, `states: Tensor[..., width]` where state 1 means present.
- Produces: the sum of independent per-field `Linear(1,d) -> GELU -> Linear(d,d)` projections and independent four-state embeddings using batched tensor contractions.

- [ ] Add a legacy numeric reference and copy every field's two linear layers and state embedding into packed tensors.
- [ ] Assert forward and all parameter gradients match in FP32, absent numeric values cannot affect output, default-present state matches explicit state 1, and invalid shapes fail closed.
- [ ] Run the focused test and verify it fails before packed numeric parameters exist.
- [ ] Implement batched first-layer affine, GELU, batched second-layer affine, present-state masking, and disjoint-offset state embedding without a Python field loop.
- [ ] Run focused tests and the real-sample A/B benchmark independently from Task 2 evidence.

### Task 4: Complete Hierarchical State Memory

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/model/config.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/model/state_encoder.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_model.py`

**Interfaces:**
- Consumes: unchanged `DecisionBatch` global/card/resource/event tensors and prototype embeddings.
- Produces: `EncodedState(tokens, mask, summary, cards)` with all family tokens present, plus explicit family spans for audit tests if needed.

- [ ] Add config fields for board, resource, event, and fusion layer counts with validation and serialization.
- [ ] Add tests proving output token count/mask still equals `1 + cards + resources + events`, padded tokens remain zero, card gather semantics remain stable, and every family can affect summary and option logits.
- [ ] Run the focused tests against the old joint encoder and capture the expected contract failure for family-specific architecture metadata.
- [ ] Encode global plus relation-enriched cards with the deep board stack, resources with a lightweight stack, and participant-enriched events with a shallow temporal stack.
- [ ] Build a small fusion bank from masked family summaries and inject fused context into every family while preserving all per-instance tokens for option retrieval.
- [ ] Benchmark the hierarchical encoder against the fused-field joint encoder using identical real batches and retain only a materially faster configuration with finite gradients and semantic sensitivity.

### Task 5: Regional Compile and Optimizer A/B

**Files:**
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/benchmark_training.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/run_bc.py`
- Modify: `train/0031_rule_faithful_semantic_foundation_pretraining/tests/test_training_pipeline.py`

**Interfaces:**
- Consumes: finite bucket-upper-bound padded batch shapes and the accepted eager hierarchical model.
- Produces: explicit `--compile-regions` and fused-AdamW configuration only when benchmark-admitted; ordinary eager checkpoints remain standard `state_dict` weights.

- [ ] Add a benchmark-only regional compile mode and record compile warmup separately from steady-state steps.
- [ ] Compare eager dynamic padding, eager fixed bucket padding, and regional compile fixed bucket padding on high-coverage real shapes.
- [ ] Compare standard and `fused=True` AdamW independently.
- [ ] Admit runtime flags only for configurations that pass output/gradient sanity and improve steady-state throughput enough to amortize compilation.
- [ ] Test that model-only and exact epoch-resume checkpoints remain loadable by an uncompiled `SemanticPolicy`.

### Task 6: Documentation and Contract Verification

**Files:**
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.md`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DESIGN.html`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DECISIONS.md`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/manifest.json`

**Interfaces:**
- Consumes: accepted code, measured parameter count, benchmark JSON, tests, and checkpoint contract.
- Produces: synchronized authoritative model/data-flow documentation and auditable optimization evidence.

- [ ] Document unchanged semantic schema, new family-specific tensor flow, exact layer counts, parameter count, padding/compile status, checkpoint portability, and exact-resume boundary.
- [ ] Record baseline and accepted throughput, peak memory, benchmark contamination boundary, and rejected optimizations.
- [ ] Cross-check markdown and HTML figures against live code and generated benchmark artifacts.
- [ ] Run the complete 0031 suite and `git diff --check`.

### Task 7: Full Audited Dataset Materialization

**Files:**
- Create: `rl_runs/0031_rule_faithful_semantic_foundation_pretraining/dataset/V1_rule_faithful_winners_20260710_20260802/` (untracked runtime dataset)
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/manifest.json`
- Modify: `experiments/0031_rule_faithful_semantic_foundation_pretraining/DECISIONS.md`

**Interfaces:**
- Consumes: every immutable raw root named by `winners_2026-07-10_2026-08-02.json`, public/full-engine prototype assets, and complete catalog coverage.
- Produces: immutable gzipped train/validation shards plus a complete manifest with hashes, episode/decision counts, maximum lengths, source provenance, elapsed time, and payload bytes.

- [ ] Resolve and verify all raw roots, catalog commitments, free disk, RAM, and current 0030 RL CPU/GPU load before launch.
- [ ] Choose a bounded worker count from a short CPU materialization benchmark; leave GPU unused and avoid memory pressure on 0030.
- [ ] Start full materialization in a foreground monitored terminal session and inspect progress, RSS, swap, disk, and 0030 health at intervals no longer than 60 seconds.
- [ ] Require atomic completion and `status: complete`; do not accept partial directories or catalog subsets.
- [ ] Load the finished dataset, verify shard hashes/counts, sample-collate boundary lengths, and record actual materialization time and size in project records.

