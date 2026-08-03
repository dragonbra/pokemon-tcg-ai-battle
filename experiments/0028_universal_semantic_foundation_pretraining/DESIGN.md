# 0028 Universal Semantic Foundation Pretraining

Status: **V1 and V2 stopped before epoch 1; V3 completed four epochs, then failed with a CUDA
unknown error. Its latest epoch-4 checkpoint has completed eight formal zero-shot deck evaluations.**

## Objective

0028 turns the 0025 V4 canonical semantic policy into a self-contained, readable universal
pretraining project. It preserves the actor semantics and full-action imitation contract while
changing the corpus from one Raging Bolt deck to every audited official winner perspective from
2026-07-10 through the latest locally available archive, currently 2026-08-01.

This is a same-semantics rewrite, not a checkpoint-compatible wrapper. 0025 is provenance only;
0028 cannot import 0025 executable code.

## Evidence Boundaries

Three evidence layers remain distinct:

1. **Official general rules:** attack ends the turn; evolution has a turn clock; Supporter, manual
   Energy, Retreat, and Stadium have per-turn budgets; zones and Prize/deck win conditions are
   distinct. Source: `docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`.
2. **Current runtime/card facts:** Card, Attack, Skill, and Effect prototype fields come from the
   current official card data and unmodified official engine-derived sidecars.
3. **Project hypotheses:** winner-only BC, model width/depth, normalization, and checkpoint
   selection are training choices, not official rules.

## Actor Visibility

Actor-visible inputs:

- current global selection context, turn, first-player relation, action count, per-turn budgets,
  status flags, zone counts, and selection bounds;
- visible own/opponent Active and Bench instances, visible hand/discard/stadium/playing cards,
  attached Energy, tools, evolution parents, HP, damage, status, and appearance timing;
- exact registered own-deck multiset plus causal visible-zone ledger and bounded deck/Prize
  knowledge;
- recent actor-visible events and explicit known/unknown opponent-hand boundary;
- every legal option with action/source/target/attack/energy/context identities, direct numeric
  facts, source/target instance relations, and related Skill/Effect prototype identities.

Never actor-visible:

- team name, source ID, expert identity, persona, or leaderboard identity;
- opponent private card identities;
- future replay information, terminal result, or copied target action;
- legacy actor tensors or raw replay payloads.

## Target Tensor Contract

The initial 0028 schema preserves 0025 V4 field semantics. Exact field names and shapes become
authoritative only when `contracts/fields.py` and `contracts/batch.py` are implemented and tested.

| Memory | Token source | Core content |
|---|---|---|
| Global | one token | selection context, turn budgets, status, zone counts, deck/hand/Prize pressure |
| Cards | variable `C` | card instance identity, zone/owner, HP/damage, Energy/tools/evolution timing, Card prototype |
| Resources | variable `R` | exact registered-deck multiset, visible zones, bounded/exact deck and Prize knowledge |
| Events | variable `E` | visible action/effect history, movement, damage, coins, serial/target relations |
| Options | variable `O` | legal action identity, source/target, Attack ID, typed Energy choice/deficit, HP/damage facts |
| Skills | variable `S` | option-parented ability/play/attack Skill IDs and roles |
| Effects | variable `F` | option-parented Effect IDs, phases, target/select/condition semantics |

Variable lengths are padded only at collation. Boolean masks prevent padded tokens from becoming
attention keys or legal outputs. Numeric zero is not used to mean both unknown and not-applicable;
an explicit field-state tensor carries that distinction.

## Model Data Flow

```text
official prototype sidecars
    Card / Attack / Skill / Effect
                 |
                 v
raw actor-visible decision + exact deck + causal history + legal options
                 |
                 v
          CanonicalFeatureCompiler
                 |
                 v
        strict named DecisionBatch
                 |
       +---------+---------------------------+
       |                                     |
       v                                     v
PrototypeEncoder                       StateEncoder
identity + categorical + numeric       global/cards/resources/events
       |                                     |
       +----------------------+--------------+
                              v
                        full state memory
                              |
                              v
                         OptionEncoder
          option facts + source/target gathers + prototypes
                 cross-attention to full state memory
                              |
                              v
                    ordered ActionDecoder
                  option pointer + explicit STOP
                              |
                              v
                 ordered legal option indices
```

The state is not reduced to one vector before option reasoning. Each option cross-attends to the
full encoded state memory. A pooled state summary initializes the autoregressive action decoder,
but the option tokens it points at already contain option-specific retrieval from full state.

## Forward Pseudocode

```python
def forward(batch, selected_prefix=None):
    batch = DecisionBatch.validate(batch)

    prototypes = prototype_encoder.lookup_all(batch)
    state = state_encoder(
        global_token=batch.global_fields,
        cards=batch.card_instances + prototypes.cards,
        resources=batch.exact_deck_ledger + prototypes.resource_cards,
        events=batch.visible_events + prototypes.event_cards,
    )

    options = option_encoder(
        option_fields=batch.options,
        source_instances=state.gather_cards(batch.option_source),
        target_instances=state.gather_cards(batch.option_target),
        attacks=prototypes.option_attacks,
        skills=prototypes.option_skills,
        effects=prototypes.option_effects,
        memory=state.tokens,
        memory_mask=state.mask,
    )

    decoder_state = action_decoder.initialize(state.summary)
    decoder_state = action_decoder.consume_prefix(options, selected_prefix, decoder_state)
    return action_decoder.pointer_logits(options, decoder_state, batch.selection_bounds)
```

## Residual And Attention Structure

Each Transformer block uses pre-normalization residual paths:

```text
x1 = x + SelfAttention(LayerNorm(x), padding_mask)
x2 = x1 + FFN(LayerNorm(x1))
```

Each option-decoder block additionally retrieves state memory:

```text
q1 = q + SelfAttention(LayerNorm(q), option_mask)
q2 = q1 + CrossAttention(LayerNorm(q1), LayerNorm(state_memory), memory_mask)
q3 = q2 + FFN(LayerNorm(q2))
```

Residuals preserve the projected input token at each block; they do not preserve raw JSON or raw
unscaled numbers. For exact numeric reasoning, the compiler must therefore provide normalized
values and explicit field state without lossy semantic omission.

## Dataset Contract

Initial interval: 2026-07-10 through 2026-08-01, covering 23 locally available official archives.

Inclusion requires:

- terminal two-player official Episode;
- one unique positive reward winner with terminal status;
- exact 60-card registered winner deck;
- unambiguous actor perspective and first-player evidence;
- strict actor-visible observation and ordered legal-action reconstruction.

Episode ID and payload hash deduplicate repeated rolling archives. Any duplicate Episode ID with a
different payload fails closed. Whole Episodes receive deterministic train/validation assignment.
Team/source identity is stored only in audit metadata and stratified reports.

### Canonical Mid Dataset Evidence

The immutable dataset at
`rl_runs/0028_universal_semantic_foundation_pretraining/dataset/V1_mid_winners_20260710_20260801`
was materialized from all 23 official archives in the interval and passed full shard/hash reload.
These are measured facts from its manifest, not planning estimates:

| Fact | Measured value |
|---|---:|
| unique winning Episodes | 107,117 |
| train / validation Episodes | 96,349 / 10,768 |
| total decisions | 8,983,982 |
| train / validation decisions | 8,088,836 / 895,146 |
| exact registered decks | 403 |
| provenance sources | 568 |
| train / validation shards | 1,975 / 219 |
| compressed payload | 2,875,581,660 bytes (about 2.7 GiB on disk) |
| materialization | 12,309.92 seconds with 8 workers |

Episode IDs are unique, train and validation share no Episode, and actor batches exclude
`source_id`, team identity, payload hashes, legacy tensors, and ordered targets. The dataset
manifest SHA-256 is
`7b8bd85a33e756e41212f57ff43a57abd59a0626a47f10fa2101c715c42adc15`.

Measured maximum variable lengths are `card_cat=117`, `resource_cat=29`, `event_cat=64`,
`option_cat=78`, `option_skill_id=82`, and `option_effect_id=317`. These maxima define capacity
requirements for collation; padding remains masked and does not alter actor semantics.

## Training Contract

The target is ordered full-action behavior cloning. Each epoch performs exactly one train pass and
one complete teacher-forced plus greedy validation pass. Canonical facts are shared across decks;
exact deck conditioning comes from the registered-deck multiset, not team identity.

Formal records use:

- `trainer/epoch` as the step;
- `bc/optimization/*` for online train-pass diagnostics;
- `bc/validation/*` for fixed epoch-end validation;
- model-only checkpoints for latest, best validation loss, best teacher exact, and best greedy
  exact;
- JSONL first, then TensorBoard, then W&B online mirroring.

### Data Supply Pipeline

V1 decoded gzip JSONL and collated the next batch synchronously with GPU training. Boundary timing
measured about 0.175 seconds of gzip/JSON/collation plus 0.234 seconds of cached-batch GPU work for
a representative batch, so the two costs were serialized and the GPU repeatedly waited for CPU
input. V1 was stopped by the user before completing epoch 1 and produced no checkpoint.

V2 preserves the canonical JSONL and every actor tensor, but changes scheduling only:

1. training shards and batch groups remain deterministically shuffled by epoch seed;
2. records are grouped by bounded actor-visible length buckets for state, option, Skill, and Effect
   sequences, reducing padding without changing or dropping a decision;
3. one bounded producer thread prepares at most two CPU batches while CUDA executes the current
   batch; producer exceptions propagate to the trainer and early close joins the thread;
4. progress rendering is rate-limited instead of redrawing the terminal for every batch.

The 500-batch formal-model gate processed 128,000 decisions in 274.77 seconds. It measured 465.84
decisions/s, 2.01 seconds total producer wait, a 0.73% data-wait fraction, 99.27% consumer-active
time, 5.46 GB peak CUDA allocation, and 7.88 GB peak CUDA reservation. This passes the required
maximum 15% data-wait gate, so compact mmap rematerialization is neither necessary nor authorized
for V2. Evidence:
`data_audit/throughput_2026-08-03_optimized_500.json` (SHA-256
`fb0abb7d2d74db264aca8161a33a647404ab2bc265bf8711dd74aafbb6744f30`).

The formal `SemanticPolicy` contains 21,837,082 parameters. Full-dataset CUDA calibration on the
local RTX 5080 measured the following throughput and memory use:

| Batch | Decisions/s | Peak allocated | Peak reserved |
|---:|---:|---:|---:|
| 128 (20 batches) | 318.4 | 2.48 GB | 3.06 GB |
| 192 (100 batches) | 489.5 | 3.69 GB | 5.37 GB |
| 224 (100 batches) | 510.7 | 4.40 GB | 5.61 GB |
| 256 (100 batches) | 525.4 | 4.99 GB | 7.38 GB |

V2 started with train and validation batch size 256, but was stopped before epoch 1 and produced no
checkpoint when GPU-forward optimization was requested. Its measured roughly 2 batch/s is a
performance baseline, not a completed BC result.

### GPU Forward Optimization Gate

PyTorch profiling on the RTX 5080 confirms that masked attention already uses fused
memory-efficient SDPA. Three profiled train steps spent about 286 ms in CUDA kernels but 685 ms in
CPU operator scheduling; repeated small embedding, linear, reshape, and copy launches are material.
Flash SDPA cannot be forced through the current `key_padding_mask`: PyTorch 2.11 rejects the
non-null attention mask instead of silently preserving semantics. `torch.compile` is not accepted
for V3: after repairing the local compiler include path, full teacher-step compilation still failed
to complete reliably at representative batch sizes.

Two algebraically equivalent common-subexpression eliminations are retained:

1. the finite Card/Attack/Skill/Effect prototype tables are encoded once per policy forward and
   gathered by every state and option occurrence, instead of recomputing the same prototype MLP for
   each duplicate identity;
2. ordered teacher forcing computes option pointer keys and option bias once per decision and reuses
   them across action steps.

CPU-side range validation remains fail-closed, while redundant CUDA scalar assertions are skipped
in the hot path to avoid host synchronization. The actor tensors, prototype fields, state and option
attention, masks, logits, targets, and loss are unchanged. Regression tests require direct/shared
policy logits to stay within `1e-6` with identical argmax, and cached/uncached decoder logits to be
exactly equal. Fused AdamW
and fused categorical-field lookup were measured and rejected because they did not improve
throughput. The clean matched gate measured 468.0 decisions/s for direct prototype recomputation at
batch 256, 515.7 for shared prototypes at batch 256, and 609.7 for shared prototypes at batch 512.
The selected path is 30.3% faster than the direct baseline, with 8.54 GB peak allocation, 12.17 GB
peak reservation, and a 2.45% data-wait fraction. This authorizes V3 with train and validation batch
size 512. Evidence: `data_audit/throughput_2026-08-03_gpu_forward_ab.json`.

## Current Phase And Resource Gate

The 0026 RL run has been stopped and no longer owns the GPU/CPU budget. Full catalog scanning and
canonical materialization are complete, all dataset shard hashes reload successfully, and the
expanded 0028 test suite passes. `V1_mid_universal_semantic_foundation` is an immutable stopped
throughput diagnostic with zero completed epochs and zero checkpoints. V2 is also immutable and
stopped with zero completed epochs and zero checkpoints. Unit, semantic-equivalence, memory, and
uncontaminated throughput gates authorized `V3_shared_prototype_batch512`. V3 completed four
epochs and persisted model-only checkpoints before the training child exited on a CUDA unknown
error; W&B synchronization completed. The latest checkpoint is epoch 4 / global step 63,196, with
validation loss 0.3920408, exact action 77.2054%, and legal action 100%.

## Pretrained Release

The current self-contained pretrained release is
`archive/pretrained/0028_universal_semantic_foundation_pretraining/`. While V3 continues training,
it publishes the real Epoch 1 `best_validation_loss` checkpoint as `epoch1_sample`: validation loss
0.4295407, exact action 74.7020%, and legal action 100%. This is an early zero-shot integration
asset, not the final selected V3 weight and not official-engine strength evidence.

The archive freezes the 0028 inference contracts, Card/Attack/Skill/Effect prototype registries,
causal ledger, feature compiler, model source, strict loader, and a stateful zero-shot observation
adapter. It contains no source/team actor input and no optimizer, scheduler, scaler, RNG, dataset,
replay, or rollout state. The archive verifier checks every committed file hash, the source-tree
hash, model-only payload boundary, strict 21,837,082-parameter loading, and a minimal forward.
A real official-replay smoke also confirms that the adapter consumes an exact 60-card deck and
returns an ordered action satisfying the live observation's legal bounds.

V3 did not finish its planned 20 epochs, so the archived `epoch1_sample` remains unchanged.
`promote_checkpoint.py` remains the only supported in-place final-weight promotion path; this
evaluation package does not silently promote or overwrite the pretrained archive.

## Formal Zero-Shot Evaluation

Eight self-contained candidates bind the same V3 `latest.pt` (SHA-256
`5e0a6eea42bf228a9bd977cf56fdd14ad360fbceffcf5c19e2d9fe1b39713d98`) to different exact
60-card decks. They import no executable code from another numbered training project. Every run
uses `0019_foundation_51_exact_decks_v4`, 51 opponents x 10 games, balanced first/second order,
two separate persistent batched `cuda:0` inference services, eight isolated official-engine CPU
workers, and the `league_deck_quality` metric profile.

| Version | Exact deck | W-L | Overall | First | Second |
|---|---|---:|---:|---:|---:|
| V9 | Alakazam / Dudunsparce 002 | 312-198 | 61.18% | 64.71% | 57.65% |
| V6 | Dragapult ex 001 | 306-204 | 60.00% | 65.49% | 54.51% |
| V10 | Dragapult ex / Crushing Hammer 001 | 281-229 | 55.10% | 56.86% | 53.33% |
| V4 | Mega Lopunny ex 001 | 279-231 | 54.71% | 54.51% | 54.90% |
| V7 | Dragapult ex / Dusknoir 001 | 208-302 | 40.78% | 42.35% | 39.22% |
| V5 | Lucario / Hariyama | 178-332 | 34.90% | 39.61% | 30.20% |
| V8 | Raging Bolt ex, James Cox / Henry Chao 001 | 106-404 | 20.78% | 21.96% | 19.61% |
| V3 | Raging Bolt ex / Teal Mask Ogerpon ex user deck | 96-414 | 18.82% | 19.61% | 18.04% |

All 4,080 games completed with zero draws, errors, or unfinished games. The authoritative entry is
`evaluation/index.html`, with immutable V3-V8 reports and artifact backlinks. These results show
substantial exact-deck sensitivity: Alakazam / Dudunsparce 002 led at 61.18%; pure Dragapult
outperformed its Crushing Hammer variant by 4.90 points and the Dusknoir variant by 19.22 points
under the same model and pool. Mega Lopunny also exceeded 50%. This is
official-engine zero-shot strength evidence for these exact checkpoint/deck/pool contracts, not a
promotion decision or proof that one result generalizes to related decklists.

## Completion Evidence Required

0028 is not complete until code, DESIGN.md, DESIGN.html, dataset audit, manifest, model contract,
checkpoint metadata, and smoke/full-run records agree on fields, tensor shapes, parameter count,
date interval, source visibility, and phase.
