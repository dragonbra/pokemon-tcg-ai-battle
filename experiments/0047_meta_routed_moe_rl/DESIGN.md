# 0047 — Meta-Routed MoE RL

Status: V11 stopped at U4 after a routing-contract defect was confirmed; V12
failed during startup audit serialization; V13 was stopped after finding stale
candidate metadata; V14 was stopped after an unauthorized throughput downgrade;
corrected formal run
`V15_deck007_public_meta29_router29x7_cuda512_micro256` is prepared from Policy-0814.

V2 reached a complete 512-game rollout but failed before U1 when per-job
allocation contexts retained views over entire historical compacted CUDA
batches. V3 clones only each job's required row and releases rollout-only
contexts before PPO. This reduced but did not eliminate the 256-lane peak, so
V4 used 64 CUDA execution lanes and PPO forward microbatches of 128. V5 also
executes only the nonzero gate support: E0 alone for UNKNOWN/non-core decisions,
and at most E0+Em for a confirmed core meta. V5 U1 passed with max pre-update
effective-logprob error `6.4373e-06`, three PPO epochs, and behavior KL
`4.1847e-05`. V6 validated 256 rollout lanes at 5.536 games/s with max
pre-update effective-logprob error `5.2452e-06`. V7 restores the historical
0045 concurrency of 512 rollout lanes and validated 6.922 games/s with max
pre-update effective-logprob error `4.7684e-06`. V8 restores the historical
0045 PPO forward microbatch of 256 while retaining logical minibatch 2048 and
explicit cache/context release. V8 stopped cleanly at completed U3 before the
new experiment changed the Actor objective and fixed-evaluation schedule. V9
completed its deterministic U0 two-pool report, then failed before training
because the host Python lacked Matplotlib for the required Router heatmap.
Matplotlib 3.11.1 was installed for V10. V10 reached durable U4, then was
stopped by explicit user request when its full-state FP32 checkpoint was found
to consume about 502 MiB per update. Its weights were explicitly authorized for
deletion and its logs/metrics remain as a non-resumable historical record. V11
starts again from immutable Policy-0814 rather than continuing V10 U4, changes
the exact focal deck to 007, and adopts the compact persistence contract below.
V11 was then stopped at durable U4 because its public identifier incorrectly
enumerated exact decks 001–070 before folding them to Meta, while its trainable
Router was only six scalar E0-to-specialist gates. Those V11 metrics are retained
as failure evidence but are not evidence for the intended Meta-29 Router. V12
starts from immutable Policy-0814 again and loads no V11 delta.
The first corrected launch was allocated V12 and wrote only its U0 compact
checkpoint before an obsolete `router_alphas()` call in the lightweight Router
JSON serializer terminated startup. V12 performed no evaluation, rollout, or
optimization and remains a failed version. V13 fixes and tests the complete
29×7 Router audit schema and again starts from immutable Policy-0814.
V13 completed its 1,024-game U0 execution and began rollout, but a pre-commit
audit found that the portable candidate metadata still named the obsolete
exact-deck identifier and two-way topology. The games themselves used the new
runtime, but the deployment evidence is invalid under the identity hard gate.
V13 was stopped at 192/512 games of the first rollout with no PPO update. V14
fixes and tests both candidate metadata fields and again starts from Policy-0814.
V14 passed U0 and completed its first 512-game rollout, but it incorrectly used
64 CUDA lanes and PPO forward microbatch 128. That silently discarded the
previously optimized 512-lane / 256-microbatch training contract. V14 was stopped
before PPO and produced no U1. V15 restores and test-locks CUDA lanes 512,
rollout chunk 512, and PPO forward microbatch 256.

## Identities

- Focal deck: exact deck `007`, content SHA-256 `07bedfffbfad6ecb31733acc54c8110bb1934d8b1dc98bd9c4d37f6ba5c5e725`, 29-class ID 0.
- Shared focal initialization and all seven Actor seeds: immutable `Policy-0814`, actor SHA-256 `d7921f420c8f12155119d6caa0fef414f51c0a8368cd5ecf07cd7d71312c897b`.
- Critic initialization: paired 0814 Value SHA-256 `0ad6f57a32d37942cccdc78a8a9c8ef2f6f8784d08ea8ed77ca1513628c77d2d`.
- Rollout and evaluation opponent: independent complete `Policy-0814`, effective identity `476d57d55eb9c040fa4e75ce74ac5205af5d094a296db71cba4ecbf792902580`.

## Audited 0045 source graph

```text
observation -> frozen prototype/state/option prefix
            -> policy final Option Q/V LoRA -> one ActionDecoder -> root action
            -> one AllocationHead when required
            -> base final Option block -> isolated Critic/Value/Prize/29-class Meta
```

The 0045 Actor loss updated exactly ActionDecoder (1,027,202), Allocation Head
(621,761), and final-Option Q/V LoRA (10,240): 1,659,203 parameters. No Critic
tensor entered Actor inference.

## 0047 graph

```text
public observation + exact focal resources + legal options
                   |
              frozen 0814 semantic backbone
                   |
        +----------+------------------+
        |                             |
  seven Actor experts             one Critic
  E0 + E00/E01/E02/E03/E05/E27   base Option final block
  each: Q/V LoRA + Decoder        Value/Prize/29-class Meta
        + Allocation
        |
public key-Pokémon rules: UNKNOWN or current 29-class Meta ID
        |
hard warm-up route (U0-U5) or row-softmax(router_logits[meta_id, 0:7])
        |
pi_eff(action|state) = sum_k gate_k * pi_k(action|state)
```

The mixture is over each expert's complete autoregressive action distribution,
including conditional Phantom Dive allocation. Parameters and hidden states are
never interpolated. PPO stores and recomputes the joint effective log-probability.

## Public identification and leakage boundary

`0047_public_meta29_priority_rules_v1` accumulates only publicly known opponent
key-Pokémon IDs from Semantic0031 (`relative_owner=opponent`, identity known).
An explicit ordered if-else table maps those public cards directly to one of the
29 Meta classes. It never loads or predicts an exact deck ID and never reads the
001–070 deck registry or deck-to-Meta mapping at runtime. More-specific compound
rules precede base rules: for example Dragapult+Blaziken→16,
Dragapult+Dusknoir→15, otherwise Dragapult→0. As later public cards arrive, a
base classification may upgrade to its more-specific compound class. No
scheduled deck ID, evaluator label, hidden zone, final result, or Critic Meta
output enters this path. The Meta ID used at each decision is stored without
rewriting earlier decisions.

## Initialization and trainable audit

All seven Actor tensors are initially equal and have independent Parameter/storage.
0814 supplies the exact Decoder tensors. Q/V LoRA is zero-delta rank 4, alpha 8.
Because raw 0814 has no Allocation Head, one deterministic initialization
(`seed=470814002`) is copied identically to all experts; this is the only new
Actor tensor family.

| Group | Parameters | LR |
|---|---:|---:|
| Seven ActionDecoders | 7,190,414 | 1e-5 |
| Seven Allocation Heads | 4,352,327 | 1e-5 |
| Seven Q/V LoRA modules | 71,680 | 2e-5 |
| Meta × expert routing logits | 29 × 7 = 203 | 2e-6 |
| Critic Value | 3,397,121 | 2e-5 |
| Critic Value adapter | 211,665 | 2e-5 |
| Critic Prize head | 103,681 | 2e-5 |

Actor total after Router unlock: 11,614,624. Critic total: 3,712,467. The frozen
0814 Actor module has 56,352,322 parameters and receives no gradients.

## Routing phases

- U0 through the rollout producing U5: hard routing, Router frozen. UNKNOWN and
  every non-core class use E0; 00→E00, 01→E01, 02→E02, 03→E03, 05→E05,
  and 27→E27.
- Starting with the rollout sourced from U5, the complete 29×7 Router table
  becomes trainable. Every identified Meta reads exactly one table row and takes
  a seven-way softmax. Core rows initialize near their hard route (E0 0.1995,
  designated specialist 0.8, each other expert 0.0001); non-core rows initialize
  E0 0.9994 and each specialist 0.0001. UNKNOWN has no table row and remains
  exactly E0.
- `old_three` and `new_four` are evaluation labels only; they do not group or
  share expert parameters.

## PPO and evaluation

The inherited PPO settings remain: 512 rollout games, three epochs, clip 0.10,
entropy coefficient 0.003, reference-KL coefficient 0.02, decoder/allocation LR
1e-5, LoRA LR 2e-5, Value/Prize LR 2e-5. `Frozen-0047-U0` is the immutable
same-architecture reference. V15 uses a strictly win-only Actor advantage:

```text
A_actor = normalize(A_terminal_win_loss)
Prize Advantage Actor coefficient = 0.0
```

Directional Prize deltas, `V_prize`, and its Critic-side auxiliary loss remain
available for variance/diagnostic work, but Prize Advantage is never added to
the Actor policy loss.

## V15 checkpoint and retention contract

Training checkpoints use schema
`0047_meta_routed_moe_compact_fp32_delta_v2_meta29x7`. The immutable Policy-0814 Actor
is never copied into an update file. Each file records the exact actor/value/
effective base hashes and stores every non-`actor.*` effective tensor in FP32:
all seven expert Decoders/Allocation Heads/QV LoRA modules, Router state, the
trainable Critic path, Value adapter, Prize head, and required buffers. A fresh
audited Policy-0814 model plus this delta must strict-load to the exact complete
effective policy; changed base hashes, missing/unexpected tensors, wrong shapes,
wrong dtypes, optimizer state, replay, or RNG state hard-fail. The expected
training checkpoint is about 59 MiB rather than the prior 502 MiB.

U0 and every update are saved before downstream work. U0/U5/U10/etc evaluation
reloads the just-saved compact checkpoint into a fresh base and requires its
effective tensor hash to equal the live policy before any official-engine game.
Only after the evaluation report returns `PASS` and the canonical local/W&B
metric is logged may checkpoints between evaluated nodes be deleted. Thus a
successful U5 removes U1-U4 while retaining U0/U5; a failed or interrupted
evaluation deletes nothing. This `evaluated_nodes_only_after_successful_eval`
exception was explicitly authorized by the user on 2026-08-16. Router JSON
remains as lightweight audit metadata.

Formal greedy evaluation runs first at U0, before any rollout or PPO update, and
then at U5/U10/etc. It uses FP16 storage followed by strict FP32 materialization
from the source compact FP32 checkpoint, records all three candidate identities,
and uses complete Frozen Policy-0814 opposition. There are exactly two physical
CUDA-512 pools:

- `focus_seven`: 256 games balanced within `old_three` exact decks 001/002/011,
  plus 256 games balanced within `new_four` exact decks 007/003/009/023;
- `remain_meta`: every other exact deck, Meta-first/deck-balanced, CUDA 512.

Reports and W&B retain separate `old_three` and `new_four` aggregates despite
their shared physical pool, plus `focus_seven`, `remain_meta`, every evaluated
exact deck, and every observed meta. The evaluator's true Meta is used only for
the fixed schedule and reporting. The Router still sees only public evidence.
`router/meta_expert_heatmap` is emitted at U0 and every later eval point.

## Training opponent schedule

Every rollout still contains 512 games and every opponent is the complete
Frozen Policy-0814. Exactly 256 dedicated branch slots are split as evenly as
integer arithmetic permits across decks `007/003/001/002/009/011/023` (36–37 each,
with deterministic remainder rotation). The other 256 slots use the existing
Meta-first then exact-deck-balanced schedule over all active metas. Core decks
may therefore also receive a small number of games from the general half.
Deck 009 maps to meta04 (Lucario) and deck 023 maps to meta27 (Hydrapple).
The branch seeds, exact counts, and final per-deck weights are checkpoint/run
metadata. Rollout chunks emit 64-game progress lines; PPO emits per-minibatch
progress lines without publishing partial strength metrics as completed updates.

## Lightweight gates

- U0 root action parity with raw 0814: PASS, exact actions/lengths.
- Seven-Actor initial equality and storage independence: PASS (CPU).
- Core hard routes, strict E0 fallback, two-way 0.20/0.80 gates, and Router
  gradient: PASS (CPU).
- Seven-Actor V12 full-softmax CUDA PPO smoke: PASS; eight official-engine games,
  694 decisions, 8/8 valid, lane identity audit PASS, all seven experts with
  nonzero effective use, 100% game identification, 2.58% UNKNOWN decisions, and
  finite Router update norm `3.9427250158041716e-06`. The trainer's pre-update
  behavior-logprob hard gate (`1e-4`) passed before optimization.
- V9 two-pool schedule contract: PASS; focus is exactly 256 old-three plus 256
  new-four, remain-meta is exactly 512 and excludes all seven focus decks.
- V12 public Meta-29 rules match the frozen labels for all 70 audited deck
  samples when their Pokémon signatures are fully public: PASS, 70/70 (CPU).
- V12 29×7 Router gradients, compact FP32 delta round-trip, immutable-base
  rejection, exact inventory, and PASS-gated evaluated-node pruning: PASS (CPU).

The Pokémon TCG rules and official-engine action contract are unchanged by V15.
The focal exact-deck conditioning remains 007/ID0. Opponent routing uses only
public opponent key-Pokémon evidence and produces a Meta ID, never a deck ID.
