# 0047 — Meta-Routed MoE RL

Status: seven-Actor sparse V8 running; formal run
`V8_deck070_policy0814_moe7_sparse_512lane_micro256`.

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
explicit cache/context release. The
512-game schedules, seeds, policy, routing, PPO minibatch/objective, and update
semantics are unchanged.

## Identities

- Focal deck: exact deck `070`, content SHA-256 `2693a7b610ccdf2ed10a76a8872003dcb53456e54b006d3a3feff580520374fb`, 29-class ID 15.
- Shared focal initialization and all seven Actor seeds: immutable `Policy-0814`, actor SHA-256 `d7921f420c8f12155119d6caa0fef414f51c0a8368cd5ecf07cd7d71312c897b`.
- Critic initialization: paired 0814 Value SHA-256 `0ad6f57a32d37942cccdc78a8a9c8ef2f6f8784d08ea8ed77ca1513628c77d2d`.
- Rollout and evaluation opponent: independent complete `Policy-0814`, effective identity `476d57d55eb9c040fa4e75ce74ac5205af5d094a296db71cba4ecbf792902580`.

## Audited 0045 source graph

```text
observation -> frozen prototype/state/option prefix
            -> policy final Option Q/V LoRA -> one ActionDecoder -> root action
            -> one AllocationHead when required
            -> base final Option block -> isolated Critic/Value/Prize/15-class Meta
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
  each: Q/V LoRA + Decoder        Value/Prize/15-class Meta
        + Allocation
        |
public exact-Meta memory: UNKNOWN or locked 29-class ID
        |
hard warm-up route (U0-U5) or six independent E0↔Em sigmoid gates
        |
pi_eff(action|state) = sum_k gate_k * pi_k(action|state)
```

The mixture is over each expert's complete autoregressive action distribution,
including conditional Phantom Dive allocation. Parameters and hidden states are
never interpolated. PPO stores and recomputes the joint effective log-probability.

## Public identification and leakage boundary

`0047_public_exact_deck_candidates_v1` accumulates only publicly known opponent
Pokémon IDs from Semantic0031 (`relative_owner=opponent`, identity known). It
eliminates incompatible exact decks from the immutable 001–070 catalog and locks
only when every remaining candidate has the same 29-class ID. It never consumes
the scheduled deck ID, evaluator label, hidden zone, final game label, or Critic
Meta output. The routing ID is stored at each decision; earlier UNKNOWN rows are
never rewritten after later identification.

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
| Six routing logits | 6 | 2e-6 |
| Critic Value | 3,397,121 | 2e-5 |
| Critic Value adapter | 211,665 | 2e-5 |
| Critic Prize head | 103,681 | 2e-5 |

Actor total after Router unlock: 11,614,427. Critic total: 3,712,467. The frozen
0814 Actor module has 56,352,322 parameters and receives no gradients.

## Routing phases

- U0 through the rollout producing U5: hard routing, Router frozen. UNKNOWN and
  every non-core class use E0; 00→E00, 01→E01, 02→E02, 03→E03, 05→E05,
  and 27→E27.
- Starting with the rollout sourced from U5, six scalar Router logits become
  trainable. A confirmed core meta uses only E0 and its own Specialist, starting
  at 0.20/0.80. UNKNOWN and every other confirmed meta remain exactly E0.
- `old_three` and `new_four` are evaluation labels only; they do not group or
  share expert parameters.

## PPO and evaluation

The inherited PPO settings remain: 512 rollout games, three epochs, clip 0.10,
entropy coefficient 0.003, reference-KL coefficient 0.02, decoder/allocation LR
1e-5, LoRA LR 2e-5, Value/Prize LR 2e-5. `Frozen-0047-U0` is the immutable
same-architecture reference.

Every five updates, formal greedy evaluation uses FP16 storage followed by strict
FP32 materialization and complete Frozen Policy-0814 opposition:

- `old_three`: exact decks 001 / 002 / 011, CUDA 512;
- `new_four`: exact decks 007 / 003 / 009 / 023, CUDA 512;
- `remain_meta`: every other exact deck, Meta-first/deck-balanced, CUDA 512.

The evaluator's true Meta is used only for the fixed schedule and reporting.

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
- Seven-Actor CUDA PPO smoke: PASS; six official games, 473 decisions,
  pre-update max effective-logprob error `3.4570693969726562e-06`, three PPO
  epochs, finite Router update norm, no NaN/Inf.
- Three-pool CUDA smoke: PASS; two official games per pool, no
  errors/unfinished, FP16-storage/FP32-runtime identity gate PASS.
