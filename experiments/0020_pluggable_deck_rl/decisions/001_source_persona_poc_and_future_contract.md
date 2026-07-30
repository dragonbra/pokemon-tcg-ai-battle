# Decision 001: source persona PoC and future actor contract

**Date:** 2026-07-30  
**Decision:** Preserve 0015–0019 source-conditioned checkpoints as historical assets, but prohibit
team/source identity from entering future universal Actor forward paths. Source remains required
metadata for provenance, split, deduplication, sampling and grouped evaluation.

## Why source conditioning existed

The actor-visible Persona path began in `0015_dragapult_conditioned_bc`, not 0019. Project 0014's
R15 had no source embedding. Project 0015 combined demonstrations from multiple teams and explicitly
defined its target as `pi(a|s,D,E)` so conflicting labels could be separated by exact deck `D` and
expert identity `E`; its exported target fixed `E` to THIRD PTCG Club. Project 0016 inherited the
pattern, 0017 used target source 1, 0018 deployed source 0, and 0019 expanded it to 509 non-neutral
sources while selecting Epoch 13 by conditioned validation loss.

That chain is coherent for reproducing a named teacher. It does not match the current objective:
given actor-visible state/history and an exact deck, learn one source-agnostic policy `pi(a|s,D)`
that remains free to improve under RL.

## Offline evidence

On the same 707,126 Epoch-13 validation decisions:

| Forward | Loss | Greedy exact action |
|---|---:|---:|
| real expert source | 0.253080647 | 81.2871% |
| neutral source 0 | 0.368793374 | 74.8431% |
| difference | +0.115712727 neutral loss | -6.4440pp neutral exact |

The 81.2871% headline is therefore conditioned teacher reproduction, not the final neutral
deployment metric. The source branch contains 368,320 parameters. In the selected checkpoint,
source 98 has embedding norm 1.2938 and produces state/option residual norms 3.8120/1.0897 after
the learned gate and projections. Persona is an effective model input, not a report-only label.

## Official-engine PoC

V8 and V10 use the same epoch-13 checkpoint, Raging Bolt exact deck, 30-opponent catalog, 10 games
per opponent and 150/150 first/second seat schedule. V8 uses source 0; V10 changes only the policy
condition to source 98 (`James Cox & Henry Chao`). The official runtime does not expose a seeded
battle ABI, so the runs are independent same-protocol samples rather than paired identical deals.

| Version | Source | W-L-D | Win rate | First | Second | Wilson 95% |
|---|---:|---:|---:|---:|---:|---:|
| V8 neutral | 0 | 84-216-0 | 28.00% | 26.67% | 29.33% | 23.22%-33.33% |
| V10 Persona PoC | 98 | 79-221-0 | 26.33% | 25.33% | 27.33% | 21.67%-31.59% |

The unpaired difference is -1.67pp with a normal 95% interval of -8.78pp to +5.45pp. Source 98
does not recover strength and provides no evidence that the Persona residual smuggled a strong
Raging Bolt policy into the checkpoint. It did alter offline imitation substantially, so the
architecture mismatch remains real even though this Persona is not an Arena-strength asset.

Authoritative report:
`experiments/0020_pluggable_deck_rl/evaluation/V10_raging_bolt_source98_persona_poc.html`.

## RL consequence

Using source 0 is not weight decay: it removes an additive residual while preserving all shared
Backbone and Decoder weights. However, knowledge represented only in a nonzero Persona residual is
absent from the neutral forward. V9 trains only the autoregressive Decoder and value head; its
Encoder, deck-conditioning path and source branch are frozen. Decoder PPO may exploit remaining
shared features, but no component guarantees reconstruction of Persona-only representations.

V9 continues unchanged as an auditable test of RL on the legacy neutral foundation. V10 is
audit-only and must never become a deployment candidate, KL reference or RL warm start. Future
Raging Bolt RL also remains source 0; any strength claim must state that it starts from the legacy
neutral foundation.

## Successor contract

The clean successor removes `source_id` from model input and deletes the Persona embedding,
state/option projections and gate. With the otherwise identical 0019 R15 topology, this removes
368,320 parameters (17,756,162 to 17,387,842). Exact deck conditioning remains actor-visible.

Multi-source data remains allowed only with:

- exact team/source provenance retained outside the Actor;
- complete-Episode splits and duplicate/conflict audits;
- per-source and per-deck validation metrics;
- source/deck-aware sampling when a large source dominates;
- checkpoint selection using exactly the same source-free forward used at deployment;
- official-engine deck-specific strength evaluation.

Source dropout and conditioned-to-neutral Persona distillation are rejected because they preserve
the unwanted semantic channel. High-quality players select better demonstrations; their identity
does not become a policy input.
