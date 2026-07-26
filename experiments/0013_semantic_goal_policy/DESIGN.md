# 0013 Semantic Goal Policy

**Project ID:** `0013_semantic_goal_policy`
**Status:** data contracts and causal semantic input layer implemented; no checkpoint, formal training version, or empirical policy-strength result exists.

## Purpose and boundaries

This project researches a reusable representation architecture for deck-specific policies. The
initial policy remains restricted to the Alakazam–Dudunsparce deck and the named expert Yushin
Ito. It does not mix decks, teams, or experts; train cross-deck shared weights; introduce a card
text language model; generate actions outside official legal options; modify `engine/source/`;
automatically admit candidates to the opponent pool; or perform a Kaggle submission.

The three project roots are:

- implementation: `../../train/0013_semantic_goal_policy/`
- tracked design and evidence: this directory
- generated runtime assets: `../../rl_runs/0013_semantic_goal_policy/`

No `V<n>_<tag>` training version is allocated by this scaffold.

## Approved source contract

The planned source is official episode-player trajectories dated 2026-07-18 through 2026-07-25,
as declared by `data/raw/episodes/source_manifest.json`.
Records are restricted by `casefold_whitespace_collapse_exact` comparison to expert name `Yushin Ito`,
that player's winning trajectories, causal actor-visible decisions, and each episode-player's actual
registered deck. The declared patch
`data/raw/episodes/patches/2026-07-24/episode-87841523-replay.json`
replaces any archive member with the same episode ID; it must not be added as a duplicate. The source boundary is named-expert, date-bounded, and
winner-only; submission IDs are unavailable, so it is not claimed to be one verified binary
policy. The future builder must preserve ordered full actions and perform group-isolated
train/validation splitting. These are approved contracts, not completed audit findings.

## Input and knowledge contract

Only the acting player's current observation, causally available redacted logs, own prior actions,
own submitted deck manifest, and prior player-local knowledge state may become features. Future
frames, outcomes, rewards, omniscient visualizer state, hidden opponent construction, and hidden
ordering are prohibited.

The implemented typed representation (`semantic_goal_typed_input_v1`) contains state, entity,
registered-deck capability, causal ledger/belief, event, and legal-option tokens plus validated typed
relations. Observed, remembered, inferred-exact, bounded, unknown, missing/malformed,
not-applicable, padding, and overflow are distinct states. The deterministic
`card_effect_ontology_v1` registry loads the read-only official card CSV, combines repeated card
rows into ordered move/effect sequences, and shares auditable structural semantics and functional
capabilities across all card-bearing token kinds; card ID remains only an identity residual.

The implemented `causal_knowledge_v1` state treats every non-null `select.deck` as the actor's full
ordered current-deck view and legal options as the eligible subset. Prize identities remain unknown
before a verified full view and become inferable only on subsequent decisions, never through future
backfill. Opponent hand state consists of legally known physical instances plus anonymous unknown
slots and degrades when correspondence is lost. Serial is a battle-local tracking key, not a numeric
strategy feature. See [`data_audit/visibility_audit.md`](data_audit/visibility_audit.md) for the
read-only engine evidence. The player's registered deck is known, while the opponent's complete
deck is never supplied. Offline building and official-engine runtime must use this same causal state.

## Model and action contract

The reference research architecture is implemented for CPU/GPU smoke at `d_model=384`, six
pre-norm state layers, eight attention heads, 1536-wide FFNs, two semantic option cross-attention
layers, and approximately 18.63M parameters per explicit M0–M5 variant. M0–M3 ignore live ledger
and event channels by contract; M4 adds independent live-ledger Goal-QKV routing; M5 adds event and
relation paths plus an explicitly uncalibrated finite `V(s)` head. This is an implemented model
shape and smoke result, not a trained policy or strategy-strength result.

Four learned Goal-QKV roles query registered capabilities and live ledger state: setup/board
development, attack/Prize progress, resource access/recovery, and tempo/survival. They are learned
query roles, not handwritten action labels.

The official engine supplies legal options. A full action is an ordered sequence of distinct legal
option indices plus STOP. STOP is masked before `minCount`; reaching `maxCount` forces termination
with log-probability and entropy contribution zero. Teacher forcing, greedy selection, stochastic
rollout, and future PPO reevaluation must share one probability implementation. Policy and
player-relative action-before state-value heads share the contextual state; value calibration and
PPO remain future stages.

## Research ladder and evidence

The approved M0–M5 ladder progresses from an 0012-compatible baseline on the new contracts,
through masks and structured semantics, registered-deck summary, Goal-QKV, causal ledger, and
finally relation/event/value-ready contracts. Each variant must be explicit and independently
auditable.

Offline exact-action and legal-action metrics are imitation and contract evidence only. Strategy
strength may be concluded only from formal official-engine evaluation with frozen candidate,
opponent catalog, engine, seat/seed, and metric contracts. No such evaluation exists yet.

## Current and next stage

Current stage: the frozen protocol, eight-day canonical source reader, ordered-action contract,
deterministic group split, atomic BC/RL-compatible shards, structured card semantics, typed input
schema, visibility audit, and causal player-local ledger are implemented. The real source audit
validated 36,700 canonical episode bodies and found 2,110 eligible complete unique Yushin-winner
groups; see [`data_audit/source_audit.json`](data_audit/source_audit.json). No trained checkpoint or
policy-strength claim exists.

Next gate: publish the deterministic train/validation dataset after deriving the complete private
split assignment and corpus distribution/conflict audit, then implement and smoke M0–M5 with the
centralized full-action probability and exact offline/online causal parity. Formal version
allocation, online W&B, long GPU training, export, and official-engine evaluation remain downstream.
