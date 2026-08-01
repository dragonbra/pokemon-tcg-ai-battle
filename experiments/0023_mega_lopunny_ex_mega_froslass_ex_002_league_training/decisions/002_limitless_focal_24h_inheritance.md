# Decision 002: Limitless Mega Lopunny Focal and 24h Continuation

Date: 2026-08-01

## Decision

The next formal 0023 version is `V7_from_v6_limitless_focal_24h`. The focal policy changes to
`mega_lopunny_ex_001`, the user-provided Limitless champion exact deck with one Abra. Its exact
deck SHA-256 is
`f03203e5bc6fc1cd4c29b4e3e728f360d37abe55aaaf34a73b053085dbe21553`.

The 50-deck League catalog and exact deck identities remain unchanged. The former focal
`mega_lopunny_ex_mega_froslass_ex_002` remains a Live opponent. Every V7 Live decoder/value branch
is inherited from the matching V6
`checkpoint/live/<deck>/update-000013.pt` model-only checkpoint. No branch is initialized from
Foundation. Optimizers are newly initialized for V7, as required for a new training version.

V6 is recorded as a failed source because its official-engine rollout gate raised an error at
update 13. Reusing its model-only outputs is an explicit continuation choice, not a claim that V6
was a successful strength run. V7 must retain the source status and exact checkpoint identities in
`artifact/initialization_audit.json`.

## Runtime Contract

V7 must receive at least 24 GPU hours. The training child and
`monitor_training.py` will run inside one foreground terminal exec session. The main agent retains
the session ID and waits on it repeatedly at intervals below 60 seconds. The monitor's flushed
stdout events are the notification channel; a process exit, failed status, stale metrics/checkpoint,
official-engine error, or resource guard is handled immediately by the main agent. No packaging,
formal evaluation, or other GPU-heavy job is started while V7 is active.

Sampled rollout and Live-pool curves remain diagnostics. Any strength statement requires a later
fixed-seed, balanced-seat official-engine Frozen Arena evaluation under the matching version
contract.
