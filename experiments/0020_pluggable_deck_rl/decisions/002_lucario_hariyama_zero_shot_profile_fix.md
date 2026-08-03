# Lucario/Hariyama zero-shot metric profile correction

Date: 2026-08-03

## Decision

Retain `V17_lucario_hariyama_zero_shot_frozen51` as an immutable historical evaluation and run a new version, `V18_lucario_hariyama_zero_shot_profile_fix_frozen51`, after registering the candidate with the league-quality metric profile.

## Evidence

- V17 completed all 510 official-engine games with 210 wins, 300 losses, no engine errors, and no unfinished games.
- The outcome metric is valid, but `league_quality` failed in all 510 games because `0024_lucario_hariyama_zero_shot` was absent from the static profile registry.
- The registry audit also found two enabled Frozen-51 identities without explicit coverage: `mega_lopunny_ex_mega_froslass_ex_002` and `rmy_teal_mask_ogerpon_001`.

## Controlled variable

V18 changes only metric-profile routing. It uses the same candidate package, deck hash, foundation checkpoint, source ID, training-update count, official engine runtime, Frozen-51 catalog, games-per-opponent count, and seat-balancing contract as V17.

Because games are stochastic, any V17/V18 win-rate difference is sampling variance and must not be interpreted as a policy-strength change. V18 is authoritative for process diagnostics; both reports remain valid records of their own official-engine outcomes.
