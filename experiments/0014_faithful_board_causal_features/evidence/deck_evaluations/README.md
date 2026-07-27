# 0014 archived V12 R2 deck evaluation evidence

This directory preserves the 2026-07-28 local official-engine comparison requested for
`archive/submission/0014_faithful_board_causal_features_v12_r2_epoch8_loss_best`.

Both runs use the same archived model weights, the same BC Marnie opponent, 100 games,
eight isolated workers with one CPU thread each, and alternating candidate seats (50 first,
50 second). The only package change is `2 Nighttime Mine (1266) -> 2 Battle Cage (1264)`.

| Deck | Overall | First | Second | Errors |
| --- | ---: | ---: | ---: | ---: |
| Ito original | 37/100 | 22/50 | 15/50 | 0 |
| Ito + 2 Battle Cage | 41/100 | 26/50 | 15/50 | 0 |

Observed difference: +4 percentage points overall. The native official runtime does not expose
a caller-supplied shuffle seed, so the runs share game IDs and seat allocation but are not
claimed to share identical per-game random streams. Treat this as one 100-game estimate rather
than a paired deterministic test.

See `manifest.json` for immutable hashes, run IDs, commands, and report names.
