# Frozen65 CUDA 2.0 server parity — 2026-08-12

## Scope

- Frozen Pool: `0812_top100_top500_stadiums_mill_plus_limitless_dragapult_v3`
- Exact decks: 65
- Pair matrix: every unordered pair, 50 deterministic coverage-policy seeds
- Expected coverage: 2,080 pairs and 104,000 battles
- Oracle: unmodified official CPU engine
- Compared implementations: official CPU, host POD, and CUDA POD
- Server GPU: NVIDIA GeForce RTX 4080 SUPER, compiled for SM 89
- Rules: private official `3aaeaa92` rule pack; the binary rule pack is not committed
- Case-list SHA-256: `49fb59d2bd17f3502730b3a887f5b26684c3ca9d6af38eacf81024a92fd58831`

The matrix uses a 512-decision bound. A legal coverage policy can revisit
reversible actions indefinitely, so a battle reaching the bound is reported as
unfinished. All decisions before the bound are still compared. This is finite
trajectory evidence and does not claim exhaustive proof over every legal action
sequence.

## Top500 Stadium and Mill additions

The committed audit `top500_stadium_coverage_audit.json` enumerates 20 distinct
Stadium cards in the supplied 2026-08-09 Top500 snapshot. The prior 62-deck pool
covered 18. The two missing cards and one explicit mill deck were added as exact
Top500 lists:

- 063, rank 170 Mega Lucario ex / Solrock: Gravity Mountain (`card1252`) x2
- 064, rank 399 Mega Lopunny ex / Mega Froslass ex: Lumiose City (`card1267`) x1
- 065, rank 360 Crustle / Great Tusk Mill: Great Tusk (`card58`) x4 and
  Neutralization Zone

After the additions, all 20/20 Top500 Stadium IDs are represented.

The three new decks were also tested pairwise for 50 seeds each. Across 150
battles and 28,783 compared decisions, CPU/POD/CUDA had zero state, status, or
outcome mismatch. All 150 battles finished naturally.

Official semantic traces prove that the target cards were exercised:

- Gravity Mountain was observed in 50/50 seeds, including Stadium-to-trash
  replacement events.
- Lumiose City was observed in 44/50 seeds; its Stadium Ability was selected
  172 times across 31 seeds.
- Great Tusk was observed in 50/50 seeds, including 128 attack events.

The machine-readable targeted result is
`frozen65_targeted_semantic_20260812.json`.

## Full matrix result

The clean matrix passed all 2,080/2,080 pairs and all 104,000 battles:

- decisions compared: 20,558,750
- state, status, canonical-byte, and outcome mismatches: 0
- player-0 wins: 49,013
- player-1 wins: 53,042
- official draws: 143
- unfinished at the 512-decision coverage bound: 1,802
- missing/unexpected/bad pair ordinals: 0/0/0

Wins, losses, draws, and unfinished battles sum to exactly 104,000. The
machine-readable aggregate is `frozen65_server_matrix_20260812.json`.

## Policy-0809 Docker checks

The 0809 portable checkpoint was loaded with FP16 storage and FP32 runtime in
the CUDA 13 PyTorch Docker image. Decks 063, 064, and 065 each completed a
32-game smoke: 96 games total, with candidate audit PASS, opponent audit PASS,
zero inference fallback, and zero engine error.

The focused Docker regression suite passed 12/12 tests, including the 65-deck
catalog/evaluator contract, enemy second-effect selection, and binary
provenance. The model was not retrained in this change; these checks establish
that the existing 0809 model and CUDA runtime can load and play the expanded
catalog.
