# Initial Top-100 Alakazam Archive Scan

Date: 2026-07-27

## Frozen cohort

The 2026-07-27 environment snapshot contains 28 Top-100 players whose representative exact deck
contains Alakazam (`card_id=743`). The archive scan uses exact replay `TeamNames` identity and keeps
only player views that are `DONE`, have an exact 60-card registered deck containing Alakazam, and
have own reward greater than zero. Battle Cage is an audit statistic, not an inclusion filter.

## Result

| Source | Winning Alakazam Episodes | Battle Cage | Wondrous Patch |
|---|---:|---:|---:|
| 17 official daily archives, 2026-07-10 through 2026-07-26 | 10,755 | 158 | 65 |
| `goonew` targeted submission `54960905` | 166 | 166 | 166 |
| Episode overlap | 49 | 49 | 49 |
| Deduplicated union | **10,872** | **275** | **182** |

All 10,755 archive matches have distinct Episode IDs across the 17 local zip files. The prefix
schema audit reported zero parse errors. The complete local scan payload, including every Episode
ID, archive membership and player counts, is stored at
`.tmp/0016_source_audit/top100_alakazam_archive_counts.json`. The targeted replay manifest is at
`data/replays/0016_alakazam_multideck_bc/goonew/submission-54960905/manifest.json`; all 166 targeted
files passed team, reward, completion, exact-deck and required-card checks.

## Largest sources

| Frozen rank | Team | Independent wins | Battle Cage | Wondrous Patch |
|---:|---|---:|---:|---:|
| 3 | Yushin Ito | 4,341 | 0 | 0 |
| 52 | 齐乐 | 1,341 | 0 | 0 |
| 47 | bono | 1,215 | 0 | 0 |
| 80 | ei ei ei yikuso | 896 | 18 | 0 |
| 18 | Team Rot-Weiß | 575 | 0 | 0 |
| 69 | matsurih | 548 | 24 | 0 |
| 34 | Benarg | 519 | 0 | 0 |
| 55 | miya | 271 | 0 | 0 |
| 25 | Raja Biswas | 178 | 0 | 0 |
| 62 | haggle | 175 | 45 | 0 |
| 85 | goonew | 65 | 65 | 65 |

The natural pool is highly imbalanced: Yushin Ito alone supplies 39.9% of the deduplicated union
before decision-count weighting, and the targeted overlay increases `goonew` from 65 archive wins
to 182 unique Wondrous Patch wins in the union. Dataset materialization must therefore report both
episode and decision counts per source and deck hash before training allocation.

## Frozen source catalog

The validated union is frozen at
`rl_runs/0016_alakazam_multideck_bc/dataset/source_catalog.json`:

- 25 exact UTF-8 team sources have at least one selected Episode; source ID `0` is reserved for
  neutral inference;
- 9,779 whole Episodes are train and 1,093 are validation;
- split membership is deterministic within exact source x first/second-seat strata, with singleton
  strata kept in train;
- catalog SHA-256 is
  `8da02dd59e3853ce1d0e343ac4a832c89c6e580ff43f78a4eda5d385a6e351d0`;
- source-vocabulary SHA-256 is
  `588c1bc174d6c1f18021e3f0119f952f46ef16f6ac573a6eab4cfecb4b7ddb88`.

All 23 unique card IDs in the frozen deployment deck occur in the selected registered decks.
Battle Cage occurs in 275 Episodes and Wondrous Patch in 182. This proves registered-deck exposure,
not that either card was legal or selected in a particular decision; the decision materializer
reports those counts separately.

The source scan's completion contract is actor-side: the selected player must be `DONE` with a
positive finite reward. An opponent may be `TIMEOUT` with a null reward. Episode `85179355` is the
regression case for this official replay shape and contains aligned actor decisions, so it remains
in the user-confirmed corpus rather than being silently filtered.

## Full-action space audit

`experiments/0016_alakazam_multideck_bc/data_audit/action_space_audit.json` scans every selected
actor decision without a model-side cap. Across 845,405 decisions, the observed maxima are:

| Field | Maximum | Decisions above 16 | Decisions above 32 |
|---|---:|---:|---:|
| selected action length | 26 | 360 | 0 |
| `minCount` | 26 | 360 | 0 |
| `maxCount` | 26 | 360 | 0 |
| legal option count | 50 | 95,601 | 4,802 |

The official API contract is relational (`minCount <= len(action) <= maxCount <= len(option)`) and
does not declare a numeric 64 limit. The model/cache horizon is 64 because it exceeds the observed
current environment and the standard 60-card physical card-selection envelope; it is not described
as an engine constant. Runtime retains a capacity-independent legal fallback for future drift.
