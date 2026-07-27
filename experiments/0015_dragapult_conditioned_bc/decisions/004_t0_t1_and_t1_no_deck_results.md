# Decision 004: T0, T1 and exploratory T1-no-deck results

Date: 2026-07-27

## Decision

Keep V2 T1 as the current strongest 0015 frozen candidate. Treat adding all audited pure
Dragapult/Dragapult+Blaziken demonstrations as a successful practical data intervention. Preserve
the exact registered-deck input in the deployed policy.

Do not allocate T2, T3 or T3-no-deck until a genuine Starmie + Dusknoir registered deck is
identified. V3 is an exploratory T1-no-deck diagnostic and does not fill any of those matrix cells.

## Evidence

All versions use the same R15 architecture, source conditioning, seed, target validation, target
THIRD PTCG Club persona/deck, best-target-greedy-exact checkpoint rule and 20-opponent official
engine catalog. Each formal evaluation ran 10 games per opponent (200 total), with 100% completion
and zero errors.

| Version | Train decisions | Best target exact | Epoch | Formal W/L | Win rate |
|---|---:|---:|---:|---:|---:|
| V1 T0 | 12,988 | 54.3% | 11 | 15 / 185 | 7.5% |
| V2 T1 | 54,764 | 58.7% | 10 | 39 / 161 | 19.5% |
| V3 T1-no-deck | 54,764 | 60.1% | 11 | 30 / 170 | 15.0% |

The independent temporary V1/V2 engine runs were directionally consistent at 7.0% and 20.0%.
This does not turn the formal runs into paired deterministic trials, but it reduces concern that
the headline ordering came from one random batch.

The formal V2 gain is not a seat artifact: V1 won 9/100 first and 6/100 second, while V2 won
20/100 first and 19/100 second. Wilson 95% intervals for the individual formal win rates are
V1 `[4.6%, 12.0%]`, V2 `[14.6%, 25.5%]`, and V3 `[10.7%, 20.6%]`. An unpaired normal interval for
the V2−V1 difference is approximately `[+5.4pp, +18.6pp]`. Treat the V3−V2 difference as
directional rather than conclusive: its corresponding interval crosses zero.

## Interpretation

T0 to T1 improves both target imitation (+4.4 percentage points) and official-engine strength
(+12.0 percentage points). The added data therefore did not merely improve auxiliary-source
labels; it transferred to the one deployed target deck.

V3 separates the two endpoints. Removing initial-deck information raises target exact action by
1.4 percentage points relative to V2 but lowers engine win rate by 4.5 percentage points. Initial
deck conditioning appears useful to actual play even when it makes literal teacher matching
slightly harder. Offline exact action must remain a checkpoint-selection and diagnostic measure,
not a standalone strength ranking.

## Provenance

- V1 report: `evaluation/V1_t0_target_only.html`, run
  `run-5078c2f1ba6e4a48816f8611299a1c5a`.
- V2 report: `evaluation/V2_t1_plus_pure_dragapult.html`, run
  `run-feac4e0403134af997755c575ab7d963`.
- V3 report: `evaluation/V3_t1_no_deck_exploratory.html`, run
  `run-961a2a9e01884116a8a789ad95869f11`.
- W&B URLs and stable run IDs are recorded in each version's `artifact/status.json`.
