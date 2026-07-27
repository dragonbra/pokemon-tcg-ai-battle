# Decision 003: Starmie–Dusknoir source gap and false archetype label

**Date:** 2026-07-27
**Status:** open data blocker for T2 only; T0/T1 preparation continues

## Evidence

The audited 0726 official archive contains 4,554 Episode JSON files and 9,108 registered player
decks. A registration-frame scan found:

- zero decks containing both Mega Starmie ex (`1031`) and any Duskull-line card
  (`131`, `132`, `133`);
- two Top-100 Mega Starmie sources in the earlier environment snapshot, `Marshall Maximizer` and
  `stardom`, but their exact 60-card lists contain Mega Froslass/Froslass rather than Dusknoir;
- the current `stardom` submission `55013094`, checked through Episode `88361434`, still contains
  Mega Starmie ex ×3 and no Duskull, Dusclops or Dusknoir.

The misleading label comes from
`data/processed/environment_daily/generate_live_snapshot.py::_classify_archetype`: its fixed table
maps every deck containing `Mega Starmie ex` to the display string
`Mega Starmie ex / Dusknoir` without checking for Dusknoir. This display heuristic is not valid
deck evidence and must not be used for training selection.

## Decision

T2 requires registration-frame proof of both the Starmie line and Dusknoir line. Mega
Starmie/Froslass data is not a substitute: Froslass's damage behavior and evolution/KO timing do
not supervise Cursed Blast. T2 and the complete T3 matrix remain pending until a genuine source is
identified. In the meantime, continue all non-destructive shared infrastructure and the T0/T1
data/model smoke work. Do not allocate a formal T2/T3 version with an empty or mislabeled source.

This decision does not modify the historical environment report during the 0015 experiment. A
separate data-report correction can replace the hard-coded label with evidence-based classification
without changing official Episode data.
