# Frozen baseline

This directory records the official-engine starting strength of exact decks under the frozen
0019 Universal Winner BC foundation policy. These reports are static evaluation evidence; they
are not League training results and do not authorize a deck to enter the League catalog.

## Contract

- checkpoint: 0019 Universal Winner BC, epoch 13
- model weight SHA-256:
  `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`
- source conditioning: neutral `source_id=0`
- runtime: official engine
- opponents: the 30 enabled Arena packages at evaluation time
- games: 10 per opponent, 300 per deck, balanced first/second seat
- execution: 8 workers, 1 CPU inference thread per worker
- metric profile: `core`

## Results

| Frozen deck | W-L-D | Win rate | First | Second | Errors | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mega Lucario ex / Solrock | 163-137-0 | 54.33% | 61.33% | 47.33% | 0 | see report |
| Mega Lucario ex / Hariyama | 135-165-0 | 45.00% | 48.00% | 42.00% | 0 | 188.84 s |
| Mega Lopunny ex / Mega Froslass ex | 150-150-0 | 50.00% | 57.33% | 42.67% | 0 | 207.57 s |
| Festival Lead / Dipplin | 230-70-0 | 76.67% | 75.33% | 78.00% | 0 | 186.12 s |
| N's Zoroark ex / N's Zekrom | 71-229-0 | 23.67% | 29.33% | 18.00% | 0 | 212.65 s |

`mega_lucario_ex_solrock.html` is a physical copy of the existing 0020 report because the supplied
`mega_lucario_ex.csv` has the exact same sorted 60-card multiset as the previously evaluated 0020
candidate. The other three reports were produced specifically for this baseline.

The source file named `N's_zoroark.csv` contains no Zoroark cards. Its actual 60-card composition
is Mega Lucario ex / Hariyama, so the report and baseline identity use the card contents rather
than the source filename.

See `manifest.json` for immutable deck hashes, run IDs, and original report provenance.

## Imported 0020 reports

`imported_0020/` contains physical copies of the six requested 0020 reports whose exact 60-card
multisets match a current 0022 deck. Five are neutral zero-shot baselines. V10 uses the same
Raging Bolt deck as V8 but is a source-98 persona POC, so it is retained as a labeled comparison
and is not treated as the Frozen neutral baseline.

The four requested Dragapult reports (V2, V11, V14, and V15) all use the same historical exact
deck, which does not match any current 0022 Dragapult deck. They were deliberately not copied.
See `matched_0020_reports.json` for the complete match audit.
