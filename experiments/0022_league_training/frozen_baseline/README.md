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
- execution: 8 workers and 1 CPU thread per worker; newer reports route the same frozen candidate
  policy through one shared `cuda:0` service (`batch_size=16`, `batch_wait_ms=0.5`)
- metric profile: `core`

## Results

| Frozen deck | Device | W-L-D | Win rate | First | Second | Errors | Wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mega Lucario ex / Solrock | CPU | 163-137-0 | 54.33% | 61.33% | 47.33% | 0 | see report |
| Mega Lucario ex / Hariyama | CPU | 135-165-0 | 45.00% | 48.00% | 42.00% | 0 | 188.84 s |
| Mega Lopunny ex / Mega Froslass ex | CPU | 150-150-0 | 50.00% | 57.33% | 42.67% | 0 | 207.57 s |
| Festival Lead / Dipplin 001 | CPU | 230-70-0 | 76.67% | 75.33% | 78.00% | 0 | 186.12 s |
| N's Zoroark ex / N's Zekrom | CPU | 71-229-0 | 23.67% | 29.33% | 18.00% | 0 | 212.65 s |
| Dragapult ex / Mega Froslass ex | shared CUDA | 113-187-0 | 37.67% | 42.00% | 33.33% | 0 | 187.05 s |
| Hydrapple ex / Meganium | shared CUDA | 121-179-0 | 40.33% | 42.67% | 38.00% | 0 | 146.69 s |
| Arboliva ex / Meganium | shared CUDA | 120-180-0 | 40.00% | 43.33% | 36.67% | 0 | 152.86 s |
| Raging Bolt ex / Mega Kangaskhan ex 002 | shared CUDA | 59-241-0 | 19.67% | 20.67% | 18.67% | 0 | 149.64 s |
| Raging Bolt ex / Mega Kangaskhan ex 003 | shared CUDA | 68-232-0 | 22.67% | 24.00% | 21.33% | 0 | 143.87 s |
| Dragapult ex / Crushing Hammer | shared CUDA | 176-124-0 | 58.67% | 63.33% | 54.00% | 0 | 200.85 s |
| Dragapult ex / Dusknoir 002 | shared CUDA | 123-177-0 | 41.00% | 44.67% | 37.33% | 0 | 180.38 s |
| Dragapult ex / Blaziken ex | shared CUDA | 144-156-0 | 48.00% | 52.67% | 43.33% | 0 | 202.14 s |
| Dragapult ex 001 - Limitless | shared CUDA | 172-128-0 | 57.33% | 62.67% | 52.00% | 0 | 205.54 s |
| Mega Kangaskhan ex / Crustle 008 | shared CUDA | 214-86-0 | 71.33% | 70.67% | 72.00% | 0 | 135.64 s |
| Festival Lead / Dipplin 003 | shared CUDA | 169-131-0 | 56.33% | 61.33% | 51.33% | 0 | 141.34 s |
| Dragapult ex / Dudunsparce 002 | shared CUDA | 147-153-0 | 49.00% | 51.33% | 46.67% | 0 | 201.58 s |

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
