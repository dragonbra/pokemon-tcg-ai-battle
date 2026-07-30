# Daily Deck Catalog

The Kaggle section of this catalog is generated from the frozen 2026-07-30 daily report. Its 32 unique full `deck_sha256` values account for all 100 audited leaderboard rows. Kaggle rank, score, usage count, W-L-D, and effective win rate apply only to those 32 decks.

The catalog also contains four separately identified external references:

- `dragapult_ex_limitless`: Andrew Hedrick's Los Angeles-winning Limitless list. The exact list includes a 1-1 Dunsparce/Dudunsparce tech line.
- `ionos_bellibolt_ex_kilowattrel_01`: an exact copy of the enabled official-engine arena opponent, useful as a stable benchmark target.
- `mega_lucario_ex_solrock_002`: the user-provided `mega_lucario_ex.csv`, numbered after the Kaggle `_001` variant.
- `ns_zoroark_ex_001`: the corrected user-provided N's Zoroark ex / N's Zekrom list, fully mapped to the current official engine card pool.

External references do not increase `unique_full_hashes` and are not assigned Kaggle snapshot rank, score, usage, or win-rate values. Every `deck.csv` contains exactly 60 engine Card IDs. Use `index.html` for readable card names, images, provenance, leaderboard evidence where available, and aggregate records.

Refresh in place with `python3 train/0022_league_training/deck/build_catalog.py --report docs/environment-daily_kaggle_top100/daily/2026-07-30.html --output train/0022_league_training/deck --refresh`.


# League deck plugin staging

One immediate child directory is one exact-deck plugin. The staging root may remain empty while
the Foundation is being verified, but a League version cannot be initialized until at least one
plugin has `"role": "live"` and `"focal": true`.

```text
deck/
  dragapult_dusknoir/
    manifest.json
    deck.csv
```

`deck.csv` contains exactly 60 lines. Every line is one positive integer card ID, including
duplicate copies. Do not add a header, blank lines, comments, counts, or card names.

Complete `manifest.json` example:

```json
{
  "schema_version": "0022_league_deck_plugin_v1",
  "deck_id": "dragapult_dusknoir",
  "display_name": "Dragapult ex + Dusknoir",
  "role": "live",
  "focal": true,
  "decoder_ref": "foundation",
  "decoder_sha256": null,
  "provenance": {
    "source": "Kaggle online environment",
    "evidence": "submission and Episode identity reference",
    "captured_at": "2026-07-31"
  }
}
```

`decoder_ref: "foundation"` means zero-shot initialization. A baseline Frozen entry creates no
duplicate weight file. A promoted Frozen entry uses a repository-relative immutable `.pt` path
and must provide its 64-character lowercase `decoder_sha256`.

Validate staging without allocating a run:

```bash
python3 -m train.0022_league_training validate-decks
```
