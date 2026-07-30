# Daily Deck Catalog

This catalog is generated from the frozen 2026-07-30 daily report. Each directory is one unique full `deck_sha256`; `deck.csv` contains exactly 60 engine Card IDs. Use `index.html` for readable card names, images, leaderboard evidence, and aggregate records.

Regenerate with `python3 deck/build_catalog.py --report docs/environment-daily_kaggle_top100/daily/2026-07-30.html --output deck`.


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
