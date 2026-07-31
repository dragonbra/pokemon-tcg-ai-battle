# Daily Deck Catalog

The Kaggle section of this catalog is generated from the frozen 2026-07-30 daily report. Its 32 unique full `deck_sha256` values account for all 100 audited leaderboard rows. Kaggle rank, score, usage count, W-L-D, and effective win rate apply only to those 32 decks.

The catalog also contains sixteen separately identified external references:

- `dragapult_ex_001`: Andrew Hedrick's Los Angeles-winning Limitless list. The exact list includes a 1-1 Dunsparce/Dudunsparce tech line.
- `ionos_bellibolt_ex_kilowattrel_01`: an exact copy of the enabled official-engine arena opponent, useful as a stable benchmark target.
- `mega_lucario_ex_solrock_002`: the user-provided `mega_lucario_ex.csv`, numbered after the Kaggle `_001` variant.
- `ns_zoroark_ex_001`: the corrected user-provided N's Zoroark ex / N's Zekrom list, fully mapped to the current official engine card pool.
- `dragapult_ex_mega_froslass_ex_001`: the user-provided Dragapult ex / Mega Froslass ex ASC list, mapped to canonical engine Card IDs.
- `hydrapple_ex_meganium_001`: the user-provided Hydrapple ex / Meganium / Teal Mask Ogerpon ex list.
- `arboliva_ex_meganium_001`: the user-provided Arboliva ex / Meganium / Teal Mask Ogerpon ex list, preserving the ASC Chikorita identity.
- `raging_bolt_ex_mega_kangaskhan_ex_002`: the user-provided Raging Bolt list with a four-copy Mega Kangaskhan ex line.
- `raging_bolt_ex_mega_kangaskhan_ex_003`: the user-provided three-copy Mega Kangaskhan ex Raging Bolt variant with Passimian.
- `dragapult_ex_crushing_hammer_001`: the user-provided pure Dragapult ex disruption list with Crushing Hammer and Risky Ruins.
- `dragapult_ex_dusknoir_002`: the user-provided Dragapult ex / Dusknoir variant with Bloodmoon Ursaluna ex and Risky Ruins.
- `dragapult_ex_blaziken_ex_001`: the user-provided Dragapult ex / Blaziken ex Rare Candy variant.
- `dragapult_ex_dunsparce_002`: the user-provided Dragapult ex / Dudunsparce variant including Dudunsparce ex.
- `marnies_grimmsnarl_ex_froslass_limitless`: the user-designated Limitless Marnie's Grimmsnarl ex / Froslass list.
- `mega_kangaskhan_ex_crustle_008`: the user-provided control list with Pokemon Center Lady replaced by a second Xerosic's Machinations because the former is absent from the official engine card catalog.
- `festival_lead_dipplin_003`: the user-provided Festival Grounds list with the Grookey-Thwackey engine and Rabsca line.

External references do not increase `unique_full_hashes` and are not assigned Kaggle snapshot rank, score, usage, or win-rate values. Every `deck.csv` contains exactly 60 engine Card IDs. Use `index.html` for readable card names, images, provenance, leaderboard evidence where available, and aggregate records.

Refresh in place with `python3 train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/deck/build_catalog.py --report docs/environment-daily_kaggle_top100/daily/2026-07-30.html --output train/0023_mega_lopunny_ex_mega_froslass_ex_002_league_training/deck --refresh`.


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
  "schema_version": "0023_league_deck_plugin_v1",
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
python3 -m train.0023_mega_lopunny_ex_mega_froslass_ex_002_league_training validate-decks
```
