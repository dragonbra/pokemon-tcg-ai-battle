# Daily Environment Data

`generate_live_snapshot.py` is the canonical, date-parameterized Kaggle Top 100 environment-daily
pipeline. It freezes the live leaderboard once, binds each row by `submissionDate`, fetches bounded
official Episode Meta, downloads the cutoff-latest replay, extracts that submission's own exact
60-card deck, validates all 100 identity chains, renders the fixed 0726/0725 UI contract, and updates
the report index.

Run from the repository root:

```bash
python3 -m data.processed.environment_daily.generate_live_snapshot --date YYYY-MM-DD
```

Raw evidence stays in an immutable `.tmp/environment_daily/<date>/run-*` directory. The formal HTML
is written to `docs/environment-daily_kaggle_top100/daily/YYYY-MM-DD.html`. Existing reports are not
overwritten unless `--overwrite` is explicit. Read the full evidence and UI contract in
`docs/environment-daily_kaggle_top100/README.md`.
