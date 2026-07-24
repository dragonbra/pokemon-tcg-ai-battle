# Arena Combat Matrix

This directory is the durable audit package for the current formal opponent pool.

- `index.html` is the generated matrix dashboard.
- `matrix.json` is the generated aggregate payload embedded by the dashboard.
- `reports/<package>/<run_id>/report.html` retains each package's full official-engine
  evaluation report against the complete enabled catalog.

Every directed package matchup contains 10 games, including self-play. A complete round
combines the first-player phase and following second-player phase and is calculated as
`ceil(engine_turn / 2)`. Action-selection counts are not used as game length.
