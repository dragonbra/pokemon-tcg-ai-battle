# Competition facts

Source of truth is the official Kaggle competition pages and the official cabt documentation. External card databases are only for interpretation; simulator assets and behavior decide legality.

## Verified from the public pages

- Competition type: Simulation Competition.
- Submission: executable agent, not a static prediction CSV.
- Official SDK/simulator is provided for local debugging and reinforcement learning.
- Observation exposes logs, current state, and selectable actions. The opponent's hand is hidden except for its count.
- Data includes English/Japanese card metadata and reference PDFs.
- Public simulator documentation: https://matsuoinstitute.github.io/cabt/
- Kaggle simulation CLI documentation: https://github.com/Kaggle/kaggle-cli/blob/main/docs/simulation_competitions.md
- Public Code page currently includes RL/MCTS sample code, probabilistic agents, replay analysis, and custom vectorized-engine work.
- Public Discussion page currently includes simulator-rule differences, 30,000-game analysis, RL journeys, heuristic-vs-RL comparisons, and first/second-player analysis.

## Evaluation facts to re-check before every submission

The competition uses agent episodes and an estimated skill rating. New submissions undergo a validation episode, then enter the pool. The competition page states daily submission limits and final-submission rules; do not rely on stale notes when planning uploads.

## Research caveat

The competition card pool is simulator-specific and should be treated as a closed world. Do not infer legality from the complete real-world Pokémon TCG card pool or Standard format.

## Retrieval URLs

- Overview: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview
- Data: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/data
- Code: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/code
- Discussion: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion
- Rules: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules
