# Pokémon TCG AI Battle Research

Research-first repository for the Kaggle Pokémon TCG AI Battle Challenge Simulation.

Competition: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview
Simulator docs: https://matsuoinstitute.github.io/cabt/

## Repository boundary

This repository is the research and coordination layer. It stores:

- verified competition facts and simulator notes;
- discussion/code/replay research;
- baseline design and experiment protocols;
- implementation briefs that can be handed to Claude Code on another machine;
- small analysis utilities that do not require GPU hardware.

The implementation/training layer may live on a separate 5080 machine. It should consume the briefs and write back reproducible results, rather than requiring this Mac to have the full GPU stack.

## Initial workflow

1. Read and record official competition surfaces in `notes/competition-facts.md`.
2. Download competition assets locally with the Kaggle CLI after joining the competition.
3. Build a simulator smoke test and a minimal baseline.
4. Read hot/top discussions and public Code notebooks; record claims with URLs and dates in `notes/discussion-digest.md`.
5. Define experiments before optimizing in `experiments/`.
6. Write an implementation brief in `reports/implementation-brief.md` for Claude Code or the GPU machine.
7. Keep submissions manual and explicit; do not automate uploads by default.

## Current status

The repository scaffold is ready. Kaggle authentication, data download, simulator installation, and baseline implementation are intentionally separate steps because they may require accepting competition rules and using competition-only data.
