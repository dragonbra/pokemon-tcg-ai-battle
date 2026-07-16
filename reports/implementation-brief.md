# Implementation brief

Status: research scaffold; implementation not started.

## Objective

Build a locally runnable baseline agent for the Kaggle Pokémon TCG AI Battle Challenge Simulation using the official simulator/SDK. The first milestone is correctness and reproducibility, not leaderboard optimization.

## Required deliverables

- official simulator dependency setup;
- minimal agent entrypoint in the required submission layout;
- legal-action handling for every observed decision type;
- deterministic seed/configuration where supported;
- local smoke test that completes at least one battle;
- JSON battle/replay output for debugging;
- packaging script that lists the archive contents;
- baseline report with command lines and results.

## Research inputs

- `notes/competition-facts.md`
- `notes/working-agreement.md`
- official docs: https://matsuoinstitute.github.io/cabt/
- competition pages listed in `notes/competition-facts.md`

## Acceptance tests

1. Import the SDK from a clean environment.
2. Start a local battle with the provided sample or minimal legal deck.
3. Return a legal action whenever `select` is non-empty.
4. Handle initialization/deck-selection states.
5. Complete a battle without an exception.
6. Emit a replay/debug JSON artifact.
7. Produce the exact archive layout expected by the competition.

## Not in scope for milestone 1

- GPU training;
- large-scale self-play;
- automatic Kaggle submission;
- claims about leaderboard strength;
- assuming official TCG rules where the simulator differs.

## Handoff notes

Claude Code may implement this on the 5080 machine. It should read this brief, update it with actual SDK/version findings, and return exact test output and artifact paths. If the SDK cannot be installed or data access is blocked, report the blocker instead of substituting fabricated results.
