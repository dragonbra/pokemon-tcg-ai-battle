# Working agreement

## Research vs implementation

Research on this Mac should answer:

- What does the official competition actually evaluate?
- What are the simulator observation/action semantics?
- Which rules differ from real Pokémon TCG rules?
- What do high-signal discussions and public notebooks establish?
- Which baseline and experiments are worth implementing?

Implementation on a GPU machine should answer:

- Does the code run locally against the official SDK?
- Does it reproduce the stated baseline metrics?
- Does a change improve performance under a fixed evaluation protocol?
- Is it packaged in the exact submission format?

## Handoff contract

Every handoff to Claude Code should include:

- objective;
- relevant files and URLs;
- environment assumptions;
- exact acceptance tests;
- compute budget;
- expected artifacts;
- open questions and known simulator caveats.

Claude Code should return:

- changed files or commit/hash;
- commands actually run;
- test and benchmark output;
- unresolved issues;
- artifacts and their paths.

## Hardware policy

Do not assume a GPU is required until a stable simulator baseline exists. Start with rule-based/probabilistic baselines and replay analysis. Introduce vectorized simulation, MCTS, self-play, imitation learning, or RL only when an experiment document justifies it.

## Submission policy

Downloading, publishing, and submitting are account-visible or competition-sensitive actions. Use dry-run/read-only operations first. Upload only after explicit confirmation in the active conversation.
