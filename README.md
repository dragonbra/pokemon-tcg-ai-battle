# Pokemon TCG AI Battle — Behavior Cloning + Reinforcement Learning

This is the public, single-system view of our Kaggle Pokemon TCG AI Battle project. It contains the semantic full-action policy, behavior-cloned initialization, PPO specialist training, policy/value separation, public-information routing, official-engine evaluation, and final competition packages.

The original research workspace grew through dozens of numbered experiments. That history is intentionally absent from this branch so the code and the write-up describe the same system. The immutable pre-cleanup snapshot is tag `archive/final-competition-repo-2026-09-13`.

## What the model does

The policy compiles an official game observation and its legal actions into three families of tokens:

- static card prototypes: identity plus HP, evolution, attack costs, weaknesses, effects, and related mechanics;
- dynamic state: zones, board entities, damage, attached cards, public knowledge, events, and resources;
- legal options: action type, source, target, card/effect identity, and conditional allocation fields.

A semantic Transformer produces state and option representations. The committed Policy-0814 weights supply the behavior-cloned initial policy. PPO adapts the action decoder, allocation head, final attention/FFN LoRA paths, final Option LayerNorm, and a separately owned critic. The deployable router uses only public opponent Pokemon evidence and never receives a hidden exact-deck label or critic output.

See [architecture](docs/architecture.md), [model design](docs/model/DESIGN.md), and [evidence boundaries](docs/evaluation.md).

## Repository map

```text
src/pokemon_tcg_ai/
  model/              stable public model API
  semantic_runtime/   feature schema, encoders, Transformer, deployment internals
  policy/             Actor/Critic, decoder, LoRA, checkpoint identity
  training/           one canonical Policy-0814 → PPO training path
  rollout/            official-engine CUDA trajectory collection
  inference/          stable portable/routed inference API
  evaluation/         final routing, export, and frozen evaluation contracts
docs/model/           authoritative final-system design and evaluation reports
docs/writeup/         retrospective audit and GitHub write-up plan
docs/rl/              immutable policy/deployment identity protocol
evaluation/           official-engine match and package evaluation runtime
engine/               read-only official CPU engine source/build boundary
engine_cuda_2_0/      CUDA engine used for batched rollout and evaluation
rl_environment/       canonical training metric logging
archive/submission/   final Git-LFS-backed competition packages
```

## Setup and checks

Git LFS is required because policy weights and final packages are intentionally versioned:

```bash
git lfs install
git lfs pull
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[rl]'
PYTHONPATH=src:. pytest -q src/pokemon_tcg_ai/tests
sha256sum -c archive/submission/dist/2026-09-13-final-packages.sha256
```

Inspect the final training contract without starting a run:

```bash
ptcg-train
```

An actual PPO run requires the private competition-derived CUDA rule pack and an explicit launch
flag; see [reproduction](docs/reproducing.md).

The unit suite validates model ownership, policy identity, checkpoint reconstruction, deployment dtype, public routing, and schedule contracts. Match strength claims require real games through the official engine; unit tests and offline imitation scores are supporting evidence only.

## Final artifacts

The seven frozen archives are listed in [archive/submission/README.md](archive/submission/README.md). Git stores each archive as an LFS pointer and the checksum file binds its downloaded content. Package names and schema identifiers retain historical `0045` labels because those strings are part of immutable provenance.

## Evidence and limitations

- The strongest retained 0045 public-router report is 1,382–666–0 (67.48%) over 2,048 common-seed official-engine CUDA games against complete Policy-0809. This was experimental selection-set evidence, not an independent holdout and not an automatic promotion decision.
- Historical Hybrid-0806 results are explicitly invalid as Frozen-0806 evidence.
- The repository does not contain sufficient evidence to claim which late local package was actually submitted to Kaggle; package smoke tests are not leaderboard receipts.
- The official engine, Pokemon card data, imagery links, and competition assets may have redistribution terms distinct from this repository's original code. See [publication checklist](docs/public-release/publication-checklist.md) before making the repository public.

## Status

This branch is a curated technical release, not an active experiment ledger. New work should use ordinary feature branches and human-readable names rather than reintroducing the historical numbered-project layout.
