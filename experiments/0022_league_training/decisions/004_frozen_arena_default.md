# 004 Frozen Arena becomes the default strength gate

Date: 2026-07-31

## Decision

V11 stopped cleanly after complete update 81 at the user's request. Its last rollout used source
policy update 80, finished 512/512 official-engine games with zero rollout errors, and produced
trajectory for all 48 Live decks.

Future training is focal-primary: one deck receives the main sampling and optimization budget;
the other 47 Live decoders may update opportunistically only from their own real actor trajectory.
Training curves and Live cross-play remain diagnostics.

The default checkpoint acceptance gate is the immutable 48-deck Frozen Arena at
`evaluation/arena/frozen/`. It binds exact deck identities to the 0019 Epoch 13 Foundation policy,
SHA-256 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`, with deployment
`source_id=0`. A 96-game balanced-seat pass is diagnostic; standard acceptance is 48 × 10 = 480
official-engine games, five games in each seat per deck.

Each deck maintains immutable `foundation`, mutable `latest/challenger`, and accepted `champion`
identities. Latest never replaces champion without a Frozen Arena gate.

Evaluation uses two persistent batched GPU services: one for the candidate and one shared by all
48 Frozen deck identities. The official engine remains isolated in CPU workers. The legacy
heterogeneous `evaluation/arena/opponents/` pool remains available through `--pool opponents` as a
secondary external-generalization test.
