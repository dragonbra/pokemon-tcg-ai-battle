# Frozen Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an immutable 48-deck 0019 Foundation pool under `evaluation/arena/frozen/` the default checkpoint-strength evaluation surface, with candidate and Frozen policy decisions served by persistent batched GPU processes while the official engine remains authoritative.

**Architecture:** `evaluation/arena/frozen/` stores one audited Foundation policy package plus 48 lightweight exact-deck identities; `evaluation/configs/frozen.json` binds names, display metadata, representative cards, and paths. The existing batch evaluator starts one candidate policy server and one Frozen Foundation policy server, sends each game session's exact deck through IPC, and leaves engine workers Torch-light. The legacy heterogeneous `arena/opponents/` pool remains available through an explicit `--pool opponents` compatibility choice.

**Tech Stack:** Python 3.11, PyTorch/CUDA, AF_UNIX multiprocessing connections, official engine runtime, unittest, repository HTML reporting.

## Global Constraints

- Never modify `engine/source/`; official engine runtime is the only strength evidence.
- Frozen Foundation identity is 0019 Epoch 13, SHA-256 `da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb`, neutral `source_id=0`.
- The pool contains exactly 48 unique exact 60-card decks and one physically stored Foundation policy.
- Frozen evaluation is greedy, fixed-catalog, equal-games-per-deck, and balanced by seat for an even game count.
- GPU policy serving must not import PyTorch inside engine worker processes.
- Existing `evaluation/arena/opponents/` remains unchanged and explicitly selectable.
- Candidate checkpoint strength comes from Frozen Arena results; sampled rollout and PPO loss remain diagnostics.

---

### Task 1: Frozen asset and catalog contract

**Files:**
- Create: `evaluation/configs/frozen.json`
- Create: `evaluation/arena/frozen/manifest.json`
- Create: `evaluation/arena/frozen/_policy/**`
- Create: `evaluation/arena/frozen/<deck_id>/manifest.json`
- Create: `evaluation/arena/frozen/<deck_id>/deck.csv`
- Test: `tests/test_evaluation_frozen_assets.py`

**Interfaces:**
- Consumes: the 48 validated plugins returned by `train.0022_league_training.decks.load_deck_plugins` at asset-generation time.
- Produces: an immutable on-disk catalog whose deck directories require no model copy and whose `_policy` package exposes the shared inference contract.

- [ ] Write failing asset tests requiring exactly 48 enabled unique deck identities, 60 official card IDs per deck, immutable Foundation SHA/source ID, and one `_policy/strategy/model.bin`.
- [ ] Run `python3 -m unittest -v tests.test_evaluation_frozen_assets` and verify the missing frozen catalog fails.
- [ ] Materialize the 48 deck identities and one update-0 Foundation policy package without symlinks or cache files.
- [ ] Run the asset test and `python3 -m evaluation --pool frozen list-opponents` until both pass.

### Task 2: Pool-aware catalog loading and default CLI

**Files:**
- Modify: `evaluation/cli.py`
- Create: `evaluation/frozen.py`
- Modify: `tests/test_evaluation_catalog.py`

**Interfaces:**
- Produces: `load_frozen_catalog(path: Path, evaluation_root: Path) -> list[SubmissionPackage]` and CLI `--pool {frozen,opponents}` with `frozen` as default.
- Consumes: the frozen catalog from Task 1 and the existing legacy catalog unchanged.

- [ ] Add failing tests that default listing resolves 48 Frozen identities and `--pool opponents` resolves the legacy catalog.
- [ ] Add strict frozen-path validation limited to `evaluation/arena/frozen/<deck_id>` and reject `_policy`, candidates, opponents, path escapes, duplicate deck hashes, and wrong Foundation identity.
- [ ] Route `run` and `list-opponents` through the selected pool; record pool ID and catalog hash in the run manifest.
- [ ] Run targeted catalog tests and preserve all legacy catalog tests under explicit opponent-pool selection.

### Task 3: Deck-routed persistent policy server

**Files:**
- Modify: `evaluation/runner/inference_server.py`
- Modify: `tests/test_evaluation_inference_server.py`

**Interfaces:**
- Produces: IPC requests shaped as `{"observation": object, "deck": [60 card IDs]}`; a session encoder is rebuilt when actor or deck changes.
- Preserves: existing requests without `deck` fall back to the policy package deck.

- [ ] Add failing tests for exact 60-card request deck validation and session deck routing.
- [ ] Extend `_InferenceRequest` and `PolicyServer.call` to carry a deck tuple.
- [ ] Return the supplied deck for initialization observations and construct `OnlineCausalEncoder(actor, deck, config)` for decisions.
- [ ] Reject malformed, nonpositive, or non-60-card IPC decks and run inference-server tests.

### Task 4: Two-sided GPU batch execution

**Files:**
- Modify: `evaluation/runner/batch.py`
- Modify: `evaluation/runner/worker.py`
- Modify: `evaluation/runner/models.py` only if typed pool provenance is needed.
- Test: `tests/test_evaluation_targeted.py`
- Test: `tests/test_evaluation_inference_server.py`

**Interfaces:**
- Adds to `BatchConfig`: `opponent_inference_root: Path | None`, `opponent_inference_device: str | None`, and shared batch settings.
- Worker environment: `EVALUATION_CANDIDATE_INFERENCE_SOCKET` and `EVALUATION_OPPONENT_INFERENCE_SOCKET`.

- [ ] Add failing worker tests proving both candidate and opponent select through remote sockets and receive their own exact decks.
- [ ] Generalize server lifecycle code to start two independently audited persistent policy services.
- [ ] Pass both sockets to workers; remote opponent use is mandatory for the Frozen pool and legacy opponents retain local package execution.
- [ ] Record both inference devices, policy package hashes, batch size, wait time, workers, and CPU thread count in the report manifest.
- [ ] Run batch/worker/inference tests.

### Task 5: Documentation and default evaluation contract

**Files:**
- Modify: `evaluation/README.md`
- Modify: `experiments/0022_league_training/DESIGN.md`
- Modify: `experiments/0022_league_training/DESIGN.html`

**Interfaces:**
- Documents: default standard gate `48×10=480`, fast diagnostic `48×2=96`, and explicit legacy pool invocation.

- [ ] Update the evaluation README with Frozen-default and `--pool opponents` commands.
- [ ] Update both 0022 authority documents with V11 user interruption at update 81, focal-only primary optimization, side-deck opportunistic updates, champion/challenger promotion, and Frozen Arena priority.
- [ ] Cross-check hashes, deck count, paths, devices, and evidence boundaries against current assets and code.

### Task 6: Verification and official-engine smoke

**Files:**
- Output: `.tmp/evaluation/0022_frozen_arena_smoke/<run-id>/report.html`

**Interfaces:**
- Runs the normal evaluation CLI against the new Frozen pool, not a mock or custom engine.

- [ ] Run all evaluation and 0022 unit tests plus `git diff --check`.
- [ ] Run a balanced two-seat Frozen GPU smoke against at least two frozen deck identities and require all games finished with zero errors.
- [ ] Verify `nvidia-smi` shows both resident policy server processes during the smoke and engine workers do not import Torch.
- [ ] Inspect the HTML manifest for Frozen catalog/policy hashes and two-sided GPU provenance.
- [ ] Commit only implementation, tests, immutable Frozen assets, plan, and synchronized authority docs.
