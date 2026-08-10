# OFFICIAL CPU SEARCH GROUND TRUTH

Scope: the unmodified official CPU engine under `engine/source/ptcgProgram 22/` only. CUDA engines,
RL/inference wrappers, and production policy behavior are not used as semantic evidence.

## 1. Official implementation

- File: `engine/source/ptcgProgram 22/Search.h`
- Class: `Search`; supporting structs: `SearchStartConfig`, `SearchState`, `SearchInfo`
- Core methods:

```cpp
SearchInfo start(const SearchStartConfig& config, const State& state);
SearchInfo step(long long id, const std::vector<int>& selected);
SearchInfo shuffle(long long id, int playerIndex);  // internal; not exported by Export.cpp
int clearSingle(long long id);
void clear();
```

- Public C ABI in `engine/source/ptcgProgram 22/Export.cpp`:

```cpp
const char8_t* SearchBegin(
    ApiData* data, const char* serialized, int count,
    int* myDeck, int* myPrize, int* enemyDeck, int* enemyPrize,
    int* enemyHand, int* enemyActive, int manualCoin);

const char8_t* SearchStep(
    ApiData* data, long long searchId, int* select, int selectCount);

void SearchEnd(ApiData* data);
void SearchRelease(ApiData* data, long long searchId);
```

The canonical call chain is:

```text
GetBattleData(live Battle ApiData)
  -> copy State
  -> erasePlayerData(selectPlayer)
  -> State::serialize(base64)

AgentStart() -> distinct agent ApiData

SearchBegin(agent ApiData, serialized observation-state, hidden-zone guesses, manualCoin)
  -> SetBattleData -> State::deserialize
  -> construct SearchStartConfig
  -> ApiSearchBegin
  -> Search::start
  -> Search::alloc: *new_state = source_state
  -> fill erased hidden zones in the cloned root

SearchStep(agent ApiData, searchId, selected option indices)
  -> ApiSearchStep
  -> Search::step
  -> Search::alloc: *branch_state = *source_search_state
  -> State::checkPlayerSelect
  -> State::step
  -> repeat State::step while selectMax == 0 and not terminal
  -> ToJsonSearch -> ToJsonApi
```

Source anchors: `Export.cpp:96-159`, `Api.h:99-175`, `Search.h:89-240`,
`State.h:243-315, 1733-1779`, and `ToJson.h:285-324`.

There is no separate official `simulate`, `apply_action`, `clone_and_step`, or `rollout` API in this
source tree. `State::step()` is the generic CPU transition primitive. `ApiSelect()` drives the live
battle state in place; `Search::step()` is the branch-producing search operation discussed here.
`Search::shuffle()` creates another branch and is internal only.

## 2. Input semantics

- State type at the public boundary: compressed base64 bytes from `State::serialize`, passed as
  `const char* serialized, int count`.
- State type internally: `const State&` for `Search::start`; `SearchState` stores `State*`.
- Action type at the public boundary: `int* select, int selectCount`.
- Action type internally: `const std::vector<int>& selected`.
- The integers are indices into the source state's current `State::options`; they are not card IDs
  and there is no separate `Action` object.
- Actor/player: no actor argument is accepted. The actor is `State::selectPlayer`; legality is
  checked by `State::checkPlayerSelect()` against `options`, `selectMin`, and `selectMax`.
- Context: `State::selectType`, `State::selectContext`, `contextCard`, option payloads, and the
  pending `functionStack` are all serialized in the input state.
- RNG: no seed or RNG object is passed to `SearchBegin` or `SearchStep`. `AgentStart()` creates a
  separate `ApiData::game` and seeds its `std::mt19937`; `manualCoin` is the only search randomness
  control in the public signature.
- Hidden information: `GetBattleData()` erases the acting player's hidden deck/prize information
  and the opponent's deck/prize/hand (plus a facedown Active) before serialization. `SearchBegin`
  requires caller-supplied card-ID arrays to fill those erased slots.

Therefore the public search input is **not the live omniscient game state**, and it is more than the
JSON observation. It is an agent-visible serialized `State` plus an explicit caller-provided
determinization of hidden zones. After `Search::start`, the root is again a full internal `State`,
but its hidden contents are the caller's supplied hypothesis, not privileged engine truth.

## 3. Output semantics

The public C ABI returns a UTF-8 JSON buffer (`const char8_t*`), not a `GameState*` or cloned engine:

```json
{
  "state": {
    "observation": {
      "select": {},
      "logs": [],
      "current": {}
    },
    "searchId": 1
  },
  "error": 0
}
```

On error, `state` is `null` and `error` is nonzero. Internally, `SearchInfo` contains:

```cpp
State* state;
int searchId;
int errorCode;
```

The returned `searchId` identifies the newly allocated branch and must be supplied to a later
`SearchStep` or `SearchRelease`. The observation exposes the next selection (`select`), newly
visible logs, and the current public state (`current`). It does not expose the internal `State*`,
the `Game*`, or RNG state. The official Python SDK merely parses this JSON into
`ApiResult -> SearchState(observation, searchId)`; that dataclass is not the engine's ground-truth
return type.

## 4. Mutation semantics

- Original live battle `State` mutated: **NO**.
- Original live battle RNG mutated: **NO**.
- Source search `State` payload mutated by `Search::step`: **NO**.
- Returned branch and source state are the same object: **NO**.
- Search-side `Game`/RNG fully isolated per branch: **NO**.

Mechanism: `Search::alloc` takes a free `State` and executes `*state = src`. This copies the State
payload and creates a new branch object; there is no mutate-and-rollback. However, `State` contains
`Game* game`, so assignment shallow-copies that pointer. The reconstructed agent state, search root,
and every branch share the same agent-side `Game`, including `Game::rng`, `config.manualCoin`, and
scratch containers. `Search::start` itself writes `s.game->config.manualCoin`.

This shared search-side `Game` is distinct from the live battle's `Game`, because the public API
requires an `AgentStart()` pointer and rejects a battle pointer (`apiDataType != 2`). Thus search does
not pollute the running battle, but random operations inside one search tree advance a shared RNG
visible to its other roots/branches. Branching does not clone RNG state.

Runtime evidence from `tests/official_cpu_search_probe.cpp`:

```text
identity ... sameObject:0, sharedGame:1
mutation_after_deterministic_steps {
  originalBefore:c71886b8e813e840, originalAfter:c71886b8e813e840,
  rootBefore:74ebb311a4b9529c, rootAfter:74ebb311a4b9529c,
  rngBefore:85b2546ea309ac46, rngAfter:85b2546ea309ac46
}
rng_shared_state_probe {
  sameGame:1,
  rngBefore:85b2546ea309ac46, rngAfter:fe709a142e130624,
  sourceRootBefore:74ebb311a4b9529c, sourceRootAfter:74ebb311a4b9529c
}
live_battle_isolation {
  sameGame:0,
  stateBefore:883fa7fe6ab1c411, stateAfter:883fa7fe6ab1c411,
  rngBefore:4a0822d1d1d98abf, rngAfter:4a0822d1d1d98abf
}
```

The three deterministic public `SearchStep` traces did not consume RNG. The RNG-change probe calls
the official internal `Search::shuffle` method, which is not public C ABI; it proves the shared
`Game::rng` behavior directly. Any public `SearchStep` transition that reaches engine code using
`state.game->rng` has the same sharing mechanism. Full internal source/root serialization stayed
unchanged, covering hidden-zone payload, actor, legal options, and pending transition state.

## 5. Basic execution boundary

The test fixture is a deterministic, legal 60-card deck and official CPU state at turn 1 Main. The
root has actor 0, Active card 947, no Bench, Basic cards and Energy in hand.

### Test A: play a Basic Pokemon to Bench

```text
S0: Main; actor 0; bench=[]; hand contains card 989
  -> SearchStep(rootId, [Play option])
  -> SelectedMain -> SelectedPlay
  -> MoveCard Hand -> Bench
  -> ToMain -> MainSelect
S1: Main; actor 0; bench=[989]; hand has one fewer card
```

The play and all non-interactive handlers resolve in one `SearchStep`, then search returns at the
next Main decision. The source root remains unchanged.

### Test B: attach an Energy to Active

```text
S0: Main; actor 0; Attach option already encodes Hand index + Active target
  -> SearchStep(playStateId, [Attach-to-Active option])
  -> SelectedMain -> SelectedAttach -> AttachProc
  -> ToMain -> MainSelect
S1: Main; actor 0; attached Energy count +1; energyAttached=true
```

Because a Main `Attach` option already contains both source and target, no extra target decision is
needed for this testcase. Search returns at Main after completing the attachment.

### Test C: retreat with cost 1

```text
S0: Main; actor 0; Active 947; Bench [989]; one Energy attached
  -> SearchStep(attachStateId, [Retreat])
  -> SelectedRetreat -> SelectTrashSinglePokemonEnergy
S1: Energy / DiscardEnergy decision; exactly one legal Energy option

  -> SearchStep(retreatEnergyStateId, [Energy 0])
  -> discard retreat Energy -> SelectSwitchPokemon
S2: Card / Switch decision; exactly one legal Bench target

  -> SearchStep(retreatSwitchStateId, [Card 0])
  -> SelectedSwitchPokemon -> ToMain -> MainSelect
S3: Main; Active 989; Bench [947]; retreated=true
```

Even when a forced decision has exactly one legal option, search does not choose it automatically.
It returns that decision to the caller. It does automatically run function-stack work that requires
no player selection.

**Conclusion:** `Search::step` stops at the next player decision point (or terminal state), after
auto-resolving intervening engine work with no selection. It is neither strictly "apply one semantic
action and return immediately" nor a rollout that automatically advances through later decisions.

## 6. Minimal reproducible testcase

- Driver: `tests/official_cpu_search_probe.py`
- C++ probe: `tests/official_cpu_search_probe.cpp`
- Build output: `.tmp/official_cpu_search_ground_truth/official_cpu_search_probe`

Run:

```bash
python3 tests/official_cpu_search_probe.py
```

Observed result on 2026-08-11:

```text
OFFICIAL_CPU_SEARCH_PROBE_V1
fixture {battleSeed:1, energyId:6, actor:0}
TEST_A ... Play -> Main, Bench +1
TEST_B ... Attach -> Main, energyAttached:1
TEST_C1 ... Retreat -> Energy/DiscardEnergy
TEST_C2 ... Energy -> Card/Switch
TEST_C3 ... Card -> Main, Active/Bench switched
... source State hashes unchanged ...
... deterministic RNG unchanged ...
... search-side shuffle RNG changed ...
... live battle State/RNG unchanged ...
RESULT PASS
```

The probe compiles the unmodified official `Export.cpp` directly, pins only fixture seeds, uses real
official legal options, and asserts every stated transition and isolation property. Pointer values
vary by process. Internal binary hashes are intended for before/after equality within a run, not as
portable cross-build state IDs.

## 7. One-paragraph ground-truth conclusion

The official CPU engine search is a two-stage `SearchBegin` + branchable `SearchStep` API backed by
`Search::start/step`: it accepts an agent-visible serialized `State`, caller-supplied hidden-zone
determinization, and later a list of legal option indices; it returns JSON containing an
agent-visible observation and a new branch `searchId`. It does not mutate the live battle or the
source branch's State payload, because each step copies `State`, but search branches share the
agent-side `Game*` and therefore share RNG rather than cloning it. For simple actions it executes
all non-interactive engine handlers and stops at the next decision point or terminal state; Bench
and direct Attach return to Main, while Retreat stops separately for Energy payment and switch
target selection.
