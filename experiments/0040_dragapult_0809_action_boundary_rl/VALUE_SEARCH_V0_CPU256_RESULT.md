# 0040 U270 VALUE SEARCH V0 — CPU256 RESULT

## 1. Exact assets

- U270 policy/package: `.tmp/evaluation/0040_u270_value_search_v0/package`
- RL checkpoint: `rl_runs/0040_dragapult_0809_action_boundary_rl/versions/V2_snapshot_loader_fix_long_run/checkpoint/update-000270.pt`
- Update: `270`
- RL checkpoint SHA-256: `c87bcf82ca2b22eed65876da7d0ec0648a85d72d40dd13dd614290e7de002142`
- Portable model SHA-256: `4e729ea30287a40d223e9696ee5303ab80b9c88a6c620f8534f8627011d1bd2e`
- Effective deployment SHA-256: `8d973e185780d873214dcbb61ae258a3dbb2d12680b5eabea29ca2687c456a0a`
- Value checkpoint: the matching `value_head.*` tensors inside the same U270 RL checkpoint
- Value-head tensor SHA-256: `4e6d97266ffadc85867988aaf1e9995a3aaff707031621e763870c975007abe7`
- Featurizer/compiler SHA-256: `03c97b5d87919a64152c066ed5df9bf5f7e91fd018ad423fc74236317c1c372e`
- Causal runtime/state SHA-256: `50fd856e84ed32995d55d99aa985447592162db15b5a203ff2ca150391a8c2a8` / `93f65da8c744ad7e3487646b8915efc7fba489c530e06d6e0c8241740d7346f3`
- Prototype SHA-256: `7228b2612da118e1ec353c7a2e08d4a306de0d4128f0c9e68b4075f2c6b54f73`
- Package manifest SHA-256: `d8f7ed4864d17976c52743b9335191714686d8c575d7cf44306544dfa74fc6e5`
- CPU256 protocol: Frozen-0806 agent-choice v3, evaluation seed `341512806`, one canonical 256-game frequency unit, 16 isolated official-CPU workers, shared CUDA inference servers
- Opponent: full `Policy-0806`, checkpoint SHA-256 `0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8`, effective SHA-256 `94a9aff63de8e2e728413e84d0647c1409326b2e5e41876a5df9062107dd2d5b`
- Opponent pool: `0806_kaggle_top100_plus_v1`
- Schedule ID: `1b09d2fcdf55be55d2c386f8776df8ee0963ade116f9e190f7f3e0fe618fafb4`
- Baseline run: `run-6570434814e543f3bcb8a0ce442589e9`
- Search run: `run-a0a7a96b6b7141e8a5d0fe9871dfb8ea`
- Paired manifest: exact match over game ID, opponent, engine/policy/Search seeds, replica/slot, toss result and actual first player

Commands:

```bash
python3 -m train.0040_dragapult_0809_action_boundary_rl.value_search_v0.run_cpu256 \
  --mode off --games 256 \
  --output .tmp/evaluation/0040_u270_value_search_v0/cpu256_baseline
python3 -m train.0040_dragapult_0809_action_boundary_rl.value_search_v0.run_cpu256 \
  --mode on --games 256 \
  --output .tmp/evaluation/0040_u270_value_search_v0/cpu256_search
```

## 2. Value semantics

The focal terminal reward is win `+1`, loss `-1`, draw `0`. PPO GAE trains Query 0 of the matching U270 `value_head` toward the focal return. Runtime maps its logit to `2 * sigmoid(logit) - 1`, so the output range is `[-1, 1]`, the viewpoint is the focal player encoded by the same-perspective observation, and higher is better. V0 therefore uses pure `argmax`, retaining the Policy selection on an exact tie.

## 3. Implementation

U270 Policy first chooses the normal official selection. The evaluation-only hook recognizes only exact Phase-3-audited handler shapes, groups alternatives inside that selected semantic family, forks every candidate from the same official CPU Search root and the same causal root, collates all branch `DecisionBatch` objects, and performs one batched Value forward. Mutable causal knowledge is copied per fork; static prototypes and model weights are shared. Any unknown shape, Search error, terminal/perspective/turn violation, forbidden log, feature failure, non-finite Value or mapping ambiguity falls back to the original Policy selection. The feature hook is default-off and no official engine source was changed.

## 4. Enabled V0 families

1. `Main/Main` Basic Pokémon Bench placement, only after Policy chose Play Basic.
2. `Main/Main` manual Energy attachment, fixed source Energy and alternative target Pokémon.
3. `Energy/DiscardEnergy` Retreat payment with multiple official selections.
4. focal-owned own-board `Card/Switch` Retreat/Switch target.
5. focal-owned Boss effect ID `1182` opponent-board `Card/Switch` target.
6. focal-owned Energy Switch effect ID `1116` visible attached-resource destination.

The sixth family was enabled but did not occur in this deck/run. Attack, End, Evolve, arbitrary Trainer/Ability/Card nodes, Tool, Draw, deck search, shuffle, coin, Prize, forced singleton, multi-step and opponent-owned decisions remained excluded.

## 5. Regression tests

| Regression | Result |
|---|---|
| New code + Search OFF vs historical U270 CPU256 | PASS: all 256 common behavior fields exact; `162/94/0` both runs |
| Causal root and sibling fork isolation | PASS |
| Phase-3 Search vs equivalent-live feature parity | PASS, all six audited fixtures |
| Actual U270 Value output parity | PASS, exact tensor equality |
| Hidden determinization perturbation invariance | PASS, all enabled fixture families |
| Hidden Draw counterexample remains rejected/sensitive | PASS |
| Policy-conditioned grouping cannot cross Main families | PASS |
| Value winner maps to original official `selected` vector | PASS; tie keeps Policy |
| Official CPU eligibility suite | PASS |
| Evaluation batch regression | PASS |
| Full focused unit suite | PASS: 58 tests plus official Search/Value probe |

## 6. Baseline CPU256

- Result: `162 W / 94 L / 0 D`
- Win rate: `63.28125%`
- Errors/unfinished: `0 / 0`
- Wall time: `761.972 s`
- Report: `.tmp/evaluation/0040_u270_value_search_v0/cpu256_baseline/report/run-6570434814e543f3bcb8a0ce442589e9/report.html`

## 7. Value Search CPU256

- Result: `120 W / 136 L / 0 D`
- Win rate: `46.875%`
- Errors/unfinished: `0 / 0`
- Wall time: `573.583 s`
- Report: `.tmp/evaluation/0040_u270_value_search_v0/cpu256_search/report/run-a0a7a96b6b7141e8a5d0fe9871dfb8ea/report.html`

## 8. Win-rate delta

- Baseline: `63.28125%`
- Value Search: `46.875%`
- Absolute/percentage-point delta: `-16.40625 pp`

Paired transitions:

| Transition | Games |
|---|---:|
| W -> W | 106 |
| W -> L | 56 |
| L -> W | 14 |
| L -> L | 80 |
| Any draw transition | 0 |

The net paired change is `-42` wins. This is a strong negative smoke signal for pure local Value argmax under this fixed V0 contract; it is not a general conclusion about all possible Value Search designs.

## 9. Search statistics

- Total focal Policy decisions: `26,728`
- V0 eligible/search invocations: `2,257` (`8.444%` of Policy decisions)
- Candidate afterstates evaluated: `9,519` (`4.217` per searched decision)
- Actual overrides: `858`
- Agreement rate among searched decisions: `61.985%`
- Override rate among searched decisions: `38.015%`
- Override rate among all Policy decisions: `3.210%`
- All-search Value margin: mean `0.016390`, median `0`
- Override-only Value margin: mean `0.043115`, median `0.023401`, max `0.567585`

## 10. Family breakdown

| Family | Occurrences/searches | Candidates | Overrides | Agreement | Mean margin | Median margin |
|---|---:|---:|---:|---:|---:|---:|
| Basic placement | 243 | 529 | 123 | 49.38% | 0.029623 | 0.000120 |
| Energy target | 1,432 | 6,798 | 584 | 59.22% | 0.016686 | 0 |
| Retreat payment | 85 | 189 | 49 | 42.35% | 0.017500 | 0.001727 |
| Retreat/Switch target | 329 | 1,355 | 36 | 89.06% | 0.004291 | 0 |
| Opponent target | 168 | 648 | 66 | 60.71% | 0.017861 | 0 |
| Attached-resource destination | 0 | 0 | 0 | N/A | N/A | N/A |

## 11. Representative disagreements

- `crustle_cornerstone_mask_ogerpon_ex_c385b53bf42c-002`, turn 7, Energy target, `L -> W`: Policy `[19]`, Value `[18]`; values `[0.1925, 0.3069, 0.2449, 0.3416, 0.3859, 0.2530]`, margin `0.132941`. Focal board had Active Dragapult ex with Psychic/Fire Energy and five Bench Pokémon; Value moved the manual attachment from Bench index 4 to Bench index 3.
- `marnie_s_grimmsnarl_ex_froslass_a9cf3d228c6a-002`, turn 18, Retreat payment, `L -> W`: Policy `[0]`, Value `[1]`; values `[0.8721, 0.9756]`, margin `0.103553`. Active Dragapult ex had Fire/Dark Energy; Value discarded the second attached Energy instead of the first.
- `teal_mask_ogerpon_ex_hero_s_cape_e40278fd83d9-001`, turn 10, Energy target, `W -> L`: Policy `[10]`, Value `[9]`; values `[-0.2976, 0.2375, -0.2068, -0.2148, -0.2687, -0.0965]`, margin `0.444308`. Focal board was Active Budew plus Dragapult ex, two Munkidori, Fezandipiti ex and Meowth ex; Value redirected the attachment from Bench index 4 to Active.
- `marnie_s_grimmsnarl_ex_froslass_c20a8a46f5c6-024`, turn 7, Energy target, `W -> L`: Policy `[4]`, Value `[6]`; values `[0.2216, 0.1408, 0.2803, 0.4001, 0.3691]`, margin `0.259293`. Focal board had Active Dragapult ex and four Bench Pokémon; Value redirected the attachment from Active to Bench index 2.

These examples identify real divergences but do not establish that a single override caused the final game result.

## 12. Failures / fallbacks

- Search/Value invariant fallback: `0`
- `HANDLER_SHAPE_NOT_WHITELISTED`: `16,440`
- `POLICY_PLAY_NOT_BASIC`: `4,080`
- `UNKNOWN_SELECTION_CARDINALITY`: `1,947`
- `UNKNOWN_SELECTION_SHAPE`: `1,041`
- `FORCED_OR_SINGLETON`: `963`

All noneligible decisions executed the unchanged U270 Policy selection. There were no Search errors, feature failures, non-finite Values, perspective/turn violations or mapping failures.

## 13. Performance overhead

- Baseline wall: `761.972 s`
- Search wall: `573.583 s`
- Recorded official CPU Search time: `11.201 s` total
- Recorded branch encoding/batched Value service time: `648.358 s` summed across concurrent worker requests
- Recorded additional worker time: `2.580 s/game`, `0.2927 s/searched decision`

The raw Search run was `188.389 s` faster, not slower, because RUN A initially shared the GPU with another evaluation. Therefore the wall-time difference is confounded and is not an intrinsic `-24.7%` overhead claim. Internal timings show official CPU Search itself was small; branch feature compilation and queued Value inference dominated. Requests overlap across 16 workers, so summed request time must not be added directly to wall time.

## 14. Final conclusion

Value Search V0 successfully ran in the real U270 official-CPU 256-game evaluation, evaluated 9,519 official Search branches, and executed 858 real action overrides with zero fallback. The exact paired baseline was `63.28125%`; pure Value argmax scored `46.875%`, a `-16.40625 pp` delta with 14 `L->W` versus 56 `W->L`. This first smoke test is sufficiently and consistently negative that a larger evaluation of the unchanged V0 configuration is not justified. Further work, if authorized, should first diagnose Value calibration/local ranking and the family/margin behavior rather than expand the whitelist or simply increase sample size.
