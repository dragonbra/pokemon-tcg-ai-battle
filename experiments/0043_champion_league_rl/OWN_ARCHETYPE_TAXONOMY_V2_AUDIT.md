# 0043 Own Archetype Taxonomy V2 Audit

Date: 2026-08-13  
Status: **PASS — taxonomy/migration gates complete; no RL update was launched.**

## Executive conclusion

- Audited all exact lists 001–067 against official card data: Pokémon/evolution lines, Trainers/Stadiums, Energy, Rare Candy, Prize/tempo, bench allocation, resource engines, and attack/win routes.
- V1 had 15 classes. V2 has 29 append-only classes: 15 meanings preserved and 14 added strategic rows. No exact-deck-per-class naming was introduced.
- All known decks are explicitly mapped; `other` remains fallback-only and currently maps no 001–067 deck.
- Champion-G1 remains unchanged. V1-Focal-Seed expands both own embeddings from `[15,16]` to `[29,16]` by exact old-row preservation and parent-copy initialization.
- Zero-step parity: **PASS** for logits, probabilities, greedy action, Value, and both adapter paths.

## Repository audit

1. Formal base source: root `FrozenPool_65_decks_2026-08-12.zip`, SHA-256 `2937c23a85c0e3805b40fb014f0274dbfc3f08e04ce3816c120105616ce320ee`; user-approved Dragapult lists were appended as 066/067 with exact-list provenance.
2. Deck registry: `train/0043_champion_league_rl/assets/decks/registry.json`; all 67 training decks are exact 60-card assets. FrozenMeta256-V1 remains 55 decks.
3. V1 own taxonomy: 0042 trigger-priority JSON with fallback class 14; it hard-coded 15 in ID validation and both embeddings.
4. Opponent Meta is a separate vocabulary/type and the pretrained Value head remains 15-way. It was not retrained or expanded.
5. OwnArchetypeId enters `ValueResidualAdapter` and `PolicyStrategyAdapter`. Those are the only own embedding tables found; both are width 16.
6. G1 checkpoint metadata stores vocabulary version and taxonomy SHA-256; loader/export historically constructed 15-row tables. 0043 runtime now derives rows from checkpoint tensors and validates version/schema.
7. CPU/Kaggle project-local compound runtime shares that loader. CUDA routes complete policy identity after materialization; the Frozen evaluation registry is unchanged.
8. G1 source is `archive/pretrained/0042_champion_g1`, copied into 0043 and pinned by two artifact hashes. Optimizers train both adapters; the V1 focal seed explicitly starts with empty optimizer state.
9. Active own-vocabulary magic numbers were removed from runtime construction. Remaining literal 15 values occur only in the explicit V1→V2 migration/parity contract.

## 67-deck decision table

| Deck | Name | Old | New | Decision | Parent | Strategic axis |
|---|---|---|---|---|---|---|
| 001 | Marnie's Grimmsnarl ex / Froslass | marnies_grimmsnarl_ex | marnies_grimmsnarl_ex | keep | — | Punk Up energy acceleration |
| 002 | Alakazam / Dudunsparce | alakazam | alakazam | keep | — | hand-size damage and draw |
| 003 | Mega Lopunny ex / Mega Froslass ex | mega_lopunny_ex | mega_lopunny_ex | keep | — | switch-tempo attacker |
| 004 | Mega Lopunny ex / Mega Froslass ex | mega_lopunny_ex | mega_lopunny_ex | keep | — | switch-tempo attacker |
| 005 | Marnie's Grimmsnarl ex / Froslass | marnies_grimmsnarl_ex | marnies_grimmsnarl_ex | keep | — | Punk Up energy acceleration |
| 006 | Teal Mask Ogerpon ex / Hero’s Cape | teal_mask_ogerpon_heros_cape | teal_mask_ogerpon_heros_cape | keep | — | Ogerpon energy stacking |
| 007 | Dragapult ex | dragapult_ex | dragapult_ex | keep | — | Phantom Dive spread |
| 008 | Teal Mask Ogerpon ex / Hero’s Cape | teal_mask_ogerpon_heros_cape | teal_mask_ogerpon_heros_cape | keep | — | Ogerpon energy stacking |
| 009 | Mega Lucario ex / Solrock | mega_lucario_ex | mega_lucario_ex | keep | — | Fighting attack rotation |
| 010 | Festival Lead / Dipplin | festival_lead | festival_lead | keep | — | Festival double attack |
| 011 | Mega Kangaskhan ex / Crustle | mega_kangaskhan_ex | mega_kangaskhan_ex | keep | — | Kangaskhan active draw with wall |
| 012 | Mega Lopunny ex / Mega Froslass ex | mega_lopunny_ex | mega_lopunny_ex | keep | — | switch-tempo attacker |
| 013 | Cynthia's Garchomp ex / Roserade | cynthias_garchomp_ex | cynthias_garchomp_ex | keep | — | Cynthia evolution engine |
| 014 | Mega Kangaskhan ex / Crustle | mega_kangaskhan_ex | mega_kangaskhan_ex | keep | — | Kangaskhan active draw with wall |
| 015 | Mega Kangaskhan ex / Crustle | mega_kangaskhan_ex | area_zero_tera_toolbox | split | mega_kangaskhan_ex | Area Zero eight-bench toolbox |
| 016 | Mega Kangaskhan ex / Crustle | mega_kangaskhan_ex | slowking_toolbox | split | mega_kangaskhan_ex | top-deck copied-attack toolbox |
| 017 | Marnie's Grimmsnarl ex / Froslass | marnies_grimmsnarl_ex | marnies_grimmsnarl_ex | keep | — | Punk Up energy acceleration |
| 018 | Dragapult ex | dragapult_ex | dragapult_ex | keep | — | Phantom Dive spread |
| 019 | Barbaracle / Cornerstone Mask Ogerpon ex | crustle_barbaracle_wall | crustle_barbaracle_wall | keep | — | single-prize wall and denial |
| 020 | Mega Venusaur ex / Meganium | meganium_grass_evolution | meganium_grass_evolution | keep | — | Meganium Grass evolution |
| 021 | Team Rocket's Mewtwo ex / Spidops | team_rockets_mewtwo_ex | team_rockets_mewtwo_ex | keep | — | Rocket board and energy sacrifice |
| 022 | Mega Lopunny ex / Mega Froslass ex | mega_lopunny_ex | mega_lopunny_ex | keep | — | switch-tempo attacker |
| 023 | Hydrapple ex / Meganium | festival_lead | hydrapple_meganium | split | festival_lead | Hydrapple energy-wide scaling |
| 024 | Crustle / Cornerstone Mask Ogerpon ex | crustle_barbaracle_wall | crustle_barbaracle_wall | keep | — | single-prize wall and denial |
| 025 | Mega Lopunny ex / Mega Froslass ex | mega_lopunny_ex | teal_mask_ogerpon_toolbox | split | mega_lopunny_ex | Ogerpon toolbox with thin Lopunny |
| 026 | Mega Kangaskhan ex / Crustle | mega_kangaskhan_ex | mega_venusaur_meganium | split | mega_kangaskhan_ex | dual Stage-2 Grass energy transfer |
| 027 | Mega Kangaskhan ex / Crustle | mega_kangaskhan_ex | area_zero_tera_toolbox | split | mega_kangaskhan_ex | Area Zero eight-bench toolbox |
| 028 | Mega Lopunny ex / Mega Froslass ex | mega_lopunny_ex | mega_lopunny_ex | keep | — | switch-tempo attacker |
| 029 | Teal Mask Ogerpon ex / Hero’s Cape | teal_mask_ogerpon_heros_cape | teal_mask_ogerpon_heros_cape | keep | — | Ogerpon energy stacking |
| 030 | Mega Kangaskhan ex / Crustle | mega_kangaskhan_ex | mega_kangaskhan_ex | keep | — | Kangaskhan active draw with wall |
| 031 | Marnie's Grimmsnarl ex / Froslass | marnies_grimmsnarl_ex | marnies_grimmsnarl_ex | keep | — | Punk Up energy acceleration |
| 032 | Mega Lopunny ex / Mega Froslass ex | mega_lopunny_ex | teal_mask_ogerpon_toolbox | split | mega_lopunny_ex | Ogerpon toolbox with thin Lopunny |
| 033 | Mega Kangaskhan ex / Crustle | mega_kangaskhan_ex | mega_kangaskhan_ex | keep | — | Kangaskhan active draw with wall |
| 034 | Alakazam / Dudunsparce | alakazam | alakazam | keep | — | hand-size damage and draw |
| 035 | Mega Kangaskhan ex / Crustle | mega_kangaskhan_ex | mega_kangaskhan_ex | keep | — | Kangaskhan active draw with wall |
| 036 | Teal Mask Ogerpon ex / Hero’s Cape | teal_mask_ogerpon_heros_cape | teal_mask_ogerpon_heros_cape | keep | — | Ogerpon energy stacking |
| 037 | Marnie's Grimmsnarl ex / Froslass | marnies_grimmsnarl_ex | marnies_grimmsnarl_ex | keep | — | Punk Up energy acceleration |
| 038 | Crustle / Cornerstone Mask Ogerpon ex | crustle_barbaracle_wall | crustle_barbaracle_wall | keep | — | single-prize wall and denial |
| 039 | Alakazam / Dudunsparce | alakazam | alakazam | keep | — | hand-size damage and draw |
| 040 | Alakazam / Dudunsparce | alakazam | alakazam | keep | — | hand-size damage and draw |
| 041 | Hydrapple ex / Meganium | festival_lead | hydrapple_meganium | split | festival_lead | Hydrapple energy-wide scaling |
| 042 | Arboliva ex / Meganium / Teal Mask Ogerpon ex | meganium_grass_evolution | meganium_grass_evolution | keep | — | Meganium Grass evolution |
| 043 | Dragapult ex / Dusknoir | dragapult_ex | dragapult_dusknoir | split | dragapult_ex | Dusknoir self-KO and damage placement |
| 044 | Mega Starmie ex / Mega Froslass ex | mega_starmie_ex | mega_starmie_ex | keep | — | Starmie fast spread |
| 045 | Mega Starmie ex / Mega Froslass ex | mega_starmie_ex | mega_starmie_ex | keep | — | Starmie fast spread |
| 046 | Mega Starmie ex / Dusknoir | mega_starmie_ex | mega_starmie_dusknoir | split | mega_starmie_ex | Dusknoir self-KO and Starmie breakpoints |
| 047 | Erika's Vileplume ex / Cinderace | other | erikas_vileplume_ex | split | other | team healing and status control |
| 048 | Archaludon ex / Cinderace | archaludon_ex | archaludon_ex | keep | — | metal discard acceleration |
| 049 | N's Zoroark ex / Munkidori | other | ns_zoroark_ex | split | other | benched attack library and copying |
| 050 | Mega Starmie ex / Dusknoir | mega_starmie_ex | mega_starmie_dusknoir | split | mega_starmie_ex | Dusknoir self-KO and Starmie breakpoints |
| 051 | Archaludon ex / Cinderace | archaludon_ex | archaludon_ex | keep | — | metal discard acceleration |
| 052 | Archaludon ex / Cinderace | archaludon_ex | archaludon_ex | keep | — | metal discard acceleration |
| 053 | Mega Starmie ex / Mega Froslass ex | mega_starmie_ex | mega_starmie_ex | keep | — | Starmie fast spread |
| 054 | N's Zoroark ex / Munkidori | other | ns_zoroark_ex | split | other | benched attack library and copying |
| 055 | Archaludon ex / Cinderace | archaludon_ex | archaludon_ex | keep | — | metal discard acceleration |
| 056 | Chandelure Sylveon | other | chandelure_hand_control | split | other | opponent-hand control and safeguard |
| 057 | Dragapult Ex Dusknoir Self Destruct | dragapult_ex | dragapult_dusknoir | split | dragapult_ex | Dusknoir self-KO and damage placement |
| 058 | Mega Kangaskhan Ex Crustle | mega_kangaskhan_ex | pecharunt_area_zero_poison | split | mega_kangaskhan_ex | Area Zero poison checkup route |
| 059 | Team Rocket S Mewtwo Ex Spidops | team_rockets_mewtwo_ex | team_rockets_mewtwo_ex | keep | — | Rocket board and energy sacrifice |
| 060 | Cynthia S Garchomp Ex Roserade | cynthias_garchomp_ex | cynthias_garchomp_ex | keep | — | Cynthia evolution engine |
| 061 | Dragapult Ex Blaziken Ex | dragapult_ex | dragapult_blaziken | split | dragapult_ex | Blaziken energy recursion and dual Stage-2 |
| 062 | Mega Starmie ex / Mega Froslass ex | mega_starmie_ex | mega_starmie_ex | keep | — | Starmie/Froslass fast spread |
| 063 | Mega Lucario Ex Solrock | mega_lucario_ex | mega_lucario_ex | keep | — | Fighting attack rotation |
| 064 | Mega Lopunny Ex Mega Froslass Ex | mega_lopunny_ex | mega_gardevoir_leafeon | split | mega_lopunny_ex | Grand Tree and board-wide Psychic energy |
| 065 | Crustle Great Tusk | crustle_barbaracle_wall | crustle_great_tusk_mill | split | crustle_barbaracle_wall | Great Tusk alternate deck-out route |
| 066 | Dragapult ex / Dusknoir Control | dragapult_dusknoir | dragapult_dusknoir | keep | dragapult_ex | Dusknoir self-KO, spread breakpoints, and Hammer/Watchtower control |
| 067 | Dragapult ex / Munkidori Control | dragapult_ex | dragapult_ex | keep | — | Phantom Dive spread with Munkidori and Hammer/Stadium control |

## Existing classes retained

- `0` `dragapult_ex` — decks 007, 018, 067; Dragapult ex Stage-2 spread-damage plan without a second structural Stage-2 engine.
- `1` `mega_lopunny_ex` — decks 003, 004, 012, 022, 028, 064; Switch-tempo Mega Lopunny plan centered on moving from Bench to Active.
- `2` `marnies_grimmsnarl_ex` — decks 001, 005, 017, 031, 037; Stage-2 dark-energy acceleration with Froslass/Munkidori damage support.
- `3` `alakazam` — decks 002, 034, 039, 040; Hand-size damage and evolution-trigger draw with Dudunsparce cycling.
- `4` `mega_lucario_ex` — decks 009, 063; Fast Fighting attacker with discard-to-bench energy acceleration.
- `5` `mega_kangaskhan_ex` — decks 011, 014, 030, 033, 035; Active-position draw/attack shell with resilient secondary lines.
- `6` `festival_lead` — decks 010; Festival Grounds bench-scaling double-attack strategy.
- `7` `crustle_barbaracle_wall` — decks 019, 024, 038; Single-prize wall and denial plans centered on Crustle or Barbaracle.
- `8` `meganium_grass_evolution` — decks 020, 042; Multi-line Grass evolution board using Meganium and Forest of Vitality.
- `9` `team_rockets_mewtwo_ex` — decks 021, 059; Team Rocket board-count and benched-energy sacrifice strategy.
- `10` `teal_mask_ogerpon_heros_cape` — decks 006, 008, 029, 036; Four-Ogerpon energy-stacking beatdown with repeated Teal Dance.
- `11` `cynthias_garchomp_ex` — decks 013, 060; Cynthia-tagged evolution and Fighting-energy attack engine.
- `12` `mega_starmie_ex` — decks 044, 045, 053, 062; Water Stage-1 spread attacker paired with Mega Froslass hand-pressure variants.
- `13` `archaludon_ex` — decks 048, 051, 052, 055; Metal discard-recovery acceleration and durable Stage-1 attacker.
- `14` `other` — decks none (fallback); Unknown, unmodeled, or not-yet-justified fallback only.

## Newly appended classes

- `15` `dragapult_dusknoir` ← `dragapult_ex` row 0; decks 043, 057, 066. Rare Candy, bench slots, self-KO Prize concessions, and damage-counter breakpoints systematically change optimal sequencing.
- `16` `dragapult_blaziken` ← `dragapult_ex` row 0; decks 061. Rare Candy allocation, bench evolution timing, and persistent energy recovery create a distinct resource plan.
- `17` `mega_starmie_dusknoir` ← `mega_starmie_ex` row 12; decks 046, 050. Self-KO Prize tempo and a second evolution line change bench and KO sequencing.
- `18` `slowking_toolbox` ← `mega_kangaskhan_ex` row 5; decks 016. Top-deck control and copied-attack selection are structurally unlike Kangaskhan beatdown.
- `19` `area_zero_tera_toolbox` ← `mega_kangaskhan_ex` row 5; decks 015, 027. Expanded bench capacity, Tera retention, and cross-board energy routing define decisions.
- `20` `pecharunt_area_zero_poison` ← `mega_kangaskhan_ex` row 5; decks 058. Active Pecharunt placement, poison checkup timing, and eight-bench allocation change the win route.
- `21` `teal_mask_ogerpon_toolbox` ← `mega_lopunny_ex` row 1; decks 025, 032. Energy stacking/search dominates; the one-copy Lopunny is a matchup option, not the strategy axis.
- `22` `mega_gardevoir_leafeon` ← `mega_lopunny_ex` row 1; decks 064. Grand Tree evolution, bench-wide energy allocation, and distributed-energy damage are structural axes.
- `23` `erikas_vileplume_ex` ← `other` row 14; decks 047. Multi-line evolution, all-board healing, and sleep/poison tempo merit an explicit prior.
- `24` `ns_zoroark_ex` ← `other` row 14; decks 049, 054. Bench attack library construction and copied-attack choice systematically alter actions.
- `25` `chandelure_hand_control` ← `other` row 14; decks 056. The policy manages both players' hand sizes, Rare Candy, lock pieces, and nonstandard damage scaling.
- `26` `crustle_great_tusk_mill` ← `crustle_barbaracle_wall` row 7; decks 065. Supporter timing and attacks target deck-out progress instead of ordinary Prize tempo.
- `27` `hydrapple_meganium` ← `festival_lead` row 6; decks 023, 041. Energy-wide damage, healing attachment, and evolution acceleration replace Festival double attacks.
- `28` `mega_venusaur_meganium` ← `mega_kangaskhan_ex` row 5; decks 026. Evolution-line allocation and movable Grass energy dominate; the thin Kangaskhan line is secondary.

## Other

`other` (ID 14) remains an unknown/unmodeled fallback. The audit found every 001–067 deck sufficiently modeled by a stable strategic class, so none remains in Other. Newly promoted classes 23–25 inherit the G1 Other row only because that is the exact historical prior those decks received; RL may now differentiate them.

## Migration and identity

- Taxonomy SHA-256: `9d9bc5cb95d253731270b064d27279448e1a099e0f548dfeb6bc8215bd891f60`
- Exact mapping SHA-256: `6b7d97027d465031344d6851f8826fcf3955e2a9cc9fbd8ba49ff32fd6cb5120`
- Old/new size: 15 → 29; added rows: 14; width: 16 unchanged.
- Policy own embedding: `[15,16]` → `[29,16]`; Value own embedding: `[15,16]` → `[29,16]`.
- Old rows: bitwise equal in FP32 and portable FP16 artifacts. New rows: exactly equal to declared parent. No unrelated model tensor changed.
- G1 artifact SHA-256 remains `aea408e...20fe3` (model-only) and `cd5c057...de84` (portable).
- V1 focal seed deployment-effective tensor hash: `ffdb3caa42777fd67d6ee3edbb5f9eb8f6d5f9463c4a086a2521c4a33ebde7d3`.
- V1 focal optimizer is fresh with zero state entries, both expanded embeddings included, and the frozen opponent Meta head excluded.

## Zero-step parity

**PASS**

- Historical own rows tested: 15/15; exact-deck G1 resolution checked: 55/55.
- Policy logits: EXACT; action probabilities: EXACT; greedy actions: EXACT.
- Value: EXACT; Policy adapter output/delta: EXACT; Value adapter output/delta: EXACT.
- G1 unchanged audit: PASS; V1 and V2 strict-load: PASS.
- Evidence: `experiments/0043_champion_league_rl/focal_seed_zero_step_parity.json`.

## Readiness

- `OWN ARCHETYPE TAXONOMY V2 READY: YES`
- `V1 FOCAL SEED READY FOR RL: YES` for the taxonomy/migration gate.
- Overall 0043 formal launch remains governed by the already-existing independent official-engine CPU/CUDA observation/action parity and fresh-version/W&B preflight in DESIGN; this task did not bypass or execute those gates.

## Human-review boundaries

- `062` 的上游名称是 “Starmie/Dusknoir”，但 exact list 包含完整 Starmie/Froslass line 且没有 Duskull line；0043 正式显示名已纠正为 `Mega Starmie ex / Mega Froslass ex`，上游名称仅保留为 provenance。

## Per-deck lookup assets

- 浏览入口：`train/0043_champion_league_rl/assets/decks/index.html`
- 逐套图文详情：`train/0043_champion_league_rl/assets/decks/definitions/001/index.html` 至 `067/index.html`
- 逐套纯文本：`train/0043_champion_league_rl/assets/decks/text/001.txt` 至 `067.txt`
- 每份档案记录 exact 60-card list、官方卡名/类型、Own Archetype V2 分类、战略特点和来源边界。
- `025/032` and `064` expose old trigger-priority misclassification (thin Lopunny before their actual Ogerpon/Gardevoir axes). V2 corrects them with append rows parent-copied from the exact G1 row for parity.
- `020` stays in the broad Meganium Grass class while `026` splits: 020 matches that class’s existing multi-line Grass definition, whereas 026 was historically mislabeled Kangaskhan and adds explicit Venusaur energy-transfer sequencing.
- Taxonomy granularity should be revisited only if official-game training evidence shows a stable action-function collision; do not split on card-count tuning alone.
