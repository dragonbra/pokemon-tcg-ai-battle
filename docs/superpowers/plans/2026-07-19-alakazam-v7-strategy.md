# Alakazam V7 Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在固定 V6 `deck.csv` 的前提下，实现 V7 的规则语义策略，并用用户反馈的 8 个 replay case 验证首选动作和后续动作链。

**Architecture:** V7 从 `submission/alakazam_v6/main.py` 复制运行时和合法 option 映射，只在 V7 目录维护策略状态与排序逻辑。`TurnMemory` 扩展为记录对手上一回合实际使用的 Budew `Itchy Pollen`、每个 Pokémon 的进化时钟和本回合预算；主行动每次只选择当前合法 option 中的最高优先级动作，攻击始终是终止提交。测试使用现有 V6 observation fixture 风格，增加 V7-01 至 V7-08 的可执行行为断言。

**Tech Stack:** Python 3.11+, `unittest`, 官方 simulator option schema, existing `cg/` runtime.

## Global Constraints

- `submission/alakazam_v7/deck.csv` 必须与 `submission/alakazam_v6/deck.csv` 字节一致。
- 不修改 V6、用户现有的 `submission/alakazam_v7/IMPROVEMENT.md` 或官方卡牌数据。
- 不按 opponent ID 写 matchup 分支；仅允许硬编码 `Budew=235`、`Itchy Pollen=323` 的实际规则事件。
- Active Abra 永不选择任何攻击；`Dunsparce Trading Places (423)` 永不作为有效换位策略。
- 所有返回的 action 必须来自当前 simulator 提供的合法 option。
- 每个新行为必须先有会失败的测试，再写策略代码；每个测试循环都运行 V7 测试与既有 V6 回归测试。

---

### Task 1: 建立 V7 运行时和测试基线

**Files:**
- Create: `submission/alakazam_v7/main.py`，复制 V6 运行时作为 V7 策略基线。
- Create: `submission/alakazam_v7/deck.csv`，复制 V6 固定卡组。
- Create: `submission/alakazam_v7/cg/__init__.py`, `api.py`, `game.py`, `sim.py`, `utils.py` 及运行库文件，保持 V6 simulator bridge 不变。
- Create: `submission/alakazam_v7/README.md` 与 `submission/alakazam_v7/STRATEGY.md`，说明 V7 只改策略不改卡组。
- Create: `tests/test_alakazam_v7_strategy.py`，复制现有 V6 fixture 并把 import/path/class 改为 V7。

**Interfaces:**
- V7 继续暴露 `DECK`, `agent(obs_dict) -> list[int]`、`_main_action()` 和 `_select_effect()`。
- 测试模块通过 `importlib.util.spec_from_file_location` 加载 V7 `main.py`，不依赖 package import。

- [ ] 复制 V6 的策略入口、deck、cg runtime 和测试 fixture，避免覆盖 V7 的设计文档和用户反馈文件。
- [ ] 将 V7 测试中的 `MAIN_PATH`、module name、测试类名和新游戏测试名称改为 V7。
- [ ] 运行 `python3 -m unittest -v tests.test_alakazam_v7_strategy`，确认复制基线通过。
- [ ] 运行 `cmp submission/alakazam_v6/deck.csv submission/alakazam_v7/deck.csv`，确认固定卡组一致。

### Task 2: 先写 V7 行为红测试

**Files:**
- Modify: `tests/test_alakazam_v7_strategy.py`，增加 Item Lock、Abra 攻击过滤、进化目标、Dudunsparce 过牌、终局闭环和 8 个 replay case 的最小测试。

**Interfaces:**
- Tests call `TurnMemory.sync(current, player, logs=logs)`, `_main_action(obs)`, `_select_effect(obs)`, `_choose_evolution_target(options, current, player)`, and `_item_lock_active(current)`.
- Every test compares normalized option identity (`type`, `cardId`/`attackId`, target area/index), never assumes a fixed simulator option ordering outside the fixture.

- [ ] Add tests that fail on the copied V6 behavior: Active Abra with `attackId=1070` chooses `END`; an actual opponent log `{type: 15, cardId: 235, attackId: 323, playerIndex: 1}` makes Rare Candy unavailable; Budew merely present without that log does not lock Rare Candy.
- [ ] Add V7-01/V7-02/V7-08 tests for Bench Abra natural evolution before Active direct Rare Candy, Active Kadabra natural evolution, and Shaymin plus two Abra role assignment.
- [ ] Add V7-03/V7-07 tests for Item Lock natural evolution and Hilda selecting Dudunsparce plus Enriching Energy when no Abra base exists.
- [ ] Add V7-04/V7-05/V7-06 tests for terminal Dudunsparce handoff, no Abra attack, and post-Unfair-Stamp draw-before-Kadabra.
- [ ] Run the new focused tests and record the expected failures before changing `main.py`.

### Task 3: 实现事件状态和资源路线模型

**Files:**
- Modify: `submission/alakazam_v7/main.py` around constants, `TurnMemory`, route helpers, and effect selection.

**Interfaces:**
- Add constants `BUDEW = 235`, `ITCHY_POLLEN_ATTACK = 323`, `TELEPORTATION_ATTACK = 1070`.
- Add `TurnMemory.item_lock_turn_key: tuple[int, int] | None` and `sync(current, player, logs=None) -> None`.
- Add `_item_lock_active(current) -> bool`, `_rare_candy_route_available(current, player) -> bool`, `_is_abra_attack(option, active_id) -> bool`, and `_enriching_draw_route(current, player) -> bool`.

- [ ] On a log with opponent `type=15`, `cardId=235`, `attackId=323`, set Item Lock only for the immediately observed own turn; clear it when the turn key changes and reset it on a new game.
- [ ] Make all `sync` call sites pass `obs.get("logs")`; preserve V6 supporter, energy, retreat, evolution and resource ledger tracking.
- [ ] Make direct Abra-to-Alakazam, Hilda search, and Rare Candy scoring consult `_rare_candy_route_available`; do not infer lock from Budew presence or absent options.
- [ ] Make `_enriching_draw_route` cover both the existing ready-Alakazam handoff and the Item Lock fallback where Dunsparce is the available draw engine.
- [ ] Make `_main_draw_gain`, deck protection, and Dudunsparce net deck delta continue to model draw-three plus returned Pokémon/attachments accurately.
- [ ] Run only the Task 2 tests and make them pass without weakening existing V6 legality guards.

### Task 4: 实现 V7 主行动优先级

**Files:**
- Modify: `submission/alakazam_v7/main.py` in `_choose_evolution_target`, `_choose_energy_option`, `_choose_switch_option`, and `_main_action`.

**Interfaces:**
- `_choose_evolution_target(...) -> list[int]` must rank natural evolution and Rare Candy targets by the primary attacker route, not by Active/Bench position alone.
- `_main_action(obs) -> list[int]` must keep attack options below all required same-turn actions and must record the chosen action in `TurnMemory`.

- [ ] Rank an unenergized Bench Abra -> Kadabra before the energized Active Abra when Active has a complete Rare Candy + Alakazam route.
- [ ] Rank Active Kadabra -> Alakazam before Bench setup and before non-KO attack; after that, allow Bench Abra -> Kadabra and Ability before attack.
- [ ] Once an Alakazam can attack, preserve second/third Rare Candy and prefer natural Kadabra handoff unless it closes an immediate Prize.
- [ ] In Item Lock, rank natural evolution, Enriching Energy to Dunsparce/Dudunsparce, Dudunsparce Ability, and recovery/draw above Rare Candy and weak attacks.
- [ ] Reject all Abra attack options, including `attackId=1070`, in every main-action branch; preserve Kadabra non-KO attack only as the final fallback.
- [ ] Keep Boss's Orders highest-HP guaranteed KO targeting, Xerosic only for a wounded non-KO Active with opponent hand count at least 6, and normal Alakazam attack after preparation.
- [ ] Allow `Run Away Draw -> Bench Alakazam -> Powerful Hand` or legal Retreat to bypass the deck gate only when it closes the final Prize; never revive Trading Places.
- [ ] Run V7 tests, then the full V6 test suite to detect regressions in copied shared logic.

### Task 5: 完善效果选择与 replay case 报告

**Files:**
- Modify: `tests/test_alakazam_v7_strategy.py` with normalized action helpers and case labels.
- Modify: `submission/alakazam_v7/STRATEGY.md` with the final priority list and case reporting format.

**Interfaces:**
- Add test helper `normalize_option(option, current, player) -> tuple` and `assert_first_action(...)` to report `type + card/attack + target`.
- Keep effect selection legal for Hilda, Dawn, Lana's Aid, Night Stretcher, Kadabra/Alakazam/Dudunsparce Ability, damage target and switch target contexts.

- [ ] Verify V7-01 through V7-08 first actions and immediate follow-up actions, including the corrected Dudunsparce identity in V7-04/V7-07.
- [ ] Verify Lana's Aid is chosen over the Supporter-plus-Night-Stretcher sequence when it recovers Pokémon and Psychic Energy together.
- [ ] Verify deck protection uses 10 as the danger line, 15 as the watch line, and terminal Prize closure as the only bypass.
- [ ] Document that each evaluation reports V6 action, V7 first action, V7 replanned action chain, and PASS/FAIL reason.

### Task 6: 资产验证、打包和最终回归

**Files:**
- Modify: `submission/alakazam_v7/README.md` with commands and current limitations.
- Create: `dist/alakazam_v7.tar.gz` through the repository packaging script.

- [ ] Run `python3 scripts/check_assets.py` and verify V7 has 60 cards and the V6 deck counts.
- [ ] Run `python3 -m unittest -v tests.test_alakazam_v7_strategy tests.test_alakazam_v6_strategy`.
- [ ] Run `python3 -m compileall -q scripts submission`.
- [ ] Run a raw-exec probe from inside `submission/alakazam_v7` to verify `DECK`, `agent`, and the no-`__file__` loader path.
- [ ] Run `bash scripts/package_submission.sh alakazam_v7` and `tar -tzf dist/alakazam_v7.tar.gz`; confirm top-level `main.py`, `deck.csv`, and `cg/` are present.
- [ ] Run the smallest local battle smoke test with output under `/tmp`; do not treat local win rate as the V7 evaluation result.
