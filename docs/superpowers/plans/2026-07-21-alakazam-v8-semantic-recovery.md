# Alakazam V8 Semantic Recovery Implementation Plan

> For agentic workers: execute this plan task-by-task with review checkpoints. Steps use checkbox syntax for tracking.

**Goal:** Restore work/alakazam_v8_current to the approved V8 design and the submission/alakazam_v8_luna_deck_opt behavior oracle, then validate it with the adjacent Auto-Iteration Sample protocol.

**Architecture:** Keep the refactored Facts -> Routes/Plan -> Policies -> Effects pipeline. Use the old single-file agent only as a behavior oracle and use focused fixtures plus external replay traces to port one semantic rule at a time. The current repository evaluation/ package is excluded from execution.

**Tech Stack:** Python 3.11+, unittest, official cg runtime, adjacent ptcg-agent-kaggle/eval/alakazam_replay.py, scripts/alakazam_auto_iter.py analyze, and /tmp for disposable full traces.

## Global Constraints

- The comparison protocol is 17 opponents x 10 games = 170 games per Sample batch.
- First/second turn is a reported sample dimension; alternating game order is not a strategy requirement.
- submission/alakazam_v8_luna_deck_opt is the Target oracle; work/alakazam_v8_current is the only production candidate changed.
- Do not use the current repository evaluation/ runner for this work.
- Do not modify the adjacent evaluator, official engine source, or Kaggle submission directories.
- Attack is a terminal turn submission; Trading Places is never treated as an ordinary switch.
- Production code changes require a failing regression test before implementation.
- Full traces stay in /tmp; preserve only lightweight AutoIter summaries and selected cases.
- Check df -h / /tmp before and after full evaluation; below 10 GB available space, FIFO-delete the oldest disposable evaluation directory before continuing.
- Do not run git commit or git push unless the user explicitly requests it.

---

### Task 1: Record and protect the external Sample baseline

**Files:**
- Create: docs/reports/kaggle/alakazam-v8-semantic-recovery/README.md
- Create: docs/reports/kaggle/alakazam-v8-semantic-recovery/target-baseline.md
- Create: docs/reports/kaggle/alakazam-v8-semantic-recovery/start-baseline.md
- Read: /tmp/ptcg-v8-target-baseline/
- Read: /tmp/ptcg-v8-start-baseline/

**Interfaces:**
- Consumes: completed external evaluator directories and analyzer outputs.
- Produces: lightweight baseline references with command, scope, metrics, and limitations; no full trace is copied into the repository.

- [x] Target and Start external Samples have been run with the fixed 17-opponent list and 10 games per opponent.
- [x] Target and Start full traces have been analyzed with scripts/alakazam_auto_iter.py analyze.
- [ ] Write only aggregate baseline notes and run df -h / /tmp. Do not copy game_*.json.

Recorded evidence:
- Target: 118W/50L/2D, 0 errors, 69.4% raw win rate, 32/170 second-turn Powerful Hand games.
- Start: 8W/128L/34D, 34 errors, 4.7% raw win rate, 0/170 second-turn Powerful Hand games.
- Start root cause: NameError: name KADABRA is not defined in the Night Stretcher dispatch branch.

### Task 2: Repair effect dispatch correctness with TDD

**Files:**
- Modify: work/alakazam_v8_current/strategy/effects/dispatcher.py
- Test: tests/test_alakazam_v8_effects.py

**Interfaces:**
- Consumes: select_effect(obs, facts, plan, memory, profile) and the existing effect_obs fixture.
- Produces: a Night Stretcher selection that can rank KADABRA without a runtime NameError.

- [ ] Add this test to EffectSelectorTests and import NIGHT_STRETCHER from strategy.cards:

~~~python
def test_night_stretcher_can_rank_kadabra_without_dispatch_name_error(self) -> None:
    discard = [KADABRA]
    obs = effect_obs(
        player(active=pokemon(DUNSPARCE, 1), discard=discard),
        player(active=pokemon(900, 2)),
        [{"type": 3, "area": 3, "index": 0}],
        effect_id=NIGHT_STRETCHER,
        context=7,
    )
    self.assertEqual(self._choose(obs).option_indexes, (0,))
~~~

- [ ] Run the focused test and confirm the expected NameError:

~~~bash
python3 -m unittest -v tests.test_alakazam_v8_effects.EffectSelectorTests.test_night_stretcher_can_rank_kadabra_without_dispatch_name_error
~~~

- [ ] Add KADABRA to the existing import tuple in dispatcher.py, with no other behavior change.
- [ ] Verify the focused test, all effects tests, and py_compile:

~~~bash
python3 -m unittest -v tests.test_alakazam_v8_effects
python3 -m py_compile work/alakazam_v8_current/main.py work/alakazam_v8_current/strategy/effects/dispatcher.py
~~~

### Task 3: Restore the basic evolution-energy-attack chain

**Files:**
- Modify: work/alakazam_v8_current/strategy/routes.py
- Modify: work/alakazam_v8_current/strategy/planner.py
- Modify: work/alakazam_v8_current/strategy/policies/continuity.py
- Modify: work/alakazam_v8_current/strategy/policies/resources.py
- Modify: work/alakazam_v8_current/strategy/policies/commit.py
- Test: tests/test_alakazam_v8_agent.py
- Test: tests/test_alakazam_v8_planner.py
- Test: tests/test_alakazam_v8_facts.py

**Interfaces:**
- Consumes: TurnFacts, RouteAnalysis, TurnPlan, ActionIntent, semantic options, and match_intent.
- Produces: deterministic intents for legal evolution, one Psychic attachment, Psychic Draw, and a final attack; matching uses card IDs and Pokémon keys rather than option positions.

- [ ] Add failing fixtures for these behaviors:
  1. Active Kadabra with Psychic and Alakazam in hand selects evolution before Powerful Hand.
  2. Active Abra with Psychic and Rare Candy plus Alakazam in hand selects Rare Candy before attack.
  3. A charged Active Alakazam keeps a legal Bench Kadabra evolution/ability route before a non-terminal attack.
- [ ] Run only the new fixtures and confirm at least one fails for the current intent order:
  
~~~bash
python3 -m unittest -v tests.test_alakazam_v8_agent tests.test_alakazam_v8_planner
~~~

- [ ] Make the smallest policy correction: classify the current attacker route first; continuity emits legal Active/Bench evolution or Psychic Draw; resources emits only the one needed energy attachment; commit remains the terminal fallback. Do not introduce a global numeric score or put strategy branches in main.py.
- [ ] Run the V8 strategy suites:

~~~bash
python3 -m unittest -v tests.test_alakazam_v8_agent tests.test_alakazam_v8_facts tests.test_alakazam_v8_planner tests.test_alakazam_v8_luna_deck_opt_strategy
~~~

### Task 4: Restore continuity and effect semantics

**Files:**
- Modify: work/alakazam_v8_current/strategy/routes.py
- Modify: work/alakazam_v8_current/strategy/policies/continuity.py
- Modify: work/alakazam_v8_current/strategy/policies/control.py
- Modify: work/alakazam_v8_current/strategy/policies/resources.py
- Modify: work/alakazam_v8_current/strategy/effects/search.py
- Modify: work/alakazam_v8_current/strategy/effects/recovery.py
- Modify: work/alakazam_v8_current/strategy/effects/targeting.py
- Test: tests/test_alakazam_v8_effects.py
- Test: tests/test_alakazam_v8_agent.py
- Test: tests/test_alakazam_v8_planner.py

**Interfaces:**
- Consumes: semantic facts and effect dispatcher; old main.py branches and selected external trace cases are references only.
- Produces: tested semantics for Dudunsparce handoff, Fezandipiti after KO, legal Enhanced Hammer target priority, recovery selection, and Boss/Xerosic conditions.

- [ ] Add one failing test for each rule family: Dudunsparce Ability hands off to a ready Bench Alakazam; Fezandipiti draws after a non-terminal KO; Enhanced Hammer selects a legal special-energy target; Trading Places never becomes a switch intent.
- [ ] Run the new tests and verify behavioral failures:

~~~bash
python3 -m unittest -v tests.test_alakazam_v8_effects tests.test_alakazam_v8_agent tests.test_alakazam_v8_planner
~~~

- [ ] Implement each rule in its owning module. Keep effect choice in effects/, route certainty in routes.py, timing in policies, and option-to-index matching in options.py.
- [ ] Run all V8 tests and compile the candidate package:

~~~bash
python3 -m unittest -v tests.test_alakazam_v8_agent tests.test_alakazam_v8_effects tests.test_alakazam_v8_facts tests.test_alakazam_v8_planner tests.test_alakazam_v8_luna_deck_opt_strategy
python3 -m compileall -q work/alakazam_v8_current
~~~

### Task 5: Run external Sample iterations with storage cleanup

**Files:**
- Create: docs/reports/kaggle/alakazam-v8-semantic-recovery/iteration-logs/README.md
- Create: docs/reports/kaggle/alakazam-v8-semantic-recovery/iteration-logs/iteration-001.md
- Read: scripts/alakazam_auto_iter.py
- Read: /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/eval/alakazam_replay.py

**Interfaces:**
- Consumes: a correctness-passing candidate and the fixed 17-opponent list.
- Produces: one lightweight iteration record per full Sample with exact command, W/L/D, errors, process metrics, cases, decision, and next hypothesis.

- [ ] Run a focused external evaluator on a small relevant opponent subset. Store full traces under a unique directory such as /tmp/ptcg-v8-iteration-001-focus, then increment the numeric suffix for later iterations.
- [ ] Require py_compile, all V8 unit tests, asset validation, and a focused external run with zero candidate errors before a full Sample.
- [ ] Run the exact external evaluator command shape from Task 1 with a unique /tmp path. Analyze with python3 -m scripts.alakazam_auto_iter analyze. Do not run python3 -m evaluation or the run subcommand of the rewritten compatibility script.
- [ ] Keep a candidate only when errors are zero, core health does not materially regress, and the fresh 17x10 evidence reaches or exceeds Target. Independent random batches are not paired A/B; near results require another same-protocol Sample.
- [ ] Before and after each full Sample run df -h / /tmp. If Avail is below 10Gi, sort disposable /tmp/ptcg-v8-* directories by mtime, verify the explicit oldest path, delete that full-trace directory, and recheck space before continuing.

### Task 6: Final verification and review

**Files:**
- Read: docs/superpowers/specs/2026-07-21-alakazam-v8-semantic-recovery-design.md
- Read: docs/superpowers/plans/2026-07-21-alakazam-v8-semantic-recovery.md
- Read: git diff --check output

**Interfaces:**
- Consumes: the promoted candidate and latest external Sample evidence.
- Produces: a verified handoff summary; no commit, push, Kaggle upload, or source-history deletion.

- [ ] Run the complete local correctness suite, assets, compileall, and git diff --check:

~~~bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 scripts/check_assets.py
python3 -m compileall -q work/alakazam_v8_current scripts
git diff --check
~~~

- [ ] Re-read the final external Sample: 170 games, 0 candidate errors, explicit W/L/D, first/second counts, second-turn Powerful Hand, post-KO relay, and trace root.
- [ ] Run df -h / /tmp and git status --short. Report unrelated existing changes without reverting them and make no completion claim without fresh verification evidence.
