from __future__ import annotations

import copy
from dataclasses import replace
import importlib
from pathlib import Path
from types import MappingProxyType
import time
import unittest


BASE = "train.0035_lifetime_aware_feature_compiler"
DYNAMIC = importlib.import_module(f"{BASE}.features.dynamic_fragments")
LAYERS = importlib.import_module(f"{BASE}.features.layers")
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
STATE = importlib.import_module(f"{BASE}.knowledge.state")
BENCHMARK = importlib.import_module(f"{BASE}.benchmark_incremental_features")
EXTENDED = importlib.import_module(f"{BASE}.tests.extended_fixture")


class DynamicFragmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assets = Path(BASE.replace(".", "/")) / "assets"
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            assets / "official_public_prototypes_v1.json",
            assets / "official_full_engine_prototypes_v2.json",
        )

    def _assert_fixture_parity(self, loader, expected_decisions: int) -> None:
        decisions = 0
        resource_hits = 0
        option_hits = 0
        for trajectory in loader():
            knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            resources = DYNAMIC.ResourceFragmentCompiler()
            options = DYNAMIC.OptionFragmentCompiler()
            for decision in trajectory.decisions:
                snapshot = knowledge.consume(decision.observation, decision.event_cursor)
                row = decision.row()
                cards = LAYERS.compile_card_layer(row, snapshot, self.prototypes)
                self.assertEqual(
                    resources.compile(row, snapshot),
                    LAYERS.compile_resource_layer(row, snapshot),
                )
                self.assertEqual(
                    options.compile(row, snapshot, self.prototypes, cards),
                    LAYERS.compile_option_layer(row, snapshot, self.prototypes, cards),
                )
                decisions += 1
            resource_hits += resources.snapshot()["core_hits"]
            option_hits += options.snapshot()["semantic_hits"]
        self.assertEqual(decisions, expected_decisions)
        self.assertGreater(resource_hits, 0)
        self.assertGreater(option_hits, 0)

    def test_original_35_decision_fixture_has_exact_layer_parity(self) -> None:
        self._assert_fixture_parity(BENCHMARK.load_parity_trajectories, 35)

    def test_extended_525_decision_fixture_has_exact_layer_parity(self) -> None:
        self._assert_fixture_parity(EXTENDED.load_extended_trajectories, 525)

    def _first_work_item(self):
        trajectory = EXTENDED.load_extended_trajectories()[0]
        knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
        decision = trajectory.decisions[0]
        snapshot = knowledge.consume(decision.observation, decision.event_cursor)
        row = decision.row()
        cards = LAYERS.compile_card_layer(row, snapshot, self.prototypes)
        return row, snapshot, cards

    def test_resource_mutation_matrix_patches_age_and_invalidates_counts(self) -> None:
        row, snapshot, _cards = self._first_work_item()
        compiler = DYNAMIC.ResourceFragmentCompiler()
        self.assertEqual(
            compiler.compile(row, snapshot),
            LAYERS.compile_resource_layer(row, snapshot),
        )
        identity, entry = next(iter(snapshot.self_ledger.items()))

        age_entry = replace(entry, deck=replace(entry.deck, age=entry.deck.age + 1))
        age_ledger = dict(snapshot.self_ledger)
        age_ledger[identity] = age_entry
        age_snapshot = replace(snapshot, self_ledger=MappingProxyType(age_ledger))
        before_age = compiler.snapshot()
        self.assertEqual(
            compiler.compile(row, age_snapshot),
            LAYERS.compile_resource_layer(row, age_snapshot),
        )
        after_age = compiler.snapshot()
        self.assertEqual(
            after_age["core_hits"] - before_age["core_hits"],
            len(snapshot.self_ledger),
        )
        self.assertEqual(after_age["core_misses"], before_age["core_misses"])

        visible = dict(age_entry.visible)
        visible["hand"] = visible.get("hand", 0) + 1
        visible_entry = replace(age_entry, visible=MappingProxyType(visible))
        visible_ledger = dict(age_snapshot.self_ledger)
        visible_ledger[identity] = visible_entry
        visible_snapshot = replace(
            age_snapshot, self_ledger=MappingProxyType(visible_ledger)
        )
        before_visible = compiler.snapshot()
        self.assertEqual(
            compiler.compile(row, visible_snapshot),
            LAYERS.compile_resource_layer(row, visible_snapshot),
        )
        after_visible = compiler.snapshot()
        self.assertEqual(
            after_visible["core_misses"], before_visible["core_misses"] + 1
        )

        reordered = dict(reversed(tuple(visible_snapshot.self_ledger.items())))
        reordered_snapshot = replace(
            visible_snapshot, self_ledger=MappingProxyType(reordered)
        )
        before_order = compiler.snapshot()
        self.assertEqual(
            compiler.compile(row, reordered_snapshot),
            LAYERS.compile_resource_layer(row, reordered_snapshot),
        )
        self.assertEqual(
            compiler.snapshot()["layout_hits"], before_order["layout_hits"] + 1
        )

    def test_option_mutation_matrix_reuses_semantics_not_relations_or_ordinals(self) -> None:
        row, snapshot, cards = self._first_work_item()
        compiler = DYNAMIC.OptionFragmentCompiler()
        expected = LAYERS.compile_option_layer(
            row, snapshot, self.prototypes, cards
        )
        self.assertEqual(
            compiler.compile(row, snapshot, self.prototypes, cards), expected
        )

        # Relation indices are action-local.  A changed card layout must be
        # reflected even though the prototype semantic cache is a hit.
        shifted_cards = replace(
            cards,
            context_index=cards.context_index + 1,
            effect_card_index=cards.effect_card_index + 1,
        )
        before_relation = compiler.snapshot()
        actual = compiler.compile(row, snapshot, self.prototypes, shifted_cards)
        self.assertEqual(
            actual,
            LAYERS.compile_option_layer(row, snapshot, self.prototypes, shifted_cards),
        )
        self.assertEqual(actual.context, tuple([shifted_cards.context_index] * len(actual.cat)))
        self.assertGreater(
            compiler.snapshot()["semantic_hits"], before_relation["semantic_hits"]
        )

        # Reordering options rebuilds their ordinal and parent arrays while
        # retaining the same semantic expansions.
        reordered = copy.deepcopy(row)
        raw_options = reordered["actor_observation"]["select"]["option"]
        raw_options.reverse()
        reordered_cards = LAYERS.compile_card_layer(
            reordered, snapshot, self.prototypes
        )
        before_reorder = compiler.snapshot()
        reordered_actual = compiler.compile(
            reordered, snapshot, self.prototypes, reordered_cards
        )
        self.assertEqual(
            reordered_actual,
            LAYERS.compile_option_layer(
                reordered, snapshot, self.prototypes, reordered_cards
            ),
        )
        self.assertEqual(
            tuple(item[13] for item in reordered_actual.cat),
            tuple(range(1, len(reordered_actual.cat) + 1)),
        )
        self.assertGreater(
            compiler.snapshot()["semantic_hits"], before_reorder["semantic_hits"]
        )

        # Attack semantics are part of the normalized key and must miss.
        changed_attack = copy.deepcopy(row)
        changed_option = changed_attack["actor_observation"]["select"]["option"][0]
        changed_option["attackId"] = int(changed_option.get("attackId", 0)) + 100_000
        changed_cards = LAYERS.compile_card_layer(
            changed_attack, snapshot, self.prototypes
        )
        before_attack = compiler.snapshot()
        self.assertEqual(
            compiler.compile(
                changed_attack, snapshot, self.prototypes, changed_cards
            ),
            LAYERS.compile_option_layer(
                changed_attack, snapshot, self.prototypes, changed_cards
            ),
        )
        self.assertGreater(
            compiler.snapshot()["semantic_misses"], before_attack["semantic_misses"]
        )

    def test_component_microbench_cached_path_reduces_materialization_time(self) -> None:
        trajectory = EXTENDED.load_extended_trajectories()[0]
        knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
        work = []
        for decision in trajectory.decisions:
            snapshot = knowledge.consume(decision.observation, decision.event_cursor)
            row = decision.row()
            work.append((
                row,
                snapshot,
                LAYERS.compile_card_layer(row, snapshot, self.prototypes),
            ))

        resources = DYNAMIC.ResourceFragmentCompiler()
        options = DYNAMIC.OptionFragmentCompiler()

        def baseline() -> None:
            for row, snapshot, cards in work:
                LAYERS.compile_resource_layer(row, snapshot)
                LAYERS.compile_option_layer(row, snapshot, self.prototypes, cards)

        def cached() -> None:
            for row, snapshot, cards in work:
                resources.compile(row, snapshot)
                options.compile(row, snapshot, self.prototypes, cards)

        baseline()
        cached()

        def best_ns(function) -> int:
            samples = []
            for _ in range(12):
                started = time.perf_counter_ns()
                function()
                samples.append(time.perf_counter_ns() - started)
            return min(samples)

        baseline_ns = best_ns(baseline)
        cached_ns = best_ns(cached)
        self.assertLess(cached_ns, baseline_ns)
        self.assertGreater(resources.snapshot()["core_hits"], 0)
        self.assertGreater(options.snapshot()["semantic_hits"], 0)


if __name__ == "__main__":
    unittest.main()
