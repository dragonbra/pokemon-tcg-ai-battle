from __future__ import annotations

import importlib
from pathlib import Path
import time
import unittest


BASE = "train.0035_lifetime_aware_feature_compiler"
ASSEMBLY = importlib.import_module(f"{BASE}.features.assembly")
BENCHMARK = importlib.import_module(f"{BASE}.benchmark_incremental_features")
COMPILER = importlib.import_module(f"{BASE}.features.compiler")
EXTENDED = importlib.import_module(f"{BASE}.tests.extended_fixture")
LAYERS = importlib.import_module(f"{BASE}.features.layers")
PROTOTYPES = importlib.import_module(f"{BASE}.domain.prototypes")
STATE = importlib.import_module(f"{BASE}.knowledge.state")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _layers(row, snapshot, prototypes):
    cards = LAYERS.compile_card_layer(row, snapshot, prototypes)
    return (
        cards,
        LAYERS.compile_resource_layer(row, snapshot),
        LAYERS.compile_event_layer(row, snapshot, cards),
        LAYERS.compile_option_layer(row, snapshot, prototypes, cards),
        LAYERS.compile_global_layer(row, snapshot),
    )


class FastAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assets = PROJECT_ROOT / "assets"
        cls.prototypes = PROTOTYPES.PrototypeIndex.load(
            assets / "official_public_prototypes_v1.json",
            assets / "official_full_engine_prototypes_v2.json",
        )

    def _assert_trajectories(self, trajectories) -> tuple[int, int]:
        decisions = 0
        elapsed_ns = 0
        for trajectory in trajectories:
            knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
            assembler = ASSEMBLY.IncrementalRecordAssembler()
            for decision in trajectory.decisions:
                snapshot = knowledge.consume(decision.observation, decision.event_cursor)
                row = decision.row()
                layers = _layers(row, snapshot, self.prototypes)
                expected = LAYERS.assemble_canonical_record(row, *layers)
                started = time.perf_counter_ns()
                actual = assembler.assemble(row, *layers)
                elapsed_ns += time.perf_counter_ns() - started
                self.assertEqual(actual, expected)
                decisions += 1
            stats = assembler.stats.snapshot()
            self.assertEqual(stats["calls"], len(trajectory.decisions))
            self.assertEqual(stats["deck_cache_misses"], 1)
            self.assertEqual(stats["deck_cache_hits"], len(trajectory.decisions) - 1)
            self.assertGreater(stats["elapsed_ns"], 0)
            self.assertGreater(stats["mean_elapsed_ns"], 0)
        return decisions, elapsed_ns

    def test_exact_record_parity_on_35_decision_fixture(self) -> None:
        decisions, elapsed_ns = self._assert_trajectories(
            BENCHMARK.load_parity_trajectories()
        )
        self.assertEqual(decisions, 35)
        self.assertGreater(elapsed_ns, 0)

    def test_exact_record_parity_on_525_decision_fixture(self) -> None:
        decisions, elapsed_ns = self._assert_trajectories(
            EXTENDED.load_extended_trajectories()
        )
        self.assertEqual(decisions, 525)
        self.assertGreater(elapsed_ns, 0)

    def test_cached_templates_never_escape_as_mutable_aliases(self) -> None:
        trajectory = BENCHMARK.load_parity_trajectories()[0]
        knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
        decision = trajectory.decisions[0]
        snapshot = knowledge.consume(decision.observation, decision.event_cursor)
        row = decision.row()
        layers = _layers(row, snapshot, self.prototypes)
        expected = LAYERS.assemble_canonical_record(row, *layers)
        assembler = ASSEMBLY.IncrementalRecordAssembler()

        polluted = assembler.assemble(row, *layers)
        polluted["actor"]["global_cat"][0] = -999
        polluted["actor"]["card_cat"][0][0] = -999
        polluted["actor"]["option_skill_id"].append(-999)
        polluted["target"]["ordered_action"].append(-999)
        polluted["audit"]["identity"]["episode_id"] = -999
        polluted["registered_deck"][0] = -999

        clean = assembler.assemble(row, *layers)
        self.assertEqual(clean, expected)
        self.assertIsNot(clean["actor"]["card_cat"], polluted["actor"]["card_cat"])
        self.assertIsNot(clean["actor"]["card_cat"][0], polluted["actor"]["card_cat"][0])
        self.assertIsNot(clean["registered_deck"], polluted["registered_deck"])
        stats = assembler.stats.snapshot()
        self.assertGreater(stats["materialization_cache_hits"], 0)
        self.assertEqual(stats["deck_cache_hits"], 1)
        self.assertEqual(stats["deck_cache_misses"], 1)

    def test_reset_battle_invalidates_private_templates_and_deck(self) -> None:
        trajectory = BENCHMARK.load_parity_trajectories()[0]
        decision = trajectory.decisions[0]
        knowledge = STATE.CausalKnowledge(trajectory.actor, trajectory.deck)
        snapshot = knowledge.consume(decision.observation, decision.event_cursor)
        row = decision.row()
        layers = _layers(row, snapshot, self.prototypes)
        assembler = ASSEMBLY.IncrementalRecordAssembler()
        assembler.assemble(row, *layers)
        assembler.reset_battle()
        self.assertEqual(assembler.assemble(row, *layers), LAYERS.assemble_canonical_record(row, *layers))
        stats = assembler.stats.snapshot()
        self.assertEqual(stats["deck_cache_misses"], 2)


if __name__ == "__main__":
    unittest.main()
