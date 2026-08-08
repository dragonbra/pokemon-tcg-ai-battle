from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from evaluation.runtime.seeded import (
    SeededBattle,
    build_seeded_runtime,
    source_tree_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
DECK0 = ROOT / "evaluation/arena/opponents/alakazam_dudunsparce_01/deck.csv"
DECK1 = ROOT / "evaluation/arena/opponents/dragapult_ex_01/deck.csv"


def read_deck(path: Path) -> list[int]:
    cards = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(cards) != 60:
        raise AssertionError(f"fixture deck must contain 60 cards: {path}")
    return cards


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_action(observation: dict, *, first_player: int = 0) -> list[int]:
    selection = observation["select"]
    context = int(selection["context"])
    options = selection["option"]
    minimum = int(selection["minCount"])
    maximum = int(selection["maxCount"])
    if context == 41:
        return [0 if first_player == 0 else 1]
    if int(selection["type"]) == 0:
        end = next(
            (index for index, option in enumerate(options) if option.get("type") == 14),
            None,
        )
        if end is not None and minimum <= 1 <= maximum:
            return [end]
    return list(range(minimum))


class SeededOfficialRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = build_seeded_runtime()
        cls.deck0 = read_deck(DECK0)
        cls.deck1 = read_deck(DECK1)

    def test_build_is_hash_addressed_and_outside_official_source(self) -> None:
        self.assertTrue(self.manifest.library_path.is_file())
        self.assertEqual(self.manifest.library_path.suffix, ".so")
        self.assertTrue(self.manifest.library_path.is_relative_to(ROOT / "engine/build"))
        self.assertEqual(self.manifest.runtime_version, 2)
        self.assertEqual(
            self.manifest.library_path.parent,
            ROOT / "engine/build/seeded_official/0002",
        )
        self.assertEqual(self.manifest.official_source_sha256, source_tree_sha256())
        self.assertEqual(
            hashlib.sha256(self.manifest.library_path.read_bytes()).hexdigest(),
            self.manifest.library_sha256,
        )
        self.assertEqual(self.manifest.abi_schema, "seeded_official_engine_abi_v1")

    def _trace(self, *, engine_seed: int, first_player: int, steps: int = 40) -> list[dict]:
        battle = SeededBattle(
            self.manifest.library_path,
            self.deck0,
            self.deck1,
            engine_seed=engine_seed,
            search_seed=engine_seed ^ 0x5EA2C001,
        )
        trace: list[dict] = []
        try:
            observation = battle.start()
            for _ in range(steps):
                action = deterministic_action(observation, first_player=first_player)
                trace.append({"observation": observation, "action": action})
                observation = battle.select(action)
                if int((observation.get("current") or {}).get("result", -1)) >= 0:
                    break
            trace.append({"observation": observation})
            return trace
        finally:
            battle.finish()

    def test_same_seed_seat_and_actions_have_identical_trace(self) -> None:
        left = self._trace(engine_seed=424242, first_player=0)
        right = self._trace(engine_seed=424242, first_player=0)
        self.assertEqual(canonical_sha256(left), canonical_sha256(right))

    def test_different_seed_changes_post_setup_trace(self) -> None:
        left = self._trace(engine_seed=424242, first_player=0)
        right = self._trace(engine_seed=424243, first_player=0)
        self.assertNotEqual(canonical_sha256(left), canonical_sha256(right))

    def test_same_seed_can_force_opposite_first_player_with_fixed_deck_slots(self) -> None:
        first = self._trace(engine_seed=919191, first_player=0, steps=2)
        second = self._trace(engine_seed=919191, first_player=1, steps=2)
        self.assertEqual(first[0]["observation"], second[0]["observation"])
        self.assertEqual(first[1]["observation"]["current"]["firstPlayer"], 0)
        self.assertEqual(second[1]["observation"]["current"]["firstPlayer"], 1)


if __name__ == "__main__":
    unittest.main()
