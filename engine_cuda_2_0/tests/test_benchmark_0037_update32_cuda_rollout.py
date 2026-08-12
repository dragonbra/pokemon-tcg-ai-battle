from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from engine_cuda_2_0.tools import benchmark_0037_update32_cuda_rollout as benchmark


class Benchmark0037Update32CudaRolloutTest(unittest.TestCase):
    def test_load_schedule_preserves_order_seat_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deck_root = root / "decks"
            opponent_root = deck_root / "opponent_a"
            opponent_root.mkdir(parents=True)
            focal = tuple(range(1, 61))
            opponent = tuple(range(101, 161))
            (opponent_root / "deck.csv").write_text(
                "\n".join(str(card) for card in opponent) + "\n", encoding="ascii"
            )
            schedule = root / "schedule.json"
            schedule.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "engine_seed": 101,
                                "focal_first": True,
                                "opponent_id": "opponent_a",
                            },
                            {
                                "engine_seed": 101,
                                "focal_first": False,
                                "opponent_id": "opponent_a",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            loaded = benchmark.load_schedule(schedule, deck_root, focal)

        self.assertEqual(loaded.engine_seeds, (101, 101))
        self.assertEqual(loaded.focal_players, (0, 1))
        self.assertEqual(loaded.deck_rows[0], (focal, opponent))
        self.assertEqual(loaded.deck_rows[1], (opponent, focal))
        self.assertEqual(loaded.opponent_ids, ("opponent_a", "opponent_a"))

    def test_load_schedule_resolves_numbered_directory_by_manifest_deck_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            deck_root = root / "decks"
            opponent_root = deck_root / "001_readable_name"
            opponent_root.mkdir(parents=True)
            focal = tuple(range(1, 61))
            opponent = tuple(range(101, 161))
            (opponent_root / "deck.csv").write_text(
                "\n".join(str(card) for card in opponent) + "\n", encoding="ascii"
            )
            (opponent_root / "manifest.json").write_text(
                json.dumps({"deck_id": "opaque_internal_id_deadbeef0000"}),
                encoding="utf-8",
            )
            schedule = root / "schedule.json"
            schedule.write_text(
                json.dumps(
                    {
                        "jobs": [
                            {
                                "engine_seed": 17,
                                "focal_first": True,
                                "opponent_id": "opaque_internal_id_deadbeef0000",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            loaded = benchmark.load_schedule(schedule, deck_root, focal)

        self.assertEqual(loaded.deck_rows, ((focal, opponent),))

    def test_expanded_portable_state_restores_prototype_aliases(self) -> None:
        tensor = torch.tensor([1.0])
        expanded = benchmark.expanded_portable_state(
            {"prototype_encoder.example": tensor, "action_decoder.bias": tensor.clone()}
        )
        self.assertIs(expanded["state_encoder.prototypes.example"], tensor)
        self.assertIs(expanded["option_encoder.prototypes.example"], tensor)

    def test_validate_completion_rejects_errors_or_unfinished_lanes(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not finish cleanly"):
            benchmark.validate_completion(total=8, completed=7, errors=1)

    def test_compact_semantic_batch_selects_only_routed_rows(self) -> None:
        batch = {
            "global_cat": torch.tensor([[1], [2], [3]]),
            "option_mask": torch.tensor([[True, False], [False, True], [True, True]]),
        }
        compact = benchmark.compact_semantic_batch(
            batch, torch.tensor([2, 0], dtype=torch.long)
        )
        self.assertEqual(compact["global_cat"].tolist(), [[3], [1]])
        self.assertEqual(
            compact["option_mask"].tolist(), [[True, True], [True, False]]
        )

    def test_routing_row_metrics_distinguish_dense_work_from_routed_rows(self) -> None:
        dense = benchmark.routing_row_metrics(
            routing_mode="dense_masked",
            total=64,
            decisions=10,
            routed_rows=123,
        )
        compact = benchmark.routing_row_metrics(
            routing_mode="compact",
            total=64,
            decisions=10,
            routed_rows=123,
        )

        self.assertEqual(dense["actual_decoder_rows"], 1280)
        self.assertEqual(dense["actual_dense_decoder_rows_avoided"], 0)
        self.assertEqual(compact["actual_decoder_rows"], 123)
        self.assertEqual(compact["actual_dense_decoder_rows_avoided"], 1157)

    def test_compact_semantic_prefixes_trims_only_masked_suffixes(self) -> None:
        batch = {
            "card_cat": torch.arange(12).view(2, 3, 2),
            "card_mask": torch.tensor([[True, False, False], [True, True, False]]),
            "option_cat": torch.arange(16).view(2, 4, 2),
            "option_mask": torch.tensor(
                [[True, False, False, False], [True, True, True, False]]
            ),
            "global_cat": torch.ones((2, 4), dtype=torch.long),
        }
        compact, widths = benchmark.compact_semantic_prefixes(batch)

        self.assertEqual(widths, {"card": 2, "option": 3})
        self.assertEqual(compact["card_cat"].shape, (2, 2, 2))
        self.assertEqual(compact["option_cat"].shape, (2, 3, 2))
        self.assertIs(compact["global_cat"], batch["global_cat"])


if __name__ == "__main__":
    unittest.main()
