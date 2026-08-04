from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import torch


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.legacy_codecs import (  # noqa: E402
    marnie_prize_control_v4_static_fields,
    policy_codec_v1_to_idonly_codec_v1,
)


def policy_batch(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    return {
        "global_cat": torch.tensor(
            [[11, 12, 2, 1, 0, 2, 3, 29]], dtype=torch.long, device=device
        ),
        "global_num": torch.tensor(
            [
                [
                    0.5,
                    0.25,
                    9.0,
                    8.0,
                    0.75,
                    0.5,
                    0.4,
                    0.3,
                    0.5,
                    1.0 / 3.0,
                    0.125,
                    0.2,
                    0.3,
                    7.0,
                    0.4,
                    0.6,
                ]
            ],
            dtype=torch.float32,
            device=device,
        ),
        "entity_cat": torch.tensor(
            [
                [
                    [101, 1, 1, 1, 2, 3],
                    [102, 1, 5, 1, 1, 0],
                    [103, 2, 6, 1, 2, 4],
                    [104, 0, 10, 1, 6, 0],
                    [105, 1, 13, 1, 3, 0],
                ]
            ],
            dtype=torch.long,
            device=device,
        ),
        "entity_num": torch.tensor(
            [
                [
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
                    [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 0.0, 0.0, 0.0, 0.0],
                    [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 0.0, 0.0, 0.0, 0.0],
                    [3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 0.0, 0.0, 0.0, 0.0],
                    [4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 1.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        ),
        "entity_parent": torch.tensor(
            [[-1, -1, -1, -1, 3]], dtype=torch.long, device=device
        ),
        "entity_mask": torch.tensor(
            [[True, True, True, True, True]], device=device
        ),
        "option_cat": torch.tensor(
            [
                [
                    [7, 3, 2, 1, 104, 102, 999, 8, 4, 2, 1, 3],
                    [13, 0, 0, 1, 0, 0, 77, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ],
            dtype=torch.long,
            device=device,
        ),
        "option_mask": torch.tensor([[True, True, False]], device=device),
        "min_count": torch.tensor([1], dtype=torch.long, device=device),
        "max_count": torch.tensor([2], dtype=torch.long, device=device),
    }


class LegacyCodecTest(unittest.TestCase):
    def test_policy_codec_maps_to_exact_idonly_layout(self) -> None:
        converted = policy_codec_v1_to_idonly_codec_v1(policy_batch())

        self.assertEqual(converted["global_cat"].tolist(), [[11, 12, 2, 29]])
        self.assertTrue(
            torch.equal(
                converted["global_num"],
                torch.tensor(
                    [
                        [
                            0.5,
                            0.25,
                            0.75,
                            0.5,
                            0.4,
                            0.3,
                            0.5,
                            1.0 / 3.0,
                            0.125,
                            0.2,
                            0.3,
                            0.5,
                        ]
                    ]
                ),
            )
        )
        self.assertEqual(
            converted["entity_cat"].tolist(),
            [
                [
                    [101, 1, 1, 1, 2, 3, 0],
                    [103, 2, 5, 1, 2, 4, 0],
                    [104, 0, 8, 1, 6, 0, 0],
                    [105, 1, 11, 1, 3, 0, 3],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ],
        )
        self.assertTrue(
            torch.equal(
                converted["entity_num"][0, :4],
                torch.tensor(
                    [
                        [0.3, 0.4, 0.5, 0.6, 1.0],
                        [2.3, 2.4, 2.5, 2.6, 0.0],
                        [3.3, 3.4, 3.5, 3.6, 0.0],
                        [4.3, 4.4, 4.5, 4.6, 1.0],
                    ],
                    dtype=torch.float32,
                ),
            )
        )
        self.assertEqual(
            converted["option_cat"].tolist(),
            [
                [
                    [7, 3, 2, 1, 104, 102, 8, 1, 3, 3, 0, 1],
                    [13, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ],
        )
        self.assertEqual(converted["entity_mask"].tolist(), [[True] * 4 + [False]])
        self.assertEqual(converted["option_mask"].tolist(), [[True, True, False]])
        self.assertEqual(converted["min_count"].tolist(), [1])
        self.assertEqual(converted["max_count"].tolist(), [2])
        self.assertEqual(converted["action_min_count"].tolist(), [1])
        self.assertEqual(converted["action_max_count"].tolist(), [2])

    def test_action_contract_preserves_required_count_above_model_horizon(self) -> None:
        batch = policy_batch()
        batch["min_count"] = torch.tensor([20], dtype=torch.long)
        batch["max_count"] = torch.tensor([20], dtype=torch.long)
        converted = policy_codec_v1_to_idonly_codec_v1(
            batch,
            max_action_steps=16,
        )
        self.assertEqual(converted["min_count"].tolist(), [16])
        self.assertEqual(converted["max_count"].tolist(), [16])
        self.assertEqual(converted["action_min_count"].tolist(), [20])
        self.assertEqual(converted["action_max_count"].tolist(), [20])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_conversion_stays_on_cuda(self) -> None:
        converted = policy_codec_v1_to_idonly_codec_v1(policy_batch("cuda"))
        self.assertTrue(all(value.is_cuda for value in converted.values()))

    def test_hot_path_has_no_host_materialization(self) -> None:
        source = inspect.getsource(policy_codec_v1_to_idonly_codec_v1)
        for forbidden in (".cpu(", ".item(", ".tolist(", ".numpy("):
            self.assertNotIn(forbidden, source)

    def test_conversion_pads_to_frozen_model_capacity(self) -> None:
        converted = policy_codec_v1_to_idonly_codec_v1(
            policy_batch(),
            target_entity_capacity=192,
            target_option_capacity=128,
        )
        self.assertEqual(tuple(converted["entity_cat"].shape), (1, 192, 7))
        self.assertEqual(tuple(converted["entity_num"].shape), (1, 192, 5))
        self.assertEqual(tuple(converted["entity_mask"].shape), (1, 192))
        self.assertEqual(tuple(converted["option_cat"].shape), (1, 128, 12))
        self.assertEqual(tuple(converted["option_mask"].shape), (1, 128))
        self.assertTrue(converted["entity_mask"][0, :4].all())
        self.assertFalse(converted["entity_mask"][0, 4:].any())
        self.assertTrue(converted["option_mask"][0, :2].all())
        self.assertFalse(converted["option_mask"][0, 2:].any())

    def test_marnie_control_fields_are_static_registered_deck_features(self) -> None:
        deck = [101] * 4 + [202] * 2 + list(range(300, 354))
        fields = marnie_prize_control_v4_static_fields(deck, "cpu")
        self.assertEqual(tuple(fields["prize_card"].shape), (1, 56))
        self.assertEqual(fields["prize_card"][0, :2].tolist(), [101, 202])
        self.assertEqual(fields["prize_num"][0, 0, 0].item(), 1.0)
        self.assertEqual(fields["prize_num"][0, 1, 0].item(), 0.5)
        self.assertFalse(fields["prize_num"][..., 1:].any())
        self.assertTrue(fields["prize_mask"].all())


if __name__ == "__main__":
    unittest.main()
