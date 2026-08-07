from __future__ import annotations

import inspect
import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


CUDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ROOT / "python"))

from ptcg_cuda_engine.semantic0031_bridge import (  # noqa: E402
    FIELD_NOT_APPLICABLE,
    FIELD_PRESENT,
    FIELD_UNKNOWN,
    _REQUIRED_PACKAGE_FILES,
    Semantic0031DeviceAdapter,
    inspect_semantic0031_archive,
    load_semantic0031_package,
    policy_codec_v1_to_semantic0031_v2,
    semantic0031_greedy_decode_device,
    semantic0031_static_fields,
    semantic0031_v2_ready_batch,
)


def policy_batch() -> dict[str, torch.Tensor]:
    global_cat = torch.tensor([[14, 7, 1, 1, 0, 1, 2, 5]], dtype=torch.long)
    global_num = torch.zeros((1, 16), dtype=torch.float32)
    global_num[0] = torch.tensor(
        [
            0.5,
            0.12,
            0.0,
            0.0,
            0.5,
            0.4,
            0.35,
            0.25,
            0.5,
            2 / 6,
            2 / 128,
            0.1,
            0.2,
            3 / 60,
            0.2,
            0.4,
        ]
    )
    entity_cat = torch.tensor(
        [
            [
                [848, 1, 1, 1, 2, 3],
                [305, 1, 13, 1, 3, 0],
                [861, 2, 6, 1, 2, 1],
                [999, 1, 5, 1, 1, 0],
            ]
        ],
        dtype=torch.long,
    )
    entity_num = torch.zeros((1, 4, 10), dtype=torch.float32)
    entity_num[0, 0] = torch.tensor([0.25, 0.3, 0.05, 0.2, 0.25, 0.25, 1.0, 0.0, 0.0, 0.0])
    entity_num[0, 2] = torch.tensor([0.4, 0.5, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    option_cat = torch.tensor(
        [
            [
                [14, 5, 5, 1, 848, 861, 7, 0, 1, 3, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ],
        dtype=torch.long,
    )
    batch = {
        "global_cat": global_cat,
        "global_num": global_num,
        "entity_cat": entity_cat,
        "entity_num": entity_num,
        "entity_parent": torch.tensor([[-1, 0, -1, -1]], dtype=torch.long),
        "entity_mask": torch.tensor([[True, True, True, True]]),
        "option_cat": option_cat,
        "option_num": torch.zeros((1, 3, 4), dtype=torch.float32),
        "option_equiv": torch.tensor([[0, 1, 0]], dtype=torch.long),
        "option_mask": torch.tensor([[True, True, False]]),
        "min_count": torch.tensor([1], dtype=torch.long),
        "max_count": torch.tensor([2], dtype=torch.long),
    }
    batch.update(semantic0031_static_fields([848] * 2 + [305] * 58, "cpu"))
    return batch


class SemanticProjectionTests(unittest.TestCase):
    def test_shapes_field_states_and_option_alignment(self) -> None:
        converted = policy_codec_v1_to_semantic0031_v2(policy_batch())
        self.assertEqual(tuple(converted["global_cat"].shape), (1, 12))
        self.assertEqual(tuple(converted["global_num"].shape), (1, 24))
        self.assertEqual(tuple(converted["card_cat"].shape), (1, 4, 9))
        self.assertEqual(tuple(converted["card_num"].shape), (1, 4, 7))
        self.assertEqual(tuple(converted["option_cat"].shape), (1, 3, 19))
        self.assertEqual(tuple(converted["option_num"].shape), (1, 3, 2))
        self.assertEqual(converted["global_cat"][0].tolist(), [14, 7, 1, 2, 1, 2, 1, 4, 2, 2, 1, 0])
        self.assertEqual(
            converted["global_num"][0, :15].tolist(),
            [
                10.0,
                6.0,
                30.0,
                24.0,
                7.0,
                5.0,
                3.0,
                2.0,
                1.0,
                2.0,
                2.0,
                1.0,
                2.0,
                30.0,
                2.0,
            ],
        )
        self.assertEqual(int(converted["global_state"][0, 15]), FIELD_UNKNOWN)
        self.assertEqual(float(converted["global_num"][0, 16]), 5.0)
        self.assertEqual(int(converted["global_state"][0, 16]), FIELD_PRESENT)

        self.assertEqual(converted["card_mask"].tolist(), [[True, True, True, False]])
        self.assertEqual(converted["card_cat"][0, 0].tolist(), [848, 0, 1, 1, 1, 2, 3, 0, 1])
        self.assertEqual(converted["card_cat"][0, 1, 3].item(), 11)
        self.assertEqual(converted["card_cat"][0, 2, 3].item(), 5)
        self.assertEqual(converted["card_parent"].tolist(), [[0, 1, 0, 0]])
        self.assertEqual(
            converted["card_num"][0, 0].tolist(), [100.0, 120.0, 2.0, 0.0, 1.0, 1.0, 1.0]
        )
        self.assertEqual(int(converted["card_state"][0, 0, 3]), FIELD_UNKNOWN)
        self.assertTrue(torch.all(converted["card_state"][0, 1] == FIELD_NOT_APPLICABLE))

        first = converted["option_cat"][0, 0]
        self.assertEqual(first[:8].tolist(), [14, 1, 5, 2, 5, 848, 861, 7])
        self.assertEqual(first[9:16].tolist(), [14, 7, 0, 0, 1, 1, 1])
        self.assertEqual(converted["option_source"].tolist(), [[1, 0, 0]])
        self.assertEqual(converted["option_target"].tolist(), [[3, 0, 0]])
        self.assertEqual(converted["option_num"][0, 1].tolist(), [3.0, 0.0])
        self.assertEqual(converted["option_state"][0, 1].tolist(), [FIELD_PRESENT, FIELD_UNKNOWN])
        self.assertFalse(bool(converted["event_mask"].any()))
        self.assertFalse(bool(converted["option_skill_mask"].any()))
        self.assertFalse(bool(converted["option_effect_mask"].any()))

    def test_registered_resources_assert_only_initial_counts(self) -> None:
        converted = policy_codec_v1_to_semantic0031_v2(policy_batch())
        self.assertEqual(converted["resource_cat"][0, :, 0].tolist(), [305, 848])
        self.assertEqual(converted["resource_num"][0, :, 0].tolist(), [58.0, 2.0])
        self.assertTrue(torch.all(converted["resource_state"][0, :, 0] == FIELD_PRESENT))
        self.assertTrue(torch.all(converted["resource_state"][0, :, 1:] == FIELD_UNKNOWN))

    def test_semantic_history_projects_setup_events(self) -> None:
        batch = policy_batch()
        batch["semantic0031_history_total_count"] = torch.tensor([9], dtype=torch.int64)
        batch["semantic0031_history_write_index"] = torch.tensor([9], dtype=torch.int32)
        batch["semantic0031_history_log_type"] = torch.zeros((1, 64), dtype=torch.uint8)
        batch["semantic0031_history_param_count"] = torch.zeros((1, 64), dtype=torch.uint8)
        batch["semantic0031_history_params"] = torch.zeros((1, 64, 7), dtype=torch.int32)
        batch["semantic0031_history_log_type"][0, :9] = torch.tensor(
            [0, 2, 4, 4, 6, 7, 22, 3, 23], dtype=torch.uint8
        )
        batch["semantic0031_history_param_count"][0, :9] = torch.tensor(
            [1, 1, 3, 3, 5, 3, 2, 1, 2], dtype=torch.uint8
        )
        batch["semantic0031_history_params"][0, 0, 0] = 0
        batch["semantic0031_history_params"][0, 1, 0] = 0
        batch["semantic0031_history_params"][0, 2, :3] = torch.tensor([0, 848, 2], dtype=torch.int32)
        batch["semantic0031_history_params"][0, 3, :3] = torch.tensor([1, 848, 2], dtype=torch.int32)
        batch["semantic0031_history_params"][0, 4, :5] = torch.tensor(
            [0, 848, 1, 2, 3], dtype=torch.int32
        )
        batch["semantic0031_history_params"][0, 5, :3] = torch.tensor(
            [1, 6, 2], dtype=torch.int32
        )
        batch["semantic0031_history_params"][0, 6, :2] = torch.tensor([1, 1], dtype=torch.int32)
        batch["semantic0031_history_params"][0, 7, 0] = 0
        batch["semantic0031_history_params"][0, 8, :2] = torch.tensor([1, 9], dtype=torch.int32)

        converted = policy_codec_v1_to_semantic0031_v2(batch)

        self.assertEqual(tuple(converted["event_cat"].shape), (1, 64, 31))
        self.assertEqual(
            converted["event_mask"][0, :10].tolist(),
            [True, True, True, True, True, True, True, True, True, False],
        )
        self.assertEqual(converted["event_cat"][0, :9, 0].tolist(), [1, 3, 5, 6, 7, 8, 23, 4, 24])
        self.assertEqual(converted["event_cat"][0, :9, 1].tolist(), [1, 1, 1, 2, 1, 2, 2, 1, 3])
        self.assertEqual(converted["event_cat"][0, :9, 2].tolist(), [0, 0, 848, 0, 848, 0, 0, 0, 0])
        self.assertEqual(converted["event_cat"][0, :9, 5].tolist(), [0, 0, 0, 0, 3, 7, 0, 0, 0])
        self.assertEqual(converted["event_cat"][0, :9, 6].tolist(), [0, 0, 0, 0, 4, 3, 0, 0, 0])
        self.assertEqual(converted["event_cat"][0, :9, 7].tolist(), [1, 1, 2, 1, 2, 1, 1, 1, 1])
        self.assertEqual(converted["event_cat"][0, :9, 8].tolist(), [1, 1, 2, 1, 2, 1, 1, 1, 1])
        self.assertEqual(converted["event_cat"][0, :9, 14].tolist(), [0, 0, 3, 0, 2, 0, 0, 0, 0])
        self.assertEqual(converted["event_cat"][0, :9, 22].tolist(), [0, 0, 0, 0, 0, 0, 2, 0, 0])
        self.assertEqual(converted["event_cat"][0, :9, 24].tolist(), [0, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertEqual(converted["event_cat"][0, :9, 25].tolist(), [0, 0, 0, 0, 0, 0, 0, 0, 10])
        self.assertEqual(
            converted["event_num"][0, :9, 0].tolist(),
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        )
        self.assertTrue(torch.all(converted["event_state"][0, :9, 0] == FIELD_PRESENT))
        self.assertTrue(torch.all(converted["event_state"][0, :9, 1:] == FIELD_UNKNOWN))
        self.assertTrue(torch.all(converted["event_state"][0, 9:] == 0))
        self.assertEqual(converted["event_source"][0, :9].tolist(), [0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_semantic_history_reorders_wrapped_ring_oldest_to_newest(self) -> None:
        batch = policy_batch()
        total_count = 70
        capacity = 64
        write_index = total_count % capacity
        batch["semantic0031_history_total_count"] = torch.tensor(
            [total_count], dtype=torch.int64
        )
        batch["semantic0031_history_write_index"] = torch.tensor(
            [write_index], dtype=torch.int32
        )
        batch["semantic0031_history_log_type"] = torch.zeros(
            (1, capacity), dtype=torch.uint8
        )
        batch["semantic0031_history_param_count"] = torch.zeros(
            (1, capacity), dtype=torch.uint8
        )
        batch["semantic0031_history_params"] = torch.zeros(
            (1, capacity, 7), dtype=torch.int32
        )
        retained_events = list(range(total_count - capacity, total_count))
        for event_index in retained_events:
            slot = event_index % capacity
            batch["semantic0031_history_log_type"][0, slot] = event_index % 24

        converted = policy_codec_v1_to_semantic0031_v2(batch)

        self.assertTrue(bool(converted["event_mask"].all()))
        self.assertEqual(
            converted["event_cat"][0, :, 0].tolist(),
            [(event_index % 24) + 1 for event_index in retained_events],
        )
        self.assertEqual(
            converted["event_num"][0, :, 0].tolist(),
            [float(value) for value in range(capacity - 1, -1, -1)],
        )

    def test_semantic_history_projects_has_basic_exactly(self) -> None:
        batch = policy_batch()
        batch["semantic0031_history_total_count"] = torch.tensor([2], dtype=torch.int64)
        batch["semantic0031_history_write_index"] = torch.tensor([2], dtype=torch.int32)
        batch["semantic0031_history_log_type"] = torch.zeros((1, 64), dtype=torch.uint8)
        batch["semantic0031_history_param_count"] = torch.zeros((1, 64), dtype=torch.uint8)
        batch["semantic0031_history_params"] = torch.zeros((1, 64, 7), dtype=torch.int32)
        batch["semantic0031_history_log_type"][0, :2] = torch.tensor([1, 1], dtype=torch.uint8)
        batch["semantic0031_history_param_count"][0, :2] = 2
        batch["semantic0031_history_params"][0, 0, :2] = torch.tensor([0, 1], dtype=torch.int32)
        batch["semantic0031_history_params"][0, 1, :2] = torch.tensor([1, 0], dtype=torch.int32)

        converted = policy_codec_v1_to_semantic0031_v2(batch)

        self.assertEqual(converted["event_mask"][0, :3].tolist(), [True, True, False])
        self.assertEqual(converted["event_cat"][0, :2, 0].tolist(), [2, 2])
        self.assertEqual(converted["event_cat"][0, :2, 1].tolist(), [1, 2])
        self.assertEqual(converted["event_cat"][0, :2, 21].tolist(), [2, 1])

    def test_semantic_history_projects_all_gameplay_event_payloads(self) -> None:
        batch = policy_batch()
        event_types = torch.arange(8, 22, dtype=torch.uint8)
        params = [
            [0, 848, 2, 861, 3],
            [1, 861, 3, 848, 2],
            [0, 848, 2],
            [0, 305, 1, 848, 2],
            [0, 848, 2, 861, 3],
            [1, 861, 3, 848, 2],
            [0, 305, 1, 848, 2, 861, 3],
            [0, 848, 2, 777],
            [1, 861, 3, -30, 1],
            [0, 0, 848, 2],
            [1, 1, 861, 3],
            [0, 0, 848, 2],
            [1, 1, 861, 3],
            [0, 0, 848, 2],
        ]
        batch["semantic0031_history_total_count"] = torch.tensor([14], dtype=torch.int64)
        batch["semantic0031_history_write_index"] = torch.tensor([14], dtype=torch.int32)
        batch["semantic0031_history_log_type"] = torch.zeros((1, 64), dtype=torch.uint8)
        batch["semantic0031_history_param_count"] = torch.zeros((1, 64), dtype=torch.uint8)
        batch["semantic0031_history_params"] = torch.zeros((1, 64, 7), dtype=torch.int32)
        batch["semantic0031_history_log_type"][0, :14] = event_types
        for index, values in enumerate(params):
            batch["semantic0031_history_param_count"][0, index] = len(values)
            batch["semantic0031_history_params"][0, index, : len(values)] = torch.tensor(
                values, dtype=torch.int32
            )

        converted = policy_codec_v1_to_semantic0031_v2(batch)
        cat = converted["event_cat"][0, :14]

        self.assertEqual(cat[:, 0].tolist(), list(range(9, 23)))
        self.assertEqual(cat[:, 1].tolist(), [1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1])
        self.assertEqual(cat[0, [10, 11, 16, 17]].tolist(), [848, 861, 3, 4])
        self.assertEqual(cat[1, [12, 13, 18, 19]].tolist(), [861, 848, 4, 3])
        self.assertEqual(cat[2, [2, 7, 8, 14]].tolist(), [848, 2, 2, 3])
        for index in (3, 4, 5):
            self.assertEqual(int(cat[index, 9]), 2)
            self.assertGreater(int(cat[index, 3]), 0)
            self.assertGreater(int(cat[index, 15]), 0)
        self.assertEqual(cat[6, [2, 12, 13, 14, 18, 19]].tolist(), [305, 848, 861, 2, 3, 4])
        self.assertEqual(cat[7, [2, 4, 14]].tolist(), [848, 777, 3])
        self.assertEqual(cat[8, [2, 14, 23]].tolist(), [861, 4, 2])
        self.assertEqual(cat[9:14, 20].tolist(), [1, 2, 1, 2, 1])
        self.assertEqual(converted["event_num"][0, 8, 1].item(), -30.0)
        self.assertEqual(converted["event_state"][0, 8, 1].item(), FIELD_PRESENT)
        self.assertTrue(torch.all(converted["event_state"][0, :14, 2:] == FIELD_UNKNOWN))

    def test_semantic0031_v2_ready_batch_normalizes_cuda_contract(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for semantic0031 v2 ready-batch normalization")
        converted = policy_codec_v1_to_semantic0031_v2(policy_batch())
        raw = {
            name: value.to("cuda", non_blocking=True)
            for name, value in converted.items()
        }
        for name in (
            "card_mask",
            "resource_mask",
            "event_mask",
            "option_mask",
            "option_skill_mask",
            "option_effect_mask",
        ):
            raw[name] = raw[name].to(dtype=torch.uint8)

        normalized = semantic0031_v2_ready_batch(raw, max_action_steps=3)

        for name in (
            "card_mask",
            "resource_mask",
            "event_mask",
            "option_mask",
            "option_skill_mask",
            "option_effect_mask",
        ):
            self.assertIs(normalized[name].dtype, torch.bool)
            self.assertEqual(normalized[name].device.type, "cuda")
        self.assertEqual(tuple(normalized["min_count"].shape), (1,))
        self.assertEqual(tuple(normalized["max_count"].shape), (1,))
        self.assertEqual(tuple(normalized["targets"].shape), (1, 1))
        self.assertTrue(torch.equal(normalized["max_count"], torch.tensor([2], device="cuda")))

    def test_hot_path_has_no_host_materialization(self) -> None:
        sources = "\n".join(
            inspect.getsource(value)
            for value in (
                policy_codec_v1_to_semantic0031_v2,
                semantic0031_v2_ready_batch,
                semantic0031_greedy_decode_device,
                Semantic0031DeviceAdapter._encode_options,
                Semantic0031DeviceAdapter.act_device_shared_static,
                Semantic0031DeviceAdapter.act_device_semantic0031_v2,
                Semantic0031DeviceAdapter.act_ready_lanes_semantic0031_v2,
            )
        )
        for forbidden in (".cpu(", ".item(", ".tolist(", ".numpy("):
            self.assertNotIn(forbidden, sources)


class _OptionBias(nn.Module):
    def forward(self, options: torch.Tensor) -> torch.Tensor:
        return options[..., :1]


class _Stop(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.new_full((hidden.shape[0], 1), -100.0)


class _Recurrent(nn.Module):
    def forward(self, selected: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + selected * 0.0


class DecoderTests(unittest.TestCase):
    def test_decoder_is_unique_bounded_and_route_masked(self) -> None:
        initial = nn.Linear(2, 2)
        key = nn.Linear(2, 2, bias=False)
        query = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            initial.weight.zero_()
            initial.bias.zero_()
            key.weight.copy_(torch.eye(2))
            query.weight.zero_()
        decoder = SimpleNamespace(
            initial=initial,
            key=key,
            query=query,
            option_bias=_OptionBias(),
            stop=_Stop(),
            recurrent=_Recurrent(),
        )
        batch = SimpleNamespace(
            option_mask=torch.tensor([[True, True, True], [True, True, True]]),
            min_count=torch.tensor([1, 1]),
            max_count=torch.tensor([2, 2]),
        )
        options = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]]] * 2)
        actions, lengths = semantic0031_greedy_decode_device(
            decoder,
            batch,
            options,
            torch.zeros((2, 2)),
            max_select=4,
            route_mask=torch.tensor([True, False]),
        )
        self.assertEqual(actions[0].tolist(), [1, 2, -1, -1])
        self.assertEqual(int(lengths[0]), 2)
        self.assertEqual(actions[1].tolist(), [-1, -1, -1, -1])
        self.assertEqual(int(lengths[1]), 0)


class ArchiveSafetyTests(unittest.TestCase):
    def _write_tar(self, root: Path, member: tarfile.TarInfo, data: bytes = b"") -> Path:
        path = root / "candidate.tar.gz"
        with tarfile.open(path, "w:gz") as handle:
            handle.addfile(member, io.BytesIO(data) if member.isfile() else None)
        return path

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            member = tarfile.TarInfo("../escape")
            member.size = 1
            path = self._write_tar(Path(raw), member, b"x")
            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                inspect_semantic0031_archive(path)

    def test_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            member = tarfile.TarInfo("deck.csv")
            member.type = tarfile.SYMTYPE
            member.linkname = "outside"
            path = self._write_tar(Path(raw), member)
            with self.assertRaisesRegex(ValueError, "links are forbidden"):
                inspect_semantic0031_archive(path)

    def test_rejects_uncommitted_archive_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "candidate.tar.gz"
            with tarfile.open(path, "w:gz") as handle:
                for name in sorted(_REQUIRED_PACKAGE_FILES):
                    member = tarfile.TarInfo(name)
                    member.size = 0
                    handle.addfile(member, io.BytesIO())
            with self.assertRaisesRegex(ValueError, "archive hash mismatch"):
                load_semantic0031_package(path, "cpu")


if __name__ == "__main__":
    unittest.main()
