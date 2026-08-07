from __future__ import annotations

import importlib
import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.runner.batch import _candidate_socket_path
from evaluation.runner.inference_server import (
    _AbilityRepeatGuard,
    _InferenceProfile,
    _InferenceRequest,
    _apply_ability_repeat_guard,
    _canonical_record_batching,
    _collate_raw_records_equivalent,
    _inject_source_id_if_required,
    _normalize_request_deck,
    _PinnedBatchPool,
    PolicyServer,
    _resolve_inference_dtype,
    _stack_batches,
    _forced_action,
    _install_static_loader_cache,
    _install_raw_record_batching,
    _handle_connection,
)
from evaluation.runner.worker import _remote_policy_agent


class CandidateInferenceServerTest(unittest.TestCase):
    def test_resident_tensor_batch_matches_existing_server_collation(self) -> None:
        event_cache = importlib.import_module(
            "train.0035_lifetime_aware_feature_compiler.deployment.event_embedding_cache"
        )
        session_cache = importlib.import_module(
            "train.0035_lifetime_aware_feature_compiler.deployment.session_tensor_cache"
        )
        collate = importlib.import_module(
            "train.0035_lifetime_aware_feature_compiler.features.collate"
        )
        records = [
            self._canonical_record(
                {"card": 2, "resource": 3, "event": 1, "option": 2, "skill": 1, "effect": 3},
                [1],
            ),
            self._canonical_record(
                {"card": 4, "resource": 1, "event": 3, "option": 5, "skill": 4, "effect": 1},
                [2, 0],
            ),
        ]
        for record in records:
            record["actor"]["event_cat"] = [
                [0] * len(row) for row in record["actor"]["event_cat"]
            ]
            record["_runtime_event_source_ids"] = tuple(
                range(len(record["actor"]["event_cat"]))
            )
        requests = [
            _InferenceRequest(
                f"session-{index}",
                {"current": {"yourIndex": index}, "select": {}},
                tuple(range(1, 61)),
            )
            for index in range(2)
        ]
        server = object.__new__(PolicyServer)
        model_module = importlib.import_module(
            "train.0035_lifetime_aware_feature_compiler.model"
        )
        domain_module = importlib.import_module(
            "train.0035_lifetime_aware_feature_compiler.domain"
        )
        model = model_module.SemanticPolicy(
            model_module.ModelConfig(
                d_model=32,
                heads=4,
                state_layers=1,
                event_layers=1,
                option_layers=1,
                max_card_id=2048,
                max_attack_id=2048,
                max_skill_id=512,
                max_effect_id=4096,
            ),
            domain_module.PrototypeIndex.empty(),
        ).eval()
        server._resident_tensor_store = event_cache.EventEmbeddingStore(
            model, max_sessions=4
        )
        server._resident_tensor_key_type = session_cache.SessionTensorKey
        server._session_tensor_keys = {}
        server._profile = _InferenceProfile(enabled=True)
        server._raw_record_collator = collate.collate_canonical_records
        server._model_dtype = torch.float32
        server._device = torch.device("cpu")

        actual = server._collate_resident(requests, records)
        components = actual.pop("_runtime_event_static_components")
        expected = _collate_raw_records_equivalent(
            collate.collate_canonical_records, records
        )
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            with self.subTest(name=name):
                self.assertEqual(actual[name].dtype, expected[name].dtype)
                self.assertEqual(actual[name].shape, expected[name].shape)
                if name != "event_cat":
                    self.assertTrue(torch.equal(actual[name], expected[name]))
        self.assertEqual(len(components), 2)
        self.assertEqual(components[0].shape[:2], expected["event_mask"].shape)
        self.assertEqual(components[1].shape, components[0].shape)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA pinned memory is unavailable")
    def test_pinned_batch_pool_reuses_capacity_and_skips_small_tensors(self) -> None:
        pool = _PinnedBatchPool(torch, slots=1, minimum_pinned_bytes=32)
        small = torch.arange(2, dtype=torch.long)
        large = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        slot, first = pool.copy({"small": small, "large": large})
        first_pointer = first["large"].data_ptr()
        self.assertIs(first["small"], small)
        self.assertTrue(first["large"].is_pinned())
        self.assertTrue(torch.equal(first["large"], large))
        pool.release(slot)

        replacement = torch.full((2, 4), 7.0)
        slot, second = pool.copy({"small": small, "large": replacement})
        self.assertEqual(second["large"].data_ptr(), first_pointer)
        self.assertEqual(tuple(second["large"].shape), (2, 4))
        self.assertTrue(torch.equal(second["large"], replacement))
        pool.release(slot)

    def test_connection_routes_explicit_sessions_and_closes_them_on_command(self) -> None:
        class Server:
            def __init__(self):
                self.calls = []
                self.closed = []

            def call(self, session_id, observation, deck, record=None, **_timing):
                self.calls.append((session_id, observation, deck))
                return [7]

            def close_session(self, session_id):
                self.closed.append(session_id)

            def record_response_send(self, _nanoseconds):
                pass

        class Connection:
            def __init__(self):
                self.requests = iter(
                    (
                        {"session_id": "game-a:candidate", "observation": {"select": {}}, "deck": [1] * 60},
                        {"command": "close_session", "session_id": "game-a:candidate"},
                    )
                )
                self.responses = []

            def recv(self):
                try:
                    return next(self.requests)
                except StopIteration as exc:
                    raise EOFError from exc

            def send(self, payload):
                self.responses.append(payload)

            def close(self):
                pass

        server = Server()
        connection = Connection()

        _handle_connection(server, connection)

        self.assertEqual(server.calls[0][0], "game-a:candidate")
        self.assertEqual(server.closed[0], "game-a:candidate")
        self.assertEqual(len(server.closed), 2)
        self.assertEqual(connection.responses, [{"ok": True, "action": [7]}, {"ok": True}])

    def test_compiler_contract_command_returns_server_contract(self) -> None:
        class Server:
            def compiler_contract(self):
                return {"supported": True, "record_contract": "canonical_raw_record_v1"}

            def close_session(self, _session_id):
                pass

        class Connection:
            def __init__(self):
                self.first = True
                self.responses = []

            def recv(self):
                if self.first:
                    self.first = False
                    return {"command": "compiler_contract"}
                raise EOFError

            def send(self, payload):
                self.responses.append(payload)

            def close(self):
                pass

        connection = Connection()
        _handle_connection(Server(), connection)
        self.assertEqual(
            connection.responses,
            [{"ok": True, "contract": {"supported": True, "record_contract": "canonical_raw_record_v1"}}],
        )

    def test_connection_rejects_empty_explicit_session_id(self) -> None:
        class Server:
            def close_session(self, _session_id):
                pass

        class Connection:
            def __init__(self):
                self.responses = []

            def recv(self):
                if self.responses:
                    raise EOFError
                return {"session_id": "", "observation": {"select": {}}}

            def send(self, payload):
                self.responses.append(payload)

            def close(self):
                pass

        connection = Connection()

        _handle_connection(Server(), connection)

        self.assertEqual(connection.responses[0]["ok"], False)
        self.assertIn("session_id", connection.responses[0]["error"])

    @staticmethod
    def _canonical_record(lengths: dict[str, int], action: list[int]) -> dict:
        fields = importlib.import_module(
            "evaluation.arena.candidates.0034_dragapult_third_large_model_zero_shot"
            ".strategy.contracts.fields"
        )
        widths = fields.WIDTHS

        def rows(count: int, width: int, value: int = 0) -> list[list[int]]:
            return [[value + index] * width for index in range(count)]

        actor = {
            "global_cat": [1] * widths.global_cat,
            "global_num": [0.5] * widths.global_num,
            "global_state": [2] * widths.global_state,
            "min_count": 1,
            "max_count": 2,
        }
        for prefix in ("card", "resource", "event", "option"):
            count = lengths[prefix]
            actor[f"{prefix}_cat"] = rows(count, getattr(widths, f"{prefix}_cat"), 1)
            actor[f"{prefix}_num"] = rows(count, getattr(widths, f"{prefix}_num"), 2)
            actor[f"{prefix}_state"] = rows(count, getattr(widths, f"{prefix}_state"), 3)
        actor["card_parent"] = list(range(lengths["card"]))
        for name in ("source", "target", "before", "after"):
            actor[f"event_{name}"] = list(range(lengths["event"]))
        for name in ("source", "target", "context", "effect_card"):
            actor[f"option_{name}"] = list(range(lengths["option"]))
        for prefix, family in (("option_skill", "skill"), ("option_effect", "effect")):
            actor[f"{prefix}_id"] = list(range(1, lengths[family] + 1))
            actor[f"{prefix}_role"] = list(range(lengths[family]))
            actor[f"{prefix}_parent"] = [index % lengths["option"] for index in range(lengths[family])]
        return {"actor": actor, "target": {"ordered_action": action}}

    def test_canonical_raw_records_are_collated_once_per_server_batch(self) -> None:
        calls = []

        def collate(records, **kwargs):
            calls.append((list(records), kwargs))
            return {"values": torch.tensor(records)}

        module_name = "tests._fake_canonical_online_runtime"
        module = type(sys)(module_name)
        module.collate_canonical_records = collate
        encoder_type = type("Encoder", (), {"__module__": module_name})
        canonical = type("Canonical", (), {"requires_source_id": False})()
        with patch.dict(sys.modules, {module_name: module}):
            batch_collator = _install_raw_record_batching(encoder_type, canonical)
            first = module.collate_canonical_records([1])
            second = module.collate_canonical_records([2])
            combined = batch_collator([first, second])

        self.assertEqual(first, 1)
        self.assertEqual(second, 2)
        self.assertEqual(combined["values"].tolist(), [1, 2])
        self.assertEqual(len(calls), 1)

    def test_explicit_record_encoder_avoids_collator_monkeypatch(self) -> None:
        def collate(records):
            return {"records": list(records)}

        module_name = "tests._fake_explicit_record_online_runtime"
        module = type(sys)(module_name)
        module.collate_canonical_records = collate
        encoder_type = type(
            "Encoder",
            (),
            {"__module__": module_name, "encode_record": lambda self, value: value},
        )
        canonical = type("Canonical", (), {"requires_source_id": False})()
        with patch.dict(sys.modules, {module_name: module}):
            batch_collator, direct = _canonical_record_batching(
                encoder_type, canonical
            )

        self.assertTrue(direct)
        self.assertIs(batch_collator, collate)
        self.assertIs(module.collate_canonical_records, collate)

    def test_canonical_batch_collation_is_exactly_equivalent_to_old_stack(self) -> None:
        collate_module = importlib.import_module(
            "evaluation.arena.candidates.0034_dragapult_third_large_model_zero_shot"
            ".strategy.features.collate"
        )
        records = [
            self._canonical_record(
                {"card": 2, "resource": 3, "event": 1, "option": 2, "skill": 1, "effect": 3},
                [1],
            ),
            self._canonical_record(
                {"card": 4, "resource": 1, "event": 3, "option": 5, "skill": 4, "effect": 1},
                [2, 0],
            ),
        ]
        old = _stack_batches(
            torch,
            [collate_module.collate_canonical_records([record]) for record in records],
            torch.device("cpu"),
        )
        new = _collate_raw_records_equivalent(
            collate_module.collate_canonical_records, records
        )

        self.assertEqual(set(new), set(old))
        for name in sorted(old):
            with self.subTest(name=name):
                self.assertEqual(new[name].dtype, old[name].dtype)
                self.assertEqual(tuple(new[name].shape), tuple(old[name].shape))
                self.assertTrue(torch.equal(new[name], old[name]))

    def test_legacy_source_conditioned_policy_keeps_single_row_collation(self) -> None:
        module_name = "tests._fake_legacy_online_runtime"
        module = type(sys)(module_name)
        module.collate_canonical_records = lambda records: records
        encoder_type = type("Encoder", (), {"__module__": module_name})
        legacy = type("Legacy", (), {"requires_source_id": True})()
        with patch.dict(sys.modules, {module_name: module}):
            self.assertIsNone(_install_raw_record_batching(encoder_type, legacy))

    def test_static_prototype_loader_is_cached_per_argument_tuple(self) -> None:
        class PrototypeIndex:
            calls = 0

            @classmethod
            def load(cls, path):
                cls.calls += 1
                return object()

        module_name = "tests._fake_online_runtime"
        module = type(sys)(module_name)
        module.PrototypeIndex = PrototypeIndex
        encoder_type = type("Encoder", (), {"__module__": module_name})
        with patch.dict(sys.modules, {module_name: module}):
            _install_static_loader_cache(encoder_type)
            first = PrototypeIndex.load(Path("one.json"))
            second = PrototypeIndex.load(Path("one.json"))
            third = PrototypeIndex.load(Path("two.json"))

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(PrototypeIndex.calls, 2)

    def test_profile_snapshot_aggregates_batches_and_request_latency(self) -> None:
        profile = _InferenceProfile(enabled=True)
        profile.increment("encoder_initializations")
        profile.add_seconds("feature_encode_seconds", 0.25)
        profile.record_batch(2)
        profile.record_batch(4)
        profile.record_latency_ms("queue_wait", 1.0)
        profile.record_latency_ms("queue_wait", 3.0)
        profile.record_batch_ready(1_000_000)
        profile.record_batch_ready(4_000_000)
        request = _InferenceRequest("session", {"select": {}}, tuple(range(1, 61)))
        profile.finish_request(request)

        snapshot = profile.snapshot()

        self.assertEqual(snapshot["counters"]["encoder_initializations"], 1)
        self.assertEqual(snapshot["counters"]["completed_requests"], 1)
        self.assertEqual(snapshot["seconds"]["feature_encode_seconds"], 0.25)
        self.assertEqual(snapshot["batches"], 2)
        self.assertEqual(snapshot["mean_batch_size"], 3.0)
        self.assertGreaterEqual(snapshot["request_latency_ms"]["p50"], 0.0)
        self.assertEqual(snapshot["latency_ms"]["queue_wait"]["count"], 2)
        self.assertEqual(snapshot["latency_ms"]["queue_wait"]["mean"], 2.0)
        self.assertEqual(snapshot["latency_ms"]["queue_wait"]["p95"], 1.0)
        self.assertEqual(snapshot["latency_ms"]["batch_ready_interval"]["count"], 1)
        self.assertEqual(snapshot["latency_ms"]["batch_ready_interval"]["mean"], 3.0)

    def test_forced_action_requires_exactly_one_required_option(self) -> None:
        required = {
            "select": {"option": [{"type": 1}], "minCount": 1, "maxCount": 1}
        }
        self.assertEqual(_forced_action(required), [0])

        optional = {
            "select": {"option": [{"type": 1}], "minCount": 0, "maxCount": 1}
        }
        multiple = {
            "select": {
                "option": [{"type": 1}, {"type": 2}],
                "minCount": 1,
                "maxCount": 1,
            }
        }
        zero = {"select": {"option": [], "minCount": 0, "maxCount": 0}}
        self.assertIsNone(_forced_action(optional))
        self.assertIsNone(_forced_action(multiple))
        self.assertIsNone(_forced_action(zero))

    def test_repeat_guard_ends_turn_after_eight_identical_ability_entries(self) -> None:
        observation = {
            "select": {
                "type": 0,
                "option": [
                    {"type": 10, "area": 5, "index": 1},
                    {"type": 13, "attackId": 120},
                    {"type": 14},
                ],
            },
            "current": {"turn": 10, "yourIndex": 1},
        }
        guard = _AbilityRepeatGuard(limit=8)
        for _ in range(8):
            self.assertEqual(_apply_ability_repeat_guard(observation, [0], guard), [0])
        self.assertEqual(_apply_ability_repeat_guard(observation, [0], guard), [2])

    def test_repeat_guard_does_not_change_attacks_or_disabled_contract(self) -> None:
        observation = {
            "select": {"type": 0, "option": [{"type": 13}, {"type": 14}]},
            "current": {"turn": 4, "yourIndex": 0},
        }
        self.assertEqual(
            _apply_ability_repeat_guard(observation, [0], _AbilityRepeatGuard(limit=8)),
            [0],
        )
        ability = {
            "select": {"type": 0, "option": [{"type": 10}, {"type": 14}]},
            "current": {"turn": 4, "yourIndex": 0},
        }
        guard = _AbilityRepeatGuard(limit=0)
        for _ in range(20):
            self.assertEqual(_apply_ability_repeat_guard(ability, [0], guard), [0])

    def test_source_identity_is_only_injected_for_declared_policies(self) -> None:
        canonical = type("Canonical", (), {"requires_source_id": False})()
        legacy = type("Legacy", (), {})()
        canonical_row = {"option_mask": torch.tensor([[True]])}
        legacy_row = {"option_mask": torch.tensor([[True]])}
        _inject_source_id_if_required(torch, canonical_row, canonical)
        _inject_source_id_if_required(torch, legacy_row, legacy)
        self.assertNotIn("source_id", canonical_row)
        self.assertEqual(legacy_row["source_id"].tolist(), [0])

    def test_canonical_policy_declares_fail_closed_inference(self) -> None:
        module = __import__(
            "train.0025_semantic_foundation_pretraining.deployment.canonical_inference",
            fromlist=["PortableCanonicalPolicy"],
        )
        self.assertTrue(module.PortableCanonicalPolicy.fail_closed_inference_errors)

    def test_remote_policy_agent_routes_exact_deck_and_role(self) -> None:
        class Connection:
            def __init__(self) -> None:
                self.requests = []

            def send(self, payload) -> None:
                self.requests.append(payload)

            @staticmethod
            def recv():
                return {"ok": True, "action": [7]}

        connection = Connection()
        deck = list(range(1, 61))
        with patch("evaluation.runner.worker.Client", return_value=connection):
            agent = _remote_policy_agent("/tmp/frozen.sock", deck=deck, role="opponent")
            self.assertEqual(agent({"select": None}), [7])

        self.assertEqual(
            connection.requests,
            [{"observation": {"select": None}, "deck": tuple(deck)}],
        )

    def test_remote_policy_agent_error_names_the_routed_role(self) -> None:
        class Connection:
            @staticmethod
            def send(_payload) -> None:
                return None

            @staticmethod
            def recv():
                return {"ok": False, "error": "boom"}

        with patch("evaluation.runner.worker.Client", return_value=Connection()):
            agent = _remote_policy_agent(
                "/tmp/frozen.sock", deck=list(range(1, 61)), role="opponent"
            )
            with self.assertRaisesRegex(RuntimeError, "shared opponent inference"):
                agent({"select": None})

    def test_socket_path_fits_linux_af_unix_limit(self) -> None:
        path = _candidate_socket_path()
        self.assertEqual(path.parent, Path(tempfile.gettempdir()))
        self.assertLessEqual(len(str(path).encode()), 107)
        self.assertFalse(path.exists())
        self.assertNotEqual(path, _candidate_socket_path())

    def test_stack_batches_pads_variable_tensor_axes(self) -> None:
        rows = [
            {
                "option_mask": torch.tensor([[True, True]]),
                "option_cat": torch.tensor([[[1, 2], [3, 4]]]),
                "targets": torch.tensor([[0, 2]]),
            },
            {
                "option_mask": torch.tensor([[True]]),
                "option_cat": torch.tensor([[[5, 6]]]),
                "targets": torch.tensor([[0]]),
            },
        ]
        batch = _stack_batches(torch, rows, torch.device("cpu"))
        self.assertEqual(tuple(batch["option_cat"].shape), (2, 2, 2))
        self.assertEqual(batch["option_mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch["targets"].tolist(), [[0, 2], [0, -100]])

    def test_request_deck_is_exact_60_positive_integer_cards(self) -> None:
        expected = tuple(range(1, 61))
        self.assertEqual(_normalize_request_deck(list(expected), (999,) * 60), expected)
        self.assertEqual(_normalize_request_deck(None, expected), expected)
        for invalid in ([1] * 59, [1] * 61, [0] * 60, [True] * 60, "1,2"):
            with self.subTest(invalid=type(invalid).__name__, length=len(invalid)):
                with self.assertRaisesRegex(ValueError, "60 positive integer"):
                    _normalize_request_deck(invalid, expected)

    def test_inference_dtype_requires_supported_cuda_precision(self) -> None:
        self.assertIs(
            _resolve_inference_dtype(torch, "fp32", torch.device("cpu")),
            torch.float32,
        )
        self.assertIs(
            _resolve_inference_dtype(torch, "fp16", torch.device("cuda:0")),
            torch.float16,
        )
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            _resolve_inference_dtype(torch, "fp16", torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "unsupported inference dtype"):
            _resolve_inference_dtype(torch, "tf32", torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
