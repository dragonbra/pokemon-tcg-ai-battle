from __future__ import annotations

import importlib
import inspect
import unittest
from collections import Counter
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch.distributions import Categorical

from .. import FOCAL_DECK_ID
from ..checkpoint import FORBIDDEN_FIELDS, _validate, load_model_checkpoint, save_model_checkpoint
from ..foundation import verify_foundation
from ..league import load_frozen_catalog
from ..policy import load_actor_critic
from ..policy.action_distribution import cached_features, evaluate_actions_encoded
from ..policy.action_distribution import SampledAction
from ..foundation.contracts.fields import WIDTHS
from ..training.run import RunConfig, build_jobs


class ProjectContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _, cls.identity = load_actor_critic("cpu")

    def test_foundation_and_trainable_contract(self):
        self.assertEqual(verify_foundation().checkpoint_sha256, "5e0a6eea42bf228a9bd977cf56fdd14ad360fbceffcf5c19e2d9fe1b39713d98")
        self.model.assert_trainable_contract()
        self.assertTrue(all(name.startswith(("actor.action_decoder.", "value_head.")) for name in self.model.trainable_parameter_names()))
        self.assertFalse(any(p.requires_grad for p in self.model.opponent_head.parameters()))

    def test_legacy_foundation_identity(self):
        from ..legacy_foundation import verify_legacy_foundation

        identity = verify_legacy_foundation()
        self.assertEqual(identity.asset_id, "0019-0730-epoch13")
        self.assertEqual(identity.deployment_source_id, 0)
        self.assertEqual(
            identity.provenance_weights_sha256,
            "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb",
        )
        self.assertEqual(
            identity.exported_model_sha256,
            "2a3e3224b9fbc0bda95de45583faa1f3923d37bc55e26c6279315cc720d7f918",
        )
        self.assertEqual(
            identity.ontology_sha256,
            "8144c63e512a2a00fabaaf2b19cd002c5b59c25fb0763a112256848460113a4d",
        )
        with self.assertRaises(FileNotFoundError):
            verify_legacy_foundation(Path("/definitely/missing/legacy-policy"))

    def test_legacy_foundation_service_is_frozen(self):
        from ..legacy_foundation import LegacyFoundationService

        service = LegacyFoundationService(torch.device("cpu"))
        self.assertTrue(all(not value.requires_grad for value in service.actor.parameters()))
        self.assertEqual(service.actor.config.d_model, 320)

    def test_legacy_feature_collation_pads_variable_width(self):
        from ..legacy_foundation import collate_legacy_batches

        first = {
            "global_cat": torch.zeros(1, 4, dtype=torch.long),
            "entity_cat": torch.ones(1, 2, 7, dtype=torch.long),
            "entity_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        second = {
            "global_cat": torch.ones(1, 4, dtype=torch.long),
            "entity_cat": torch.full((1, 4, 7), 2, dtype=torch.long),
            "entity_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        batch = collate_legacy_batches([first, second])
        self.assertEqual(tuple(batch["entity_cat"].shape), (2, 4, 7))
        self.assertFalse(batch["entity_mask"][0, 2:].any())
        self.assertTrue(torch.equal(batch["global_cat"][:, 0], torch.tensor([0, 1])))

    def test_legacy_collector_requires_verified_service(self):
        from ..rollout.collector import HeterogeneousRolloutCollector

        with self.assertRaises(ValueError):
            HeterogeneousRolloutCollector(
                self.model,
                device=torch.device("cpu"),
                workers=1,
                mode="greedy",
                opponent_foundation="0019",
            )

    def test_collector_records_focal_action_after_routing_split(self):
        from ..rollout import collector as collector_module
        from ..rollout.collector import HeterogeneousRolloutCollector

        collector = HeterogeneousRolloutCollector(
            self.model, device=torch.device("cpu"), workers=1, mode="greedy"
        )
        encoder = mock.Mock()
        encoder.encode.return_value = {"stub": torch.zeros(1)}
        item = SimpleNamespace(
            focal_encoder=encoder,
            opponent_encoder=encoder,
            decisions=[],
            job=SimpleNamespace(source_policy_update=7),
        )
        validated = SimpleNamespace(
            option_mask=torch.ones(1, 2, dtype=torch.bool),
            min_count=torch.ones(1, dtype=torch.long),
            max_count=torch.ones(1, dtype=torch.long),
        )
        action = SampledAction((1,), True, -0.5, 0.25, 0.75)
        with (
            mock.patch.object(
                collector_module,
                "collate_feature_batches",
                return_value={"stub": torch.zeros(1)},
            ),
            mock.patch.object(
                collector_module,
                "move_batch",
                side_effect=lambda batch, device: batch,
            ),
            mock.patch.object(
                self.model,
                "encode",
                return_value=(
                    validated,
                    torch.zeros(1, 320),
                    torch.zeros(1, 2, 320),
                ),
            ),
            mock.patch.object(
                collector_module,
                "infer_actions",
                return_value=[action],
            ),
        ):
            routed = collector._infer([(item, {}, True)])
        self.assertEqual(routed, [(1,)])
        self.assertEqual(len(item.decisions), 1)
        self.assertEqual(item.decisions[0].policy_update, 7)

    def test_dual_evaluation_reuses_schedule_and_separates_namespaces(self):
        run_module = importlib.import_module(
            "train.0030_dragapult_shared_encoder_decoder_rl.training.run"
        )

        observed = []

        def fake_build_jobs(*, opponent_foundations, **kwargs):
            foundation = opponent_foundations[0]
            return [
                SimpleNamespace(
                    opponent_id=f"deck-{index}",
                    opponent_foundation=foundation,
                    focal_first=index == 0,
                    seed=100 + index,
                )
                for index in range(2)
            ]

        class FakeCollector:
            def __init__(self, *args, opponent_foundation="0028", **kwargs):
                self.foundation = opponent_foundation

            def collect(self, received):
                observed.append((self.foundation, id(received), tuple(
                    (job.opponent_id, job.focal_first, job.seed) for job in received
                )))
                return [
                    SimpleNamespace(
                        valid=True,
                        reward=1.0 if index == 0 else -1.0,
                        job=job,
                        turns=4,
                        decisions=[object()],
                    )
                    for index, job in enumerate(received)
                ]

            def metrics(self):
                return {"rollout/inference_requests": 2.0}

        config = RunConfig(version="V999_dual_eval", workers=1)
        with (
            mock.patch.object(run_module, "build_jobs", side_effect=fake_build_jobs),
            mock.patch.object(
                run_module, "HeterogeneousRolloutCollector", FakeCollector
            ),
        ):
            metrics, selection_score = run_module._evaluate_dual(
                self.model, config, 5, object()
            )
        self.assertNotEqual(observed[0][1], observed[1][1])
        self.assertEqual(observed[0][2], observed[1][2])
        self.assertEqual(observed[0][0], "0019")
        self.assertEqual(observed[1][0], "0028")
        self.assertEqual(metrics["eval/checkpoint_update"], 5.0)
        self.assertEqual(metrics["eval/foundation_0019/win_rate"], 0.5)
        self.assertEqual(metrics["eval/foundation_0028/win_rate"], 0.5)
        self.assertEqual(selection_score, 0.5)
        self.assertNotIn("eval/win_rate", metrics)
        self.assertNotIn("rollout/inference_requests", metrics)

    def test_frozen_catalog_and_schedule(self):
        catalog = load_frozen_catalog()
        self.assertEqual(len(catalog), 51)
        self.assertEqual(len(next(row for row in catalog if row.deck_id == FOCAL_DECK_ID).deck), 60)
        jobs = build_jobs(
            count=102,
            source_policy_update=0,
            seed=30,
            greedy_eval=True,
            opponent_foundations=("0028",),
        )
        self.assertEqual(len({job.opponent_id for job in jobs}), 51)
        self.assertEqual(sum(job.focal_first for job in jobs), 51)

    def test_mixed_training_schedule_is_equal_by_foundation_and_seat(self):
        jobs = build_jobs(count=512, source_policy_update=4, seed=20260804)
        counts = Counter(job.opponent_foundation for job in jobs)
        self.assertEqual(counts, {"0019": 256, "0028": 256})
        seats = Counter(
            (job.opponent_foundation, job.focal_first) for job in jobs
        )
        self.assertEqual(seats[("0019", True)], 128)
        self.assertEqual(seats[("0019", False)], 128)
        self.assertEqual(seats[("0028", True)], 128)
        self.assertEqual(seats[("0028", False)], 128)
        self.assertEqual(
            {job.opponent_id for job in jobs if job.opponent_foundation == "0019"},
            {job.opponent_id for job in jobs if job.opponent_foundation == "0028"},
        )

    def test_frozen_evaluation_seeds_are_fixed_across_checkpoints(self):
        first = build_jobs(
            count=102,
            source_policy_update=0,
            seed=20260804,
            greedy_eval=True,
            opponent_foundations=("0019",),
        )
        later = build_jobs(
            count=102,
            source_policy_update=75,
            seed=20260804,
            greedy_eval=True,
            opponent_foundations=("0019",),
        )
        self.assertEqual([job.seed for job in first], [job.seed for job in later])
        self.assertEqual(
            [(job.opponent_id, job.focal_first) for job in first],
            [(job.opponent_id, job.focal_first) for job in later],
        )

    def test_model_only_checkpoint_rejects_training_state(self):
        for forbidden in FORBIDDEN_FIELDS:
            with self.assertRaises(ValueError):
                _validate({forbidden: torch.tensor(0)})

    def test_checkpoint_contract_preserves_every_update(self):
        run_module = importlib.import_module(
            "train.0030_dragapult_shared_encoder_decoder_rl.training.run"
        )
        config = RunConfig(version="V999_all_checkpoints", workers=1)
        self.assertEqual(config.checkpoint_retention, "all")
        config.validate()
        self.assertNotIn(".unlink(", inspect.getsource(run_module))

    def test_exporter_enforces_kaggle_raw_exec_contract(self):
        exporter = Path(__file__).resolve().parents[1] / "export_candidate.py"
        source = exporter.read_text(encoding="utf-8")
        self.assertIn('globals().get("__file__", Path.cwd())', source)
        self.assertIn('Path("/kaggle_simulations/agent/deck.csv")', source)
        self.assertIn("validate_kaggle_raw_exec(", source)
        self.assertLess(
            source.index("validate_kaggle_raw_exec("),
            source.index("staging.replace(output)"),
        )

    def test_run_config_rejects_missing_initialization_checkpoint(self):
        config = RunConfig(
            version="V999_missing_checkpoint",
            initialization_checkpoint="/definitely/missing/update-0005.pt",
        )
        with self.assertRaises(FileNotFoundError):
            config.validate()

    def test_model_only_checkpoint_initializes_fresh_model(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "update-0005.pt"
            first_parameter = next(self.model.actor.action_decoder.parameters())
            original = first_parameter.detach().clone()
            with torch.no_grad():
                first_parameter.add_(0.001)
            try:
                expected_decoder = self.model.decoder_sha256()
                expected_representation = self.model.representation_sha256()
                expected_file = save_model_checkpoint(
                    path, self.model, policy_version="V2_test", update=5
                )
            finally:
                with torch.no_grad():
                    first_parameter.copy_(original)
            fresh, _, _ = load_actor_critic("cpu")
            identity = load_model_checkpoint(path, fresh)
            self.assertEqual(identity["checkpoint_sha256"], expected_file)
            self.assertEqual(identity["update"], 5)
            self.assertEqual(fresh.decoder_sha256(), expected_decoder)
            self.assertEqual(fresh.representation_sha256(), expected_representation)

    def test_cached_decoder_matches_full_representation_path(self):
        b, c, r, e, o, s, f = 2, 3, 4, 2, 5, 3, 4
        batch = {
            "global_cat": torch.zeros(b, WIDTHS.global_cat, dtype=torch.long),
            "global_num": torch.zeros(b, WIDTHS.global_num),
            "card_cat": torch.zeros(b, c, WIDTHS.card_cat, dtype=torch.long),
            "card_num": torch.zeros(b, c, WIDTHS.card_num),
            "card_parent": torch.zeros(b, c, dtype=torch.long),
            "card_mask": torch.ones(b, c, dtype=torch.bool),
            "resource_cat": torch.zeros(b, r, WIDTHS.resource_cat, dtype=torch.long),
            "resource_num": torch.zeros(b, r, WIDTHS.resource_num),
            "resource_mask": torch.ones(b, r, dtype=torch.bool),
            "event_cat": torch.zeros(b, e, WIDTHS.event_cat, dtype=torch.long),
            "event_num": torch.zeros(b, e, WIDTHS.event_num),
            "event_mask": torch.ones(b, e, dtype=torch.bool),
            "option_cat": torch.zeros(b, o, WIDTHS.option_cat, dtype=torch.long),
            "option_num": torch.zeros(b, o, WIDTHS.option_num),
            "option_state": torch.zeros(b, o, WIDTHS.option_state, dtype=torch.long),
            "option_source": torch.zeros(b, o, dtype=torch.long),
            "option_target": torch.zeros(b, o, dtype=torch.long),
            "option_mask": torch.ones(b, o, dtype=torch.bool),
            "option_skill_id": torch.zeros(b, s, dtype=torch.long),
            "option_skill_role": torch.zeros(b, s, dtype=torch.long),
            "option_skill_parent": torch.zeros(b, s, dtype=torch.long),
            "option_skill_mask": torch.ones(b, s, dtype=torch.bool),
            "option_effect_id": torch.zeros(b, f, dtype=torch.long),
            "option_effect_role": torch.zeros(b, f, dtype=torch.long),
            "option_effect_parent": torch.zeros(b, f, dtype=torch.long),
            "option_effect_mask": torch.ones(b, f, dtype=torch.bool),
            "min_count": torch.ones(b, dtype=torch.long),
            "max_count": torch.full((b,), 2, dtype=torch.long),
            "targets": torch.zeros(b, 2, dtype=torch.long),
        }
        self.model.eval()
        with torch.inference_mode():
            validated, state, options = self.model.actor.encode(batch)
            self.model.prepare_inference_cache()
            cached_batch, cached_summary, cached_options = self.model.encode(batch)
            decoder_state = self.model.actor.action_decoder.initialize(validated, state.summary)
            distribution = Categorical(logits=self.model.actor.action_decoder.logits(validated, options, decoder_state).float())
            direct_log_prob = distribution.log_prob(torch.zeros(b, dtype=torch.long))
            direct_entropy = distribution.entropy()
            direct_value = self.model.value_head(state.summary).squeeze(-1)
            result = evaluate_actions_encoded(
                self.model.head,
                cached_features(validated, state.summary, options),
                torch.zeros(b, 1, dtype=torch.long),
                torch.ones(b, dtype=torch.long),
                torch.zeros(b, dtype=torch.bool),
            )
        torch.testing.assert_close(result.log_prob, direct_log_prob)
        torch.testing.assert_close(result.entropy, direct_entropy)
        torch.testing.assert_close(result.value, direct_value)
        torch.testing.assert_close(cached_summary, state.summary)
        torch.testing.assert_close(cached_options, options)
        self.assertTrue(torch.equal(cached_batch.option_mask, validated.option_mask))

    def test_no_cross_numbered_runtime_imports(self):
        root = Path(__file__).resolve().parents[1]
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in root.rglob("*.py")
            if "tests" not in path.parts
        )
        self.assertNotIn("from train.0028_", sources)
        self.assertNotIn("from train.0026_", sources)
        self.assertNotIn("import train.0028_", sources)
        self.assertNotIn("import train.0026_", sources)


if __name__ == "__main__":
    unittest.main()
