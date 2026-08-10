from __future__ import annotations

import copy
import importlib
from pathlib import Path
import tempfile
import unittest

import torch
from torch.distributions import Categorical


PROJECT = "train.0042_full_model_design"
ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "archive/pretrained/0031_friend_0809_gsb_v5_value_v9/model.pt"
DECK = ROOT / "train/0042_full_model_design/league/decks/007_dragapult_ex/deck.csv"


def _has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in module.parameters()
    )


class StrategyArchitectureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        policy = importlib.import_module(f"{PROJECT}.policy")
        audit = importlib.import_module(f"{PROJECT}.diagnostics.strategy_adapter_v2_audit")
        flags = importlib.import_module(f"{PROJECT}.integrated.presets").preset("FULL_MODEL")
        cls.features, cls.deck, _ = audit._public_feature_batch(audit.DEFAULT_REPLAY)
        cls.model, cls.identity = policy.load_actor_critic(
            CHECKPOINT, cls.deck, "cpu", integrated_flags=flags
        )
        cls.actions = importlib.import_module(f"{PROJECT}.policy.action_distribution")

    def setUp(self) -> None:
        self.model.zero_grad(set_to_none=True)
        self.model.value_adapter.gate.data.zero_()
        self.model.policy_strategy_adapter.gate.data.zero_()

    def test_frozen_trainable_and_optimizer_contract(self) -> None:
        trainer_cls = importlib.import_module(
            f"{PROJECT}.training.ppo_full_semantic"
        ).PPOTrainer
        trainer = trainer_cls(self.model, device=torch.device("cpu"))
        self.assertEqual(self.identity.actor_parameter_count, 56_352_322)
        self.assertEqual(sum(p.numel() for p in self.model.actor.option_encoder.parameters() if p.requires_grad), 0)
        self.model.assert_no_option_adaptation()
        names = [str(group["name"]) for group in trainer.optimizer.param_groups]
        self.assertEqual(names, [
            "action_decoder", "policy_strategy_adapter", "value_win",
            "value_adapter", "allocation_head", "value_prize",
        ])
        self.assertFalse(any("lora" in name.lower() or "option" in name.lower() for name in names))
        value_ids = {id(p) for p in self.model.value_adapter.parameters()}
        policy_ids = {id(p) for p in self.model.policy_strategy_adapter.parameters()}
        self.assertTrue(value_ids.isdisjoint(policy_ids))
        self.assertFalse(any(p.requires_grad for p in self.model.value_head.heads.archetype.parameters()))

    def test_numbered_deck_copy_matches_authoritative_frozen_pool(self) -> None:
        target = ROOT / "train/0042_full_model_design/league/decks"
        source = ROOT / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1/decks"
        target_dirs = sorted(path.name for path in target.iterdir() if path.is_dir())
        source_dirs = sorted(path.name for path in source.iterdir() if path.is_dir())
        self.assertEqual(target_dirs, source_dirs)
        self.assertEqual(len(target_dirs), 55)
        self.assertEqual([name[:3] for name in target_dirs], [f"{i:03d}" for i in range(1, 56)])
        for name in target_dirs:
            for filename in ("deck.csv", "manifest.json"):
                self.assertEqual((target / name / filename).read_bytes(), (source / name / filename).read_bytes())

    def test_zero_gate_is_exact_0042_base(self) -> None:
        self.model.eval()
        with torch.inference_mode():
            validated, state, options = self.model.actor.encode(self.features)
            memory = torch.cat((state.tokens, options), dim=1)
            mask = torch.cat((state.mask, validated.option_mask), dim=1)
            queries = self.model.value_head.decode(memory, mask)
            legacy_value = 2.0 * self.model.value_head.heads.value(
                queries[:, 0]
            ).squeeze(-1).sigmoid() - 1.0
            legacy_meta = self.model.value_head.heads.archetype(queries[:, 1])
            value, auxiliary = self.model.value_and_aux_from_encoded(validated, state, options)
            context = self.model.strategy_context(validated, value, auxiliary)
            decoder_state = self.model.actor.action_decoder.initialize(validated, state.summary)
            legacy_logits = self.model.actor.action_decoder.logits(validated, options, decoder_state)
            adapted_logits = self.model.head.logits(validated, options, decoder_state, context)
            legacy_action = self.model.actor.action_decoder.greedy(validated, options, state.summary)
            adapted_action = self.actions.greedy_actions(self.model, self.features)[0]
        torch.testing.assert_close(value, legacy_value, rtol=0, atol=0)
        torch.testing.assert_close(auxiliary["meta_logits"], legacy_meta, rtol=0, atol=0)
        torch.testing.assert_close(adapted_logits, legacy_logits, rtol=0, atol=0)
        self.assertEqual(adapted_action.indices, tuple(
            legacy_action.sequences[0, : legacy_action.lengths[0]].tolist()
        ))

    def test_policy_adapter_changes_readout_not_recurrent_transition(self) -> None:
        self.model.policy_strategy_adapter.gate.data.fill_(0.7)
        validated, state, options, value, auxiliary, context = self.model.encode_with_strategy(
            self.features
        )
        decoder = self.model.actor.action_decoder
        decoder_state = decoder.initialize(validated, state.summary)
        hidden_before = decoder_state.hidden.detach().clone()
        legacy = decoder.logits(validated, options, decoder_state)
        adapted = self.model.head.logits(validated, options, decoder_state, context)
        self.assertFalse(torch.equal(legacy, adapted))
        torch.testing.assert_close(decoder_state.hidden, hidden_before, rtol=0, atol=0)
        choice = adapted[:, :-1].argmax(dim=1)
        selected = options[:, choice.item()]
        expected = decoder.recurrent(selected, hidden_before)
        consumed = decoder.consume(options, decoder_state, choice)
        torch.testing.assert_close(consumed.hidden, expected, rtol=0, atol=0)

    def _startup(self, module, call) -> None:
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        optimizer.zero_grad(set_to_none=True)
        call().square().mean().backward()
        self.assertIsNotNone(module.gate.grad)
        self.assertNotEqual(float(module.gate.grad), 0.0)
        self.assertEqual(sum(
            float(p.grad.abs().sum()) for p in module.mlp.parameters() if p.grad is not None
        ), 0.0)
        optimizer.step()
        self.assertNotEqual(float(module.gate.detach()), 0.0)
        optimizer.zero_grad(set_to_none=True)
        call().square().mean().backward()
        self.assertGreater(sum(
            float(p.grad.abs().sum()) for p in module.mlp.parameters() if p.grad is not None
        ), 0.0)

    def test_both_zero_gates_start_in_two_steps(self) -> None:
        policy = importlib.import_module(f"{PROJECT}.policy")
        torch.manual_seed(42)
        value_module = policy.ValueResidualAdapter()
        z = torch.randn(4, 320)
        own = torch.zeros(4, dtype=torch.long)
        self._startup(value_module, lambda: value_module(z, own)[0])

        policy_module = policy.PolicyStrategyAdapter()
        context = policy.StrategyContext.build(
            relative_first_player=torch.tensor([1, 2, 1, 2]),
            z_meta=torch.randn(4, 320),
            meta_logits=torch.randn(4, 15),
            value=torch.randn(4),
            own_archetype_id=own,
        )
        hidden = torch.randn(4, 320)
        self._startup(policy_module, lambda: policy_module(hidden, context)[0])

    def test_measured_gradient_boundaries(self) -> None:
        modules = {
            "prototype": self.model.actor.prototype_encoder,
            "state": self.model.actor.state_encoder,
            "option": self.model.actor.option_encoder,
            "value_trunk": self.model.value_head,
            "value_adapter": self.model.value_adapter,
            "meta_head": self.model.value_head.heads.archetype,
            "decoder": self.model.actor.action_decoder,
            "policy_adapter": self.model.policy_strategy_adapter,
        }

        def snapshot(loss):
            self.model.zero_grad(set_to_none=True)
            loss().backward()
            return {name: _has_nonzero_grad(module) for name, module in modules.items()}

        def policy_loss():
            batch, state, options, value, auxiliary, context = self.model.encode_with_strategy(self.features)
            decoder_state = self.model.actor.action_decoder.initialize(batch, state.summary)
            logits = self.model.head.logits(batch, options, decoder_state, context)
            choice = logits.argmax(dim=1)
            return -Categorical(logits=logits.float()).log_prob(choice).mean()

        def value_loss():
            _, _, _, value, _, _ = self.model.encode_with_strategy(self.features)
            return (value - 0.75).square().mean()

        def meta_loss():
            batch, state, options = self.model.actor.encode(self.features)
            _, auxiliary = self.model.value_and_aux_from_encoded(batch, state, options)
            target = (auxiliary["meta_logits"].argmax(dim=-1) + 1) % 15
            return torch.nn.functional.cross_entropy(auxiliary["meta_logits"], target)

        matrix = {
            "policy": snapshot(policy_loss),
            "value": snapshot(value_loss),
            "meta": snapshot(meta_loss),
        }
        self.assertEqual(matrix["policy"], {
            "prototype": False, "state": False, "option": False,
            "value_trunk": False, "value_adapter": False, "meta_head": False,
            "decoder": True, "policy_adapter": True,
        })
        self.assertEqual(matrix["value"], {
            "prototype": False, "state": False, "option": False,
            "value_trunk": True, "value_adapter": True, "meta_head": False,
            "decoder": False, "policy_adapter": False,
        })
        self.assertEqual(matrix["meta"], {
            "prototype": False, "state": False, "option": False,
            "value_trunk": True, "value_adapter": False, "meta_head": False,
            "decoder": False, "policy_adapter": False,
        })

    def test_greedy_rollout_and_replay_share_logprob(self) -> None:
        sampled, _, _, _, _, _, _ = self.actions.sample_actions_with_encoding(
            self.model, self.features, greedy=True
        )
        action = sampled[0]
        width = max(1, len(action.indices))
        sequences = torch.full((1, width), -1, dtype=torch.long)
        if action.indices:
            sequences[0, : len(action.indices)] = torch.tensor(action.indices)
        batch, state, options, value, auxiliary, context = self.model.encode_with_strategy(
            self.features
        )
        replay = self.actions.evaluate_actions_encoded(
            self.model.head, batch, state.summary, options, sequences,
            torch.tensor([len(action.indices)]), torch.tensor([action.stopped]), value, context,
        )
        torch.testing.assert_close(replay.log_prob, torch.tensor([action.log_prob]), rtol=0, atol=1e-6)


class StrategyCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        policy = importlib.import_module(f"{PROJECT}.policy")
        flags = importlib.import_module(f"{PROJECT}.integrated.presets").preset("FULL_MODEL")
        deck = tuple(map(int, DECK.read_text().splitlines()))
        cls.model, _ = policy.load_actor_critic(CHECKPOINT, deck, "cpu", integrated_flags=flags)
        cls.storage = importlib.import_module(f"{PROJECT}.training.storage_full_semantic")

    def test_strict_roundtrip_and_fail_closed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.storage.save_model_only(self.model, path, update=0, metadata={})
            payload = self.storage.load_adapted_model_only(self.model, path)
            state = payload["state_dict"]
            self.assertTrue(any(name.startswith("value_adapter.") for name in state))
            self.assertTrue(any(name.startswith("policy_strategy_adapter.") for name in state))
            self.assertFalse(any("lora" in name.lower() or ".parametrizations." in name for name in state))

            broken = copy.deepcopy(payload)
            broken["state_dict"].pop(next(
                name for name in broken["state_dict"] if name.startswith("policy_strategy_adapter.")
            ))
            missing = Path(directory) / "missing.pt"
            torch.save(broken, missing)
            with self.assertRaisesRegex(ValueError, "state inventory mismatch"):
                self.storage.load_adapted_model_only(self.model, missing)

            broken = copy.deepcopy(payload)
            broken["metadata"]["own_archetype_taxonomy_sha256"] = "0" * 64
            mismatch = Path(directory) / "taxonomy.pt"
            torch.save(broken, mismatch)
            with self.assertRaisesRegex(ValueError, "taxonomy metadata mismatch"):
                self.storage.load_adapted_model_only(self.model, mismatch)

            broken = copy.deepcopy(payload)
            broken["state_dict"]["actor.option_encoder.fake_lora"] = torch.zeros(1)
            lora = Path(directory) / "lora.pt"
            torch.save(broken, lora)
            with self.assertRaisesRegex(ValueError, "forbidden Option LoRA"):
                self.storage.load_adapted_model_only(self.model, lora)


if __name__ == "__main__":
    unittest.main()
