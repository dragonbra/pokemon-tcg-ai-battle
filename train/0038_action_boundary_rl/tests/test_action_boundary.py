from __future__ import annotations

import dataclasses
import importlib
import math
import unittest

import torch

gate_module = importlib.import_module("train.0038_action_boundary_rl.action_boundary.decision_gate")
dragapult_module = importlib.import_module("train.0038_action_boundary_rl.action_boundary.dragapult")
macro_module = importlib.import_module("train.0038_action_boundary_rl.action_boundary.macro_protocol")
head_module = importlib.import_module("train.0038_action_boundary_rl.policy.allocation_head")
protocol_module = importlib.import_module("train.0038_action_boundary_rl.rollout.protocol")
gae_module = importlib.import_module("train.0038_action_boundary_rl.training.compound_gae")
ppo_module = importlib.import_module("train.0038_action_boundary_rl.training.compound_ppo")
checkpoint_module = importlib.import_module("train.0038_action_boundary_rl.checkpoint")

DecisionClass, DecisionGate = gate_module.DecisionClass, gate_module.DecisionGate
DragapultDamageAllocation = dragapult_module.DragapultDamageAllocation
StableTargetIdentity, enumerate_allocations = dragapult_module.StableTargetIdentity, dragapult_module.enumerate_allocations
MacroProtocolError, PendingMacroTransaction = macro_module.MacroProtocolError, macro_module.PendingMacroTransaction
DragapultAllocationHead = head_module.DragapultAllocationHead
CanonicalMacroAction, PolicyTransition = protocol_module.CanonicalMacroAction, protocol_module.PolicyTransition
compound_gae, strategic_metric = gae_module.compound_gae, gae_module.strategic_metric
compound_policy_loss = ppo_module.compound_policy_loss


def observation(options, minimum=1, maximum=1, *, result=-1):
    return {
        "current": {"result": result, "yourIndex": 0, "players": [{}, {}]},
        "select": {"option": options, "minCount": minimum, "maxCount": maximum},
    }


class DecisionGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = DecisionGate()

    def test_terminal_and_empty_contracts(self):
        self.assertEqual(self.gate.classify(observation([], 0, 0, result=0)).classification, DecisionClass.TERMINAL)
        self.assertEqual(self.gate.classify(observation([], 0, 0)).classification, DecisionClass.LEGAL_EMPTY_PASS)
        self.assertEqual(self.gate.classify(observation([], 1, 1)).classification, DecisionClass.MASK_ERROR)

    def test_only_single_mandatory_completion_is_forced(self):
        forced = self.gate.classify(observation([{"type": "Card"}]))
        self.assertEqual(forced.classification, DecisionClass.FORCED)
        self.assertEqual(forced.forced_action, (0,))
        for option in ({"type": "No"}, {"type": 2}, {"type": "Cancel"},
                       {"type": "Pass"}, {"type": 14}):
            self.assertEqual(
                self.gate.classify(observation([option])).classification,
                DecisionClass.STRATEGIC,
            )
        self.assertEqual(
            self.gate.classify(observation([{"type": "Card"}], 0, 1)).classification,
            DecisionClass.STRATEGIC,
        )

    def test_multiple_options_and_canonical_aliases(self):
        result = self.gate.classify(observation([{}, {}], 1, 2), canonical_completion_count=1)
        self.assertEqual(result.legal_completion_count, 4)
        self.assertEqual(result.classification, DecisionClass.STRATEGIC)  # STOP remains legal


class DragapultTests(unittest.TestCase):
    @staticmethod
    def targets(n):
        return tuple(StableTargetIdentity(1, 100 + i, 500 + i, i) for i in range(n))

    def test_exhaustive_counts_and_uniqueness(self):
        for n, expected in enumerate(
            (1, 7, 28, 84, 210, 462, 924, 1716), start=1,
        ):
            allocations = enumerate_allocations(self.targets(n))
            self.assertEqual(len(allocations), expected)
            self.assertEqual(len({item.counters for item in allocations}), expected)
            self.assertTrue(all(sum(item.counters) == 6 for item in allocations))
            self.assertEqual(expected, math.comb(n + 5, 6))

    def test_macro_relocates_by_serial_and_returns_original_shape(self):
        targets = self.targets(2)
        allocation = DragapultDamageAllocation(targets, (2, 4))
        transaction = PendingMacroTransaction("battle-a", 0, allocation)
        bench = [
            {"serial": targets[1].serial, "id": targets[1].card_id},
            {"serial": targets[0].serial, "id": targets[0].card_id},
        ]
        for remain, expected_index in ((6, 1), (5, 1), (4, 0), (3, 0), (2, 0), (1, 0)):
            obs = {
                "current": {"yourIndex": 0, "players": [{}, {"bench": bench}]},
                "select": {
                    "context": 14, "remainDamageCounter": remain,
                    "minCount": 1, "maxCount": 1,
                    "option": [
                        {"playerIndex": 1, "index": 0},
                        {"playerIndex": 1, "index": 1},
                    ],
                },
            }
            self.assertEqual(transaction.next_primitive(obs, battle_id="battle-a"), [expected_index])
        self.assertTrue(transaction.complete)

    def test_macro_fails_closed_on_identity_drift(self):
        target = self.targets(1)[0]
        transaction = PendingMacroTransaction("battle-a", 0, DragapultDamageAllocation((target,), (6,)))
        obs = {
            "current": {"yourIndex": 0, "players": [{}, {"bench": [{"serial": 999, "id": target.card_id}]}]},
            "select": {"context": 14, "remainDamageCounter": 6, "minCount": 1, "maxCount": 1,
                       "option": [{"playerIndex": 1, "index": 0}]},
        }
        with self.assertRaises(MacroProtocolError):
            transaction.next_primitive(obs, battle_id="battle-a")
        self.assertIsNotNone(transaction.invalid_reason)

    def test_macro_allows_bench_reordering_but_not_target_universe_drift(self):
        targets = self.targets(2)
        transaction = PendingMacroTransaction(
            "battle-a", 0, DragapultDamageAllocation(targets, (1, 5))
        )

        def callback(remain, bench):
            return {
                "current": {"yourIndex": 0, "players": [{}, {"bench": bench}]},
                "select": {
                    "context": 14, "remainDamageCounter": remain,
                    "minCount": 1, "maxCount": 1,
                    "option": [
                        {"playerIndex": 1, "index": 0},
                        {"playerIndex": 1, "index": 1},
                    ],
                },
            }

        original = [
            {"serial": targets[0].serial, "id": targets[0].card_id},
            {"serial": targets[1].serial, "id": targets[1].card_id},
        ]
        reordered = list(reversed(original))
        self.assertEqual(
            transaction.next_primitive(callback(6, original), battle_id="battle-a"),
            [0],
        )
        self.assertEqual(
            transaction.next_primitive(callback(5, reordered), battle_id="battle-a"),
            [0],
        )

        drifted = [reordered[0], {"serial": 999, "id": reordered[1]["id"]}]
        with self.assertRaises(MacroProtocolError):
            transaction.next_primitive(callback(4, drifted), battle_id="battle-a")


class AllocationHeadTests(unittest.TestCase):
    def test_permutation_invariant_and_root_independent(self):
        torch.manual_seed(7)
        head = DragapultAllocationHead(8)
        state, root = torch.randn(1, 8), torch.randn(1, 8)
        targets = torch.randn(1, 3, 4, 8)
        visible = torch.randn(1, 3, 4, 12)
        visible[..., 0] = torch.tensor([[[1, 2, 3, 0], [6, 0, 0, 0], [0, 1, 0, 5]]])
        mask = torch.ones(1, 3, 4, dtype=torch.bool)
        valid = torch.ones(1, 3, dtype=torch.bool)
        first = head(state, root, targets, visible, mask, valid)
        permutation = torch.tensor([2, 0, 3, 1])
        second = head(state, root, targets[:, :, permutation], visible[:, :, permutation], mask[:, :, permutation], valid)
        torch.testing.assert_close(first, second)


class PolicyTrajectoryTests(unittest.TestCase):
    def test_joint_logprob_and_terminal_contract(self):
        transition = PolicyTransition(
            {}, CanonicalMacroAction("phantom_dive", (2,), False, {"counters": [6]}),
            -0.4, -0.7, -1.1, 0.2, 1.0, 0.99, None, 0.0, True, (10, 17),
        )
        self.assertEqual(transition.engine_event_span, (10, 17))
        with self.assertRaises(ValueError):
            dataclasses.replace(transition, joint_old_logprob=-9.0)

    def test_forced_callback_count_does_not_change_gae(self):
        action = CanonicalMacroAction("phantom_dive", (2,), False, {"counters": [6]})
        transitions = [
            PolicyTransition({}, action, -0.2, -0.3, -0.5, 0.1, 0.0, 0.99,
                             {}, 0.4, False, (0, 7)),
            PolicyTransition({}, action, -0.1, 0.0, -0.1, 0.4, 1.0, 0.99,
                             None, 0.0, True, (7, 9)),
        ]
        targets = compound_gae(transitions, gae_lambda=0.95, credit_clock="selection")
        # Six internal selects occupy event span 0:7, but only one lambda factor exists.
        expected_last = 1.0 - 0.4
        expected_first = (0.99 * 0.4 - 0.1) + 0.99 * 0.95 * expected_last
        self.assertAlmostEqual(float(targets.advantages[0]), expected_first, places=6)

    def test_turn_clock_does_not_apply_lambda_within_one_turn(self):
        action = CanonicalMacroAction("root", (2,), False, {})
        transitions = [
            PolicyTransition({}, action, -0.2, 0.0, -0.2, 0.1, 0.0, 1.0,
                             {}, 0.4, False, (0, 1), metadata={"turn": 4}),
            PolicyTransition({}, action, -0.1, 0.0, -0.1, 0.4, 1.0, 1.0,
                             None, 0.0, True, (1, 2), metadata={"turn": 4}),
        ]

        targets = compound_gae(transitions, gae_lambda=0.95, credit_clock="turn")

        expected_last = 1.0 - 0.4
        expected_first = (0.4 - 0.1) + expected_last
        self.assertAlmostEqual(float(targets.advantages[0]), expected_first, places=6)

    def test_policy_statistics_ignore_non_strategic_rows_and_old_logprob_is_frozen(self):
        old = torch.tensor([-0.5, -9.0], requires_grad=True)
        result = compound_policy_loss(
            current_root_logprob=torch.tensor([-0.4, 3.0], requires_grad=True),
            current_parameter_logprob=torch.tensor([-0.2, 3.0], requires_grad=True),
            old_joint_logprob=old,
            root_entropy=torch.tensor([0.7, 99.0]), parameter_entropy=torch.tensor([0.3, 99.0]),
            advantage=torch.tensor([1.0, 1.0]), policy_mask=torch.tensor([True, False]),
            clip_ratio=0.1,
        )
        self.assertAlmostEqual(float(result.entropy), 1.0)
        result.policy_loss.backward()
        self.assertIsNone(old.grad)


class CheckpointContractTests(unittest.TestCase):
    def test_metadata_contains_all_action_contract_versions(self):
        metadata = checkpoint_module.checkpoint_metadata(
            source_actor_sha256="actor", source_value_sha256="value"
        )
        for key in (
            "action_schema_version", "decision_gate_version", "canonicalizer_version",
            "trajectory_schema_version", "official_protocol_adapter_version",
        ):
            self.assertIn(key, metadata)

    def test_rejects_optimizer_and_rollout_state(self):
        payload = {
            "schema_version": checkpoint_module.CHECKPOINT_SCHEMA_VERSION,
            "metadata": checkpoint_module.checkpoint_metadata(
                source_actor_sha256="actor", source_value_sha256="value"
            ),
            "optimizer_state_dict": {},
        }
        with self.assertRaisesRegex(ValueError, "forbidden"):
            checkpoint_module.validate_model_only_payload(payload)


if __name__ == "__main__":
    unittest.main()
