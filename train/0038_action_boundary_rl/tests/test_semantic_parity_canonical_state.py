from __future__ import annotations

import copy
import importlib
import unittest


module = importlib.import_module("train.0038_action_boundary_rl.semantic_parity.canonical_state")


def snapshot(backend: str = "official_cpu") -> dict:
    player = lambda seat: {
        "player": seat, "active": [1 + seat], "bench": [3 + seat],
        "hand": [5 + seat], "deck": [7 + seat, 9 + seat], "discard": [],
        "prizes": [11 + seat], "energy": [13 + seat], "tools": [],
        "pre_evolution": [], "temporary": [], "turn_flags": 0,
        "once_per_turn": [], "supporter_used": False, "retreat_used": False,
        "energy_attached": False,
    }
    return {
        "backend": backend,
        "game": {"episode_id": 1, "turn": 4, "phase": 1, "seat": 0,
                 "first_player": 1, "turn_action_count": 3,
                 "effect_action_count": 2, "move_counter": 19},
        "players": [player(0), player(1)],
        "cards": [{"stable_serial": 1, "card_id": 101, "owner": 0,
                   "zone": "active", "zone_index": 0, "hp": 200,
                   "damage": 30, "status": "none", "energy": [13],
                   "tools": [], "pre_evolution": [], "turn_flags": 0,
                   "continual_flags": 0, "ability_used": []}],
        "zones": {"stadium": [], "looking": [], "playing": []},
        "pending": {"select_type": 4, "actor": 0, "min": 1, "max": 1},
        "effect_stack": [{"effect": 7}],
        "continuation_stack": [{"opcode": 79, "args": [0, 1, 0]}],
        "legal_options": [{"type": 4, "params": [1]}],
        "event_history": [{"type": "damage", "amount": 30}],
        "result": {"terminal": False, "winner": None, "game_result": 0,
                   "finish_reason": 0, "error": 0, "error_detail": 0},
        "reward": {"reward": 0.0, "prize_delta": [0, 0], "terminal_reward": 0.0},
    }


class CanonicalAuthorityStateTest(unittest.TestCase):
    def test_cpu_cuda_identical_rule_state_hashes(self):
        cpu = module.CanonicalAuthorityState.from_official(snapshot())
        cuda = module.CanonicalAuthorityState.from_cuda(snapshot("cuda"))
        self.assertEqual(cpu.sha256, cuda.sha256)
        self.assertEqual(module.canonical_state_diff(cpu, cuda), [])

    def test_complete_deck_prize_order_and_continuation_are_rule_fields(self):
        left = module.CanonicalAuthorityState.from_official(snapshot())
        changed = snapshot()
        changed["players"][0]["deck"].reverse()
        changed["continuation_stack"][0]["opcode"] = 80
        right = module.CanonicalAuthorityState.from_official(changed)
        paths = {difference.path for difference in module.canonical_state_diff(left, right)}
        self.assertIn("players[0].deck[0]", paths)
        self.assertIn("continuation_stack[0].opcode", paths)

    def test_missing_rule_field_fails_closed(self):
        broken = snapshot()
        del broken["cards"][0]["damage"]
        with self.assertRaisesRegex(ValueError, "damage"):
            module.CanonicalAuthorityState.from_official(broken)

    def test_only_explicit_non_rule_exclusions_are_allowed(self):
        allowed = snapshot()
        allowed["host_pointer"] = "0x123"
        allowed["excluded_fields"] = ["host_pointer"]
        state = module.CanonicalAuthorityState.from_official(allowed)
        self.assertNotIn("host_pointer", state.payload)
        denied = copy.deepcopy(allowed)
        denied["excluded_fields"] = ["damage"]
        with self.assertRaisesRegex(ValueError, "unapproved"):
            module.CanonicalAuthorityState.from_official(denied)


if __name__ == "__main__":
    unittest.main()
