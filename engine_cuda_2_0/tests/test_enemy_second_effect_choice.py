from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EnemySecondEffectChoiceContractTest(unittest.TestCase):
    def test_play_path_preserves_enemy_choice_identity(self) -> None:
        main = (ROOT / "include/ptcg_cuda/official_main_pod.cuh").read_text()
        interpreter = (
            ROOT / "include/ptcg_cuda/official_effect_interpreter_pod.cuh"
        ).read_text()
        continuation = (
            ROOT / "include/ptcg_cuda/official_continuation_pod.cuh"
        ).read_text()
        effects = (ROOT / "include/ptcg_cuda/official_effects_pod.cuh").read_text()
        core = (ROOT / "include/ptcg_cuda/official_core_pod.cuh").read_text()
        paired = (
            ROOT / "extractor/official_battle_end_turn_paired.cpp"
        ).read_text()

        self.assertIn("kSkillSecondEffectStartEnemy", main)
        self.assertIn("kOfficialEffectSelectContextActivate", main)
        self.assertIn("1 - player", main)
        self.assertIn("kEnemySkillChooseEffect = 9", interpreter)
        self.assertIn("own_second > 0 ? own_second : enemy_second", interpreter)
        self.assertIn("kEnemySelectedWhichEffect", continuation)
        self.assertIn(
            "if (resume == OfficialEffectResumeKind::kSelectActivate)",
            continuation,
        )
        prize_case = effects.index("case OfficialEffectTypeId::kPrizeToHand:")
        prize_break = effects.index("break;", prize_case)
        self.assertIn(
            "target.card, OfficialArea::kHand, false, false, 1",
            effects[prize_case:prize_break],
        )
        self.assertNotIn("official_pod_mark_changed", effects[prize_case:prize_break])
        self.assertIn("kCardKoEnemyTerastalAttackDamage", core)
        self.assertIn("count += attacker_turn_state[6]", core)
        self.assertIn("count += attacker_turn_state[7]", core)
        ko_case = effects.index("case OfficialEffectTypeId::kKo:")
        ko_break = effects.index("break;", ko_case)
        self.assertIn(
            "official_effect_blocks_target_effect",
            effects[ko_case:ko_break],
        )
        switch_preflight = interpreter.index(
            "if (effect_type == OfficialEffectTypeId::kSwitch)"
        )
        self.assertIn(
            "official_effect_blocks_active_effect",
            interpreter[switch_preflight:switch_preflight + 2200],
        )
        terminal = paired.index("OfficialGameResult::kNone")
        self.assertIn("state->select_context = 0", paired[terminal:terminal + 1200])


if __name__ == "__main__":
    unittest.main()
