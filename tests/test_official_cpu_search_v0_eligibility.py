from __future__ import annotations

import unittest
from dataclasses import replace

from tests.research.official_cpu_search_v0_eligibility import (
    CandidateEvidence,
    Eligibility,
    is_v0_value_search_candidate,
)


class OfficialCpuSearchV0EligibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eligible = CandidateEvidence(
            focal_player=0,
            root_select_player=0,
            legal_selection_count=2,
            distinct_value_visible_afterstates=2,
            branch_terminal=False,
            branch_select_player=0,
            root_turn=7,
            branch_turn=7,
            root_active_player=0,
            branch_active_player=0,
            root_main_phase=True,
            branch_main_phase=True,
            rng_unchanged=True,
            hidden_invariant=True,
            feature_compatible=True,
        )

    def test_passes_only_complete_positive_evidence(self) -> None:
        self.assertEqual(
            is_v0_value_search_candidate(self.eligible), Eligibility.ELIGIBLE
        )

    def test_rejects_forced_or_value_equivalent_choices(self) -> None:
        for update in (
            {"legal_selection_count": 1},
            {"distinct_value_visible_afterstates": 1},
        ):
            with self.subTest(update=update):
                self.assertEqual(
                    is_v0_value_search_candidate(replace(self.eligible, **update)),
                    Eligibility.REJECT_NOT_ACTUAL_CHOICE,
                )

    def test_rejects_each_unsafe_boundary(self) -> None:
        cases = (
            ({"root_select_player": 1}, Eligibility.REJECT_PERSPECTIVE_FLIP),
            ({"branch_select_player": 1}, Eligibility.REJECT_PERSPECTIVE_FLIP),
            ({"branch_terminal": True}, Eligibility.REJECT_TERMINAL),
            ({"branch_turn": 8}, Eligibility.REJECT_TURN_CHANGE),
            ({"branch_active_player": 1}, Eligibility.REJECT_TURN_CHANGE),
            ({"root_active_player": 1, "branch_active_player": 1}, Eligibility.REJECT_TURN_CHANGE),
            ({"root_main_phase": False}, Eligibility.REJECT_TURN_CHANGE),
            ({"branch_main_phase": False}, Eligibility.REJECT_TURN_CHANGE),
            ({"rng_unchanged": False}, Eligibility.REJECT_RNG),
            ({"hidden_invariant": False}, Eligibility.REJECT_HIDDEN_SENSITIVE),
            (
                {"feature_compatible": False},
                Eligibility.REJECT_FEATURE_INCOMPATIBLE,
            ),
        )
        for update, expected in cases:
            with self.subTest(update=update):
                self.assertEqual(
                    is_v0_value_search_candidate(replace(self.eligible, **update)),
                    expected,
                )

    def test_missing_evidence_fails_closed(self) -> None:
        for field in CandidateEvidence.__dataclass_fields__:
            with self.subTest(field=field):
                evidence = replace(self.eligible, **{field: None})
                self.assertEqual(
                    is_v0_value_search_candidate(evidence), Eligibility.UNKNOWN
                )


if __name__ == "__main__":
    unittest.main()
