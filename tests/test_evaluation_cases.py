from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.cases import CaseCandidate, case_record, select_cases, write_case_records


def candidate(
    game_id: str,
    *,
    opponent: str,
    status: str = "finished",
    failure_class: str = "",
    is_loss: bool = True,
    metric_ids: tuple[str, ...] = (),
) -> CaseCandidate:
    return CaseCandidate(
        game_id=game_id,
        opponent=opponent,
        status=status,
        failure_class=failure_class,
        is_loss=is_loss,
        metric_ids=metric_ids,
        evidence_steps=(
            {
                "step": 7,
                "turn": 4,
                "role": "candidate",
                "state_summary": {"deck_count": 12, "prize_count": 3},
                "actual_action": [2],
                "expected_action": "play_rare_candy",
                "expected_reason": "complete the active Alakazam route",
            },
        ),
        trace_path=Path("/tmp") / f"{game_id}.json",
    )


class EvaluationCaseTests(unittest.TestCase):
    def test_select_cases_prioritizes_failures_and_diversifies_ties(self) -> None:
        candidates = [
            candidate(
                "04-normal-loss",
                opponent="water",
                failure_class="ordinary_loss",
            ),
            candidate(
                "06-post-ko",
                opponent="bolt",
                failure_class="post_ko_zero_ready",
                metric_ids=("post_ko_relay",),
            ),
            candidate(
                "03-rare-candy",
                opponent="psychic",
                failure_class="rare_candy_not_played",
                metric_ids=("rare_candy",),
            ),
            candidate(
                "02-candidate-crash",
                opponent="bolt",
                status="candidate_error",
                failure_class="candidate_error",
            ),
            candidate(
                "01-engine-error",
                opponent="water",
                status="engine_error",
                failure_class="engine_error",
            ),
            candidate(
                "05-duplicate-opponent",
                opponent="water",
                failure_class="attack_not_declared",
                metric_ids=("rare_candy",),
            ),
        ]

        selected = select_cases(candidates, limit=3)

        self.assertEqual(
            [item.game_id for item in selected],
            ["01-engine-error", "02-candidate-crash", "03-rare-candy"],
        )
        self.assertEqual({item.opponent for item in selected}, {"water", "bolt", "psychic"})
        self.assertEqual(
            {item.failure_class for item in selected},
            {"engine_error", "candidate_error", "rare_candy_not_played"},
        )

    def test_select_cases_is_order_independent_and_excludes_ordinary_losses(self) -> None:
        candidates = [
            candidate("z-normal", opponent="water", failure_class="ordinary_loss"),
            candidate(
                "b-resource",
                opponent="water",
                failure_class="empty_bench_run_away_draw",
                metric_ids=("run_away_draw",),
            ),
            candidate(
                "a-target",
                opponent="water",
                failure_class="attack_not_declared",
                metric_ids=("rare_candy",),
            ),
            candidate(
                "c-resource-different-opponent",
                opponent="bolt",
                failure_class="post_ko_zero_ready",
                metric_ids=("post_ko_relay",),
            ),
        ]

        selected = select_cases(candidates, limit=3)
        reversed_selected = select_cases(list(reversed(candidates)), limit=3)

        self.assertEqual([item.game_id for item in selected], [
            "a-target",
            "c-resource-different-opponent",
            "b-resource",
        ])
        self.assertEqual(
            [item.game_id for item in reversed_selected],
            [item.game_id for item in selected],
        )
        self.assertEqual(select_cases([candidates[0]]), [])

    def test_select_cases_only_uses_losses_for_target_and_resource_cases(self) -> None:
        candidates = [
            candidate(
                "winning-target",
                opponent="water",
                failure_class="attack_not_declared",
                is_loss=False,
            ),
            candidate(
                "winning-resource",
                opponent="bolt",
                failure_class="library_deck_out",
                metric_ids=("library_pressure",),
                is_loss=False,
            ),
            candidate(
                "losing-target",
                opponent="psychic",
                failure_class="attack_not_declared",
                is_loss=True,
            ),
            candidate(
                "non-loss-error",
                opponent="fire",
                status="visualization_error",
                failure_class="visualization_error",
                is_loss=False,
            ),
        ]

        selected = select_cases(candidates, limit=3)

        self.assertEqual(
            [item.game_id for item in selected],
            ["non-loss-error", "losing-target"],
        )

    def test_select_cases_validates_limit(self) -> None:
        with self.assertRaises(ValueError):
            select_cases([], limit=-1)
        self.assertEqual(select_cases([], limit=0), [])

    def test_select_cases_caps_positive_limit_at_three(self) -> None:
        selected = select_cases(
            [
                candidate("target-a", opponent="water", failure_class="attack_not_declared"),
                candidate("target-b", opponent="bolt", failure_class="rare_candy_not_played"),
                candidate("target-c", opponent="psychic", failure_class="no_legal_attack"),
                candidate("target-d", opponent="fire", failure_class="alakazam_not_active"),
            ],
            limit=99,
        )

        self.assertEqual([item.game_id for item in selected], ["target-a", "target-b", "target-c"])

    def test_select_cases_rejects_duplicate_game_ids(self) -> None:
        candidates = [
            candidate("duplicate", opponent="water", failure_class="attack_not_declared"),
            candidate("duplicate", opponent="bolt", failure_class="rare_candy_not_played"),
        ]

        with self.assertRaisesRegex(ValueError, r"duplicate game_id: duplicate"):
            select_cases(candidates)

    def test_select_cases_normalizes_failure_classes_for_diversity(self) -> None:
        selected = select_cases(
            [
                candidate(
                    "a-spaced",
                    opponent="water",
                    failure_class="Attack Not Declared",
                ),
                candidate(
                    "b-normalized-duplicate",
                    opponent="water",
                    failure_class="attack_not_declared",
                ),
                candidate(
                    "c-distinct",
                    opponent="water",
                    failure_class="rare_candy_not_played",
                ),
            ],
            limit=2,
        )

        self.assertEqual([item.game_id for item in selected], ["a-spaced", "c-distinct"])

    def test_select_cases_recognizes_crash_failure_class_as_an_error(self) -> None:
        selected = select_cases(
            [
                candidate(
                    "target",
                    opponent="water",
                    failure_class="rare_candy_not_played",
                ),
                candidate(
                    "crash",
                    opponent="bolt",
                    failure_class="candidate_crash",
                ),
            ],
            limit=1,
        )

        self.assertEqual([item.game_id for item in selected], ["crash"])

    def test_case_record_and_jsonl_are_explainable_and_json_serializable(self) -> None:
        item = candidate(
            "rare-candy-case",
            opponent="water",
            failure_class="rare_candy_not_played",
            metric_ids=("rare_candy",),
        )

        record = case_record(item)

        self.assertEqual(record["game_id"], "rare-candy-case")
        self.assertEqual(record["metric_ids"], ["rare_candy"])
        self.assertEqual(record["state_summary"], {"deck_count": 12, "prize_count": 3})
        self.assertEqual(record["actual_action"], [2])
        self.assertEqual(record["expected_action"], "play_rare_candy")
        self.assertEqual(record["expected_reason"], "complete the active Alakazam route")
        self.assertEqual(record["trace_path"], "/tmp/rare-candy-case.json")
        json.dumps(record)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cases.jsonl"
            write_case_records([item], output)
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(persisted, record)


if __name__ == "__main__":
    unittest.main()
