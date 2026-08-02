from __future__ import annotations

import importlib
import unittest


BASE = "train.0026_raging_bolt_canonical_decoder_rl"


class RolloutContractTests(unittest.TestCase):
    def test_smoke_schedule_balances_seats_and_uses_frozen_decks(self) -> None:
        smoke = importlib.import_module(f"{BASE}.smoke")
        jobs = smoke.smoke_jobs(4)
        self.assertEqual(len(jobs), 4)
        self.assertEqual([job.focal_first for job in jobs], [True, False, True, False])
        self.assertEqual(jobs[0].opponent_id, jobs[1].opponent_id)
        self.assertEqual(jobs[2].opponent_id, jobs[3].opponent_id)
        self.assertNotEqual(jobs[0].opponent_id, jobs[2].opponent_id)
        self.assertTrue(all(len(job.focal_deck) == len(job.opponent_deck) == 60 for job in jobs))
        self.assertTrue(all((job.runtime_root / "cg/game.py").is_file() for job in jobs))

    def test_formal_schedule_covers_all_51_and_balances_seats(self) -> None:
        run = importlib.import_module(f"{BASE}.training.run")
        jobs = run.build_jobs(count=512, source_policy_update=3, seed=17)
        self.assertEqual(len(jobs), 512)
        self.assertEqual(len({job.opponent_id for job in jobs}), 51)
        self.assertEqual(sum(job.focal_first for job in jobs), 256)
        self.assertEqual({job.source_policy_update for job in jobs}, {3})

    def test_frozen_eval_is_one_game_per_opponent_and_seat(self) -> None:
        run = importlib.import_module(f"{BASE}.training.run")
        jobs = run.build_jobs(
            count=102, source_policy_update=5, seed=19, greedy_eval=True
        )
        pairs = {(job.opponent_id, job.focal_first) for job in jobs}
        self.assertEqual(len(pairs), 102)


if __name__ == "__main__":
    unittest.main()
