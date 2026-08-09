from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class RandomOutcomeInjectionIsolationTest(unittest.TestCase):
    def test_fixture_header_is_not_in_production_translation_units(self):
        needle = "ptcg_cuda/testing/random_outcome_fixture.cuh"
        allowed = {
            ROOT / "engine_cuda/benchmarks/official_random_outcome_injection_paired.cu",
        }
        hits = []
        for base in (ROOT / "engine_cuda/src", ROOT / "engine_cuda/include"):
            for path in base.rglob("*"):
                if not path.is_file() or path.suffix not in {".cu", ".cuh", ".cpp", ".h"}:
                    continue
                if needle in path.read_text(encoding="utf-8", errors="ignore"):
                    hits.append(path)
        self.assertEqual(hits, [])
        self.assertIn(needle, next(iter(allowed)).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
