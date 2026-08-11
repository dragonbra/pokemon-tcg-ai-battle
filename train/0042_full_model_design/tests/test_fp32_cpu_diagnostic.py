from __future__ import annotations

import importlib
from types import SimpleNamespace
import unittest


diagnostic = importlib.import_module(
    "train.0042_full_model_design.diagnostics.fp32_cpu2048_policy0809"
)


def episode(game_id: str, *, valid: bool = True, error=None, reason=None):
    return SimpleNamespace(
        valid=valid,
        error=error,
        job=SimpleNamespace(game_id=game_id),
        diagnostics={"macro_fallback_reason": reason},
    )


class CpuDiagnosticHealthTest(unittest.TestCase):
    def test_cpu_health_does_not_require_cuda_metrics(self) -> None:
        rows = [episode("g1"), episode("g2")]
        diagnostic._assert_cpu_chunk_health(rows, expected_games=2)

    def test_cpu_health_allows_audited_chance_boundary(self) -> None:
        rows = [episode("g1", reason=diagnostic.runner.CHANCE_BOUNDARY_FALLBACK)]
        diagnostic._assert_cpu_chunk_health(rows, expected_games=1)

    def test_cpu_health_rejects_semantic_fallback(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "semantic_fallbacks=1"):
            diagnostic._assert_cpu_chunk_health(
                [episode("g1", reason="wrong_semantics")], expected_games=1
            )


if __name__ == "__main__":
    unittest.main()
