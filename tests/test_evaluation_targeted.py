from __future__ import annotations

import unittest

from evaluation.cli import _validate_research_coverage
from evaluation.packages.loader import PackageValidationError


class TargetedEvaluationTests(unittest.TestCase):
    def test_temporary_evaluation_allows_catalog_subset(self) -> None:
        _validate_research_coverage(
            "one",
            100,
            tuple(range(1)),  # type: ignore[arg-type]
            require_full_catalog=False,
        )

    def test_formal_evaluation_still_requires_full_catalog(self) -> None:
        with self.assertRaisesRegex(PackageValidationError, "opponents all"):
            _validate_research_coverage(
                "one",
                100,
                tuple(range(1)),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
