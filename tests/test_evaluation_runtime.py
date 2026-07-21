from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

from evaluation.packages.loader import PackageValidationError, SubmissionPackage
from evaluation.runtime.loader import assert_cg_compatible, compute_cg_manifest, load_game_api


class EvaluationRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_runtime(self, name: str, *, api_value: int = 1, native: bool = True) -> Path:
        runtime_root = self.root / name
        cg_root = runtime_root / "cg"
        cg_root.mkdir(parents=True)
        (cg_root / "__init__.py").write_text("", encoding="utf-8")
        (cg_root / "api.py").write_text(f"API_VALUE = {api_value}\n", encoding="utf-8")
        (cg_root / "game.py").write_text(
            "from cg.api import API_VALUE\nGAME_VALUE = API_VALUE\n",
            encoding="utf-8",
        )
        if native:
            (cg_root / "libcg.so").write_bytes(b"test-runtime")
        return runtime_root

    def package(self, name: str, runtime_root: Path) -> SubmissionPackage:
        return SubmissionPackage(
            name=name,
            root=runtime_root,
            deck=[1] * 60,
            entrypoint=runtime_root / "main.py",
            package_hash=name,
            deck_hash=name,
            cg_manifest=compute_cg_manifest(runtime_root / "cg"),
        )

    def test_identical_runtime_manifests_are_compatible(self) -> None:
        candidate_root = self.make_runtime("candidate")
        opponent_root = self.make_runtime("opponent")

        assert_cg_compatible(
            self.package("candidate", candidate_root),
            self.package("opponent", opponent_root),
        )

    def test_runtime_change_is_incompatible(self) -> None:
        candidate_root = self.make_runtime("candidate")
        opponent_root = self.make_runtime("opponent", api_value=2)

        with self.assertRaisesRegex(PackageValidationError, "cg hash mismatch"):
            assert_cg_compatible(
                self.package("candidate", candidate_root),
                self.package("opponent", opponent_root),
            )

    def test_manifest_requires_game_api_and_native_library(self) -> None:
        runtime_root = self.make_runtime("candidate", native=False)

        with self.assertRaisesRegex(PackageValidationError, "native"):
            compute_cg_manifest(runtime_root / "cg")

    def test_manifest_requires_game_and_api_modules(self) -> None:
        runtime_root = self.make_runtime("candidate")
        (runtime_root / "cg" / "api.py").unlink()

        with self.assertRaisesRegex(PackageValidationError, "api.py"):
            compute_cg_manifest(runtime_root / "cg")

    def test_load_game_api_uses_only_requested_runtime(self) -> None:
        candidate_root = self.make_runtime("candidate", api_value=1)
        opponent_root = self.make_runtime("opponent", api_value=2)

        candidate_game = load_game_api(candidate_root)
        self.assertEqual(candidate_game.GAME_VALUE, 1)
        self.assertNotIn("cg", sys.modules)
        self.assertNotIn("cg.api", sys.modules)

        opponent_game = load_game_api(opponent_root)

        self.assertEqual(opponent_game.GAME_VALUE, 2)
        self.assertNotIn("cg", sys.modules)
        self.assertNotIn("cg.api", sys.modules)


if __name__ == "__main__":
    unittest.main()
