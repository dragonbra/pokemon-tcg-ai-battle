from __future__ import annotations

import importlib
import json
from pathlib import Path
import re
import unittest


runtime = importlib.import_module(
    "train.0034_dragapult_third_large_model_rl.training.runtime"
)
transfer = importlib.import_module("train.0034_dragapult_third_large_model_rl.transfer")
exporter = importlib.import_module(
    "train.0034_dragapult_third_large_model_rl.export_zero_shot_candidate"
)
full_runner = importlib.import_module(
    "train.0034_dragapult_third_large_model_rl.training.run_full_semantic"
)


class ProjectIdentityTest(unittest.TestCase):
    def test_exact_focal_deck_identity(self) -> None:
        deck = runtime.read_deck(runtime.FOCAL_DECK_PATH)
        self.assertEqual(len(deck), 60)
        self.assertEqual(
            runtime.deck_multiset_sha256(deck), runtime.FOCAL_DECK_MULTISET_SHA256
        )
        manifest = json.loads(runtime.FOCAL_DECK_PATH.with_name("manifest.json").read_text())
        self.assertEqual(manifest["deck_id"], runtime.FOCAL_DECK_ID)
        self.assertEqual(
            manifest["deck_file_sha256"], runtime.FOCAL_DECK_FILE_SHA256
        )
        self.assertEqual(
            manifest["deck_multiset_sha256"], runtime.FOCAL_DECK_MULTISET_SHA256
        )

    def test_large_model_0806_epoch11_source_identity(self) -> None:
        self.assertEqual(transfer.SOURCE_EPOCH, 11)
        self.assertEqual(transfer.SOURCE_GLOBAL_STEP, 141878)
        self.assertEqual(
            transfer.SOURCE_VERSION,
            "V2_full_winners_bs1024_20260616_20260803",
        )
        self.assertEqual(
            transfer.SOURCE_CHECKPOINT_SHA256,
            "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8",
        )

    def test_no_executable_cross_project_imports(self) -> None:
        project = Path("train/0034_dragapult_third_large_model_rl")
        forbidden = re.compile(r"(?:from|import)\s+train\.00(?:0[0-9]|[12][0-9]|3[0-3])[_\w.]*")
        violations = []
        for path in project.rglob("*.py"):
            if match := forbidden.search(path.read_text(encoding="utf-8")):
                violations.append(f"{path}: {match.group(0)}")
        self.assertEqual(violations, [])

    def test_exporter_normalizes_relative_repository_path(self) -> None:
        relative = Path(
            "archive/pretrained/0031_friend_0806_epoch11_best_validation_loss/model.pt"
        )
        self.assertEqual(exporter._repo_relative(relative), relative)

    def test_wandb_display_name_keeps_project_prefix(self) -> None:
        self.assertEqual(
            full_runner.WANDB_DISPLAY_PREFIX,
            "0034 · dragapult_third_large_model_rl",
        )


if __name__ == "__main__":
    unittest.main()
