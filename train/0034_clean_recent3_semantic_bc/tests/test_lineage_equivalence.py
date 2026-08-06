from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAIN = importlib.import_module(
    "train.0034_clean_recent3_semantic_bc.domain"
)
MODEL = importlib.import_module(
    "train.0034_clean_recent3_semantic_bc.model"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LineageEquivalenceTests(unittest.TestCase):
    def test_immutable_prototype_asset_commitments(self) -> None:
        self.assertEqual(
            _sha256(PROJECT_ROOT / "assets/official_public_prototypes_v1.json"),
            "f4212a6c7b3e33d88c980599668c85fd3f86393964922629ef00630ba86e8776",
        )
        self.assertEqual(
            _sha256(PROJECT_ROOT / "assets/official_full_engine_prototypes_v2.json"),
            "cf37489f00152d2de77a81bc9ebdff9663575bb65f7660f96a0791cbbcb35fa1",
        )

    def test_default_policy_has_effect_summary_0034_capacity(self) -> None:
        prototypes = DOMAIN.PrototypeIndex.load(
            PROJECT_ROOT / "assets/official_public_prototypes_v1.json"
        )
        policy = MODEL.SemanticPolicy(MODEL.ModelConfig(), prototypes)
        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        self.assertEqual(parameter_count, 22_595_202)


if __name__ == "__main__":
    unittest.main()
