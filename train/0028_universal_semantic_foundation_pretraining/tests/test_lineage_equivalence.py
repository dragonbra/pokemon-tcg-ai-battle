from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAIN = importlib.import_module(
    "train.0028_universal_semantic_foundation_pretraining.domain"
)
MODEL = importlib.import_module(
    "train.0028_universal_semantic_foundation_pretraining.model"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LineageEquivalenceTests(unittest.TestCase):
    def test_immutable_prototype_asset_commitments(self) -> None:
        self.assertEqual(
            _sha256(PROJECT_ROOT / "assets/official_public_prototypes_v1.json"),
            "5b45041cd14beed8a847f2e99ce955a105fbeb63d09256769d41f8cc88bdbc9d",
        )
        self.assertEqual(
            _sha256(PROJECT_ROOT / "assets/official_full_engine_prototypes_v1.json"),
            "5ab0b28e21d40a17a7b153744332424f08d4e32cde4e2ab20f11cb262e799845",
        )

    def test_default_policy_matches_0025_v4_capacity(self) -> None:
        prototypes = DOMAIN.PrototypeIndex.load(
            PROJECT_ROOT / "assets/official_public_prototypes_v1.json"
        )
        policy = MODEL.SemanticPolicy(MODEL.ModelConfig(), prototypes)
        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        self.assertEqual(parameter_count, 21_837_082)


if __name__ == "__main__":
    unittest.main()
