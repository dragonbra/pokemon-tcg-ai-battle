from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOMAIN = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.domain"
)
MODEL = importlib.import_module(
    "train.0031_rule_faithful_semantic_foundation_pretraining.model"
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
            _sha256(PROJECT_ROOT / "assets/official_full_engine_prototypes_v2.json"),
            "7228b2612da118e1ec353c7a2e08d4a306de0d4128f0c9e68b4075f2c6b54f73",
        )

    def test_default_policy_has_audited_0031_capacity(self) -> None:
        prototypes = DOMAIN.PrototypeIndex.load(
            PROJECT_ROOT / "assets/official_public_prototypes_v1.json"
        )
        policy = MODEL.SemanticPolicy(MODEL.ModelConfig(), prototypes)
        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        self.assertEqual(parameter_count, 55_868_802)


if __name__ == "__main__":
    unittest.main()
