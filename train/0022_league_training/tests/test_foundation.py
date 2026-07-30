from __future__ import annotations

import unittest
from unittest import mock

from .. import foundation
from ..foundation import contract


class FoundationTest(unittest.TestCase):
    def test_identity_model_and_neutral_source_contract(self) -> None:
        model, identity = foundation.load_foundation("cpu")
        self.assertEqual(identity.epoch, 13)
        self.assertEqual(identity.parameter_count, 17_756_162)
        self.assertEqual(
            identity.weights_sha256,
            "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb",
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(foundation.neutral_source_id(4).tolist(), [0, 0, 0, 0])

    def test_changed_archive_sha_is_rejected(self) -> None:
        with mock.patch.object(contract, "_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "weights file SHA mismatch"):
                foundation.verify_foundation()


if __name__ == "__main__":
    unittest.main()
