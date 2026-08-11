from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest


parity = importlib.import_module("train.0042_full_model_design.parity")


class ParityModuleIsolationTest(unittest.TestCase):
    def test_runtime_types_do_not_leave_strategy_namespace_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployment = root / "strategy/deployment"
            deployment.mkdir(parents=True)
            for path in (
                root / "strategy/__init__.py",
                deployment / "__init__.py",
            ):
                path.write_text("")
            (deployment / "online_runtime.py").write_text(
                "class OnlineCausalEncoder:\n    pass\n"
            )
            (deployment / "inference.py").write_text(
                "class PortableSemanticPolicy:\n    pass\n"
            )
            encoder, policy = parity._load_evaluated_runtime(root)
            self.assertEqual(encoder.__name__, "OnlineCausalEncoder")
            self.assertEqual(policy.__name__, "PortableSemanticPolicy")
            self.assertNotIn(str(root.resolve()), sys.path)
            self.assertFalse(any(
                name == "strategy" or name.startswith("strategy.")
                for name in sys.modules
            ))

    def test_preexisting_strategy_namespace_fails_closed(self) -> None:
        sys.modules["strategy"] = object()
        try:
            with self.assertRaisesRegex(RuntimeError, "stale strategy modules"):
                parity._load_evaluated_runtime(Path("."))
        finally:
            sys.modules.pop("strategy", None)


if __name__ == "__main__":
    unittest.main()
