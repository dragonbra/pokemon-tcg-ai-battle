from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prune_experiment_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("prune_experiment_checkpoints", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CheckpointRetentionTest(unittest.TestCase):
    def test_priority_and_apply_keep_one_recognized_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_a = root / "run_a"
            run_b = root / "nested" / "run_b"
            run_a.mkdir()
            run_b.mkdir(parents=True)

            for name in ("policy_iter_0001.pt", "policy_iter_0009.pt", "checkpoint_last.pt"):
                (run_a / name).write_bytes(name.encode())
            for name in ("bc_transformer_epoch01.pt", "bc_transformer_epoch06.pt", "policy_best.pt"):
                (run_b / name).write_bytes(name.encode())
            unrelated = run_b / "semantic_aux_probe.pt"
            unrelated.write_bytes(b"unrelated")

            groups = MODULE.build_plan(root)
            self.assertEqual(len(groups), 2)
            kept = {Path(group.keep.path).name for group in groups}
            self.assertEqual(kept, {"checkpoint_last.pt", "policy_best.pt"})

            removed_files, _ = MODULE.apply_plan(root, groups)
            self.assertEqual(removed_files, 4)
            self.assertEqual(
                {path.name for path in root.rglob("*.pt")},
                {"checkpoint_last.pt", "policy_best.pt", "semantic_aux_probe.pt"},
            )


if __name__ == "__main__":
    unittest.main()
