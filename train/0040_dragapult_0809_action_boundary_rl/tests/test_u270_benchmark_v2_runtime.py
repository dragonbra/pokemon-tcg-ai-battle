import importlib
from pathlib import Path
import unittest


target = importlib.import_module(
    "train.0040_dragapult_0809_action_boundary_rl.evaluate_u270_benchmark_v2"
)


class U270BenchmarkV2RuntimeTest(unittest.TestCase):
    def test_current_engine2_runtime_is_bound_before_legacy_collector_import(self) -> None:
        expected_python = target.ROOT / "engine_cuda_2_0/python"
        expected_extension = target.ROOT / "engine_cuda_2_0/build/native"
        self.assertEqual(target.CURRENT_CUDA_PYTHON, expected_python)
        self.assertEqual(target.legacy_evaluation.CUDA_EXTENSION, expected_extension)
        self.assertTrue(Path(target.ptcg_cuda_engine.__file__).is_relative_to(expected_python))


if __name__ == "__main__":
    unittest.main()
