from __future__ import annotations

import importlib
from pathlib import Path
import unittest


builder = importlib.import_module(
    "train.0042_full_model_design.tools.build_cuda_extension"
)


class BuildCudaExtensionTest(unittest.TestCase):
    def test_normalizes_torch_compute_capability_for_cmake(self) -> None:
        self.assertEqual(builder.normalize_architecture((12, 0)), "120")
        self.assertEqual(builder.normalize_architecture((8, 6)), "86")
        self.assertEqual(builder.normalize_architecture("120"), "120")
        self.assertEqual(builder.normalize_architecture("sm_86"), "86")

    def test_rejects_invalid_architecture(self) -> None:
        for value in ("", "8.6", "compute_86", (8,), (8, -1)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                builder.normalize_architecture(value)

    def test_default_output_matches_0042_runtime_contract(self) -> None:
        self.assertEqual(
            builder.default_build_dir(Path("/repo")),
            Path("/repo/.tmp/engine_cuda_benchmark/build_sm120_staged"),
        )

    def test_build_commands_bind_torch_and_detected_architecture(self) -> None:
        configure, compile_command = builder.build_commands(
            repository_root=Path("/repo"),
            build_dir=Path("/repo/.tmp/build"),
            torch_cmake_dir=Path("/python/torch/share/cmake/Torch"),
            architecture="86",
            jobs=3,
        )
        self.assertEqual(configure[:3], ["cmake", "-S", "/repo/engine_cuda"])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", configure)
        self.assertIn("-DPTCG_CUDA_BUILD_TORCH=ON", configure)
        self.assertIn("-DTorch_DIR=/python/torch/share/cmake/Torch", configure)
        self.assertEqual(
            compile_command,
            ["cmake", "--build", "/repo/.tmp/build", "--target", "_ptcg_cuda", "--parallel", "3"],
        )


if __name__ == "__main__":
    unittest.main()
