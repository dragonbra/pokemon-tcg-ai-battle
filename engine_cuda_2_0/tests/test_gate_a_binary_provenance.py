from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine_cuda_2_0.tools.run_official_battle_ordered_matrix_cuda import (
    persistent_build_contract,
    sha256_file,
    validate_persistent_build_manifest,
)


class GateABinaryProvenanceTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        engine = root / "engine_cuda_2_0"
        source = root / "official"
        rules = root / "official_rules.bin"
        executable = root / "paired"
        (engine / "include" / "ptcg_cuda").mkdir(parents=True)
        (engine / "extractor").mkdir(parents=True)
        (engine / "benchmarks").mkdir(parents=True)
        (engine / "src").mkdir(parents=True)
        source.mkdir()
        (source / "State.h").write_text("official-v1\n", encoding="utf-8")
        (engine / "include" / "ptcg_cuda" / "state.cuh").write_text(
            "cuda-v1\n", encoding="utf-8"
        )
        (engine / "extractor" / "bridge.cpp").write_text(
            "bridge-v1\n", encoding="utf-8"
        )
        (engine / "benchmarks" / "official_battle_ordered_matrix_cuda_paired.cu").write_text(
            "matrix-v1\n", encoding="utf-8"
        )
        (engine / "benchmarks" / "official_battle_end_turn_cuda_paired.cu").write_text(
            "paired-v1\n", encoding="utf-8"
        )
        (engine / "src" / "official_engine_kernels.cu").write_text(
            "kernel-v1\n", encoding="utf-8"
        )
        rules.write_bytes(b"rules-v1")
        executable.write_bytes(b"binary-v1")
        return engine, source, rules, executable

    def test_matching_manifest_allows_cached_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, source, rules, executable = self._fixture(root)
            expected = persistent_build_contract(
                source=source,
                rules=rules,
                image="cuda:test",
                branch_coverage=False,
                nvcc_threads=2,
                engine_root=engine,
            )
            manifest = root / "build_manifest.json"
            manifest.write_text(
                json.dumps({
                    **expected,
                    "executable_sha256": sha256_file(executable),
                }),
                encoding="utf-8",
            )
            validate_persistent_build_manifest(manifest, executable, expected)

    def test_source_change_rejects_cached_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, source, rules, executable = self._fixture(root)
            recorded = persistent_build_contract(
                source=source,
                rules=rules,
                image="cuda:test",
                branch_coverage=False,
                nvcc_threads=2,
                engine_root=engine,
            )
            manifest = root / "build_manifest.json"
            manifest.write_text(
                json.dumps({
                    **recorded,
                    "executable_sha256": sha256_file(executable),
                }),
                encoding="utf-8",
            )
            (engine / "include" / "ptcg_cuda" / "state.cuh").write_text(
                "cuda-v2\n", encoding="utf-8"
            )
            expected = persistent_build_contract(
                source=source,
                rules=rules,
                image="cuda:test",
                branch_coverage=False,
                nvcc_threads=2,
                engine_root=engine,
            )
            with self.assertRaisesRegex(SystemExit, "stale Gate-A binary"):
                validate_persistent_build_manifest(manifest, executable, expected)

    def test_missing_manifest_rejects_cached_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine, source, rules, executable = self._fixture(root)
            expected = persistent_build_contract(
                source=source,
                rules=rules,
                image="cuda:test",
                branch_coverage=False,
                nvcc_threads=2,
                engine_root=engine,
            )
            with self.assertRaisesRegex(SystemExit, "unprovenanced"):
                validate_persistent_build_manifest(
                    root / "missing.json", executable, expected
                )


if __name__ == "__main__":
    unittest.main()
