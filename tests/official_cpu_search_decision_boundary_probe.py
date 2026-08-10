#!/usr/bin/env python3
"""Build and run the official CPU Search decision-boundary probe."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_suffix(".cpp")
OFFICIAL_SOURCE = ROOT / "engine/source/ptcgProgram 22"
BUILD_DIR = ROOT / ".tmp/official_cpu_search_decision_boundary"
BINARY = BUILD_DIR / "official_cpu_search_decision_boundary_probe"


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "g++",
            "-std=c++20",
            "-O1",
            "-g",
            "-pthread",
            "-I",
            str(OFFICIAL_SOURCE),
            str(SOURCE),
            "-o",
            str(BINARY),
        ],
        cwd=ROOT,
        check=True,
    )
    return subprocess.run([str(BINARY)], cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
