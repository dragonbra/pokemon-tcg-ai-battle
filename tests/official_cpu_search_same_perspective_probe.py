#!/usr/bin/env python3
"""Build and run the Phase 3 probe against the unmodified official CPU engine."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_suffix(".cpp")
OFFICIAL_SOURCE = ROOT / "engine/source/ptcgProgram 22"
BUILD_DIR = ROOT / ".tmp/official_cpu_search_same_perspective"
BINARY = BUILD_DIR / "official_cpu_search_same_perspective_probe"
OUTPUT = BUILD_DIR / "probe_output.jsonl"


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
    completed = subprocess.run(
        [str(BINARY)], cwd=ROOT, text=True, capture_output=True, check=False
    )
    OUTPUT.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=__import__("sys").stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
