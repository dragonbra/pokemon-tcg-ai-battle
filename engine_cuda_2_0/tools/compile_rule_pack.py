from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.schema import RulePack  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or compile a PTCG CUDA numeric rule pack.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pack = RulePack.load(args.input)
    report = pack.report()
    if args.output is not None:
        if args.check:
            raise SystemExit("--check and --output are mutually exclusive")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(pack.to_bytes())
        report["output"] = str(args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
