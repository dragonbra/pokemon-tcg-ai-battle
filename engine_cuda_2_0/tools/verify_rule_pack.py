from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CUDA_ENGINE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CUDA_ENGINE_ROOT.parent
sys.path.insert(0, str(CUDA_ENGINE_ROOT / "python"))

from ptcg_cuda_engine.official_manifest import verify_rule_pack_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a private typed-IR extraction and deterministic binary rule pack."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--card-data", type=Path, nargs="*")
    parser.add_argument(
        "--skip-docker-check",
        action="store_true",
        help="skip local Docker image ID verification when inspecting a copied manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_rule_pack_manifest(
        args.manifest,
        workspace=WORKSPACE_ROOT,
        source=args.source.resolve() if args.source else None,
        oracle=args.oracle.resolve() if args.oracle else None,
        card_data=[path.resolve() for path in args.card_data] if args.card_data else None,
        verify_docker=not args.skip_docker_check,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "manifest": str(Path(args.manifest).resolve()),
                "output_sha256": result["manifest"]["output_sha256"],
                "rule_pack": result["rule_pack"],
                "source_git_commit": result["manifest"]["source_git_commit"],
                "oracle_sha256": result["manifest"]["official_oracle"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
