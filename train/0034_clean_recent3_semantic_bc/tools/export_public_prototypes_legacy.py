"""CLI for exporting the committed official public prototype sidecar."""
from __future__ import annotations

import argparse
from pathlib import Path

from ..features.prototypes import export_public_runtime_prototypes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = export_public_runtime_prototypes(args.output)
    print(
        f"wrote {len(payload['cards'])} cards and {len(payload['attacks'])} attacks "
        f"to {args.output} ({payload['content_sha256']})"
    )


if __name__ == "__main__":
    main()
