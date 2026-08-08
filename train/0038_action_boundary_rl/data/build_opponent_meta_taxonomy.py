"""Materialize the audited opponent catalog as a versioned training-label taxonomy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experiments/0038_action_boundary_rl/opponent_meta_taxonomy_v1.json"
VERSION = "0038_opponent_meta_taxonomy_v1"


def taxonomy() -> dict[str, object]:
    from ..league import load_frozen_catalog
    opponents = load_frozen_catalog()
    mapping = {item.deck_id: item.display_name for item in opponents}
    classes = ["unknown", "ambiguous", *sorted(set(mapping.values()) - {"unknown", "ambiguous"})]
    return {
        "opponent_meta_taxonomy_version": VERSION,
        "source_catalog": "train/0038_action_boundary_rl/league/frozen_catalog.json",
        "classes": classes,
        "deck_id_to_archetype": mapping,
        "inference_contract": "label_only; true deck ID is forbidden from actor inputs",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = json.dumps(taxonomy(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
