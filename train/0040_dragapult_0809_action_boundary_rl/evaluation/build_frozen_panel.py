"""Write the immutable 0038 2,048-game Frozen evaluation manifest."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ..league import load_frozen_catalog
from .frozen_panel import build_panel

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experiments/0040_dragapult_0809_action_boundary_rl/frozen_panel_2048_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    catalog = load_frozen_catalog()
    rows = build_panel([(item.deck_id, item.display_name) for item in catalog])
    games = [asdict(row) for row in rows]
    canonical = json.dumps(games, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "schema_version": "0038_frozen_panel_manifest_v1",
        "frozen_panel_version": rows[0].frozen_panel_version,
        "games": len(rows), "shards": 8, "games_per_shard": 256,
        "unique_seeds": len({row.seed for row in rows}),
        "first_player_games": sum(row.focal_first for row in rows),
        "second_player_games": sum(not row.focal_first for row in rows),
        "game_list_sha256": hashlib.sha256(canonical).hexdigest(),
        "entries": games,
        "interpretation": "Designed for paired ~2pp direction/stability; tiny differences are not automatically significant.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
