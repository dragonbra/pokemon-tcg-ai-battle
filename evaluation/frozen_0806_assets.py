"""Materialize exact decks and fixed frequencies for Frozen-0806."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.cards import load_card_catalog
from evaluation.frozen_0806 import (
    POLICY_0019_SHA256,
    POLICY_0806_SHA256,
    POOL_ID,
    allocate_largest_remainder,
    exact_deck_sha256,
    frozen_deck_directory_name,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / "docs" / "environment-daily_kaggle_top100" / "ranked" / "data"
    / "2026-08-06-top500-exact.json"
)
DEFAULT_TARGET = ROOT / "evaluation" / "arena" / "frozen_pools" / POOL_ID
TAIL_SPECS = (
    (101, "Arboliva ex / Meganium / Teal Mask Ogerpon ex", 2),
    (204, "Mega Starmie ex / Mega Froslass ex", 1),
    (210, "Mega Starmie ex / Mega Froslass ex", 1),
    (425, "Mega Starmie ex / Mega Froslass ex", 1),
    (228, "Mega Starmie ex / Dusknoir", 1),
    (350, "Mega Starmie ex / Dusknoir", 1),
    (282, "Archaludon ex / Cinderace", 1),
    (387, "Archaludon ex / Cinderace", 1),
    (412, "Archaludon ex / Cinderace", 1),
    (457, "Archaludon ex / Cinderace", 1),
    (230, "Dragapult ex / Dusknoir", 2),
    (244, "Erika's Vileplume ex / Cinderace", 1),
    (347, "N's Zoroark ex / Munkidori", 1),
    (433, "N's Zoroark ex / Munkidori", 1),
)
REPRESENTATIVE_NAMES = {
    "Arboliva ex / Meganium / Teal Mask Ogerpon ex": ("Arboliva ex", "Meganium"),
    "Mega Starmie ex / Mega Froslass ex": ("Mega Starmie ex", "Mega Froslass ex"),
    "Mega Starmie ex / Dusknoir": ("Mega Starmie ex", "Dusknoir"),
    "Archaludon ex / Cinderace": ("Archaludon ex", "Cinderace"),
    "Dragapult ex / Dusknoir": ("Dragapult ex", "Dusknoir"),
    "Erika's Vileplume ex / Cinderace": ("Erika's Vileplume ex", "Cinderace"),
    "N's Zoroark ex / Munkidori": ("N's Zoroark ex", "Munkidori"),
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ascii(value: str) -> str:
    return unicodedata.normalize("NFKD", value.replace("’", "'")).encode(
        "ascii", "ignore"
    ).decode("ascii")


def _slug(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", _ascii(value).lower())).strip("_")


def _representative_cards(
    archetype: str, cards: list[int], official_cards: dict[int, dict[str, Any]]
) -> list[int]:
    wanted = REPRESENTATIVE_NAMES.get(archetype)
    if wanted is None:
        wanted = tuple(part.strip() for part in archetype.split("/")[:2])
    result: list[int] = []
    for name in wanted:
        normalized = _ascii(name).casefold()
        match = next(
            (
                card_id for card_id in cards
                if _ascii(official_cards[card_id]["name"]).casefold() == normalized
            ),
            None,
        )
        if match is not None and match not in result:
            result.append(match)
    if not result:
        result = [
            card_id for card_id in dict.fromkeys(cards)
            if "Pokémon" in official_cards[card_id]["stage_or_type"]
        ][:2]
    return result[:2]


def _group_players(players: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for player in players:
        groups[exact_deck_sha256(player["deck"])].append(player)
    return groups


def materialize(*, source_snapshot: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Frozen-0806 assets already exist: {target}")
    payload = json.loads(source_snapshot.read_text(encoding="utf-8"))
    if payload.get("schema") != "pokemon_tcg_top500_exact_v1":
        raise ValueError("unexpected Top 500 snapshot schema")
    players = payload.get("players")
    if not isinstance(players, list) or len(players) != 500:
        raise ValueError("Frozen-0806 source must contain exactly 500 players")
    by_rank = {player["rank"]: player for player in players}
    if set(by_rank) != set(range(1, 501)) or any(len(player.get("deck", [])) != 60 for player in players):
        raise ValueError("Frozen-0806 source ranks or exact decks are incomplete")

    top_players = [player for player in players if player["rank"] <= 100]
    top_groups = _group_players(top_players)
    if len(top_groups) != 41:
        raise ValueError(f"expected 41 Top 100 exact decks, got {len(top_groups)}")
    top_allocation = allocate_largest_remainder(
        [
            (deck_hash, len(group), min(player["rank"] for player in group))
            for deck_hash, group in top_groups.items()
        ],
        population=100,
        total=240,
    )

    all_groups = _group_players(players)
    rows: list[dict[str, Any]] = []
    for deck_hash, group in sorted(
        top_groups.items(), key=lambda item: (min(p["rank"] for p in item[1]), item[0])
    ):
        group = sorted(group, key=lambda player: player["rank"])
        rows.append(
            {
                "archetype": group[0]["archetype"],
                "best_rank": group[0]["rank"],
                "binding_counts": dict(sorted(Counter(p["binding"] for p in group).items())),
                "cards": group[0]["deck"],
                "exact_deck_sha256": deck_hash,
                "games": top_allocation[deck_hash],
                "observed_players": len(group),
                "segment": "top100",
                "selection_rank": None,
                "source_ranks": [player["rank"] for player in group],
            }
        )

    selected_tail_hashes: set[str] = set()
    for selection_rank, corrected_archetype, games in TAIL_SPECS:
        selected = by_rank[selection_rank]
        deck_hash = exact_deck_sha256(selected["deck"])
        if deck_hash in top_groups or deck_hash in selected_tail_hashes:
            raise ValueError(f"tail exact deck is duplicated: rank {selection_rank}")
        selected_tail_hashes.add(deck_hash)
        same_deck_tail = sorted(
            (p for p in all_groups[deck_hash] if p["rank"] > 100),
            key=lambda player: player["rank"],
        )
        rows.append(
            {
                "archetype": corrected_archetype,
                "best_rank": same_deck_tail[0]["rank"],
                "binding_counts": dict(
                    sorted(Counter(p["binding"] for p in same_deck_tail).items())
                ),
                "cards": selected["deck"],
                "exact_deck_sha256": deck_hash,
                "games": games,
                "observed_players": len(same_deck_tail),
                "segment": "potential_101_500",
                "selection_rank": selection_rank,
                "source_ranks": [player["rank"] for player in same_deck_tail],
            }
        )

    if len(rows) != 55 or sum(row["games"] for row in rows) != 256:
        raise ValueError("Frozen-0806 materialized schedule size mismatch")

    for row in rows:
        row["deck_id"] = (
            f"{_slug(row['archetype'])}_{row['exact_deck_sha256'][:12]}"
        )
    frequency_order = sorted(
        rows,
        key=lambda row: (-row["games"], row["best_rank"], row["deck_id"]),
    )
    deck_number_by_id = {
        row["deck_id"]: number
        for number, row in enumerate(frequency_order, start=1)
    }

    official_cards = load_card_catalog(ROOT / "data" / "official" / "EN_Card_Data.csv")
    decks_root = target / "decks"
    decks_root.mkdir(parents=True)
    schedule_entries: list[dict[str, Any]] = []
    for row in rows:
        deck_id = row["deck_id"]
        deck_root = decks_root / frozen_deck_directory_name(
            deck_id, row["archetype"], deck_number_by_id[deck_id]
        )
        deck_root.mkdir()
        (deck_root / "deck.csv").write_text(
            "".join(f"{card_id}\n" for card_id in row["cards"]), encoding="ascii"
        )
        deck_manifest = {
            "schema_version": "evaluation_frozen_0806_deck_v1",
            "pool_id": POOL_ID,
            "deck_id": deck_id,
            "archetype": row["archetype"],
            "exact_deck_sha256": row["exact_deck_sha256"],
            "representative_card_ids": _representative_cards(
                row["archetype"], row["cards"], official_cards
            ),
            "provenance": {
                "source_snapshot": source_snapshot.relative_to(ROOT).as_posix(),
                "segment": row["segment"],
                "source_ranks": row["source_ranks"],
                "selection_rank": row["selection_rank"],
                "binding_counts": row["binding_counts"],
            },
        }
        (deck_root / "manifest.json").write_text(
            _canonical_json(deck_manifest), encoding="utf-8"
        )
        schedule_entries.append(
            {key: value for key, value in row.items() if key != "cards"}
        )

    schedule = {
        "schema_version": "evaluation_frozen_schedule_v1",
        "pool_id": POOL_ID,
        "total_games": 256,
        "allocation": {
            "top100": {
                "games": 240,
                "method": "largest_remainder",
                "population": 100,
                "tie_break": ["fractional_remainder_desc", "best_rank_asc", "exact_deck_sha256_asc"],
            },
            "potential_101_500": {"games": 16, "method": "explicit_fixed_selection"},
        },
        "entries": schedule_entries,
    }
    (target / "schedule.json").write_text(_canonical_json(schedule), encoding="utf-8")
    manifest = {
        "schema_version": "evaluation_frozen_distribution_v1",
        "pool_id": POOL_ID,
        "deck_count": len(rows),
        "total_games": 256,
        "schedule_path": "schedule.json",
        "schedule_sha256": _sha256(target / "schedule.json"),
        "source_snapshot": {
            "path": source_snapshot.relative_to(ROOT).as_posix(),
            "sha256": _sha256(source_snapshot),
            "captured_at_utc": payload["captured_at_utc"],
        },
        "policies": {
            "opponent": {
                "role": "opponent",
                "policy_id": "Policy-0019",
                "asset_id": "0019-0730-epoch13",
                "archive": "archive/pretrained/0019_universal_winner_bc_0730_epoch13",
                "weights_sha256": POLICY_0019_SHA256,
                "deployment_source_id": 0,
                "shared_across_decks": True,
            },
            "main": {
                "role": "main",
                "policy_id": "Policy-0806",
                "asset_id": "0031-friend-0806-epoch11-best-validation-loss",
                "archive": "archive/pretrained/0031_friend_0806_epoch11_best_validation_loss",
                "weights_sha256": POLICY_0806_SHA256,
                "exact_deck_conditioned": True,
                "shared_across_decks": True,
            },
        },
        "usage_contract": {
            "arena_validation": True,
            "ppo_rollout_batch": True,
            "final_frozen_evaluation": True,
            "fixed_integer_counts": True,
            "runtime_resampling": False,
            "reward": "terminal_win_loss_only",
            "official_engine_required": True,
        },
    }
    (target / "manifest.json").write_text(_canonical_json(manifest), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="materialize Frozen-0806 deck distribution")
    parser.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args(argv)
    materialize(source_snapshot=args.source_snapshot.resolve(), target=args.target.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
