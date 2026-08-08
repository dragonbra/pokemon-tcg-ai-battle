"""Strict loader for the immutable Frozen-0806 deck distribution."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from evaluation.cards import load_card_catalog
from evaluation.packages.loader import PackageValidationError, _read_deck


POOL_ID = "0806_kaggle_top100_plus_v1"
POLICY_0019_SHA256 = (
    "da9b13d6f82d19d4521b0bf43369adf5a41e9b4fd752795cc77b5c4ba467e5bb"
)
POLICY_0806_SHA256 = (
    "0ca395a5f08ca21f22417a04736d1bf42274800586729e6cc0917a0c79aa5df8"
)
EXPECTED_TAIL_GAMES = {
    "Arboliva ex / Meganium / Teal Mask Ogerpon ex": 2,
    "Mega Starmie ex / Mega Froslass ex": 3,
    "Mega Starmie ex / Dusknoir": 2,
    "Archaludon ex / Cinderace": 4,
    "Dragapult ex / Dusknoir": 2,
    "Erika's Vileplume ex / Cinderace": 1,
    "N's Zoroark ex / Munkidori": 2,
}

_DECK_SLUG_OVERRIDES = {
    "mega_kangaskhan_ex_crustle_310ede704da1": (
        "wellspring_mask_ogerpon_ex_teal_mask_ogerpon_ex"
    ),
    "mega_kangaskhan_ex_crustle_df6f74437196": "slowking_toolbox",
}


@dataclass(frozen=True)
class Frozen0806Deck:
    deck_id: str
    root: Path
    cards: tuple[int, ...]
    exact_deck_sha256: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class Frozen0806ScheduleEntry:
    deck_id: str
    archetype: str
    segment: str
    games: int
    observed_players: int
    best_rank: int
    selection_rank: int | None
    source_ranks: tuple[int, ...]
    exact_deck_sha256: str
    binding_counts: dict[str, int]


@dataclass(frozen=True)
class Frozen0806Pool:
    pool_id: str
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    policies: dict[str, dict[str, Any]]
    decks: tuple[Frozen0806Deck, ...]
    schedule: tuple[Frozen0806ScheduleEntry, ...]
    total_games: int


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PackageValidationError(f"could not read Frozen-0806 {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PackageValidationError(f"Frozen-0806 {label} must be a JSON object")
    return payload


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PackageValidationError(f"could not hash Frozen-0806 asset {path}: {exc}") from exc


def exact_deck_sha256(cards: Iterable[int]) -> str:
    canonical = ",".join(str(card_id) for card_id in sorted(cards)).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def frozen_deck_number_by_id(
    schedule: Iterable[Frozen0806ScheduleEntry],
) -> dict[str, int]:
    ordered = sorted(
        schedule,
        key=lambda entry: (-entry.games, entry.best_rank, entry.deck_id),
    )
    return {
        entry.deck_id: number for number, entry in enumerate(ordered, start=1)
    }


def frozen_deck_directory_name(
    deck_id: str, archetype: str, number: int
) -> str:
    slug = _DECK_SLUG_OVERRIDES.get(deck_id)
    if slug is None:
        slug, separator, suffix = deck_id.rpartition("_")
        if (
            not separator
            or len(suffix) != 12
            or any(character not in "0123456789abcdef" for character in suffix)
        ):
            slug = "_".join(
                part for part in archetype.lower().replace("'", "").split() if part
            )
            slug = slug.replace("/", "_")
    return f"{number:03d}_{slug}"


def allocate_largest_remainder(
    groups: Iterable[tuple[str, int, int]], *, population: int, total: int
) -> dict[str, int]:
    """Allocate integer slots by count, then remainder, rank, and stable identity."""
    rows = list(groups)
    if population <= 0 or total <= 0:
        raise ValueError("population and total must be positive")
    if not rows or sum(count for _, count, _ in rows) != population:
        raise ValueError("group counts must exactly match population")
    if len({identity for identity, _, _ in rows}) != len(rows):
        raise ValueError("group identities must be unique")
    if any(count <= 0 or best_rank <= 0 for _, count, best_rank in rows):
        raise ValueError("counts and best ranks must be positive")

    result = {
        identity: (count * total) // population for identity, count, _ in rows
    }
    remaining = total - sum(result.values())
    order = sorted(
        rows,
        key=lambda row: (-(row[1] * total % population), row[2], row[0]),
    )
    for identity, _, _ in order[:remaining]:
        result[identity] += 1
    return result


def _validate_policy(
    repo_root: Path, policy: dict[str, Any], *, expected_sha256: str, expected_role: str
) -> None:
    if (
        policy.get("role") != expected_role
        or policy.get("weights_sha256") != expected_sha256
        or not isinstance(policy.get("archive"), str)
        or not isinstance(policy.get("asset_id"), str)
    ):
        raise PackageValidationError(f"Frozen-0806 {expected_role} policy identity mismatch")
    archive = (repo_root / policy["archive"]).resolve()
    if repo_root not in archive.parents or not archive.is_dir():
        raise PackageValidationError(f"Frozen-0806 {expected_role} archive is missing")
    archive_manifest = _read_json(archive / "manifest.json", f"{expected_role} archive manifest")
    if (
        archive_manifest.get("asset_id") != policy["asset_id"]
        or (archive_manifest.get("weights") or {}).get("sha256") != expected_sha256
        or _sha256(archive / "model.pt") != expected_sha256
    ):
        raise PackageValidationError(f"Frozen-0806 {expected_role} archive hash mismatch")


def load_frozen_0806_pool(config_path: Path, evaluation_root: Path) -> Frozen0806Pool:
    """Load and fully validate the fixed 256-game Frozen-0806 schedule."""
    config = _read_json(config_path, "config")
    if set(config) != {"schema_version", "pool_id", "pool_manifest"} or (
        config.get("schema_version") != "evaluation_frozen_pool_config_v1"
        or config.get("pool_id") != POOL_ID
        or not isinstance(config.get("pool_manifest"), str)
    ):
        raise PackageValidationError("Frozen-0806 config identity mismatch")

    repo_root = evaluation_root.parent.resolve()
    manifest_path = (evaluation_root / config["pool_manifest"]).resolve()
    expected_root = (evaluation_root / "arena" / "frozen_pools" / POOL_ID).resolve()
    if manifest_path != expected_root / "manifest.json":
        raise PackageValidationError("Frozen-0806 manifest path mismatch")
    manifest = _read_json(manifest_path, "pool manifest")
    if (
        manifest.get("schema_version") != "evaluation_frozen_distribution_v1"
        or manifest.get("pool_id") != POOL_ID
        or manifest.get("deck_count") != 55
        or manifest.get("total_games") != 256
        or manifest.get("schedule_path") != "schedule.json"
    ):
        raise PackageValidationError("Frozen-0806 pool manifest identity mismatch")

    source = manifest.get("source_snapshot")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise PackageValidationError("Frozen-0806 source snapshot is invalid")
    source_path = (repo_root / source["path"]).resolve()
    if repo_root not in source_path.parents or _sha256(source_path) != source.get("sha256"):
        raise PackageValidationError("Frozen-0806 source snapshot hash mismatch")
    source_payload = _read_json(source_path, "source snapshot")
    source_players = source_payload.get("players")
    if not isinstance(source_players, list) or len(source_players) != 500:
        raise PackageValidationError("Frozen-0806 source snapshot population mismatch")
    source_by_rank = {player.get("rank"): player for player in source_players}
    if set(source_by_rank) != set(range(1, 501)):
        raise PackageValidationError("Frozen-0806 source snapshot ranks mismatch")

    policies = manifest.get("policies")
    if not isinstance(policies, dict) or set(policies) != {"opponent", "main"}:
        raise PackageValidationError("Frozen-0806 policy roles mismatch")
    _validate_policy(
        repo_root, policies["opponent"], expected_sha256=POLICY_0019_SHA256,
        expected_role="opponent",
    )
    _validate_policy(
        repo_root, policies["main"], expected_sha256=POLICY_0806_SHA256,
        expected_role="main",
    )

    official_ids = set(
        load_card_catalog(repo_root / "data" / "official" / "EN_Card_Data.csv")
    )
    decks: list[Frozen0806Deck] = []
    deck_by_id: dict[str, Frozen0806Deck] = {}
    seen_hashes: set[str] = set()
    decks_root = expected_root / "decks"
    if not decks_root.is_dir():
        raise PackageValidationError("Frozen-0806 decks directory is missing")
    for deck_root in sorted(path for path in decks_root.iterdir() if path.is_dir()):
        if set(item.name for item in deck_root.iterdir()) != {"deck.csv", "manifest.json"}:
            raise PackageValidationError(f"Frozen-0806 deck must be lightweight: {deck_root.name}")
        cards = tuple(_read_deck(deck_root / "deck.csv", official_ids))
        deck_manifest = _read_json(deck_root / "manifest.json", f"deck {deck_root.name}")
        exact_hash = exact_deck_sha256(cards)
        deck_id = deck_manifest.get("deck_id")
        if (
            deck_manifest.get("schema_version") != "evaluation_frozen_0806_deck_v1"
            or not isinstance(deck_id, str)
            or not deck_id
            or deck_manifest.get("exact_deck_sha256") != exact_hash
            or deck_manifest.get("pool_id") != POOL_ID
            or not isinstance(deck_manifest.get("representative_card_ids"), list)
            or not set(deck_manifest["representative_card_ids"]).issubset(cards)
        ):
            raise PackageValidationError(f"Frozen-0806 deck identity mismatch: {deck_root.name}")
        if exact_hash in seen_hashes:
            raise PackageValidationError(f"duplicate Frozen-0806 exact deck: {deck_root.name}")
        seen_hashes.add(exact_hash)
        if deck_id in deck_by_id:
            raise PackageValidationError(f"duplicate Frozen-0806 deck ID: {deck_id}")
        deck = Frozen0806Deck(deck_id, deck_root, cards, exact_hash, deck_manifest)
        decks.append(deck)
        deck_by_id[deck.deck_id] = deck
    if len(decks) != manifest["deck_count"]:
        raise PackageValidationError("Frozen-0806 deck count mismatch")

    schedule_path = expected_root / "schedule.json"
    if _sha256(schedule_path) != manifest.get("schedule_sha256"):
        raise PackageValidationError("Frozen-0806 schedule hash mismatch")
    schedule_payload = _read_json(schedule_path, "schedule")
    raw_entries = schedule_payload.get("entries")
    if (
        schedule_payload.get("schema_version") != "evaluation_frozen_schedule_v1"
        or schedule_payload.get("pool_id") != POOL_ID
        or schedule_payload.get("total_games") != 256
        or not isinstance(raw_entries, list)
    ):
        raise PackageValidationError("Frozen-0806 schedule identity mismatch")
    schedule: list[Frozen0806ScheduleEntry] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise PackageValidationError(f"Frozen-0806 schedule row {index} is invalid")
        try:
            entry = Frozen0806ScheduleEntry(
                deck_id=raw["deck_id"], archetype=raw["archetype"],
                segment=raw["segment"], games=raw["games"],
                observed_players=raw["observed_players"], best_rank=raw["best_rank"],
                selection_rank=raw["selection_rank"], source_ranks=tuple(raw["source_ranks"]),
                exact_deck_sha256=raw["exact_deck_sha256"],
                binding_counts=dict(raw["binding_counts"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PackageValidationError(f"Frozen-0806 schedule row {index} fields invalid") from exc
        deck = deck_by_id.get(entry.deck_id)
        if (
            deck is None
            or deck.exact_deck_sha256 != entry.exact_deck_sha256
            or entry.segment not in {"top100", "potential_101_500"}
            or type(entry.games) is not int or entry.games <= 0
            or type(entry.observed_players) is not int or entry.observed_players <= 0
            or not entry.source_ranks or min(entry.source_ranks) != entry.best_rank
            or not all(type(rank) is int and 1 <= rank <= 500 for rank in entry.source_ranks)
            or tuple(sorted(set(entry.source_ranks))) != entry.source_ranks
            or not isinstance(entry.archetype, str) or not entry.archetype
            or not entry.binding_counts
            or not all(
                isinstance(binding, str) and binding and type(count) is int and count > 0
                for binding, count in entry.binding_counts.items()
            )
            or sum(entry.binding_counts.values()) != entry.observed_players
        ):
            raise PackageValidationError(f"Frozen-0806 schedule row {index} contract mismatch")
        source_rows = [source_by_rank[rank] for rank in entry.source_ranks]
        if (
            any(exact_deck_sha256(player.get("deck", [])) != entry.exact_deck_sha256 for player in source_rows)
            or Counter(player.get("binding") for player in source_rows) != Counter(entry.binding_counts)
        ):
            raise PackageValidationError(f"Frozen-0806 schedule row {index} source mismatch")
        if entry.segment == "top100" and (
            entry.selection_rank is not None or max(entry.source_ranks) > 100
        ):
            raise PackageValidationError(f"Frozen-0806 Top 100 row {index} provenance mismatch")
        if entry.segment == "potential_101_500" and (
            entry.selection_rank not in entry.source_ranks or min(entry.source_ranks) <= 100
        ):
            raise PackageValidationError(f"Frozen-0806 tail row {index} provenance mismatch")
        schedule.append(entry)
    if len(schedule) != len(decks) or {item.deck_id for item in schedule} != set(deck_by_id):
        raise PackageValidationError("Frozen-0806 schedule/deck identity set mismatch")
    if sum(item.games for item in schedule) != 256:
        raise PackageValidationError("Frozen-0806 schedule total mismatch")
    if sum(item.games for item in schedule if item.segment == "top100") != 240:
        raise PackageValidationError("Frozen-0806 Top 100 allocation mismatch")
    if sum(item.games for item in schedule if item.segment == "potential_101_500") != 16:
        raise PackageValidationError("Frozen-0806 tail allocation mismatch")
    top = [item for item in schedule if item.segment == "top100"]
    if len(top) != 41 or sum(item.observed_players for item in top) != 100:
        raise PackageValidationError("Frozen-0806 Top 100 population mismatch")
    expected_top = allocate_largest_remainder(
        [
            (item.exact_deck_sha256, item.observed_players, item.best_rank)
            for item in top
        ],
        population=100,
        total=240,
    )
    if {item.exact_deck_sha256: item.games for item in top} != expected_top:
        raise PackageValidationError("Frozen-0806 Top 100 largest-remainder mismatch")
    tail_games: dict[str, int] = {}
    for item in schedule:
        if item.segment == "potential_101_500":
            tail_games[item.archetype] = tail_games.get(item.archetype, 0) + item.games
    if tail_games != EXPECTED_TAIL_GAMES:
        raise PackageValidationError("Frozen-0806 potential-deck allocation mismatch")

    number_by_id = frozen_deck_number_by_id(schedule)
    for deck in decks:
        schedule_entry = next(item for item in schedule if item.deck_id == deck.deck_id)
        expected_directory = frozen_deck_directory_name(
            deck.deck_id,
            schedule_entry.archetype,
            number_by_id[deck.deck_id],
        )
        if deck.root.name != expected_directory:
            raise PackageValidationError(
                "Frozen-0806 deck directory identity mismatch: "
                f"{deck.root.name} != {expected_directory}"
            )

    return Frozen0806Pool(
        pool_id=POOL_ID, root=expected_root, manifest=manifest,
        manifest_sha256=_sha256(manifest_path), policies=policies,
        decks=tuple(decks), schedule=tuple(schedule), total_games=256,
    )


__all__ = [
    "Frozen0806Deck", "Frozen0806Pool", "Frozen0806ScheduleEntry", "POOL_ID",
    "POLICY_0019_SHA256", "POLICY_0806_SHA256", "allocate_largest_remainder",
    "exact_deck_sha256", "frozen_deck_directory_name",
    "frozen_deck_number_by_id", "load_frozen_0806_pool",
]
