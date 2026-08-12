"""Materialize the immutable 2026-08-12 Top100/500 + Limitless Frozen pool."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
BASE_POOL = ROOT / "evaluation/arena/frozen_pools/0806_kaggle_top100_plus_v1"
POOL_ID = "0812_top100_top500_stadiums_mill_plus_limitless_dragapult_v3"
POOL_ROOT = ROOT / "evaluation/arena/frozen_pools" / POOL_ID
SNAPSHOT_PATH = (
    ROOT / "docs/environment-daily_kaggle_top100/ranked/data/2026-08-12-top100-exact.json"
)
REPORT_PATH = Path(
    r"D:\Users\admin\Documents\xwechat_files\wxid_ljstzlmxrfeu12_640d"
    r"\msg\file\2026-08\2026-08-12.html"
)
TOP500_REPORT_PATH = Path(r"D:\Users\admin\Desktop\2026-08-09-top500.html")
TOP500_SNAPSHOT_PATH = (
    ROOT / "docs/environment-daily_kaggle_top100/ranked/data/2026-08-09-top500-exact.json"
)
CARD_IMPL_PATH = ROOT / "engine/source/ptcgProgram 22/CardImpl.h"
TOP500_SELECTIONS = (
    (159, "Mega Kangaskhan ex / Crustle", [756, 96], "dragapult_counter", "top500_limitless_counter_selection"),
    (193, "Team Rocket's Mewtwo ex / Spidops", [431, 401], "lucario_counter", "top500_limitless_counter_selection"),
    (222, "Cynthia's Garchomp ex / Roserade", [381, 342], "dragapult_and_lucario_counter", "top500_limitless_counter_selection"),
    (240, "Dragapult ex / Blaziken ex", [121, 326], "alakazam_counter", "top500_limitless_counter_selection"),
    (294, "Mega Starmie ex / Dusknoir", [1031, 861], "lucario_counter", "top500_limitless_counter_selection"),
    (170, "Mega Lucario ex / Solrock", [675, 1252], "stadium_coverage_gravity_mountain", "top500_stadium_coverage"),
    (399, "Mega Lopunny ex / Mega Froslass ex", [849, 1267], "stadium_coverage_lumiose_city", "top500_stadium_coverage"),
    (360, "Crustle / Great Tusk", [345, 58, 1247], "great_tusk_mill_coverage", "top500_mill_coverage"),
)

LIMITLESS_SOURCE = "https://labs.limitlesstcg.com/0063/player/1074/decklist"
LIMITLESS_CARDS = {
    119: 4, 120: 4, 121: 3, 131: 2, 132: 2, 133: 1,
    1071: 2, 112: 2, 140: 1, 235: 1,
    1227: 4, 1182: 3, 1198: 2, 1240: 1, 1213: 1, 1231: 1,
    1121: 4, 1152: 4, 1086: 4, 1097: 2, 1080: 1,
    1260: 2, 1246: 1, 5: 3, 2: 3, 7: 2,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = (item for item in root.rglob("*") if item.is_file())
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def exact_hash(cards: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(card) for card in sorted(cards)).encode("ascii")
    ).hexdigest()


def expanded(counts: dict[int, int]) -> list[int]:
    return [card for card, count in sorted(counts.items()) for _ in range(count)]


def stadium_cards() -> dict[int, str]:
    source = CARD_IMPL_PATH.read_text(encoding="utf-8")
    starts = list(re.finditer(r"CreateCard\((\d+),.*?,\s*Stadium\s*,", source))
    result = {}
    for match in starts:
        end = source.find("CreateCard(", match.end())
        block = source[match.start():end if end >= 0 else len(source)]
        name = re.search(r'\.nameEn\(u8"([^"]+)"\)', block)
        result[int(match.group(1))] = name.group(1) if name else "unknown"
    # CardImpl.h contains this apostrophe in the legacy source encoding.
    result[1253] = "N's Castle"
    return result


def parse_report(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    audit_match = re.search(
        r'<script type="application/json" id="snapshot-audit">(.*?)</script>',
        source,
        re.DOTALL,
    )
    if audit_match is None:
        raise ValueError("Top100 report has no snapshot-audit payload")
    audit = json.loads(html.unescape(audit_match.group(1)))
    required = (
        "all_episode_times_at_or_before_capture",
        "all_submission_matches_unique",
        "all_leaderboard_scores_match",
        "all_decks_exactly_60",
    )
    if audit.get("audited_players") != 100 or not all(audit.get(key) for key in required):
        raise ValueError("Top100 report failed its embedded audit")

    starts = list(re.finditer(
        r'<details class="person-card"[^>]*id="player-(\d+)"[^>]*'
        r'data-archetype="([^"]+)"',
        source,
    ))
    if len(starts) not in {99, 100}:
        raise ValueError(f"expected 99 or 100 player details, found {len(starts)}")
    players_by_rank = {}
    for index, match in enumerate(starts):
        end = source.index("</details>", match.start()) + len("</details>")
        block = source[match.start():end]
        counts = {
            int(card): int(count)
            for card, count in re.findall(
                r'<small>Card ID\s+(\d+)\D+(\d+)\D*</small>', block
            )
        }
        cards = expanded(counts)
        rank = int(match.group(1))
        if len(cards) != 60:
            raise ValueError(f"invalid exact deck at rank {rank}: {len(cards)} cards")
        players_by_rank[rank] = {
            "rank": rank,
            "archetype": html.unescape(match.group(2)),
            "deck": cards,
            "exact_deck_sha256": exact_hash(cards),
        }
    selected = {int(row["rank"]): row for row in audit.get("selected", [])}
    if set(selected) != set(range(1, 101)):
        raise ValueError("embedded audit does not contain ranks 1..100")
    for rank, player in players_by_rank.items():
        if player["exact_deck_sha256"] != selected[rank].get("deck_sha256"):
            raise ValueError(f"detail/audit deck hash mismatch at rank {rank}")
    for rank in sorted(set(range(1, 101)) - set(players_by_rank)):
        wanted_hash = selected[rank].get("deck_sha256")
        matches = []
        for manifest_path in (BASE_POOL / "decks").glob("*/manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("exact_deck_sha256") == wanted_hash:
                cards = [
                    int(value) for value in
                    (manifest_path.parent / "deck.csv").read_text(encoding="utf-8").split()
                ]
                matches.append((manifest, cards))
        if len(matches) != 1:
            raise ValueError(f"could not recover omitted rank {rank} from its audited hash")
        manifest, cards = matches[0]
        players_by_rank[rank] = {
            "rank": rank,
            "archetype": manifest["archetype"],
            "deck": cards,
            "exact_deck_sha256": wanted_hash,
            "detail_recovered_from_frozen_exact_hash": True,
        }
    players = [players_by_rank[rank] for rank in range(1, 101)]
    return {
        "schema_version": "daily_top100_exact_snapshot_v1",
        "date": "2026-08-12",
        "source_report": {"path": str(path), "sha256": sha256(path)},
        "embedded_audit": audit,
        "players": players,
    }


def parse_top500_report(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    starts = list(re.finditer(
        r'<details class="player-detail" id="player-(\d+)"', source
    ))
    if len(starts) != 500:
        raise ValueError(f"expected 500 player details, found {len(starts)}")
    players = []
    for match in starts:
        rank = int(match.group(1))
        end = source.index("</details>", match.start()) + len("</details>")
        block = source[match.start():end]
        summary = block[:block.index("</summary>")]
        labels = re.findall(r'</span></span><span>([^<>]+)</span>', summary)
        if not labels:
            raise ValueError(f"Top500 rank {rank} has no archetype label")
        counts = [
            (int(card), int(count))
            for count, card in re.findall(
                r'class="count">\D*(\d+)</b>.*?<small>ID\s+(\d+)</small>',
                block,
                re.DOTALL,
            )
        ]
        cards = [card for card, count in counts for _ in range(count)]
        if len(cards) != 60:
            raise ValueError(f"invalid Top500 exact deck at rank {rank}")
        players.append({
            "rank": rank,
            "archetype": html.unescape(labels[-1]),
            "deck": cards,
            "exact_deck_sha256": exact_hash(cards),
        })
    return {
        "schema_version": "daily_top500_exact_snapshot_v1",
        "date": "2026-08-09",
        "source_report": {"path": str(path), "sha256": sha256(path)},
        "players": players,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_deck(root: Path, cards: list[int], manifest: dict) -> None:
    root.mkdir(parents=True)
    (root / "deck.csv").write_text(
        "".join(f"{card}\n" for card in cards), encoding="utf-8"
    )
    _write_json(root / "manifest.json", manifest)


def materialize(report_path: Path, top500_report_path: Path, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(output_root)
    snapshot = parse_report(report_path)
    _write_json(SNAPSHOT_PATH, snapshot)
    top500_snapshot = parse_top500_report(top500_report_path)
    _write_json(TOP500_SNAPSHOT_PATH, top500_snapshot)

    old_manifest = json.loads((BASE_POOL / "manifest.json").read_text(encoding="utf-8"))
    old_schedule = json.loads((BASE_POOL / "schedule.json").read_text(encoding="utf-8"))
    old_hashes = {row["exact_deck_sha256"] for row in old_schedule["entries"]}
    new_by_hash: dict[str, list[dict]] = defaultdict(list)
    for player in snapshot["players"]:
        if player["exact_deck_sha256"] not in old_hashes:
            new_by_hash[player["exact_deck_sha256"]].append(player)
    old_archetypes = {row["archetype"] for row in old_schedule["entries"]}
    new_archetypes = sorted({
        player["archetype"] for player in snapshot["players"]
        if player["archetype"] not in old_archetypes
    })
    if new_archetypes != ["Chandelure / Sylveon"]:
        raise ValueError(f"unexpected new archetypes: {new_archetypes}")
    chandelure = next(
        player for player in snapshot["players"]
        if player["archetype"] == "Chandelure / Sylveon"
    )
    limitless_cards = expanded(LIMITLESS_CARDS)
    if len(limitless_cards) != 60:
        raise ValueError("Limitless self-destruct Dragapult deck is not 60 cards")

    shutil.copytree(BASE_POOL, output_root)
    for deck_root in sorted((output_root / "decks").iterdir()):
        manifest_path = deck_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["pool_id"] = POOL_ID
        manifest["schema_version"] = "evaluation_frozen_0812_deck_v1"
        _write_json(manifest_path, manifest)

    chandelure_id = f"chandelure_sylveon_{chandelure['exact_deck_sha256'][:12]}"
    _write_deck(
        output_root / "decks/056_chandelure_sylveon",
        chandelure["deck"],
        {
            "schema_version": "evaluation_frozen_0812_deck_v1",
            "pool_id": POOL_ID,
            "deck_id": chandelure_id,
            "archetype": chandelure["archetype"],
            "exact_deck_sha256": chandelure["exact_deck_sha256"],
            "representative_card_ids": [164, 145],
            "provenance": {
                "source_snapshot": SNAPSHOT_PATH.relative_to(ROOT).as_posix(),
                "source_ranks": [chandelure["rank"]],
                "segment": "top100_new_archetype",
            },
        },
    )

    selected_top500 = []
    for offset, (rank, archetype, representatives, purpose, segment) in enumerate(
        TOP500_SELECTIONS, start=58
    ):
        player = top500_snapshot["players"][rank - 1]
        if player["rank"] != rank or player["archetype"] != archetype:
            raise ValueError(f"Top500 selection identity changed at rank {rank}")
        if player["exact_deck_sha256"] in old_hashes:
            raise ValueError(f"Top500 selection rank {rank} already exists in base pool")
        slug = re.sub(r"[^a-z0-9]+", "_", archetype.lower()).strip("_")
        deck_id = f"{slug}_{player['exact_deck_sha256'][:12]}"
        _write_deck(
            output_root / "decks" / f"{offset:03d}_{slug}",
            player["deck"],
            {
                "schema_version": "evaluation_frozen_0812_deck_v1",
                "pool_id": POOL_ID,
                "deck_id": deck_id,
                "archetype": archetype,
                "exact_deck_sha256": player["exact_deck_sha256"],
                "representative_card_ids": representatives,
                "provenance": {
                    "source_snapshot": TOP500_SNAPSHOT_PATH.relative_to(ROOT).as_posix(),
                    "source_ranks": [rank],
                    "segment": segment,
                    "selection_purpose": purpose,
                },
            },
        )
        selected_top500.append({
            "rank": rank, "archetype": archetype, "deck_id": deck_id,
            "exact_deck_sha256": player["exact_deck_sha256"], "purpose": purpose,
            "segment": segment,
        })
    limitless_hash = exact_hash(limitless_cards)
    limitless_id = f"dragapult_ex_dusknoir_self_destruct_{limitless_hash[:12]}"
    _write_deck(
        output_root / "decks/057_dragapult_ex_dusknoir_self_destruct",
        limitless_cards,
        {
            "schema_version": "evaluation_frozen_0812_deck_v1",
            "pool_id": POOL_ID,
            "deck_id": limitless_id,
            "archetype": "Dragapult ex / Dusknoir (Self-Destruct)",
            "exact_deck_sha256": limitless_hash,
            "representative_card_ids": [121, 133],
            "provenance": {
                "source_url": LIMITLESS_SOURCE,
                "event": "Regional Los Angeles 0063",
                "player": "Nicholas Moffitt",
                "placement": 3,
                "record": "13-3",
                "segment": "limitless_representative",
            },
        },
    )

    entries = old_schedule["entries"]
    for row in entries:
        if row["deck_id"] == "marnie_s_grimmsnarl_ex_froslass_c20a8a46f5c6":
            row["games"] -= 9
        if row["deck_id"] == "dragapult_ex_dusknoir_b85bae9f3a21":
            row["games"] -= 1
    entries.extend([
        {
            "archetype": chandelure["archetype"], "best_rank": chandelure["rank"],
            "binding_counts": {"provided_audited_top100": 1},
            "deck_id": chandelure_id,
            "exact_deck_sha256": chandelure["exact_deck_sha256"],
            "games": 1, "observed_players": 1, "segment": "top100_new_archetype",
            "selection_rank": chandelure["rank"], "source_ranks": [chandelure["rank"]],
        },
        {
            "archetype": "Dragapult ex / Dusknoir (Self-Destruct)", "best_rank": 3,
            "binding_counts": {"limitless_event_result": 1}, "deck_id": limitless_id,
            "exact_deck_sha256": limitless_hash, "games": 1, "observed_players": 1,
            "segment": "limitless_representative", "selection_rank": 3,
            "source_ranks": [3],
        },
    ])
    entries.extend({
        "archetype": item["archetype"], "best_rank": item["rank"],
        "binding_counts": {"provided_audited_top500": 1},
        "deck_id": item["deck_id"],
        "exact_deck_sha256": item["exact_deck_sha256"],
        "games": 1, "observed_players": 1,
        "segment": item["segment"],
        "selection_rank": item["rank"], "source_ranks": [item["rank"]],
    } for item in selected_top500)
    old_schedule.update({
        "pool_id": POOL_ID,
        "allocation": {
            "base_0806_legacy": {"games": 246, "method": "immutable_slot_preserving"},
            "top100_new_archetype": {"games": 1, "method": "archetype_coverage"},
            "limitless_representative": {"games": 1, "method": "audited_event_representative"},
            "top500_limitless_counter_selection": {"games": 5, "method": "ranked_counter_coverage"},
            "top500_stadium_coverage": {"games": 2, "method": "exact_top500_stadium_gap_coverage"},
            "top500_mill_coverage": {"games": 1, "method": "exact_top500_archetype_coverage"},
        },
    })
    if len(entries) != 65 or sum(row["games"] for row in entries) != 256:
        raise ValueError("new Frozen pool must contain 65 decks and 256 slots")
    _write_json(output_root / "schedule.json", old_schedule)

    diff_rows = []
    for deck_hash, players in sorted(new_by_hash.items(), key=lambda item: min(p["rank"] for p in item[1])):
        diff_rows.append({
            "exact_deck_sha256": deck_hash,
            "archetype": players[0]["archetype"],
            "ranks": [player["rank"] for player in players],
            "selected_for_pool": players[0]["archetype"] in new_archetypes,
        })
    audit = {
        "schema_version": "frozen_0812_top100_diff_audit_v1",
        "interpretation": "archetype coverage; exact-list differences retained for audit",
        "source_report_sha256": snapshot["source_report"]["sha256"],
        "top100_players": 100,
        "top100_unique_exact_decks": len({p["exact_deck_sha256"] for p in snapshot["players"]}),
        "exact_decks_absent_from_0806_pool": len(diff_rows),
        "new_archetypes_absent_from_0806_pool": new_archetypes,
        "exact_deck_differences": diff_rows,
        "selected_additions": [chandelure_id, limitless_id]
        + [item["deck_id"] for item in selected_top500],
        "top500_source_report_sha256": top500_snapshot["source_report"]["sha256"],
        "top500_counter_selections": selected_top500,
        "limitless_matchup_evidence": {
            "dragapult_dusknoir_pbl": "https://play.limitlesstcg.com/decks/dragapult-dusknoir/matchups/?format=standard&rotation=2026&set=PBL",
            "mega_lucario_cri": "https://play.limitlesstcg.com/decks/mega-lucario-ex/matchups?format=standard&rotation=2026&set=CRI",
            "alakazam_cri": "https://play.limitlesstcg.com/decks/alakazam-meg/matchups?format=standard&rotation=2026&set=CRI"
        },
        "resulting_deck_count": 65,
        "resulting_schedule_games": 256,
    }
    _write_json(output_root / "top100_diff_audit.json", audit)

    stadiums = stadium_cards()
    top500_stadium_decks = Counter()
    top500_stadium_copies = Counter()
    for player in top500_snapshot["players"]:
        for card_id in set(player["deck"]):
            if card_id in stadiums:
                top500_stadium_decks[card_id] += 1
        for card_id in player["deck"]:
            if card_id in stadiums:
                top500_stadium_copies[card_id] += 1
    base_covered = {
        int(value)
        for deck_path in (output_root / "decks").glob("*/deck.csv")
        if int(deck_path.parent.name.split("_", 1)[0]) <= 62
        for value in deck_path.read_text(encoding="utf-8").split()
        if int(value) in stadiums
    }
    final_covered = {
        int(value)
        for deck_path in (output_root / "decks").glob("*/deck.csv")
        for value in deck_path.read_text(encoding="utf-8").split()
        if int(value) in stadiums
    }
    missing_before = sorted(set(top500_stadium_decks) - base_covered)
    missing_after = sorted(set(top500_stadium_decks) - final_covered)
    if missing_before != [1252, 1267] or missing_after:
        raise ValueError(
            f"unexpected Top500 stadium coverage: before={missing_before}, after={missing_after}"
        )
    _write_json(output_root / "top500_stadium_coverage_audit.json", {
        "schema_version": "top500_stadium_coverage_audit_v1",
        "source_snapshot": TOP500_SNAPSHOT_PATH.relative_to(ROOT).as_posix(),
        "top500_stadiums": [{
            "card_id": card_id,
            "name": stadiums[card_id],
            "deck_count": top500_stadium_decks[card_id],
            "copy_count": top500_stadium_copies[card_id],
            "covered_before_v3": card_id in base_covered,
            "covered_after_v3": card_id in final_covered,
        } for card_id in sorted(top500_stadium_decks)],
        "missing_before_v3": missing_before,
        "missing_after_v3": missing_after,
        "selected_exact_top500_ranks": [170, 399],
        "great_tusk_mill_rank": 360,
    })

    old_manifest.update({
        "pool_id": POOL_ID,
        "deck_count": 65,
        "schedule_sha256": sha256(output_root / "schedule.json"),
        "source_snapshot": {
            "path": SNAPSHOT_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256(SNAPSHOT_PATH),
        },
        "derived_from": {
            "pool_id": "0806_kaggle_top100_plus_v1",
            "manifest_sha256": sha256(BASE_POOL / "manifest.json"),
        },
        "additions": {
            "top100_new_archetype": chandelure_id,
            "limitless_self_destruct_dragapult": limitless_id,
            "top500_limitless_counter_selections": [
                item["deck_id"] for item in selected_top500
                if item["segment"] == "top500_limitless_counter_selection"
            ],
            "top500_stadium_coverage_selections": [
                item["deck_id"] for item in selected_top500
                if item["segment"] == "top500_stadium_coverage"
            ],
            "top500_mill_coverage_selections": [
                item["deck_id"] for item in selected_top500
                if item["segment"] == "top500_mill_coverage"
            ],
        },
    })
    for role, runtime in (old_manifest.get("policy_runtimes") or {}).items():
        runtime["tree_sha256"] = tree_sha256(output_root / runtime["path"])
    _write_json(output_root / "manifest.json", old_manifest)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--top500-report", type=Path, default=TOP500_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=POOL_ROOT)
    args = parser.parse_args()
    print(json.dumps(
        materialize(args.report, args.top500_report, args.output),
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
