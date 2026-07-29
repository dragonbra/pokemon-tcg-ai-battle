from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .fetch import FetchResult, fetch_text
from .kaggle import parse_kaggle_archetypes
from .labs import parse_deck_meta_payload, parse_decklist_payload, parse_standings_payload
from .models import Match, MatchupCell
from .parsers import (
    parse_deck_distribution,
    parse_filter_audit,
    parse_labs_event_id,
    parse_pairings_payload,
    parse_tournament_index,
    validate_filter_query,
)
from .stats import (
    aggregate_matchups,
    evidence_grade,
    hhi,
    jensen_shannon_divergence,
    wilson_interval,
)
from .render import render_report


ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / ".tmp" / "environment_limitless" / "cache"
SNAPSHOT_PATH = Path(__file__).with_name("snapshot.json")
KAGGLE_PATH = ROOT / "docs" / "environment-daily_kaggle_top100" / "daily" / "2026-07-30.html"
FILTER_QUERY = "time=all&type=all&format=TEF-POR&region=all&division=all"
DECKS_URL = f"https://limitlesstcg.com/decks?{FILTER_QUERY}&show=100"
VARIANTS_URL = f"https://limitlesstcg.com/decks?{FILTER_QUERY}&variants=on&show=100"
TOURNAMENTS_URL = f"https://limitlesstcg.com/tournaments?{FILTER_QUERY}"
COMPARISON_MAPPING_RULES = (
    ("dragapult", "Dragapult"),
    ("marnie's grimmsnarl", "Marnie's Grimmsnarl"),
    ("alakazam", "Alakazam"),
    ("mega kangaskhan", "Mega Kangaskhan"),
    ("rocket's mewtwo", "Rocket's Mewtwo"),
    ("festival lead", "Festival Lead"),
    ("cynthia's garchomp", "Cynthia's Garchomp"),
    ("mega lopunny", "Mega Lopunny"),
    ("crustle", "Crustle"),
    ("mega lucario", "Mega Lucario"),
    ("archaludon", "Archaludon"),
)


def _fetch_many(urls: list[str], refresh: bool) -> dict[str, FetchResult]:
    results: dict[str, FetchResult] = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(fetch_text, url, CACHE_DIR, refresh): url for url in urls
        }
        for future in as_completed(futures):
            url = futures[future]
            results[url] = future.result()
    return results


def _source(result: FetchResult, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "url": result.url,
        "final_url": result.final_url,
        "fetched_at": result.fetched_at,
        "sha256": result.sha256,
    }


def _aggregate_deck_meta(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = totals.setdefault(
            row["deck_id"],
            {
                "deck_id": row["deck_id"],
                "name": row["name"],
                "primary_id": row["primary_id"],
                "primary_name": row["primary_name"],
                "icons": row["icons"],
                "players": 0,
                "day2s": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "events": 0,
            },
        )
        if (item["name"], item["primary_id"]) != (row["name"], row["primary_id"]):
            raise ValueError(f"deck identity changed across events: {row['deck_id']}")
        for key in ("players", "day2s", "wins", "losses", "ties"):
            item[key] += row[key]
        item["events"] += 1
    for item in totals.values():
        item["matches"] = item["wins"] + item["losses"] + item["ties"]
        item["effective_win_rate"] = (
            (item["wins"] + 0.5 * item["ties"]) / item["matches"]
            if item["matches"]
            else None
        )
        item["day2_rate"] = item["day2s"] / item["players"] if item["players"] else None
    return sorted(totals.values(), key=lambda item: (-item["players"], item["name"]))


def _aggregate_primary(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for row in variants:
        item = totals.setdefault(
            row["primary_id"],
            {
                "deck_id": row["primary_id"],
                "name": row["primary_name"],
                "players": 0,
                "day2s": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "variants": 0,
            },
        )
        for key in ("players", "day2s", "wins", "losses", "ties"):
            item[key] += row[key]
        item["variants"] += 1
    for item in totals.values():
        item["matches"] = item["wins"] + item["losses"] + item["ties"]
        item["effective_win_rate"] = (
            (item["wins"] + 0.5 * item["ties"]) / item["matches"]
            if item["matches"]
            else None
        )
        item["day2_rate"] = item["day2s"] / item["players"] if item["players"] else None
    return sorted(totals.values(), key=lambda item: (-item["players"], item["name"]))


def _serialize_matrix(
    matches: list[Match],
    names: dict[str, str],
) -> list[dict[str, Any]]:
    matrix = aggregate_matchups(matches)
    pair_events: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_players: dict[tuple[str, str], set[str]] = defaultdict(set)
    for match in matches:
        for key, player in (
            ((match.deck1, match.deck2), match.player1),
            ((match.deck2, match.deck1), match.player2),
        ):
            pair_events[key].add(match.tournament_id)
            pair_players[key].add(f"{match.tournament_id}:{player}")
    rows: list[dict[str, Any]] = []
    for (deck, opponent), cell in matrix.items():
        self_matchup = deck == opponent
        if self_matchup:
            low = high = None
            evidence = "self_matchup"
        else:
            low, high = wilson_interval(cell.wins + 0.5 * cell.draws, cell.n)
            evidence = evidence_grade(cell.wins, cell.losses, cell.draws)
        rows.append(
            {
                "deck_id": deck,
                "deck_name": names.get(deck, deck),
                "opponent_id": opponent,
                "opponent_name": names.get(opponent, opponent),
                "wins": cell.wins,
                "losses": cell.losses,
                "ties": cell.draws,
                "decisive": cell.decisive,
                "n": cell.n,
                "effective_win_rate": None if self_matchup else cell.effective_rate,
                "wilson_low": low,
                "wilson_high": high,
                "evidence": evidence,
                "self_matchup": self_matchup,
                "events": len(pair_events[(deck, opponent)]),
                "players": len(pair_players[(deck, opponent)]),
            }
        )
    return sorted(rows, key=lambda row: (row["deck_name"], row["opponent_name"]))


def _canonical_name(name: str) -> tuple[str, str]:
    lowered = name.lower()
    for token, canonical in COMPARISON_MAPPING_RULES:
        if token in lowered:
            return canonical, f"contains:{token}"
    return name, "passthrough"


def _comparison(
    shares: list[dict[str, Any]], kaggle_counts: Counter[str]
) -> dict[str, Any]:
    limitless: Counter[str] = Counter()
    mapping_audit: list[dict[str, Any]] = []
    for row in shares:
        canonical, rule = _canonical_name(row["name"])
        limitless[canonical] += row["share"]
        mapping_audit.append(
            {
                "source": "limitless",
                "original": row["name"],
                "canonical": canonical,
                "rule": rule,
                "mass": row["share"],
            }
        )
    kaggle: Counter[str] = Counter()
    for name, count in kaggle_counts.items():
        canonical, rule = _canonical_name(name)
        mass = count / sum(kaggle_counts.values())
        kaggle[canonical] += mass
        mapping_audit.append(
            {
                "source": "kaggle",
                "original": name,
                "canonical": canonical,
                "rule": rule,
                "mass": mass,
            }
        )
    labels = sorted(set(limitless) | set(kaggle))
    rows = [
        {
            "name": label,
            "limitless_share": limitless[label],
            "kaggle_share": kaggle[label],
            "delta": kaggle[label] - limitless[label],
        }
        for label in labels
    ]
    rows.sort(key=lambda row: (-abs(row["delta"]), row["name"]))
    l_values = [limitless[label] for label in labels]
    k_values = [kaggle[label] for label in labels]
    return {
        "rows": rows,
        "limitless_hhi": hhi(l_values),
        "kaggle_hhi": hhi(k_values),
        "limitless_top2": sum(sorted(l_values, reverse=True)[:2]),
        "kaggle_top2": sum(sorted(k_values, reverse=True)[:2]),
        "limitless_top5": sum(sorted(l_values, reverse=True)[:5]),
        "kaggle_top5": sum(sorted(k_values, reverse=True)[:5]),
        "jensen_shannon_divergence": jensen_shannon_divergence(l_values, k_values),
        "mapping_rules": [
            {"priority": index, "match": f"contains:{token}", "canonical": canonical}
            for index, (token, canonical) in enumerate(COMPARISON_MAPPING_RULES, start=1)
        ],
        "mapping_audit": mapping_audit,
    }


def _classification_audit(deck_id: str, cards: list[dict[str, Any]]) -> dict[str, Any]:
    pokemon: Counter[str] = Counter()
    for card in cards:
        if card["group"] == "pokemon":
            pokemon[card["name"]] += card["count"]
    audit: dict[str, Any] = {
        "label_source": "Limitless Labs deck category",
        "scope": "best-placement representative deck only",
        "status": "representative_only",
        "key_cards": [],
        "note": "该标签未对该牌型所有参赛卡表逐份重分类。",
    }
    if not deck_id.startswith("dragapult"):
        return audit
    axes = {
        "dragapult-dusknoir": ("Dusknoir line", sum(pokemon[name] for name in ("Duskull", "Dusclops", "Dusknoir"))),
        "dragapult-dudunsparce": ("Dudunsparce line", pokemon["Dudunsparce"] + pokemon["Dudunsparce ex"]),
        "dragapult-blaziken": ("Blaziken line", pokemon["Blaziken ex"] + pokemon["Blaziken"]),
        "dragapult-froslass": ("Froslass line", pokemon["Froslass"] + pokemon["Mega Froslass ex"]),
    }
    audit["key_cards"] = [
        {"name": name, "count": count}
        for name, count in pokemon.items()
        if name in {
            "Duskull", "Dusclops", "Dusknoir", "Dunsparce", "Dudunsparce",
            "Dudunsparce ex", "Torchic", "Combusken", "Blaziken", "Blaziken ex",
            "Snorunt", "Froslass", "Mega Froslass ex",
        }
    ]
    if deck_id in axes:
        label, count = axes[deck_id]
        audit["status"] = "representative_confirmed" if count else "representative_conflict"
        audit["note"] = f"代表卡表中的 {label} 合计 {count} 张。"
    else:
        dudunsparce = pokemon["Dudunsparce"] + pokemon["Dudunsparce ex"]
        audit["status"] = "base_or_other_representative"
        audit["note"] = (
            "站点未单列副轴；代表卡表含 "
            f"{pokemon['Dunsparce']}-{dudunsparce} 土龙 tech，不等于所有样本均为纯基础型。"
        )
    return audit


def _representatives(
    standings: list[dict[str, Any]], variants: list[dict[str, Any]], refresh: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_deck: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in standings:
        if row["deck_id"]:
            by_deck[row["deck_id"]].append(row)
        if row["player_id"] is not None:
            by_player[int(row["player_id"])].append(row)
    target_decks = {row["deck_id"] for row in variants[:18]}
    target_decks.update(row["deck_id"] for row in variants if row["primary_id"] == "dragapult-ex")
    selected: list[dict[str, Any]] = []
    urls: list[str] = []
    for deck_id in target_decks:
        candidates = [row for row in by_deck.get(deck_id, []) if row["has_decklist"]]
        if not candidates:
            continue
        best = min(
            candidates,
            key=lambda row: (
                row["placement"] is None,
                row["placement"] or 10**9,
                -row["points"],
                row["name"],
            ),
        )
        url = (
            "https://mew.limitlesstcg.com/labs/data/tcg/decklist"
            f"?tournamentId={best['lab_event_id']}&playerId={best['labs_player_id']}"
        )
        best = dict(best)
        best["decklist_api_url"] = url
        selected.append(best)
        urls.append(url)
    fetched = _fetch_many(urls, refresh)
    sources: list[dict[str, Any]] = []
    decklists: list[dict[str, Any]] = []
    for player in selected:
        response = fetched[player["decklist_api_url"]]
        cards = parse_decklist_payload(response.text)
        total = sum(card["count"] for card in cards)
        if total != 60:
            raise ValueError(f"representative deck is not 60 cards: {player['name']} ({total})")
        history = by_player[int(player["player_id"])]
        decklists.append(
            {
                "deck_id": player["deck_id"],
                "deck_name": player["deck_name"],
                "player_id": player["player_id"],
                "labs_player_id": player["labs_player_id"],
                "player_name": player["name"],
                "country": player["country"],
                "event_id": player["lab_event_id"],
                "event_name": player["event_name"],
                "placement": player["placement"],
                "record": [player["wins"], player["losses"], player["ties"]],
                "history": {
                    "events": len(history),
                    "wins": sum(row["wins"] for row in history),
                    "losses": sum(row["losses"] for row in history),
                    "ties": sum(row["ties"] for row in history),
                    "best_placement": min(
                        (row["placement"] for row in history if row["placement"] is not None),
                        default=None,
                    ),
                    "decks": sorted({row["deck_name"] for row in history if row["deck_name"]}),
                },
                "cards": cards,
                "total_cards": total,
                "classification_audit": _classification_audit(player["deck_id"], cards),
                "source_url": (
                    f"https://labs.limitlesstcg.com/{player['lab_event_id']}"
                    f"/player/{player['labs_player_id']}/decklist"
                ),
            }
        )
        sources.append(_source(response, f"representative_decklist:{player['deck_id']}"))
    decklists.sort(key=lambda row: (row["placement"] or 10**9, row["deck_name"]))
    player_summaries = []
    for player_id, history in by_player.items():
        player_summaries.append(
            {
                "player_id": player_id,
                "name": history[0]["name"],
                "country": history[0]["country"],
                "events": len(history),
                "wins": sum(row["wins"] for row in history),
                "losses": sum(row["losses"] for row in history),
                "ties": sum(row["ties"] for row in history),
                "points": sum(row["points"] for row in history),
                "best_placement": min(
                    (row["placement"] for row in history if row["placement"] is not None),
                    default=None,
                ),
                "decks": sorted({row["deck_name"] for row in history if row["deck_name"]}),
            }
        )
    player_summaries.sort(key=lambda row: (-row["points"], -row["wins"], row["name"]))
    return decklists, player_summaries, sources


def collect_snapshot(refresh: bool = False) -> dict[str, Any]:
    requested_filters = validate_filter_query(FILTER_QUERY)
    landing = _fetch_many([DECKS_URL, VARIANTS_URL, TOURNAMENTS_URL], refresh)
    sources = [
        _source(landing[DECKS_URL], "limitless_primary_distribution"),
        _source(landing[VARIANTS_URL], "limitless_variant_distribution"),
        _source(landing[TOURNAMENTS_URL], "limitless_tournament_filter"),
    ]
    for url in (DECKS_URL, VARIANTS_URL, TOURNAMENTS_URL):
        parse_filter_audit(landing[url].text)
    events = parse_tournament_index(landing[TOURNAMENTS_URL].text)
    if len(events) != 8 or sum(event.players for event in events) != 10_923:
        raise ValueError(
            f"TEF-POR filter drift: events={len(events)}, players={sum(e.players for e in events)}"
        )
    if any(event.format.lower() != "standard" for event in events):
        raise ValueError("TEF-POR tournament filter returned a non-Standard event")
    event_urls = {
        event.tournament_id: f"https://limitlesstcg.com/tournaments/{event.tournament_id}"
        for event in events
    }
    event_pages = _fetch_many(list(event_urls.values()), refresh)
    discovered_labs: dict[int, str | None] = {}
    event_format_verified: dict[int, bool] = {}
    for event in events:
        response = event_pages[event_urls[event.tournament_id]]
        discovered_labs[event.tournament_id] = parse_labs_event_id(response.text)
        event_format_verified[event.tournament_id] = "format=TEF-POR" in response.text
        if not event_format_verified[event.tournament_id]:
            raise ValueError(f"event page does not confirm TEF-POR: {event.name}")
        sources.append(_source(response, f"tournament_detail:{event.tournament_id}"))
    if sum(lab_id is not None for lab_id in discovered_labs.values()) != 7:
        raise ValueError(f"expected 7 discoverable Labs events, got {discovered_labs}")
    primary_shares = [asdict(row) for row in parse_deck_distribution(landing[DECKS_URL].text)]
    variant_shares = [asdict(row) for row in parse_deck_distribution(landing[VARIANTS_URL].text)]
    if sum(row["points"] for row in primary_shares) != sum(row["points"] for row in variant_shares):
        raise ValueError("primary and variant Points totals do not match")

    all_matches: list[Match] = []
    all_meta: list[dict[str, Any]] = []
    all_standings: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    deck_catalog: dict[str, tuple[str, str, str]] = {}
    for event in events:
        lab_id = discovered_labs[event.tournament_id]
        if lab_id is None:
            coverage.append(
                {
                    "tournament_id": event.tournament_id,
                    "lab_event_id": None,
                    "name": event.name,
                    "players": event.players,
                    "pairings_available": False,
                    "labs_link_discovered": False,
                    "event_format_verified": event_format_verified[event.tournament_id],
                    "reason": "Tournament detail page has no Labs standings/pairings link",
                }
            )
            continue
        base = "https://mew.limitlesstcg.com/labs/data/tcg"
        tournament_url = f"{base}/tournament?id={lab_id}&division=MA"
        decks_url = f"{base}/decks?tournamentId={lab_id}&division=MA"
        standings_url = f"{base}/standings?tournamentId={lab_id}&division=MA"
        core = _fetch_many([tournament_url, decks_url, standings_url], refresh)
        meta_document = json.loads(core[tournament_url].text)
        meta = meta_document["message"]
        if not meta_document.get("ok") or not meta.get("completed"):
            raise ValueError(f"Labs event is not complete: {lab_id}")
        if int(meta["players"]) != event.players:
            raise ValueError(f"player count mismatch for {event.name}")
        event_decks = parse_deck_meta_payload(core[decks_url].text)
        standings = parse_standings_payload(core[standings_url].text)
        for row in event_decks:
            identity = (row["name"], row["primary_id"], row["primary_name"])
            if row["deck_id"] in deck_catalog and deck_catalog[row["deck_id"]] != identity:
                raise ValueError(f"deck catalog conflict: {row['deck_id']}")
            deck_catalog[row["deck_id"]] = identity
            all_meta.append(row)
        for row in standings:
            row.update(
                {
                    "lab_event_id": lab_id,
                    "event_name": event.name,
                    "event_date": event.date,
                }
            )
            all_standings.append(row)
        round_urls = [
            f"{base}/pairings?tournamentId={lab_id}&division=MA&round={round_number}"
            for round_number in range(1, int(meta["round"]) + 1)
        ]
        round_results = _fetch_many(round_urls, refresh)
        totals = Counter()
        event_matches: list[Match] = []
        for round_number, url in enumerate(round_urls, start=1):
            result = round_results[url]
            matches, audit = parse_pairings_payload(result.text, lab_id, round_number)
            event_matches.extend(matches)
            totals.update(asdict(audit))
            sources.append(_source(result, f"pairings:{lab_id}:round:{round_number}"))
        all_matches.extend(event_matches)
        coverage.append(
            {
                "tournament_id": event.tournament_id,
                "lab_event_id": lab_id,
                "name": event.name,
                "players": event.players,
                "players_round_1": int(meta["players_r1"]),
                "rounds": int(meta["round"]),
                "pairings_available": True,
                "labs_link_discovered": True,
                "event_format_verified": event_format_verified[event.tournament_id],
                "pairing_rows": totals["rows"],
                "accepted_matches": totals["accepted"],
                "byes": totals["byes"],
                "unresolved": totals["unresolved"],
                "incomplete": totals["incomplete"],
                "no_result": totals["no_result"],
                "classified_players": sum(row["players"] for row in event_decks),
                "deck_categories": len(event_decks),
            }
        )
        sources.extend(
            [
                _source(core[tournament_url], f"labs_tournament:{lab_id}"),
                _source(core[decks_url], f"labs_decks:{lab_id}"),
                _source(core[standings_url], f"labs_standings:{lab_id}"),
            ]
        )

    variants = _aggregate_deck_meta(all_meta)
    primaries = _aggregate_primary(variants)
    variant_names = {key: value[0] for key, value in deck_catalog.items()}
    primary_names = {value[1]: value[2] for value in deck_catalog.values()}
    primary_matches: list[Match] = []
    for match in all_matches:
        left = deck_catalog.get(match.deck1)
        right = deck_catalog.get(match.deck2)
        if left is None or right is None:
            continue
        primary_matches.append(
            replace(
                match,
                deck1=left[1],
                deck1_name=left[2],
                deck2=right[1],
                deck2_name=right[2],
            )
        )

    decklists, player_summaries, decklist_sources = _representatives(
        all_standings, variants, refresh
    )
    sources.extend(decklist_sources)
    kaggle_text = KAGGLE_PATH.read_text(encoding="utf-8")
    kaggle_counts = parse_kaggle_archetypes(kaggle_text)
    if sum(kaggle_counts.values()) != 100:
        raise ValueError(f"Kaggle snapshot must contain 100 person cards, got {sum(kaggle_counts.values())}")
    kaggle_source = {
        "role": "kaggle_2026_07_30",
        "path": str(KAGGLE_PATH.relative_to(ROOT)),
        "sha256": hashlib.sha256(kaggle_text.encode("utf-8")).hexdigest(),
    }
    sources.append(kaggle_source)
    snapshot = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "filter_contract": {
            "query": FILTER_QUERY,
            "format": "TEF-POR",
            "events": len(events),
            "players": sum(event.players for event in events),
            "points": sum(row["points"] for row in primary_shares),
            "active_filters": parse_filter_audit(landing[TOURNAMENTS_URL].text).active_filters,
            "requested_filters": requested_filters,
        },
        "sources": sources,
        "events": [asdict(event) for event in events],
        "coverage": coverage,
        "primary_points_share": primary_shares,
        "variant_points_share": variant_shares,
        "labs_primary_meta": primaries,
        "labs_variant_meta": variants,
        "matchups_primary": _serialize_matrix(primary_matches, primary_names),
        "matchups_variant": _serialize_matrix(all_matches, variant_names),
        "pairing_summary": {
            "events_with_pairings": sum(row["pairings_available"] for row in coverage),
            "events_total": len(coverage),
            "players_in_pairing_events": sum(
                row["players"] for row in coverage if row["pairings_available"]
            ),
            "accepted_matches": sum(row.get("accepted_matches", 0) for row in coverage),
            "unresolved_rows": sum(row.get("unresolved", 0) for row in coverage),
            "no_result_rows": sum(row.get("no_result", 0) for row in coverage),
            "byes": sum(row.get("byes", 0) for row in coverage),
        },
        "representative_decklists": decklists,
        "players": player_summaries[:100],
        "kaggle": {
            "date": "2026-07-30",
            "players": sum(kaggle_counts.values()),
            "archetypes": [
                {"name": name, "count": count, "share": count / 100}
                for name, count in kaggle_counts.most_common()
            ],
        },
        "comparison": _comparison(primary_shares, kaggle_counts),
        "review_checks": {
            "filter_is_tef_por": True,
            "eight_events": len(events) == 8,
            "players_10923": sum(event.players for event in events) == 10_923,
            "points_conserved": sum(row["points"] for row in primary_shares)
            == sum(row["points"] for row in variant_shares),
            "representative_decks_are_60": all(
                row["total_cards"] == 60 for row in decklists
            ),
            "kaggle_has_100_rows": sum(kaggle_counts.values()) == 100,
        },
    }
    return snapshot


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Limitless TEF-POR 环境审计快照")
    parser.add_argument("--refresh", action="store_true", help="忽略本地网络缓存")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs" / "environment" / "limitless.html"
    )
    args = parser.parse_args()
    snapshot = collect_snapshot(refresh=args.refresh)
    report = render_report(snapshot)
    snapshot["report_sha256"] = hashlib.sha256(report.encode("utf-8")).hexdigest()
    _atomic_write_json(args.snapshot, snapshot)
    _atomic_write_text(args.output, report)
    print(
        json.dumps(
            {
                "snapshot": str(args.snapshot),
                "output": str(args.output),
                "events": snapshot["filter_contract"]["events"],
                "players": snapshot["filter_contract"]["players"],
                "matches": snapshot["pairing_summary"]["accepted_matches"],
                "representative_decks": len(snapshot["representative_decklists"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
