"""Freeze and render the current Kaggle Top 500 exact-deck distribution."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable
import zipfile
from zoneinfo import ZoneInfo

from kaggle.api.kaggle_api_extended import ApiGetLeaderboardRequest, KaggleApi

from .generate_live_snapshot import (
    COMPETITION,
    LEADERBOARD_BINDINGS,
    _card_catalog,
    _datetime,
    _decks_from_replay,
    _image_url,
    _materialize_submission_deck,
    _model_value,
    _rate_call,
    _scored_submissions_at_or_before,
    _select_distinct_deck_evidence,
    _select_leaderboard_submission,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
RANKED_ROOT = REPOSITORY_ROOT / "docs/environment-daily_kaggle_top100/ranked"
REFERENCE_ROOT = REPOSITORY_ROOT / "train/0045_single_deck_expert_minimal_lora/assets"
DECK_REGISTRY = REFERENCE_ROOT / "decks/registry.json"
TAXONOMY = REFERENCE_ROOT / "taxonomy/own_archetypes_v2.json"
MAPPING = REFERENCE_ROOT / "taxonomy/deck_own_archetype_mapping_v2.json"
CARD_DATA = REPOSITORY_ROOT / "data/official/EN_Card_Data.csv"
SCHEMA = "pokemon_tcg_current_top500_decks_v1"
SCORE_BANDS = ("1000+", "900-1000", "800-900", "700-800", "<700")
DYNAMIC_BINDING = "active_submission_score_at_team_capture"
VALID_BINDINGS = frozenset((*LEADERBOARD_BINDINGS, DYNAMIC_BINDING))


def canonical_deck_sha256(deck: Iterable[int]) -> str:
    encoded = ",".join(str(card_id) for card_id in sorted(int(card) for card in deck))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_band(value: object) -> str:
    score = float(value)
    if score >= 1000:
        return "1000+"
    if score >= 900:
        return "900-1000"
    if score >= 800:
        return "800-900"
    if score >= 700:
        return "700-800"
    return "<700"


def _card_group(row: dict[str, str] | None) -> str:
    if not row:
        return "unknown"
    kind = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
    category = row.get("Category", "")
    if "Pokémon" in kind and category not in {"Fossil", "Technical Machine"}:
        return "pokemon"
    if "Energy" in kind:
        return "energy"
    if "Stadium" in kind:
        return "stadium"
    return "trainer"


def _similarity_weight(row: dict[str, str] | None) -> float:
    return {"pokemon": 4.0, "stadium": 2.0, "trainer": 0.5, "energy": 0.35}.get(
        _card_group(row), 0.5
    )


def _weighted_jaccard(
    left: Iterable[int], right: Iterable[int], catalog: dict[int, dict[str, str]]
) -> float:
    a, b = Counter(int(card) for card in left), Counter(int(card) for card in right)
    union = set(a) | set(b)
    numerator = sum(min(a[card], b[card]) * _similarity_weight(catalog.get(card)) for card in union)
    denominator = sum(max(a[card], b[card]) * _similarity_weight(catalog.get(card)) for card in union)
    return numerator / denominator if denominator else 0.0


def _representative_cards(
    deck: Iterable[int], catalog: dict[int, dict[str, str]], *, limit: int = 2
) -> list[int]:
    counts = Counter(int(card) for card in deck)
    candidates: list[tuple[int, int, str, int]] = []
    for card_id, count in counts.items():
        row = catalog.get(card_id)
        if _card_group(row) != "pokemon":
            continue
        name = (row or {}).get("Card Name", "")
        stage = (row or {}).get("Stage (Pokémon)/Type (Energy and Trainer)", "")
        priority = 3 if "Stage 2" in stage else 2 if "Stage 1" in stage else 1
        priority += 2 if " ex" in name or name.startswith("Mega ") else 0
        candidates.append((priority, count, name, card_id))
    candidates.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    return [row[3] for row in candidates[:limit]]


def load_reference_catalog() -> dict[str, Any]:
    registry = json.loads(DECK_REGISTRY.read_text(encoding="utf-8"))
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    classes = {int(row["archetype_id"]): row for row in taxonomy["classes"]}
    mapping_by_id = {str(row["deck_id"]): row for row in mapping["decks"]}
    catalog = _card_catalog()
    decks = []
    for row in registry["decks"]:
        deck_id = str(row["deck_id"])
        if row.get("cards"):
            cards = [
                int(card["card_id"])
                for card in row["cards"]
                for _ in range(int(card["count"]))
            ]
        else:
            deck_path = REFERENCE_ROOT.parent / str(row["deck_path"])
            cards = [int(value.strip()) for value in deck_path.read_text().splitlines() if value.strip()]
        if len(cards) != 60:
            raise ValueError(f"reference deck {deck_id} has {len(cards)} cards")
        mapping_row = mapping_by_id[deck_id]
        archetype = classes[int(mapping_row["archetype_id"])]
        decks.append(
            {
                "deck_id": deck_id,
                "name": str(row["name"]),
                "deck": sorted(cards),
                "deck_sha256": canonical_deck_sha256(cards),
                "meta_id": int(archetype["archetype_id"]),
                "meta_slug": str(archetype["name"]),
                "meta_name": str(archetype["display_name"]),
                "representative_card_ids": _representative_cards(cards, catalog),
            }
        )
    if [row["deck_id"] for row in decks] != [f"{number:03d}" for number in range(1, 71)]:
        raise ValueError("reference registry must contain contiguous exact decks 001-070")
    return {
        "decks": decks,
        "classes": [classes[index] for index in range(29)],
        "card_catalog": catalog,
        "registry_sha256": _sha256(DECK_REGISTRY),
        "taxonomy_sha256": _sha256(TAXONOMY),
        "mapping_sha256": _sha256(MAPPING),
    }


def classify_decks(payload: dict[str, Any], *, reference: dict[str, Any] | None = None) -> None:
    reference = reference or load_reference_catalog()
    catalog = reference["card_catalog"]
    by_hash = {row["deck_sha256"]: row for row in reference["decks"]}
    classes = {int(row["archetype_id"]): row for row in reference["classes"]}
    unknown: dict[str, dict[str, Any]] = {}
    for player in payload["players"]:
        for evidence in player["decks"]:
            deck = sorted(int(card) for card in evidence["deck"])
            digest = canonical_deck_sha256(deck)
            evidence["deck"] = deck
            evidence["deck_sha256"] = digest
            known = by_hash.get(digest)
            if known:
                evidence["classification"] = {
                    "kind": "catalog_exact",
                    "deck_id": known["deck_id"],
                    "report_deck_id": known["deck_id"],
                    "deck_name": known["name"],
                    "meta_id": known["meta_id"],
                    "meta_slug": known["meta_slug"],
                    "meta_name": known["meta_name"],
                    "audit": {"exact_deck_sha256": digest},
                }
                continue
            scores = sorted(
                (
                    (_weighted_jaccard(deck, row["deck"], catalog), row["deck_id"], row)
                    for row in reference["decks"]
                ),
                key=lambda item: (-item[0], item[1]),
            )
            best_score, _, best = scores[0]
            other_meta_score = next(
                (score for score, _, row in scores if row["meta_id"] != best["meta_id"]), 0.0
            )
            unknown[digest] = {
                "kind": "meta_nearest",
                "deck_id": None,
                "report_deck_id": None,
                "deck_name": f"Non-catalog {best['meta_name']}",
                "meta_id": best["meta_id"],
                "meta_slug": best["meta_slug"],
                "meta_name": classes[best["meta_id"]]["display_name"],
                "audit": {
                    "method": "weighted_multiset_jaccard_v1",
                    "weights": {"pokemon": 4.0, "stadium": 2.0, "trainer": 0.5, "energy": 0.35},
                    "nearest_deck_id": best["deck_id"],
                    "similarity": round(best_score, 6),
                    "next_meta_similarity": round(other_meta_score, 6),
                    "meta_margin": round(best_score - other_meta_score, 6),
                },
            }
    by_meta: dict[str, list[str]] = defaultdict(list)
    for digest, result in unknown.items():
        by_meta[result["meta_slug"]].append(digest)
    for meta_slug, digests in by_meta.items():
        for number, digest in enumerate(sorted(digests), 1):
            unknown[digest]["report_deck_id"] = f"00_{meta_slug}_P{number:02d}"
    for player in payload["players"]:
        for evidence in player["decks"]:
            digest = evidence["deck_sha256"]
            if digest in unknown:
                evidence["classification"] = dict(unknown[digest])
    payload["classification_contract"] = {
        "priority": ["exact_001_070", "nearest_29_meta"],
        "unknown_method": "weighted_multiset_jaccard_v1",
        "reference_registry_sha256": reference["registry_sha256"],
        "taxonomy_version": "own_archetypes_v2",
        "taxonomy_sha256": reference["taxonomy_sha256"],
        "mapping_sha256": reference["mapping_sha256"],
        "class_count": 29,
    }


def validate_snapshot(
    payload: dict[str, Any], *, expected_count: int | None = None, replay_root: Path | None = None
) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported snapshot schema: {payload.get('schema')!r}")
    players = list(payload.get("players") or [])
    expected_count = expected_count if expected_count is not None else int(payload["leaderboard_count"])
    if len(players) != expected_count:
        raise ValueError(f"expected {expected_count} players, got {len(players)}")
    if [int(row["rank"]) for row in players] != list(range(1, expected_count + 1)):
        raise ValueError("ranks must be contiguous and unique")
    cutoff = _datetime(payload.get("captured_at_utc"))
    if cutoff is None:
        raise ValueError("invalid captured_at_utc")
    bands = Counter()
    dual = evidence_count = exact_count = strict_bindings = dynamic_bindings = 0
    for player in players:
        rank = int(player["rank"])
        binding = str(player.get("binding"))
        if binding not in VALID_BINDINGS:
            raise ValueError(f"rank {rank}: invalid leaderboard binding")
        scores_match = Decimal(str(player["score"])) == Decimal(
            str(player["submission_public_score"])
        )
        expected_delta = Decimal(str(player["submission_public_score"])) - Decimal(
            str(player["score"])
        )
        if abs(Decimal(str(player.get("score_binding_delta"))) - expected_delta) > Decimal(
            "0.000001"
        ):
            raise ValueError(f"rank {rank}: invalid score binding delta")
        if binding in LEADERBOARD_BINDINGS and not scores_match:
            raise ValueError(f"rank {rank}: leaderboard score mismatch")
        if binding == DYNAMIC_BINDING and scores_match:
            raise ValueError(f"rank {rank}: dynamic binding unexpectedly has an exact score match")
        strict_bindings += binding in LEADERBOARD_BINDINGS
        dynamic_bindings += binding == DYNAMIC_BINDING
        leaderboard_submitted_at = _datetime(player.get("submission_date"))
        if leaderboard_submitted_at is None or leaderboard_submitted_at > cutoff:
            raise ValueError(f"rank {rank}: invalid leaderboard submission cutoff")
        decks = list(player.get("decks") or [])
        if not 1 <= len(decks) <= 2:
            raise ValueError(f"rank {rank}: expected one or two decks")
        if decks[0].get("role") != "leaderboard_primary":
            raise ValueError(f"rank {rank}: first deck is not leaderboard primary")
        if int(decks[0]["submission_id"]) != int(player["submission_id"]):
            raise ValueError(f"rank {rank}: primary submission mismatch")
        hashes = set()
        for evidence in decks:
            evidence_count += 1
            deck = [int(card) for card in evidence.get("deck") or []]
            if len(deck) != 60:
                raise ValueError(f"rank {rank}: deck does not have exactly 60 cards")
            digest = canonical_deck_sha256(deck)
            if digest != evidence.get("deck_sha256"):
                raise ValueError(f"rank {rank}: deck hash mismatch")
            if digest in hashes:
                raise ValueError(f"rank {rank}: duplicate retained deck")
            hashes.add(digest)
            submitted_at = _datetime(evidence.get("submitted_at"))
            episode_at = _datetime(evidence.get("episode_create_time"))
            if submitted_at is None or submitted_at > cutoff:
                raise ValueError(f"rank {rank}: invalid submission cutoff")
            if episode_at is None or episode_at > cutoff:
                raise ValueError(f"rank {rank}: invalid Episode cutoff")
            if int(evidence.get("episode_player_index", -1)) not in (0, 1):
                raise ValueError(f"rank {rank}: invalid Episode player index")
            if evidence.get("classification", {}).get("kind") == "catalog_exact":
                exact_count += 1
            if replay_root is not None:
                replay = replay_root / f"episode-{int(evidence['episode_id'])}.json"
                if not replay.exists():
                    raise ValueError(f"rank {rank}: missing replay {replay.name}")
                replay_decks = _decks_from_replay(json.loads(replay.read_text(encoding="utf-8")))
                replay_deck = replay_decks[int(evidence["episode_player_index"])]
                if canonical_deck_sha256(replay_deck) != digest:
                    raise ValueError(f"rank {rank}: replay deck identity mismatch")
        dual += len(decks) == 2
        bands[score_band(player["score"])] += 1
    audit = {
        "status": "PASS",
        "players": len(players),
        "deck_evidence": evidence_count,
        "dual_deck_players": dual,
        "catalog_exact_evidence": exact_count,
        "non_catalog_evidence": evidence_count - exact_count,
        "score_bands": {band: bands[band] for band in SCORE_BANDS},
        "strict_score_bindings": strict_bindings,
        "dynamic_score_bindings": dynamic_bindings,
        "all_scores_bound": dynamic_bindings == 0,
        "all_decks_exact_60": True,
        "all_episode_times_at_or_before_capture": True,
        "replay_identity_checked": replay_root is not None,
    }
    if sum(audit["score_bands"].values()) != len(players):
        raise ValueError("score-band population does not sum to players")
    return audit


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _fetch_leaderboard(api: KaggleApi, limit: int) -> list[object]:
    rows: list[object] = []
    token = None
    with api.build_kaggle_client() as kaggle:
        while len(rows) < limit:
            request = ApiGetLeaderboardRequest()
            request.competition_name = COMPETITION
            api._set_paging(request, min(200, limit - len(rows)), token)
            response = _rate_call(
                lambda: kaggle.competitions.competition_api_client.get_leaderboard(request)
            )
            batch = list(response.submissions or [])
            rows.extend(batch)
            token = response.next_page_token
            if not batch or not token:
                break
    return rows[:limit]


def _leaderboard_fingerprint(rows: list[object]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            int(_model_value(row, "team_id", "teamId")),
            str(_model_value(row, "score")),
            str(_model_value(row, "submission_date", "submissionDate")),
        )
        for row in rows
    )


def _fetch_team_submissions_parallel(
    api: KaggleApi,
    rows: list[object],
    *,
    cache_path: Path,
    existing: dict[int, list[object]] | None = None,
    force: bool = False,
) -> dict[int, list[object]]:
    result = dict(existing or {})
    unique_team_ids = list(
        dict.fromkeys(int(_model_value(row, "team_id", "teamId")) for row in rows)
    )
    targets = unique_team_ids if force else [team_id for team_id in unique_team_ids if team_id not in result]
    for completed, team_id in enumerate(targets, 1):
        result[team_id] = list(
            _rate_call(lambda team_id=team_id: api.competition_team_submissions(team_id))
        )
        _atomic_json(
            cache_path,
            {
                "schema": "pokemon_tcg_team_submission_capture_cache_v1",
                "teams": {
                    str(key): [_serialize_submission(row) for row in value]
                    for key, value in result.items()
                },
            },
        )
        if completed % 10 == 0 or completed == len(targets):
            print(
                f"[submission views {completed:03d}/{len(targets)} · cached {len(result):03d}]",
                flush=True,
            )
    return result


def _download_leaderboard_snapshot(
    api: KaggleApi, limit: int, work: Path
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    download_root = work / "leaderboard_download"
    download_root.mkdir(parents=True, exist_ok=True)
    api.competition_leaderboard_download(COMPETITION, str(download_root), quiet=True)
    archive = download_root / f"{COMPETITION}.zip"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        raise RuntimeError("Kaggle leaderboard download did not produce a valid ZIP archive")
    with zipfile.ZipFile(archive) as package:
        csv_names = [name for name in package.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise RuntimeError(f"expected one leaderboard CSV, got {csv_names}")
        csv_name = csv_names[0]
        match = re.search(r"-(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.csv$", csv_name)
        if not match:
            raise RuntimeError(f"leaderboard CSV lacks a capture timestamp: {csv_name}")
        captured_at = datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)
        with package.open(csv_name) as stream:
            text = stream.read().decode("utf-8-sig").splitlines()
    source_rows = list(csv.DictReader(text))
    if len(source_rows) < limit:
        raise RuntimeError(f"leaderboard CSV has {len(source_rows)} rows, expected at least {limit}")
    rows = [
        {
            "rank": int(row["Rank"]),
            "team_id": int(row["TeamId"]),
            "team_name": str(row["TeamName"]),
            "score": float(row["Score"]),
            "submission_date": str(row["LastSubmissionDate"]),
        }
        for row in source_rows[:limit]
    ]
    if [row["rank"] for row in rows] != list(range(1, limit + 1)):
        raise RuntimeError("downloaded leaderboard ranks are not contiguous")
    return rows, captured_at.isoformat(), {
        "leaderboard_source": "kaggle_competition_leaderboard_download",
        "leaderboard_archive": archive.name,
        "leaderboard_csv": csv_name,
        "leaderboard_archive_sha256": _sha256(archive),
    }


def _capture_leaderboard(
    api: KaggleApi, limit: int, cache_path: Path, work: Path
) -> tuple[list[dict[str, Any]], str, dict[int, list[object]], dict[str, Any]]:
    cached_payload = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    cached_teams = {
        int(team_id): [_as_submission(row) for row in submissions]
        for team_id, submissions in (cached_payload.get("teams") or {}).items()
    }
    leaderboard, captured_at, capture_audit = _download_leaderboard_snapshot(api, limit, work)
    print("ensuring all downloaded leaderboard teams have cached submission views", flush=True)
    submissions_by_team = _fetch_team_submissions_parallel(
        api, leaderboard, cache_path=cache_path, existing=cached_teams
    )
    strict = 0
    for row in leaderboard:
        try:
            _select_leaderboard_submission(row, submissions_by_team[row["team_id"]])
            strict += 1
        except ValueError:
            pass
    capture_audit.update(
        {
            "team_submission_capture": "asynchronous_cached_public_team_views",
            "strict_score_bindings_at_capture": strict,
            "dynamic_score_bindings_at_capture": limit - strict,
            "binding_boundary": (
                "rank and score come from one server-generated leaderboard CSV; deck identity "
                "comes from asynchronously captured active team submissions"
            ),
        }
    )
    return leaderboard, captured_at, submissions_by_team, capture_audit


def _serialize_submission(submission: object) -> dict[str, Any]:
    return {
        "id": int(_model_value(submission, "id", default=0) or 0),
        "public_score": str(_model_value(submission, "public_score", "publicScore", default="")),
        "date_submitted": str(_model_value(submission, "date_submitted", "dateSubmitted", default="")),
    }


def _as_submission(row: dict[str, Any]) -> object:
    class Submission:
        pass

    result = Submission()
    result.id = int(row["id"])
    result.public_score = row.get("public_score")
    result.date_submitted = row.get("date_submitted")
    return result


def _select_ranked_submission(
    row: dict[str, Any], submissions: list[object], cutoff: datetime
) -> tuple[object, str]:
    eligible = _scored_submissions_at_or_before(submissions, cutoff)
    if not eligible:
        raise ValueError(f"rank {row['rank']}: no scored submission exists at the leaderboard cutoff")
    try:
        return _select_leaderboard_submission(row, eligible)
    except ValueError:
        return eligible[0], DYNAMIC_BINDING


def _materialize_player(
    api: KaggleApi,
    row: dict[str, Any],
    cutoff: datetime,
    replay_root: Path,
    catalog: dict[int, dict[str, str]],
    submissions: list[object],
) -> dict[str, Any]:
    selected, binding = _select_ranked_submission(row, submissions, cutoff)
    primary = _materialize_submission_deck(api, selected, cutoff, replay_root, catalog)
    if primary is None:
        raise RuntimeError(f"rank {row['rank']}: primary submission has no eligible Episode")
    primary["role"] = "leaderboard_primary"
    retained = [primary]
    audited_submission_ids = [int(primary["submission_id"])]
    for submission in _scored_submissions_at_or_before(submissions, cutoff):
        submission_id = int(_model_value(submission, "id", default=0) or 0)
        if submission_id == int(primary["submission_id"]):
            continue
        audited_submission_ids.append(submission_id)
        evidence = _materialize_submission_deck(api, submission, cutoff, replay_root, catalog)
        if evidence is None:
            continue
        evidence["role"] = "alternate_high_score"
        retained = _select_distinct_deck_evidence(retained + [evidence], limit=2)
        if len(retained) == 2:
            break
    submission_score = Decimal(str(_model_value(selected, "public_score", "publicScore")))
    leaderboard_score = Decimal(str(row["score"]))
    return {
        **row,
        "submission_id": int(_model_value(selected, "id")),
        "submission_public_score": str(_model_value(selected, "public_score", "publicScore")),
        "submission_date_actual": str(_model_value(selected, "date_submitted", "dateSubmitted")),
        "binding": binding,
        "score_binding_delta": float(submission_score - leaderboard_score),
        "decks": retained,
        "alternate_audit": {
            "scored_submissions_available": len(_scored_submissions_at_or_before(submissions, cutoff)),
            "submission_ids_examined": audited_submission_ids,
            "stopped_after_distinct_deck_found": len(retained) == 2,
        },
    }


def collect_snapshot(work: Path, snapshot_path: Path, *, limit: int = 500) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    replay_root = work / "representative_replays"
    replay_root.mkdir(exist_ok=True)
    state_path = work / "snapshot.collecting.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete":
            raise RuntimeError("completed Top 500 work directory is immutable")
        if int(state["leaderboard_count"]) != limit:
            raise ValueError("work directory belongs to another leaderboard limit")
    else:
        api = KaggleApi()
        api.authenticate()
        leaderboard, captured_at, submissions_by_team, capture_audit = _capture_leaderboard(
            api, limit, work / "team_submissions.capture.json", work
        )
        rows = list(leaderboard)
        state = {
            "schema": SCHEMA,
            "status": "collecting",
            "competition": COMPETITION,
            "captured_at_utc": captured_at,
            "leaderboard_capture": capture_audit,
            "leaderboard_count": limit,
            "rows": rows,
            "submissions_by_team": {
                str(team_id): [_serialize_submission(row) for row in submissions]
                for team_id, submissions in submissions_by_team.items()
            },
            "players_by_rank": {},
        }
        _atomic_json(state_path, state)
    api = KaggleApi()
    api.authenticate()
    cutoff = _datetime(state["captured_at_utc"])
    if cutoff is None:
        raise ValueError("invalid capture cutoff")
    catalog = _card_catalog()
    frozen_submissions = state.get("submissions_by_team")
    if not isinstance(frozen_submissions, dict):
        raise RuntimeError(
            "work directory predates frozen team-submission capture; start a new Top 500 run"
        )
    for row in state["rows"]:
        key = str(row["rank"])
        if key in state["players_by_rank"]:
            continue
        print(f"[{int(row['rank']):03d}/{limit}] {row['team_name']} · {row['score']:.1f}", flush=True)
        state["players_by_rank"][key] = _materialize_player(
            api,
            row,
            cutoff,
            replay_root,
            catalog,
            [_as_submission(item) for item in frozen_submissions[str(row["team_id"])]],
        )
        _atomic_json(state_path, state)
    payload = {
        key: value
        for key, value in state.items()
        if key not in {"rows", "players_by_rank", "submissions_by_team"}
    }
    payload["players"] = [state["players_by_rank"][str(rank)] for rank in range(1, limit + 1)]
    payload["coverage"] = {
        "lowest_rank": limit,
        "lowest_score": float(payload["players"][-1]["score"]),
        "reached_700": float(payload["players"][-1]["score"]) <= 700,
    }
    classify_decks(payload)
    payload["identity_audit"] = validate_snapshot(
        payload, expected_count=limit, replay_root=replay_root
    )
    payload["status"] = "complete"
    _atomic_json(snapshot_path, payload)
    state["status"] = "complete"
    state["formal_snapshot"] = str(snapshot_path)
    _atomic_json(state_path, state)
    return payload


def _card_thumb(card_id: int, catalog: dict[int, dict[str, str]], *, large: bool = False) -> str:
    row = catalog.get(card_id)
    name = (row or {}).get("Card Name", f"Card ID {card_id}")
    source = _image_url(row)
    preview = _image_url(row, hires=True)
    width, height = ((92, 129) if large else (38, 53))
    image = (
        f'<img src="{html.escape(source)}" alt="{html.escape(name)}" width="{width}" height="{height}" '
        'loading="lazy" decoding="async" referrerpolicy="no-referrer">'
        if source
        else f'<span>ID {card_id}</span>'
    )
    return (
        f'<span class="card-thumb {"large" if large else ""}" tabindex="0" '
        f'data-preview="{html.escape(preview)}" data-name="{html.escape(name)}">{image}</span>'
    )


def _deck_grid(deck: list[int], catalog: dict[int, dict[str, str]]) -> str:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for card_id, count in Counter(deck).items():
        groups[_card_group(catalog.get(card_id))].append((card_id, count))
    sections = []
    labels = {"pokemon": "Pokémon", "trainer": "Trainer", "stadium": "Stadium", "energy": "Energy", "unknown": "Unknown"}
    for group in ("pokemon", "trainer", "stadium", "energy", "unknown"):
        cards = groups.get(group, [])
        if not cards:
            continue
        cards.sort(key=lambda item: catalog.get(item[0], {}).get("Card Name", str(item[0])))
        tiles = "".join(
            f'<article class="card-tile"><div>{_card_thumb(card_id, catalog, large=True)}<b>×{count}</b></div>'
            f'<span>{html.escape(catalog.get(card_id, {}).get("Card Name", f"ID {card_id}"))}</span></article>'
            for card_id, count in cards
        )
        sections.append(f'<section class="deck-group"><h5>{labels[group]} · {sum(count for _, count in cards)} 张</h5><div class="deck-grid">{tiles}</div></section>')
    return "".join(sections)


def _time(value: object) -> str:
    parsed = _datetime(value)
    return parsed.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S CST") if parsed else str(value)


def render_report(
    payload: dict[str, Any], output: Path, *, reference: dict[str, Any] | None = None
) -> None:
    reference = reference or load_reference_catalog()
    catalog = reference["card_catalog"]
    players = list(payload["players"])
    audit = validate_snapshot(payload, expected_count=len(players))
    primary = [player["decks"][0] for player in players]
    all_evidence = [deck for player in players for deck in player["decks"]]
    exact_primary = Counter(
        deck["classification"]["deck_id"]
        for deck in primary
        if deck["classification"]["kind"] == "catalog_exact"
    )
    exact_alternate = Counter(
        deck["classification"]["deck_id"]
        for player in players
        for deck in player["decks"][1:]
        if deck["classification"]["kind"] == "catalog_exact"
    )
    max_exact = max([exact_primary[row["deck_id"]] + exact_alternate[row["deck_id"]] for row in reference["decks"]] or [1])
    bars = []
    for row in reference["decks"]:
        deck_id = row["deck_id"]
        p, a = exact_primary[deck_id], exact_alternate[deck_id]
        total = p + a
        height = 210 * total / max_exact if max_exact else 0
        alt_height = height * a / total if total else 0
        art = "".join(_card_thumb(card, catalog) for card in row["representative_card_ids"][:1])
        bars.append(
            f'<div class="catalog-bar" data-catalog-bar="{deck_id}" title="{html.escape(row["name"])} · primary {p} · alternate {a}">'
            f'<div class="bar-value">{total}</div><div class="bar-track"><i class="primary" style="height:{height-alt_height:.2f}px"></i>'
            f'<i class="alternate" style="height:{alt_height:.2f}px"></i></div>{art}<b>{deck_id}</b></div>'
        )
    exact_band_rows = []
    for row in reference["decks"]:
        deck_id = row["deck_id"]
        scoped = [
            player
            for player in players
            if player["decks"][0]["classification"]["kind"] == "catalog_exact"
            and player["decks"][0]["classification"]["deck_id"] == deck_id
        ]
        band_cells = "".join(
            f'<td>{sum(score_band(player["score"]) == band for player in scoped)}</td>'
            for band in SCORE_BANDS[:-1]
        )
        exact_band_rows.append(
            f'<tr><td><b>{deck_id}</b><small>{html.escape(row["name"])}</small></td>'
            f'<td><b>{len(scoped)}</b></td><td>{exact_alternate[deck_id]}</td>{band_cells}</tr>'
        )
    band_counts = Counter(score_band(player["score"]) for player in players)
    band_cards = "".join(
        f'<article><b>{band_counts[band]}</b><span>{band}</span><small>{100*band_counts[band]/len(players):.1f}%</small></article>'
        for band in SCORE_BANDS if band_counts[band]
    )
    meta_rows = []
    for meta in reference["classes"]:
        meta_id = int(meta["archetype_id"])
        scoped = [player for player in players if int(player["decks"][0]["classification"]["meta_id"]) == meta_id]
        evidence_n = sum(int(deck["classification"]["meta_id"]) == meta_id for deck in all_evidence)
        if not scoped and not evidence_n:
            continue
        cells = "".join(f'<td>{sum(score_band(player["score"]) == band for player in scoped)}</td>' for band in SCORE_BANDS[:-1])
        meta_rows.append(
            f'<tr><td><b>{meta_id:02d} · {html.escape(str(meta["display_name"]))}</b><small>{html.escape(str(meta["name"]))}</small></td>'
            f'<td><b>{len(scoped)}</b></td><td>{evidence_n}</td>{cells}</tr>'
        )
    unknown_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for player in players:
        for deck in player["decks"]:
            if deck["classification"]["kind"] == "meta_nearest":
                unknown_groups[deck["deck_sha256"]].append((player, deck))
    unknown_rows = []
    for deck_hash, observations in sorted(unknown_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        first_player, deck = observations[0]
        cls = deck["classification"]
        ranks = sorted({int(player["rank"]) for player, _ in observations})
        band_cells = "".join(
            f'<td>{sum(score_band(player["score"]) == band for player, _ in observations)}</td>'
            for band in SCORE_BANDS[:-1]
        )
        unknown_rows.append(
            f'<tr><td><code>{html.escape(cls["report_deck_id"])}</code></td><td>{html.escape(cls["meta_name"])}</td>'
            f'<td>{len(observations)}</td><td>#{min(ranks)}</td><td>{cls["audit"]["similarity"]:.3f}</td>'
            f'<td>nearest {cls["audit"]["nearest_deck_id"]}</td>{band_cells}<td><code>{deck_hash[:12]}</code></td></tr>'
        )
    rank_rows, details = [], []
    for player in players:
        primary_deck = player["decks"][0]
        cls = primary_deck["classification"]
        leaderboard_submitted_at = player["submission_date"]
        binding_label = (
            "严格同分"
            if player["binding"] in LEADERBOARD_BINDINGS
            else f"抓取漂移 {float(player['score_binding_delta']):+.1f}"
        )
        search = f'{player["rank"]} {player["team_name"]} {cls["report_deck_id"]} {cls["meta_name"]}'.lower()
        deck_labels = " + ".join(deck["classification"]["report_deck_id"] for deck in player["decks"])
        rank_rows.append(
            f'<tr data-player-row data-band="{score_band(player["score"])}" data-search="{html.escape(search)}"><td><b>#{player["rank"]}</b></td>'
            f'<td><a href="#player-{int(player["rank"]):03d}">{html.escape(str(player["team_name"]))}</a></td><td><b>{float(player["score"]):.1f}</b></td>'
            f'<td>{html.escape(score_band(player["score"]))}</td><td>{html.escape(deck_labels)}</td><td>{html.escape(cls["meta_name"])}</td>'
            f'<td><time datetime="{html.escape(str(leaderboard_submitted_at))}">{html.escape(_time(leaderboard_submitted_at))}</time></td>'
            f'<td>{html.escape(binding_label)}</td></tr>'
        )
        deck_sections = []
        for index, deck in enumerate(player["decks"], 1):
            deck_cls = deck["classification"]
            badge = "榜单主构筑" if index == 1 else "第二套高分构筑"
            audit_text = (
                "001–070 exact match"
                if deck_cls["kind"] == "catalog_exact"
                else f"29 Meta nearest {deck_cls['audit']['nearest_deck_id']} · similarity {deck_cls['audit']['similarity']:.3f} · margin {deck_cls['audit']['meta_margin']:.3f}"
            )
            deck_sections.append(
                f'<section class="deck-evidence" data-deck-evidence><div class="deck-head"><div><span class="pill">{badge}</span>'
                f'<h4>{html.escape(deck_cls["report_deck_id"])} · {html.escape(deck_cls["deck_name"])}</h4></div>'
                f'<div><b>{float(deck["public_score"]):.1f} 分</b><small>提交时间 {_time(deck["submitted_at"])}</small></div></div>'
                f'<p>submission {deck["submission_id"]} · Episode {deck["episode_id"]} · P{deck["episode_player_index"]} · '
                f'<code>{deck["deck_sha256"][:12]}</code> · {html.escape(audit_text)}</p>{_deck_grid(deck["deck"], catalog)}</section>'
            )
        details.append(
            f'<details class="player-detail" id="player-{int(player["rank"]):03d}" data-player-detail data-band="{score_band(player["score"])}" data-search="{html.escape(search)}">'
            f'<summary><span><b>#{player["rank"]} {html.escape(str(player["team_name"]))}</b><small>{float(player["score"]):.1f} 分 · 榜单提交时间 {_time(leaderboard_submitted_at)} · {html.escape(binding_label)}</small></span>'
            f'<span>{html.escape(deck_labels)} · {len(player["decks"])} 套</span></summary><div class="detail-body">{"".join(deck_sections)}</div></details>'
        )
    captured = _time(payload["captured_at_utc"])
    lowest = float(payload.get("coverage", {}).get("lowest_score", players[-1]["score"]))
    css = """
:root{--ink:#18211d;--muted:#66716c;--line:#d5ddd8;--paper:#fff;--bg:#eef1ed;--green:#155d46;--green2:#2e8b68;--gold:#d69a2d;--blue:#3273a8}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,system-ui,"Microsoft YaHei",sans-serif;letter-spacing:0}a{color:var(--green);text-decoration:none}nav{position:sticky;top:0;z-index:20;display:flex;gap:4px;overflow:auto;padding:9px max(16px,calc((100% - 1480px)/2));background:#143c30;border-bottom:1px solid #3e6456}nav a{flex:none;padding:7px 9px;color:#eff8f4}.hero{padding:38px max(18px,calc((100% - 1440px)/2));background:#174b3c;color:#fff;border-bottom:5px solid var(--gold)}.hero h1{max-width:1050px;margin:4px 0 10px;font-size:40px;line-height:1.12}.hero p{max-width:1000px;color:#dcebe5}.metrics,.band-cards{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:9px;margin-top:20px}.metric,.band-cards article{padding:13px;border:1px solid #ffffff30;border-radius:7px;background:#ffffff0d}.metric b,.metric span,.band-cards b,.band-cards span,.band-cards small{display:block}.metric b,.band-cards b{font-size:25px}.band{padding:32px max(16px,calc((100% - 1480px)/2))}.band:nth-of-type(even){background:#f9faf9}.heading{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:16px}.heading h2{margin:0;font-size:27px}.heading p{max-width:820px;margin:0;color:var(--muted)}.notice{padding:13px 15px;border-left:4px solid var(--gold);background:#fff7e7}.band-cards article{background:#fff;border-color:var(--line)}.band-cards small{color:var(--muted)}.chart-scroll,.table-scroll{overflow:auto;border:1px solid var(--line);border-radius:7px;background:#fff}.catalog-chart{display:flex;align-items:end;gap:8px;min-width:5200px;height:330px;padding:16px}.catalog-bar{display:grid;grid-template-rows:20px 210px 55px 20px;justify-items:center;width:62px}.bar-value{font-weight:800}.bar-track{display:flex;flex-direction:column-reverse;justify-content:flex-start;width:28px;height:210px;background:#e5ebe7}.bar-track i{display:block;width:100%}.bar-track .primary{background:var(--green2)}.bar-track .alternate{background:var(--gold)}.card-thumb{position:relative;display:inline-grid;place-items:center;width:38px;height:53px;overflow:hidden;border:1px solid #c9d2cd;border-radius:4px;background:#edf1ef}.card-thumb.large{width:92px;height:129px}.card-thumb img{width:100%;height:100%;object-fit:contain}.legend{display:flex;gap:20px;margin:10px 0}.legend i{display:inline-block;width:12px;height:12px;margin-right:5px}.legend .p{background:var(--green2)}.legend .a{background:var(--gold)}table{width:100%;border-collapse:collapse}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}th{position:sticky;top:0;z-index:2;background:#e8eee9;white-space:nowrap}td small{display:block;color:var(--muted)}.meta-table,.rank-table{min-width:1000px}.tools{display:flex;gap:9px;margin-bottom:12px}.tools input,.tools select{min-height:38px;padding:7px 9px;border:1px solid #b8c4bd;border-radius:5px;background:#fff}.tools input{flex:1}.player-detail{margin:9px 0;border:1px solid var(--line);border-radius:7px;background:#fff}.player-detail>summary{display:flex;justify-content:space-between;gap:18px;padding:13px;cursor:pointer}.player-detail>summary span:first-child b,.player-detail>summary span:first-child small{display:block}.detail-body{padding:0 14px 16px}.deck-evidence{margin-top:14px;padding-top:14px;border-top:2px solid #e2e8e4}.deck-head{display:flex;justify-content:space-between;gap:20px}.deck-head h4{margin:6px 0}.deck-head small{display:block;color:var(--muted)}.pill{display:inline-block;padding:3px 7px;border-radius:4px;background:#e3f2eb;color:#155d46;font-size:11px;font-weight:800}.deck-group h5{margin:15px 0 7px;color:var(--green)}.deck-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(92px,1fr));gap:9px}.card-tile{min-width:0}.card-tile>div{position:relative;width:92px;height:129px}.card-tile>div>b{position:absolute;z-index:2;top:4px;right:4px;padding:2px 5px;border-radius:4px;background:#153e32;color:#fff}.card-tile>span{display:block;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}.evidence-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.evidence-grid article{padding:15px;border-left:4px solid var(--blue);background:#fff}code{padding:2px 4px;border-radius:3px;background:#edf1ef}.preview{position:fixed;z-index:100;display:none;width:244px;padding:6px;border:1px solid #b8c4bd;border-radius:7px;background:#fff;box-shadow:0 18px 45px #10231c55;pointer-events:none}.preview.visible{display:block}.preview img{width:230px;height:322px;object-fit:contain}.preview b{display:block;text-align:center}@media(max-width:850px){.metrics,.band-cards{grid-template-columns:repeat(2,1fr)}.heading,.deck-head,.player-detail>summary{align-items:flex-start;flex-direction:column}.evidence-grid{grid-template-columns:1fr}.hero h1{font-size:31px}.deck-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.card-tile>div,.card-thumb.large{width:100%;height:auto;aspect-ratio:2.5/3.5}}
"""
    script = """
const q=document.getElementById('player-search'),b=document.getElementById('band-filter');function filter(){const term=q.value.trim().toLowerCase();document.querySelectorAll('[data-player-row],[data-player-detail]').forEach(row=>row.hidden=(term&&!row.dataset.search.includes(term))||(b.value&&row.dataset.band&&row.dataset.band!==b.value));}q.oninput=filter;b.onchange=filter;
const p=document.createElement('div');p.className='preview';p.innerHTML='<img width="230" height="322" alt="card preview"><b></b>';document.body.appendChild(p);document.addEventListener('pointerover',e=>{const t=e.target.closest?.('.card-thumb');if(!t||!t.dataset.preview)return;p.querySelector('img').src=t.dataset.preview;p.querySelector('b').textContent=t.dataset.name;p.classList.add('visible')});document.addEventListener('pointermove',e=>{if(!p.classList.contains('visible'))return;p.style.left=Math.min(innerWidth-260,e.clientX+16)+'px';p.style.top=Math.min(innerHeight-360,e.clientY+16)+'px'});document.addEventListener('pointerout',e=>{if(e.target.closest?.('.card-thumb'))p.classList.remove('visible')});
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kaggle 当前 Top {len(players)} 构筑环境分布</title><style>{css}</style></head><body>
<nav><a href="#summary">摘要</a><a href="#score-bands">分段人数</a><a href="#catalog-distribution">001–070</a><a href="#meta-distribution">29 Meta</a><a href="#rank-index">排名</a><a href="#player-decks">逐人构筑</a><a href="#evidence-boundary">证据</a></nav>
<header class="hero" id="summary"><small>POKÉMON TCG AI BATTLE · CURRENT RANKED SNAPSHOT</small><h1>当前 Top {len(players)} 真实环境构筑分布</h1><p>榜单排名与分数冻结于 {html.escape(captured)} 的一次性 Kaggle leaderboard CSV；卡组来自异步抓取的 active submission 官方 replay。{audit["strict_score_bindings"]} 行可与榜分严格同分绑定，{audit["dynamic_score_bindings"]} 行在抓取期间已发生分数漂移并逐行标出；每位选手最多保留两套不同 exact deck。</p><div class="metrics"><div class="metric"><b>{len(players)}</b><span>榜单选手</span></div><div class="metric"><b>{audit["dual_deck_players"]}</b><span>双构筑选手</span></div><div class="metric"><b>{audit["deck_evidence"]}</b><span>exact deck 证据</span></div><div class="metric"><b>{audit["catalog_exact_evidence"]}</b><span>命中 001–070</span></div><div class="metric"><b>{lowest:.1f}</b><span>最低覆盖分</span></div></div></header>
<main><section class="band" id="score-bands"><div class="heading"><div><h2>各分段人数</h2></div><p>人数只按 leaderboard 行计一次；第二套构筑不重复增加分段人口。</p></div><div class="band-cards">{band_cards}</div><p class="notice">本次 Top {len(players)} 最低分 {lowest:.1f}；{'已覆盖到 700 分或以下。' if lowest <= 700 else '榜单前 500 尚未下降到 700 分，因此报告止于实际第 500 名。'}</p></section>
<section class="band" id="catalog-distribution"><div class="heading"><div><h2>001–070 exact deck 出现频率</h2></div><p>横轴固定为卡组编号并附代表卡图；绿色为当前榜单主构筑，金色为同选手第二套高分构筑证据。</p></div><div class="legend"><span><i class="p"></i>主构筑</span><span><i class="a"></i>第二套构筑</span></div><div class="chart-scroll"><div class="catalog-chart">{"".join(bars)}</div></div><h3>001–070 主构筑分数层</h3><div class="table-scroll"><table class="meta-table" id="catalog-score-table"><thead><tr><th>Deck</th><th>主构筑人数</th><th>第二套证据</th><th>1000+</th><th>900–1000</th><th>800–900</th><th>700–800</th></tr></thead><tbody>{"".join(exact_band_rows)}</tbody></table></div></section>
<section class="band" id="meta-distribution"><div class="heading"><div><h2>29 Meta 分类分布</h2></div><p>先执行 001–070 exact match；未命中的构筑才使用带权多重集 Jaccard 映射到最接近的注册 Meta，并保留 nearest deck、similarity 与跨 Meta margin。</p></div><div class="table-scroll"><table class="meta-table"><thead><tr><th>Meta</th><th>主构筑人数</th><th>全部构筑证据</th><th>1000+</th><th>900–1000</th><th>800–900</th><th>700–800</th></tr></thead><tbody>{"".join(meta_rows)}</tbody></table></div><h3>非 001–070 构筑编号</h3>{('<div class="table-scroll"><table><thead><tr><th>报告编号</th><th>29 Meta</th><th>出现证据</th><th>最佳排名</th><th>相似度</th><th>最近基准</th><th>1000+</th><th>900–1000</th><th>800–900</th><th>700–800</th><th>hash</th></tr></thead><tbody>'+''.join(unknown_rows)+'</tbody></table></div>') if unknown_rows else '<p class="notice">本次保留的构筑全部 exact 命中 001–070。</p>'}</section>
<section class="band" id="rank-index"><div class="heading"><div><h2>静态排名、分数与提交时间</h2></div><p>时间统一显示为中国标准时间；排名、分数和本列提交时间均来自同一 leaderboard CSV。绑定列说明卡组 submission 在异步抓取时是否仍与榜分一致。</p></div><div class="tools"><input id="player-search" type="search" placeholder="筛选排名、选手、卡组编号或 Meta"><select id="band-filter"><option value="">全部分段</option>{''.join(f'<option value="{band}">{band}</option>' for band in SCORE_BANDS)}</select></div><div class="table-scroll"><table class="rank-table"><thead><tr><th>Rank</th><th>选手</th><th>分数</th><th>分段</th><th>构筑编号</th><th>29 Meta</th><th>榜单提交时间</th><th>卡组证据绑定</th></tr></thead><tbody>{''.join(rank_rows)}</tbody></table></div></section>
<section class="band" id="player-decks"><div class="heading"><div><h2>逐人 exact 60-card deck</h2></div><p>每套均显示自己的 submission 分数与提交时间；第二套构筑不混入榜单主 policy 的人口统计。</p></div>{''.join(details)}</section>
<section class="band" id="evidence-boundary"><div class="heading"><div><h2>证据边界与审计</h2></div></div><div class="evidence-grid"><article><h3>榜单与绑定</h3><p>排名、分数、榜单提交时间来自单个 server-generated CSV。卡组取证耗时更长，因此只有 {audit["strict_score_bindings"]}/{audit["players"]} 行仍可 exact score bind；另 {audit["dynamic_score_bindings"]} 行使用抓取窗口内最高分 active submission，并保留 score delta，不能解读为产生该冻结榜分的唯一 submission。</p></article><article><h3>卡组身份链</h3><p>{audit["players"]}/{audit["players"]} 行均完成 selected submission → PUBLIC + COMPLETED Episode → own player index → exact 60-card replay deck，卡组 identity audit 为 <b>{audit["status"]}</b>。</p></article><article><h3>分类边界</h3><p>001–070 使用完整 exact-deck SHA-256；未知构筑的 29 Meta 是相对当前 70 套注册样本的结构相似度推断，不是官方规则判定或对战强度证据。</p></article><article><h3>双构筑</h3><p>只保留按 publicScore 降序审计时遇到的首个不同 deck hash，最多两套。第二套仅作构筑环境证据。</p></article><article><h3>复现标识</h3><p>captured_at: <code>{html.escape(str(payload["captured_at_utc"]))}</code><br>leaderboard CSV: <code>{html.escape(str(payload.get("leaderboard_capture", {}).get("leaderboard_csv", "n/a")))}</code><br>registry: <code>{reference["registry_sha256"][:16]}</code><br>taxonomy: <code>{reference["taxonomy_sha256"][:16]}</code></p></article></div></section></main><script>{script}</script></body></html>''', encoding="utf-8")


def _default_date() -> str:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=_default_date())
    parser.add_argument("--work", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        parser.error("--date must use YYYY-MM-DD")
    if args.limit < 1 or args.limit > 500:
        parser.error("--limit must be between 1 and 500")
    if args.render_only and args.validate_only:
        parser.error("--render-only and --validate-only are mutually exclusive")
    stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    args.work = args.work or REPOSITORY_ROOT / ".tmp/environment_daily" / args.date / f"{stamp}-top500"
    args.snapshot = args.snapshot or RANKED_ROOT / "data" / f"{args.date}-top500-exact.json"
    args.report = args.report or RANKED_ROOT / f"{args.date}-top500.html"
    for path in (args.snapshot, args.report):
        if path.exists() and not (args.overwrite or args.validate_only):
            parser.error(f"output exists; pass --overwrite explicitly: {path}")
    if args.render_only or args.validate_only:
        payload = json.loads(args.snapshot.read_text(encoding="utf-8"))
        audit = validate_snapshot(payload, expected_count=int(payload["leaderboard_count"]))
        if args.validate_only:
            print(json.dumps(audit, ensure_ascii=False))
            return
    else:
        payload = collect_snapshot(args.work, args.snapshot, limit=args.limit)
    render_report(payload, args.report)
    print(args.report)


if __name__ == "__main__":
    main()
