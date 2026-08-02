"""Build a timestamp-bound Kaggle Top 100 environment snapshot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import html
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import RequestException

COMPETITION = "pokemon-tcg-ai-battle"
CURRENT_USER_TEAM_ID = 16383960
DRAGAPULT_EX = 121
MARNIES_GRIMMSNARL_EX = 648

UI_BASELINE = "2026-07-26.html"
CARD_POOL_BASELINE = "2026-07-25.html"
SNAPSHOT_SCHEMA = "pokemon_tcg_environment_daily_v3"
LEADERBOARD_BINDINGS = {
    "leaderboard_score",
    "leaderboard_score_submission_date",
}

CARD_IMAGE_SET_BY_EXPANSION = {
    "BLK": "zsv10pt5", "DRI": "sv10", "JTG": "sv9", "MEG": "me1",
    "PAL": "sv2", "PFL": "me2", "PRE": "sv8pt5", "SCR": "sv7",
    "SFA": "sv6pt5", "SSP": "sv8", "SVE": "sve", "SVI": "sv1",
    "TEF": "sv5", "TWM": "sv6", "WHT": "rsv10pt5",
}
SCRYDEX_SET_BY_EXPANSION = {"ASC": "me2pt5", "POR": "me3"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DAILY_REPORT_ROOT = REPOSITORY_ROOT / "docs/environment-daily_kaggle_top100/daily"


def _card_catalog() -> dict[int, dict[str, str]]:
    path = Path(__file__).parents[2] / "official" / "EN_Card_Data.csv"
    catalog: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            catalog.setdefault(int(row["Card ID"]), row)
    return catalog


def _rate_call(fn):
    transient_failures = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # Kaggle currently rate-limits team-submission lookups.
            if "429" in str(exc):
                time.sleep(65)
                continue
            if isinstance(exc, RequestException) and transient_failures < 4:
                transient_failures += 1
                time.sleep(5)
                continue
            raise


def _result(episodes, submission_id: int) -> tuple[int, int, int, int]:
    wins = losses = draws = 0
    for episode in episodes:
        player_index = _episode_player_index(episode, submission_id)
        agents = list(_model_value(episode, "agents", default=[]) or [])
        agent = next(
            row
            for fallback_index, row in enumerate(agents)
            if int(_model_value(row, "index", default=fallback_index)) == player_index
        )
        reward = _model_value(agent, "reward")
        if reward is None:
            continue
        if float(reward) > 0:
            wins += 1
        elif float(reward) < 0:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws, wins + losses + draws


def _archetype(deck: list[int], catalog: dict[int, dict[str, str]]) -> str:
    """Name a deck from its primary evolution line plus its strongest distinct partner."""
    counts: dict[int, int] = {card_id: deck.count(card_id) for card_id in set(deck)}
    pokemon = []
    for card_id, count in counts.items():
        card = catalog.get(card_id, {})
        stage = card.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
        if "Pokémon" not in stage:
            continue
        level = 3 if "Stage 2" in stage else 2 if "Stage 1" in stage else 1
        if "ex" in card.get("Card Name", "") or "Mega" in card.get("Card Name", ""):
            level += 1
        pokemon.append((level, count, card_id, card))
    pokemon.sort(key=lambda item: (-item[0], -item[1], item[3].get("Card Name", "")))
    if not pokemon:
        return "Unknown Pokémon"
    primary = pokemon[0]
    ancestors = {primary[3].get("Card Name", "")}
    previous = primary[3].get("Previous stage", "")
    while previous and previous != "n/a":
        ancestors.add(previous)
        previous = next(
            (row.get("Previous stage", "") for row in catalog.values()
             if row.get("Card Name") == previous),
            "",
        )
    partner = next((item for item in pokemon[1:] if item[3].get("Card Name") not in ancestors), None)
    if partner is None:
        return primary[3]["Card Name"]
    return f"{primary[3]['Card Name']} / {partner[3]['Card Name']}"


def _classify_archetype(deck: list[int], catalog: dict[int, dict[str, str]]) -> str:
    names = {catalog.get(card_id, {}).get("Card Name", "") for card_id in deck}
    fixed = (
        ("Marnie's Grimmsnarl ex", "Marnie's Grimmsnarl ex / Froslass"),
        ("Alakazam", "Alakazam / Dudunsparce"),
        ("Team Rocket's Mewtwo ex", "Team Rocket's Mewtwo ex / Spidops"),
        ("Cynthia's Garchomp ex", "Cynthia's Garchomp ex / Roserade"),
        ("Mega Kangaskhan ex", "Mega Kangaskhan ex / Crustle"),
        ("Mega Lopunny ex", "Mega Lopunny ex / Mega Froslass ex"),
        ("Mega Lucario ex", "Mega Lucario ex / Solrock"),
        ("Mega Starmie ex", "Mega Starmie ex / Dusknoir"),
        ("Mega Abomasnow ex", "Mega Abomasnow ex / Kyogre"),
    )
    if "Dragapult ex" in names and "Dusknoir" in names:
        return "Dragapult ex / Dusknoir"
    if "Dragapult ex" in names and "Blaziken ex" in names:
        return "Dragapult ex / Blaziken ex"
    if "Dragapult ex" in names:
        return "Dragapult ex"
    if "Dipplin" in names and "Thwackey" in names:
        return "Festival Lead / Dipplin"
    for card_name, archetype in fixed:
        if card_name in names:
            return archetype
    return _archetype(deck, catalog)


def _datetime(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _model_value(model: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            return value
    return default


def _timestamp(value: object) -> str:
    parsed = _datetime(value)
    return parsed.astimezone(timezone.utc).isoformat() if parsed is not None else str(value)


def _score(value: object) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _select_leaderboard_submission(
    leaderboard_row: object, submissions: list[object]
) -> tuple[object, str]:
    if not submissions:
        raise ValueError("team has no submissions")
    leaderboard_score = _score(_model_value(leaderboard_row, "score"))
    if leaderboard_score is None:
        raise ValueError("leaderboard row has no valid score")
    score_matches = [
        row
        for row in submissions
        if _score(_model_value(row, "public_score", "publicScore")) == leaderboard_score
    ]
    if len(score_matches) == 1:
        return score_matches[0], "leaderboard_score"
    if not score_matches:
        raise ValueError(
            f"cannot bind leaderboard score {leaderboard_score}: no submission matches"
        )
    leaderboard_date = _model_value(leaderboard_row, "submission_date", "submissionDate")
    exact_dates = [
        row
        for row in score_matches
        if _timestamp(_model_value(row, "date_submitted", "dateSubmitted"))
        == _timestamp(leaderboard_date)
    ]
    if len(exact_dates) == 1:
        return exact_dates[0], "leaderboard_score_submission_date"
    leaderboard_datetime = _datetime(leaderboard_date)
    dated_rows = []
    if leaderboard_datetime is not None:
        for row in score_matches:
            submitted_at = _datetime(_model_value(row, "date_submitted", "dateSubmitted"))
            if submitted_at is not None:
                dated_rows.append(
                    (abs((submitted_at - leaderboard_datetime).total_seconds()), row)
                )
    dated_rows.sort(key=lambda item: item[0])
    if (
        dated_rows
        and dated_rows[0][0] <= 2.0
        and (len(dated_rows) == 1 or dated_rows[0][0] < dated_rows[1][0])
    ):
        return dated_rows[0][1], "leaderboard_score_submission_date"
    raise ValueError(
        f"cannot uniquely bind leaderboard score {leaderboard_score} across "
        f"{len(score_matches)} submissions"
    )


def _capture_score_bound_leaderboard(
    api: KaggleApi,
) -> tuple[list[object], dict[int, list[object]], str]:
    """Warm submission views before freezing the final score-bound leaderboard."""
    preliminary = api.competition_leaderboard_view(COMPETITION, page_size=100)[:100]
    if len(preliminary) != 100:
        raise RuntimeError(f"leaderboard returned {len(preliminary)} rows, expected 100")
    submissions_by_team = {
        int(_model_value(row, "team_id", "teamId")): list(
            _rate_call(
                lambda team_id=int(_model_value(row, "team_id", "teamId")): (
                    api.competition_team_submissions(team_id)
                )
            )
        )
        for row in preliminary
    }

    leaderboard = api.competition_leaderboard_view(COMPETITION, page_size=100)[:100]
    captured_at_utc = datetime.now(timezone.utc).isoformat()
    if len(leaderboard) != 100:
        raise RuntimeError(f"leaderboard returned {len(leaderboard)} rows, expected 100")
    for row in leaderboard:
        team_id = int(_model_value(row, "team_id", "teamId"))
        submissions = submissions_by_team.get(team_id, [])
        try:
            _select_leaderboard_submission(row, submissions)
        except ValueError:
            submissions = list(
                _rate_call(lambda: api.competition_team_submissions(team_id))
            )
            _select_leaderboard_submission(row, submissions)
        submissions_by_team[team_id] = submissions
    return leaderboard, submissions_by_team, captured_at_utc


def _episode_agents(episode: object) -> list[dict[str, object]]:
    result = []
    for fallback_index, agent in enumerate(_model_value(episode, "agents", default=[]) or []):
        result.append(
            {
                "index": int(_model_value(agent, "index", default=fallback_index)),
                "submission_id": int(
                    _model_value(agent, "submission_id", "submissionId", default=0) or 0
                ),
            }
        )
    return result


def _episode_player_index(episode: object, submission_id: int) -> int:
    matches = [
        int(row["index"])
        for row in _episode_agents(episode)
        if int(row["submission_id"]) == submission_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"submission {submission_id} is not unique in Episode "
            f"{_model_value(episode, 'id')}: {matches}"
        )
    return matches[0]


def _decks_from_replay(payload: dict[str, object]) -> list[list[int]]:
    for step in payload.get("steps") or []:
        if not isinstance(step, list):
            continue
        for row in step:
            if not isinstance(row, dict):
                continue
            for frame in row.get("visualize") or []:
                actions = frame.get("action") if isinstance(frame, dict) else None
                if (
                    isinstance(actions, list)
                    and len(actions) == 2
                    and all(isinstance(deck, list) and len(deck) == 60 for deck in actions)
                ):
                    return [[int(card_id) for card_id in deck] for deck in actions]
    raise ValueError("replay has no public two-player 60-card initial deck frame")


def _eligible_episodes(api: KaggleApi, submission_id: int, limit: int) -> list[object]:
    episodes = list(api.competition_list_episodes(submission_id) or [])
    eligible = []
    for episode in episodes:
        if "PUBLIC" not in str(_model_value(episode, "type", default="")):
            continue
        if "COMPLETED" not in str(_model_value(episode, "state", default="")):
            continue
        try:
            _episode_player_index(episode, submission_id)
        except ValueError:
            continue
        eligible.append(episode)
    eligible.sort(
        key=lambda row: (
            _datetime(_model_value(row, "create_time", "createTime"))
            or datetime.min.replace(tzinfo=timezone.utc),
            int(_model_value(row, "id")),
        ),
        reverse=True,
    )
    return eligible[:limit]


def _episodes_at_or_before(episodes: list[object], cutoff_value: object) -> list[object]:
    """Return only Episodes whose creation time is inside one frozen snapshot boundary."""
    cutoff = _datetime(cutoff_value)
    if cutoff is None:
        raise ValueError(f"invalid leaderboard capture timestamp: {cutoff_value!r}")
    bounded: list[tuple[datetime, int, object]] = []
    for episode in episodes:
        created_at = _datetime(_model_value(episode, "create_time", "createTime"))
        if created_at is None or created_at > cutoff:
            continue
        bounded.append((created_at, int(_model_value(episode, "id")), episode))
    bounded.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in bounded]


def _report_date(state: dict[str, object]) -> str:
    explicit = str(state.get("report_date") or "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", explicit):
        return explicit
    captured_at = _datetime(state.get("captured_at_utc"))
    if captured_at is None:
        raise ValueError("snapshot has neither a valid report_date nor captured_at_utc")
    return captured_at.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()


def _report_id(report_date: str) -> str:
    return report_date[5:7] + report_date[8:10]


def _previous_report_path(report_date: str) -> Path:
    candidates = sorted(
        path for path in DAILY_REPORT_ROOT.glob("????-??-??.html") if path.stem < report_date
    )
    return candidates[-1] if candidates else DAILY_REPORT_ROOT / UI_BASELINE


def _episode_views(episodes: list[object], submission_id: int) -> list[dict[str, object]]:
    views: list[dict[str, object]] = []
    for episode in episodes:
        player_index = _episode_player_index(episode, submission_id)
        agents = list(_model_value(episode, "agents", default=[]) or [])
        own_matches = [
            agent
            for fallback_index, agent in enumerate(agents)
            if int(_model_value(agent, "index", default=fallback_index)) == player_index
        ]
        if len(own_matches) != 1:
            raise ValueError(f"Episode {_model_value(episode, 'id')} has no unique own agent")
        own = own_matches[0]
        others = [
            agent
            for fallback_index, agent in enumerate(agents)
            if int(_model_value(agent, "index", default=fallback_index)) != player_index
        ]
        if len(others) != 1:
            raise ValueError(f"Episode {_model_value(episode, 'id')} is not a two-player game")
        other = others[0]
        created_at = _datetime(_model_value(episode, "create_time", "createTime"))
        ended_at = _datetime(_model_value(episode, "end_time", "endTime"))
        views.append(
            {
                "episode_id": int(_model_value(episode, "id")),
                "submission_id": submission_id,
                "reward": _model_value(own, "reward"),
                "player_index": player_index,
                "other": {
                    "index": int(_model_value(other, "index", default=1 - player_index)),
                    "submission_id": int(
                        _model_value(other, "submission_id", "submissionId", default=0) or 0
                    ),
                    "team_id": int(
                        _model_value(other, "team_id", "teamId", default=0) or 0
                    ),
                    "team_name": str(
                        _model_value(other, "team_name", "teamName", default="") or ""
                    ),
                    "reward": _model_value(other, "reward"),
                },
                "create_time": created_at.isoformat() if created_at else None,
                "end_time": ended_at.isoformat() if ended_at else None,
            }
        )
    return views


def _validate_snapshot(
    state: dict[str, object],
    meta_payload: dict[str, object] | None,
    replay_root: Path,
) -> dict[str, object]:
    """Fail closed unless all 100 leaderboard → submission → replay → deck chains agree."""
    cutoff = _datetime(state.get("captured_at_utc"))
    rows = list(state.get("rows") or [])
    players = dict(state.get("players") or {})
    views_by_submission = dict((meta_payload or {}).get("views") or {})
    if cutoff is None or len(rows) != 100 or len(players) != 100:
        raise ValueError("snapshot must contain one valid cutoff, 100 rows, and 100 players")
    if state.get("schema") == SNAPSHOT_SCHEMA:
        stabilization = dict(state.get("stabilization") or {})
        if int(stabilization.get("consecutive_stable_sweeps") or 0) < 2:
            raise ValueError("snapshot has not reached two stable full Meta sweeps")
    selected: list[dict[str, object]] = []
    for expected_rank, row in enumerate(rows, 1):
        player = players.get(str(expected_rank))
        if not isinstance(player, dict):
            raise ValueError(f"rank {expected_rank}: missing player record")
        for field in ("rank", "team_id", "team_name", "score", "submission_date"):
            if player.get(field) != row.get(field):
                raise ValueError(f"rank {expected_rank}: leaderboard field {field} diverged")
        if player.get("binding") not in LEADERBOARD_BINDINGS:
            raise ValueError(f"rank {expected_rank}: leaderboard score was not uniquely bound")
        if _score(player.get("submission_public_score")) != _score(row.get("score")):
            raise ValueError(f"rank {expected_rank}: submission publicScore diverged")
        submission_id = int(player.get("submission_id") or 0)
        bounded_views = [
            view
            for view in views_by_submission.get(str(submission_id), [])
            if _datetime(view.get("create_time")) is not None
            and _datetime(view.get("create_time")) <= cutoff
        ]
        if not bounded_views:
            raise ValueError(f"rank {expected_rank}: no bounded PUBLIC + COMPLETED Episode")
        latest = max(
            bounded_views,
            key=lambda view: (_datetime(view["create_time"]), int(view["episode_id"])),
        )
        episode_id = int(player.get("episode_id") or 0)
        player_index = int(player.get("episode_player_index", -1))
        if episode_id != int(latest["episode_id"]) or player_index != int(latest["player_index"]):
            raise ValueError(f"rank {expected_rank}: representative Episode is not cutoff-latest")
        if _datetime(player.get("episode_create_time")) != _datetime(latest.get("create_time")):
            raise ValueError(f"rank {expected_rank}: representative Episode time diverged")
        if int(latest.get("submission_id", submission_id)) != submission_id:
            raise ValueError(f"rank {expected_rank}: Episode submission identity diverged")
        replay_path = replay_root / f"episode-{episode_id}.json"
        if not replay_path.is_file():
            raise ValueError(f"rank {expected_rank}: missing replay {episode_id}")
        decks = _decks_from_replay(json.loads(replay_path.read_text(encoding="utf-8")))
        deck = [int(card_id) for card_id in player.get("deck") or []]
        if player_index not in (0, 1) or len(deck) != 60 or sorted(decks[player_index]) != sorted(deck):
            raise ValueError(f"rank {expected_rank}: replay player index or exact deck diverged")
        deck_sha256 = hashlib.sha256(
            ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
        ).hexdigest()
        if player.get("deck_sha256") != deck_sha256:
            raise ValueError(f"rank {expected_rank}: deck hash diverged")
        rewards = [float(view["reward"]) for view in bounded_views if view.get("reward") is not None]
        expected_result = (
            sum(reward > 0 for reward in rewards),
            sum(reward < 0 for reward in rewards),
            sum(reward == 0 for reward in rewards),
        )
        actual_result = (player.get("wins"), player.get("losses"), player.get("draws"))
        if actual_result != expected_result or player.get("valid_games") != len(rewards):
            raise ValueError(f"rank {expected_rank}: bounded win/loss/draw statistics diverged")
        expected_win_rate = (
            (expected_result[0] + 0.5 * expected_result[2]) / len(rewards)
            if rewards
            else None
        )
        if player.get("win_rate") != expected_win_rate:
            raise ValueError(f"rank {expected_rank}: bounded win rate diverged")
        selected.append(
            {
                "rank": expected_rank,
                "team_name": player["team_name"],
                "leaderboard_score": str(row["score"]),
                "submission_id": submission_id,
                "submission_public_score": player["submission_public_score"],
                "episode_id": episode_id,
                "episode_create_time": latest["create_time"],
                "episode_player_index": player_index,
                "deck_sha256": deck_sha256,
            }
        )
    return {
        "contract": (
            "latest PUBLIC + COMPLETED Episode for the exact leaderboard submission "
            "at or before captured_at_utc"
        ),
        "audited_players": 100,
        "all_episode_times_at_or_before_capture": True,
        "all_submission_matches_unique": True,
        "all_leaderboard_scores_match": True,
        "all_decks_exactly_60": True,
        "bounded_player_views": sum(
            int(player.get("valid_games") or 0) for player in players.values()
        ),
        "stabilization": state.get("stabilization"),
        "selected": selected,
    }


def _card_group(row: dict[str, str]) -> str:
    kind = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "").strip()
    category = row.get("Category", "").strip()
    if kind.endswith("Energy"):
        return "Energy"
    if kind.endswith("Pokémon") and category not in {"Fossil", "Technical Machine"}:
        return "Pokémon"
    return "Trainer"


def _image_url(row: dict[str, str] | None, *, hires: bool = False) -> str:
    if not row:
        return ""
    number = row["Collection No."]
    scrydex_set = SCRYDEX_SET_BY_EXPANSION.get(row["Expansion"])
    if scrydex_set:
        return f"https://images.scrydex.com/pokemon/{scrydex_set}-{number}/{'large' if hires else 'small'}"
    set_id = CARD_IMAGE_SET_BY_EXPANSION.get(row["Expansion"])
    if not set_id:
        return ""
    return f"https://images.pokemontcg.io/{set_id}/{number}{'_hires' if hires else ''}.png"


def _card_thumb(card_id: int, catalog: dict[int, dict[str, str]], *, compact: bool) -> str:
    row = catalog.get(card_id)
    name = row["Card Name"] if row else f"Card ID {card_id}"
    source = _image_url(row)
    preview = _image_url(row, hires=True)
    cls = "card-thumb compact" if compact else "card-thumb"
    image = (
        f'<img src="{html.escape(source)}" alt="{html.escape(name)}" loading="lazy" '
        f'width="{20 if compact else 30}" height="{28 if compact else 42}" '
        f'style="width:{20 if compact else 30}px;height:{28 if compact else 42}px;'
        f'max-width:{20 if compact else 30}px;max-height:{28 if compact else 42}px" '
        'decoding="async" referrerpolicy="no-referrer" '
        'onerror="this.hidden=true;this.parentElement.classList.add(\'thumb-failed\')">'
        if source else ""
    )
    return (
        f'<span class="{cls}" tabindex="0" data-card-name="{html.escape(name)}" '
        f'data-card-preview-url="{html.escape(preview)}" title="{html.escape(name)} · Card ID {card_id}">'
        f'{image}<span class="thumb-fallback">ID {card_id}<small>{html.escape(name)}</small></span></span>'
    )


def _deck_hash(deck: list[int]) -> str:
    encoded = ",".join(str(card_id) for card_id in sorted(deck)).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def _same_deck_hash(left: str, right: str) -> bool:
    left = left.strip().lower()
    right = right.strip().lower()
    return bool(left and right) and (left.startswith(right) or right.startswith(left))


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def _baseline_css(report: Path) -> str:
    baseline = DAILY_REPORT_ROOT / UI_BASELINE
    match = re.search(r"<style>(.*?)</style>", baseline.read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError(f"missing style block in UI baseline: {baseline}")
    return match.group(1)


def _previous_players(report_date: str) -> dict[str, dict[str, str | int | float]]:
    text = _previous_report_path(report_date).read_text(encoding="utf-8")
    result: dict[str, dict[str, str | int | float]] = {}
    for row in re.findall(r'<tr data-index-row.*?</tr>', text, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.S)
        if len(cells) < 8:
            continue
        plain = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", cell)).strip() for cell in cells]
        rank = re.search(r"#(\d+)", plain[0])
        team_name = re.search(r'<a href="#player-\d+">(.*?)</a>', row, re.S)
        archetype = re.search(r'data-archetype="([^"]*)"', row)
        deck_hash = re.search(r"([0-9a-f]{10,12})", plain[-1])
        if rank and team_name:
            name = html.unescape(re.sub(r"<[^>]+>", "", team_name.group(1))).strip()
            result[name] = {
                "rank": int(rank.group(1)),
                "archetype": html.unescape(archetype.group(1)) if archetype else "",
                "deck_hash": deck_hash.group(1) if deck_hash else "",
            }
    return result


def _render_report(
    state: dict[str, object],
    players: list[dict[str, object]],
    catalog: dict[int, dict[str, str]],
    report: Path,
    meta_payload: dict[str, object] | None = None,
    replay_root: Path | None = None,
) -> None:
    report_date = _report_date(state)
    report_id = _report_id(report_date)
    previous_report = _previous_report_path(report_date)
    previous_date = previous_report.stem
    if replay_root is not None:
        state["identity_audit"] = _validate_snapshot(state, meta_payload, replay_root)
    report.parent.mkdir(parents=True, exist_ok=True)
    css = _baseline_css(report) + """
    .snapshot-note{padding:16px 18px;border:1px solid #e5bb76;border-radius:14px;background:#fff8e9;color:#5e4218}
    .archetype-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.archetype-card{padding:15px;border:1px solid var(--line);border-radius:14px;background:#fff}.archetype-card h3{margin:0 0 8px}.bar-track{height:9px;border-radius:99px;background:#e8eee9;overflow:hidden}.bar-fill{height:100%;background:linear-gradient(90deg,#1e7c60,#8bbf9a)}
    .deck-group{margin:18px 0 28px}.deck-group h4{padding-bottom:7px;border-bottom:2px solid #a8cdbd}.card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:12px}.card-tile{min-width:0}.card-tile .card-art{position:relative;aspect-ratio:2.5/3.5;overflow:hidden;border-radius:8px;background:#edf2ef}.card-tile .card-thumb{width:100%!important;height:100%!important;min-width:0!important;max-width:none!important;min-height:0!important;max-height:none!important}.card-tile .card-thumb img{width:100%!important;height:100%!important;object-fit:contain!important}.card-tile b,.card-tile small{display:block}.card-tile b{margin-top:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.card-tile small{color:var(--muted)}.deck-count{position:absolute;z-index:4;top:7px;right:7px;padding:3px 7px;border-radius:999px;color:#fff;background:#173e31e8;font-weight:900}
    .archetype-profile{margin:10px 0;border:1px solid var(--line);border-radius:14px}.archetype-profile summary{display:flex;justify-content:space-between;gap:12px;padding:15px;cursor:pointer;font-weight:800}.archetype-profile>div{padding:0 15px 15px}.tag-list{display:flex;flex-wrap:wrap;gap:7px}.tag{padding:5px 8px;border-radius:999px;background:#edf6f1;color:#0d6349;font-size:12px}.unavailable-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.unavailable-grid article{padding:16px;border:1px dashed #c9d2cd;border-radius:14px;background:#f7f8f7}.comparison-table small,.summary-table small{display:block;color:var(--muted)}
    .current-user{outline:2px solid #c88b19;outline-offset:-2px;background:#fffaf0!important}.current-user-badge{display:inline-flex;align-items:center;margin-left:7px;padding:3px 7px;border:1px solid #d8aa54;border-radius:999px;background:#fff0c9;color:#754b08;font-size:11px;font-weight:900;vertical-align:middle}.current-user-banner{display:flex;flex-wrap:wrap;align-items:center;gap:9px 16px;margin:16px 0;padding:15px 18px;border:2px solid #c88b19;border-radius:12px;background:#fffaf0}.current-user-banner b{font-size:18px}.current-user-banner span{color:#5f584c}.current-user-banner a{margin-left:auto;font-weight:850}
    @media(max-width:900px){.archetype-grid,.unavailable-grid{grid-template-columns:1fr 1fr}}@media(max-width:680px){.archetype-grid,.unavailable-grid{grid-template-columns:1fr}}
    """
    css = "\n".join(line.rstrip() for line in css.splitlines())
    for player in players:
        player["archetype"] = _classify_archetype(player["deck"], catalog)
        player["deck_hash"] = _deck_hash(player["deck"])
        player["card_counts"] = Counter(int(card_id) for card_id in player["deck"])
    current_user = next(
        (player for player in players if int(player["team_id"]) == CURRENT_USER_TEAM_ID),
        None,
    )

    by_submission = {int(player["submission_id"]): player for player in players}
    cutoff = _datetime(state["captured_at_utc"])
    views_by_submission: dict[int, list[dict[str, object]]] = {}
    for submission_id, player in by_submission.items():
        raw_views = (meta_payload or {}).get("views", {}).get(str(submission_id), [])
        bounded_views = []
        for view in raw_views:
            created_at = _datetime(view.get("create_time"))
            if cutoff is not None and created_at is not None and created_at > cutoff:
                continue
            bounded_views.append(view)
        views_by_submission[submission_id] = bounded_views
        if bounded_views:
            rewards = [float(view["reward"]) for view in bounded_views if view.get("reward") is not None]
            player["wins"] = sum(reward > 0 for reward in rewards)
            player["losses"] = sum(reward < 0 for reward in rewards)
            player["draws"] = sum(reward == 0 for reward in rewards)
            player["valid_games"] = len(rewards)
            player["win_rate"] = (
                (player["wins"] + 0.5 * player["draws"]) / len(rewards) if rewards else None
            )
        high_rewards: list[float] = []
        other_rewards: list[float] = []
        for view in bounded_views:
            reward = view.get("reward")
            if reward is None:
                continue
            other = view.get("other") or {}
            opponent_submission = int(other.get("submission_id") or 0)
            (high_rewards if opponent_submission in by_submission else other_rewards).append(float(reward))
        player["high_rewards"] = high_rewards
        player["other_rewards"] = other_rewards

    distribution = Counter(str(player["archetype"]) for player in players)
    presence: Counter[int] = Counter()
    archetype_presence: dict[str, Counter[int]] = defaultdict(Counter)
    archetype_decks: dict[str, list[Counter[int]]] = defaultdict(list)
    for player in players:
        unique_ids = set(player["deck"])
        presence.update(unique_ids)
        archetype_presence[str(player["archetype"])].update(unique_ids)
        archetype_decks[str(player["archetype"])].append(player["card_counts"])

    ordered_archetypes = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    archetype_names = [name for name, _ in ordered_archetypes]
    archetype_matrix: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    player_matrix: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    top100_player_views = 0
    for submission_id, views in views_by_submission.items():
        player = by_submission[submission_id]
        for view in views:
            other = view.get("other") or {}
            opponent = by_submission.get(int(other.get("submission_id") or 0))
            reward = view.get("reward")
            if opponent is None or reward is None:
                continue
            result = "wins" if float(reward) > 0 else "losses" if float(reward) < 0 else "draws"
            archetype_matrix[(str(player["archetype"]), str(opponent["archetype"]))][result] += 1
            player_matrix[(str(player["team_name"]), str(opponent["team_name"]))][result] += 1
            top100_player_views += 1
    capped_submissions = sum(
        bool(row.get("possibly_censored_at_1000"))
        for row in (meta_payload or {}).get("audit", {}).values()
    )
    audit_json = json.dumps(
        state.get("identity_audit") or {}, ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    previous = _previous_players(report_date)
    current_names = {str(player["team_name"]) for player in players}
    joined = [player for player in players if player["team_name"] in previous]
    entered = [player for player in players if player["team_name"] not in previous]
    exited = sorted(set(previous) - current_names)
    changed = [
        player for player in joined
        if not _same_deck_hash(
            str(previous[str(player["team_name"])]["deck_hash"]),
            str(player["deck_hash"]),
        )
    ]

    total_games = sum(int(player["valid_games"]) for player in players)
    total_wins = sum(int(player["wins"]) for player in players)
    total_losses = sum(int(player["losses"]) for player in players)
    total_draws = sum(int(player["draws"]) for player in players)
    overall = (total_wins + 0.5 * total_draws) / total_games if total_games else None

    representative_ids: dict[str, list[int]] = {}
    for name, _ in ordered_archetypes:
        selected: list[int] = []
        requested_names = [part.strip() for part in name.split("/")]
        aliases = {
            "Spidops": "Team Rocket's Spidops",
            "Roserade": "Cynthia's Roserade",
            "Festival Lead": "Dipplin",
        }
        for requested in requested_names:
            requested = aliases.get(requested, requested)
            match = next(
                (
                    card_id for card_id in archetype_presence[name]
                    if catalog.get(card_id, {}).get("Card Name") == requested
                ),
                None,
            )
            if match is not None and match not in selected:
                selected.append(match)
        candidates: list[tuple[int, int, int]] = []
        for card_id, deck_count in archetype_presence[name].items():
            row = catalog.get(card_id, {})
            if _card_group(row) != "Pokémon":
                continue
            stage = row.get("Stage (Pokémon)/Type (Energy and Trainer)", "")
            priority = 3 if "Stage 2" in stage else 2 if "Stage 1" in stage else 1
            if " ex" in row.get("Card Name", "") or row.get("Card Name", "").startswith("Mega "):
                priority += 2
            candidates.append((priority, deck_count, card_id))
        candidates.sort(reverse=True)
        selected.extend(
            card_id for _, _, card_id in candidates if card_id not in selected
        )
        representative_ids[name] = selected[:2]

    def archetype_visual(name: str, *, compact: bool = True) -> str:
        thumbs = "".join(
            _card_thumb(card_id, catalog, compact=compact)
            for card_id in representative_ids.get(name, [])
        )
        return (
            f'<span class="archetype-visual" title="{html.escape(name)}">'
            f'<span class="archetype-thumbs">{thumbs}</span>'
            f'<span class="archetype-label">{html.escape(name)}</span></span>'
        )

    def reward_rate(rewards: list[float]) -> float | None:
        if not rewards:
            return None
        return (sum(reward > 0 for reward in rewards) + 0.5 * sum(reward == 0 for reward in rewards)) / len(rewards)

    def matrix_cell(counter: Counter[str] | None) -> str:
        counter = counter or Counter()
        n = sum(counter.values())
        if not n:
            return '<td class="no-sample">—<small>n=0</small></td>'
        rate = (counter["wins"] + 0.5 * counter["draws"]) / n
        hue = max(6.0, min(145.0, 8.0 + rate * 137.0))
        return (
            f'<td style="background:hsl({hue:.1f} 56% 89%)" '
            f'title="真实 {report_id} 玩家视角，n={n}">'
            f'<b>{_pct(rate)}</b><small>{counter["wins"]}-{counter["losses"]}-{counter["draws"]} · n={n}</small></td>'
        )

    matrix_head = "".join(f'<th>{archetype_visual(name)}</th>' for name in archetype_names)
    archetype_matrix_rows = "".join(
        f'<tr><th>{archetype_visual(left)}</th>'
        + "".join(matrix_cell(archetype_matrix.get((left, right))) for right in archetype_names)
        + "</tr>"
        for left in archetype_names
    )

    def badge(player: dict[str, object]) -> str:
        return f'<span class="archetype-badge visual-badge">{archetype_visual(str(player["archetype"]))}</span>'

    def is_current_user(player: dict[str, object]) -> bool:
        return int(player["team_id"]) == CURRENT_USER_TEAM_ID

    def current_user_attributes(player: dict[str, object]) -> str:
        return ' class="current-user" data-current-user="true"' if is_current_user(player) else ""

    def current_user_badge(player: dict[str, object]) -> str:
        return '<span class="current-user-badge">我的位置</span>' if is_current_user(player) else ""

    def player_row(player: dict[str, object], attribute: str) -> str:
        search = f'{player["rank"]} {player["team_name"]} {player["team_id"]} {player["archetype"]}'.lower()
        return (
            f'<tr {attribute}{current_user_attributes(player)} data-archetype="{html.escape(str(player["archetype"]))}" '
            f'data-search="{html.escape(search)}"><td><b>#{player["rank"]}</b></td>'
            f'<td><a href="#player-{int(player["rank"]):03d}">{html.escape(str(player["team_name"]))}</a>{current_user_badge(player)}'
            f'<small>Team {player["team_id"]} · submission {player["submission_id"]}</small></td>'
            f'<td>{badge(player)}</td><td>{float(player["score"]):.1f}</td>'
            f'<td>{int(player["valid_games"]):,}</td><td>{player["wins"]}-{player["losses"]}-{player["draws"]}</td>'
            f'<td><b>{_pct(player["win_rate"])}</b></td><td><code>{player["deck_hash"]}</code>'
            f'<small>Episode {player["episode_id"]} · P{player.get("episode_player_index", "—")}'
            f' · {html.escape(str(player.get("episode_create_time") or "时间未记录"))}</small></td></tr>'
        )

    archetype_cards = []
    max_count = max(distribution.values(), default=1)
    for name, count in ordered_archetypes:
        ids = archetype_presence[name].most_common(5)
        tags = "".join(
            f'<span class="tag">{html.escape(catalog[card_id]["Card Name"])} · {n}/{count}</span>'
            for card_id, n in ids if card_id in catalog
        )
        archetype_cards.append(
            f'<article class="archetype-card"><h3>{archetype_visual(name, compact=False)} <small>{count} 人</small></h3>'
            f'<p>{_pct(count / 100)} · 代表 replay exact deck</p>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{100 * count / max_count:.1f}%"></div></div>'
            f'<div class="tag-list">{tags}</div></article>'
        )

    comparison_rows = []
    for player in joined:
        old = previous[str(player["team_name"])]
        is_changed = not _same_deck_hash(str(old["deck_hash"]), str(player["deck_hash"]))
        comparison_rows.append(
            f'<tr class="{"changed-deck" if is_changed else "same-deck"}"><td><b>{html.escape(str(player["team_name"]))}</b></td>'
            f'<td>#{old["rank"]} → <b>#{player["rank"]}</b></td>'
            f'<td>{html.escape(str(old["archetype"]))} → <b>{html.escape(str(player["archetype"]))}</b></td>'
            f'<td><span class="status {"provisional" if is_changed else "strict"}">{"卡组变化" if is_changed else "卡组不变"}</span>'
            f'<small>{old["deck_hash"] or "—"} → {player["deck_hash"]}</small></td></tr>'
        )

    person_cards = []
    for player in players:
        high_rate = reward_rate(player["high_rewards"])
        other_rate = reward_rate(player["other_rewards"])
        current_user_class = " current-user" if is_current_user(player) else ""
        current_user_data = 'data-current-user="true" ' if is_current_user(player) else ""

        def rate_card(label: str, rate: float | None, n: int) -> str:
            rate_class = "good" if rate is not None and rate >= 0.60 else "bad" if rate is not None and rate < 0.45 else "mid"
            return f'<div class="rate {rate_class}"><span>{label}</span><b>{_pct(rate)}</b><small>n={n:,}</small></div>'

        search = f'{player["rank"]} {player["team_name"]} {player["team_id"]} {player["archetype"]}'.lower()
        person_cards.append(
            f'<article class="person-card{current_user_class}" data-person-card '
            f'{current_user_data}'
            f'data-archetype="{html.escape(str(player["archetype"]))}" '
            f'data-search="{html.escape(search)}"><div class="person-head"><span class="rank-chip">#{player["rank"]}</span>'
            f'<div><a href="#player-{int(player["rank"]):03d}"><b>{html.escape(str(player["team_name"]))}</b></a>'
            f'{current_user_badge(player)}{archetype_visual(str(player["archetype"]))}</div></div><div class="win-grid">'
            f'{rate_card("全样本", player["win_rate"], int(player["valid_games"]))}'
            f'{rate_card("高分段 · Top 100 对手", high_rate, len(player["high_rewards"]))}'
            f'{rate_card("低分段 · 其余对手", other_rate, len(player["other_rewards"]))}</div>'
            f'<div class="person-foot"><span>{player["wins"]}-{player["losses"]}-{player["draws"]}</span>'
            f'<span>submission {player["submission_id"]}</span><span>deck {player["deck_hash"]}</span></div></article>'
        )

    top20 = players[:20]
    top20_names = [str(player["team_name"]) for player in top20]
    top20_head = "".join(
        f'<th title="{html.escape(name)}">{index}<small>{html.escape(name[:12])}</small></th>'
        for index, name in enumerate(top20_names, 1)
    )
    top20_matrix_rows = "".join(
        f'<tr><th>{html.escape(left[:18])}</th>'
        + "".join(
            '<td class="no-sample">—</td>' if left == right
            else matrix_cell(player_matrix.get((left, right)))
            for right in top20_names
        )
        + "</tr>"
        for left in top20_names
    )
    top20_rows = "".join(
        f'<tr><td>#{player["rank"]}</td><td><a href="#player-{int(player["rank"]):03d}">{html.escape(str(player["team_name"]))}</a></td>'
        f'<td>{archetype_visual(str(player["archetype"]))}</td><td>{player["valid_games"]}</td>'
        f'<td>{player["wins"]}-{player["losses"]}-{player["draws"]}</td><td>{_pct(player["win_rate"])}</td></tr>'
        for player in top20
    )

    investment_hist: Counter[int] = Counter()
    for player in players:
        investment_hist.update(player["card_counts"].values())
    investment_html = "".join(
        f'<div><b>×{copies}</b><span>{deck_cards:,} 次 deck-card 投入</span></div>'
        for copies, deck_cards in sorted(investment_hist.items())
    )
    pool_cards = []
    for card_id, deck_count in presence.most_common():
        row = catalog.get(card_id, {})
        name = row.get("Card Name", f"Card ID {card_id}")
        investments = [int(player["card_counts"].get(card_id, 0)) for player in players]
        used = [count for count in investments if count]
        count_hist = Counter(used)
        distribution_text = " · ".join(
            f"×{copies}: {decks}套" for copies, decks in sorted(count_hist.items())
        )
        facts = " · ".join(
            value for value in (
                row.get("Stage (Pokémon)/Type (Energy and Trainer)", ""),
                f'{row.get("Expansion", "")} {row.get("Collection No.", "")}'.strip(),
            ) if value
        )
        pool_cards.append(
            f'<tr data-pool-card><td><span class="pool-card">{_card_thumb(card_id, catalog, compact=False)}'
            f'<span><b>{html.escape(name)}</b><small>ID {card_id}</small></span></span></td>'
            f'<td><small class="card-facts">{html.escape(facts)}</small></td>'
            f'<td><b>{deck_count}</b> / 100</td><td>{_pct(deck_count / 100)}</td>'
            f'<td>{sum(used)}</td><td>{statistics.mean(used):.2f}</td>'
            f'<td>{statistics.median(used):.1f}</td><td>{min(used)}–{max(used)}</td>'
            f'<td><small>{distribution_text}</small></td></tr>'
        )

    archetype_profiles = []
    for name, count in ordered_archetypes:
        deck_counters = archetype_decks[name]
        typical = [item for item in archetype_presence[name].most_common() if item[1] / count >= 0.5]
        typical_rows = []
        for card_id, deck_count in typical:
            row = catalog.get(card_id, {})
            card_name = row.get("Card Name", f"Card ID {card_id}")
            investments = [deck[card_id] for deck in deck_counters if deck.get(card_id)]
            mode = Counter(investments).most_common(1)[0][0]
            typical_rows.append(
                f'<tr><td><span class="pool-card">{_card_thumb(card_id, catalog, compact=False)}'
                f'<span><b>{html.escape(card_name)}</b><small>ID {card_id}</small></span></span></td>'
                f'<td>{_pct(deck_count / count)}</td><td>{mode}</td>'
                f'<td>{statistics.median(investments):.1f}</td><td>{min(investments)}–{max(investments)}</td></tr>'
            )
        exceptions = []
        for card_id, deck_count in sorted(archetype_presence[name].items(), key=lambda item: (item[1], item[0])):
            if deck_count / count > 0.2:
                continue
            card_name = catalog.get(card_id, {}).get("Card Name", f"Card ID {card_id}")
            exceptions.append(f"{card_name} ({deck_count})")
        archetype_profiles.append(
            f'<details class="archetype-profile"><summary>{archetype_visual(name, compact=False)}'
            f'<span>{count} 人 · {len(archetype_presence[name])} 张局部卡池 · {len(typical)} 张典型卡</span></summary>'
            f'<div class="profile-grid"><div><h4>典型投入（牌型内覆盖 ≥ 50%）</h4>'
            f'<div class="table-scroll"><table class="compact"><thead><tr><th>卡</th><th>覆盖</th><th>众数</th><th>中位</th><th>范围</th></tr></thead>'
            f'<tbody>{"".join(typical_rows)}</tbody></table></div></div><div class="audit-verdict">'
            f'<h4>个人特例 / 低频卡</h4><p>{html.escape("、".join(exceptions[:24]) or "没有 ≤20% 覆盖的低频卡")}</p>'
            f'<p class="compact-note">只描述当前代表 deck，不推断单卡因果强度。</p></div></div></details>'
        )

    detail_blocks = []
    for player in players:
        groups: dict[str, list[str]] = {"Pokémon": [], "Trainer": [], "Energy": []}
        totals: Counter[str] = Counter()
        for card_id, count in sorted(player["card_counts"].items(), key=lambda item: catalog.get(item[0], {}).get("Card Name", "")):
            row = catalog.get(card_id, {})
            group = _card_group(row)
            totals[group] += count
            name = row.get("Card Name", f"Card ID {card_id}")
            groups[group].append(
                f'<article class="card-tile"><div class="card-art">{_card_thumb(card_id, catalog, compact=False)}'
                f'<span class="deck-count">×{count}</span></div><b title="{html.escape(name)}">{html.escape(name)}</b>'
                f'<small>Card ID {card_id} · {count} 张</small></article>'
            )
        deck_html = "".join(
            f'<section class="deck-group"><h4>{group} <small>{totals[group]} 张 · {len(groups[group])} 种</small></h4>'
            f'<div class="card-grid">{"".join(groups[group])}</div></section>'
            for group in ("Pokémon", "Trainer", "Energy")
        )
        matchup_items = [
            (right, counter)
            for (left, right), counter in player_matrix.items()
            if left == str(player["team_name"])
        ]
        matchup_items.sort(key=lambda item: (-sum(item[1].values()), item[0]))
        matchup_rows = []
        for opponent_name, counter in matchup_items[:12]:
            opponent = next(item for item in players if item["team_name"] == opponent_name)
            n = sum(counter.values())
            rate = (counter["wins"] + 0.5 * counter["draws"]) / n
            matchup_rows.append(
                f'<tr><td>{html.escape(opponent_name)}</td><td>{archetype_visual(str(opponent["archetype"]))}</td>'
                f'<td>{counter["wins"]}-{counter["losses"]}-{counter["draws"]}</td><td>{_pct(rate)}</td><td>{n}</td></tr>'
            )
        matchup_html = (
            '<div class="table-scroll"><table class="wide-table"><thead><tr><th>对手</th><th>牌型</th><th>W-L-D</th><th>胜率</th><th>n</th></tr></thead>'
            f'<tbody>{"".join(matchup_rows)}</tbody></table></div>'
            if matchup_rows else '<p class="empty">冻结边界内没有当前 Top100 最终 submission 的直接对局。</p>'
        )
        search = f'{player["rank"]} {player["team_name"]} {player["team_id"]} {player["archetype"]}'.lower()
        current_user_class = " current-user" if is_current_user(player) else ""
        current_user_data = 'data-current-user="true" ' if is_current_user(player) else ""
        detail_blocks.append(
            f'<details class="person-card{current_user_class}" '
            f'id="player-{int(player["rank"]):03d}" data-player-detail '
            f'{current_user_data}'
            f'data-archetype="{html.escape(str(player["archetype"]))}" data-search="{html.escape(search)}">'
            f'<summary><span><b>#{player["rank"]} {html.escape(str(player["team_name"]))}</b>'
            f'{current_user_badge(player)} · {badge(player)}</span>'
            f'<span>{player["wins"]}-{player["losses"]}-{player["draws"]} · {_pct(player["win_rate"])}</span></summary>'
            f'<div class="detail-body"><p><b>submission {player["submission_id"]}</b> · Episode {player["episode_id"]} · '
            f'player index {player.get("episode_player_index", "—")} · '
            f'createTime {html.escape(str(player.get("episode_create_time") or "未记录"))} · '
            f'deck hash <code>{player["deck_hash"]}</code></p><h4>已识别 matchup</h4>{matchup_html}'
            f'<h4>Exact 60-card deck</h4>{deck_html}</div></details>'
        )

    current_user_banner = ""
    if current_user is not None:
        current_user_banner = (
            '<aside class="current-user-banner" data-current-user="true">'
            '<span class="current-user-badge">我的位置</span>'
            f'<b>#{current_user["rank"]} {html.escape(str(current_user["team_name"]))}</b>'
            f'<span>榜分 {float(current_user["score"]):.1f} · submission '
            f'{current_user["submission_id"]} · {html.escape(str(current_user["archetype"]))}</span>'
            f'<a href="#player-{int(current_user["rank"]):03d}">查看完整构筑与对局</a></aside>'
        )

    report_html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Top 100 今日环境快照 · {report_id} · {report_date}</title><style>{css}</style></head>
<body id="top"><main class="page">
<nav class="section-nav" aria-label="{report_id} 完整分析导航"><a href="#new-summary">新版总览</a><a href="#personal-winrates">个人胜率</a><a href="#construction-distribution">构筑分布</a><a href="#snapshot-comparison">跨日变化</a><a href="#matchup-boundary">Match-up</a><a href="#card-pool">构筑卡池</a><a href="#archetype-builds">牌型卡池</a><a href="#rank-index">静态排名</a><a href="#player-details">逐人展开</a></nav>

<section class="panel" id="new-summary"><div class="heading"><div><p class="eyebrow">{report_date} SNAPSHOT · {report_id}</p><h2>新版环境分析总览</h2></div><p>整体组件与 UI 固定继承 0726；构筑卡池固定继承 0725 的卡图网格与覆盖率表达。</p></div>
{current_user_banner}
<div class="metrics new-metrics"><div class="metric"><b>100</b><span>最终 submissions</span></div><div class="metric"><b>{total_games:,}</b><span>公开 Meta 玩家视角</span></div><div class="metric"><b>{len(presence)}</b><span>Card Pool 并集</span></div><div class="metric"><b>{len(distribution)}</b><span>实际牌型</span></div><div class="metric"><b>100</b><span>exact decks 已审计</span></div><div class="metric"><b>100</b><span>代表 replays</span></div></div>
<div class="evidence-grid"><article><b>榜单层</b><p>冻结官方 Top100，以 score 匹配 submission publicScore；同分才用 submissionDate 消歧。</p></article><article><b>Meta 层</b><p>只计 exact submission 的 PUBLIC + COMPLETED Episode Meta。</p></article><article><b>Replay 层</b><p>逐 submission 最新合格 replay 的 exact 60-card deck。</p></article></div>
<h3>读数摘要</h3><div class="finding-grid"><article><b>榜首与分差</b><p>{html.escape(str(players[0]["team_name"]))} · {float(players[0]["score"]):.1f}；第 100 名 {html.escape(str(players[-1]["team_name"]))} · {float(players[-1]["score"]):.1f}。</p></article><article><b>公开 Meta</b><p>{total_wins:,}-{total_losses:,}-{total_draws:,}，玩家视角胜率 {_pct(overall)}（n={total_games:,}）。</p></article><article><b>环境集中度</b><p>{html.escape(ordered_archetypes[0][0])} {ordered_archetypes[0][1]} 人；前两类合计 {sum(count for _, count in ordered_archetypes[:2])}/100。</p></article><article><b>卡池审计</b><p>100 份代表 deck 均为 60 张，共覆盖 {len(presence)} 个 Card ID。</p></article></div>
<div class="snapshot-note"><b>证据护栏：</b>单 submission 的 Episode Meta 最多暴露约 1,000 局；本次有 {capped_submissions} 个 submission 命中端点上限。逐局 opponent Meta 已按 leaderboard 冻结时间截断后用于真实 matchup，firstPlayer / final turn 仍只按代表 replay 审计，不外推为全量结论。</div></section>

<section class="panel" id="personal-winrates"><div class="heading"><div><p class="eyebrow">PLAYER CONSTRUCTION UI · FINAL SUBMISSION META</p><h2>个人构筑与双分段胜率</h2></div><p>胜率按 0726 视觉合同分级高亮；高分段 = 对手 submission 属于冻结 Top 100，低分段 = 其余公开 Episode 对手。</p></div><div class="toolbar"><input id="search" type="search" placeholder="搜索选手、Team ID、牌型或 rank…"><select id="archetype-filter"><option value="">全部牌型</option>{''.join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name, _ in ordered_archetypes)}</select><span id="visible-count" class="count-visible">显示 100</span></div><div class="person-grid">{''.join(person_cards)}</div><div class="table-scroll"><table class="wide-table summary-table"><thead><tr><th>Rank</th><th>选手</th><th>牌型</th><th>榜分</th><th>Meta 场次</th><th>W-L-D</th><th>胜率</th><th>deck evidence</th></tr></thead><tbody>{''.join(player_row(player, 'data-player-row') for player in players)}</tbody></table></div></section>

<section class="panel" id="construction-distribution"><div class="heading"><div><p class="eyebrow">CONSTRUCTION DISTRIBUTION</p><h2>构筑使用分布</h2></div><p>牌型只由 {report_id} 最终 submission 的代表 replay exact 60-card deck 分类。</p></div><div class="archetype-grid">{''.join(archetype_cards)}</div></section>

<section class="panel" id="snapshot-comparison"><div class="heading"><div><p class="eyebrow">SNAPSHOT DELTA</p><h2>{previous_date} → {report_date} 同榜选手与卡组变化</h2></div><p>以前一份已发布日报的 rank、牌型和 deck hash 前缀为基线。</p></div><div class="metrics comparison-metrics"><div class="metric"><b>{len(joined)}</b><span>两次同时上榜</span></div><div class="metric"><b>{len(changed)}</b><span>deck hash 变化</span></div><div class="metric"><b>{len(entered)}</b><span>新进榜</span></div><div class="metric"><b>{len(exited)}</b><span>退出榜</span></div></div><p class="muted"><b>新进榜：</b>{html.escape('、'.join(str(player['team_name']) for player in entered) or '—')}<br><b>退出榜：</b>{html.escape('、'.join(exited) or '—')}</p><label><input id="changed-only" type="checkbox"> 只看卡组变化</label><div class="table-scroll"><table class="comparison-table"><thead><tr><th>选手</th><th>Rank</th><th>牌型</th><th>deck hash</th></tr></thead><tbody>{''.join(comparison_rows)}</tbody></table></div></section>

<section class="panel" id="matchup-boundary"><div class="heading"><div><p class="eyebrow">{len(archetype_names)} ARCHETYPE MATRIX · EXACT SUBMISSIONS</p><h2>牌型对战胜率热力图</h2></div><p>行是当前玩家牌型，列是对手牌型；只统计 leaderboard 冻结时间以前、双方均为本次冻结最终 submission 的 PUBLIC + COMPLETED Meta。</p></div><div class="table-scroll matrix-scroll"><table class="matrix-table heatmap"><thead><tr><th>行 / 列</th>{matrix_head}</tr></thead><tbody>{archetype_matrix_rows}</tbody></table></div><p class="muted">共 {top100_player_views:,} 个 Top100 玩家视角；每格显示胜率、W-L-D 与 n。n=0 显示“—”，不冒充 0% 胜率。</p><h3>Top 20 直接 Match-up</h3><div class="table-scroll"><table class="top20-table"><thead><tr><th>Rank</th><th>选手</th><th>牌型</th><th>Meta 场次</th><th>W-L-D</th><th>胜率</th></tr></thead><tbody>{top20_rows}</tbody></table></div><div class="table-scroll"><table class="top20-matrix heatmap"><thead><tr><th>行 / 列</th>{top20_head}</tr></thead><tbody>{top20_matrix_rows}</tbody></table></div><div class="snapshot-note"><b>先后攻/回合护栏：</b>逐局 Meta 已补齐 opponent 与时间字段，但 firstPlayer 和 final turn 只存在于 replay；当前只有每个 submission 一份代表 replay，因此不将其扩张为全量先后攻或节奏结论。</div></section>

<section class="panel" id="card-pool"><div class="heading"><div><p class="eyebrow">6,000 SLOTS · 0725 CARD-POOL UI</p><h2>Top 100 全局构筑卡池</h2></div><p>100 份 audited exact 60-card deck 的并集共有 {len(presence)} 个 Card ID；投入分布按每份构筑中该卡的张数统计。</p></div><div class="investment-grid">{investment_html}</div><div class="table-scroll card-pool-scroll"><table class="pool-table"><thead><tr><th>卡牌</th><th>卡面资料</th><th>使用构筑</th><th>覆盖率</th><th>合计投入</th><th>使用时均值</th><th>中位数</th><th>范围</th><th>投入分布</th></tr></thead><tbody>{''.join(pool_cards)}</tbody></table></div></section>

<section class="panel" id="archetype-builds"><div class="heading"><div><p class="eyebrow">ARCHETYPE BUILD AUDIT</p><h2>各牌型典型构筑卡池</h2></div><p>每张卡的 n 表示该牌型中有多少套代表 deck 包含它；不是平均投入张数或因果强度。</p></div>{''.join(archetype_profiles)}</section>

<section class="panel" id="rank-index"><div class="heading"><div><p class="eyebrow">STATIC RANK INDEX</p><h2>Top 100 静态排名索引</h2></div><p>固定为 {html.escape(str(state["captured_at_utc"]))} 的榜单，不随后续 leaderboard 变化重排。</p></div><div class="table-scroll"><table class="summary-table"><thead><tr><th>Rank</th><th>选手</th><th>牌型</th><th>榜分</th><th>Meta 场次</th><th>W-L-D</th><th>胜率</th><th>deck evidence</th></tr></thead><tbody>{''.join(player_row(player, 'data-index-row') for player in players)}</tbody></table></div></section>

<section class="panel" id="player-details"><div class="heading"><div><p class="eyebrow">PLAYER DETAIL / EXACT DECK</p><h2>逐人 exact 60-card deck 展开</h2></div><p>点击展开；卡图、分组、张数、Card ID 与大图预览沿用 0726。</p></div><div class="toolbar"><button id="open-visible" type="button">展开当前筛选</button><button id="close-all" type="button">全部收起</button></div><div class="person-grid">{''.join(detail_blocks)}</div></section>

<section class="panel provenance" id="boundaries"><div class="heading"><div><p class="eyebrow">BOUNDARIES</p><h2>数据边界与复现</h2></div></div><ul><li>报告 ID：<code>{report_id}</code>；leaderboard 冻结：<code>{html.escape(str(state["captured_at_utc"]))}</code>。</li><li>100 行均通过 <code>leaderboard score = submission publicScore</code> 绑定实际高分 submission；只有同分候选才使用 submissionDate 消歧。</li><li>只统计 PUBLIC + COMPLETED 且 submission ID 精确匹配、createTime 不晚于 leaderboard 冻结时间的公开 Episode Meta。</li><li>逐局 Meta 玩家视角 {total_games:,}，其中当前 Top100 最终 submission 互局视角 {top100_player_views:,}；命中约 1,000 条端点上限的 submission 为 {capped_submissions} 个。</li><li>每个 deck 来自该 submission 最新合格代表 replay 的自身 player index，且恰为 60 个 Card ID；渲染前已执行 100/100 身份链强制审计。</li><li>UI 基准：<a href="{UI_BASELINE}">0726</a>；构筑卡池专项基准：<a href="{CARD_POOL_BASELINE}">0725</a>；归档入口：<a href="../index.html">环境日报索引</a>。</li><li>本地冻结事实源：<code>{html.escape(str(state.get("evidence_root", ".tmp/environment_daily")))}/snapshot.json</code>；逐局 Meta 缓存：<code>{html.escape(str(state.get("evidence_root", ".tmp/environment_daily")))}/meta_views.json</code>；生成入口：<code>python3 -m data.processed.environment_daily.generate_live_snapshot</code>。</li></ul></section>
</main><script type="application/json" id="snapshot-audit">{audit_json}</script><script>
const search=document.getElementById('search'), archetype=document.getElementById('archetype-filter'), visible=document.getElementById('visible-count');
function apply(){{const q=search.value.trim().toLowerCase();let n=0;document.querySelectorAll('[data-player-row],[data-player-detail],[data-person-card]').forEach(row=>{{const ok=(!q||row.dataset.search.includes(q))&&(!archetype.value||row.dataset.archetype===archetype.value);row.hidden=!ok;if(ok&&row.matches('[data-player-row]'))n++;}});visible.textContent=`显示 ${{n}}`;}}search.addEventListener('input',apply);archetype.addEventListener('change',apply);apply();
document.getElementById('changed-only').addEventListener('change',event=>document.querySelectorAll('.comparison-table tbody tr').forEach(row=>row.hidden=event.target.checked&&!row.classList.contains('changed-deck')));
document.getElementById('open-visible').addEventListener('click',()=>document.querySelectorAll('[data-player-detail]:not([hidden])').forEach(row=>row.open=true));document.getElementById('close-all').addEventListener('click',()=>document.querySelectorAll('[data-player-detail]').forEach(row=>row.open=false));
const preview=document.createElement('div');preview.className='card-preview';preview.innerHTML='<img alt="卡牌大图预览" width="234" height="328"><b></b>';document.body.appendChild(preview);const previewImg=preview.querySelector('img'),previewName=preview.querySelector('b');let active=null;function position(e){{const w=preview.offsetWidth||250,h=preview.offsetHeight||370,m=14;let l=e.clientX+18,t=e.clientY+18;if(l+w+m>innerWidth)l=e.clientX-w-18;if(t+h+m>innerHeight)t=innerHeight-h-m;preview.style.left=`${{Math.max(m,l)}}px`;preview.style.top=`${{Math.max(m,t)}}px`;}}function show(thumb,e){{const img=thumb.querySelector('img');if(!img||img.hidden)return;active=thumb;previewImg.src=thumb.dataset.cardPreviewUrl||img.src;previewName.textContent=thumb.dataset.cardName||img.alt;preview.classList.add('visible');position(e);}}function hide(){{active=null;preview.classList.remove('visible');}}document.addEventListener('pointerover',e=>{{const t=e.target.closest&&e.target.closest('.card-thumb');if(t&&t!==active)show(t,e);}});document.addEventListener('pointermove',e=>{{if(active)position(e);}});document.addEventListener('pointerout',e=>{{if(active&&!e.relatedTarget?.closest?.('.card-thumb'))hide();}});
</script></body></html>'''
    report.write_text(report_html, encoding="utf-8")


def _refresh_frozen_player(
    api: KaggleApi,
    player: dict[str, object],
    capture_cutoff: datetime,
    replay_root: Path,
    catalog: dict[int, dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    submission_id = int(player["submission_id"])
    all_eligible = _rate_call(lambda: _eligible_episodes(api, submission_id, 1000))
    episodes = _episodes_at_or_before(all_eligible, capture_cutoff)
    if not episodes:
        raise RuntimeError(f"submission {submission_id} has no Episode at snapshot boundary")
    episode = episodes[0]
    episode_id = int(_model_value(episode, "id"))
    player_index = _episode_player_index(episode, submission_id)
    episode_create_time = _datetime(_model_value(episode, "create_time", "createTime"))
    replay = replay_root / f"episode-{episode_id}.json"
    if not replay.exists():
        _rate_call(
            lambda: api.competition_episode_replay(
                episode_id, path=str(replay_root), quiet=True
            )
        )
        (replay_root / f"episode-{episode_id}-replay.json").replace(replay)
    deck = _decks_from_replay(json.loads(replay.read_text(encoding="utf-8")))[player_index]
    if len(deck) != 60:
        raise RuntimeError(
            f"submission {submission_id} has no exact 60-card replay at snapshot boundary"
        )
    wins, losses, draws, known = _result(episodes, submission_id)
    deck_sha256 = hashlib.sha256(
        ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
    ).hexdigest()
    player.update(
        {
            "episode_id": episode_id,
            "episode_create_time": (
                episode_create_time.isoformat() if episode_create_time else None
            ),
            "episode_player_index": player_index,
            "deck_sha256": deck_sha256,
            "valid_games": len(episodes),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": (wins + 0.5 * draws) / known if known else None,
            "archetype": _classify_archetype(deck, catalog),
            "deck": deck,
        }
    )
    return _episode_views(all_eligible, submission_id), {
        "eligible": len(all_eligible),
        "bounded": len(episodes),
        "possibly_censored_at_1000": len(all_eligible) >= 1000,
    }


def _bounded_meta_fingerprint(
    views: list[dict[str, object]], capture_cutoff: datetime
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (
                int(view["episode_id"]),
                view.get("reward"),
                int(view.get("player_index", -1)),
                int((view.get("other") or {}).get("submission_id") or 0),
            )
            for view in views
            if _datetime(view.get("create_time")) is not None
            and _datetime(view.get("create_time")) <= capture_cutoff
        )
    )


def _merge_episode_views(
    existing: list[dict[str, object]], fresh: list[dict[str, object]]
) -> list[dict[str, object]]:
    by_episode = {int(view["episode_id"]): view for view in existing}
    by_episode.update({int(view["episode_id"]): view for view in fresh})
    return sorted(
        by_episode.values(),
        key=lambda view: (
            _datetime(view.get("create_time"))
            or datetime.min.replace(tzinfo=timezone.utc),
            int(view["episode_id"]),
        ),
        reverse=True,
    )


def _apply_bounded_meta_stats(
    player: dict[str, object],
    views: list[dict[str, object]],
    capture_cutoff: datetime,
) -> None:
    rewards = [
        float(view["reward"])
        for view in views
        if view.get("reward") is not None
        and _datetime(view.get("create_time")) is not None
        and _datetime(view.get("create_time")) <= capture_cutoff
    ]
    wins = sum(reward > 0 for reward in rewards)
    losses = sum(reward < 0 for reward in rewards)
    draws = sum(reward == 0 for reward in rewards)
    player.update(
        {
            "valid_games": len(rewards),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": (wins + 0.5 * draws) / len(rewards) if rewards else None,
        }
    )


def collect(work: Path, report: Path, report_date: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    replay_root = work / "representative_replays"
    replay_root.mkdir(exist_ok=True)
    state_path = work / "snapshot.json"
    meta_path = work / "meta_views.json"
    if state_path.exists():
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        if saved_state.get("status") == "complete":
            raise RuntimeError(
                "completed leaderboard capture is immutable; use a new empty --work directory "
                "for a fresh daily freeze, or pass --render-only to render this exact capture"
            )
        if _report_date(saved_state) != report_date:
            raise ValueError("work directory belongs to a different report date")
        state = saved_state
    else:
        state = {}
    meta_payload = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.exists()
        else {"views": {}, "audit": {}}
    )
    api = KaggleApi()
    api.authenticate()
    catalog = _card_catalog()
    prefetched_submissions: dict[int, list[object]] = {}
    if not state:
        leaderboard, prefetched_submissions, captured_at_utc = (
            _capture_score_bound_leaderboard(api)
        )
        state = {
            "schema": SNAPSHOT_SCHEMA,
            "status": "collecting",
            "report_date": report_date,
            "captured_at_utc": captured_at_utc,
            "evidence_root": str(work),
            "rows": [
                {"rank": rank, "team_id": row.team_id, "team_name": row.team_name,
                 "score": row.score, "submission_date": row.submission_date.isoformat()}
                for rank, row in enumerate(leaderboard, 1)
            ],
            "players": {},
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    capture_cutoff = _datetime(state["captured_at_utc"])
    if capture_cutoff is None:
        raise ValueError(f"invalid captured_at_utc: {state['captured_at_utc']!r}")

    # Phase 1 freezes the exact submission represented by every leaderboard row before
    # probing any Episode. This gives Kaggle's Episode index time to settle around the cutoff.
    for row in state["rows"]:
        key = str(row["rank"])
        existing = state["players"].get(key)
        if isinstance(existing, dict) and existing.get("submission_id"):
            continue
        print(f"[{row['rank']:03d}/100] binding {row['team_name']}", flush=True)
        submissions = prefetched_submissions.get(int(row["team_id"]))
        if submissions is None:
            submissions = _rate_call(
                lambda: api.competition_team_submissions(row["team_id"])
            )

        class Model:
            pass

        model = Model()
        model.team_id = row["team_id"]
        model.team_name = row["team_name"]
        model.score = row["score"]
        model.submission_date = datetime.fromisoformat(row["submission_date"])
        selected, binding = _select_leaderboard_submission(model, list(submissions))
        if binding not in LEADERBOARD_BINDINGS:
            raise RuntimeError(
                f"rank {row['rank']}: leaderboard score did not bind uniquely"
            )
        state["players"][key] = {
            **row,
            "submission_id": int(selected.id),
            "submission_public_score": str(
                _model_value(selected, "public_score", "publicScore", default="")
            ),
            "submission_date_actual": str(
                _model_value(selected, "date_submitted", "dateSubmitted", default="")
            ),
            "binding": binding,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    # Phase 2 refreshes Episode Meta for every frozen submission and derives the exact deck
    # only from the latest eligible Episode at or before the one shared leaderboard cutoff.
    for row in state["rows"]:
        key = str(row["rank"])
        player = state["players"][key]
        submission_id = int(player["submission_id"])
        print(f"[{row['rank']:03d}/100] freezing Episodes for {row['team_name']}", flush=True)
        views, meta_audit = _refresh_frozen_player(
            api, player, capture_cutoff, replay_root, catalog
        )
        views = _merge_episode_views(
            meta_payload["views"].get(str(submission_id), []), views
        )
        _apply_bounded_meta_stats(player, views, capture_cutoff)
        meta_audit["bounded_union"] = len(_bounded_meta_fingerprint(views, capture_cutoff))
        meta_payload["views"][str(submission_id)] = views
        meta_payload["audit"][str(submission_id)] = meta_audit
        meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False), encoding="utf-8")
        state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        time.sleep(1.0)

    # Phase 3 keeps re-querying until two consecutive full sweeps observe no change in either
    # the cutoff-latest Episode or the complete bounded Meta set. This is data-driven rather
    # than assuming Kaggle's eventually consistent Episode index settles after a fixed delay.
    consecutive_stable_sweeps = 0
    stabilization_history = []
    for sweep in range(1, 7):
        changed_submissions = []
        for row in state["rows"]:
            key = str(row["rank"])
            player = state["players"][key]
            submission_id = int(player["submission_id"])
            old_episode_id = player.get("episode_id")
            old_fingerprint = _bounded_meta_fingerprint(
                meta_payload["views"].get(str(submission_id), []), capture_cutoff
            )
            print(
                f"[sweep {sweep} · {row['rank']:03d}/100] finalizing {row['team_name']}",
                flush=True,
            )
            views, meta_audit = _refresh_frozen_player(
                api, player, capture_cutoff, replay_root, catalog
            )
            views = _merge_episode_views(
                meta_payload["views"].get(str(submission_id), []), views
            )
            _apply_bounded_meta_stats(player, views, capture_cutoff)
            new_fingerprint = _bounded_meta_fingerprint(views, capture_cutoff)
            if old_episode_id != player.get("episode_id") or old_fingerprint != new_fingerprint:
                changed_submissions.append(submission_id)
            meta_audit["bounded_union"] = len(new_fingerprint)
            meta_payload["views"][str(submission_id)] = views
            meta_payload["audit"][str(submission_id)] = meta_audit
            meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False), encoding="utf-8")
            state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        stabilization_history.append(
            {
                "sweep": sweep,
                "changed_submissions": changed_submissions,
                "changed_count": len(changed_submissions),
            }
        )
        consecutive_stable_sweeps = (
            consecutive_stable_sweeps + 1 if not changed_submissions else 0
        )
        state["stabilization"] = {
            "required_consecutive_stable_sweeps": 2,
            "consecutive_stable_sweeps": consecutive_stable_sweeps,
            "history": stabilization_history,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        if consecutive_stable_sweeps >= 2:
            break
    else:
        raise RuntimeError("Kaggle Episode Meta did not stabilize after six full sweeps")
    players = [state["players"][str(i)] for i in range(1, 101)]
    state["identity_audit"] = _validate_snapshot(state, meta_payload, replay_root)
    state["status"] = "complete"
    state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    _render_report(state, players, catalog, report, meta_payload, replay_root)


def _update_index(report_date: str) -> None:
    index_path = DAILY_REPORT_ROOT.parent / "index.html"
    text = index_path.read_text(encoding="utf-8")
    report_id = _report_id(report_date)
    article = f'''\n    <article class="report">
      <time datetime="{report_date}">{report_date}</time>
      <div>
        <h3>Top 100 实时环境快照 · {report_id}</h3>
        <p>冻结官方 Top 100，并审计最终 submission、公开 completed Episode、胜率与 exact deck。</p>
      </div>
      <a class="button" href="daily/{report_date}.html">查看日报</a>
    </article>'''
    matches = list(
        re.finditer(r'    <article class="[^"]*\breport\b[^"]*">.*?    </article>', text, re.S)
    )
    if not matches:
        raise ValueError(f"report articles missing: {index_path}")
    articles = [match.group(0) for match in matches]
    if f'href="daily/{report_date}.html"' not in text:
        articles.append(article.strip("\n"))
    articles.sort(
        key=lambda value: re.search(r'<time datetime="([^"]+)">', value).group(1),
        reverse=True,
    )
    updated = text[: matches[0].start()] + "\n".join(articles) + text[matches[-1].end() :]
    index_path.write_text(updated, encoding="utf-8")


def _default_report_date() -> str:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the live Kaggle Top 100 and build one audited environment daily."
    )
    parser.add_argument("--date", default=_default_report_date())
    parser.add_argument("--work", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date):
        parser.error("--date must use YYYY-MM-DD")
    if args.render_only and args.validate_only:
        parser.error("--render-only and --validate-only are mutually exclusive")
    if args.work is None:
        if args.render_only or args.validate_only:
            parser.error("--work is required for render-only or validate-only")
        stamp = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
        args.work = REPOSITORY_ROOT / ".tmp/environment_daily" / args.date / stamp
    if args.report is None:
        args.report = DAILY_REPORT_ROOT / f"{args.date}.html"
    if args.report.exists() and not (args.overwrite or args.validate_only):
        parser.error(f"report already exists; pass --overwrite explicitly: {args.report}")

    if args.render_only or args.validate_only:
        snapshot = json.loads((args.work / "snapshot.json").read_text(encoding="utf-8"))
        meta_payload = json.loads((args.work / "meta_views.json").read_text(encoding="utf-8"))
        replay_root = args.work / "representative_replays"
        snapshot["identity_audit"] = _validate_snapshot(snapshot, meta_payload, replay_root)
        if args.validate_only:
            print(json.dumps(snapshot["identity_audit"], ensure_ascii=False))
            return
        snapshot_players = [snapshot["players"][str(rank)] for rank in range(1, 101)]
        _render_report(
            snapshot,
            snapshot_players,
            _card_catalog(),
            args.report,
            meta_payload,
            replay_root,
        )
    else:
        collect(args.work, args.report, args.date)
    if args.report.resolve().parent == DAILY_REPORT_ROOT.resolve():
        _update_index(args.date)


if __name__ == "__main__":
    main()
