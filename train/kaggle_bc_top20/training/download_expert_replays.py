"""Download and audit one exact Kaggle expert source's public replays.

The Kaggle API currently exposes at most the latest 1,000 episodes for a
submission.  The manifest records that limit explicitly instead of implying
that the result is the expert's complete lifetime history.  The command only
accepts a frozen single-expert source manifest and identifies the player by
submission ID; display names are retained for audit but never used as identity.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiGetEpisodeReplayRequest


_THREAD_LOCAL = threading.local()
_RATE_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0
SOURCE_SCHEMA_VERSION = "ptcg_single_expert_bc_source_v1"
EXACT_SOURCE_SCHEMA_VERSION = "ptcg_single_expert_bc_source_v2"
REPLAY_SCHEMA_VERSION = "ptcg_single_expert_exact_replays_v2"


def _api() -> KaggleApi:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = KaggleApi()
        client.authenticate()
        _THREAD_LOCAL.client = client
    return client


def _discard_thread_api_client() -> None:
    """Force the next request in this worker to authenticate a fresh SDK client."""
    if hasattr(_THREAD_LOCAL, "client"):
        delattr(_THREAD_LOCAL, "client")


def _wait_for_request_slot(interval_seconds: float) -> None:
    global _NEXT_REQUEST_AT
    while True:
        with _RATE_LOCK:
            now = time.monotonic()
            delay = _NEXT_REQUEST_AT - now
            if delay <= 0:
                _NEXT_REQUEST_AT = now + interval_seconds
                return
        # A concurrent 429 may extend the shared deadline while this thread
        # sleeps.  Recheck after waking instead of using a stale reservation.
        time.sleep(delay)


def _defer_requests(seconds: float) -> None:
    global _NEXT_REQUEST_AT
    with _RATE_LOCK:
        _NEXT_REQUEST_AT = max(_NEXT_REQUEST_AT, time.monotonic() + seconds)


def _retry_after_seconds(exc: Exception, *, now: datetime | None = None) -> float | None:
    """Return a server-requested retry delay without depending on requests internals."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (retry_at - current).total_seconds())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_deck_sha256(deck: list[int]) -> str:
    """Hash an exact 60-card multiset independent of source-file order."""
    normalized = ",".join(str(card_id) for card_id in sorted(int(card) for card in deck))
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _model_value(model: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            return value
    return default


def _timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _episode_agents(episode: Any) -> list[dict[str, Any]]:
    agents: list[dict[str, Any]] = []
    for fallback_index, agent in enumerate(_model_value(episode, "agents", default=[]) or []):
        agents.append(
            {
                "index": int(_model_value(agent, "index", default=fallback_index) or 0),
                "submission_id": int(
                    _model_value(agent, "submission_id", "submissionId", default=0) or 0
                ),
                "team_id": int(_model_value(agent, "team_id", "teamId", default=0) or 0),
                "team_name": str(
                    _model_value(agent, "team_name", "teamName", default="") or ""
                ),
                "reward": _model_value(agent, "reward"),
            }
        )
    return agents


def episode_player_index(episode: Any, submission_id: int) -> int:
    matches = [
        int(agent["index"])
        for agent in _episode_agents(episode)
        if int(agent["submission_id"]) == submission_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"submission {submission_id} is not unique in episode "
            f"{_model_value(episode, 'id')}: {matches}"
        )
    return matches[0]


def load_exact_source_manifest(path: Path) -> dict[str, Any]:
    """Load and fail closed on a sampled single-policy source identity."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        SOURCE_SCHEMA_VERSION,
        EXACT_SOURCE_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported source manifest schema: {path}")
    if payload.get("single_policy_constraint") is not True:
        raise ValueError(f"source manifest does not freeze one policy: {path}")
    policy = payload.get("source_policy")
    if not isinstance(policy, dict):
        raise ValueError(f"source manifest is missing source_policy: {path}")
    for field in ("team_id", "submission_id", "team_name"):
        if policy.get(field) in {None, ""}:
            raise ValueError(f"source manifest is missing source_policy.{field}: {path}")
    deck_file = payload.get("deck_file")
    if not isinstance(deck_file, str) or not deck_file:
        raise ValueError(f"source manifest is missing deck_file: {path}")
    deck_path = (path.parent / deck_file).resolve()
    deck = [
        int(line.strip())
        for line in deck_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"source deck must contain exactly 60 cards: {deck_path}")
    expected_hash = str((payload.get("deck_profile") or {}).get("deck_sha256", ""))
    actual_hash = canonical_deck_sha256(deck)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            f"source deck hash mismatch: manifest={expected_hash!r}, actual={actual_hash}"
        )
    return {
        **payload,
        "_manifest_path": str(path.resolve()),
        "_manifest_sha256": _sha256(path),
        "_deck_path": str(deck_path),
        "_deck": deck,
        "_deck_sha256": actual_hash,
    }


def eligible_episode_rows(episodes: list[Any], submission_id: int) -> list[dict[str, Any]]:
    """Return only completed public episodes containing the exact submission once."""
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        episode_type = str(_model_value(episode, "type", default=""))
        episode_state = str(_model_value(episode, "state", default=""))
        if "PUBLIC" not in episode_type or "COMPLETED" not in episode_state:
            continue
        player_index = episode_player_index(episode, submission_id)
        rows.append(
            {
                "episode_id": int(_model_value(episode, "id")),
                "create_time": _timestamp(
                    _model_value(episode, "create_time", "createTime")
                ),
                "end_time": _timestamp(_model_value(episode, "end_time", "endTime")),
                "state": episode_state,
                "type": episode_type,
                "player_index": player_index,
                "agents": _episode_agents(episode),
            }
        )
    rows.sort(key=lambda row: (row["create_time"], row["episode_id"]))
    return rows


def _download_one(
    episode_id: int,
    output_root: Path,
    retries: int,
    request_interval: float,
    network_timeout: float = 60.0,
) -> tuple[int, str, int]:
    destination = output_root / f"episode-{episode_id}-replay.json"
    if destination.is_file() and destination.stat().st_size > 0:
        try:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if int((payload.get("info") or {}).get("EpisodeId")) == episode_id:
                return episode_id, "cached", destination.stat().st_size
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    last_error: Exception | None = None
    for attempt in range(retries):
        temporary_root = Path(tempfile.mkdtemp(prefix=f"episode-{episode_id}-", dir=output_root))
        try:
            _wait_for_request_slot(request_interval)
            client = _api()
            with client.build_kaggle_client() as kaggle:
                http_client = kaggle.http_client()
                http_client._init_session()
                session = http_client._session
                original_send = session.send

                def send_with_timeout(request: Any, **kwargs: Any) -> Any:
                    kwargs.setdefault("timeout", network_timeout)
                    return original_send(request, **kwargs)

                session.send = send_with_timeout
                request = ApiGetEpisodeReplayRequest()
                request.episode_id = episode_id
                response = kaggle.competitions.competition_api_client.get_episode_replay(request)
                downloaded = temporary_root / destination.name
                client.download_file(
                    response,
                    str(downloaded),
                    http_client,
                    quiet=True,
                    max_retries=1,
                    timeout=network_timeout,
                )
            if not downloaded.is_file() or downloaded.stat().st_size == 0:
                raise RuntimeError(f"Kaggle did not create {downloaded}")
            os.replace(downloaded, destination)
            return episode_id, "downloaded", destination.stat().st_size
        except Exception as exc:  # noqa: BLE001 - retry network/API errors
            last_error = exc
            message = str(exc)
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 401 or "401 Client Error" in message:
                # Long replay campaigns can outlive the SDK client's access
                # token.  Reusing that client makes every retry unauthorized.
                _discard_thread_api_client()
            fallback = min(60.0, (15.0 if "429" in message else 2.0) * (2**attempt))
            retry_after = _retry_after_seconds(exc)
            # Kaggle currently returns Retry-After values greater than the old
            # 60-second local cap.  Retrying below that boundary perpetuates the
            # throttle, so give the server deadline a one-second safety margin.
            backoff = max(fallback, (retry_after + 1.0) if retry_after is not None else 0.0)
            _defer_requests(backoff)
        finally:
            try:
                for child in temporary_root.iterdir():
                    if child.is_file():
                        child.unlink()
                temporary_root.rmdir()
            except OSError:
                pass
    raise RuntimeError(f"failed to download episode {episode_id}: {last_error}")


def _deck_from_replay(payload: dict[str, Any], agent_index: int) -> list[int]:
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
                    and len(actions) > agent_index
                    and isinstance(actions[agent_index], list)
                    and len(actions[agent_index]) == 60
                ):
                    return [int(card_id) for card_id in actions[agent_index]]
    raise ValueError("replay has no public 60-card initial deck frame")


def _audit_replay(
    path: Path,
    *,
    expected_episode_id: int,
    player_index: int,
    submission_id: int,
    team_id: int,
    team_name: str,
    expected_deck: list[int],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode_id = int((payload.get("info") or {}).get("EpisodeId"))
    if episode_id != expected_episode_id:
        raise ValueError(f"episode ID mismatch in {path}: {episode_id} != {expected_episode_id}")
    agents = ((payload.get("info") or {}).get("Agents") or [])
    names = [str(agent.get("Name", "")) for agent in agents if isinstance(agent, dict)]
    if player_index < 0 or player_index >= len(names):
        raise ValueError(f"player index {player_index} is absent in episode {episode_id}")
    deck = _deck_from_replay(payload, player_index)
    actual_hash = canonical_deck_sha256(deck)
    expected_hash = canonical_deck_sha256(expected_deck)
    if actual_hash != expected_hash:
        raise ValueError(
            f"exact deck mismatch in episode {episode_id}: {actual_hash} != {expected_hash}"
        )
    if len(payload.get("steps") or []) < 2:
        raise ValueError(f"episode {episode_id} has no action steps")
    return {
        "episode_id": episode_id,
        "file": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "agent_index": player_index,
        "agents": names,
        "reward": (payload.get("rewards") or [None, None])[player_index],
        "steps": len(payload["steps"]),
        "deck_sha256": actual_hash,
        "expert_players": [
            {
                "player_index": player_index,
                "submission_id": submission_id,
                "team_id": team_id,
                "team_name": team_name,
            }
        ],
    }


def seed_replays_from_dataset_roots(
    output_root: Path,
    *,
    episode_ids: list[int],
    dataset_roots: list[Path],
) -> dict[str, Any]:
    """Seed exact Episode files from read-only mounted Kaggle daily datasets.

    Daily episode datasets name each replay ``<episode_id>.json``.  The
    frozen source manifest remains authoritative: mounted files are considered
    only when their numeric filename is an expected Episode ID, and the normal
    replay audit later verifies the embedded ID, player index, and exact deck.
    Symlinks avoid copying multi-gigabyte replay payloads into working storage.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    roots = [root.resolve() for root in dataset_roots]
    missing_roots = [root for root in roots if not root.is_dir()]
    if missing_roots:
        raise FileNotFoundError(f"mounted replay dataset root is missing: {missing_roots[0]}")

    seeded: list[dict[str, Any]] = []
    preexisting: list[int] = []
    missing: list[int] = []
    for episode_id in sorted(set(int(value) for value in episode_ids)):
        destination = output_root / f"episode-{episode_id}-replay.json"
        if destination.is_file():
            preexisting.append(episode_id)
            continue
        candidates = [root / f"{episode_id}.json" for root in roots]
        matches = [path for path in candidates if path.is_file()]
        if len(matches) > 1:
            raise ValueError(
                f"episode {episode_id} appears in multiple mounted datasets: {matches}"
            )
        if not matches:
            missing.append(episode_id)
            continue
        source = matches[0].resolve()
        destination.symlink_to(source)
        seeded.append(
            {
                "episode_id": episode_id,
                "source": str(source),
                "bytes": source.stat().st_size,
            }
        )
    return {
        "schema_version": "ptcg_mounted_episode_seed_v1",
        "dataset_roots": [str(root) for root in roots],
        "expected_episode_count": len(set(episode_ids)),
        "seeded_episode_count": len(seeded),
        "seeded_bytes": sum(int(row["bytes"]) for row in seeded),
        "preexisting_cache_count": len(preexisting),
        "missing_episode_count": len(missing),
        "seeded": seeded,
        "preexisting_episode_ids": preexisting,
        "missing_episode_ids": missing,
    }


def download(
    output_root: Path,
    *,
    source_manifest: Path,
    workers: int = 2,
    retries: int = 12,
    request_interval: float = 1.0,
    network_timeout: float = 60.0,
    seed_roots: list[Path] | None = None,
) -> dict[str, Any]:
    if workers < 1 or workers > 2:
        raise ValueError("workers must be between 1 and 2")
    if request_interval < 1.0:
        raise ValueError("request_interval must be at least 1 second")
    source = load_exact_source_manifest(source_manifest)
    policy = source["source_policy"]
    submission_id = int(policy["submission_id"])
    team_id = int(policy["team_id"])
    team_name = str(policy["team_name"])
    expected_deck = list(source["_deck"])
    output_root.mkdir(parents=True, exist_ok=True)
    frozen_rows = source.get("episodes")
    if frozen_rows is not None:
        if source.get("schema_version") != EXACT_SOURCE_SCHEMA_VERSION:
            raise ValueError("only an exact v2 source manifest may freeze Episode rows")
        episode_rows = []
        for row in frozen_rows:
            agents = row.get("agents") or []
            matches = [
                int(agent["index"])
                for agent in agents
                if int(agent.get("submission_id", 0)) == submission_id
            ]
            if (
                "PUBLIC" not in str(row.get("type", ""))
                or "COMPLETED" not in str(row.get("state", ""))
                or matches != [int(row.get("player_index", -1))]
            ):
                raise ValueError(
                    f"frozen Episode row violates exact source contract: {row.get('episode_id')}"
                )
            episode_rows.append(dict(row))
        episode_rows.sort(key=lambda row: (row["create_time"], int(row["episode_id"])))
    else:
        episode_rows = eligible_episode_rows(
            list(_api().competition_list_episodes(submission_id) or []), submission_id
        )
    if not episode_rows:
        raise RuntimeError(f"submission {submission_id} has no completed public episodes")
    episode_rows_by_id = {row["episode_id"]: row for row in episode_rows}
    seed_report = seed_replays_from_dataset_roots(
        output_root,
        episode_ids=list(episode_rows_by_id),
        dataset_roots=list(seed_roots or []),
    )
    seeded_ids = {int(row["episode_id"]) for row in seed_report["seeded"]}
    fetch_statuses: dict[int, str] = {}
    download_failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _download_one,
                episode_id,
                output_root,
                retries,
                request_interval,
                network_timeout,
            ): episode_id
            for episode_id in sorted(episode_rows_by_id)
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                episode_id, status, _ = future.result()
                fetch_statuses[int(episode_id)] = str(status)
            except Exception as exc:  # noqa: BLE001 - persist exact failed Episode
                episode_id = int(futures[future])
                download_failures.append(
                    {
                        "episode_id": episode_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                for pending in futures:
                    pending.cancel()
                break
            if index % 50 == 0 or index == len(futures):
                print(f"downloaded {index}/{len(futures)} replays", flush=True)
    failure_path = output_root / "download_failures.json"
    if download_failures:
        failure_path.write_text(
            json.dumps(download_failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"replay download failed closed; inspect and resume from {failure_path}"
        )
    if failure_path.is_file():
        failure_path.unlink()

    audited: list[dict[str, Any]] = []
    for episode_id in sorted(episode_rows_by_id):
        path = output_root / f"episode-{episode_id}-replay.json"
        episode_row = episode_rows_by_id[episode_id]
        row = {
            **episode_row,
            **_audit_replay(
                path,
                expected_episode_id=episode_id,
                player_index=int(episode_row["player_index"]),
                submission_id=submission_id,
                team_id=team_id,
                team_name=team_name,
                expected_deck=expected_deck,
            ),
            "replay_source": (
                "mounted_daily_dataset"
                if episode_id in seeded_ids
                else fetch_statuses.get(episode_id, "unknown")
            ),
        }
        audited.append(row)
    # Older episodes are useful as held-out data, while the newest public games
    # remain the natural test slice for a current submission policy.
    audited.sort(key=lambda row: (row["create_time"], row["episode_id"]))
    total = len(audited)
    train_end = max(1, int(total * 0.8))
    validation_end = max(train_end + 1, int(total * 0.9)) if total > 1 else total
    for index, row in enumerate(audited):
        if index < train_end:
            row["split"] = "train"
        elif index < validation_end:
            row["split"] = "validation"
        else:
            row["split"] = "test"
    manifest = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": source["_manifest_sha256"],
        "source_identity": {
            "team_id": team_id,
            "submission_id": submission_id,
            "deck_sha256": source["_deck_sha256"],
        },
        "team_id": team_id,
        "submission_id": submission_id,
        "team_name": team_name,
        "deck": expected_deck,
        "deck_counts": dict(sorted(collections.Counter(expected_deck).items())),
        "deck_hash_algorithm": "sha256(comma-separated sorted integer card IDs)",
        "deck_sha256": source["_deck_sha256"],
        "episode_api_limit": 1000,
        "episode_count": total,
        "episode_type": "PUBLIC",
        "episode_state": "COMPLETED",
        "player_resolution": "exact submission_id from Kaggle Episode agents",
        "single_policy_constraint": True,
        "api_note": (
            "competition_list_episodes exposes the latest 1000 episodes for this submission"
        ),
        "mounted_dataset_seed": seed_report,
        "replay_source_counts": dict(
            sorted(collections.Counter(row["replay_source"] for row in audited).items())
        ),
        "replay_root": str(output_root.resolve()),
        "episodes": audited,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "submission_id": submission_id,
        "episode_count": total,
        "replay_root": str(output_root.resolve()),
        "split_counts": dict(collections.Counter(row["split"] for row in audited)),
        "replay_source_counts": dict(
            sorted(collections.Counter(row["replay_source"] for row in audited).items())
        ),
        "bytes": sum(int(row["bytes"]) for row in audited),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--request-interval", type=float, default=1.0)
    parser.add_argument("--network-timeout", type=float, default=60.0)
    parser.add_argument(
        "--seed-root",
        type=Path,
        action="append",
        default=[],
        help="mounted daily episode Dataset directory; may be repeated",
    )
    args = parser.parse_args()
    print(json.dumps(download(
        args.output,
        source_manifest=args.source_manifest,
        workers=args.workers,
        retries=args.retries,
        request_interval=args.request_interval,
        network_timeout=args.network_timeout,
        seed_roots=args.seed_root,
    ), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
