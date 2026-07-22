"""Download and audit a Kaggle expert submission's public replays.

The Kaggle API currently exposes at most the latest 1,000 episodes for a
submission.  The manifest records that limit explicitly instead of implying
that the result is the expert's complete lifetime history.
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
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiGetEpisodeReplayRequest


_THREAD_LOCAL = threading.local()
_RATE_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0


def _api() -> KaggleApi:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = KaggleApi()
        client.authenticate()
        _THREAD_LOCAL.client = client
    return client


def _wait_for_request_slot(interval_seconds: float) -> None:
    global _NEXT_REQUEST_AT
    with _RATE_LOCK:
        now = time.monotonic()
        ready_at = max(now, _NEXT_REQUEST_AT)
        _NEXT_REQUEST_AT = ready_at + interval_seconds
    delay = ready_at - now
    if delay > 0:
        time.sleep(delay)


def _defer_requests(seconds: float) -> None:
    global _NEXT_REQUEST_AT
    with _RATE_LOCK:
        _NEXT_REQUEST_AT = max(_NEXT_REQUEST_AT, time.monotonic() + seconds)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            backoff = min(60.0, (15.0 if "429" in message else 2.0) * (2**attempt))
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
    agent_name: str,
    expected_deck: list[int],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episode_id = int((payload.get("info") or {}).get("EpisodeId"))
    if episode_id != expected_episode_id:
        raise ValueError(f"episode ID mismatch in {path}: {episode_id} != {expected_episode_id}")
    agents = ((payload.get("info") or {}).get("Agents") or [])
    names = [str(agent.get("Name", "")) for agent in agents if isinstance(agent, dict)]
    matches = [index for index, name in enumerate(names) if name == agent_name]
    if len(matches) != 1:
        raise ValueError(f"{agent_name!r} is not unique in episode {episode_id}: {names}")
    agent_index = matches[0]
    deck = _deck_from_replay(payload, agent_index)
    if collections.Counter(deck) != collections.Counter(expected_deck):
        raise ValueError(f"exact deck mismatch in episode {episode_id}")
    if len(payload.get("steps") or []) < 2:
        raise ValueError(f"episode {episode_id} has no action steps")
    return {
        "episode_id": episode_id,
        "file": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "agent_index": agent_index,
        "agents": names,
        "reward": (payload.get("rewards") or [None, None])[agent_index],
        "steps": len(payload["steps"]),
    }


def download(
    output_root: Path,
    *,
    submission_id: int,
    agent_name: str,
    expected_deck: list[int],
    workers: int = 2,
    retries: int = 12,
    request_interval: float = 1.0,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    episodes = _api().competition_list_episodes(submission_id)
    episode_rows = [
        {
            "episode_id": int(episode.id),
            "create_time": str(episode.create_time),
            "end_time": str(episode.end_time),
            "state": str(episode.state),
            "type": str(episode.type),
        }
        for episode in episodes
    ]
    if not episode_rows:
        raise RuntimeError(f"submission {submission_id} has no public episodes")
    episode_rows_by_id = {row["episode_id"]: row for row in episode_rows}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [
            pool.submit(_download_one, episode_id, output_root, retries, request_interval)
            for episode_id in sorted(episode_rows_by_id)
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if index % 50 == 0 or index == len(futures):
                print(f"downloaded {index}/{len(futures)} replays", flush=True)

    audited: list[dict[str, Any]] = []
    for episode_id in sorted(episode_rows_by_id):
        path = output_root / f"episode-{episode_id}-replay.json"
        row = {**episode_rows_by_id[episode_id], **_audit_replay(
            path,
            expected_episode_id=episode_id,
            agent_name=agent_name,
            expected_deck=expected_deck,
        )}
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
        "schema_version": "yushin_exact_replay_manifest_v1",
        "submission_id": submission_id,
        "agent_name": agent_name,
        "deck": expected_deck,
        "deck_counts": dict(sorted(collections.Counter(expected_deck).items())),
        "episode_api_limit": 1000,
        "episode_count": total,
        "episode_type": "public",
        "api_note": (
            "competition_list_episodes exposes the latest 1000 episodes for this submission"
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
        "bytes": sum(int(row["bytes"]) for row in audited),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--agent-name", default="Yushin Ito")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--request-interval", type=float, default=1.0)
    args = parser.parse_args()
    deck = [
        int(line.strip())
        for line in args.deck.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        parser.error("--deck must contain exactly 60 card IDs")
    print(json.dumps(download(
        args.output,
        submission_id=args.submission_id,
        agent_name=args.agent_name,
        expected_deck=deck,
        workers=args.workers,
        retries=args.retries,
        request_interval=args.request_interval,
    ), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
