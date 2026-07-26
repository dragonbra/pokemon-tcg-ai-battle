"""Read immutable daily Episode ZIPs with canonical patch precedence."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import shutil
import struct
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date as calendar_date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping

from .records import SourceIdentity

_EXPERT = "Yushin Ito"
_READ_CHUNK = 1024 * 1024
MAX_EPISODE_BYTES = 320 * 1024 * 1024
MAX_SPOOL_BYTES = 8 * 1024 * 1024 * 1024
MIN_TEMP_FREE_BYTES = MAX_SPOOL_BYTES + 1024 * 1024 * 1024
_FRAME = struct.Struct(">Q")
_MEMBER_ID_RE = re.compile(r"^(?:episode-)?(?P<id>\d+)(?:-replay)?$")


def normalize_team_identity(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_date(value: str, label: str) -> None:
    try:
        parsed = calendar_date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid {label} date: {value!r}") from error
    if parsed.isoformat() != value:
        raise ValueError(f"invalid {label} date: {value!r}")


@dataclass(frozen=True, slots=True)
class EpisodePatch:
    date: str
    episode_id: int
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class EpisodeSource:
    date: str
    archive_path: Path
    archive_sha256: str
    patches: tuple[EpisodePatch, ...] = ()


@dataclass(frozen=True, slots=True)
class CanonicalEpisode:
    source: SourceIdentity
    payload: Mapping[str, Any]
    player_trajectory: tuple[Mapping[str, Any], ...]
    payload_sha256: str
    origin: str


@dataclass(frozen=True, slots=True)
class SourceAudit:
    seen: int = 0
    eligible: int = 0
    ineligible_noncomplete: int = 0
    spool_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _ArchiveMember:
    filename: str
    episode_id: int


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    source: EpisodeSource
    members: tuple[_ArchiveMember, ...]
    patches: Mapping[int, EpisodePatch]


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _encode_episode(episode: CanonicalEpisode) -> bytes:
    return pickle.dumps(
        (episode.source, _thaw(episode.payload), episode.payload_sha256, episode.origin),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def _spool_episode(spool: Any, episode: CanonicalEpisode, spool_bytes: int) -> int:
    payload = _encode_episode(episode)
    frame_bytes = _FRAME.size + len(payload)
    if spool_bytes + frame_bytes > MAX_SPOOL_BYTES:
        raise ValueError("validated episode spool byte cap exceeded")
    spool.write(_FRAME.pack(len(payload)))
    spool.write(payload)
    return spool_bytes + frame_bytes


def _decode_spooled_episode(payload_bytes: bytes) -> CanonicalEpisode:
    source, payload, payload_sha256, origin = pickle.loads(payload_bytes)
    frozen = _freeze(payload)
    return CanonicalEpisode(
        source,
        frozen,
        _trajectory(frozen, source.player_index, f"{source.date}/{source.episode_id}"),
        payload_sha256,
        origin,
    )


def _read_frame(spool: Any) -> CanonicalEpisode | None:
    header = spool.read(_FRAME.size)
    if not header:
        return None
    if len(header) != _FRAME.size:
        raise ValueError("truncated spool frame header")
    (size,) = _FRAME.unpack(header)
    payload = spool.read(size)
    if len(payload) != size:
        raise ValueError("truncated spool frame payload")
    return _decode_spooled_episode(payload)


class ReplayableEpisodeRecords:
    """Bounded, replayable spool view valid only during the transactional sink call."""

    def __init__(self, spool: Any, count: int) -> None:
        self._spool = spool
        self._count = count
        self._valid = True
        self._iterating = False

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[CanonicalEpisode]:
        if not self._valid:
            raise RuntimeError("episode records are no longer valid")
        if self._iterating:
            raise RuntimeError("concurrent or reentrant episode iteration is forbidden")
        self._iterating = True
        self._spool.seek(0)
        try:
            for _ in range(self._count):
                if not self._valid:
                    raise RuntimeError("episode records are no longer valid")
                record = _read_frame(self._spool)
                if record is None:
                    raise ValueError("spool ended before declared record count")
                yield record
            if self._spool.read(1):
                raise ValueError("spool contains undeclared trailing bytes")
        finally:
            self._iterating = False

    def invalidate(self) -> None:
        self._valid = False


def _read_stream(handle: Any) -> bytes:
    payload = bytearray()
    while chunk := handle.read(_READ_CHUNK):
        payload.extend(chunk)
        if len(payload) > MAX_EPISODE_BYTES:
            raise ValueError("episode payload exceeds 320 MiB safety limit")
    return bytes(payload)


def _decode(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed episode JSON: {label}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"malformed episode JSON object: {label}")
    return payload


def _episode_id(payload: Mapping[str, Any], label: str) -> int:
    info = payload.get("info")
    value = info.get("EpisodeId") if isinstance(info, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid canonical episode ID: {label}")
    return value


def _member_id(filename: str) -> int:
    match = _MEMBER_ID_RE.fullmatch(Path(filename).stem)
    if match is None:
        raise ValueError(f"archive member has invalid episode ID stem: {filename}")
    return int(match.group("id"))


def _validate_replay(
    payload: Mapping[str, Any], label: str, *, require_finite_rewards: bool = True
) -> tuple[list[str], list[float]]:
    info = payload.get("info")
    teams = info.get("TeamNames") if isinstance(info, dict) else None
    rewards = payload.get("rewards")
    statuses = payload.get("statuses")
    steps = payload.get("steps")
    if not isinstance(teams, list) or not teams or not all(isinstance(x, str) for x in teams):
        raise ValueError(f"team names must be nonempty strings: {label}")
    player_count = len(teams)
    if not isinstance(rewards, list) or len(rewards) != player_count:
        raise ValueError(f"rewards must match player dimension: {label}")
    if require_finite_rewards and not all(
        not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(float(x))
        for x in rewards
    ):
        raise ValueError(f"rewards must be finite numeric rewards with player dimension: {label}")
    if not isinstance(statuses, list) or len(statuses) != player_count or not all(
        isinstance(x, str) for x in statuses
    ):
        raise ValueError(f"statuses must match player dimension: {label}")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"episode steps must be nonempty: {label}")
    for step_index, step in enumerate(steps):
        if not isinstance(step, list) or len(step) != player_count:
            raise ValueError(f"episode step {step_index} has inconsistent player dimension: {label}")
        if not all(isinstance(member, dict) for member in step):
            raise ValueError(f"episode step {step_index} members must be objects: {label}")
    numeric_rewards = [float(value) for value in rewards] if require_finite_rewards else []
    return teams, numeric_rewards


def _completion_is_eligible(payload: Mapping[str, Any]) -> bool:
    statuses = payload["statuses"]
    steps = payload["steps"]
    return statuses == ["DONE"] * len(statuses) and all(
        member.get("status") == "DONE" for member in steps[-1]
    )


def _winner_index(payload: Mapping[str, Any], label: str, expert: str) -> int | None:
    teams, rewards = _validate_replay(payload, label)
    matches = [i for i, team in enumerate(teams) if normalize_team_identity(team) == normalize_team_identity(expert)]
    if len(matches) > 1:
        raise ValueError(f"ambiguous expert identity: {label}")
    if not matches:
        return None
    maximum = max(rewards)
    winners = [i for i, reward in enumerate(rewards) if reward == maximum]
    return matches[0] if winners == matches else None


def _trajectory(payload: Mapping[str, Any], player_index: int, label: str) -> tuple[Mapping[str, Any], ...]:
    statuses = payload["statuses"]
    steps = payload["steps"]
    if statuses[player_index] != "DONE" or steps[-1][player_index].get("status") != "DONE":
        raise ValueError(f"winning player trajectory is not terminal DONE: {label}")
    return tuple(step[player_index] for step in steps)


def _preflight(sources: Iterable[EpisodeSource]) -> tuple[_PreparedSource, ...]:
    prepared: list[_PreparedSource] = []
    dates: set[str] = set()
    global_ids: set[int] = set()
    for source in sources:
        _valid_date(source.date, "source")
        if source.date in dates:
            raise ValueError(f"duplicate source date: {source.date}")
        dates.add(source.date)
        expected_archive = f"pokemon-tcg-ai-battle-episodes-{source.date}.zip"
        if source.archive_path.name != expected_archive:
            raise ValueError(f"archive path filename/date mismatch: {source.archive_path}")
        if _sha256(source.archive_path) != source.archive_sha256:
            raise ValueError(f"archive SHA-256 mismatch: {source.archive_path}")
        members: list[_ArchiveMember] = []
        day_ids: set[int] = set()
        try:
            with zipfile.ZipFile(source.archive_path) as bundle:
                for entry in bundle.infolist():
                    if entry.is_dir() or not entry.filename.endswith(".json"):
                        continue
                    episode_id = _member_id(entry.filename)
                    if episode_id in day_ids:
                        raise ValueError(f"duplicate canonical episode ID {episode_id}")
                    day_ids.add(episode_id)
                    members.append(_ArchiveMember(entry.filename, episode_id))
        except zipfile.BadZipFile as error:
            raise ValueError(f"malformed ZIP archive: {source.archive_path}") from error
        patch_map: dict[int, EpisodePatch] = {}
        for item in source.patches:
            _valid_date(item.date, "patch")
            if item.date != source.date:
                raise ValueError(f"patch date mismatch: {item.path}")
            if item.path.parent.name != item.date:
                raise ValueError(f"patch path date mismatch: {item.path}")
            expected_patch = f"episode-{item.episode_id}-replay.json"
            if item.path.name != expected_patch:
                raise ValueError(f"patch path episode ID mismatch: {item.path}")
            if item.episode_id in patch_map:
                raise ValueError(f"duplicate patch declaration: {item.episode_id}")
            if _sha256(item.path) != item.sha256:
                raise ValueError(f"patch SHA-256 mismatch: {item.path}")
            patch_map[item.episode_id] = item
        canonical_ids = day_ids | set(patch_map)
        overlap = global_ids & canonical_ids
        if overlap:
            raise ValueError(f"duplicate canonical episode ID {min(overlap)} across dates")
        global_ids.update(canonical_ids)
        prepared.append(_PreparedSource(source, tuple(members), MappingProxyType(patch_map)))
    return tuple(prepared)


def _episode(
    raw: bytes,
    date: str,
    declared_id: int,
    origin: str,
    expert: str,
    kind: str,
) -> tuple[CanonicalEpisode | None, bool]:
    label = f"{date}/{declared_id}"
    payload = _decode(raw, label)
    actual_id = _episode_id(payload, label)
    if actual_id != declared_id:
        raise ValueError(f"{kind} episode ID mismatch: declared {declared_id}, payload {actual_id}")
    _validate_replay(payload, label, require_finite_rewards=False)
    if not _completion_is_eligible(payload):
        return None, True
    _validate_replay(payload, label)
    player_index = _winner_index(payload, label, expert)
    if player_index is None:
        return None, False
    frozen = _freeze(payload)
    return (
        CanonicalEpisode(
            SourceIdentity(date, declared_id, player_index, True),
            frozen,
            _trajectory(frozen, player_index, label),
            hashlib.sha256(raw).hexdigest(),
            origin,
        ),
        False,
    )


def _scan_to_spool(
    sources: Iterable[EpisodeSource], spool: Any, expert_team: str
) -> SourceAudit:
    seen = eligible = ineligible = spool_bytes = 0
    for prepared in _preflight(sources):
        source = prepared.source
        patch_ids = set(prepared.patches)
        with zipfile.ZipFile(source.archive_path) as bundle:
            for member in prepared.members:
                if member.episode_id in patch_ids:
                    continue
                with bundle.open(member.filename) as handle:
                    raw = _read_stream(handle)
                episode, noncomplete = _episode(
                    raw,
                    source.date,
                    member.episode_id,
                    f"archive:{member.filename}",
                    expert_team,
                    "archive member",
                )
                seen += 1
                ineligible += int(noncomplete)
                if episode is not None:
                    spool_bytes = _spool_episode(spool, episode, spool_bytes)
                    eligible += 1
        for episode_id in sorted(patch_ids):
            item = prepared.patches[episode_id]
            with item.path.open("rb") as handle:
                raw = _read_stream(handle)
            episode, noncomplete = _episode(
                raw,
                source.date,
                episode_id,
                f"patch:{item.path}",
                expert_team,
                "patch",
            )
            seen += 1
            ineligible += int(noncomplete)
            if episode is not None:
                spool_bytes = _spool_episode(spool, episode, spool_bytes)
                eligible += 1
    return SourceAudit(seen, eligible, ineligible, spool_bytes)


def iter_canonical_episodes(
    sources: Iterable[EpisodeSource],
    sink: Callable[[ReplayableEpisodeRecords], Any],
    *,
    expert_team: str = _EXPERT,
) -> SourceAudit:
    """Validate the whole corpus, then call ``sink`` once with a temporary replay stream.

    The sink receives a lazy, replayable spool view only after the corpus has validated. It must
    consume records during the call and atomically commit its own durable output; the view is
    invalidated when the sink returns or raises. The sink is never called when validation fails.
    The spool is outside the raw source tree and is deleted on success and failure.
    """

    free_bytes = shutil.disk_usage(tempfile.gettempdir()).free
    if free_bytes < MIN_TEMP_FREE_BYTES:
        raise OSError(
            f"insufficient temporary free space: need {MIN_TEMP_FREE_BYTES}, have {free_bytes}"
        )
    with tempfile.TemporaryFile(mode="w+b") as spool:
        audit = _scan_to_spool(sources, spool, expert_team)
        records = ReplayableEpisodeRecords(spool, audit.eligible)
        try:
            sink(records)
        finally:
            records.invalidate()
        return audit


def _scan_prepared_to_path(arguments: tuple[EpisodeSource, str, str]) -> tuple[str, SourceAudit]:
    source, expert_team, path = arguments
    with open(path, "w+b") as spool:
        audit = _scan_to_spool((source,), spool, expert_team)
    return path, audit


def iter_canonical_episodes_parallel(
    sources: Iterable[EpisodeSource],
    sink: Callable[[ReplayableEpisodeRecords], Any],
    *,
    expert_team: str = _EXPERT,
    workers: int = 4,
) -> SourceAudit:
    """Validate independent daily archives in parallel and replay in declared source order."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    prepared = _preflight(sources)
    free_bytes = shutil.disk_usage(tempfile.gettempdir()).free
    if free_bytes < MIN_TEMP_FREE_BYTES:
        raise OSError(
            f"insufficient temporary free space: need {MIN_TEMP_FREE_BYTES}, have {free_bytes}"
        )
    with tempfile.TemporaryDirectory() as directory:
        paths = [str(Path(directory) / f"source-{index:04d}.spool") for index in range(len(prepared))]
        arguments = tuple(
            (item.source, expert_team, paths[index]) for index, item in enumerate(prepared)
        )
        with ProcessPoolExecutor(max_workers=min(workers, len(arguments) or 1)) as executor:
            results = tuple(executor.map(_scan_prepared_to_path, arguments))
        seen = eligible = ineligible = spool_bytes = 0
        with tempfile.TemporaryFile(mode="w+b") as combined:
            for path, audit in results:
                if spool_bytes + audit.spool_bytes > MAX_SPOOL_BYTES:
                    raise ValueError("validated episode spool byte cap exceeded")
                with open(path, "rb") as source_handle:
                    shutil.copyfileobj(source_handle, combined, _READ_CHUNK)
                seen += audit.seen
                eligible += audit.eligible
                ineligible += audit.ineligible_noncomplete
                spool_bytes += audit.spool_bytes
            audit = SourceAudit(seen, eligible, ineligible, spool_bytes)
            records = ReplayableEpisodeRecords(combined, eligible)
            try:
                sink(records)
            finally:
                records.invalidate()
            return audit
