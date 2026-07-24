"""Reusable bounded-memory iterators for large training corpora."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar


Record = dict[str, Any]
T = TypeVar("T")
Key = TypeVar("Key")


def iter_jsonl(path: str | Path) -> Iterator[Record]:
    """Read one JSON object per line without materializing the full file."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record at {source}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at {source}:{line_number}")
            yield record


def shuffle_buffer(
    records: Iterable[T], *, seed: int, buffer_size: int
) -> Iterator[T]:
    """Approximate a deterministic shuffle while keeping bounded memory."""
    if buffer_size < 1:
        raise ValueError("buffer_size must be positive")
    if buffer_size == 1:
        yield from records
        return
    generator = random.Random(seed)
    buffer: list[T] = []
    for record in records:
        if len(buffer) < buffer_size:
            buffer.append(record)
            continue
        index = generator.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = record
    generator.shuffle(buffer)
    yield from buffer


def chunks(records: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Yield stable batches without requiring a sized collection."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunk: list[T] = []
    for record in records:
        chunk.append(record)
        if len(chunk) == batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def aligned_records(
    primary: Iterable[T],
    sidecar: Iterable[T],
    *,
    identity: Callable[[T], Key],
) -> Iterator[tuple[T, T]]:
    """Pair two ordered streams and fail closed on missing, extra, or shifted rows."""
    sidecar_iter = iter(sidecar)
    for row in primary:
        try:
            annotation = next(sidecar_iter)
        except StopIteration as exc:
            raise ValueError("sidecar ended before the primary stream") from exc
        row_identity = identity(row)
        annotation_identity = identity(annotation)
        if row_identity != annotation_identity:
            raise ValueError(
                "stream identity mismatch: "
                f"primary={row_identity!r}, sidecar={annotation_identity!r}"
            )
        yield row, annotation
    try:
        extra = next(sidecar_iter)
    except StopIteration:
        return
    raise ValueError(f"sidecar has extra record {identity(extra)!r}")
