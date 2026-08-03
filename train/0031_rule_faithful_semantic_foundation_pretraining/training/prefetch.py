"""Bounded background preparation for CPU training batches."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import queue
import threading
import time
from types import TracebackType
from typing import Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _Message(Generic[T]):
    kind: str
    value: T | None = None
    error: BaseException | None = None
    traceback: TracebackType | None = None


class PrefetchIterator(Iterator[T], Generic[T]):
    """Prepare a bounded number of items while the consumer is working."""

    def __init__(self, iterable: Iterable[T], depth: int = 2):
        if depth < 1:
            raise ValueError("prefetch depth must be positive")
        self._source = iter(iterable)
        self._queue: queue.Queue[_Message[T]] = queue.Queue(maxsize=depth)
        self._cancel = threading.Event()
        self._closed = False
        self._terminal = False
        self.produced = 0
        self.consumed = 0
        self.wait_seconds = 0.0
        self._thread = threading.Thread(
            target=self._produce,
            name="0031-prefetch-producer",
            daemon=True,
        )
        self._thread.start()

    def _put(self, message: _Message[T]) -> bool:
        while not self._cancel.is_set():
            try:
                self._queue.put(message, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            for item in self._source:
                if not self._put(_Message(kind="item", value=item)):
                    return
                self.produced += 1
        except BaseException as error:
            self._put(
                _Message(
                    kind="error",
                    error=error,
                    traceback=error.__traceback__,
                )
            )
            return
        self._put(_Message(kind="end"))

    def __iter__(self) -> PrefetchIterator[T]:
        return self

    def __next__(self) -> T:
        if self._terminal:
            raise StopIteration
        started = time.perf_counter()
        message = self._queue.get()
        self.wait_seconds += time.perf_counter() - started
        if message.kind == "item":
            self.consumed += 1
            return message.value  # type: ignore[return-value]
        self._terminal = True
        if message.kind == "error":
            assert message.error is not None
            raise message.error.with_traceback(message.traceback)
        raise StopIteration

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cancel.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("prefetch producer did not stop")

    def __enter__(self) -> PrefetchIterator[T]:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["PrefetchIterator"]
