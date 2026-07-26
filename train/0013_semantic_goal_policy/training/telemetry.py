"""Non-overlapping GPU activity accounting."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

@dataclass(frozen=True,slots=True)
class GpuInterval:
    start:float;end:float;kind:str
    def __post_init__(self)->None:
        if self.end<self.start or self.kind not in {"healthy","idle","failure"}:raise ValueError("invalid GPU interval")

@dataclass(frozen=True,slots=True)
class GpuTimeSummary:
    healthy_seconds:float;idle_seconds:float;failure_seconds:float;overlap_seconds:float

def _union(intervals:Sequence[GpuInterval])->float:
    values=sorted((item.start,item.end) for item in intervals);total=0.0;end=None
    for start,stop in values:
        if end is None or start>end:total+=stop-start;end=stop
        elif stop>end:total+=stop-end;end=stop
    return total

def summarize_intervals(intervals:Sequence[GpuInterval],*,merge_gap_seconds:float)->GpuTimeSummary:
    healthy=[item for item in intervals if item.kind=="healthy"];raw=sum(item.end-item.start for item in healthy);unique=_union(healthy)
    return GpuTimeSummary(unique,sum(item.end-item.start for item in intervals if item.kind=="idle"),sum(item.end-item.start for item in intervals if item.kind=="failure"),raw-unique)

__all__=["GpuInterval","GpuTimeSummary","summarize_intervals"]
