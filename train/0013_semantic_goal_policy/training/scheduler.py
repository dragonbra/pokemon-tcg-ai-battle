"""Deterministic adaptive training scheduler state."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping,Sequence


@dataclass(frozen=True,slots=True)
class VariantProgress:
    epochs:int=0;healthy_seconds:float=0.0;score:float=float("-inf");failed:bool=False

@dataclass(frozen=True,slots=True)
class SchedulerState:
    variants:tuple[str,...];minimum_epochs:int;target_active_seconds:float;progress:Mapping[str,VariantProgress];healthy_seconds:float=0.0
    @classmethod
    def new(cls,variants:Sequence[str],*,minimum_epochs:int,target_active_seconds:float)->SchedulerState:
        values=tuple(variants)
        return cls(values,minimum_epochs,target_active_seconds,MappingProxyType({name:VariantProgress() for name in values}))
    @property
    def complete(self)->bool:return self.healthy_seconds>=self.target_active_seconds and all(value.failed or value.epochs>=self.minimum_epochs for value in self.progress.values())
    def next_variant(self)->str|None:
        for name in self.variants:
            value=self.progress[name]
            if not value.failed and value.epochs<self.minimum_epochs:return name
        healthy=[(value.score,name) for name,value in self.progress.items() if not value.failed]
        return max(healthy)[1] if healthy and not self.complete else None
    def record_epoch(self,variant:str,*,healthy_seconds:float,score:float)->SchedulerState:
        if variant not in self.progress or healthy_seconds<0:raise ValueError("invalid scheduler update")
        updated=dict(self.progress);old=updated[variant];updated[variant]=VariantProgress(old.epochs+1,old.healthy_seconds+healthy_seconds,score,old.failed)
        return SchedulerState(self.variants,self.minimum_epochs,self.target_active_seconds,MappingProxyType(updated),self.healthy_seconds+healthy_seconds)
    def record_failure(self,variant:str)->SchedulerState:
        updated=dict(self.progress);old=updated[variant];updated[variant]=VariantProgress(old.epochs,old.healthy_seconds,old.score,True)
        return SchedulerState(self.variants,self.minimum_epochs,self.target_active_seconds,MappingProxyType(updated),self.healthy_seconds)

__all__=["SchedulerState","VariantProgress"]
