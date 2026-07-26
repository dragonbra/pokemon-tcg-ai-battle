"""Deterministic episode-player split with private details and aggregate-only audit."""
from __future__ import annotations
import hashlib, json, re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date as calendar_date
from types import MappingProxyType
from typing import Any
from ..protocol import PreRunProtocol

_HEX = re.compile(r"^[0-9a-f]{64}$")
_VERSION = "semantic_goal_policy_split_v1"
_GROUP_HASH_VERSION = "episode_player_deck_sha256_v1"
_SEED = 20260726
_DIGEST_INPUT = "semantic_goal_policy_split_v1|20260726|<date>|<deck_hash>|<episode_id>|<player>"

@dataclass(frozen=True, slots=True)
class SplitGroup:
    date: str; deck_manifest_hash: str; episode_id: int; player_index: int
    def __post_init__(self) -> None:
        if not isinstance(self.date, str): raise TypeError("date must be a string")
        try: parsed = calendar_date.fromisoformat(self.date)
        except ValueError as error: raise ValueError(f"invalid date: {self.date!r}") from error
        if parsed.isoformat() != self.date: raise ValueError(f"invalid date: {self.date!r}")
        if not isinstance(self.deck_manifest_hash, str) or not _HEX.fullmatch(self.deck_manifest_hash): raise ValueError("deck_manifest_hash must be a lowercase SHA-256 hex digest")
        for label, value in (("episode_id",self.episode_id),("player_index",self.player_index)):
            if isinstance(value,bool) or not isinstance(value,int): raise TypeError(f"{label} must be an integer")
            if value < 0: raise ValueError(f"{label} must be non-negative")
    @property
    def trajectory_key(self) -> tuple[str,int,int]: return self.date,self.episode_id,self.player_index
    @property
    def stratum(self) -> str: return f"{self.date}|{self.deck_manifest_hash}"

@dataclass(frozen=True, slots=True)
class SplitAssignment:
    group_hash: str; group_hash_version: str; ranking_digest: str; algorithm_version: str; split: str

@dataclass(frozen=True, slots=True)
class SplitAudit:
    algorithm: str; digest_algorithm: str; digest_input: str; seed: int; protocol_hash: str; group_unit: str
    counts: Mapping[str,int]; validation_target: int; validation_actual: int; validation_delta: int
    private_assignments_count: int; private_assignments_sha256: str; stratum_size_histogram: Mapping[str,int]
    large_strata_count: int; rare_strata_count: int; rare_groups_count: int; rare_validation_groups: int
    def as_dict(self) -> dict[str,object]:
        return {"algorithm":self.algorithm,"counts":dict(self.counts),"digest_algorithm":self.digest_algorithm,"digest_input":self.digest_input,"group_unit":self.group_unit,"large_strata_count":self.large_strata_count,"private_assignments_count":self.private_assignments_count,"private_assignments_sha256":self.private_assignments_sha256,"protocol_hash":self.protocol_hash,"rare_groups_count":self.rare_groups_count,"rare_strata_count":self.rare_strata_count,"rare_validation_groups":self.rare_validation_groups,"seed":self.seed,"stratum_size_histogram":dict(self.stratum_size_histogram),"validation_actual":self.validation_actual,"validation_delta":self.validation_delta,"validation_target":self.validation_target}

@dataclass(frozen=True, slots=True)
class SplitManifest:
    audit: SplitAudit | None; assignments: Mapping[str,tuple[SplitAssignment,...]]; private_strata: Mapping[str,Mapping[str,int]]
    def canonical_bytes(self) -> bytes:
        if self.audit is None: raise ValueError("private manifest has no audit")
        return _bytes(self.audit.as_dict())
    def sha256(self) -> str: return hashlib.sha256(self.canonical_bytes()).hexdigest()
    def private_canonical_bytes(self) -> bytes:
        return _bytes({"algorithm_version":_VERSION,"assignments":{k:[{"algorithm_version":x.algorithm_version,"group_hash":x.group_hash,"group_hash_version":x.group_hash_version,"ranking_digest":x.ranking_digest,"split":x.split} for x in v] for k,v in self.assignments.items()},"strata":{k:dict(v) for k,v in self.private_strata.items()}})
    def private_sha256(self) -> str: return hashlib.sha256(self.private_canonical_bytes()).hexdigest()
    def assignment_for(self, group: SplitGroup) -> SplitAssignment:
        if self.audit is None: raise ValueError("split manifest has no committed audit")
        if self.private_sha256()!=self.audit.private_assignments_sha256: raise ValueError("private split assignment commitment mismatch")
        wanted=_group_hash(group)
        matches=[item for values in self.assignments.values() for item in values if item.group_hash==wanted]
        if len(matches)!=1: raise ValueError("trajectory group is absent or ambiguous in split manifest")
        return matches[0]

def _bytes(value: object) -> bytes: return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode("utf-8")
def _digest(group: SplitGroup) -> str: return hashlib.sha256("|".join((_VERSION,str(_SEED),group.date,group.deck_manifest_hash,str(group.episode_id),str(group.player_index))).encode("ascii")).hexdigest()
def _group_hash(group: SplitGroup) -> str: return hashlib.sha256(_bytes((group.date,group.deck_manifest_hash,group.episode_id,group.player_index))).hexdigest()
def _half_up_tenth(count: int) -> int: return (count+5)//10
def _target(count: int) -> int:
    if count==0: raise ValueError("groups must not be empty")
    return 0 if count==1 else max(1,min(count-1,_half_up_tenth(count)))
def _protocol(value: PreRunProtocol|Mapping[str,Any]) -> PreRunProtocol:
    result=value if isinstance(value,PreRunProtocol) else PreRunProtocol(value)
    if not isinstance(result,PreRunProtocol): raise TypeError("protocol must be a PreRunProtocol or mapping")
    result.validate()
    if result.data["split"]["seed"]!=_SEED: raise ValueError("split.seed must equal 20260726")
    if result.data["split"]["digest_input"]!=_DIGEST_INPUT: raise ValueError("split.digest_input must equal frozen digest input")
    return result

def assign_groups(groups: Sequence[SplitGroup], protocol: PreRunProtocol|Mapping[str,Any]) -> SplitManifest:
    """Constrained Hamilton apportionment; compact audit contains aggregate distributions only."""
    frozen,values=_protocol(protocol),tuple(groups)
    if not values: raise ValueError("groups must not be empty")
    if not all(isinstance(x,SplitGroup) for x in values): raise TypeError("groups must contain only SplitGroup values")
    if len({x.trajectory_key for x in values})!=len(values): raise ValueError("duplicate episode-player identity")
    strata: dict[str,list[tuple[str,SplitGroup]]]={}
    for x in values: strata.setdefault(x.stratum,[]).append((_digest(x),x))
    for members in strata.values(): members.sort(key=lambda x:(x[0],x[1].trajectory_key))
    quotas={}; remainders={}
    for key,members in strata.items():
        size=len(members); quotas[key]=max(1 if size>=10 else 0,size//10); remainders[key]=size%10
    target=_target(len(values)); order=sorted(strata,key=lambda k:(-remainders[k],hashlib.sha256(k.encode()).hexdigest()))
    while sum(quotas.values())<target:
        progressed=False
        for key in order: # one Hamilton remainder seat per stratum per round
            maximum=len(strata[key])-1 if len(strata[key])>=10 else len(strata[key])
            if quotas[key]<maximum:
                quotas[key]+=1; progressed=True
                if sum(quotas.values())==target: break
        if not progressed: raise ValueError("validation target is infeasible")
    while sum(quotas.values())>target:
        progressed=False
        for key in reversed(order):
            minimum=1 if len(strata[key])>=10 else 0
            if quotas[key]>minimum:
                quotas[key]-=1; progressed=True
                if sum(quotas.values())==target: break
        if not progressed: raise ValueError("validation target violates large-stratum constraints")
    output={"train":[],"validation":[],"test":[]}; private_strata={}
    for key,members in sorted(strata.items()):
        quota=quotas[key]; private_strata[key]=MappingProxyType({"groups":len(members),"train_groups":len(members)-quota,"validation_groups":quota})
        for index,(digest,group) in enumerate(members):
            bucket="validation" if index<quota else "train"; output[bucket].append(SplitAssignment(_group_hash(group),_GROUP_HASH_VERSION,digest,_VERSION,bucket))
    for v in output.values(): v.sort(key=lambda x:x.group_hash)
    private=MappingProxyType({k:tuple(v) for k,v in output.items()}); pstrata=MappingProxyType(private_strata)
    private_manifest=SplitManifest(None,private,pstrata); histogram=Counter(len(v) for v in strata.values()); actual=len(private["validation"])
    audit=SplitAudit("constrained_hamilton_half_up_v1","sha256",_DIGEST_INPUT,_SEED,frozen.sha256(),"complete_episode_player",MappingProxyType({k:len(v) for k,v in private.items()}),target,actual,actual-target,len(values),private_manifest.private_sha256(),MappingProxyType({str(k):v for k,v in sorted(histogram.items())}),sum(len(v)>=10 for v in strata.values()),sum(len(v)<10 for v in strata.values()),sum(len(v) for v in strata.values() if len(v)<10),sum(quotas[k] for k,v in strata.items() if len(v)<10))
    return SplitManifest(audit,private,pstrata)
