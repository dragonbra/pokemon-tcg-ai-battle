"""Fast schema discovery and mini-corpus extraction before formal dataset publication."""
from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from collections.abc import Iterable,Mapping,Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .source import EpisodeSource,normalize_team_identity

@dataclass(frozen=True,slots=True)
class SchemaAudit:
    episodes:int;eligible:int;observation_fields:tuple[str,...];current_fields:tuple[str,...];player_fields:tuple[str,...];entity_fields:tuple[str,...];select_fields:tuple[str,...];option_fields:tuple[str,...];log_fields:tuple[str,...];shape_counts:Mapping[str,int]
    def as_dict(self)->dict[str,object]:return {"episodes":self.episodes,"eligible":self.eligible,"observation_fields":list(self.observation_fields),"current_fields":list(self.current_fields),"player_fields":list(self.player_fields),"entity_fields":list(self.entity_fields),"select_fields":list(self.select_fields),"option_fields":list(self.option_fields),"log_fields":list(self.log_fields),"shape_counts":dict(self.shape_counts)}

def _fields(value:object,target:set[str])->None:
    if isinstance(value,Mapping):target.update(str(key) for key in value)

def _eligible(payload:Mapping[str,Any],expert:str)->int|None:
    info=payload.get("info");teams=info.get("TeamNames") if isinstance(info,Mapping) else None;rewards=payload.get("rewards");statuses=payload.get("statuses")
    if not isinstance(teams,list) or not isinstance(rewards,list) or not isinstance(statuses,list) or statuses!=["DONE"]*len(teams) or len(rewards)!=len(teams):return None
    matches=[i for i,name in enumerate(teams) if normalize_team_identity(name)==normalize_team_identity(expert)]
    if len(matches)!=1 or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in rewards):return None
    winners=[i for i,value in enumerate(rewards) if value==max(rewards)]
    return matches[0] if winners==matches else None

def audit_sources(sources:Sequence[EpisodeSource],*,expert:str="Yushin Ito",sample_per_date:int=4,output_dir:Path|None=None)->SchemaAudit:
    sets={name:set() for name in ("observation","current","player","entity","select","option","log")};shapes=Counter();seen=eligible=0
    if output_dir is not None:output_dir.mkdir(parents=True,exist_ok=True)
    for source in sources:
        candidates=[]
        with zipfile.ZipFile(source.archive_path) as bundle:
            for entry in bundle.infolist():
                if not entry.filename.endswith(".json"):continue
                seen+=1
                with bundle.open(entry) as handle:payload=json.load(handle)
                actor=_eligible(payload,expert)
                if actor is None:continue
                eligible+=1;frames=payload["steps"][0][0].get("visualize")
                if not isinstance(frames,list):continue
                local_fields=set();decision_count=0
                for frame in frames:
                    obs=frame.get("obs");current=obs.get("current") if isinstance(obs,Mapping) else None
                    if not isinstance(current,Mapping) or current.get("yourIndex")!=actor:continue
                    decision_count+=frame.get("selected") is not None;_fields(obs,sets["observation"]);_fields(current,sets["current"]);select=obs.get("select");_fields(select,sets["select"])
                    shapes[f"looking:{type(current.get('looking')).__name__}"]+=1;shapes[f"select.deck:{type(select.get('deck') if isinstance(select,Mapping) else None).__name__}"]+=1
                    for player in current.get("players") or []:
                        _fields(player,sets["player"])
                        if isinstance(player,Mapping):
                            for zone in ("active","bench","hand","discard","prize"):
                                for entity in player.get(zone) or []:_fields(entity,sets["entity"])
                    if isinstance(select,Mapping):
                        for option in select.get("option") or []:_fields(option,sets["option"])
                    for log in obs.get("logs") or []:_fields(log,sets["log"])
                novelty=sum(len(sets[name]&local_fields) for name in sets);candidates.append((decision_count+novelty,entry.filename,payload,actor))
        if output_dir is not None:
            for _,filename,payload,actor in sorted(candidates,reverse=True)[:sample_per_date]:
                destination=output_dir/f"{source.date}-{Path(filename).stem}-p{actor}.json";destination.write_text(json.dumps(payload,separators=(",",":"),allow_nan=False))
    return SchemaAudit(seen,eligible,*(tuple(sorted(sets[name])) for name in ("observation","current","player","entity","select","option","log")),dict(sorted(shapes.items())))

__all__=["SchemaAudit","audit_sources"]
