"""Collate serialized decision records into Semantic Goal Policy tensors."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping,Sequence
from typing import Any

import torch
from torch import Tensor


def _int(value:object,default:int=0)->int:
    return int(value) if isinstance(value,int) and not isinstance(value,bool) else default

def collate_records(records:Sequence[Mapping[str,Any]])->dict[str,Tensor]:
    if not records:raise ValueError("records must be nonempty")
    batch=len(records);max_options=max(len(row["legal_options"]) for row in records);max_entities=1;max_deck=max(len(row["deck_manifest"]["counts"]) for row in records);max_steps=max(len(row["ordered_action"])+(0 if row["action_termination"]=="forced_max" else 1) for row in records)
    out={"state_cat":torch.zeros(batch,8,dtype=torch.long),"state_num":torch.zeros(batch,16),"entities_cat":torch.zeros(batch,max_entities,8,dtype=torch.long),"entities_num":torch.zeros(batch,max_entities,12),"entity_mask":torch.zeros(batch,max_entities,dtype=torch.bool),"deck_card_ids":torch.zeros(batch,max_deck,dtype=torch.long),"deck_multiplicity":torch.zeros(batch,max_deck),"deck_mask":torch.zeros(batch,max_deck,dtype=torch.bool),"ledger_cat":torch.zeros(batch,1,6,dtype=torch.long),"ledger_num":torch.zeros(batch,1,12),"ledger_mask":torch.zeros(batch,1,dtype=torch.bool),"events_cat":torch.zeros(batch,1,6,dtype=torch.long),"events_num":torch.zeros(batch,1,8),"event_mask":torch.zeros(batch,1,dtype=torch.bool),"relations":torch.zeros(batch,max_entities,max_entities,dtype=torch.long),"options_cat":torch.zeros(batch,max_options,12,dtype=torch.long),"options_num":torch.zeros(batch,max_options,8),"option_mask":torch.zeros(batch,max_options,dtype=torch.bool),"min_count":torch.zeros(batch,dtype=torch.long),"max_count":torch.zeros(batch,dtype=torch.long),"targets":torch.full((batch,max_steps),max_options,dtype=torch.long),"target_mask":torch.zeros(batch,max_steps,dtype=torch.bool)}
    for row_index,row in enumerate(records):
        obs=row["actor_observation"];current=obs["current"];select=obs["select"];players=current["players"];actor=_int(current.get("yourIndex"));own=players[actor];opp=players[1-actor]
        out["state_cat"][row_index,:4]=torch.tensor([actor,_int(current.get("firstPlayer"))+1,_int(select.get("type"))+1,_int(select.get("context"))+1])
        out["state_num"][row_index,:8]=torch.tensor([_int(current.get("turn")),_int(current.get("turnActionCount")),_int(own.get("deckCount")),_int(opp.get("deckCount")),_int(own.get("handCount")),_int(opp.get("handCount")),_int(select.get("minCount")),_int(select.get("maxCount"))],dtype=torch.float32)
        counts=row["deck_manifest"]["counts"]
        for index,(card_id,count) in enumerate(counts):out["deck_card_ids"][row_index,index]=card_id;out["deck_multiplicity"][row_index,index]=count;out["deck_mask"][row_index,index]=True
        options=row["legal_options"]
        for index,option in enumerate(options):out["options_cat"][row_index,index,0]=_int(option.get("type"))+1;out["options_cat"][row_index,index,1]=_int(option.get("cardId"));out["option_mask"][row_index,index]=True
        minimum,maximum=_int(select.get("minCount")),_int(select.get("maxCount"));out["min_count"][row_index]=minimum;out["max_count"][row_index]=maximum
        action=list(row["ordered_action"]);targets=action if row["action_termination"]=="forced_max" else action+[max_options];out["targets"][row_index,:len(targets)]=torch.tensor(targets);out["target_mask"][row_index,:len(targets)]=True
    return out

__all__=["collate_records"]
