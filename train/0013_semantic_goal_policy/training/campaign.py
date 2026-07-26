"""GPU campaign entrypoint for immutable M0-M5 formal versions."""
from __future__ import annotations

import argparse
import torch

PROJECT_ID="0013_semantic_goal_policy"


def _batch(batch_size:int,device:torch.device)->dict[str,torch.Tensor]:
    options=48
    return {"state_cat":torch.randint(0,64,(batch_size,8),device=device),"state_num":torch.randn(batch_size,16,device=device),"entities_cat":torch.randint(0,64,(batch_size,24,8),device=device),"entities_num":torch.randn(batch_size,24,12,device=device),"entity_mask":torch.ones(batch_size,24,dtype=torch.bool,device=device),"deck_card_ids":torch.randint(1,1268,(batch_size,22),device=device),"deck_multiplicity":torch.ones(batch_size,22,device=device),"deck_mask":torch.ones(batch_size,22,dtype=torch.bool,device=device),"ledger_cat":torch.randint(0,64,(batch_size,24,6),device=device),"ledger_num":torch.randn(batch_size,24,12,device=device),"ledger_mask":torch.ones(batch_size,24,dtype=torch.bool,device=device),"events_cat":torch.randint(0,64,(batch_size,64,6),device=device),"events_num":torch.randn(batch_size,64,8,device=device),"event_mask":torch.ones(batch_size,64,dtype=torch.bool,device=device),"relations":torch.zeros(batch_size,24,24,dtype=torch.long,device=device),"options_cat":torch.randint(0,64,(batch_size,options,12),device=device),"options_num":torch.randn(batch_size,options,8,device=device),"option_mask":torch.ones(batch_size,options,dtype=torch.bool,device=device),"min_count":torch.ones(batch_size,dtype=torch.long,device=device),"max_count":torch.full((batch_size,),3,dtype=torch.long,device=device)}

def run(variant:str,version:str,epochs:int,steps_per_epoch:int,batch_size:int)->None:
    raise RuntimeError(
        "synthetic campaign batches are smoke-only and cannot allocate a formal training version; "
        "use the record-backed trainer after dataset publication"
    )

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--variant",required=True);parser.add_argument("--version",required=True);parser.add_argument("--epochs",type=int,default=3);parser.add_argument("--steps-per-epoch",type=int,default=1000);parser.add_argument("--batch-size",type=int,default=16);args=parser.parse_args();run(args.variant,args.version,args.epochs,args.steps_per_epoch,args.batch_size)
if __name__=="__main__":main()
