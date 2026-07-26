"""Formal epoch trainer with local-first canonical logging."""
from __future__ import annotations

import json
import math
import time
from collections.abc import Callable,Iterable,Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor,nn

from rl_environment.logging import TrainingLogger
from rl_environment.runs import VersionPaths,write_version_status
from .checkpoints import save_checkpoint

Batch=Mapping[str,Tensor]


def _atomic_json(path:Path,value:dict[str,Any])->None:
    temp=path.with_name(path.name+".tmp");temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n");temp.replace(path)

def train_version(paths:VersionPaths,*,model:nn.Module,optimizer:torch.optim.Optimizer,train_batches:Callable[[],Iterable[Batch]],validation_batches:Callable[[],Iterable[Batch]],loss_fn:Callable[[nn.Module,Batch],Tensor],epochs:int,config:dict[str,Any],device:torch.device)->dict[str,Any]:
    if paths.config.exists() or paths.metrics.exists() or any(paths.checkpoints.iterdir()):raise FileExistsError("formal version paths already occupied")
    _atomic_json(paths.config,config);write_version_status(paths,{"state":"training","started_at":time.time()})
    best_loss=math.inf;best_exact=-math.inf;global_step=0;history=[]
    model.to(device)
    try:
        with TrainingLogger(paths.metrics,paths.tensorboard) as logger:
            for epoch in range(1,epochs+1):
                started=time.perf_counter();model.train();train_total=train_count=0
                for batch in train_batches():
                    batch={k:v.to(device) for k,v in batch.items()};optimizer.zero_grad(set_to_none=True);loss=loss_fn(model,batch)
                    if not torch.isfinite(loss):raise FloatingPointError("nonfinite training loss")
                    loss.backward();optimizer.step();train_total+=float(loss.item());train_count+=1;global_step+=1
                model.eval();validation_total=validation_count=exact_total=0.0
                with torch.no_grad():
                    for batch in validation_batches():
                        batch={k:v.to(device) for k,v in batch.items()};loss=loss_fn(model,batch)
                        if not torch.isfinite(loss):raise FloatingPointError("nonfinite validation loss")
                        validation_total+=float(loss.item());validation_count+=1;exact_total+=float(batch.get("exact",torch.tensor(0.0,device=device)).float().mean().item())
                metrics={"trainer/epoch":epoch,"bc/train/loss":train_total/max(train_count,1),"bc/validation/loss":validation_total/max(validation_count,1),"bc/validation/exact_action":exact_total/max(validation_count,1),"system/epoch_seconds":time.perf_counter()-started,"system/parameter_count":sum(p.numel() for p in model.parameters())}
                logger.log(epoch,metrics);criteria=["latest"]
                if metrics["bc/validation/loss"]<best_loss:best_loss=metrics["bc/validation/loss"];criteria.append("best_validation_loss")
                if metrics["bc/validation/exact_action"]>best_exact:best_exact=metrics["bc/validation/exact_action"];criteria.append("best_validation_exact")
                manifest=save_checkpoint(paths.checkpoints,model=model,optimizer=optimizer,epoch=epoch,global_step=global_step,metadata={**config,"metrics":metrics},criteria=criteria)
                history.append({**metrics,"checkpoint":manifest})
        summary={"state":"complete","epochs":epochs,"global_step":global_step,"best_validation_loss":best_loss,"best_validation_exact":best_exact,"history":history}
        _atomic_json(paths.summary,summary);write_version_status(paths,{"state":"complete","completed_at":time.time()});return summary
    except BaseException as error:
        write_version_status(paths,{"state":"failed","reason":f"{type(error).__name__}: {error}","failed_at":time.time()});raise

__all__=["train_version"]
