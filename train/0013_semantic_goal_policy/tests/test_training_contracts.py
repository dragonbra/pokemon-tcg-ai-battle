from __future__ import annotations

import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

checkpoints = importlib.import_module("train.0013_semantic_goal_policy.training.checkpoints")
selection = importlib.import_module("train.0013_semantic_goal_policy.training.selection")
scheduler = importlib.import_module("train.0013_semantic_goal_policy.training.scheduler")
telemetry = importlib.import_module("train.0013_semantic_goal_policy.training.telemetry")


class TrainingContractsTest(unittest.TestCase):
    def test_content_addressed_checkpoints_and_atomic_selection_pointers(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters())
            manifest=checkpoints.save_checkpoint(root,model=model,optimizer=optimizer,epoch=1,global_step=3,metadata={"variant":"M0","dataset_sha256":"a"*64},criteria=("latest","best_validation_loss"))
            path=root/manifest["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),manifest["sha256"])
            best_before = json.loads((root/"criteria"/"best_validation_loss.json").read_text())
            second=checkpoints.save_checkpoint(root,model=model,optimizer=optimizer,epoch=2,global_step=6,metadata={"variant":"M0","dataset_sha256":"a"*64},criteria=("latest",))
            latest = json.loads((root/"criteria"/"latest.json").read_text())
            best_after = json.loads((root/"criteria"/"best_validation_loss.json").read_text())
            self.assertEqual(latest["sha256"], second["sha256"])
            self.assertEqual(best_after, best_before)
            with self.assertRaises(FileExistsError):checkpoints.save_checkpoint(root,model=model,optimizer=optimizer,epoch=2,global_step=6,metadata={"variant":"M0","dataset_sha256":"a"*64},criteria=("latest",))

    def test_selection_hard_gates_composite_and_tie_break(self):
        candidates=[
            {"path":"a.pt","sha256":"a"*64,"epoch":2,"validation_exact":0.5,"validation_loss":1.0,"decision_quality_composite":0.8,"gates":{"legality":True,"invariance":True,"counterfactual":True,"numerical":True,"runtime":True}},
            {"path":"b.pt","sha256":"b"*64,"epoch":1,"validation_exact":0.5,"validation_loss":0.9,"decision_quality_composite":0.8,"gates":{"legality":True,"invariance":True,"counterfactual":True,"numerical":True,"runtime":True}},
            {"path":"c.pt","sha256":"c"*64,"epoch":0,"validation_exact":1.0,"validation_loss":0.1,"decision_quality_composite":1.0,"gates":{"legality":False,"invariance":True,"counterfactual":True,"numerical":True,"runtime":True}},
        ]
        result=selection.select_checkpoint(candidates)
        self.assertEqual(result["selected"]["path"],"b.pt")
        self.assertIn("legality",result["rejected"][0]["reasons"])

    def test_scheduler_minimum_allocation_then_adaptive(self):
        state=scheduler.SchedulerState.new(["M0","M1","M2","M3","M4","M5"],minimum_epochs=3,target_active_seconds=36000)
        for variant in state.variants:
            self.assertEqual(state.next_variant(),variant)
            state=state.record_epoch(variant,healthy_seconds=10,score=float(variant[1]))
            state=state.record_epoch(variant,healthy_seconds=10,score=float(variant[1]))
            state=state.record_epoch(variant,healthy_seconds=10,score=float(variant[1]))
        self.assertEqual(state.next_variant(),"M5")
        self.assertFalse(state.complete)

    def test_telemetry_merges_nonoverlap_and_excludes_idle_failure_overlap(self):
        intervals=[telemetry.GpuInterval(0,10,"healthy"),telemetry.GpuInterval(5,15,"healthy"),telemetry.GpuInterval(20,25,"idle"),telemetry.GpuInterval(30,40,"failure")]
        summary=telemetry.summarize_intervals(intervals,merge_gap_seconds=0)
        self.assertEqual(summary.healthy_seconds,15)
        self.assertEqual(summary.idle_seconds,5)
        self.assertEqual(summary.failure_seconds,10)
        self.assertEqual(summary.overlap_seconds,5)


if __name__=="__main__":unittest.main()
