"""Replay only persisted Frozen error/fallback seeds with full traces."""

from __future__ import annotations

from dataclasses import replace
import json

import torch

from ..initialization import build_preset_from_common_update0
from ..integrated.presets import preset
from ..rollout import FullSemanticRolloutCollector
from ..training.run_full_semantic import FROZEN_PANEL, focal_deck, load_frozen_opponent, runtime_root
from .frozen_jobs import build_frozen_jobs
from .run_update0_frozen import RESULTS


def main() -> None:
    persisted = json.loads(RESULTS.read_text())
    seeds = {
        int(row["seed"]) for row in persisted["games"]
        if not row["valid"] or row["fallback"]
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = build_preset_from_common_update0(
        focal_deck(), preset("INTEGRATED"), device=device
    )
    opponent = load_frozen_opponent(device)
    jobs = build_frozen_jobs(
        FROZEN_PANEL, focal_deck=focal_deck(), runtime_root=runtime_root(),
        source_policy_update=0,
    )
    jobs = [replace(job, trace_policy="full") for job in jobs if job.seed in seeds]
    collector = FullSemanticRolloutCollector(
        model, opponent, device=device, worker_processes=4, engines_per_worker=4,
        inference_channels_per_role=4, mode="greedy", coalesce_ms=5.0,
    )
    for episode in collector.collect(jobs):
        diagnostics = episode.diagnostics
        row = {
            "seed": episode.job.seed, "valid": episode.valid, "error": episode.error,
            "fallback": diagnostics.get("macro_fallback"),
            "fallback_reason": diagnostics.get("macro_fallback_reason"),
            "trace_length": len(diagnostics.get("engine_event_trace") or []),
        }
        if not episode.valid:
            hits = []
            for index, event in enumerate(diagnostics.get("engine_event_trace") or []):
                observation = event.get("observation") or {}
                logs = observation.get("logs") or []
                types = [item.get("type") for item in logs if isinstance(item, dict)]
                if 22 in types or "Coin" in types:
                    prior = ((diagnostics.get("engine_event_trace") or [])[max(0, index - 1)]
                             .get("observation") or {})
                    hits.append({
                        "event_index": index,
                        "turn": (observation.get("current") or {}).get("turn"),
                        "context": (observation.get("select") or {}).get("context"),
                        "log_types": types,
                        "recent_logs": logs[-4:],
                        "prior_focal_active": ((((prior.get("current") or {}).get("players") or [{}, {}])[0]
                                                  or {}).get("active")),
                        "prior_focal_player": (((prior.get("current") or {}).get("players") or [{}, {}])[0]),
                    })
            row["coin_log_hits"] = hits[-3:]
        print(json.dumps(row, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
