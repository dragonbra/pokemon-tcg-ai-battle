from __future__ import annotations

import importlib
from pathlib import Path


module = importlib.import_module("train.0043_champion_league_rl.evaluation.schedule")
ROOT = Path(__file__).resolve().parents[1]
FOCAL = "a" * 64


def test_cpu_and_cuda_share_exact_frequency_unit() -> None:
    cpu = module.materialize(ROOT, focal_deck_id="048", focal_deployment_identity=FOCAL, replicas=1)
    cuda = module.materialize(ROOT, focal_deck_id="048", focal_deployment_identity=FOCAL, replicas=8)
    assert len(cpu["jobs"]) == 256
    assert len(cuda["jobs"]) == 2048
    assert cpu["base_schedule_sha256"] == cuda["base_schedule_sha256"]
    assert cpu["jobs"] == cuda["jobs"][:256]
    assert len({row["engine_seed"] for row in cuda["jobs"]}) == 2048


def test_schedule_is_identity_bound_and_pinned() -> None:
    first = module.materialize(ROOT, focal_deck_id="048", focal_deployment_identity=FOCAL, replicas=1)
    second = module.materialize(ROOT, focal_deck_id="048", focal_deployment_identity="b" * 64, replicas=1)
    assert first["base_schedule_sha256"] == "2f465d715174afd0d9957101d9d1a475d13bea576b92a9742449e58066c0c070"
    assert first["schedule_sha256"] != second["schedule_sha256"]
