"""Render an immutable 0042 periodic Frozen-0809 CUDA-2048 JSON as HTML."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid

from .render_custom_deck_cuda2048 import _audit_from_report, render


ROOT = Path(__file__).resolve().parents[3]
REPORT_DATA = re.compile(
    r'(<script id="report-data" type="application/json">)(.*?)(</script>)',
    re.S,
)


def _load(path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entries = result.get("entries")
    candidate = result.get("candidate_deployment_identity_audit") or {}
    opponent = result.get("policy_identity_audit") or {}
    if (
        result.get("schema_version") != "0042_frozen_per_game_results_policy0809_v1"
        or result.get("benchmark_kind") != "cuda_2048"
        or result.get("expected_games") != 2048
        or not isinstance(entries, list)
        or len(entries) != 2048
    ):
        raise ValueError("source is not a complete 0042 Frozen-0809 CUDA-2048 result")
    if (
        candidate.get("status") != "PASS"
        or candidate.get("contract_id") != "kaggle_fp16_storage_fp32_runtime_v1"
        or candidate.get("storage_dtype") != "fp16"
        or candidate.get("runtime_dtype") != "fp32"
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != "Policy-0809"
    ):
        raise ValueError("candidate/opponent deployment identity audit is not PASS")
    if any(
        row.get("valid") is not True
        or row.get("error") is not None
        or row.get("semantic_fallback") is not False
        for row in entries
    ):
        raise ValueError("result contains invalid, error, or semantic-fallback games")
    if len({row.get("game_id") for row in entries}) != 2048 or len(
        {row.get("seed") for row in entries}
    ) != 2048:
        raise ValueError("result does not contain 2,048 unique game IDs and seeds")
    if int(result.get("checkpoint_update", -1)) != int(candidate.get("checkpoint_update", -2)):
        raise ValueError("checkpoint update and candidate audit disagree")
    return result, config


def _normalize(result: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    entries = [
        {**row, "opponent_id": row["opponent"]}
        for row in result["entries"]
    ]
    wins = sum(row["outcome"] == 1 for row in entries)
    losses = sum(row["outcome"] == -1 for row in entries)
    draws = sum(row["outcome"] == 0 for row in entries)
    first = [row for row in entries if row["focal_first"]]
    second = [row for row in entries if not row["focal_first"]]
    deck_path = Path(config["focal_deck_path"])
    if not deck_path.is_absolute():
        deck_path = ROOT / deck_path
    cards = [int(line) for line in deck_path.read_text(encoding="utf-8").splitlines()]
    if len(cards) != 60:
        raise ValueError("configured focal deck is not exact 60")
    return {
        "status": "PASS",
        "checkpoint_update": result["checkpoint_update"],
        "focal_deck_cards": cards,
        "focal_deck_id": config["focal_deck_id"],
        "focal_deck_display_name": config["focal_deck_display_name"],
        "focal_exact_deck_sha256": config["focal_exact_deck_sha256"],
        "schedule_sha256": result["schedule_sha256"],
        "candidate_deployment_identity_audit": result[
            "candidate_deployment_identity_audit"
        ],
        "opponent_policy_identity_audit": result["policy_identity_audit"],
        "summary": {
            "games": 2048,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / 2048,
            "focal_first_games": len(first),
            "focal_first_win_rate": sum(row["outcome"] == 1 for row in first) / len(first),
            "focal_second_games": len(second),
            "focal_second_win_rate": sum(row["outcome"] == 1 for row in second) / len(second),
        },
        "health": {
            "eval/core/chance_boundaries": sum(bool(row["chance_boundary"]) for row in entries),
            "eval/core/semantic_fallbacks": 0,
            "eval/core/errors": 0,
            "eval/core/unfinished": 0,
        },
        "entries": entries,
    }


def _index_payload(report: dict[str, Any], source: Path) -> dict[str, Any]:
    summary = report["summary"]
    persisted_at = datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat()
    return {
        "manifest": {
            "run_id": (
                f'0042-v9-u{int(report["checkpoint_update"]):03d}-'
                "periodic-frozen0809-cuda2048"
            ),
            "candidate": {
                "name": report["focal_deck_id"],
                "display_name": (
                    f'{report["focal_deck_display_name"]} · V9 U'
                    f'{int(report["checkpoint_update"])}'
                ),
            },
            "finished_at": persisted_at,
            "source_result": str(source.relative_to(ROOT)),
            "evidence_kind": "periodic_frozen_0809_cuda_2048_rendered_snapshot",
        },
        "summary": {
            "total_games": summary["games"],
            "wins": summary["wins"],
            "losses": summary["losses"],
            "draws": summary["draws"],
            "errors": 0,
            "unfinished": 0,
            "win_rate": summary["win_rate"],
            "completion_rate": 1.0,
        },
        "report": report,
    }


def render_file(source: Path, config_path: Path, output: Path) -> dict[str, Any]:
    result, config = _load(source, config_path)
    report = _normalize(result, config)
    audit = _audit_from_report(report)
    document = render(report, audit)
    document = document.replace(
        "0042 candidate vs full immutable Policy-0809 · Official CUDA engine",
        "0042 candidate vs full immutable Policy-0809 · Official CUDA engine · 基于已持久化 periodic JSON 渲染，未重跑",
    ).replace(
        "2,048 terminal · 0 error · 0 unsupported · 0 semantic fallback",
        "2,048 terminal · 0 error · 0 semantic fallback",
    ).replace(
        '<div class="label">Feature residency</div><div class="pass">GPU resident · feature D2H bytes = 0</div>',
        '<div class="label">Feature residency</div><div>该 periodic 结果 JSON 未内嵌此项 telemetry，本报告不作声明</div>',
    )
    embedded = json.dumps(
        _index_payload(report, source), ensure_ascii=False, separators=(",", ":")
    ).replace("</", "<\\/")
    document, replacements = REPORT_DATA.subn(
        lambda match: f"{match.group(1)}{embedded}{match.group(3)}", document, count=1
    )
    if replacements != 1:
        raise RuntimeError("rendered HTML has no unique report-data payload")
    if output.exists():
        raise FileExistsError(f"evaluation report already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return _index_payload(report, source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = render_file(args.source.resolve(), args.config.resolve(), args.output.resolve())
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
