"""Evaluate the 0031 epoch-6 checkpoint on selected Frozen exact decks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from evaluation.reporting.index import _embedded_report_data


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = REPOSITORY_ROOT / "evaluation/arena/frozen"
CHECKPOINT = REPOSITORY_ROOT / (
    "rl_runs/0031_rule_faithful_semantic_foundation_pretraining/versions/"
    "V4_lr5e4_no_early_stop_b512/checkpoint/rule_faithful_semantic/latest.pt"
)
CG_SOURCE = REPOSITORY_ROOT / "evaluation/arena/frozen/_policy/cg"
PACKAGE_ROOT = REPOSITORY_ROOT / ".tmp/model_loading/0031_epoch6/candidates"
TEMP_ROOT = REPOSITORY_ROOT / ".tmp/evaluation/0031_epoch6_frozen_test"
OUTPUT_ROOT = REPOSITORY_ROOT / "evaluation/arena/combat_mat/0031_epoch6_frozen_test"
PROJECT_ID = "0031_rule_faithful_semantic_foundation_pretraining"
VERSION = "V4_lr5e4_no_early_stop_b512"
ARM = "rule_faithful_semantic"
POOL_ID = "0019_foundation_51_exact_decks_v4"
EXPECTED_GAMES = 510
DECKS = (
    "mega_lopunny_ex_001",
    "dragapult_ex_001",
    "dragapult_ex_dusknoir_001",
    "alakazam_dudunsparce_003",
    "marnies_grimmsnarl_ex_froslass_limitless",
    "raging_bolt_ex_james_cox_henry_chao_001",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deck_hash(deck: list[int]) -> str:
    encoded = ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_decks(path: Path = FROZEN_ROOT) -> list[dict[str, object]]:
    result = []
    for deck_id in DECKS:
        deck_root = path / deck_id
        record = json.loads((deck_root / "manifest.json").read_text(encoding="utf-8"))
        deck = [
            int(line)
            for line in (deck_root / "deck.csv").read_text(encoding="ascii").splitlines()
            if line.strip()
        ]
        if (
            record.get("deck_id") != deck_id
            or len(deck) != 60
            or not all(type(card_id) is int and card_id > 0 for card_id in deck)
            or record.get("deck_sha256") != _deck_hash(deck)
        ):
            raise ValueError(f"Frozen deck identity mismatch: {deck_id}")
        record["deck"] = deck
        result.append(record)
    return result


def validate_checkpoint(path: Path = CHECKPOINT) -> dict[str, object]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    contract = metadata.get("model_config") if isinstance(metadata, dict) else None
    if (
        set(payload) != {"schema_version", "state_dict", "metadata"}
        or payload.get("schema_version") != "0031_model_only_checkpoint_v1"
        or not isinstance(metadata, dict)
        or metadata.get("project_id") != PROJECT_ID
        or metadata.get("version") != VERSION
        or metadata.get("arm") != ARM
        or metadata.get("epoch") != 6
        or metadata.get("global_step") != 99006
        or not isinstance(contract, dict)
        or contract.get("model") != "SemanticPolicy"
    ):
        raise ValueError("checkpoint does not match the audited 0031 latest contract")
    return metadata


def materialize_package(
    deck_record: dict[str, object], checkpoint: Path = CHECKPOINT
) -> Path:
    deck_id = str(deck_record["deck_id"])
    output = PACKAGE_ROOT / deck_id
    checkpoint_sha256 = _sha256(checkpoint)
    if output.exists():
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        if (
            manifest.get("checkpoint_sha256") == checkpoint_sha256
            and manifest.get("deck_sha256") == deck_record["deck_sha256"]
            and manifest.get("deck_id") == deck_id
        ):
            return output
        raise FileExistsError(f"conflicting candidate package already exists: {output}")
    deck_path = PACKAGE_ROOT.parent / "decks" / f"{deck_id}.csv"
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    deck_path.write_text(
        "".join(f"{card_id}\n" for card_id in deck_record["deck"]), encoding="ascii"
    )
    exporter = importlib.import_module(
        "train.0031_rule_faithful_semantic_foundation_pretraining.export_candidate"
    )
    exporter.export_candidate(
        checkpoint=checkpoint,
        deck_path=deck_path,
        cg_source=CG_SOURCE,
        output=output,
        deck_id=deck_id,
    )
    return output


def validate_report(
    report_path: Path,
    deck_record: dict[str, object],
    checkpoint_sha256: str,
    *,
    expected_games: int = EXPECTED_GAMES,
) -> dict[str, object]:
    payload = _embedded_report_data(report_path)
    manifest, summary, games = (
        payload.get("manifest"), payload.get("summary"), payload.get("games")
    )
    if not isinstance(manifest, dict) or not isinstance(summary, dict) or not isinstance(games, list):
        raise ValueError("evaluation report payload is incomplete")
    candidate = manifest.get("candidate")
    package = candidate.get("package_manifest") if isinstance(candidate, dict) else None
    pool = manifest.get("opponent_pool")
    if not isinstance(candidate, dict) or not isinstance(package, dict):
        raise ValueError("evaluation report lacks candidate identity")
    if not isinstance(pool, dict) or pool.get("pool_id") != POOL_ID:
        raise ValueError("evaluation report Frozen pool identity mismatch")
    if (
        package.get("project_id") != PROJECT_ID
        or package.get("version") != VERSION
        or package.get("arm") != ARM
        or package.get("checkpoint_selection") != "latest"
        or package.get("checkpoint_sha256") != checkpoint_sha256
        or package.get("deck_id") != deck_record["deck_id"]
        or package.get("deck_sha256") != deck_record["deck_sha256"]
        or Counter(candidate.get("deck", [])) != Counter(deck_record["deck"])
    ):
        raise ValueError("evaluation report candidate provenance mismatch")
    if (
        manifest.get("games") != expected_games
        or len(manifest.get("opponents", [])) != 51
        or summary.get("total_games") != expected_games
        or summary.get("completed_games") != expected_games
        or summary.get("errors") != 0
        or summary.get("unfinished") != 0
        or len(games) != expected_games
    ):
        raise ValueError("evaluation report is partial or contains errors")
    turn_order = {
        "first": {"wins": 0, "losses": 0, "draws": 0},
        "second": {"wins": 0, "losses": 0, "draws": 0},
    }
    for game in games:
        if not isinstance(game, dict) or game.get("status") != "finished":
            raise ValueError("evaluation report contains an unfinished game")
        order = "first" if game.get("candidate_first") else "second"
        key = "wins" if game.get("winner") == 0 else "losses" if game.get("winner") == 1 else "draws"
        turn_order[order][key] += 1
    return {
        "deck_id": deck_record["deck_id"],
        "display_name": deck_record["display_name"],
        "deck_sha256": deck_record["deck_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "pool_id": POOL_ID,
        "run_id": manifest.get("run_id"),
        "wall_time_seconds": manifest.get("wall_time_seconds"),
        "games": expected_games,
        "wins": summary.get("wins"),
        "losses": summary.get("losses"),
        "draws": summary.get("draws"),
        "errors": 0,
        "completion_rate": summary.get("completion_rate"),
        "win_rate": summary.get("win_rate"),
        "turn_order": turn_order,
        "report_sha256": _sha256(report_path),
        "report": f"reports/{deck_record['deck_id']}.html",
    }


def publish_report(
    source: Path,
    deck_record: dict[str, object],
    checkpoint_sha256: str,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, object]:
    record = validate_report(source, deck_record, checkpoint_sha256)
    target = output_root / str(record["report"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = validate_report(target, deck_record, checkpoint_sha256)
        if existing["report_sha256"] != record["report_sha256"]:
            raise FileExistsError(f"refusing to overwrite a different report: {target}")
        return existing
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def refresh_index(
    decks: list[dict[str, object]],
    checkpoint_sha256: str,
    output_root: Path = OUTPUT_ROOT,
) -> list[dict[str, object]]:
    records = []
    for deck in decks:
        report = output_root / "reports" / f"{deck['deck_id']}.html"
        if report.is_file():
            records.append(validate_report(report, deck, checkpoint_sha256))
    support_path = output_root / "cuda_support.json"
    support = None
    if support_path.is_file():
        candidate = json.loads(support_path.read_text(encoding="utf-8"))
        if candidate.get("schema_version") != "cuda_frozen51_support_v1":
            raise ValueError("unexpected CUDA support report schema")
        support = candidate
    manifest = {
        "schema_version": "0031_epoch6_frozen_test_v1",
        "project_id": PROJECT_ID,
        "version": VERSION,
        "checkpoint": str(CHECKPOINT.relative_to(REPOSITORY_ROOT)),
        "checkpoint_selection": "latest",
        "checkpoint_epoch": 6,
        "checkpoint_global_step": 99006,
        "checkpoint_sha256": checkpoint_sha256,
        "frozen_manifest_sha256": _sha256(FROZEN_ROOT / "manifest.json"),
        "pool_id": POOL_ID,
        "games_per_opponent": 10,
        "opponents_per_report": 51,
        "requested_decks": list(DECKS),
        "published_reports": len(records),
        "published_games": sum(int(record["games"]) for record in records),
        "reports": records,
    }
    if support is not None:
        manifest["cuda_support"] = {
            "report": "cuda_support.html",
            "json": "cuda_support.json",
            "report_sha256": _sha256(support_path),
            "supported_decks": support["summary"]["supported"],
            "unsupported_decks": support["summary"]["unsupported"],
            "ordered_matchups": support["summary"]["ordered_matchups"],
            "decisions_compared": support["summary"]["decisions_compared"],
            "policy_adapter_status": "blocked_0031_observation_contract",
        }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = "".join(
        "<tr>"
        f'<td><a href="{html.escape(str(record["report"]))}">{html.escape(str(record["display_name"]))}</a>'
        f'<small>{html.escape(str(record["deck_id"]))}</small></td>'
        f'<td>{record["wins"]}-{record["losses"]}-{record["draws"]}</td>'
        f'<td>{float(record["win_rate"]):.2%}</td><td>{record["games"]}</td>'
        f'<td>{float(record["wall_time_seconds"]):.1f}s</td><td><code>{record["run_id"]}</code></td></tr>'
        for record in records
    ) or '<tr><td colspan="6">No completed reports</td></tr>'
    support_banner = ""
    if support is not None:
        support_banner = (
            '<p><a href="cuda_support.html">CUDA Frozen51 support audit</a> · '
            f'{support["summary"]["supported"]}/51 supported · '
            f'{support["summary"]["ordered_matchups"]:,} ordered parity matchups · '
            '<a href="THROUGHPUT.md">throughput</a> · '
            '<a href="RL_READINESS.md">RL readiness</a></p>'
            '<p><strong>Boundary:</strong> engine admission is complete, but the 0031 '
            'chronological observation/event contract is not yet available from the resident GPU state. '
            'The reports below remain official CPU-engine strength evaluations.</p>'
        )
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>0031 Epoch 6 Frozen Test</title>
<style>body{{font:14px/1.5 system-ui;margin:0;background:#f4f7f6;color:#17231f}}main{{max-width:1200px;margin:auto;padding:28px 20px}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:10px;border:1px solid #d9e3de;text-align:right}}th:first-child,td:first-child{{text-align:left}}small{{display:block;color:#687870}}a{{color:#176b4d;font-weight:650}}</style>
</head><body><main><h1>0031 Epoch 6 · Frozen Arena</h1><p>V4 epoch 6 / step 99006 latest checkpoint. Each complete report contains 51 Frozen opponents × 10 official-engine games.</p>
<p><code>{checkpoint_sha256}</code> · {len(records)}/{len(DECKS)} reports · {sum(int(r['games']) for r in records)} games</p>
{support_banner}
<table><thead><tr><th>Deck / report</th><th>W-L-D</th><th>Win rate</th><th>Games</th><th>Wall</th><th>Run ID</th></tr></thead><tbody>{rows}</tbody></table></main></body></html>'''
    (output_root / "index.html").write_text(document, encoding="utf-8")
    return records


def _run_evaluation(package: Path, deck_id: str, games: int, output: Path) -> Path:
    before = set(output.glob("run-*/report.html")) if output.exists() else set()
    subprocess.run(
        [
            sys.executable, "-m", "evaluation", "--pool", "frozen", "run",
            "--candidate", str(package), "--opponents", "all", "--games", str(games),
            "--output", str(output), "--workers", "8", "--worker-cpu-threads", "1",
            "--candidate-device", "cuda:0", "--opponent-device", "cuda:0",
            "--metric-profile", "league_deck_quality",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    created = set(output.glob("run-*/report.html")) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new {deck_id} report, found {len(created)}")
    return created.pop()


def run_deck(deck_record: dict[str, object], checkpoint_sha256: str) -> dict[str, object]:
    existing = OUTPUT_ROOT / "reports" / f"{deck_record['deck_id']}.html"
    if existing.is_file():
        return validate_report(existing, deck_record, checkpoint_sha256)
    package = materialize_package(deck_record)
    subprocess.run(
        [sys.executable, "-m", "evaluation", "validate", str(package)],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    source = _run_evaluation(
        package, str(deck_record["deck_id"]), 10, TEMP_ROOT / str(deck_record["deck_id"])
    )
    return publish_report(source, deck_record, checkpoint_sha256)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", action="append", choices=DECKS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    decks = load_decks()
    by_id = {str(deck["deck_id"]): deck for deck in decks}
    requested = args.deck or list(DECKS)
    validate_checkpoint()
    checkpoint_sha256 = _sha256(CHECKPOINT)
    if args.smoke:
        deck = by_id[requested[0]]
        refresh_index(decks, checkpoint_sha256)
        record = run_deck(deck, checkpoint_sha256)
        refresh_index(decks, checkpoint_sha256)
        print(
            f"0031_FROZEN_SMOKE_COMPLETE deck={deck['deck_id']} "
            f"run_id={record['run_id']} record={record['wins']}-{record['losses']}-{record['draws']}"
        )
        return 0
    refresh_index(decks, checkpoint_sha256)
    for deck_id in requested:
        record = run_deck(by_id[deck_id], checkpoint_sha256)
        refresh_index(decks, checkpoint_sha256)
        print(
            f"0031_FROZEN_COMPLETE deck={deck_id} run_id={record['run_id']} "
            f"record={record['wins']}-{record['losses']}-{record['draws']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
