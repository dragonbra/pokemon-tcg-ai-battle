"""Evaluate the 0029 multi-memory semantic checkpoint over exact League decks."""

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
import zipfile

from evaluation.reporting.index import _embedded_report_data


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEAGUE_CATALOG = REPOSITORY_ROOT / (
    "rl_runs/0022_league_training/versions/V11_multidecoder_league_20h/"
    "artifact/league_catalog.json"
)
CHECKPOINT_ZIP = REPOSITORY_ROOT / "experiments/0029_large/semantic.zip"
CHECKPOINT = REPOSITORY_ROOT / (
    ".tmp/model_loading/0029_large/semantic/best_greedy_exact.pt"
)
CG_SOURCE = REPOSITORY_ROOT / "evaluation/arena/frozen/_policy/cg"
TEMP_ROOT = REPOSITORY_ROOT / ".tmp/evaluation/0029_frozen_test"
PACKAGE_ROOT = REPOSITORY_ROOT / ".tmp/model_loading/0029_large/candidates"
OUTPUT_ROOT = REPOSITORY_ROOT / "evaluation/arena/combat_mat/0029_frozen_test"
POOL_ID = "0019_foundation_51_exact_decks_v4"
PROJECT_ID = "0029_multi_deck_semantic_foundation_bc"
EXPECTED_GAMES = 510
PRIORITY_DECKS = (
    "marnies_grimmsnarl_ex_froslass_001",
    "dragapult_ex_001",
    "mega_lucario_ex_solrock_001",
    "mega_lopunny_ex_mega_froslass_ex_001",
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


def load_league_decks(path: Path = LEAGUE_CATALOG) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decks = payload.get("decks")
    if not isinstance(decks, list) or len(decks) != 48:
        raise ValueError("0022 V11 League catalog must contain exactly 48 decks")
    by_id: dict[str, dict[str, object]] = {}
    for raw in decks:
        if not isinstance(raw, dict):
            raise ValueError("League deck record must be an object")
        deck_id = raw.get("deck_id")
        deck = raw.get("deck")
        if not isinstance(deck_id, str) or deck_id in by_id:
            raise ValueError("League deck IDs must be unique strings")
        if not isinstance(deck, list) or len(deck) != 60 or not all(
            type(card_id) is int and card_id > 0 for card_id in deck
        ):
            raise ValueError(f"League deck is not exact 60: {deck_id}")
        if _deck_hash(deck) != raw.get("deck_sha256"):
            raise ValueError(f"League deck hash mismatch: {deck_id}")
        by_id[deck_id] = raw
    return [
        by_id[deck_id]
        for deck_id in PRIORITY_DECKS
    ] + [raw for raw in decks if raw["deck_id"] not in PRIORITY_DECKS]


def ensure_checkpoint(path: Path = CHECKPOINT, archive: Path = CHECKPOINT_ZIP) -> Path:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            member = "semantic/best_greedy_exact.pt"
            info = bundle.getinfo(member)
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with bundle.open(info) as source, temporary.open("xb") as target:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    contract = metadata.get("model_config") if isinstance(metadata, dict) else None
    if (
        set(payload) != {"schema_version", "state_dict", "metadata"}
        or payload["schema_version"] != "0025_model_only_checkpoint_v1"
        or metadata.get("arm") != "semantic"
        or metadata.get("project_id") != PROJECT_ID
        or not isinstance(contract, dict)
        or contract.get("model") != "SemanticFoundationPolicy"
    ):
        raise ValueError("checkpoint does not match the audited 0029 semantic contract")
    return path


def materialize_package(
    deck_record: dict[str, object], checkpoint: Path, package_root: Path = PACKAGE_ROOT
) -> Path:
    deck_id = str(deck_record["deck_id"])
    output = package_root / deck_id
    if output.exists():
        manifest_path = output / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file()
            else {}
        )
        if (
            manifest.get("checkpoint_sha256") == _sha256(checkpoint)
            and manifest.get("deck_sha256") == deck_record["deck_sha256"]
            and manifest.get("deck_id") == deck_id
        ):
            return output
        raise FileExistsError(f"conflicting candidate package already exists: {output}")

    deck_path = package_root / "decks" / f"{deck_id}.csv"
    deck_path.parent.mkdir(parents=True, exist_ok=True)
    deck_path.write_text(
        "".join(f"{card_id}\n" for card_id in deck_record["deck"]), encoding="ascii"
    )
    exporter = importlib.import_module(
        "train.0025_semantic_foundation_pretraining.export_candidate"
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
) -> dict[str, object]:
    payload = _embedded_report_data(report_path)
    manifest = payload.get("manifest")
    summary = payload.get("summary")
    games = payload.get("games")
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
        or package.get("arm") != "semantic"
        or package.get("checkpoint_sha256") != checkpoint_sha256
        or package.get("deck_id") != deck_record["deck_id"]
        or package.get("deck_sha256") != deck_record["deck_sha256"]
    ):
        raise ValueError("evaluation report candidate provenance mismatch")
    if Counter(candidate.get("deck", [])) != Counter(deck_record["deck"]):
        raise ValueError("evaluation report exact deck mismatch")
    if (
        manifest.get("games") != EXPECTED_GAMES
        or len(manifest.get("opponents", [])) != 51
        or summary.get("total_games") != EXPECTED_GAMES
        or summary.get("completed_games") != EXPECTED_GAMES
        or summary.get("errors") != 0
        or summary.get("unfinished") != 0
        or len(games) != EXPECTED_GAMES
    ):
        raise ValueError("evaluation report is partial or contains errors")

    turn_order: dict[str, dict[str, int]] = {
        "first": {"wins": 0, "losses": 0, "draws": 0},
        "second": {"wins": 0, "losses": 0, "draws": 0},
    }
    for game in games:
        if not isinstance(game, dict) or game.get("status") != "finished":
            raise ValueError("evaluation report contains an unfinished game")
        order = "first" if game.get("candidate_first") else "second"
        winner = game.get("winner")
        key = "wins" if winner == 0 else "losses" if winner == 1 else "draws"
        turn_order[order][key] += 1
    return {
        "deck_id": deck_record["deck_id"],
        "display_name": deck_record["display_name"],
        "deck_sha256": deck_record["deck_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "pool_id": POOL_ID,
        "run_id": manifest.get("run_id"),
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "wall_time_seconds": manifest.get("wall_time_seconds"),
        "games": EXPECTED_GAMES,
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
        with source.open("rb") as read_handle, temporary.open("xb") as write_handle:
            shutil.copyfileobj(read_handle, write_handle)
            write_handle.flush()
            os.fsync(write_handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


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
    manifest = {
        "schema_version": "0029_semantic_frozen_test_v1",
        "project_id": PROJECT_ID,
        "checkpoint": str(CHECKPOINT.relative_to(REPOSITORY_ROOT)),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_selection": "best_greedy_exact",
        "league_catalog": str(LEAGUE_CATALOG.relative_to(REPOSITORY_ROOT)),
        "league_catalog_sha256": _sha256(LEAGUE_CATALOG),
        "league_deck_count": len(decks),
        "pool_id": POOL_ID,
        "games_per_opponent": 10,
        "opponents_per_report": 51,
        "published_reports": len(records),
        "published_games": sum(int(record["games"]) for record in records),
        "priority_decks": list(PRIORITY_DECKS),
        "reports": records,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    rows = "".join(_index_row(record) for record in records)
    if not rows:
        rows = '<tr><td colspan="10" class="muted">No completed reports</td></tr>'
    document = _index_html(len(decks), records, rows, checkpoint_sha256)
    temporary = output_root / f".index.html.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(document, encoding="utf-8")
        temporary.replace(output_root / "index.html")
    finally:
        temporary.unlink(missing_ok=True)
    return records


def _index_row(record: dict[str, object]) -> str:
    first = record["turn_order"]["first"]
    second = record["turn_order"]["second"]
    return (
        "<tr>"
        f'<td><a href="{html.escape(str(record["report"]), quote=True)}">{html.escape(str(record["display_name"]))}</a>'
        f'<small>{html.escape(str(record["deck_id"]))}</small></td>'
        f'<td>{record["wins"]}-{record["losses"]}-{record["draws"]}</td>'
        f'<td>{float(record["win_rate"]):.2%}</td>'
        f'<td>{first["wins"]}-{first["losses"]}-{first["draws"]}</td>'
        f'<td>{second["wins"]}-{second["losses"]}-{second["draws"]}</td>'
        f'<td>{record["games"]}</td><td>{record["errors"]}</td>'
        f'<td>{float(record["wall_time_seconds"]):.1f}s</td>'
        f'<td><code>{html.escape(str(record["run_id"]))}</code></td>'
        f'<td><code>{html.escape(str(record["deck_sha256"]))[:12]}</code></td>'
        "</tr>"
    )


def _index_html(
    deck_count: int,
    records: list[dict[str, object]],
    rows: str,
    checkpoint_sha256: str,
) -> str:
    games = sum(int(record["games"]) for record in records)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>0029 Semantic · Frozen Test</title><style>
:root{{--bg:#f4f7f6;--panel:#fff;--ink:#17231f;--muted:#687870;--line:#d9e3de;--accent:#176b4d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px 20px 56px}}h1{{margin:0;font-size:28px}}p{{color:var(--muted)}}
.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:20px 0}}
.stat{{padding:14px;border:1px solid var(--line);border-radius:6px;background:var(--panel)}}.stat b{{display:block;font-size:22px}}
.table{{overflow:auto;border:1px solid var(--line);background:var(--panel)}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}
th,td{{padding:10px 11px;border-bottom:1px solid var(--line);text-align:right}}th{{background:#eaf1ed;font-size:12px}}
th:first-child,td:first-child{{text-align:left}}small{{display:block;color:var(--muted)}}a{{color:var(--accent);font-weight:650;text-decoration:none}}
code{{font-size:11px}}.muted{{color:var(--muted)}}@media(max-width:720px){{.stats{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main><h1>0029 Semantic · Frozen Test</h1>
<p>Multi-memory semantic checkpoint zero-shot evaluation. Every published report is 51 Frozen opponents × 10 official-engine games. Priority order: Dragapult, Lucario, Mega Lopunny, Raging Bolt.</p>
<div class="stats"><div class="stat"><b>{len(records)}/{deck_count}</b>League decks</div><div class="stat"><b>{games}</b>official games</div><div class="stat"><b>{sum(int(r['errors']) for r in records)}</b>errors</div><div class="stat"><b>51 × 10</b>per report</div></div>
<p><code>checkpoint {html.escape(checkpoint_sha256)}</code> · <code>{POOL_ID}</code></p>
<div class="table"><table><thead><tr><th>Deck / report</th><th>W-L-D</th><th>Win rate</th><th>First</th><th>Second</th><th>Games</th><th>Error</th><th>Wall</th><th>Run ID</th><th>Deck hash</th></tr></thead><tbody>{rows}</tbody></table></div>
</main></body></html>"""


def run_deck(
    deck_record: dict[str, object], checkpoint: Path, *, keep_candidate: bool = False
) -> dict[str, object]:
    checkpoint_sha256 = _sha256(checkpoint)
    existing = OUTPUT_ROOT / "reports" / f"{deck_record['deck_id']}.html"
    if existing.is_file():
        return validate_report(existing, deck_record, checkpoint_sha256)
    package = materialize_package(deck_record, checkpoint)
    subprocess.run(
        [sys.executable, "-m", "evaluation", "validate", str(package)],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    temp_output = TEMP_ROOT / str(deck_record["deck_id"])
    before = set(temp_output.glob("run-*/report.html")) if temp_output.exists() else set()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation",
            "run",
            "--candidate",
            str(package),
            "--opponents",
            "all",
            "--games",
            "10",
            "--output",
            str(temp_output),
            "--workers",
            "8",
            "--worker-cpu-threads",
            "1",
            "--candidate-device",
            "cuda:0",
            "--opponent-device",
            "cuda:0",
            "--metric-profile",
            "league_deck_quality",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    created = set(temp_output.glob("run-*/report.html")) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one new evaluation report, found {len(created)}")
    record = publish_report(created.pop(), deck_record, checkpoint_sha256)
    if not keep_candidate:
        shutil.rmtree(package)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--keep-candidates", action="store_true")
    args = parser.parse_args(argv)
    decks = load_league_decks()
    by_id = {str(deck["deck_id"]): deck for deck in decks}
    requested = list(by_id) if args.all else args.deck or list(PRIORITY_DECKS)
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        parser.error(f"unknown League decks: {', '.join(unknown)}")
    checkpoint = ensure_checkpoint()
    checkpoint_sha256 = _sha256(checkpoint)
    refresh_index(decks, checkpoint_sha256)
    for deck_id in requested:
        record = run_deck(
            by_id[deck_id], checkpoint, keep_candidate=args.keep_candidates
        )
        refresh_index(decks, checkpoint_sha256)
        print(
            f"0029_FROZEN_COMPLETE deck={deck_id} run_id={record['run_id']} "
            f"record={record['wins']}-{record['losses']}-{record['draws']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
