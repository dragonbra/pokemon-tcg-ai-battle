"""Render stopped V8 three-pool history under 29-way and Value-loss 15-way taxonomies."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ..assets import AssetRegistry
from ..opponent_meta import OpponentArchetypeTaxonomy


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION = "V8_dragapult_007_u282_aggressive_meta_quota_policy0809_eval2"
VERSION_ROOT = ROOT / "runs/versions" / VERSION
V4_VERSION = "V4_dragapult_007_u200_policy0809_three_pool"
V7_VERSION = "V7_dragapult_007_u276_aggressive_meta_quota_policy0809"
V8_VERSION = VERSION
UPDATES = (
    tuple(range(200, 276, 5))
    + tuple(range(276, 282))
    + tuple(range(282, 299, 2))
)
REPORT_SPECS = (
    tuple((update, V4_VERSION) for update in range(200, 276, 5))
    + tuple((update, V7_VERSION) for update in range(276, 282))
    + tuple((update, V8_VERSION) for update in range(282, 299, 2))
)
REPORT_PATHS = tuple(
    ROOT / "runs/versions" / version
    / f"artifact/periodic_evaluation/update-{update:06d}/report.json"
    for update, version in REPORT_SPECS
)
CUDA2048_UPDATES = (40,) + tuple(range(90, 201, 5)) + (282,)
CUDA2048_PATHS = tuple(
    (
        ROOT
        / "runs/versions"
        / (
            "V2_dragapult_007_expert_cold_start_lr"
            if update == 40
            else "V7_dragapult_007_u276_aggressive_meta_quota_policy0809"
            if update == 282
            else "V3_dragapult_007_expert_continue_u45"
        )
        / "artifact/formal_evaluation"
        / (
            f"update-{update:06d}_benchmark_v2_cuda2048_identity_fixed/report.json"
            if update == 190
            else f"update-{update:06d}_benchmark_v2_cuda2048/report.json"
        )
    )
    for update in CUDA2048_UPDATES
)
OUTPUT = (
    ROOT / "docs/model/evaluation"
    / "V8_dragapult_007_u200_u298_policy0809_three_pool_cuda512.html"
)
ORACLE_REPORT = (
    ROOT / "runs/versions/"
    "V9_dragapult_007_meta_oracle_v1_cuda2048/artifact/oracle_evaluation/report.json"
)
V9_OUTPUT = (
    ROOT / "docs/model/evaluation/"
    "V9_dragapult_007_meta_oracle_v1_cuda2048.html"
)
CURRENT_TAXONOMY = PROJECT_ROOT / "assets/taxonomy/own_archetypes_v2.json"
CURRENT_MAPPING = PROJECT_ROOT / "assets/taxonomy/deck_own_archetype_mapping_v2.json"


@dataclass(frozen=True, slots=True)
class OutcomeStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.games if self.games else None


@dataclass(frozen=True, slots=True)
class HistoryAudit:
    updates: tuple[int, ...]
    current_names: Mapping[int, str]
    value_names: Mapping[int, str]
    active_current_ids: tuple[int, ...]
    current: Mapping[int, Mapping[int, OutcomeStats]]
    value: Mapping[int, Mapping[int, OutcomeStats]]
    overall: Mapping[int, OutcomeStats]
    pools: Mapping[str, Mapping[int, OutcomeStats]]
    deck_to_current: Mapping[str, int]
    deck_to_value: Mapping[str, int]
    current_to_value: Mapping[int, tuple[int, ...]]
    value_to_current: Mapping[int, tuple[int, ...]]
    current_to_decks: Mapping[int, tuple[str, ...]]
    value_to_decks: Mapping[int, tuple[str, ...]]
    report_audits: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class TradeoffDiagnostic:
    left: int
    right: int
    full_correlation: float | None
    aggressive_correlation: float | None
    full_opposing_steps: int
    full_joint_steps: int
    aggressive_opposing_steps: int
    aggressive_joint_steps: int
    full_left_net: int
    full_right_net: int
    aggressive_left_net: int
    aggressive_right_net: int
    left_games: int
    right_games: int

    @property
    def full_net_exchange(self) -> bool:
        return self.full_left_net * self.full_right_net < 0

    @property
    def aggressive_net_exchange(self) -> bool:
        return self.aggressive_left_net * self.aggressive_right_net < 0


def _stats(rows: Iterable[Mapping[str, object]]) -> OutcomeStats:
    wins = losses = draws = 0
    for row in rows:
        outcome = int(row["outcome"])
        if outcome == 1:
            wins += 1
        elif outcome == -1:
            losses += 1
        elif outcome == 0:
            draws += 1
        else:
            raise ValueError(f"invalid outcome: {outcome}")
    return OutcomeStats(wins, losses, draws)


def aggregate_entries(
    entries: Sequence[Mapping[str, object]],
    *,
    deck_to_current: Mapping[str, int],
    deck_to_value: Mapping[str, int],
) -> tuple[dict[int, OutcomeStats], dict[int, OutcomeStats]]:
    current_rows: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    value_rows: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in entries:
        deck_id = str(row["opponent_id"])
        if deck_id not in deck_to_current or deck_id not in deck_to_value:
            raise KeyError(f"unmapped opponent deck: {deck_id}")
        reported = int(row["opponent_meta_archetype_id"])
        if reported != deck_to_current[deck_id]:
            raise RuntimeError(
                f"entry current Meta mismatch for deck {deck_id}: "
                f"reported={reported} mapped={deck_to_current[deck_id]}"
            )
        current_rows[reported].append(row)
        value_rows[deck_to_value[deck_id]].append(row)
    return (
        {key: _stats(value) for key, value in current_rows.items()},
        {key: _stats(value) for key, value in value_rows.items()},
    )


def _load_taxonomies() -> tuple[
    dict[int, str], dict[int, str], dict[str, int], dict[str, int], dict[str, str]
]:
    current_raw = json.loads(CURRENT_TAXONOMY.read_text(encoding="utf-8"))
    current_names = {
        int(row["archetype_id"]): str(row["display_name"])
        for row in current_raw["classes"]
    }
    if tuple(current_names) != tuple(range(29)):
        raise RuntimeError("current taxonomy is not exact 29-way own_archetypes_v2")

    mapping_raw = json.loads(CURRENT_MAPPING.read_text(encoding="utf-8"))
    deck_to_current = {
        str(row["deck_id"]): int(row["archetype_id"])
        for row in mapping_raw["decks"]
    }
    if len(deck_to_current) != len(mapping_raw["decks"]):
        raise RuntimeError("duplicate deck IDs in current taxonomy mapping")

    registry = AssetRegistry.load(PROJECT_ROOT)
    registry.validate_all()
    value_taxonomy = OpponentArchetypeTaxonomy.load()
    value_names = {row.class_id: row.name for row in value_taxonomy.classes}
    deck_to_value: dict[str, int] = {}
    deck_names: dict[str, str] = {}
    for row in registry.decks:
        deck_id = str(row.deck_id)
        cards = tuple(
            int(value)
            for value in (PROJECT_ROOT / row.deck_path)
            .read_text(encoding="utf-8")
            .splitlines()
            if value.strip()
        )
        if len(cards) != 60:
            raise RuntimeError(f"deck {deck_id} is not exact 60 cards")
        deck_to_value[deck_id] = value_taxonomy.classify_target(cards).value
        deck_names[deck_id] = str(row.name)
    if set(deck_to_current) != set(deck_to_value):
        raise RuntimeError("current and Value-loss deck inventories differ")
    return current_names, value_names, deck_to_current, deck_to_value, deck_names


def _validate_report(
    report: Mapping[str, object], *, update: int, deck_to_current: Mapping[str, int]
) -> None:
    if report.get("status") != "PASS" or report.get("focal_checkpoint_update") != update:
        raise RuntimeError(f"U{update} report identity/status failed")
    summary = report.get("summary")
    entries = report.get("entries")
    if not isinstance(summary, Mapping) or not isinstance(entries, list):
        raise RuntimeError(f"U{update} report structure failed")
    if int(summary.get("games", -1)) != 1536 or len(entries) != 1536:
        raise RuntimeError(f"U{update} is not complete three-pool 1536")
    if any(
        not bool(row.get("valid")) or row.get("error") is not None
        for row in entries
    ):
        raise RuntimeError(f"U{update} contains invalid/error games")
    candidate = report.get("focal_policy_identity_audit")
    opponent = report.get("opponent_policy_identity_audit")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("status") != "PASS"
        or candidate.get("contract_id") != "kaggle_fp16_storage_fp32_runtime_v1"
        or candidate.get("checkpoint_update") != update
    ):
        raise RuntimeError(f"U{update} candidate deployment identity failed")
    if (
        not isinstance(opponent, Mapping)
        or opponent.get("status") != "PASS"
        or opponent.get("requested_policy_id") != "Policy-0809"
    ):
        raise RuntimeError(f"U{update} opponent identity failed")
    schedule = report.get("schedule")
    if (
        not isinstance(schedule, Mapping)
        or schedule.get("contract_id")
        != "0045_policy0809_three_meta_pools_common_seeds_cuda512_v1"
        or schedule.get("games") != 1536
        or schedule.get("common_random_numbers") is not True
    ):
        raise RuntimeError(f"U{update} schedule contract failed")
    for row in entries:
        deck_id = str(row["opponent_id"])
        if deck_id not in deck_to_current:
            raise KeyError(f"unmapped opponent deck: {deck_id}")


def load_history(report_paths: tuple[Path, ...]) -> HistoryAudit:
    if len(report_paths) != len(UPDATES):
        raise RuntimeError("V8 history requires exact U282..U298 report inventory")
    current_names, value_names, deck_to_current, deck_to_value, _ = _load_taxonomies()
    current: dict[int, dict[int, OutcomeStats]] = {
        class_id: {} for class_id in current_names
    }
    value: dict[int, dict[int, OutcomeStats]] = {
        class_id: {} for class_id in value_names
    }
    overall: dict[int, OutcomeStats] = {}
    pools: dict[str, dict[int, OutcomeStats]] = {
        name: {} for name in ("low_score", "priority", "remaining")
    }
    audits: list[Mapping[str, object]] = []
    common_pool_hashes: dict[str, str] | None = None
    opponent_hash: str | None = None
    for expected_update, path in zip(UPDATES, report_paths, strict=True):
        report = json.loads(path.read_text(encoding="utf-8"))
        _validate_report(report, update=expected_update, deck_to_current=deck_to_current)
        entries = report["entries"]
        current_rows, value_rows = aggregate_entries(
            entries, deck_to_current=deck_to_current, deck_to_value=deck_to_value
        )
        for class_id in current_names:
            current[class_id][expected_update] = current_rows.get(class_id, OutcomeStats())
        for class_id in value_names:
            value[class_id][expected_update] = value_rows.get(class_id, OutcomeStats())
        overall[expected_update] = _stats(entries)
        for pool in pools:
            pools[pool][expected_update] = _stats(
                row for row in entries if row["pool"] == pool
            )

        # Cross-check the renderer's current aggregation against the source report.
        for class_id, source in report["per_meta"].items():
            calculated = current[int(class_id)][expected_update]
            if (
                calculated.wins != int(source["wins"])
                or calculated.losses != int(source["losses"])
                or calculated.draws != int(source["draws"])
                or calculated.games != int(source["games"])
            ):
                raise RuntimeError(f"U{expected_update} current Meta aggregate mismatch")
        schedule = report["schedule"]
        hashes = {
            pool: str(schedule["pools"][pool]["common_random_schedule_sha256"])
            for pool in pools
        }
        if common_pool_hashes is None:
            common_pool_hashes = hashes
        elif common_pool_hashes != hashes:
            raise RuntimeError("V8 reports do not share exact pool schedules")
        effective = str(report["opponent_policy_identity_audit"]["effective_policy_sha256"])
        if opponent_hash is None:
            opponent_hash = effective
        elif opponent_hash != effective:
            raise RuntimeError("V8 opponent effective identity changed")
        audits.append({
            "update": expected_update,
            "path": str(path.relative_to(ROOT)),
            "status": report["status"],
            "candidate_status": report["focal_policy_identity_audit"]["status"],
            "candidate_effective_sha256": report["focal_policy_identity_audit"]["effective_candidate_sha256"],
            "opponent_status": report["opponent_policy_identity_audit"]["status"],
            "opponent_effective_sha256": effective,
            "pool_schedule_sha256": hashes,
        })

    current_to_value_sets: dict[int, set[int]] = defaultdict(set)
    value_to_current_sets: dict[int, set[int]] = defaultdict(set)
    current_to_decks: dict[int, list[str]] = defaultdict(list)
    value_to_decks: dict[int, list[str]] = defaultdict(list)
    for deck_id, current_id in deck_to_current.items():
        value_id = deck_to_value[deck_id]
        current_to_value_sets[current_id].add(value_id)
        value_to_current_sets[value_id].add(current_id)
        current_to_decks[current_id].append(deck_id)
        value_to_decks[value_id].append(deck_id)
    return HistoryAudit(
        updates=UPDATES,
        current_names=current_names,
        value_names=value_names,
        active_current_ids=tuple(class_id for class_id in current_names if current[class_id][UPDATES[0]].games),
        current=current,
        value=value,
        overall=overall,
        pools=pools,
        deck_to_current=deck_to_current,
        deck_to_value=deck_to_value,
        current_to_value={
            key: tuple(sorted(value)) for key, value in current_to_value_sets.items()
        },
        value_to_current={
            key: tuple(sorted(value)) for key, value in value_to_current_sets.items()
        },
        current_to_decks={key: tuple(sorted(value)) for key, value in current_to_decks.items()},
        value_to_decks={key: tuple(sorted(value)) for key, value in value_to_decks.items()},
        report_audits=tuple(audits),
    )


def load_default_history() -> HistoryAudit:
    return load_history(REPORT_PATHS)


def load_cuda2048_history() -> HistoryAudit:
    """Load every valid Dragapult-007 CUDA-2048 checkpoint report on disk."""
    current_names, value_names, deck_to_current, deck_to_value, _ = _load_taxonomies()
    current = {class_id: {} for class_id in current_names}
    value = {class_id: {} for class_id in value_names}
    overall: dict[int, OutcomeStats] = {}
    audits: list[Mapping[str, object]] = []
    schedule_hash: str | None = None
    opponent_hash: str | None = None
    for update, path in zip(CUDA2048_UPDATES, CUDA2048_PATHS, strict=True):
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report.get("summary")
        entries = report.get("entries")
        candidate = report.get("focal_policy_identity_audit")
        opponent = report.get("opponent_policy_identity_audit")
        schedule = report.get("schedule")
        expected_policy_id = f"0045-single-deck-expert-007-update-{update:06d}"
        if (
            report.get("status") != "PASS"
            or report.get("focal_checkpoint_update") != update
            or report.get("focal_policy_id") != expected_policy_id
            or not isinstance(summary, Mapping)
            or int(summary.get("games", -1)) != 2048
            or not isinstance(entries, list)
            or len(entries) != 2048
        ):
            raise RuntimeError(f"U{update} CUDA-2048 report identity/completion failed")
        if any(not bool(row.get("valid")) or row.get("error") is not None for row in entries):
            raise RuntimeError(f"U{update} CUDA-2048 contains invalid/error games")
        if (
            not isinstance(candidate, Mapping)
            or candidate.get("status") != "PASS"
            or candidate.get("contract_id") != "kaggle_fp16_storage_fp32_runtime_v1"
            or candidate.get("checkpoint_update") != update
        ):
            raise RuntimeError(f"U{update} CUDA-2048 candidate identity failed")
        if (
            not isinstance(opponent, Mapping)
            or opponent.get("status") != "PASS"
            or opponent.get("requested_policy_id") != "Policy-0809"
        ):
            raise RuntimeError(f"U{update} CUDA-2048 opponent identity failed")
        if (
            not isinstance(schedule, Mapping)
            or schedule.get("contract_id")
            != "0044_benchmark_v2_core16_meta_balanced_policy0809_common_seeds_cuda2048_v2"
            or schedule.get("games") != 2048
            or schedule.get("common_random_numbers") is not True
        ):
            raise RuntimeError(f"U{update} CUDA-2048 schedule failed")
        current_rows, value_rows = aggregate_entries(
            entries, deck_to_current=deck_to_current, deck_to_value=deck_to_value
        )
        for class_id in current_names:
            current[class_id][update] = current_rows.get(class_id, OutcomeStats())
        for class_id in value_names:
            value[class_id][update] = value_rows.get(class_id, OutcomeStats())
        overall[update] = _stats(entries)
        this_schedule_hash = str(schedule["common_random_schedule_sha256"])
        this_opponent_hash = str(opponent["effective_policy_sha256"])
        if schedule_hash is None:
            schedule_hash = this_schedule_hash
            opponent_hash = this_opponent_hash
        elif schedule_hash != this_schedule_hash or opponent_hash != this_opponent_hash:
            raise RuntimeError("CUDA-2048 schedule or opponent identity changed")
        audits.append({
            "update": update,
            "path": str(path.relative_to(ROOT)),
            "status": "PASS",
            "candidate_status": candidate["status"],
            "candidate_effective_sha256": candidate["effective_candidate_sha256"],
            "opponent_status": opponent["status"],
            "opponent_effective_sha256": this_opponent_hash,
            "schedule_sha256": this_schedule_hash,
        })

    current_to_value_sets: dict[int, set[int]] = defaultdict(set)
    value_to_current_sets: dict[int, set[int]] = defaultdict(set)
    current_to_decks: dict[int, list[str]] = defaultdict(list)
    value_to_decks: dict[int, list[str]] = defaultdict(list)
    for deck_id, current_id in deck_to_current.items():
        value_id = deck_to_value[deck_id]
        current_to_value_sets[current_id].add(value_id)
        value_to_current_sets[value_id].add(current_id)
        current_to_decks[current_id].append(deck_id)
        value_to_decks[value_id].append(deck_id)
    return HistoryAudit(
        updates=CUDA2048_UPDATES,
        current_names=current_names,
        value_names=value_names,
        active_current_ids=tuple(
            class_id for class_id in current_names if current[class_id][CUDA2048_UPDATES[0]].games
        ),
        current=current,
        value=value,
        overall=overall,
        pools={},
        deck_to_current=deck_to_current,
        deck_to_value=deck_to_value,
        current_to_value={key: tuple(sorted(rows)) for key, rows in current_to_value_sets.items()},
        value_to_current={key: tuple(sorted(rows)) for key, rows in value_to_current_sets.items()},
        current_to_decks={key: tuple(sorted(rows)) for key, rows in current_to_decks.items()},
        value_to_decks={key: tuple(sorted(rows)) for key, rows in value_to_decks.items()},
        report_audits=tuple(audits),
    )


def _movement_metrics(
    left: Sequence[int], right: Sequence[int]
) -> tuple[float | None, int, int, int, int]:
    left_delta = [after - before for before, after in zip(left, left[1:])]
    right_delta = [after - before for before, after in zip(right, right[1:])]
    left_mean = sum(left_delta) / len(left_delta)
    right_mean = sum(right_delta) / len(right_delta)
    left_variance = sum((value - left_mean) ** 2 for value in left_delta)
    right_variance = sum((value - right_mean) ** 2 for value in right_delta)
    correlation = None
    if left_variance > 0 and right_variance > 0:
        correlation = sum(
            (x - left_mean) * (y - right_mean)
            for x, y in zip(left_delta, right_delta)
        ) / math.sqrt(left_variance * right_variance)
    opposing = sum(x * y < 0 for x, y in zip(left_delta, right_delta))
    joint = sum(x != 0 and y != 0 for x, y in zip(left_delta, right_delta))
    return correlation, opposing, joint, left[-1] - left[0], right[-1] - right[0]


def tradeoff_diagnostics(history: HistoryAudit) -> tuple[TradeoffDiagnostic, ...]:
    aggressive_updates = tuple(update for update in history.updates if update >= 276)
    rows: list[TradeoffDiagnostic] = []
    for offset, left in enumerate(history.active_current_ids):
        full_left = [history.current[left][update].wins for update in history.updates]
        aggressive_left = [history.current[left][update].wins for update in aggressive_updates]
        for right in history.active_current_ids[offset + 1 :]:
            full_right = [history.current[right][update].wins for update in history.updates]
            aggressive_right = [history.current[right][update].wins for update in aggressive_updates]
            full = _movement_metrics(full_left, full_right)
            aggressive = _movement_metrics(aggressive_left, aggressive_right)
            rows.append(TradeoffDiagnostic(
                left=left,
                right=right,
                full_correlation=full[0],
                aggressive_correlation=aggressive[0],
                full_opposing_steps=full[1],
                full_joint_steps=full[2],
                aggressive_opposing_steps=aggressive[1],
                aggressive_joint_steps=aggressive[2],
                full_left_net=full[3],
                full_right_net=full[4],
                aggressive_left_net=aggressive[3],
                aggressive_right_net=aggressive[4],
                left_games=history.current[left][history.updates[-1]].games,
                right_games=history.current[right][history.updates[-1]].games,
            ))
    return tuple(rows)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def _cell(stats: OutcomeStats) -> str:
    if not stats.games:
        return '<td class="empty"><b>—</b><small>n=0</small></td>'
    return (
        f"<td><b>{_pct(stats.win_rate)}</b>"
        f"<small>{stats.wins}-{stats.losses}-{stats.draws} · n={stats.games}</small></td>"
    )


def _row_heat_cells(stats_by_update: Sequence[OutcomeStats]) -> str:
    """Render within-Meta heat: darker means a higher rate in this exact row."""
    rates = [stats.win_rate for stats in stats_by_update if stats.win_rate is not None]
    if not rates:
        return "".join(_cell(stats) for stats in stats_by_update)
    minimum = min(rates)
    maximum = max(rates)
    span = maximum - minimum
    cells: list[str] = []
    for stats in stats_by_update:
        rate = stats.win_rate
        if rate is None:
            cells.append(_cell(stats))
            continue
        normalized = 0.5 if span <= 1e-12 else (rate - minimum) / span
        alpha = 0.08 + 0.58 * normalized
        best = abs(rate - maximum) <= 1e-12
        css_class = "row-heat row-best" if best else "row-heat"
        label = " · 行内最高" if best else ""
        cells.append(
            f'<td class="{css_class}" style="background:rgba(23,107,75,{alpha:.3f})" '
            f'title="本 Meta 行内相对热度：{normalized * 100:.1f}%{label}">'
            f"<b>{_pct(rate)}</b>"
            f"<small>{stats.wins}-{stats.losses}-{stats.draws} · n={stats.games}{label}</small></td>"
        )
    return "".join(cells)


def _row_class(current_id: int) -> str:
    if current_id in {2, 3, 5}:
        return "target"
    if current_id in {0, 1, 4, 27}:
        return "protect"
    return ""


def _correlation(value: float | None) -> str:
    return "—" if value is None else f"{value:+.3f}"


def _heat_color(value: float | None) -> str:
    if value is None:
        return "#f2f4f3"
    strength = min(1.0, abs(value))
    if value < 0:
        return f"rgba(190,65,53,{0.12 + 0.68 * strength:.3f})"
    return f"rgba(29,128,86,{0.10 + 0.60 * strength:.3f})"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_oracle_report(path: Path = ORACLE_REPORT) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    from .run_meta_oracle_v1_cuda2048 import validate_report
    validate_report(report)
    return report


def _oracle_section(
    report: Mapping[str, object], cuda_history: HistoryAudit
) -> tuple[str, Mapping[str, object]]:
    entries = report["entries"]
    summary = report["summary"]
    audit = report["focal_oracle_identity_audit"]
    by_meta = {
        meta_id: _stats(
            row for row in entries
            if int(row["opponent_meta_archetype_id"]) == meta_id
        )
        for meta_id in cuda_history.active_current_ids
    }
    meta_rows = []
    for meta_id in cuda_history.active_current_ids:
        stats = by_meta[meta_id]
        update = int(audit["route"][f"{meta_id:02d}"])
        source = cuda_history.current[meta_id][update]
        default = cuda_history.current[meta_id][282]
        if stats != source:
            raise RuntimeError(
                f"oracle U{update} route did not reproduce source Meta {meta_id:02d}"
            )
        delta = (stats.win_rate - default.win_rate) * 100
        meta_rows.append(
            f'<tr class="{_row_class(meta_id)}" data-oracle-meta-row="1">'
            f'<th><span class="badge">{meta_id:02d}</span></th>'
            f"<td class=sticky-name><b>{html.escape(cuda_history.current_names[meta_id])}</b></td>"
            f"<td><b>U{update}</b></td>{_cell(stats)}{_cell(default)}"
            f'<td class="{"up" if delta > 0 else "down" if delta < 0 else ""}">{delta:+.2f} pp<small>{stats.wins-default.wins:+d} wins</small></td></tr>'
        )
    route_rows = []
    for update in (40, 90, 200, 282):
        row = report["route_summary"][str(update)]
        metas = " / ".join(f"{int(value):02d}" for value in row["meta_ids"])
        route_rows.append(
            f"<tr><th>U{update}</th><td>{metas}</td><td>{row['games']}</td>"
            f"<td>{row['wins']}-{row['losses']}-{row['draws']}</td>"
            f"<td><b>{_pct(float(row['win_rate']))}</b></td></tr>"
        )
    u282 = cuda_history.overall[282]
    u200 = cuda_history.overall[200]
    oracle_stats = OutcomeStats(
        int(summary["wins"]), int(summary["losses"]), int(summary["draws"])
    )
    delta_282 = (oracle_stats.win_rate - u282.win_rate) * 100
    delta_200 = (oracle_stats.win_rate - u200.win_rate) * 100
    section = f'''<section id="meta-oracle-v1" class="oracle"><div class="oracle-warning"><b>⚠ 拼好龙 V1 作弊版 / Oracle diagnostic</b><span>运行时直接读取对手 exact deck 的真实 29-way Meta，再选择 checkpoint head；不可部署、不可提交 Kaggle、不可用于 Promote Champion。</span></div><h2>Experimental-Oracle-MetaRouter-V1 · CUDA-2048</h2><p class="note">冻结语义骨干只保留一份；按 Meta 仅切换 Policy Q/V LoRA、Action Decoder、Allocation Head。Critic 固定来自 U282，且不进入 Actor。对手仍为完整 immutable Policy-0809，原 Benchmark V2 common seeds / official engine / greedy 合同保持不变。</p><div class="cards oracle-cards"><div class="card"><small>Oracle 总胜率</small><b>{_pct(oracle_stats.win_rate)}</b><span>{oracle_stats.wins}-{oracle_stats.losses}-{oracle_stats.draws}</span></div><div class="card"><small>相对默认 U282</small><b class="up">{delta_282:+.2f} pp</b><span>{oracle_stats.wins-u282.wins:+d} wins</span></div><div class="card"><small>相对静态 U200</small><b class="up">{delta_200:+.2f} pp</b><span>{oracle_stats.wins-u200.wins:+d} wins</span></div><div class="card"><small>先手 / 后手</small><b>{_pct(float(summary['focal_first_win_rate']))} / {_pct(float(summary['focal_second_win_rate']))}</b><span>{summary['focal_first_games']} / {summary['focal_second_games']} games</span></div></div><h3>真实 if / elif 路由</h3><div class="table-wrap"><table><thead><tr><th>Checkpoint head</th><th>Opponent Meta</th><th>Games</th><th>W-L-D</th><th>Win rate</th></tr></thead><tbody>{''.join(route_rows)}</tbody></table></div><h3>逐 29-way Meta 结果</h3><p class="note">“Source checkpoint” 列已逐格与同 seeds 的历史 CUDA-2048 报告核对，16/16 Meta 完全复现。最后两列显示 oracle 相对全局默认 U282 的收益。</p><div class="table-wrap"><table><thead><tr><th>Meta</th><th class=sticky-name>名称</th><th>Route</th><th>Oracle / source checkpoint</th><th>U282 default</th><th>Oracle − U282</th></tr></thead><tbody>{''.join(meta_rows)}</tbody></table></div><h3>Identity / isolation audit</h3><div class="audit"><span>Oracle policy ID</span><code>{html.escape(str(report['focal_policy_id']))}</code><span>Composite identity</span><code>{html.escape(str(audit['composite_effective_sha256']))}</code><span>Shared Actor tensors</span><b class="pass">{audit['shared_actor']['shared_tensor_count']} exact-equal · PASS</b><span>Routed Actor decoder tensors</span><b>{audit['shared_actor']['excluded_routed_actor_tensor_count']}</b><span>Routed modules</span><code>policy_option_lora + actor.action_decoder + allocation_head</code><span>Storage aliases</span><b class="pass">shared↔head 0 · head↔head 0 · focal↔opponent 0</b><span>Opponent</span><b class="pass">Policy-0809 identity PASS</b><span>Completion</span><b class="pass">2,048/2,048 terminal · 0 error · 0 unfinished</b><span>Evidence boundary</span><b class="down">diagnostic_oracle_not_promote_not_kaggle</b></div></section>'''
    payload = {
        "report_path": str(ORACLE_REPORT.relative_to(ROOT)),
        "report_sha256": _sha256_file(ORACLE_REPORT),
        "summary": summary,
        "route_summary": report["route_summary"],
        "identity_audit": audit,
    }
    return section, payload


def render(
    history: HistoryAudit, cuda_history: HistoryAudit | None = None,
    oracle_report: Mapping[str, object] | None = None,
) -> str:
    if cuda_history is None:
        cuda_history = load_cuda2048_history()
    oracle_section = ""
    oracle_payload = None
    if oracle_report is not None:
        oracle_section, oracle_payload = _oracle_section(oracle_report, cuda_history)
    latest = history.updates[-1]
    baseline = history.updates[0]
    update_headers = "".join(f"<th>U{update}</th>" for update in history.updates)
    overall_rows = []
    for label, values in (
        ("全部三池", history.overall),
        ("02 / 03 / 05", history.pools["low_score"]),
        ("00 / 01 / 04 / 27", history.pools["priority"]),
        ("Remaining", history.pools["remaining"]),
    ):
        overall_rows.append(
            f"<tr><th>{html.escape(label)}</th>"
            + "".join(_cell(values[update]) for update in history.updates)
            + "</tr>"
        )

    cuda_headers = "".join(f"<th>U{update}</th>" for update in cuda_history.updates)
    cuda_overall = "".join(_cell(cuda_history.overall[update]) for update in cuda_history.updates)
    cuda_current_rows = []
    for class_id, name in cuda_history.current_names.items():
        values = cuda_history.current[class_id]
        labels = ", ".join(
            f"{value:02d}" for value in cuda_history.current_to_value.get(class_id, ())
        ) or "—"
        decks = ", ".join(cuda_history.current_to_decks.get(class_id, ())) or "—"
        first = values[cuda_history.updates[0]]
        last = values[cuda_history.updates[-1]]
        change = (
            (last.win_rate - first.win_rate) * 100
            if first.win_rate is not None and last.win_rate is not None
            else None
        )
        change_text = "—" if change is None else f"{change:+.2f} pp"
        change_class = "up" if change is not None and change > 0 else "down" if change is not None and change < 0 else ""
        cuda_current_rows.append(
            f'<tr class="{_row_class(class_id)}" data-cuda-current-meta-row="1">'
            f'<th><span class="badge">{class_id:02d}</span></th>'
            f"<td class=sticky-name><b>{html.escape(name)}</b><small>Decks {decks}<br>Value label {labels}</small></td>"
            + _row_heat_cells([values[update] for update in cuda_history.updates])
            + f'<td class="{change_class}">{change_text}</td></tr>'
        )
    cuda_value_rows = []
    for class_id, name in cuda_history.value_names.items():
        values = cuda_history.value[class_id]
        constituents = ", ".join(
            f"{item:02d}" for item in cuda_history.value_to_current.get(class_id, ())
        ) or "—"
        decks = ", ".join(cuda_history.value_to_decks.get(class_id, ())) or "—"
        cuda_value_rows.append(
            '<tr data-cuda-value-meta-row="1">'
            f'<th><span class="badge legacy">{class_id:02d}</span></th>'
            f"<td class=sticky-name><b>{html.escape(name)}</b><small>V2 {constituents}<br>Decks {decks}</small></td>"
            + _row_heat_cells([values[update] for update in cuda_history.updates])
            + "</tr>"
        )

    cuda_key_ids = tuple(item for item in (0, 1, 2, 3, 4, 5, 27) if item in cuda_history.active_current_ids)
    cuda_tradeoff_rows = []
    for offset, left in enumerate(cuda_key_ids):
        left_wins = [cuda_history.current[left][update].wins for update in cuda_history.updates]
        for right in cuda_key_ids[offset + 1 :]:
            right_wins = [cuda_history.current[right][update].wins for update in cuda_history.updates]
            correlation, opposing, joint, left_net, right_net = _movement_metrics(left_wins, right_wins)
            flag = "净交换" if left_net * right_net < 0 else "非净交换"
            cuda_tradeoff_rows.append(
                '<tr data-cuda-tradeoff-row="1">'
                f"<td><b>{left:02d} ↔ {right:02d}</b><small>{html.escape(cuda_history.current_names[left])} / {html.escape(cuda_history.current_names[right])}</small></td>"
                f"<td>{_correlation(correlation)}<small>{opposing}/{joint} opposite/joint</small></td>"
                f"<td>{left_net:+d} / {right_net:+d}<small>U40→U282 wins（各类 n=128）</small></td>"
                f"<td>{flag}</td></tr>"
            )

    current_rows = []
    for class_id, name in history.current_names.items():
        values = history.current[class_id]
        labels = ", ".join(f"{value:02d}" for value in history.current_to_value.get(class_id, ())) or "—"
        decks = ", ".join(history.current_to_decks.get(class_id, ())) or "—"
        delta = None
        if values[baseline].win_rate is not None and values[latest].win_rate is not None:
            delta = (values[latest].win_rate - values[baseline].win_rate) * 100
        delta_text = "—" if delta is None else f"{delta:+.2f} pp"
        delta_class = "up" if delta is not None and delta > 0 else "down" if delta is not None and delta < 0 else ""
        current_rows.append(
            f'<tr class="{_row_class(class_id)}" data-current-meta-row="1">'
            f'<th><span class="badge">{class_id:02d}</span></th>'
            f"<td class=sticky-name><b>{html.escape(name)}</b><small>Decks {decks}<br>Value label {labels}</small></td>"
            + _row_heat_cells([values[update] for update in history.updates])
            + f'<td class="{delta_class}">{delta_text}</td></tr>'
        )

    value_rows = []
    conflict_rows = []
    for class_id, name in history.value_names.items():
        values = history.value[class_id]
        constituents = history.value_to_current.get(class_id, ())
        constituent_text = ", ".join(f"{item:02d}" for item in constituents) or "—"
        decks = ", ".join(history.value_to_decks.get(class_id, ())) or "—"
        rates = [
            history.current[item][latest].win_rate
            for item in constituents
            if history.current[item][latest].win_rate is not None
        ]
        span = (max(rates) - min(rates)) * 100 if len(rates) > 1 else 0.0
        level = "高" if span >= 20 else "中" if span >= 10 else "低"
        level_class = "conflict-high" if level == "高" else "conflict-mid" if level == "中" else "conflict-low"
        value_rows.append(
            '<tr data-value-meta-row="1">'
            f'<th><span class="badge legacy">{class_id:02d}</span></th>'
            f"<td class=sticky-name><b>{html.escape(name)}</b><small>V2 {constituent_text}<br>Decks {decks}</small></td>"
            + _row_heat_cells([values[update] for update in history.updates])
            + f'<td class="{level_class}"><b>{level}</b><small>{span:.2f} pp</small></td></tr>'
        )
        conflict_rows.append(
            f"<tr><td>{class_id:02d} · {html.escape(name)}</td><td>{constituent_text}</td>"
            f"<td>{decks}</td><td class={level_class}>{level} · {span:.2f} pp</td></tr>"
        )

    diagnostics = tradeoff_diagnostics(history)
    key_ids = {0, 1, 2, 3, 4, 5, 27}
    key_pairs = [
        row for row in diagnostics if row.left in key_ids and row.right in key_ids
    ]
    key_pairs.sort(key=lambda row: (
        not row.aggressive_net_exchange,
        not row.full_net_exchange,
        row.aggressive_correlation if row.aggressive_correlation is not None else 1.0,
    ))
    global_pairs = [
        row for row in diagnostics
        if row.full_joint_steps >= 3 and row.aggressive_joint_steps >= 2
    ]
    global_pairs.sort(key=lambda row: (
        not row.aggressive_net_exchange,
        not row.full_net_exchange,
        min(
            row.full_correlation if row.full_correlation is not None else 1.0,
            row.aggressive_correlation if row.aggressive_correlation is not None else 1.0,
        ),
        -row.aggressive_opposing_steps,
    ))

    def tradeoff_row(row: TradeoffDiagnostic) -> str:
        left_name = history.current_names[row.left]
        right_name = history.current_names[row.right]
        flags = []
        if row.full_net_exchange:
            flags.append("全程净交换")
        if row.aggressive_net_exchange:
            flags.append("激进段净交换")
        if not flags:
            flags.append("仅步进反向")
        reliability = "高样本" if min(row.left_games, row.right_games) >= 100 else "低样本"
        return (
            '<tr data-tradeoff-row="1">'
            f"<td><b>{row.left:02d} ↔ {row.right:02d}</b><small>{html.escape(left_name)} / {html.escape(right_name)}</small></td>"
            f"<td>{_correlation(row.full_correlation)}<small>{row.full_opposing_steps}/{row.full_joint_steps} opposite/joint</small></td>"
            f"<td>{_correlation(row.aggressive_correlation)}<small>{row.aggressive_opposing_steps}/{row.aggressive_joint_steps} opposite/joint</small></td>"
            f"<td>{row.full_left_net:+d} / {row.full_right_net:+d}<small>U200→U298 wins</small></td>"
            f"<td>{row.aggressive_left_net:+d} / {row.aggressive_right_net:+d}<small>U276→U298 wins</small></td>"
            f"<td>{' · '.join(flags)}<small>{reliability}; n={row.left_games}/{row.right_games}</small></td></tr>"
        )

    matrix_lookup = {
        (row.left, row.right): row.aggressive_correlation for row in diagnostics
    }
    matrix_header = "".join(
        f'<th title="{html.escape(history.current_names[item])}">{item:02d}</th>'
        for item in history.active_current_ids
    )
    matrix_rows = []
    for left in history.active_current_ids:
        cells = []
        for right in history.active_current_ids:
            if left == right:
                value = 1.0
            else:
                value = matrix_lookup.get(
                    (min(left, right), max(left, right))
                )
            cells.append(
                f'<td style="background:{_heat_color(value)}" title="{left:02d} {html.escape(history.current_names[left])} ↔ {right:02d} {html.escape(history.current_names[right])}">{_correlation(value)}</td>'
            )
        matrix_rows.append(
            f'<tr><th title="{html.escape(history.current_names[left])}">{left:02d}</th>{"".join(cells)}</tr>'
        )

    latest_stats = history.overall[latest]
    baseline_stats = history.overall[baseline]
    delta = (latest_stats.win_rate - baseline_stats.win_rate) * 100
    audit_payload = {
        "schema_version": "0045_v8_dual_taxonomy_history_report_v2",
        "version": VERSION,
        "lineage_versions": [V4_VERSION, V7_VERSION, V8_VERSION],
        "updates": history.updates,
        "evaluation_contract": "0045_policy0809_three_meta_pools_common_seeds_cuda512_v1",
        "current_taxonomy": "own_archetypes_v2",
        "value_loss_taxonomy": "0036_opponent_archetypes_v1",
        "value_label_function": "OpponentArchetypeTaxonomy.classify_target(exact_60_card_deck)",
        "reports": history.report_audits,
        "cuda2048": {
            "updates": cuda_history.updates,
            "evaluation_contract": "0044_benchmark_v2_core16_meta_balanced_policy0809_common_seeds_cuda2048_v2",
            "reports": cuda_history.report_audits,
        },
        "meta_oracle_v1": oracle_payload,
    }
    embedded = json.dumps(audit_payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    style = """
:root{--bg:#f5f7f6;--ink:#17211d;--muted:#65716c;--brand:#176b4b;--line:#d8e0dc;--card:#fff;--target:#fff2dc;--protect:#e9f3ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}main{max-width:1680px;margin:auto;padding:30px 20px 80px}header{background:linear-gradient(120deg,#153f32,#1c6a51);color:#fff;padding:32px;border-radius:20px}header p{margin:0;color:#d5e9e1}h1{font-size:36px;margin:5px 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px}.card,section{background:#fff;border:1px solid var(--line);border-radius:16px}.card{padding:18px}.card b{display:block;font-size:25px}.card small,.note,td small{color:var(--muted)}section{margin-top:18px;padding:22px}h2{margin:0 0 8px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}table{border-collapse:separate;border-spacing:0;min-width:1450px;width:100%;background:#fff}th,td{padding:8px 10px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);text-align:center;vertical-align:top;white-space:nowrap}thead th{position:sticky;top:0;background:#edf4f1;z-index:4}tbody th{position:sticky;left:0;background:#fff;z-index:2}.sticky-name{position:sticky;left:62px;background:#fff;z-index:1;text-align:left;white-space:normal;min-width:260px}.target th,.target .sticky-name{background:var(--target)}.protect th,.protect .sticky-name{background:var(--protect)}td b{display:block}td small{display:block}.badge{display:inline-block;min-width:34px;padding:3px 7px;border-radius:999px;background:#dcebe5}.badge.legacy{background:#e8e2f5}.up{color:#0a7a4b;font-weight:700}.down{color:#b14439;font-weight:700}.empty{color:#8b9590}.legend{display:flex;gap:12px;flex-wrap:wrap}.legend span{padding:5px 10px;border-radius:999px;border:1px solid var(--line)}.legend .target-key{background:var(--target)}.legend .protect-key{background:var(--protect)}.conflict-high{color:#a3342c;font-weight:700}.conflict-mid{color:#a06400;font-weight:700}.conflict-low{color:#16724e;font-weight:700}.audit{display:grid;grid-template-columns:220px 1fr;gap:7px 14px}.audit code{overflow-wrap:anywhere}.pass{color:var(--brand);font-weight:800}@media(max-width:900px){.cards{grid-template-columns:1fr 1fr}h1{font-size:28px}.audit{grid-template-columns:1fr}.sticky-name{position:static}}
"""
    style += """
.row-heat{transition:filter .15s}.row-heat:hover{filter:saturate(1.25)}.row-best{box-shadow:inset 0 0 0 2px #0d5239}.row-best b{color:#062f20}.legend .heat-key{background:linear-gradient(90deg,rgba(23,107,75,.08),rgba(23,107,75,.66));color:#062f20}
.oracle{border:2px solid #c44b3e}.oracle-warning{display:flex;flex-direction:column;gap:4px;background:#ffe8e4;color:#7a2118;border:1px solid #efaca4;border-radius:12px;padding:14px;margin-bottom:18px}.oracle-warning b{font-size:18px}.oracle-cards{margin-bottom:18px}
"""
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>0045 U40–U298 · 双 Meta taxonomy</title><style>{style}</style></head><body><main>
<header><p>0045 SINGLE-DECK EXPERT · ALL AVAILABLE META EVALUATIONS</p><h1>U40–U298 · 双 Meta taxonomy 与 trade-off 体检</h1><p>Deck 007 · Policy-0809 · CUDA-2048 Benchmark V2 + 三池 CUDA-512（每点合计 1,536 局）· greedy · official engine</p></header>
<div class="cards"><div class="card"><small>已纳入 update 点</small><b>{len(cuda_history.updates) + len(history.updates)} points</b><span>{len(cuda_history.updates)} CUDA-2048 + {len(history.updates)} 三池</span></div><div class="card"><small>最新三池 U{latest}</small><b>{_pct(latest_stats.win_rate)}</b><span>{latest_stats.wins}-{latest_stats.losses}-{latest_stats.draws}</span></div><div class="card"><small>三池 U{baseline} → U{latest}</small><b class="{'up' if delta >= 0 else 'down'}">{delta:+.2f} pp</b><span>{latest_stats.wins-baseline_stats.wins:+d} wins</span></div><div class="card"><small>Identity / health</small><b class="pass">{len(cuda_history.updates) + len(history.updates)} / {len(cuda_history.updates) + len(history.updates)} PASS</b><span>{len(cuda_history.updates)*2048 + len(history.updates)*1536:,} terminal · 0 error</span></div></div>
<section id="cuda2048-history"><h2>Benchmark V2 CUDA-2048 · 全部已测 update</h2><p class="note">纳入 U40、U90、U95、U100–U200（每 5 update）及 U282，共 {len(cuda_history.updates)} 点。它与后面的三池 512 不是同一个采样分布，因此独立成表；内部使用同一 Policy-0809、同一 common-random schedule，每个所选 current Meta 固定 128 局。U190 使用修复 candidate policy ID 后的 identity-fixed 报告，未重复计入旧副本。</p><div class="table-wrap"><table><thead><tr><th>Overall</th>{cuda_headers}</tr></thead><tbody><tr><th>Benchmark V2</th>{cuda_overall}</tr></tbody></table></div><h3>CUDA-2048 · 当前 29 类</h3><div class="legend"><span class="target-key">02 / 03 / 05 · 激进训练目标</span><span class="protect-key">00 / 01 / 04 / 27 · 重点保护组</span><span class="heat-key">浅 → 深：本 Meta 行内胜率低 → 高；描边为最高</span></div><div class="table-wrap"><table><thead><tr><th>Meta</th><th class=sticky-name>名称 / deck / Value label</th>{cuda_headers}<th>U40→U282</th></tr></thead><tbody>{''.join(cuda_current_rows)}</tbody></table></div><h3>CUDA-2048 · Value-loss 15 类</h3><div class="table-wrap"><table><thead><tr><th>Label</th><th class=sticky-name>名称 / 被合并的 current 类</th>{cuda_headers}</tr></thead><tbody>{''.join(cuda_value_rows)}</tbody></table></div><h3>CUDA-2048 · 重点七类 trade-off</h3><p class="note">相关系数仍基于相邻已测点胜场一阶差分；这里只描述曲线共振，不把不同训练阶段的变化解释成因果。</p><div class="table-wrap"><table><thead><tr><th>Pair</th><th>diff corr</th><th>净胜场</th><th>判读</th></tr></thead><tbody>{''.join(cuda_tradeoff_rows)}</tbody></table></div></section>
<section><h2>三池总体纵向表</h2><p class="note">每个 update 使用同一 common-random schedule。各格显示胜率、W-L-D 与样本数。</p><div class="table-wrap"><table><thead><tr><th>Pool</th>{update_headers}</tr></thead><tbody>{''.join(overall_rows)}</tbody></table></div></section>
<section id="meta-current-29"><h2>三池表一 · 当前 29 类 reporting taxonomy</h2><div class="legend"><span class="target-key">02 / 03 / 05 · 激进训练目标</span><span class="protect-key">00 / 01 / 04 / 27 · 重点保护组</span><span class="heat-key">浅 → 深：本 Meta 行内胜率低 → 高；描边为最高</span><span>14 Other 无已知 deck，明确显示 n=0</span></div><p class="note"><code>own_archetypes_v2</code>：exact deck ID 是 source of truth。最后一列为 U298 相对 U200。</p><div class="table-wrap"><table><thead><tr><th>Meta</th><th class=sticky-name>名称 / deck / Value label</th>{update_headers}<th>Δ</th></tr></thead><tbody>{''.join(current_rows)}</tbody></table></div></section>
<section id="meta-value-15"><h2>表二 · Value-loss 15-way opponent taxonomy</h2><p class="note">这是训练 Critic Meta classification loss 的真实标签体系：对同一 opponent deck ID 读取 exact 60 cards，再调用 <code>OpponentArchetypeTaxonomy.classify_target</code>，taxonomy 为 <code>opponent_archetypes_v1</code>。不是 actor own-taxonomy 的 <code>old_archetype_id</code>。</p><div class="table-wrap"><table><thead><tr><th>Label</th><th class=sticky-name>名称 / 被合并的 V2 类</th>{update_headers}<th>U298 类内跨度</th></tr></thead><tbody>{''.join(value_rows)}</tbody></table></div></section>
<section id="taxonomy-crosswalk"><h2>29 → 15 归并冲突审计</h2><p class="note">本次所有 current 29-way 类都确定性映射到一个 Value label，没有同一 current 类跨多个 Value label 的标签歧义。“冲突”表示 15-way 把多个策略类合并后，U298 各子类胜率的 max−min；它衡量信息损失，不等于 classifier 预测错误。</p><div class="table-wrap"><table><thead><tr><th>Value label</th><th>Current V2 constituents</th><th>Exact decks</th><th>U298 outcome span</th></tr></thead><tbody>{''.join(conflict_rows)}</tbody></table></div></section>
<section id="meta-tradeoffs"><h2>Meta trade-off 诊断</h2><p class="note">相关系数基于相邻已测 checkpoint 的胜场一阶差分；负值表示两个 Meta 经常反向移动。U200→U298 使用全部 30 个可比间隔，激进段使用 U276→U298。净交换表示一个 Meta 最终增加、另一个最终减少。相关不证明容量竞争或因果，低样本类尤其只能作为线索。</p><h3>重点七类全部 21 个 pair</h3><div class="table-wrap"><table><thead><tr><th>Pair</th><th>全程 diff corr</th><th>激进段 diff corr</th><th>全程净胜场</th><th>激进段净胜场</th><th>判读</th></tr></thead><tbody>{''.join(tradeoff_row(row) for row in key_pairs)}</tbody></table></div><h3>全 29 类候选排名（前 30）</h3><div class="table-wrap"><table><thead><tr><th>Pair</th><th>全程 diff corr</th><th>激进段 diff corr</th><th>全程净胜场</th><th>激进段净胜场</th><th>判读</th></tr></thead><tbody>{''.join(tradeoff_row(row) for row in global_pairs[:30])}</tbody></table></div><h3>激进段全量一阶差分相关矩阵</h3><p class="note">红色越深越倾向反向移动；绿色越深越倾向同向移动。Meta 14 无已知 deck，未进入矩阵。</p><div class="table-wrap"><table class="matrix"><thead><tr><th>Meta</th>{matrix_header}</tr></thead><tbody>{''.join(matrix_rows)}</tbody></table></div></section>
<section><h2>证据与可比性</h2><div class="audit"><span>CUDA-2048 contract</span><code>0044_benchmark_v2_core16_meta_balanced_policy0809_common_seeds_cuda2048_v2 · {len(cuda_history.updates)}/{len(cuda_history.updates)} PASS</code><span>三池 contract</span><code>0045_policy0809_three_meta_pools_common_seeds_cuda512_v1 · {len(history.updates)}/{len(history.updates)} PASS</code><span>Opponent</span><b class="pass">complete immutable Policy-0809 · 全部 identity PASS</b><span>Candidate deployment</span><b class="pass">kaggle_fp16_storage_fp32_runtime_v1 · 全部 PASS</b><span>Current taxonomy</span><code>own_archetypes_v2 · deck ID exact mapping</code><span>Value-loss taxonomy</span><code>0036_opponent_archetypes_v1 · exact deck → classify_target</code><span>Boundary</span><b>不同评测合同全部展示，但分表处理；横向阅读时必须同时看每格 n，不能把 CUDA-2048 与三池 1,536 的总体胜率直接当成同一分布。</b></div></section>
{oracle_section}
<script id="report-audit" type="application/json">{embedded}</script></main></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--oracle-report", type=Path)
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite formal report: {args.output}")
    content = render(
        load_default_history(),
        oracle_report=(
            load_oracle_report(args.oracle_report.resolve())
            if args.oracle_report else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
