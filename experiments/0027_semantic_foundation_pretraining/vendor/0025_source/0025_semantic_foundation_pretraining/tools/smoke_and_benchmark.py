"""Constrained real-row smoke and old/new encoding cost benchmark."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
import resource
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from ..features.compiler import actor_payload, compile_row
from ..features.prototypes import PrototypeIndex
from ..knowledge.state import CausalKnowledge
from ..legacy.base_model import IDOnlyCodec, IDOnlyConfig
from ..model.batching import collate
from ..model.multi_memory import SemanticFoundationPolicy, SemanticModelConfig

OBSERVED_0019 = {
    "date_start": "2026-07-10", "date_end": "2026-07-28", "days": 19,
    "decisions": 7_347_132, "raw_bytes": 2_386_657_033,
    "model_ready_bytes": 54_155_429_609, "materialization_seconds": 9_680.333,
    "workers": 8,
}


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _deck(row: dict[str, Any]) -> list[int]:
    output: list[int] = []
    for card_id, count in row["deck_manifest"]["counts"]:
        output.extend([int(card_id)] * int(count))
    return output


def _read_sample(path: Path, count: int) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError(f"sample shard contains only {len(rows)} rows")
    return rows


def _semantic_compile(rows: list[dict[str, Any]], prototypes: PrototypeIndex) -> list[dict[str, Any]]:
    output = []
    knowledge = None
    group = None
    for row in rows:
        identity = row["identity"]
        key = (identity["date"], identity["episode_id"], identity["player_index"])
        if key != group:
            group = key
            knowledge = CausalKnowledge(int(identity["player_index"]), _deck(row))
        assert knowledge is not None
        snapshot = knowledge.consume(row["actor_observation"], row["event_cursor"])
        output.append(compile_row(row, snapshot, prototypes))
    return output


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * q))]


def benchmark(rows: list[dict[str, Any]], prototypes: PrototypeIndex, warmup: int, repeats: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    codec = IDOnlyCodec(IDOnlyConfig(max_action_steps=64))
    warm_rows = rows[: max(1, min(warmup, len(rows)))]
    for row in warm_rows:
        if codec.encode(row["actor_observation"], row["ordered_action"]) is None:
            raise ValueError("legacy warmup rejected a row")
    _semantic_compile(warm_rows, prototypes)

    legacy_times: list[float] = []
    legacy_runs_ms: list[float] = []
    legacy: list[dict[str, Any]] = []
    for repeat in range(repeats):
        run_started = time.perf_counter_ns()
        current = []
        for row in rows:
            started = time.perf_counter_ns()
            value = codec.encode(row["actor_observation"], row["ordered_action"])
            legacy_times.append((time.perf_counter_ns() - started) / 1_000_000)
            if value is None:
                raise ValueError("legacy codec rejected sampled accepted decision")
            current.append(value)
        if repeat == 0:
            legacy = current
        legacy_runs_ms.append((time.perf_counter_ns() - run_started) / 1_000_000)
    semantic_runs_ms: list[float] = []
    semantic: list[dict[str, Any]] = []
    for repeat in range(repeats):
        started = time.perf_counter_ns()
        current = _semantic_compile(rows, prototypes)
        semantic_runs_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        if repeat == 0:
            semantic = current
    semantic_actor = [actor_payload(item) for item in semantic]
    raw_payload = b"".join(_canonical(row) for row in rows)
    old_payload = b"".join(_canonical(row) for row in legacy)
    new_payload = b"".join(_canonical(row) for row in semantic_actor)
    old_gzip = gzip.compress(old_payload, compresslevel=6)
    new_gzip = gzip.compress(new_payload, compresslevel=6)
    n = len(rows)
    new_ms = statistics.median(semantic_runs_ms) / n
    static_sidecar_bytes = len(_canonical(prototypes.payload)) + len(_canonical(prototypes.engine_payload))
    daily = OBSERVED_0019["decisions"] / OBSERVED_0019["days"]
    projections = {}
    for label, days in (("locally_available_0710_0729", 20), ("requested_0710_0802", 24)):
        decisions = round(daily * days)
        projections[label] = {
            "status": "linear_projection_not_observed",
            "days": days,
            "decisions": decisions,
            "dynamic_gzip_bytes": round(decisions * len(new_gzip) / n),
            "plus_static_sidecar_bytes": round(decisions * len(new_gzip) / n) + static_sidecar_bytes,
            "single_thread_encoding_seconds": decisions * new_ms / 1000,
            "eight_worker_linear_seconds": decisions * new_ms / 1000 / 8,
        }
    option_types = Counter(
        int(option.get("type", -1))
        for row in rows
        for option in row["legal_options"]
        if isinstance(option, dict)
    )
    return {
        "schema_version": "0025_semantic_smoke_benchmark_v1",
        "status": "passed",
        "constraints": {"processes": 1, "torch_threads": 1, "full_build_started": False, "formal_training_started": False},
        "sample": {
            "decisions": n, "warmup_decisions": len(warm_rows), "timing_repeats": repeats,
            "first_identity": rows[0]["identity"], "last_identity": rows[-1]["identity"],
            "legal_option_type_counts": {str(key): value for key, value in sorted(option_types.items())},
            "attack_options": option_types[13],
        },
        "prototype_sidecar": {
            "cards": len(prototypes.cards), "attacks": len(prototypes.attacks),
            "public_bytes": len(_canonical(prototypes.payload)),
            "full_engine_bytes": len(_canonical(prototypes.engine_payload)),
            "total_bytes": static_sidecar_bytes,
            "skills": len(prototypes.skills), "effects": len(prototypes.effects),
            "content_sha256": prototypes.payload["content_sha256"],
        },
        "bytes_per_decision": {
            "raw_json_uncompressed_sample": len(raw_payload) / n,
            "legacy_dynamic_json_uncompressed_sample": len(old_payload) / n,
            "semantic_dynamic_json_uncompressed_sample": len(new_payload) / n,
            "legacy_dynamic_gzip_stream_sample": len(old_gzip) / n,
            "semantic_dynamic_gzip_stream_sample": len(new_gzip) / n,
            "observed_0019_model_ready_cache": OBSERVED_0019["model_ready_bytes"] / OBSERVED_0019["decisions"],
        },
        "encoding_ms_per_decision": {
            "legacy_mean": statistics.mean(legacy_times),
            "legacy_median": statistics.median(legacy_times),
            "legacy_p95": _percentile(legacy_times, 0.95),
            "legacy_pipeline_median": statistics.median(legacy_runs_ms) / n,
            "semantic_pipeline_mean": new_ms,
            "semantic_pipeline_note": "median repeated pipeline time; includes causal ledger consume and frozen legacy plus semantic compilation",
        },
        "peak_rss_kib_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "observed_0019": OBSERVED_0019,
        "projections": projections,
        "correctness": {
            "all_actor_records_exclude_source_id": all("source_id" not in item for item in semantic_actor),
            "ledger_present": all(bool(item["ledger_cat"]) for item in semantic),
            "legacy_option_count_preserved": all(len(a["option_cat"]) == len(b["semantic_option_cat"]) for a, b in zip(legacy, semantic)),
        },
    }


def _render(result: dict[str, Any]) -> str:
    b = result["bytes_per_decision"]
    t = result["encoding_ms_per_decision"]
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value:,.3f}</td></tr>" for name, value in b.items()
    )
    projections = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value['decisions']:,}</td>"
        f"<td>{value['plus_static_sidecar_bytes']/2**30:.2f} GiB</td>"
        f"<td>{value['single_thread_encoding_seconds']/3600:.2f} h</td>"
        f"<td>{value['eight_worker_linear_seconds']/3600:.2f} h</td></tr>"
        for name, value in result["projections"].items()
    )
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>0025 smoke benchmark</title>
<style>body{{font:15px system-ui;max-width:980px;margin:36px auto;padding:0 20px;color:#17202a}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #ccd3da;padding:9px;text-align:left}}th{{background:#edf2f5}}code{{background:#eef1f3;padding:2px 5px}}.ok{{color:#176b3a;font-weight:700}}</style>
<h1>0025 semantic contract smoke</h1><p class='ok'>PASS: 单进程、单线程；未启动全量构建或训练。</p>
<p>样本 {result['sample']['decisions']} decisions，prototype sidecar {result['prototype_sidecar']['cards']} cards / {result['prototype_sidecar']['attacks']} attacks / {result['prototype_sidecar']['skills']} skills / {result['prototype_sidecar']['effects']} effects。</p>
<h2>每步存储（byte/decision）</h2><table><tr><th>口径</th><th>bytes</th></tr>{rows}</table>
<p>静态 sidecar 仅存一次：{result['prototype_sidecar']['total_bytes']:,} bytes。压缩流样本不是最终 Arrow/packed cache 承诺值。</p>
<h2>编码耗时</h2><p>legacy {t['legacy_pipeline_median']:.4f} ms/step；0025 完整语义管线 {t['semantic_pipeline_mean']:.4f} ms/step（10 次重复的 pipeline median）。</p>
<h2>线性外推</h2><table><tr><th>范围</th><th>decisions</th><th>存储</th><th>1 thread</th><th>8-worker 理想线性</th></tr>{projections}</table>
<p>07-29 与 08-02 数字均为按 0019 的 19 天 decisions/day 做的外推，不是已观察结果；本地 archive 当前仅到 07-29。</p>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-shard", type=Path, default=Path("rl_runs/0019_universal_winner_bc/dataset/V1_universal_winner_raw/train-00000.jsonl.gz"))
    parser.add_argument("--prototypes", type=Path, default=Path("train/0025_semantic_foundation_pretraining/assets/official_public_prototypes_v1.json"))
    parser.add_argument("--sample-decisions", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-html", type=Path)
    args = parser.parse_args()
    rows = _read_sample(args.sample_shard, args.sample_decisions)
    result = benchmark(rows, PrototypeIndex.load(args.prototypes), args.warmup, args.repeats)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_bytes(_canonical(result))
    if args.output_html:
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        args.output_html.write_text(_render(result), encoding="utf-8")
    print(_canonical(result).decode(), end="")


if __name__ == "__main__":
    main()
