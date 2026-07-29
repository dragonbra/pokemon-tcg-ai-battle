from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit CUDA runtime activity issued inside one NVTX range."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("--range", dest="range_name", required=True)
    parser.add_argument(
        "--scope",
        default="engine_step_device_codec_device_route; policies resident but model forwards are not in this range",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")')
    }


def require_columns(
    connection: sqlite3.Connection, table: str, required: set[str]
) -> None:
    present = columns(connection, table)
    missing = required - present
    if missing:
        raise RuntimeError(f"{table} is missing columns: {sorted(missing)}")


def find_range(
    connection: sqlite3.Connection, range_name: str
) -> tuple[int, int]:
    require_columns(connection, "NVTX_EVENTS", {"start", "end"})
    nvtx_columns = columns(connection, "NVTX_EVENTS")
    clauses: list[str] = []
    parameters: list[Any] = []
    joins = ""
    if "text" in nvtx_columns:
        clauses.append("n.text = ?")
        parameters.append(range_name)
    if "textId" in nvtx_columns:
        require_columns(connection, "StringIds", {"id", "value"})
        joins = " LEFT JOIN StringIds s ON s.id = n.textId"
        clauses.append("s.value = ?")
        parameters.append(range_name)
    if not clauses:
        raise RuntimeError("NVTX_EVENTS exposes neither text nor textId")
    rows = list(
        connection.execute(
            "SELECT n.start, n.end FROM NVTX_EVENTS n"
            + joins
            + " WHERE ("
            + " OR ".join(clauses)
            + ") AND n.end IS NOT NULL ORDER BY n.start",
            parameters,
        )
    )
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one completed NVTX range named {range_name!r}, "
            f"found {len(rows)}"
        )
    return int(rows[0][0]), int(rows[0][1])


def main() -> None:
    args = parse_args()
    connection = sqlite3.connect(args.database)
    try:
        require_columns(
            connection,
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            {"start", "end", "nameId", "correlationId"},
        )
        require_columns(connection, "StringIds", {"id", "value"})
        range_start, range_end = find_range(connection, args.range_name)
        runtime_rows = list(
            connection.execute(
                """
                SELECT s.value, COUNT(*), SUM(r.end - r.start)
                FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                JOIN StringIds s ON s.id = r.nameId
                WHERE r.start < ? AND r.end > ?
                GROUP BY s.value
                ORDER BY COUNT(*) DESC, s.value
                """,
                (range_end, range_start),
            )
        )
        api_calls = [
            {"name": str(name), "calls": int(count), "total_time_ns": int(total)}
            for name, count, total in runtime_rows
        ]

        transfer_rows: list[tuple[Any, ...]] = []
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }:
            require_columns(
                connection,
                "CUPTI_ACTIVITY_KIND_MEMCPY",
                {"correlationId", "bytes", "copyKind", "srcKind", "dstKind"},
            )
            transfer_rows = list(
                connection.execute(
                    """
                    SELECT s.value, m.copyKind, m.srcKind, m.dstKind,
                           COUNT(*), SUM(m.bytes)
                    FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                    JOIN StringIds s ON s.id = r.nameId
                    JOIN CUPTI_ACTIVITY_KIND_MEMCPY m
                      ON m.correlationId = r.correlationId
                    WHERE r.start < ? AND r.end > ?
                    GROUP BY s.value, m.copyKind, m.srcKind, m.dstKind
                    ORDER BY SUM(m.bytes) DESC
                    """,
                    (range_end, range_start),
                )
            )
        transfers = [
            {
                "api": str(name),
                "copy_kind": int(copy_kind),
                "source_kind": int(source_kind),
                "destination_kind": int(destination_kind),
                "operations": int(count),
                "bytes": int(byte_count),
            }
            for name, copy_kind, source_kind, destination_kind, count, byte_count
            in transfer_rows
        ]
        host_device_transfers = [
            row
            for row in transfers
            if row["source_kind"] != row["destination_kind"]
        ]
        same_kind_transfers = [
            row
            for row in transfers
            if row["source_kind"] == row["destination_kind"]
        ]
        synchronization_calls = [
            row
            for row in api_calls
            if "synchronize" in row["name"].lower()
            or row["name"] in {"cudaMemcpy", "cudaFree", "cudaMalloc"}
        ]
        result = {
            "database": str(args.database),
            "nvtx_range": args.range_name,
            "range_start_ns": range_start,
            "range_end_ns": range_end,
            "range_duration_ns": range_end - range_start,
            "scope": args.scope,
            "api_calls": api_calls,
            "correlated_memcpy_operations": transfers,
            "same_memory_kind_memcpy_operations": same_kind_transfers,
            "host_device_memcpy_operations": host_device_transfers,
            "synchronization_or_blocking_runtime_calls": synchronization_calls,
            "passed_no_memcpy_or_host_synchronization": (
                not transfers and not synchronization_calls
            ),
            "passed_no_host_device_transfer_or_host_synchronization": (
                not host_device_transfers and not synchronization_calls
            ),
        }
    finally:
        connection.close()
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not result["passed_no_host_device_transfer_or_host_synchronization"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
