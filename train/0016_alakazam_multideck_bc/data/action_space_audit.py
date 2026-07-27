"""Audit observed full-action and option-space bounds in the frozen 0016 catalog."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


def _observed_decisions(payload: Mapping[str, Any], actor: int) -> Iterator[dict[str, Any]]:
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("replay steps are absent")
    first = steps[0][0] if steps[0] else None
    visual = first.get("visualize") if isinstance(first, Mapping) else None
    if isinstance(visual, list):
        for frame_index, frame in enumerate(visual):
            if frame_index == 0 or not isinstance(frame, Mapping):
                continue
            observation = frame.get("obs")
            current = observation.get("current") if isinstance(observation, Mapping) else None
            if not isinstance(current, Mapping) or current.get("yourIndex") != actor:
                continue
            selected = frame.get("selected")
            select = observation.get("select")
            if not isinstance(selected, list) or not isinstance(select, Mapping):
                raise ValueError("visual decision lacks selected/select")
            yield _row(frame_index, select, selected)
        return
    for frame_index in range(len(steps) - 1):
        member = steps[frame_index][actor]
        observation = member.get("observation") if isinstance(member, Mapping) else None
        select = observation.get("select") if isinstance(observation, Mapping) else None
        current = observation.get("current") if isinstance(observation, Mapping) else None
        if (
            member.get("status") != "ACTIVE"
            or not isinstance(select, Mapping)
            or not isinstance(current, Mapping)
            or current.get("yourIndex") != actor
        ):
            continue
        selected = steps[frame_index + 1][actor].get("action")
        if selected is not None:
            yield _row(frame_index, select, selected)


def _row(frame_index: int, select: Mapping[str, Any], selected: object) -> dict[str, Any]:
    options = select.get("option")
    if not isinstance(options, list) or not isinstance(selected, list):
        raise ValueError("decision options/action must be lists")
    minimum = select.get("minCount")
    maximum = select.get("maxCount")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (minimum, maximum)):
        raise ValueError("decision bounds must be exact integers")
    if not minimum <= len(selected) <= maximum <= len(options):
        raise ValueError("observed action violates select bounds")
    if len(selected) != len(set(selected)) or any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(options)
        for index in selected
    ):
        raise ValueError("observed action indices are invalid")
    return {
        "frame_index": frame_index,
        "action_length": len(selected),
        "min_count": minimum,
        "max_count": maximum,
        "option_count": len(options),
        "select_type": select.get("type"),
        "select_context": select.get("context"),
    }


def _summarize(
    rows: list[dict[str, Any]], episodes: list[dict[str, Any]]
) -> dict[str, Any]:
    names = ("action_length", "min_count", "max_count", "option_count")
    histograms = {name: Counter() for name in names}
    maxima = {name: -1 for name in histograms}
    examples: dict[str, dict[str, Any]] = {}
    for row in rows:
        for name in histograms:
            value = int(row[name])
            histograms[name][value] += 1
            if value > maxima[name]:
                maxima[name] = value
                examples[name] = row
    return {
        "episodes": len(episodes),
        "decisions": len(rows),
        "maxima": maxima,
        "max_examples": examples,
        "histograms": {
            name: {str(key): value for key, value in sorted(counter.items())}
            for name, counter in histograms.items()
        },
        "thresholds": {
            str(limit): {
                name: sum(value for key, value in counter.items() if key > limit)
                for name, counter in histograms.items()
            }
            for limit in (16, 32, 60, 64, 99, 128)
        },
    }


def _audit_group(arguments: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    locator, episodes = arguments
    rows: list[dict[str, Any]] = []
    if locator.endswith(".zip"):
        with zipfile.ZipFile(locator) as bundle:
            for episode in episodes:
                payload = json.loads(bundle.read(episode["locator"]["member"]))
                rows.extend(_bind(episode, _observed_decisions(payload, episode["player_index"])))
    else:
        for episode in episodes:
            payload = json.loads(Path(episode["locator"]["member"]).read_bytes())
            rows.extend(_bind(episode, _observed_decisions(payload, episode["player_index"])))
    return _summarize(rows, episodes)


def _bind(episode: Mapping[str, Any], rows: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "episode_id": episode["episode_id"],
            "player_index": episode["player_index"],
            "team_name": episode["team_name"],
        }
        for row in rows
    ]


def _merge(parts: list[dict[str, Any]], catalog: Mapping[str, Any]) -> dict[str, Any]:
    names = ("action_length", "min_count", "max_count", "option_count")
    histograms = {name: Counter() for name in names}
    maxima = {name: -1 for name in histograms}
    examples: dict[str, dict[str, Any]] = {}
    for part in parts:
        for name, histogram in part["histograms"].items():
            histograms[name].update({int(key): value for key, value in histogram.items()})
            if part["maxima"][name] > maxima[name]:
                maxima[name] = part["maxima"][name]
                examples[name] = part["max_examples"][name]
    return {
        "schema_version": "0016_action_space_audit_v1",
        "catalog_sha256": catalog["catalog_sha256"],
        "episodes": sum(part["episodes"] for part in parts),
        "decisions": sum(part["decisions"] for part in parts),
        "maxima": maxima,
        "max_examples": examples,
        "histograms": {
            name: {str(key): value for key, value in sorted(counter.items())}
            for name, counter in histograms.items()
        },
        "thresholds": {
            str(limit): {
                name: sum(value for key, value in counter.items() if key > limit)
                for name, counter in histograms.items()
            }
            for limit in (16, 32, 60, 64, 99, 128)
        },
        "engine_contract": "minCount <= len(action) <= maxCount <= len(option)",
        "numeric_engine_cap": None,
    }


def build_audit(catalog_path: Path, workers: int) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = {}
    for episode in catalog["episodes"]:
        locator = episode["locator"]
        key = (
            locator["archive"]
            if locator["kind"] == "zip_member"
            else str(Path(locator["member"]).parent)
        )
        groups.setdefault(key, []).append(episode)
    arguments = sorted(groups.items())
    with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as pool:
        parts = list(pool.map(_audit_group, arguments))
    return _merge(parts, catalog)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    audit = build_audit(args.catalog, args.workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "decisions": audit["decisions"], **audit["maxima"]}
        )
    )


if __name__ == "__main__":
    main()
