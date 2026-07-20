"""Load replay JSON and prepare it for the external PTCG viewer."""

from __future__ import annotations

import html
import json
import tempfile
import webbrowser
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any


DEFAULT_VIEWER_URL = "https://ptcgvis.heroz.jp/Visualizer/Replay/0"


class ReplayFormatError(ValueError):
    """Raised when a replay cannot provide valid visualizer frames."""


@dataclass(frozen=True)
class NormalizedReplay:
    path: Path
    source: str
    frames: list[dict[str, Any]]
    record: dict[str, Any]


@dataclass(frozen=True)
class ReplayLaunch:
    launcher: Path
    viewer_url: str
    browser_opened: bool


def _validate_frames(value: object, source: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ReplayFormatError(f"回放没有可用的 visualize 帧，来源：{source}")
    if not all(isinstance(frame, dict) for frame in value):
        raise ReplayFormatError(f"visualize 帧必须是对象，来源：{source}")
    return value


def _extract_with_source(record: object) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(record, dict):
        raise ReplayFormatError("回放 JSON 顶层必须是对象")
    if "visualize" in record:
        return "top-level", _validate_frames(record["visualize"], "顶层 visualize")
    if "visualize_frames" in record:
        return "visualize_frames", _validate_frames(
            record["visualize_frames"], "顶层 visualize_frames"
        )
    steps = record.get("steps", [])
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, list):
                continue
            for item in step:
                if isinstance(item, dict) and item.get("visualize"):
                    return "kaggle-steps", _validate_frames(item["visualize"], "Kaggle steps")
    raise ReplayFormatError(
        "回放没有 visualize 帧；旧 trace 需要用 --visualize-output 重新生成"
    )


def extract_visualize_frames(record: object) -> list[dict[str, Any]]:
    """Extract the first complete visualizer frame array from a replay record."""
    return _extract_with_source(record)[1]


def load_replay(path: Path) -> NormalizedReplay:
    """Load and validate a replay JSON file without modifying its contents."""
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, JSONDecodeError) as exc:
        raise ReplayFormatError(f"无法读取回放 {path}: {exc}") from exc
    try:
        source, frames = _extract_with_source(record)
    except ReplayFormatError as exc:
        raise ReplayFormatError(f"回放 {path}: {exc}") from exc
    return NormalizedReplay(path=path, source=source, frames=frames, record=record)


def attach_trace_metadata(
    frames: list[dict[str, Any]], trace: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach local observations/actions to frame positions without overwriting fields."""
    action_entries = [
        entry for entry in trace if isinstance(entry, dict) and "action" in entry
    ]
    for frame_index, entry in enumerate(action_entries, start=1):
        if frame_index >= len(frames):
            break
        frame = frames[frame_index]
        frame.setdefault("obs", entry.get("observation"))
        action = entry.get("action")
        observation = entry.get("observation")
        current = observation.get("current") if isinstance(observation, dict) else None
        player_index = current.get("yourIndex") if isinstance(current, dict) else None
        if player_index in (0, 1):
            action_pair: list[Any] = [[], []]
            action_pair[player_index] = action
        else:
            action_pair = [action, action]
        frame.setdefault("action", action_pair)
    return frames


def create_viewer_launcher(
    frames: list[dict[str, Any]], output: Path, viewer_url: str = DEFAULT_VIEWER_URL
) -> Path:
    """Write a local page that POSTs frames to the external viewer."""
    _validate_frames(frames, "launcher")
    output = Path(output)
    payload = html.escape(json.dumps(frames, ensure_ascii=False), quote=True)
    action = html.escape(viewer_url, quote=True)
    document = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>PTCG replay launcher</title></head>
<body>
<form method="POST" action="{action}" target="_blank">
<input type="hidden" name="json" value="{payload}">
</form>
<script>window.addEventListener("load", function () {{ document.forms[0].submit(); }});</script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def show_replay(
    path: Path,
    open_browser: bool = True,
    viewer_url: str = DEFAULT_VIEWER_URL,
) -> ReplayLaunch:
    """Create a launcher for a replay and optionally open it in the browser."""
    replay = load_replay(path)
    with tempfile.NamedTemporaryFile(
        prefix="ptcg-replay-", suffix=".html", mode="w", encoding="utf-8", delete=False
    ) as handle:
        launcher = Path(handle.name)
    create_viewer_launcher(replay.frames, launcher, viewer_url)
    browser_opened = bool(webbrowser.open(launcher.as_uri())) if open_browser else False
    return ReplayLaunch(
        launcher=launcher,
        viewer_url=viewer_url,
        browser_opened=browser_opened,
    )
