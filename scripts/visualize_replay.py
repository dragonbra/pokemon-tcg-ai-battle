#!/usr/bin/env python3
"""Submit a Kaggle or local visualizer replay to the external viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from replay_visualizer import DEFAULT_VIEWER_URL, ReplayFormatError, show_replay
else:
    from .replay_visualizer import DEFAULT_VIEWER_URL, ReplayFormatError, show_replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="Kaggle 或本地 replay JSON 文件")
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="只生成 launcher，不自动打开浏览器",
    )
    parser.add_argument(
        "--viewer-url",
        default=DEFAULT_VIEWER_URL,
        help=f"viewer POST 地址（默认：{DEFAULT_VIEWER_URL}）",
    )
    args = parser.parse_args(argv)
    try:
        launch = show_replay(
            args.replay,
            open_browser=not args.no_open,
            viewer_url=args.viewer_url,
        )
    except (OSError, ReplayFormatError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print(f"launcher: {launch.launcher}")
    print(f"viewer: {launch.viewer_url}")
    print("browser: opened" if launch.browser_opened else "browser: not opened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
