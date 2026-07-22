# 回放可视化

本目录负责把 Kaggle 官方 replay 或本地生成的 `visualize` 帧交给外部 PTCG viewer 展示。核心实现位于 [`replay/`](replay/)，CLI 使用 Python 模块入口运行。

## 查看已有 replay

```bash
python3 -m visualization.replay.cli \
  replays/kaggle_v3_54813215/86731321/episode-86731321-replay.json
```

默认会生成系统临时目录中的 HTML launcher，并自动在当前标签页 POST 跳转到 viewer，不创建新窗口。只生成 launcher、不打开浏览器时：

```bash
python3 -m visualization.replay.cli \
  replays/kaggle_v3_54813215/86731321/episode-86731321-replay.json \
  --no-open
```

命令会打印 launcher 路径和 viewer endpoint。launcher 使用 `POST` 的 `json` 字段提交完整 `visualize` 帧，不把大段回放数据放到 URL 中；提交结果会替换当前标签页。默认 endpoint 是：

```text
https://ptcgvis.heroz.jp/Visualizer/Replay/0
```

需要替换 viewer 地址时：

```bash
python3 -m visualization.replay.cli path/to/replay.json \
  --viewer-url https://example.test/Visualizer/Replay/0
```

## 本地可视化 replay

根目录不再提供一次性本地对局 wrapper。需要生成本地回放时，调用方必须在对局结束前
直接从 engine 导出非空 `visualize` 帧；仅有 observation/action 的轻量 trace 无法事后还原
卡面。临时回放应写入 `/tmp`，不纳入仓库。

`attach_trace_metadata` 可以把已有 trace 中的 observation/action 补充到对应 engine 帧；
如果 observation 带有 `current.yourIndex`，action 会放到正确的玩家槽位。

## 支持的输入

工具按以下顺序读取非空的 `visualize` 帧数组：

1. 本地 replay 顶层的 `visualize`。
2. 本地 replay 顶层的 `visualize_frames`。
3. Kaggle replay 的 `steps[*][*].visualize`，例如 notebook 使用的 `steps[0][0]["visualize"]`。

仓库之前生成的旧本地 JSON 只有 observation/action `trace`，没有引擎可视化帧，不能事后无损还原卡面。直接传入会失败并提示使用 `--visualize-output` 重新运行；这是为了避免生成播放器全黑或缺少动作的假回放。

## 卡牌显示与故障排查

播放器展示依赖 viewer 能够读取对应的卡牌资源。回放帧本身包含卡牌 ID、名称和场上状态，但如果外部 viewer 的卡图资源不可用，卡牌区域可能仍然显示为黑色。

遇到黑屏时按以下顺序检查：

1. 输入 JSON 是否包含非空的 `visualize` 帧。
2. 是否误把只有 `trace` 的旧本地 JSON 交给 viewer。
3. 外部 viewer 站点和对应卡牌资源是否可以访问。
4. `--viewer-url` 是否指向接受 `POST`、字段名为 `json` 的 endpoint。

CLI 不会预先请求外部 viewer；浏览器打开 launcher 后会在当前标签页进入 viewer，页面和资源问题需要在浏览器中诊断。

## Python 调用

其他评测脚本可以复用 `visualization.replay` 的公共 API：

```python
from pathlib import Path

from visualization.replay import load_replay, show_replay

replay = load_replay(Path("episode-replay.json"))
print(replay.source, len(replay.frames))
launch = show_replay(replay.path, open_browser=True)
print(launch.launcher)
```

`visualization.replay.core` 提供相同的具体实现 API，包括 `extract_visualize_frames`、`attach_trace_metadata` 和 `create_viewer_launcher`。
