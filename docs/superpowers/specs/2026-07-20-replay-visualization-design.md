# Replay 可视化工具设计

## 目标

为 `pokemon-tcg-ai-battle` 沉淀一套统一的 replay 可视化工具，能够处理 Kaggle 官方 replay 和本地 simulator 生成的回放，并在需要时把同一局对战交给现有的外部 viewer 展示。

主要入口为：

```bash
python3 scripts/visualize_replay.py path/to/replay.json
```

默认行为是生成一个临时 HTML launcher 并打开浏览器；`--no-open` 用于只生成 launcher 并打印路径，方便脚本和无图形环境使用。

## 范围与非目标

本次工作包括：

- 识别 Kaggle replay、本地带可视化帧的 replay，以及当前本地旧 trace。
- 从支持的 JSON 中提取 viewer 所需的 `visualize` 帧数组。
- 为本地 runner 增加可选的完整可视化帧输出。
- 通过 Kaggle notebook 已验证的表单 POST 协议调用外部 viewer。
- 对格式识别、帧校验、旧 trace 诊断、launcher 生成和命令行行为增加测试。

本次工作不包括：

- 在仓库内实现新的卡牌图片资源或完整的 HTML 播放器。
- 修改任何 agent 策略、卡组或官方引擎逻辑。
- 默认把本地 smoke test 的大型可视化数据写入仓库。
- 把没有 `visualize` 帧的旧 trace 事后伪造为可播放回放。

## 输入格式

统一读取 JSON 文件后按以下优先级提取帧：

1. 顶层 `visualize`：本地工具生成的标准化 replay。
2. 顶层 `visualize_frames`：兼容显式命名的本地 replay。
3. Kaggle `steps` 中第一个非空的 `visualize` 字段：与 notebook 使用的 `steps[0][0]["visualize"]` 结构兼容。每个 `visualize` 字段本身视为完整帧数组，不把不同 episode step 的数组再次拼接。

提取结果必须是非空的 JSON 数组，数组元素必须是 JSON 对象。原始 Kaggle JSON 不被修改；本地标准化 replay 保留原始 trace，并在顶层增加 `visualize`。

如果 JSON 只有 `trace`、`steps` 中没有可用 `visualize`，工具返回带输入路径和重新生成命令的明确错误。这样不会把缺少卡面状态的 trace 发送给 viewer，避免生成黑屏回放。

## 标准化本地 replay

`run_local_battle.py` 的默认输出保持现状。只有传入 `--visualize-output PATH` 时，runner 才在 `battle_finish()` 前调用 `cg.game.visualize_data()`，并把结果写到独立文件：

```json
{
  "replay_format": "ptcg-local-v1",
  "agent0": "alakazam_v7",
  "agent1": "official_water",
  "finished": true,
  "result": 1,
  "steps": 123,
  "error": null,
  "trace": [],
  "visualize": []
}
```

示例命令：

```bash
python3 scripts/run_local_battle.py \
  --agent0 alakazam_v7 \
  --agent1 official_water \
  --output /tmp/local-battle.json \
  --visualize-output /tmp/local-battle-visualize.json
```

可视化输出保留与普通 trace 相同的对局信息和 `trace`，并新增 viewer 帧。对于没有已有 `obs`/`action` 元数据的 engine 帧，使用本地 trace 的 action 条目按帧序号补齐；如果 observation 中有 `current.yourIndex`，把 action 放入对应玩家槽位，否则保留兼容性的双槽位表示；不覆盖已经存在的字段。由于 `visualize_data()` 必须在 `battle_finish()` 之前调用，任何获取失败都记录在输出错误中，并仍然执行 `battle_finish()`。

## Viewer launcher

外部 viewer 的协议是：

- 地址：`https://ptcgvis.heroz.jp/Visualizer/Replay/0`
- 方法：`POST`
- 表单字段：`json`
- 字段值：JSON 编码后的 `visualize` 帧数组
- 目标窗口：新窗口

工具通过标准库生成 `/tmp/ptcg-replay-*.html`。HTML 使用 hidden input 和自动提交 form，把完整 replay 放在 POST body 中而不是 URL 查询参数。默认使用 `webbrowser.open()` 打开 launcher；`--no-open` 不启动浏览器，但仍生成 launcher 并打印路径和 viewer endpoint。

viewer endpoint 作为可配置参数保留，便于外部站点变更时测试和替换：

```bash
python3 scripts/visualize_replay.py replay.json \
  --viewer-url https://ptcgvis.heroz.jp/Visualizer/Replay/0
```

## Python 接口

新增的实现模块提供以下可测试接口：

- `load_replay(path: Path) -> NormalizedReplay`
- `extract_visualize_frames(record: object) -> list[dict[str, object]]`
- `create_viewer_launcher(frames: list[dict[str, object]], output: Path, viewer_url: str) -> Path`
- `show_replay(path: Path, open_browser: bool = True, viewer_url: str = DEFAULT_VIEWER_URL) -> ReplayLaunch`

`NormalizedReplay` 至少包含输入路径、来源类型、标准化帧数组和原始 JSON；`ReplayLaunch` 至少包含 launcher 路径、viewer endpoint 和是否成功调用浏览器。CLI 只负责参数解析、调用接口和将用户可执行的错误信息打印到 stderr。

## 错误处理

以下情况必须失败且返回非零退出码：

- 文件不存在或无法读取。
- JSON 语法错误。
- 顶层结构不是对象。
- 找到 `visualize` 但它不是非空数组。
- 帧数组元素不是对象。
- 只有旧 trace、没有任何可视化帧。
- launcher 无法写入目标目录。

错误信息使用中文，至少包含输入路径和下一步操作。外部 viewer 的 HTTP 响应不在 CLI 中预先请求；浏览器 POST 失败属于 viewer 运行时问题，CLI 仍打印 launcher 路径，便于复现和诊断。

## 测试策略

测试放在 `tests/test_replay_visualization.py`，覆盖：

- Kaggle `steps[0][0].visualize` 提取。
- 顶层 `visualize` 和 `visualize_frames` 提取。
- 空帧、非法帧和旧 trace 的错误。
- 本地 trace action 与可视化帧的补齐规则。
- launcher 中的 POST endpoint、`json` 字段和 JSON payload。
- CLI `--no-open` 不调用浏览器并输出 launcher 路径。

验证还包括一份仓库现有 Kaggle replay 的实际标准化和 launcher 生成，以及一次本地 battle 的可选可视化输出。普通本地 smoke test 仍使用 `/tmp`，官方 Kaggle replay 仍是长期性能分析的主要来源。
