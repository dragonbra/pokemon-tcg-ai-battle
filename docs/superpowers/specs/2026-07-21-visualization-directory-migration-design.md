# Replay 可视化目录迁移设计

## 目标

将当前位于 `scripts/` 的 replay 可视化实现和相关说明收拢到 `visualization/`，让该目录成为回放读取、校验、外部 viewer launcher 和 CLI 的唯一 ownership 入口。

迁移完成后，用户通过下面的模块命令查看 Kaggle 官方 replay 或本地带 `visualize` 帧的 replay：

```bash
python3 -m visualization.replay.cli path/to/replay.json
```

## 范围与非目标

本次工作包括：

- 把 `scripts/replay_visualizer.py` 的实现迁移到 `visualization/replay/core.py`。
- 把 `scripts/visualize_replay.py` 的 CLI 迁移到 `visualization/replay/cli.py`。
- 更新 `run_local_battle.py`、测试、README、`AGENTS.md` 和 replay 说明中的导入与命令。
- 删除旧的 `scripts/replay_visualizer.py` 和 `scripts/visualize_replay.py`，不保留兼容入口。
- 将完整使用说明收拢到 `visualization/README.md`，并说明外部 viewer、输入格式、本地可视化输出和旧 trace 限制。

本次工作不包括：

- 在仓库内实现卡牌图片资源或新的 HTML 播放器。
- 修改 agent 策略、卡组或官方引擎逻辑。
- 修改外部 viewer 的服务端行为。
- 把本地大体积 replay 写入仓库。

## 目录与职责

```text
visualization/
├── README.md
└── replay/
    ├── __init__.py
    ├── cli.py
    └── core.py
```

`core.py` 提供纯 Python 的 replay 解析、帧校验、trace 元数据补齐和 launcher 生成 API。`cli.py` 只负责参数解析、调用 core API、打印结果和返回错误码。`visualization/replay/__init__.py` 暴露需要被 runner 或测试复用的公共 API，但不复制实现。

## 数据流

输入 JSON 按以下优先级提取非空 `visualize` 帧数组：

1. 顶层 `visualize`。
2. 顶层 `visualize_frames`。
3. Kaggle `steps[*][*].visualize` 中第一个可用数组。

本地 runner 默认仍只写轻量 trace。传入 `--visualize-output PATH` 时，runner 在 `battle_finish()` 前调用 `visualize_data()`，复用 core 的校验和元数据补齐逻辑后写出独立 replay。旧的只有 `trace` 的 JSON 不伪造为可播放回放，而是返回包含重新生成命令的错误。

## Viewer launcher

CLI 默认生成系统临时目录中的 HTML launcher，并通过 `POST` 表单的 `json` hidden field 将完整帧数组提交给：

```text
https://ptcgvis.heroz.jp/Visualizer/Replay/0
```

`--no-open` 只生成 launcher 并打印路径；`--viewer-url` 可替换 endpoint。CLI 不预先发起 HTTP 请求，外部 viewer 的页面或资源问题在浏览器中诊断。

## 错误处理与验证

文件不可读、JSON 无效、顶层结构错误、visualize 帧为空或帧元素不是对象时，CLI 打印中文错误到 stderr 并返回 1。有效 replay 返回 0，并打印 launcher、viewer 和浏览器状态。

测试继续使用标准库 `unittest`，把 import 和 subprocess 命令切换到 `visualization.replay`。验证包括 replay 单元测试、CLI `--no-open`、`py_compile`/`compileall`、`check_assets.py` 和 `git diff --check`。
