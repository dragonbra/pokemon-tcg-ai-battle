# 对局回放可视化

仓库提供统一入口，把 Kaggle 官方 replay 或本地保存了 `visualize` 帧的 replay 交给外部 PTCG viewer 播放。

## 查看已有 replay

```bash
python3 scripts/visualize_replay.py \
  replays/kaggle_v3_54813215/86731321/episode-86731321-replay.json
```

默认会生成系统临时目录中的 HTML launcher，并自动打开浏览器。只生成 launcher、不打开浏览器时：

```bash
python3 scripts/visualize_replay.py \
  replays/kaggle_v3_54813215/86731321/episode-86731321-replay.json \
  --no-open
```

命令会打印 launcher 路径和 viewer endpoint。launcher 使用 POST 的 `json` 字段提交完整 `visualize` 帧，不把大段回放数据放到 URL 中。

## 生成本地可视化 replay

普通本地 smoke test 仍然只保存轻量 trace。需要浏览器回放时显式增加 `--visualize-output`：

```bash
./scripts/run_local_battle.sh \
  --agent0 alakazam_v7 \
  --agent1 official_water \
  --output /tmp/ptcg-local-battle.json \
  --visualize-output /tmp/ptcg-local-battle-visualize.json

python3 scripts/visualize_replay.py \
  /tmp/ptcg-local-battle-visualize.json
```

可视化输出保留本地 `trace`，并在 engine 帧中补充对应的 observation/action；如果 observation 带有 `current.yourIndex`，action 会放到正确的玩家槽位，方便播放器回放和之后的策略复盘。输出应放在 `/tmp` 等临时目录，不纳入仓库。

## 支持范围

工具按以下顺序读取：

1. 本地 replay 顶层的 `visualize`。
2. 本地 replay 顶层的 `visualize_frames`。
3. Kaggle replay 的 `steps[*][*].visualize`，例如 notebook 使用的 `steps[0][0]["visualize"]`。

仓库之前生成的旧本地 JSON 只有 observation/action `trace`，没有引擎可视化帧，不能事后无损还原卡面。直接传入会失败并提示使用 `--visualize-output` 重新运行；这是为了避免生成播放器全黑或缺少动作的假回放。

## 卡牌显示

播放器展示依赖 viewer 能够读取对应的卡牌资源。回放帧本身包含卡牌 ID、名称和场上状态，但如果外部 viewer 的卡图资源不可用，卡牌区域可能仍然显示为黑色；这不是本地 runner 丢失了对局状态。先确认 viewer 站点和对应卡牌资源可访问，再检查 replay 是否含有非空 `visualize` 帧。

## Python 调用

需要由其他评测脚本调用时，可以复用：

```python
from pathlib import Path

from scripts.replay_visualizer import load_replay, show_replay

replay = load_replay(Path("episode-replay.json"))
print(replay.source, len(replay.frames))
launch = show_replay(replay.path, open_browser=True)
print(launch.launcher)
```

外部 viewer 地址可以通过 CLI 的 `--viewer-url` 或 `show_replay(..., viewer_url=...)` 替换。
