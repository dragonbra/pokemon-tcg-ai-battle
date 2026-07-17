# Repository Guidelines

## 项目结构与模块组织

- `submission/<name>/` 是可独立打包的 agent，必须包含 `main.py`、60 行 `deck.csv` 和 `cg/` 运行时；策略说明放在同目录的 `README.md` 或 `STRATEGY.md`。
- `scripts/` 提供资产校验、提交打包、本地对局和官方引擎构建脚本。
- `data/official/` 是只读卡牌参考数据，`engine/source/` 是官方引擎源码；`engine/build/` 只保存本地构建产物。
- `notes/`、`reports/`、`experiments/` 保存事实、研究结论和实验记录，`replays/` 保存可复现的对局 JSON。

## 构建、测试与本地开发

项目要求 Python 3.11+。常用命令如下：

```bash
python3 scripts/check_assets.py
bash scripts/package_submission.sh alakazam_v1
./scripts/run_local_battle.sh --agent0 alakazam_v1 --agent1 official_water
```

第一条检查卡牌数据、60 张卡组和模拟器文件；第二条生成 `dist/<name>.tar.gz`；第三条运行一局本地 AI 对局并写入 `replays/local_battle.json`。若动态库不兼容，设置 `PTCG_CXX_RUNTIME=/path/to/runtime`；需要回归官方源码时运行 `./scripts/build_official_engine.sh`，再设置 `PTCG_CG_LIBRARY`。

## 编码风格与命名约定

Python 使用 4 个空格、类型注解和清晰的小函数；遵守 Ruff 的 100 字符行宽（`pyproject.toml`）。变量、函数使用 `snake_case`，常量使用 `UPPER_SNAKE_CASE`，提交目录使用小写下划线。agent 必须只返回模拟器提供的合法选项，并保持策略确定性；Shell 脚本沿用 `set -euo pipefail`。

## 测试指南

仓库当前没有 pytest 测试套件。提交前至少运行 `python3 scripts/check_assets.py`，并让目标 agent 完成一局本地对局；确认输出中的 `finished` 为 `true` 且 `error` 为 `null`。策略或引擎改动应保留 replay，并在说明中记录对手、步数和结果。必要时使用 `python3 -m compileall scripts submission` 检查语法。

## 提交与 Pull Request

提交信息采用 Git 风格操作前缀加中文说明，例如 `feat: 新增胡地策略`、`fix: 修复合法选项选择`、`docs: 补充实验记录`、`test: 记录本地对局`、`chore: 整理脚本`；一次提交聚焦一个主题。除非用户主动要求，否则不要执行 `git commit`。PR 应说明动机、改动目录、验证命令和结果，涉及策略时附 replay 或指标；本项目无 UI，截图仅在确有帮助时提供。不要提交 Kaggle 凭据、`.env`、下载的官方数据或未经许可的二进制资源，也不要自动上传正式 submission。
