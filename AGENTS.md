# Repository Guidelines

## 项目结构与模块组织

- `submission/<name>/` 是可独立打包的 agent，必须包含 `main.py`、60 行 `deck.csv` 和 `cg/` 运行时；策略说明放在同目录的 `README.md` 或 `STRATEGY.md`。
- `scripts/` 提供资产校验、提交打包、本地对局、replay 可视化和官方引擎构建脚本。
- `data/official/` 是只读卡牌参考数据，`engine/source/` 是官方引擎源码；`engine/build/` 只保存本地构建产物。
- `notes/`、`reports/`、`experiments/` 保存事实、研究结论和实验记录，`replays/` 主要保存从 Kaggle 下载的官方 Episode replay/log JSON；本地 simulator 输出写到 `/tmp`，不纳入仓库。

## 宝可梦 TCG 规则学习长期记忆（V6）

详细证据见 [`reports/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`](reports/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md)，官方规则书为 [Pokémon TCG Rules](https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/par_rulebook_en.pdf)。后续新会话设计策略时，必须同时遵守下面的官方规则和 V6 已确认的策略语义。

### 官方规则硬约束

- 回合先抽牌，再按任意顺序执行主行动，最后才可以攻击；一旦宣告攻击，本回合立即结束，不能再进化、附能、使用 Supporter、铺 Bench 或使用普通 Ability。没有伤害的招式也仍然是攻击提交。
- 只有本回合开始时已经在场的 Pokémon 才能进化；本回合刚放下的 Basic 不能进化；同一只 Pokémon 同回合不能连续进化。Rare Candy 只能让合法在场的 Basic 跳过 Stage 1，不能绕过回合时机。
- Supporter、手动从手牌附 Energy、Retreat、Stadium 各自每回合一次；Item 和 Ability 没有统一次数上限，必须遵守具体卡文。Retreat 要支付 Retreat Cost，换位后仍可攻击。
- 进化会保留附着 Energy、工具和伤害指示物；附能不会因为进化重新触发。Special Energy 的效果必须按卡牌文本判断。
- 胜利来自拿完 Prize、对手没有可战斗 Pokémon，或对手在自己回合开始无法抽牌。牌库变空本身不会立即输，卡牌效果抽不到足够数量也不会立即输；规则没有通用手牌上限。
- Prize、手牌、Active、Bench、弃牌区和牌库是不同资源区域；卡牌效果改变区域时必须更新资源位置，不要把“当前看不到”误当成“仍在牌库”。

### V6 已确认策略语义

- 攻击是回合终止提交。低阶段 Abra/Kadabra 攻击是最后手段；只要仍能完成 Alakazam 进化、后场铺场、过牌或明确的下一条攻击路线，就先准备再攻击。
- Active Kadabra 已有 Psychic Energy 且手里有 Alakazam 时，优先自然进化 Active 并攻击；之后可以把合法 Bench Abra 进化为 Kadabra，使用其 Psychic Draw。不要为了立即消耗 Rare Candy 把 Bench Abra 直进。
- Active 已经可以有效攻击，且有 Enriching Energy 和 Dudunsparce 时，V6 初版默认把 Enriching Energy 给 Dudunsparce 过牌；没有 Enriching Energy 时，其他 Energy 优先给没有 Psychic Energy 的 Abra 线。Dudunsparce 的净牌库变化按“抽 3 张 − 本体 − 所有附着卡”计算。
- 对手 Active 不能被击倒而 Bench 有确定 KO 目标时，优先使用 Boss；多个可击倒目标之间优先剩余 HP 最高者。Active 已能攻击、无 Boss KO 目标且对手手牌至少 6 张、Active 已受伤时，Xerosic 可在攻击前使用；不要把 Fezandipiti Ability 当作 Supporter。
- Lana’s Aid 是 Supporter，适合一次性从弃牌区拿回 Pokémon 与 Basic Energy；Night Stretcher 是 Item，可以和另一个 Supporter 组合。Supporter 机会一回合只有一次。
- 牌库超过 15 张时相对自由；11–15 张轻度警戒；10 张进入中局保护。所有 deck → hand 的抽牌和检索都消耗牌库预算；低于 10 张时，只有“继续过牌增加伤害 → 必要时 Boss → 本回合拿完最后 Prize”的终局闭环才允许继续消耗。
- Dunsparce 的 Trading Places 是攻击，不是普通换位；V6 初版禁用。只有 Retreat 后能让已经具备攻击条件的 Alakazam 在本回合攻击时，才立即 Retreat；换位后仍不能攻击时，才让低 Prize 的 Dunsparce 留在 Active。
- 不预设 opponent ID 或完整对手构筑，只根据 observation 中实际看到的 Pokémon、Energy、伤害、手牌、Prize 和牌库压力做决策。Land Collapse 只在实际观察到 Great Tusk/攻击压力后按通用进攻与牌库规则处理，不增加预设 matchup 分支。

### V6 必须维护的状态来源

- 新对局开始时清空本地动作历史和 effect serial；每个己方回合开始保存 Active/Bench Pokémon 实例快照。
- 记录每只 Pokémon 的进场回合、进化回合、本回合是否已使用 Supporter/手填 Energy/Retreat，以及是否已经攻击或主动结束回合。用这些时间节点判断进化时机，不能只看当前 card ID。
- 以固定卡组总量维护 Abra、Kadabra、Alakazam 等资源账本，追踪 Prize、手牌、Active、Bench、弃牌区和牌库；用可见卡牌和 `deckCount` 交叉核对，未知 Prize 保留已知/未知边界。
- Simulator 的 `supporterPlayed`、`energyAttached`、`retreated`、`appearThisTurn` 等字段用于合法性校验；agent 自己的动作历史用于补充发生时点和策略语义。

## 构建、测试与本地开发

项目要求 Python 3.11+。常用命令如下：

```bash
python3 scripts/check_assets.py
bash scripts/package_submission.sh alakazam_v1
./scripts/run_local_battle.sh --agent0 alakazam_v1 --agent1 official_water \
  --output /tmp/ptcg-local-battle.json
```

第一条检查卡牌数据、60 张卡组和模拟器文件；第二条生成 `dist/<name>.tar.gz`；本地 battle runner 仅用于临时调试，输出应写到 `/tmp`。官方真实对局以 Kaggle submission/episode/replay 为主要分析依据。若动态库不兼容，设置 `PTCG_CXX_RUNTIME=/path/to/runtime`；需要回归官方源码时运行 `./scripts/build_official_engine.sh`，再设置 `PTCG_CG_LIBRARY`。

### Replay 可视化工具

- 当用户要求查看、展示、播放或可视化某一局对战时，优先调用统一入口：`python3 scripts/visualize_replay.py <replay.json>`。默认生成临时 HTML launcher 并打开浏览器；无图形环境或只需要路径时使用 `--no-open`。
- 统一入口兼容 Kaggle 官方 replay、本地顶层包含 `visualize`/`visualize_frames` 的 replay，以及 Kaggle `steps[*][*].visualize` 帧。外部 viewer 使用 `POST` 的 `json` 字段提交完整帧，默认 endpoint 为 `https://ptcgvis.heroz.jp/Visualizer/Replay/0`；需要替换时使用 `--viewer-url`。
- 当前只有 observation/action 的旧本地 trace 不包含引擎可视化帧，不能事后伪造卡面回放。遇到这类文件时应明确提示重新运行，并使用 `--visualize-output` 生成可播放文件：

  ```bash
  ./scripts/run_local_battle.sh \
    --agent0 alakazam_v7 --agent1 official_water \
    --output /tmp/ptcg-local-battle.json \
    --visualize-output /tmp/ptcg-local-battle-visualize.json
  python3 scripts/visualize_replay.py /tmp/ptcg-local-battle-visualize.json
  ```

- `run_local_battle.py` 默认仍只保存轻量 trace；只有显式传入 `--visualize-output` 时才在 `battle_finish()` 前调用 `visualize_data()`。本地临时 replay 写入 `/tmp`，不纳入仓库；Kaggle 官方 replay 仍作为长期分析资料。
- 如果播放器黑屏，先检查输入是否含非空 `visualize` 帧；如果状态存在但卡面为黑色，再检查外部 viewer 的卡牌资源是否可访问。完整说明见 [`docs/replay-visualization.md`](docs/replay-visualization.md)。

## 编码风格与命名约定

Python 使用 4 个空格、类型注解和清晰的小函数；遵守 Ruff 的 100 字符行宽（`pyproject.toml`）。变量、函数使用 `snake_case`，常量使用 `UPPER_SNAKE_CASE`，提交目录使用小写下划线。agent 必须只返回模拟器提供的合法选项，并保持策略确定性；Shell 脚本沿用 `set -euo pipefail`。

## 测试指南

仓库当前没有 pytest 测试套件。提交前至少运行 `python3 scripts/check_assets.py`；本地对局不是当前主要评测来源，如需运行只将输出写到 `/tmp`。策略或引擎改动应优先记录 Kaggle 官方 Episode，并在说明中记录对手、步数和结果。必要时使用 `python3 -m compileall scripts submission` 检查语法。

## 提交与 Pull Request

提交信息采用 Git 风格操作前缀加中文说明，例如 `feat: 新增胡地策略`、`fix: 修复合法选项选择`、`docs: 补充实验记录`、`test: 记录本地对局`、`chore: 整理脚本`；一次提交聚焦一个主题。除非用户主动要求，否则不要执行 `git commit`。每次执行 `git commit` 后必须立即执行对应的 `git push`；如果远端、分支或权限导致无法 push，应报告失败，不得把仅完成本地 commit 当作同步完成。PR 应说明动机、改动目录、验证命令和结果，涉及策略时附 replay 或指标；本项目无 UI，截图仅在确有帮助时提供。不要提交 Kaggle 凭据、`.env`、下载的官方数据或未经许可的二进制资源，也不要自动上传正式 submission。
