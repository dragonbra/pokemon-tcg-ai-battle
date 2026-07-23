# Repository Guidelines

## 项目结构与模块组织

- `work/<name>/` 是当前可打包候选，必须包含 `main.py`、60 行 `deck.csv` 和 `cg/` 运行时；策略说明放在 `work/docs/`。`submission/<name>/` 保留历史提交源目录。
- `scripts/` 只保留训练观测入口 `start_tensorboard.sh`；通用训练基建放在 `rl_environment/`，具体训练项目放在 `train/<project>/`；其他基建必须放在所属的 Python package（如 `evaluation/`、`visualization/`）中，不再新增一次性或规则策略脚本。
- `visualization/` 提供 replay 可视化核心、外部 viewer launcher、CLI 和使用说明。
- `evaluation/` 是仓库内的评测运行入口：`configs/opponents.json` 固定 catalog，`opponents/<name>/` 下每个对手都是独立标准 package（`main.py`、60 行 `deck.csv`、物理复制的 `cg/`），不是 adapter。官方 engine runtime 是唯一运行时来源；评测代码不得修改 `engine/source/`，也不得依赖隔壁评测仓库。
- 评测 CLI 使用 `python3 -m evaluation list-opponents`、`validate <package>` 和 `run --candidate <package> --opponents all --output <report-root>`。每次 `run` 在指定报告根目录下创建独立 `run_id/`，写入 manifest、逐局记录、指标、case 及 Markdown/HTML 报告；`evaluation/opponents/` 不属于 Kaggle 正式 submission 输出目录。
- 每场评测在独立 worker 进程中运行，隔离双方策略的模块级状态、导入缓存和 cg 状态。完整 trace 仅在当前 run 的临时目录保留，结束时默认删除；长期报告最多保留三份被选中的完整 trace，只有调试时才使用 `--keep-temp`。
- AutoIteration 使用 `--metric-profile auto_iteration_v8_setup_relay`（当前 revision 2）生成语义化完整报告；报告按结果护栏、阶段一二回合基础能力、阶段二 Post-KO 接力、阶段三攻击质量和辅助审计分组，同时保留原始 metric payload。使用步骤、产物和调用边界见 [`evaluation/HANDOFF.md`](evaluation/HANDOFF.md)。
- `data/official/` 是只读卡牌参考数据，`engine/source/` 是官方引擎源码；`engine/build/` 只保存本地构建产物。
- `notes/`、`docs/reports/`、`experiments/` 保存事实、研究结论和实验记录，`replays/` 主要保存从 Kaggle 下载的官方 Episode replay/log JSON；本地 simulator 输出写到 `/tmp`，不纳入仓库。

## 官方引擎与评测硬约束

- 任何开发、调试、训练、评测或自动化行为都不得修改官方提供的 `engine/source/` 源代码，包括直接编辑、格式化、自动修复或生成补丁。允许只读分析与构建；构建产物只能写入 `engine/build/` 或仓库外临时目录。
- 所有用于判断 agent 实际能力、比较版本或形成评测结论的结果，都必须来自使用官方 engine runtime 执行的真实模拟对战。静态分析、mock、伪造 trace、单元测试或合法性检查只能作为辅助证据，不能替代真实对局评测。
- 即使用户明确要求修改 `engine/source/`，也必须在执行前停止，明确警告这会偏离官方引擎、破坏评测真实性和结果可比性，并先与用户讨论用途、影响与替代方案；只有再次取得明确确认后才能继续。

## RL 设计文档同步约定

- [`train/alakazam_bc_rl/DESIGN.html`](train/alakazam_bc_rl/DESIGN.html) 是 RL 模型输入、网络结构、训练目标和项目阶段的项目级可视化现状文档，不是一次性说明或历史快照。
- 后续任何改动只要涉及模型输入或 feature schema、模型结构或输出 head、action contract、训练 loss/reward/value/PPO 接口，或者项目从 BC、rollout、value calibration、RL fine-tuning 等阶段推进到新阶段，都必须在同一项工作中同步更新 `train/alakazam_bc_rl/DESIGN.html`。
- 同步内容至少覆盖受影响的字段与张量 shape、模型数据流与参数结构、训练目标、当前/下一阶段、正式工作包与研究 checkpoint 的边界；不得让页面继续展示已经失效的 schema、数字或阶段结论。
- 实现或配置已经变化但 `train/alakazam_bc_rl/DESIGN.html` 尚未同步时，该项工作视为未完成。交付前必须用当前代码、数据 audit、checkpoint metadata 和 run manifest 交叉核对页面内容；不能只根据旧报告手工推断。
- 纯粹的内部重构、文件移动或不改变模型/训练语义的修复不要求制造文档改动；但如果路径发生变化，页面中的事实来源链接也必须保持可用。
- BC 数据默认只允许来自同一个明确的 team/agent policy；manifest、dataset summary 和 run record 必须记录并校验该来源。不得把不同 team、不同实现逻辑的动作标签直接混成一个无条件 BC policy；只有在模型显式加入可审计的 expert/source conditioning、并单独设计冲突标签与分来源评测时，才可以开启多来源训练。

## RL 实验内迭代版本硬约束

- `rl_runs/<000N-experiment>/` 表示一个项目级实验，根目录只保存共享的 manifest、数据 manifest/audit、决策与命令记录；同一实验内每次实际训练、校准或策略更新都必须新建 `V<序号>_<tag>/` 子目录，例如 `V1_initial_contract`、`V2_card_token_fix`。
- 序号从 `V1` 开始，在同一实验内严格单调递增；`tag` 使用能说明本次假设或修复的 ASCII 小写 `snake_case`。不得复用旧序号、覆盖旧目录、向旧 metrics 追加新 run，或因为结果失败而删除旧版本。
- 每个版本的 tracked 训练产物必须写到 `rl_runs/<experiment>/V<n>_<tag>/`，至少包含独立的 training config、metrics、summary/status；TensorBoard event 必须写到 `rl_runs/tensorboard/<experiment>/V<n>_<tag>/`，checkpoint 必须写到 `rl_runs/checkpoint/<experiment>/V<n>_<tag>/`。三处版本名必须完全一致，禁止直接把 event 或 checkpoint 写在 experiment 根目录。
- 评测同样使用 `evaluation/V<n>_<tag>/`，并与产生 candidate 的训练版本对应；仅评测解析器或 opponent catalog 变化时也要新建版本并在 tag/决策记录中说明变量，不得混写已有报告。
- 失败、被中止或确认存在数据/编码缺陷的版本仍然是正式迭代记录：必须保留可恢复产物，在 version status 或 experiment decisions 中记录失败原因、发现证据和下一版本具体改进。后续版本的价值需要能从这些记录和 TensorBoard 曲线中被追溯。
- 启动新 run 前必须先检查目标 version 的 run、TensorBoard 和 checkpoint 三个目录均未被使用；若任一路径已有文件，必须分配下一个 `V<n>_<tag>`，不得依赖 TensorBoard 新建 event 文件来区分逻辑 run。

## 宝可梦 TCG 规则学习长期记忆

详细证据见 [`docs/reports/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`](docs/reports/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md)，官方规则书为 [Pokémon TCG Rules](https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/par_rulebook_en.pdf)。后续新会话设计策略时，必须同时遵守下面的官方规则和已确认的策略语义。

### 官方规则硬约束

- 回合先抽牌，再按任意顺序执行主行动，最后才可以攻击；一旦宣告攻击，本回合立即结束，不能再进化、附能、使用 Supporter、铺 Bench 或使用普通 Ability。没有伤害的招式也仍然是攻击提交。
- 只有本回合开始时已经在场的 Pokémon 才能进化；本回合刚放下的 Basic 不能进化；同一只 Pokémon 同回合不能连续进化。Rare Candy 只能让合法在场的 Basic 跳过 Stage 1，不能绕过回合时机。
- Supporter、手动从手牌附 Energy、Retreat、Stadium 各自每回合一次；Item 和 Ability 没有统一次数上限，必须遵守具体卡文。Retreat 要支付 Retreat Cost，换位后仍可攻击。
- 进化会保留附着 Energy、工具和伤害指示物；附能不会因为进化重新触发。Special Energy 的效果必须按卡牌文本判断。
- 胜利来自拿完 Prize、对手没有可战斗 Pokémon，或对手在自己回合开始无法抽牌。牌库变空本身不会立即输，卡牌效果抽不到足够数量也不会立即输；规则没有通用手牌上限。
- Prize、手牌、Active、Bench、弃牌区和牌库是不同资源区域；卡牌效果改变区域时必须更新资源位置，不要把“当前看不到”误当成“仍在牌库”。

### 已确认策略语义

- 攻击是回合终止提交。低阶段 Abra/Kadabra 攻击是最后手段；只要仍能完成 Alakazam 进化、后场铺场、过牌或明确的下一条攻击路线，就先准备再攻击。
- Active Kadabra 已有 Psychic Energy 且手里有 Alakazam 时，优先自然进化 Active 并攻击；之后可以把合法 Bench Abra 进化为 Kadabra，使用其 Psychic Draw。不要为了立即消耗 Rare Candy 把 Bench Abra 直进。
- Active 已经可以有效攻击，且有 Enriching Energy 和 Dudunsparce 时，V6 初版默认把 Enriching Energy 给 Dudunsparce 过牌；没有 Enriching Energy 时，其他 Energy 优先给没有 Psychic Energy 的 Abra 线。Dudunsparce 的净牌库变化按“抽 3 张 − 本体 − 所有附着卡”计算。
- 对手 Active 不能被击倒而 Bench 有确定 KO 目标时，优先使用 Boss；多个可击倒目标之间优先剩余 HP 最高者。Active 已能攻击、无 Boss KO 目标且对手手牌至少 6 张、Active 已受伤时，Xerosic 可在攻击前使用；不要把 Fezandipiti Ability 当作 Supporter。
- Lana’s Aid 是 Supporter，适合一次性从弃牌区拿回 Pokémon 与 Basic Energy；Night Stretcher 是 Item，可以和另一个 Supporter 组合。Supporter 机会一回合只有一次。
- 牌库超过 15 张时相对自由；11–15 张轻度警戒；10 张进入中局保护。所有 deck → hand 的抽牌和检索都消耗牌库预算；低于 10 张时，只有“继续过牌增加伤害 → 必要时 Boss → 本回合拿完最后 Prize”的终局闭环才允许继续消耗。
- Dunsparce 的 Trading Places 是攻击，不是普通换位；V6 初版禁用。只有 Retreat 后能让已经具备攻击条件的 Alakazam 在本回合攻击时，才立即 Retreat；换位后仍不能攻击时，才让低 Prize 的 Dunsparce 留在 Active。
- 不预设 opponent ID 或完整对手构筑，只根据 observation 中实际看到的 Pokémon、Energy、伤害、手牌、Prize 和牌库压力做决策。Land Collapse 只在实际观察到 Great Tusk/攻击压力后按通用进攻与牌库规则处理，不增加预设 matchup 分支。

### 必须维护的状态来源

- 新对局开始时清空本地动作历史和 effect serial；每个己方回合开始保存 Active/Bench Pokémon 实例快照。
- 记录每只 Pokémon 的进场回合、进化回合、本回合是否已使用 Supporter/手填 Energy/Retreat，以及是否已经攻击或主动结束回合。用这些时间节点判断进化时机，不能只看当前 card ID。
- 以固定卡组总量维护 Abra、Kadabra、Alakazam 等资源账本，追踪 Prize、手牌、Active、Bench、弃牌区和牌库；用可见卡牌和 `deckCount` 交叉核对，未知 Prize 保留已知/未知边界。
- Simulator 的 `supporterPlayed`、`energyAttached`、`retreated`、`appearThisTurn` 等字段用于合法性校验；agent 自己的动作历史用于补充发生时点和策略语义。

## 训练观测、评测与本地开发

项目要求 Python 3.11+。常用命令如下：

```bash
./scripts/start_tensorboard.sh
python3 -m evaluation list-opponents
python3 -m evaluation validate work/alakazam_bc_v1
```

TensorBoard 默认读取 `rl_runs/tensorboard`，监听 `127.0.0.1:6006`；可通过 `TENSORBOARD_LOGDIR`、`TENSORBOARD_HOST`、`TENSORBOARD_PORT` 和 `PYTHON` 覆盖。评测、可视化和训练分别使用各自的模块入口，不在 `scripts/` 中增加兼容 wrapper。官方真实对局仍以 Kaggle submission/episode/replay 为主要分析依据。

新增 evaluation opponent 时，先建立自包含标准 package，保证 `main.py` 从同目录读取 deck、`deck.csv` 恰为 60 行且 `cg/` 与基线 hash 一致；然后添加 catalog 条目、补充资产/策略测试，并运行 `python3 -m unittest -v tests.test_evaluation_assets`。不要把 opponent adapter、共享 cg 目录、symlink 或其他仓库的绝对路径带入运行时。

### Replay 可视化工具

- 当用户要求查看、展示、播放或可视化某一局对战时，优先调用统一入口：`python3 -m visualization.replay.cli <replay.json>`。默认生成临时 HTML launcher 并在当前标签页 POST 跳转到 viewer，不创建弹窗；无图形环境或只需要路径时使用 `--no-open`。
- 统一入口兼容 Kaggle 官方 replay、本地顶层包含 `visualize`/`visualize_frames` 的 replay，以及 Kaggle `steps[*][*].visualize` 帧。外部 viewer 使用 `POST` 的 `json` 字段提交完整帧，默认 endpoint 为 `https://ptcgvis.heroz.jp/Visualizer/Replay/0`；需要替换时使用 `--viewer-url`。
- 当前只有 observation/action 的旧本地 trace 不包含引擎可视化帧，不能事后伪造卡面回放。根目录不再提供本地对局 wrapper；需要回放时必须让实际运行器直接导出非空 `visualize` 帧，再把临时 replay 写入 `/tmp`。Kaggle 官方 replay 仍作为长期分析资料。
- 可视化核心实现位于 `visualization/replay/`；如果播放器黑屏，先检查输入是否含非空 `visualize` 帧；如果状态存在但卡面为黑色，再检查外部 viewer 的卡牌资源是否可访问。完整说明见 [`visualization/README.md`](visualization/README.md)。

## 编码风格与命名约定

Python 使用 4 个空格、类型注解和清晰的小函数；遵守 Ruff 的 100 字符行宽（`pyproject.toml`）。变量、函数使用 `snake_case`，常量使用 `UPPER_SNAKE_CASE`，提交目录使用小写下划线。agent 必须只返回模拟器提供的合法选项，并保持策略确定性；Shell 脚本沿用 `set -euo pipefail`。

## 测试指南

测试使用标准库 `unittest`。提交前运行 `python3 -m unittest discover -s tests -p 'test_*.py'`；evaluation 资产改动还必须单独运行 `python3 -m unittest -v tests.test_evaluation_assets`。评测 smoke 和临时验证 report 输出写到仓库根目录 `.tmp/evaluation/`，不要提交生成的 trace 或报告；其他不需要在 VS Code 查看且不是 report 的 simulator 临时数据仍可写到 `/tmp`。策略或引擎改动应优先记录 Kaggle 官方 Episode，并在说明中记录对手、步数和结果。必要时使用 `python3 -m compileall -q evaluation visualization rl_environment train` 检查语法，并用 `bash -n scripts/start_tensorboard.sh` 检查唯一的 shell 入口。

## 提交与 Pull Request

提交信息采用 Git 风格操作前缀加中文说明，例如 `feat: 新增胡地策略`、`fix: 修复合法选项选择`、`docs: 补充实验记录`、`test: 记录本地对局`、`chore: 整理脚本`；一次提交聚焦一个主题。除非用户主动要求，否则不要执行 `git commit`。每次执行 `git commit` 后必须立即执行对应的 `git push`；如果远端、分支或权限导致无法 push，应报告失败，不得把仅完成本地 commit 当作同步完成。PR 应说明动机、改动目录、验证命令和结果，涉及策略时附 replay 或指标；本项目无 UI，截图仅在确有帮助时提供。
- Kaggle 提交默认禁止重复提交。除非用户明确要求进行一次 Kaggle submission，否则不得执行 `kaggle competitions submit`，也不得因为上传失败、状态 pending 或结果不理想而自行重试或追加提交。用户明确要求提交一次时，只允许针对当时完成验收的最终归档执行一次打包和提交；后续再次提交、更新归档或更换说明都必须重新获得用户明确授权。
- 不要提交 Kaggle 凭据、`.env`、下载的官方数据或未经许可的二进制资源，也不要自动上传正式 submission。
