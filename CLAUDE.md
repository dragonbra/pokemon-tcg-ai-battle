# Repository Guidelines

## 当前 BC + RL 路线

- 项目当前全面采用 BC + RL 作为主要研发路线。规则策略保留为历史资产、分析基线、回归参考和必要的合法性保障，不再作为主要扩展方向。
- 胡地卡组的单 deck、单 expert BC 已经成功验证 full-action imitation 与官方引擎闭环；这不代表跨卡组泛化、rollout collection、reward/value calibration 或 RL fine-tuning 已经完成。
- 当前并行推进两条主线：一是建立可审计的 rollout、reward、value calibration 与 RL fine-tuning 闭环；二是使用相近的 BC 结构分别学习常见强力卡组，形成更强、更多样的 arena opponents，为后续 RL 提供 curriculum 和训练环境。
- 多卡组 BC 默认训练相互隔离的 deck-specific policy。每套卡组必须保持明确的 deck、expert/team policy、dataset manifest、experiment/version 和 candidate package 边界；不得将不同 deck、team 或实现逻辑的动作标签无条件混入同一个 policy。
- 未来若采用跨卡组共享模型，必须显式加入可审计的 deck/source conditioning，并为冲突标签以及按 deck、source 分组的独立评测建立明确合同。离线 exact-action 或 legal-action 指标只能证明模仿质量和动作合同，不能单独证明真实对战强度。
- 新 BC policy 必须先形成自包含 candidate package，经过 package 验证和官方 engine runtime 真实对局评测，并取得用户明确确认后，才可进入 `evaluation/arena/opponents/`。不得因训练完成、离线指标较高或单一 matchup 表现良好而自动晋级。
- 每个正式 RL run 必须记录实际 opponent catalog、可复现的 pool snapshot，或足以重建对手集合的 package 标识和版本。opponent 构成、采样权重或 curriculum 阶段变化必须作为显式实验变量，禁止在未记录的情况下跨不同 opponent 池继续同一逻辑版本或直接比较结果。

## 项目结构与模块组织

### 环境日报 UI 基准

- 新生成的 `docs/environment-daily_kaggle_top100/daily/YYYY-MM-DD.html` 必须默认继承
  `daily/2026-07-26.html` 的整体信息架构、视觉语言和交互组件，包括顶部导航、摘要指标、
  个人筛选、构筑分布、跨日变化、证据边界、静态排名、逐人 exact 60-card deck、卡图和大图预览；
  不得退化为只有标题、输入框和单表格的临时页面。
- `Top 100 全局构筑卡池` 专项以 `daily/2026-07-25.html` 为 UI 基准，必须使用带卡图的
  9 列审计表，展示卡牌资料、使用构筑数、覆盖率、合计投入、使用时均值、中位数、范围和投入分布；
  各牌型构筑必须保留带卡图的典型投入表与个人特例，不得擅自缩减为纯文本列表或卡片网格。
- 历史日报是已发布快照，不得为统一模板而回写。新日报若缺少某类逐局证据，必须保留对应分析入口并明确标注
  “当前快照不可重建”及缺失字段，禁止复制前日数值、伪造矩阵或用 `n=0` 冒充 0% 胜率。
- 每份新日报交付前必须验证 100 个榜单行、100 个逐人详情、exact deck 均为 60 张、卡池卡图组件存在、
  0726 核心 section ID 存在且索引按日期倒序更新。生成实现与合同见
  `data/processed/environment_daily/generate_live_snapshot.py` 和
  `docs/environment-daily_kaggle_top100/README.md`。

- “根据现在的快照构建日报”固定要求调用统一生成入口即时冻结官方 Top 100，以 leaderboard 每行
  `submissionDate` 唯一绑定 submission，并只选择同一冻结点以前该 submission 最新的
  `PUBLIC + COMPLETED` Episode；必须从 Episode 中 submission 自身唯一 player index 读取 exact
  60-card deck。不得复用旧 snapshot、旧 replay 或前一日报的统计冒充当前环境。
- 日报的个人胜率、W-L-D、牌型 match-up 和卡池统计必须由本次冻结 submission 的官方 Episode
  Meta 与 exact deck 计算。正式渲染前必须通过 100/100 leaderboard → submission → Episode →
  player index → replay → deck/hash 身份链审计；任一链不一致时 fail closed，不得发布。
- Kaggle Episode Meta 是最终一致的约 1,000 条滚动窗口；采集必须将多轮查询看到的冻结点前
  Episode 按 ID 做单调并集，并连续两轮全量扫描零新增后才可完成。胜率和 match-up 必须由该
  冻结并集重算，禁止使用仍在漂移的单次端点切片。
- 统一入口为 `python3 -m data.processed.environment_daily.generate_live_snapshot --date YYYY-MM-DD`；
  每次新采集使用独立 `.tmp/environment_daily/<date>/run-*`，complete run 不得继续采集，正式 HTML
  已存在时必须显式 `--overwrite` 才可重建，并自动维护日报索引。

- 仓库不再使用 `work/`；可运行待评估 package 统一放入 `evaluation/arena/candidates/<name>/`。从 `0013` 起，具体训练项目放在根目录 `train/<project_id>/`，项目归档、权威设计文档和正式评测放在 `experiments/<project_id>/`，运行时 dataset、checkpoint、TensorBoard、W&B staging 与版本记录放在 `rl_runs/<project_id>/`。`rl_environment/` 只提供通用训练基础设施；完成选择并需要长期归档的自包含 payload 统一放入 `archive/submission/<project_numbered_name>/`，对应压缩包放入 `archive/submission/dist/`；根目录 `submission/` 已退役，不得再创建新资产。
- `scripts/` 只保留训练观测入口 `start_tensorboard.sh`；常用基建必须放在所属的 Python package（如 `rl_environment/`、`train/<project>/`、`evaluation/`、`visualization/`）中，不再新增一次性或规则策略脚本。
- `visualization/` 提供 replay 可视化核心、外部 viewer launcher、CLI 和使用说明。
- `evaluation/` 是仓库内的评测运行入口：`configs/opponents.json` 固定 catalog，`arena/opponents/<archetype>_<NN>/` 是正式固定对手池，每个对手都是独立标准 package（`main.py`、60 行 `deck.csv`、物理复制的 `cg/`），不是 adapter；正式名称按关键宝可梦组合使用 ASCII `snake_case` 和两位序号，例如 `alakazam_dudunsparce_01`。catalog 同时维护页面显示名和 1–2 张代表宝可梦卡 ID，用于胜率图缩略图。`arena/candidates/<name>/` 只暂存待准入 package，不参与 `--opponents all`，在用户确认收编前保持候选原名。官方 engine runtime 是唯一运行时来源；评测代码不得修改 `engine/source/`，也不得依赖隔壁评测仓库。
- 评测 CLI 使用 `python3 -m evaluation list-opponents`、`validate <package>` 和 `run --candidate <package> --opponents all --output <report-path>`。每次正式“评测”都只对 `evaluation/arena/opponents/` 固定池运行；这里的 `arena/candidates/` 是候选 opponent 准入区，不是 CLI `--candidate` 所指的被评测卡组。自 `0013` 起，正式实验输出必须是 `experiments/<project_id>/evaluation/V<n>_<tag>.html`，并维护同目录的 `index.html` 汇总所有 `V*.html` 的 candidate、run、局数、胜负、error、胜率和完成率；每次新增正式报告后必须自动刷新 index，版本名可点击进入完整报告。manifest、逐局轻量记录、指标和 case 摘要都内嵌在版本报告中，并继续保留真实 `run_id`；同名 `rl_runs/<project_id>/versions/<V<n>_<tag>>/artifact/evaluation.json` 必须反向指向该权威报告。`.tmp/evaluation/<purpose>` 等临时输出仍可使用隔离的 `run-<id>/report.html`。`evaluation/arena/` 不属于 Kaggle 正式 submission 输出目录。
- 每场评测在独立 worker 进程中运行，隔离双方策略的模块级状态、导入缓存和 cg 状态。完整 trace 仅在当前 run 的临时目录保留，结束时默认删除，不再长期复制三份 trace；只有调试时才使用 `--keep-temp`，需要卡面帧时再显式使用 `--visualize`。
- evaluation 支持用 `--workers N` 并行调度多局，但不得因此复用同一局的 engine 或策略进程；指标分析和报告仍须按 catalog/game 固定顺序落盘，并在 manifest 中记录实际并行度。包含 PyTorch、OpenMP、MKL 或 OpenBLAS CPU 推理的候选，并行时必须同时用 `--worker-cpu-threads N` 限制每个 worker 的内部线程，避免进程乘线程造成 CPU 过度订阅。本机 8 核 BC004 的 18×10 实测为：串行 234.41 秒，`--workers 4 --worker-cpu-threads 1` 为 60.29 秒，`--workers 8 --worker-cpu-threads 1` 为 40.23 秒；未限制内部线程的 4-worker 反而耗时 488.41 秒并出现 worker error。因此本机 CPU-only BC evaluation 默认优先使用 `--workers 8 --worker-cpu-threads 1`，其他硬件先把小规模 benchmark report 写到 `.tmp/evaluation/benchmarks/` 校准，不得盲目按逻辑 CPU 数放大 worker。并行只优化吞吐，不构成策略强度证据；不同 run 的随机胜负不可用于验证并行语义一致性。
- 需要 setup/relay 语义指标时使用 `--metric-profile auto_iteration_v8_setup_relay`（兼容 ID，当前 revision 7）生成完整报告；报告按结果护栏、阶段一二回合基础能力、阶段二 Post-KO 接力、阶段三攻击质量和辅助审计分组，同时保留原始 metric payload。该 profile 只定义指标与展示合同，不代表或触发任何自动迭代、规则策略修改、晋级或淘汰流程。
- `data/official/` 是只读卡牌参考数据，`engine/source/` 是官方引擎源码；`engine/build/` 只保存本地构建产物。
- `notes/`、`docs/reports/` 保存事实和研究结论；`experiments/<project_id>/` 保存该项目的 manifest、决策、权威 `DESIGN.html`/`DESIGN.md`、正式评测和项目实验记录；`replays/` 主要保存从 Kaggle 下载的官方 Episode replay/log JSON；本地 simulator 的非报告临时输出写到 `/tmp`，不纳入仓库。
- Agent 为验证、smoke、benchmark 或临时验收生成的 Evaluation report 必须写到仓库根目录 `.tmp/evaluation/<purpose>/`，不得写到系统 `/tmp`；保留 Evaluation 自动创建的 `run_id/`，并在交付时给出仓库内可点击的 `report.html` 路径，方便直接用 VS Code 查看。`.tmp/` 只用于可删除的本机临时产物，必须保持 Git ignore，禁止提交。自 `0013` 起，正式实验评测写入 `experiments/<project_id>/evaluation/V<n>_<tag>.html`，不得用 `.tmp/` 代替可审计的正式版本产物。
- `evaluation/arena/combat_mat/` 是正式 opponents 池的长期全量循环评测资产，入口为 `index.html`；`reports/<package>/<run_id>/report.html` 长期保留每套卡组的完整源报告，`matrix.json` 保存聚合数据，不得放到可随时清理的 `.tmp/`。每次更新正式池后，必须让 catalog 中每个启用 package 作为 candidate，对完整启用 catalog（包含自身）逐项运行至少 10 局，形成有向 N×N 矩阵。页面必须同时展示可按胜率和耗时排序的完整评测表、总体/先攻/后攻胜率、package 与按关键宝可梦归类的 archetype 胜率热力图、两级平均完整回合数热力图、代表卡图、总局数、累计耗时和逐 package 耗时。完整回合数必须按官方 engine 最终 turn 的 `ceil(turn / 2)` 计算，先手与后手玩家阶段共同组成一个回合；禁止用 action selection steps 冒充。

## 官方引擎与评测硬约束

- 任何开发、调试、训练、评测或自动化行为都不得修改官方提供的 `engine/source/` 源代码，包括直接编辑、格式化、自动修复或生成补丁。允许只读分析与构建；构建产物只能写入 `engine/build/` 或仓库外临时目录。
- 所有用于判断 agent 实际能力、比较版本或形成评测结论的结果，都必须来自使用官方 engine runtime 执行的真实模拟对战。静态分析、mock、伪造 trace、单元测试或合法性检查只能作为辅助证据，不能替代真实对局评测。
- 即使用户明确要求修改 `engine/source/`，也必须在执行前停止，明确警告这会偏离官方引擎、破坏评测真实性和结果可比性，并先与用户讨论用途、影响与替代方案；只有再次取得明确确认后才能继续。

## RL 设计文档同步约定

- [`experiments/<project_id>/DESIGN.html`](experiments/<project_id>/DESIGN.html) 和同目录的 `DESIGN.md` 是该 RL 项目模型输入、网络结构、训练目标和项目阶段的权威设计文档，不是一次性说明或历史快照。
- 后续任何改动只要涉及模型输入或 feature schema、模型结构或输出 head、action contract、训练 loss/reward/value/PPO 接口，或者项目从 BC、rollout、value calibration、RL fine-tuning 等阶段推进到新阶段，都必须在同一项工作中同步更新该项目 `experiments/<project_id>/DESIGN.html` 和 `DESIGN.md`。
- 同步内容至少覆盖受影响的字段与张量 shape、模型数据流与参数结构、训练目标、当前/下一阶段、正式工作包与研究 checkpoint 的边界；不得让页面继续展示已经失效的 schema、数字或阶段结论。
- 实现或配置已经变化但 `experiments/<project_id>/DESIGN.html` 或 `DESIGN.md` 尚未同步时，该项工作视为未完成。交付前必须用当前代码、数据 audit、checkpoint metadata 和 run manifest 交叉核对页面内容；不能只根据旧报告手工推断。
- 纯粹的内部重构、文件移动或不改变模型/训练语义的修复不要求制造文档改动；但如果路径发生变化，页面中的事实来源链接也必须保持可用。
- BC 数据默认只允许来自同一个明确的 team/agent policy；manifest、dataset summary 和 run record 必须记录并校验该来源。不得把不同 team、不同实现逻辑的动作标签直接混成一个无条件 BC policy；只有在模型显式加入可审计的 expert/source conditioning、并单独设计冲突标签与分来源评测时，才可以开启多来源训练。

## RL 实验内迭代版本硬约束

- 从 `0013` 起，项目 ID 使用严格的 `0000_ascii_snake_case`，项目根分别为 `train/<project_id>/`、`experiments/<project_id>/` 和 `rl_runs/<project_id>/`。`experiments/<project_id>/` 保存共享 manifest、数据 manifest/audit、决策、命令记录与权威 DESIGN；每次实际训练、校准或策略更新必须在 `rl_runs/<project_id>/versions/V<n>_<tag>/` 新建严格递增子目录，例如 `V1_initial_contract`、`V2_card_token_fix`。`rl_runs/<project_id>/versions/` 下不得直接写入未版本化资产。`rl_runs/artifact|checkpoint|tensorboard|evaluation` 只读保留给 `0001`–`0012` 历史项目。
- 序号从 `V1` 开始，在同一实验内严格单调递增；`tag` 使用能说明本次假设或修复的 ASCII 小写 `snake_case`。不得复用旧序号、覆盖旧目录、向旧 metrics 追加新 run，或因为结果失败而删除旧版本。
- 每个版本的 tracked 训练产物必须写到 `rl_runs/<project_id>/versions/V<n>_<tag>/artifact/`，至少包含独立的 training config、metrics、summary/status；TensorBoard event、checkpoint 和 W&B staging 分别写到同名版本下的 `tensorboard/`、`checkpoint/` 和 `wandb/`。正式评测写入 `experiments/<project_id>/evaluation/V<n>_<tag>.html`，并由 `artifact/evaluation.json` 记录不可变反向链接。四处版本名必须完全一致，禁止直接把 event、checkpoint 或 W&B staging 写在项目根目录。
- 每个训练 epoch 只对 train split 执行一次参数更新 pass，并从这次已有的 teacher-forced logits 在线聚合 `bc/optimization/*` loss、token accuracy 和 teacher exact 等训练诊断；不得为了生成静态 train 快照指标再次完整扫描 train split，也不得在训练 pass 中额外运行 greedy decode。每个 epoch 仍必须用 epoch 结束时的固定模型对完整 validation split 执行同口径 teacher-forced + greedy full-action 评测，并把在线训练指标与 validation 快照指标同时写入 `training_metrics.jsonl` 和 TensorBoard；正式在线记录还必须镜像到 W&B。在线训练指标来自 epoch 内不断更新的参数，只用于优化健康诊断，不得冒充静态 train 指标、validation 指标或策略强度证据。不得用间隔 validation、缺失值、插值或仅有 console progress 代替逐 epoch validation 曲线。
- 正式评测使用 `experiments/<project_id>/evaluation/V<n>_<tag>.html`，并与产生 candidate 的训练版本对应；项目级 `experiments/<project_id>/evaluation/index.html` 是所有版本评测的总览入口。仅评测解析器或 opponent catalog 变化时也要新建版本并在 tag/决策记录中说明变量，不得混写已有报告。正式 HTML 已存在时必须拒绝覆盖并分配新的版本或明确后缀。
- 失败、被中止或确认存在数据/编码缺陷的版本仍然是正式迭代记录：必须保留可恢复产物，在 version status 或 experiment decisions 中记录失败原因、发现证据和下一版本具体改进。后续版本的价值需要能从这些记录和 TensorBoard 曲线中被追溯。
- 启动新 run 前必须先检查目标版本的 `artifact/`、TensorBoard、checkpoint 和 W&B staging 目录均未被使用；若任一路径已有文件，或权威 evaluation HTML 已存在，必须分配下一个 `V<n>_<tag>`，不得依赖 TensorBoard 新建 event 文件来区分逻辑 run。

## W&B 正式训练记录约定

- 后续正式 BC、value calibration 和 RL/PPO 训练默认都必须启用 W&B online 记录，固定使用 private 项目 `dragon_bra/pokemon-tcg-policy-learning`。临时单元测试、smoke 和诊断不得混入正式项目；确需不启用 W&B 的正式 run，必须在该版本 `status.json` 或决策记录中说明原因和本地证据是否完整。
- `training_metrics.jsonl` 始终是 canonical 事实源；同一条训练记录必须按 `training_metrics.jsonl` flush、TensorBoard、W&B 的顺序写入。W&B 缺失、断网或上传失败不能回滚本地指标、停止 checkpoint 保存或把本次训练伪装为成功同步，但必须把 mirror 失败写入版本状态。
- W&B SDK 与 CLI 在本机使用宿主机 `python3` 的用户级环境，不安装到项目 `.venv`；登录和诊断统一使用 `python3 -m wandb ...`。真正执行训练的解释器必须能够 `import wandb`，不得一边用宿主机安装、一边用默认不可见宿主 site-packages 的 `.venv/bin/python` 启动正式 online run。
- 自动 W&B 镜像只允许由 `TrainingLogger` 对 `rl_runs/<project_id>/versions/<V<n>_<tag>>/artifact/` 正式路径启用。一个 repository version 映射到一个稳定 W&B run ID；同一版本的真实断点恢复可以 resume，新训练语义、超参数或策略更新必须分配新的 `V<n>_<tag>` 和 W&B run，禁止复用旧 run 或向旧曲线追加另一项实验。
- BC 使用 `trainer/epoch` 横轴与 `bc/*` namespace，value calibration 使用 `trainer/epoch` 与 `value/*`，PPO 使用 `trainer/update` 与 `ppo/*`，rollout 诊断使用 `env/decisions`/`env/episodes` 与 `rollout/*`。不同阶段的 loss、训练内 rollout 胜率和吞吐不能互相冒充策略强度。
- 每个正式版本的 `status.json`、`training_summary.json` 或等价版本记录必须保留 W&B project、稳定 run ID 或 URL、sync 状态以及失败原因；本地 `rl_runs/<project_id>/versions/<V<n>_<tag>>/wandb/` 只作为可再生 staging 并保持 Git ignore。
- 默认只镜像有限标量、非秘密 config 和 W&B 自动元数据，不上传 checkpoint、dataset、optimizer state、完整 trace、replay、observation、source patch 或其他大文件。跨 BC/RL 的 policy 强度比较仍必须来自相同 official-engine runtime、opponent catalog、seed/先后手合同和 metric profile 下的正式 `eval/*` 结果。

## 宝可梦 TCG 规则学习长期记忆

详细证据见 [`docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md`](docs/rules/pokemon-tcg-par-rulebook-and-v6-rules-2026-07-19.md)，官方规则书为 [Pokémon TCG Rules](https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/par_rulebook_en.pdf)。后续新会话设计策略时，必须同时遵守下面的官方规则和已确认的策略语义。

### 讨论前规则证据流程

- 当与用户深入讨论游戏设计、策略语义、observation/model feature schema、模型结构或输出 head、action contract、BC/RL loss、reward、value 或训练阶段设计时，必须先完整阅读上述详细规则调研文档，再基于其结论展开讨论；回答中须明确区分通用官方规则、当前卡牌文本/engine runtime 事实与项目策略假设，不能把任一层的结论混称为另一层。
- 如果问题依赖官方规则的精确措辞、涉及调研文档未覆盖或可能已经变化的规则/卡牌交互，或该文档与当前 runtime/卡牌数据存在冲突或不足以判定，必须进一步查询上列官方规则书；具体卡牌效果仍以当前官方卡牌数据和 official engine runtime 为准。完成必要核验后，才与用户进行深度设计讨论，并说明采用的证据边界。

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
python3 -m evaluation validate evaluation/arena/candidates/<candidate>
```

TensorBoard 默认读取 `rl_runs` 中的嵌套项目版本目录，监听 `127.0.0.1:6006`；可通过 `TENSORBOARD_LOGDIR`、`TENSORBOARD_HOST`、`TENSORBOARD_PORT` 和 `PYTHON` 覆盖。评测、可视化和训练分别使用各自的模块入口，不在 `scripts/` 中增加兼容 wrapper。官方真实对局仍以 Kaggle submission/episode/replay 为主要分析依据。

新增 evaluation opponent 时，先在 `evaluation/arena/candidates/<name>/` 建立自包含标准 package，保证 `main.py` 从同目录读取 deck、`deck.csv` 恰为 60 行且 `cg/` 与基线 hash 一致；完成独立验证后由用户确认是否准入。只有获得确认才可按关键宝可梦组合分配正式 `<archetype>_<NN>` 名称、迁入 `evaluation/arena/opponents/`、添加 display name 与代表卡 ID、更新 catalog 和资产测试，并运行 `python3 -m unittest -v tests.test_evaluation_assets`。catalog 只能引用 `arena/opponents/`，不得引用 `arena/candidates/`；不要把 opponent adapter、共享 cg 目录、symlink 或其他仓库的绝对路径带入运行时。

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
