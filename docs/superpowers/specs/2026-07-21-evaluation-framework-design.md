# 评测框架建设设计

## 1. 状态与目标

本设计基于 2026-07-21 的需求讨论和对隔壁 ptcg-agent-kaggle repo 的代码调研。
设计已经确认，下一步进入实现计划阶段。

评测框架的目标是：在当前 repo 内维护一套能够运行标准 submission package 的完整评测系统，
运行真实官方对局，准确统计胜率和策略指标，生成 Markdown/HTML 报告，并为最多三场重点对局
保留完整 JSON 回放。

评测框架不负责自动晋级、回退或选择最终提交卡组。最终策略选择由用户根据整体指标和回放分析决定。

## 2. 非目标

本设计不包括：

- 固定随机 seed 或固定样本集的完全复现；
- 修改官方 Pokémon TCG 规则或官方引擎源码；
- 让某个候选策略实现专用的评测 adapter；
- 把评测结果自动转换成晋级决策；
- 长期保存全部对局的完整 trace；
- 继续依赖隔壁 repo 作为运行时评测入口。

## 3. 调研结论

### 3.1 隔壁 repo 的实际核心资产

隔壁 repo 的 17 个 opponent 并不是由 Report Builder 动态生成的，而是由以下内容提供：

- opponents/registry.py：名称到 opponent 的旧式 Python 模块映射；
- 17 个 opponents/*.py：每个文件同时包含卡组、策略和模块级状态；
- eval/alakazam_replay.py：加载 agent、运行官方引擎、处理先后手交换并保存 trace。

17 个 opponent 为：

~~~text
romanrozen_v9
pilkwang_v2
kokinn_search
penguin_915
crustle_wall
crustle_v1
kiyotah_lucario
kiyotah_dragapult
kiyotah_iono
kiyotah_abomasnow
kacchan_anti_wall
nursrijan_lucario
yakitori_raging_bolt
zoli_dragapult
sue_alakazam
maktha_1084
yanxiaohan
~~~

这些 opponent 模块合计约 9500 行代码。它们主要依赖 cg.api 和 Python 标准库，
卡组大多以内嵌列表形式存在。

### 3.2 运行时依赖

隔壁的 replay runner 通过 --cg-path 使用当前 repo submission 目录中的 cg/，
隔壁 repo 本身并没有一套独立且统一的官方运行时。

当前 repo 的 submission 运行时存在两个 Python binding 版本组：

- alakazam_v1 到 alakazam_v5；
- alakazam_v6 到 alakazam_v8。

两组 Python 文件并不相同。当前 v7_auto_iter 与 v8 的 cg/ 文件一致，
native library 文件也与其他版本基本一致。

隔壁的 agent/yanxiaohan/cg/ 是旧版 Python binding，且不包含完整 native library，
不能作为标准独立 submission runtime 使用。当前旧 evaluator 的加载顺序会先加载候选的
cg，再加载 yanxiaohan，因此后者实际复用了候选的 cg；新框架不再依赖这种隐式行为。

### 3.3 yanxiaohan 特殊依赖

旧的 opponents/yanxiaohan.py 会动态加载外部：

~~~text
agent/yanxiaohan/main.py
agent/yanxiaohan/deck.csv
~~~

实际运行的 main.py 只依赖 cg.api 和自己的 deck.csv。同目录下的 core/、
strategies/ 等辅助代码不属于当前运行路径，不迁移到第一版评测框架。

## 4. 标准 Submission Package

候选卡组和 opponent 使用同一种 package 格式：

~~~text
<package>/
├── main.py
├── deck.csv
└── cg/
~~~

deck.csv 是 60 行卡牌 ID，不包含表头。

main.py 必须提供：

~~~python
def agent(observation: dict) -> list[int]:
    ...
~~~

当 observation.get("select") is None 时，agent 应返回同目录 deck.csv 中的 60 张卡。
评测器不再依赖 module.DECK 或 module.my_deck。

策略代码内部可以暂时保留 DECK 变量，但它必须由 deck.csv 读取，不能继续以内嵌列表作为卡组真相。
所有 package 的文件读取都必须以 Path(__file__).resolve().parent 为基准，不能依赖启动时的 cwd。

### 4.1 Opponent 目录

17 个 opponent 放在：

~~~text
evaluation/opponents/<opponent_name>/
~~~

每个目录都物理包含完整的 cg/。第一版使用当前 v7_auto_iter/v8 的新版 runtime 作为模板，
复制到 17 个目录，并记录文件 hash，防止部分 opponent 被替换成不同版本。

每次实际对局只加载一份 cg。候选和 opponent 的 runtime 必须通过版本/hash 兼容检查，
不能静默混用两份不同的 cg。

### 4.2 Opponent Catalog

旧的 Python opponents/registry.py 改为配置文件：

~~~text
evaluation/configs/opponents.json
~~~

配置只描述 opponent 的名称、package 路径、启用状态和可选标签，不加载 Python 模块，也不暴露策略内部接口。

新增 opponent 时只需添加一个标准 package 和一条 catalog 配置，评测核心无需修改。

## 5. 目标目录结构

~~~text
evaluation/
├── __init__.py
├── cli.py
├── configs/
│   └── opponents.json
├── opponents/
│   ├── romanrozen_v9/
│   │   ├── main.py
│   │   ├── deck.csv
│   │   └── cg/
│   ├── ...
│   └── yanxiaohan/
├── packages/
│   ├── loader.py
│   └── validator.py
├── runtime/
│   └── loader.py
├── runner/
│   ├── batch.py
│   └── worker.py
├── metrics/
├── cases/
├── reporting/
└── schemas/
~~~

tests/ 保持在 repo 根目录，覆盖 package、runner、metrics、case selection 和报告结构。

## 6. Package Loader

通用 loader 负责所有候选和 opponent：

~~~python
@dataclass
class SubmissionPackage:
    name: str
    root: Path
    deck: list[int]
    entrypoint: Path
    cg_manifest: dict
~~~

加载过程：

1. 检查 main.py、deck.csv 和 cg/；
2. 读取并校验 60 张卡；
3. 校验卡牌 ID 存在于官方卡表；
4. 以 package 目录为 cwd 和 import 路径加载 main.py；
5. 确认 agent 可调用；
6. 校验初始 agent 返回值与 deck.csv 一致；
7. 计算 package、deck 和 cg 文件 hash。

loader 不为具体策略提供 adapter。候选和 opponent 使用同一套 loader。

## 7. Runner 与单局 Worker

### 7.1 批量流程

~~~text
加载候选 package
    ↓
加载 opponent catalog
    ↓
预检所有 package 和 cg
    ↓
生成 run manifest
    ↓
逐 opponent、逐局启动 worker
    ↓
读取逐局结果和完整 trace
    ↓
执行指标插件
    ↓
筛选最多三场重点回放
    ↓
生成 Markdown/HTML
    ↓
删除未保留的临时完整 trace
~~~

默认每个 opponent 的对局交替先后手：

~~~text
game 1：候选先手
game 2：候选后手
game 3：候选先手
~~~

### 7.2 单局 Worker

每场对局使用独立 worker 进程，以隔离：

- opponent 的模块级可变状态；
- candidate 的模块级状态；
- Python import cache；
- agent 搜索缓存和随机状态；
- cg 模块状态。

worker 接收候选 package、opponent package、先后手和 trace 配置，运行一局后输出：

- 逐局状态摘要；
- winner 和 candidate 角色；
- steps；
- error/status；
- 完整 observation trace；
- visualize 数据；
- trace 文件路径。

单局 worker 错误不能阻断后续对局。

## 8. Trace 与逐局记录

每场对局首先生成完整 JSON，并写入临时目录：

~~~text
/tmp/evaluation/<run_id>/<game_id>.json
~~~

完整 JSON 可包含每一步 observation、合法 options、action、回合、玩家、错误和可视化帧。

完整 trace 处理后转换为长期保留的精简 GameRecord。未入选的完整 JSON 删除，入选的最多三场复制到：

~~~text
reports/evaluation/<run_id>/traces/
~~~

长期保存：

- 所有对局的一份 games.jsonl；
- 指标聚合 metrics.json；
- 精炼 case cases.jsonl；
- 最多三份完整回放 JSON。

## 9. 指标插件

指标插件只描述对局发生了什么，不执行晋级决策：

~~~python
class MetricPlugin(Protocol):
    metric_id: str

    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        ...

    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        ...
~~~

每个指标必须提供：

- metric ID；
- 适用范围；
- 分子和分母；
- 状态和值；
- trace evidence step；
- 失败阶段或诊断标签。

### 9.1 初版指标

初版实现：

- 胜、负、平、错误、未完成；
- 平均步数和对局长度；
- 第二个己方回合实际使用 Powerful Hand；
- 第二个己方回合 Rare Candy → Alakazam → Attack；
- 击倒后的 ready attacker 接力；
- 空 Bench 时使用 Run Away Draw；
- 低牌库阶段的过牌和资源消耗；
- 非法动作、agent crash 和 engine error。

第二回合攻击指标根据实际攻击动作判断。对于 Powerful Hand，统计实际选择 attackId=1072，
并按先后手将 engine turn 规范化为第二个己方回合；同时报告全样本分母和到达目标回合的分母。

Rare Candy 指标检查完整动作链：

~~~text
第二个己方回合
  → 使用 Rare Candy
  → Abra 进化为 Alakazam
  → Alakazam 位于 Active
  → 实际宣告攻击
~~~

失败阶段分别记录为 rare_candy_not_played、evolution_not_completed、
alakazam_not_active、no_legal_attack 或 attack_not_declared。

## 10. CaseSelector 与回放保留

CaseSelector 独立于指标插件，负责选择值得保留的完整回放。

默认优先级：

1. engine error、agent crash、非法动作；
2. 败局中的明确目标操作失败；
3. 败局中的资源、接力或终局异常；
4. 具有代表性的 opponent 和失败类型；
5. 与已选 case 重复度较低的对局。

每次评测全局最多保留三场，而不是每个 opponent 保留三场。没有满足条件的 case 时可以保留少于三场。

## 11. 报告与长期产物

每次评测输出：

~~~text
reports/evaluation/<run_id>/
├── manifest.json
├── summary.json
├── games.jsonl
├── metrics.json
├── cases.jsonl
├── report.md
├── report.html
└── traces/
~~~

### 11.1 Manifest

Manifest 记录：

- run ID；
- candidate package 路径和 hash；
- opponent package 列表和 hash；
- deck hash；
- cg hash；
- 对局数；
- swap 规则；
- 启用的指标插件；
- trace 保留策略；
- Python、引擎和运行时间信息。

不依赖固定 seed，但保留足够的版本信息判断两次评测是否具有可比性。

### 11.2 Markdown

Markdown 报告包含：

- 总体胜率、完成率和错误数量；
- 17 个 opponent 的 W/L/D 表格；
- 指标及其分子、分母；
- 失败类型分布；
- 最多三场重点 case；
- 可选 control 对比；
- 运行异常说明。

### 11.3 HTML

HTML 与 Markdown 使用同一份 report_data，不重复计算指标。

HTML 初版包含：

- 总体结果卡片；
- 17 个 opponent 的胜率矩阵；
- matchup 横向胜率图；
- 关键指标的分子、分母和达成率；
- 失败类型分布；
- case 卡片、证据 step 和 trace 路径；
- 可选 control 差异展示。

HTML 是独立文件，不依赖 web server 或外部前端框架。

## 12. 错误处理

### 12.1 运行前错误

Package 缺失文件、卡组错误、agent 缺失、cg 不完整或版本不兼容时，评测在预检阶段终止。

### 12.2 单局错误

单局错误包括：

- candidate_error；
- opponent_error；
- engine_error；
- worker_crash；
- unfinished；
- visualization_error。

这些错误写入逐局记录并继续后续对局。

### 12.3 批次错误

输出目录不可写、报告生成失败、清理失败或进程中断时，保留已经生成的 manifest 和摘要，并标记
run_incomplete。清理操作只允许作用于当前 run 自己创建的临时目录。

## 13. CLI

第一版入口：

~~~bash
python3 -m evaluation list-opponents

python3 -m evaluation validate evaluation/opponents/kiyotah_dragapult

python3 -m evaluation run \
  --candidate work/alakazam_v8_current \
  --opponents all \
  --games 30 \
  --output reports/evaluation
~~~

可选参数：

~~~text
--control PACKAGE
--no-visualize
--keep-temp
--max-steps N
~~~

control 只生成对比结果，不参与自动晋级。

## 14. 迁移顺序

1. 实现通用 package loader 和 validator；
2. 从 17 个旧模块提取 deck.csv；
3. 将策略代码迁移为 17 个 main.py；
4. 特殊迁移 yanxiaohan；
5. 复制并校验 17 份新版 cg/；
6. 建立 opponents.json；
7. 实现单局 worker；
8. 实现批量 Runner 和 manifest；
9. 接入初版指标插件；
10. 接入 CaseSelector 和临时 trace 清理；
11. 生成 Markdown/HTML 报告；
12. 保留旧 scripts CLI 兼容入口；
13. 删除对隔壁 repo 的运行时依赖；
14. 更新 README、AGENTS.md 和测试。

迁移过程中不改变 17 个 opponent 的策略逻辑。所有卡组转换必须通过旧嵌入列表与新 deck.csv 的逐项比对验证。

## 15. 验收标准

完成后应满足：

1. 当前 repo 内存在完整 evaluation/ 框架；
2. 17 个 opponent 都是独立的 main.py、deck.csv、cg/ package；
3. 候选和 opponent 使用相同的 package loader；
4. 评测不再依赖隔壁 repo 的代码或路径；
5. 每场对局都能生成可供指标插件分析的完整临时 JSON；
6. 最终最多保留三场完整重点回放；
7. 所有对局保留精简结构化记录和完整指标分母；
8. Markdown 和 HTML 报告来自同一份统计数据；
9. 单局错误不会阻断整批评测；
10. yanxiaohan 不再通过外部 agent adapter 运行；
11. 可以只新增标准 opponent 目录和 catalog 配置来扩展 opponent 池；
    12. 评测框架不做自动晋级决策。
