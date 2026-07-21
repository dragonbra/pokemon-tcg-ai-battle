# Evaluation Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax - [ ] for tracking.

**Goal:** 在当前 repo 内实现一个不依赖隔壁 repo、能够直接运行标准 Submission Package 的完整评测框架：迁移 17 个 opponent，逐局保存临时完整 trace，插件化统计指标，最多保留三场重点回放，并生成 Markdown/HTML 报告。

**Architecture:** evaluation/packages 负责 candidate 和 opponent 的统一加载与预检；evaluation/runner 在每场对局使用独立 worker 进程并只加载一份经过 hash 校验的 cg runtime；evaluation/metrics 只分析事实，evaluation/cases 只选择回放，evaluation/reporting 使用同一份聚合数据生成 Markdown 和独立 HTML。旧的 scripts/alakazam_auto_iter.py 保留为兼容入口，但不再启动隔壁 repo。

**Tech Stack:** Python 3.11+、官方 cg Python binding 和 native library、Python 标准库（argparse、dataclasses、hashlib、importlib、json、multiprocessing/subprocess、html）；不新增第三方运行时依赖，不修改 engine/source。

## Global Constraints

- evaluation/ 是唯一评测运行入口，运行时不得 import 或访问 ptcg-agent-kaggle。
- candidate 和 opponent 必须都使用 main.py、60 行 deck.csv、cg/ 的标准 submission package。
- 17 个 opponent 的策略逻辑保持不变；只把内嵌卡组迁移到 deck.csv，并将 deck.csv 作为卡组真相。
- 17 个 opponent 各自物理复制一份 cg/，禁止 symlink、共享目录或运行时隐式复用另一份 cg。
- 第一版以 submission/alakazam_v7_auto_iter/cg 或 submission/alakazam_v8/cg 的 hash 作为新版 runtime 基线；候选和 opponent 的 cg hash 不兼容时，预检直接失败。
- 每场对局先写完整临时 JSON；长期产物只保留 games.jsonl、聚合指标、精炼 case，以及全局最多三份完整 trace。
- 不实现固定 seed 完全复现，不实现自动晋级/回退/选择。
- 单局 worker 错误必须写入逐局记录并继续后续对局；运行前 package/cg 预检错误必须终止。
- 所有 package 文件读取必须以 Path(__file__).resolve().parent 为基准，不能依赖启动 cwd。
- 不自动执行 git commit 或 git push；每个任务以测试和可复核的工作树结果作为完成条件。

---

## 文件布局与责任边界

第一阶段完成后的新增和修改范围如下。opponent 的 main.py 是从隔壁 repo 对应模块迁移后的独立策略文件，不再通过 registry 或 adapter 间接加载。

~~~text
evaluation/
├── __init__.py
├── __main__.py
├── cli.py
├── configs/
│   └── opponents.json
├── opponents/
│   ├── romanrozen_v9/
│   ├── pilkwang_v2/
│   ├── 其余 15 个已命名目录/
├── packages/
│   ├── __init__.py
│   ├── loader.py
│   └── validator.py
├── runtime/
│   ├── __init__.py
│   └── loader.py
├── runner/
│   ├── __init__.py
│   ├── models.py
│   ├── worker.py
│   └── batch.py
├── traces/
│   ├── __init__.py
│   └── store.py
├── metrics/
│   ├── __init__.py
│   ├── base.py
│   ├── registry.py
│   ├── trace_utils.py
│   ├── outcome.py
│   ├── powerful_hand.py
│   ├── rare_candy.py
│   ├── post_ko_relay.py
│   ├── run_away_draw.py
│   ├── library_pressure.py
│   └── correctness.py
├── cases/
│   ├── __init__.py
│   └── selector.py
├── reporting/
│   ├── __init__.py
│   ├── models.py
│   ├── markdown.py
│   └── html.py
└── schemas/
    ├── __init__.py
    └── json_schema.py

tests/
├── test_evaluation_packages.py
├── test_evaluation_runtime.py
├── test_evaluation_worker.py
├── test_evaluation_metrics.py
├── test_evaluation_cases.py
└── test_evaluation_reporting.py
~~~

## Task 1: 建立 package、runtime 和 JSON schema 契约

**Files:**

- Create: evaluation/__init__.py
- Create: evaluation/__main__.py
- Create: evaluation/packages/__init__.py
- Create: evaluation/packages/loader.py
- Create: evaluation/packages/validator.py
- Create: evaluation/runtime/__init__.py
- Create: evaluation/runtime/loader.py
- Create: evaluation/schemas/__init__.py
- Create: evaluation/schemas/json_schema.py
- Create: tests/test_evaluation_packages.py
- Create: tests/test_evaluation_runtime.py

**Interfaces:**

- Produces SubmissionPackage, PackageValidationError, load_submission_package, validate_submission_package, compute_cg_manifest, assert_cg_compatible, and load_game_api for all later tasks.
- SubmissionPackage fields are name: str, root: Path, deck: list[int], entrypoint: Path, package_hash: str, deck_hash: str, cg_manifest: dict[str, object].
- load_submission_package(root: Path, official_card_ids: set[int], name: str | None = None) -> SubmissionPackage.
- validate_submission_package(package_root: Path, official_card_ids: set[int]) -> SubmissionPackage.
- load_game_api(runtime_root: Path) -> object.

- [ ] **Step 1: 先写 package fixture 和失败测试**

在 tests/test_evaluation_packages.py 中建立临时标准 package，包含 main.py、deck.csv 和最小 cg/__init__.py；测试以下情况：

~~~python
def test_loader_reads_deck_and_hashes_package(tmp_path):
    package_root = make_package(tmp_path, deck=[1] * 60)
    package = load_submission_package(package_root, {1})
    assert package.deck == [1] * 60
    assert package.entrypoint == package_root / "main.py"
    assert package.deck_hash
    assert package.package_hash
    assert package.cg_manifest["file_count"] >= 1

def test_loader_rejects_wrong_deck_size(tmp_path):
    package_root = make_package(tmp_path, deck=[1] * 59)
    with pytest.raises(PackageValidationError, match="60"):
        load_submission_package(package_root, {1})

def test_loader_rejects_unknown_card_id(tmp_path):
    package_root = make_package(tmp_path, deck=[999] * 60)
    with pytest.raises(PackageValidationError, match="999"):
        load_submission_package(package_root, {1})

def test_loader_requires_agent_and_deck_callback(tmp_path):
    package_root = make_package(tmp_path, deck=[1] * 60, main_source="VALUE = 1\n")
    with pytest.raises(PackageValidationError, match="agent"):
        load_submission_package(package_root, {1})
~~~

运行：

~~~bash
python3 -m pytest tests/test_evaluation_packages.py -q
~~~

预期：测试先因模块缺失而失败。

- [ ] **Step 2: 实现统一 package loader**

deck.csv 读取规则固定为逐行读取非空文本、每行一个整数；拒绝表头、空 deck、非整数、非正数、不是 60 张以及不在官方卡表的 ID。loader 使用独立 module name 加载 main.py，确认存在可调用的 agent，并调用一次 agent({"select": None})，确认返回值逐项等于 deck.csv。读取时把 package root 放入 sys.path，加载完成后移除本次临时路径，不把 module 返回给批量 runner。

~~~python
@dataclass(frozen=True)
class SubmissionPackage:
    name: str
    root: Path
    deck: list[int]
    entrypoint: Path
    package_hash: str
    deck_hash: str
    cg_manifest: dict[str, object]

def load_submission_package(
    root: Path,
    official_card_ids: set[int],
    name: str | None = None,
) -> SubmissionPackage:
    raise NotImplementedError
~~~

package_hash 由 main.py、deck.csv 和 cg/ 下相对路径排序后的 sha256 组成；不把 __pycache__ 和 *.pyc 纳入 hash。

- [ ] **Step 3: 实现 cg manifest 和兼容性检查**

runtime/loader.py 对 cg/ 下所有文件按相对路径和 sha256 生成 manifest，并额外记录 Python 文件集合、native 文件集合和总 hash。两份 package 只有 runtime manifest 总 hash 完全一致时才视为兼容；缺少 cg/game.py、cg/api.py 或 native library 时抛出 PackageValidationError。

~~~python
def compute_cg_manifest(cg_root: Path) -> dict[str, object]:
    raise NotImplementedError

def assert_cg_compatible(
    candidate: SubmissionPackage,
    opponent: SubmissionPackage,
) -> None:
    if candidate.cg_manifest["tree_hash"] != opponent.cg_manifest["tree_hash"]:
        raise PackageValidationError("candidate/opponent cg hash mismatch")

def load_game_api(runtime_root: Path) -> object:
    raise NotImplementedError
~~~

load_game_api 只加载 runtime_root/cg/game.py；禁止先加载候选 cg 后再加载 opponent cg。测试覆盖相同目录内容通过、单文件变化失败、缺少 native 文件失败。

- [ ] **Step 4: 固化记录 schema 并运行测试**

schemas/json_schema.py 定义 manifest、GameResult、GameRecord、GameMetric、AggregateMetric、CaseRecord 的必需字段名，至少提供 validate_manifest、validate_game_record、validate_metric_payload、validate_case_record 四个纯函数。

~~~bash
python3 -m pytest tests/test_evaluation_packages.py tests/test_evaluation_runtime.py -q
python3 -m compileall -q evaluation
~~~

预期：package、hash、schema 测试全部通过。

## Task 2: 迁移 opponent pool A（并行任务）

**Files:**

- Create: evaluation/opponents/romanrozen_v9/main.py
- Create: evaluation/opponents/romanrozen_v9/deck.csv
- Copy: evaluation/opponents/romanrozen_v9/cg/ from submission/alakazam_v8/cg
- Create: evaluation/opponents/pilkwang_v2/main.py
- Create: evaluation/opponents/pilkwang_v2/deck.csv
- Copy: evaluation/opponents/pilkwang_v2/cg/ from submission/alakazam_v8/cg
- Create: evaluation/opponents/kokinn_search/main.py
- Create: evaluation/opponents/kokinn_search/deck.csv
- Copy: evaluation/opponents/kokinn_search/cg/ from submission/alakazam_v8/cg
- Create: evaluation/opponents/penguin_915/main.py
- Create: evaluation/opponents/penguin_915/deck.csv
- Copy: evaluation/opponents/penguin_915/cg/ from submission/alakazam_v8/cg
- Create: evaluation/opponents/crustle_wall/main.py
- Create: evaluation/opponents/crustle_wall/deck.csv
- Copy: evaluation/opponents/crustle_wall/cg/ from submission/alakazam_v8/cg
- Create: evaluation/opponents/crustle_v1/main.py
- Create: evaluation/opponents/crustle_v1/deck.csv
- Copy: evaluation/opponents/crustle_v1/cg/ from submission/alakazam_v8/cg
- Create: tests/test_evaluation_opponents_a.py

**Interfaces:**

- Each directory is independently loadable by load_submission_package.
- Each main.py exports agent(observation: dict) -> list[int].
- The only deck source used at runtime is the directory's deck.csv.

- [ ] **Step 1: 从旧模块提取卡组并逐项核对**

从 /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/opponents/{romanrozen_v9,pilkwang_v2,kokinn_search,penguin_915,crustle_wall,crustle_v1}.py 提取旧的 DECK、MY_DECK 或 EMBEDDED_DECK，写成 60 行 deck.csv。用以下脚本化断言逐个比较顺序和计数：

~~~python
assert old_deck == [
    int(line)
    for line in (target_dir / "deck.csv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(old_deck) == 60
~~~

- [ ] **Step 2: 迁移策略入口而不改变策略分支**

复制对应旧模块作为 main.py，仅修改卡组加载和 package 路径：

~~~python
PACKAGE_ROOT = Path(__file__).resolve().parent

def read_deck_csv() -> list[int]:
    return [
        int(line.strip())
        for line in (PACKAGE_ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

DECK = read_deck_csv()
my_deck = DECK
~~~

保留旧 agent 的 to_observation_class、全局状态、fallback、搜索开关和 card constants；删除旧的内嵌 deck list，确保 agent 在 select=None 时返回 list(DECK)。不要添加候选专用 adapter，不要把策略拆成公共基类。

- [ ] **Step 3: 物理复制 cg 并执行分组验证**

复制完整目录而不是链接，确认每个 package 都有 game.py、api.py、sim.py 以及 submission/alakazam_v8/cg 中的 native 文件。

~~~bash
for name in romanrozen_v9 pilkwang_v2 kokinn_search penguin_915 crustle_wall crustle_v1; do
  test -f evaluation/opponents/$name/main.py
  test "$(wc -l < evaluation/opponents/$name/deck.csv)" -eq 60
  test -f evaluation/opponents/$name/cg/game.py
  test -f evaluation/opponents/$name/cg/api.py
done
python3 -m pytest tests/test_evaluation_opponents_a.py -q
~~~

预期：6 个 package 都通过 loader 预检，deck 顺序与旧模块完全一致。

## Task 3: 迁移 opponent pool B（并行任务）

**Files:**

- Create: evaluation/opponents/kiyotah_lucario/main.py, deck.csv, cg/
- Create: evaluation/opponents/kiyotah_dragapult/main.py, deck.csv, cg/
- Create: evaluation/opponents/kiyotah_iono/main.py, deck.csv, cg/
- Create: evaluation/opponents/kiyotah_abomasnow/main.py, deck.csv, cg/
- Create: evaluation/opponents/kacchan_anti_wall/main.py, deck.csv, cg/
- Create: evaluation/opponents/nursrijan_lucario/main.py, deck.csv, cg/
- Create: tests/test_evaluation_opponents_b.py

**Interfaces:**

- Six directories each satisfy the exact same SubmissionPackage contract from Task 1.
- All six cg trees must have the same tree_hash as submission/alakazam_v8/cg.

- [ ] **Step 1: 迁移 deck.csv**

读取旧模块中的 DECK/MY_DECK，保留原始顺序写入 60 行 deck.csv，并运行：

~~~bash
python3 -m pytest tests/test_evaluation_opponents_b.py -q
~~~

测试必须逐个断言 deck_hash、长度、卡牌集合和重复计数；不能只断言目录存在。

- [ ] **Step 2: 迁移 main.py**

对每个模块应用以下统一入口，策略主体保持原样：

~~~python
def agent(obs_dict: dict) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return list(DECK)
    return _existing_policy_body(obs_dict, obs)
~~~

如果旧文件已经有 crash-safe wrapper，则只替换其 deck 分支，不新增第二层 wrapper。所有相对文件读取改为 PACKAGE_ROOT。

- [ ] **Step 3: 复制 cg 并做模块导入检查**

~~~bash
for name in kiyotah_lucario kiyotah_dragapult kiyotah_iono kiyotah_abomasnow kacchan_anti_wall nursrijan_lucario; do
  python3 -m evaluation validate evaluation/opponents/$name
done
~~~

预期：6 个 package 逐一输出 valid，且没有 import 另一个 opponent 或隔壁 repo 的路径。

## Task 4: 迁移 opponent pool C，包括 yanxiaohan（并行任务）

**Files:**

- Create: evaluation/opponents/yakitori_raging_bolt/main.py, deck.csv, cg/
- Create: evaluation/opponents/zoli_dragapult/main.py, deck.csv, cg/
- Create: evaluation/opponents/sue_alakazam/main.py, deck.csv, cg/
- Create: evaluation/opponents/maktha_1084/main.py, deck.csv, cg/
- Create: evaluation/opponents/yanxiaohan/main.py, deck.csv, cg/
- Create: tests/test_evaluation_opponents_c.py

**Interfaces:**

- yanxiaohan 必须是独立标准 package；运行时禁止访问隔壁 repo/agent/yanxiaohan。
- yanxiaohan/main.py 的策略行为来自隔壁 repo 的 agent/yanxiaohan/main.py，deck.csv 来自同目录原始 my_deck。

- [ ] **Step 1: 迁移四个本地 opponent**

对 yakitori_raging_bolt、zoli_dragapult、sue_alakazam、maktha_1084 按 pool A/B 的同样流程提取 deck、替换 deck 真相、复制 cg。测试覆盖所有 package 的初始 agent 返回值与 deck.csv 一致。

- [ ] **Step 2: 消除 yanxiaohan 外部 adapter**

将 /Users/hejinyu/Documents/repos/ptcg-agent-kaggle/agent/yanxiaohan/main.py 复制为 evaluation/opponents/yanxiaohan/main.py，并将其 my_deck 内容写为 evaluation/opponents/yanxiaohan/deck.csv。把相对路径统一改为 PACKAGE_ROOT；保留 agent 的策略代码和 cg.api 依赖，不保留 _load_agent_module、AGENT_DIR、AGENT_MAIN 或 sys.path 指向隔壁 repo 的逻辑。

~~~python
PACKAGE_ROOT = Path(__file__).resolve().parent
DECK = [
    int(line)
    for line in (PACKAGE_ROOT / "deck.csv").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
my_deck = DECK
~~~

- [ ] **Step 3: 复制 cg 并证明没有外部路径**

~~~bash
python3 -m evaluation validate evaluation/opponents/yanxiaohan
rg -n "ptcg-agent-kaggle|AGENT_DIR|AGENT_MAIN|agent/yanxiaohan" evaluation/opponents/yanxiaohan
~~~

预期：validate 通过，rg 无输出。

## Task 5: 建立 opponent catalog 和全池预检

**Files:**

- Create: evaluation/configs/opponents.json
- Create: evaluation/cli.py
- Create: tests/test_evaluation_catalog.py
- Modify: scripts/check_assets.py

**Interfaces:**

- Catalog entry schema：name: str、package: str、enabled: bool、tags: list[str]。
- load_opponent_catalog(path: Path, evaluation_root: Path) -> list[SubmissionPackage]。
- list_enabled_opponents(path: Path) -> list[str]。
- validate_catalog(path: Path, official_card_ids: set[int]) -> list[SubmissionPackage]。

- [ ] **Step 1: 写入唯一 catalog**

配置完整列出以下 17 个名称：romanrozen_v9、pilkwang_v2、kokinn_search、penguin_915、crustle_wall、crustle_v1、kiyotah_lucario、kiyotah_dragapult、kiyotah_iono、kiyotah_abomasnow、kacchan_anti_wall、nursrijan_lucario、yakitori_raging_bolt、zoli_dragapult、sue_alakazam、maktha_1084、yanxiaohan。每条 package 使用相对于 evaluation/ 的路径，默认 enabled=true。

- [ ] **Step 2: 实现 catalog loader 和检查脚本接入**

catalog loader 拒绝重复 name、目录越界路径、未知字段类型和不存在的 package。scripts/check_assets.py 增加 evaluation catalog 检查，但仍保留 submission/ 的既有检查。

~~~bash
python3 scripts/check_assets.py
python3 -m evaluation list-opponents
python3 -m pytest tests/test_evaluation_catalog.py -q
~~~

预期：官方卡数据、现有 submission 和 17 个 opponent 都通过，list-opponents 精确输出 17 个名称。

## Task 6: 实现单局 worker 和 worker 边界测试

**Files:**

- Create: evaluation/runner/__init__.py
- Create: evaluation/runner/models.py
- Create: evaluation/runner/worker.py
- Create: tests/test_evaluation_worker.py

**Interfaces:**

~~~python
@dataclass(frozen=True)
class GameRequest:
    run_id: str
    game_id: str
    candidate: SubmissionPackage
    opponent: SubmissionPackage
    candidate_first: bool
    max_steps: int
    visualize: bool

@dataclass(frozen=True)
class GameResult:
    game_id: str
    opponent: str
    candidate_first: bool
    candidate_physical_index: int
    finished: bool
    winner: int | None
    status: str
    error_kind: str | None
    error: str | None
    steps: int
    trace_path: Path

def run_game(request: GameRequest, trace_path: Path) -> GameResult:
    raise NotImplementedError
~~~

- [ ] **Step 1: 用 fake cg API 写失败测试**

fake battle_start 返回两个合法 deck 和最小 observation；fake battle_select 在三步后返回 result；fake agent 记录 module-level counter。测试验证 candidate_first=True/False 的 winner 归一化、每场调用 battle_finish、agent exception 变成 candidate_error、step limit 变成 unfinished、trace 包含 observation/action/state 摘要。

~~~bash
python3 -m pytest tests/test_evaluation_worker.py -q
~~~

预期：实现前失败，不能因为一侧 agent 异常抛出到 batch 层。

- [ ] **Step 2: 实现独立进程入口**

worker 以 request JSON 路径和 result JSON 路径为命令行参数；每次进程只做以下顺序：读取两个 package manifest、assert_cg_compatible、将 candidate root/cg 放入 sys.path、加载一次 cg.game、在唯一模块名下加载两份 main.py、battle_start、按 candidate_first 交换 decks 和 agent、逐步调用 battle_select、finally 调用 battle_finish、写完整 trace 和 result。

模块加载必须设置 cwd 为各 package root 的临时上下文，加载后还原 cwd；加载完成后不把 main module 暴露给 batch 进程。每个 worker 不能复用主进程 import cache。visualize=True 时在 battle_finish 前调用 visualize_data，并把原始帧放入 trace 的 visualize 字段；visualization 失败只记录 visualization_error。

- [ ] **Step 3: 运行 worker 单元测试**

~~~bash
python3 -m pytest tests/test_evaluation_worker.py -q
python3 -m compileall -q evaluation/runner
~~~

预期：worker 测试全部通过，并可序列化 GameResult。

## Task 7: 实现 trace store 和 batch runner

**Files:**

- Create: evaluation/traces/__init__.py
- Create: evaluation/traces/store.py
- Create: evaluation/runner/batch.py
- Create: tests/test_evaluation_batch.py

**Interfaces:**

~~~python
class TraceStore:
    def __init__(self, temp_root: Path, report_root: Path, retain_limit: int = 3):
        raise NotImplementedError
    def temp_path(self, game_id: str) -> Path:
        raise NotImplementedError
    def write_game_record(self, result: GameResult, trace: dict) -> None:
        raise NotImplementedError
    def retain(self, selected_game_ids: set[str]) -> dict[str, Path]:
        raise NotImplementedError
    def cleanup(self) -> None:
        raise NotImplementedError

def run_batch(config: BatchConfig) -> BatchResult:
    raise NotImplementedError
~~~

~~~python
@dataclass(frozen=True)
class BatchConfig:
    candidate: SubmissionPackage
    opponents: tuple
    games_per_opponent: int
    output_root: Path
    visualize: bool
    max_steps: int
    control: SubmissionPackage | None
    plugin_ids: tuple
    keep_temp: bool

@dataclass(frozen=True)
class BatchResult:
    run_id: str
    manifest: dict[str, object]
    game_records: tuple
    metric_results: dict[str, object]
    case_records: tuple
    report_data: ReportData
~~~

BatchConfig 至少包含 candidate、opponents、games_per_opponent、output_root、visualize、max_steps、control、plugin_ids 和 keep_temp。BatchResult 包含 run_id、manifest、game_records、metric_results、case_records 和 report_data。

- [ ] **Step 1: 写 trace 生命周期测试**

测试创建 5 个临时完整 trace，retain 选择 3 个，断言报告目录仅出现 3 个 traces/*.json；games.jsonl 的 5 条记录都存在；cleanup 不会删除 report_root 之外的路径，重复 cleanup 安全。

- [ ] **Step 2: 实现交替先后手和单局隔离**

逐 opponent、逐 game 生成 game_id；game number 为奇数时 candidate_first=true，偶数时 false。每一局通过 subprocess 启动 worker，不在 batch 进程加载策略 main.py。worker 非零退出仍生成 status=worker_crash 的 GameResult 并继续。

- [ ] **Step 3: 写 manifest 和逐局精简记录**

manifest 记录 run_id、candidate package/deck/cg hash、opponent package/deck/cg hash、games、swap_policy、plugins、python_version、engine_runtime、started_at、finished_at、trace_policy。games.jsonl 每行记录 game_id、opponent、swap、status、winner、steps、error_kind、trace_path 和指标引用，但不复制完整 observation。

~~~bash
python3 -m pytest tests/test_evaluation_batch.py -q
~~~

预期：5 场 fixture 对局全部进入 games.jsonl，只有显式 retain 的 trace 留在 report_root/traces。

## Task 8: 建立 MetricPlugin 接口、trace utilities 和基础指标

**Files:**

- Create: evaluation/metrics/__init__.py
- Create: evaluation/metrics/base.py
- Create: evaluation/metrics/registry.py
- Create: evaluation/metrics/trace_utils.py
- Create: evaluation/metrics/outcome.py
- Create: evaluation/metrics/length.py
- Create: evaluation/metrics/correctness.py
- Create: tests/test_evaluation_metrics.py

**Interfaces:**

~~~python
@dataclass(frozen=True)
class GameContext:
    game_id: str
    candidate_name: str
    opponent_name: str
    candidate_physical_index: int
    candidate_first: bool

@dataclass(frozen=True)
class GameMetric:
    metric_id: str
    status: str
    numerator: int
    denominator: int
    value: float | int | str | None
    evidence: tuple
    diagnostics: tuple

@dataclass(frozen=True)
class AggregateMetric:
    metric_id: str
    numerator: int
    denominator: int
    value: float | int | str | None
    by_opponent: dict[str, dict[str, object]]

class MetricPlugin(Protocol):
    metric_id: str
    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        raise NotImplementedError
    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        raise NotImplementedError
~~~

- [ ] **Step 1: 从 scripts/alakazam_auto_iter.py 提取只读 trace helpers**

迁移 observation/current/players/active/bench/options/action/selected_attack_id、agent turn normalization、field state 和 knockout confirmation 的纯函数到 trace_utils.py。迁移后的函数只接受 JSON dict，不读取文件、不调用 engine、不做 promotion decision。为 option 与 options 两种历史键名提供同一读取函数。

- [ ] **Step 2: 写并实现 outcome/length/correctness 基础指标**

OutcomePlugin 输出 wins、losses、draws、errors、unfinished，分母为所有尝试对局；LengthPlugin 输出 average_steps；CorrectnessPlugin 按 candidate_error、illegal_action、engine_error、worker_crash、unfinished、visualization_error 分类。结果 status 使用 success、unavailable、error 三类之一，evidence 至少含 step、turn、role 和 action。

~~~python
class OutcomePlugin:
    metric_id = "outcome"
    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        raise NotImplementedError
    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        raise NotImplementedError

class LengthPlugin:
    metric_id = "length"
    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        raise NotImplementedError
    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        raise NotImplementedError

class CorrectnessPlugin:
    metric_id = "correctness"
    def analyze_game(self, trace: dict, context: GameContext) -> GameMetric:
        raise NotImplementedError
    def aggregate(self, results: list[GameMetric]) -> AggregateMetric:
        raise NotImplementedError

def default_metric_plugins() -> tuple:
    return (
        OutcomePlugin(),
        LengthPlugin(),
        CorrectnessPlugin(),
    )
~~~

- [ ] **Step 3: 为基础指标写 fixture 测试**

覆盖 win/loss/draw、candidate_error 与 opponent_error 归因、step limit、空 trace、旧 options 键名和完整分母。运行：

~~~bash
python3 -m pytest tests/test_evaluation_metrics.py -q
~~~

预期：基础指标能在没有官方 engine 的 fixture trace 上确定性通过。

## Task 9: 实现关键策略指标插件（可并行拆给三个 sub-agent）

**Files:**

- Create: evaluation/metrics/powerful_hand.py
- Create: evaluation/metrics/rare_candy.py
- Create: evaluation/metrics/post_ko_relay.py
- Create: evaluation/metrics/run_away_draw.py
- Create: evaluation/metrics/library_pressure.py
- Modify: evaluation/metrics/registry.py
- Modify: tests/test_evaluation_metrics.py

**Interfaces:**

- Each plugin implements MetricPlugin and returns evidence that can locate its source step in a retained trace.
- All plugins use the normalized second own turn rather than assuming candidate physical index 0.

- [ ] **Step 1: 实现 Powerful Hand 指标**

第二个己方回合的攻击 step 以 candidate 的 own-turn ordinal=2 判断，实际选中的 option.attackId 必须等于 1072；不能只检查 option 是否存在。输出两个分母：all_games_denominator 和 reached_second_turn_denominator；主 GameMetric 的 denominator 使用全样本，diagnostics 记录未到达回合的原因。

测试必须覆盖 candidate 先手 engine turn 3、后手 engine turn 4、engine turn 5 不计数、存在 1072 但选中另一个 attack 不计数、没有 Powerful Hand 合法 option 的 unavailable。

- [ ] **Step 2: 实现 Rare Candy → Alakazam → Attack 指标**

在同一个 candidate 第二己方回合内按实际 trace 顺序检查：选中 Rare Candy、Active/Bench 的 Abra 变为 Alakazam、Alakazam 位于 Active、随后选中任意合法 attack option。失败阶段只能使用 rare_candy_not_played、evolution_not_completed、alakazam_not_active、no_legal_attack、attack_not_declared 五个值之一，并把对应 step 作为 evidence。

- [ ] **Step 3: 实现 post-KO relay、Run Away Draw 和 library pressure**

post_ko_relay 在击倒确认后检查候选是否仍有 ready attacker，并统计 zero-ready event；RunAwayDraw 检查空 Bench 时实际选择 Run Away Draw 的 action；library_pressure（牌库压力）在 deckCount <= 15、<=10 两个区间记录抽牌/检索消耗、end-turn deckCount、是否发生 deck-out/未完成。所有指标只报告事实，不把阈值转成晋级结论。

- [ ] **Step 4: 注册插件并跑关键 fixture**

registry 使用显式 plugin factory，允许 CLI 传入额外 module path 动态加载一个实现 MetricPlugin 的类；动态插件只能加入统计，不可覆盖核心 output。测试断言所有核心 metric_id 唯一、失败阶段可序列化、evidence 指向存在的 step。

~~~bash
python3 -m pytest tests/test_evaluation_metrics.py -q
~~~

预期：Powerful Hand 和 Rare Candy 指标能区分“有合法选项但没选”和“当回合根本不可用”，并保留完整分母。

## Task 10: 实现 CaseSelector 和最多三场 trace 保留

**Files:**

- Create: evaluation/cases/__init__.py
- Create: evaluation/cases/selector.py
- Create: tests/test_evaluation_cases.py

**Interfaces:**

~~~python
@dataclass(frozen=True)
class CaseCandidate:
    game_id: str
    opponent: str
    status: str
    failure_class: str
    is_loss: bool
    metric_ids: tuple
    evidence_steps: tuple
    trace_path: Path

def select_cases(
    candidates: list[CaseCandidate],
    limit: int = 3,
) -> list[CaseCandidate]:
    raise NotImplementedError
~~~

- [ ] **Step 1: 编写排序 fixture**

构造 engine_error、candidate crash、Rare Candy 失败、post-KO 失败、普通败局和重复 opponent 的 candidates。测试断言 limit<0 抛 ValueError，limit=0 返回空，limit=3 不超过三场；错误优先，明确目标失败次之，资源/接力异常再次之，并优先覆盖不同 opponent 和 failure_class。

- [ ] **Step 2: 实现确定性选择和 evidence 摘要**

排序键固定为 error rank、target failure rank、diagnostic rank、opponent diversity、failure diversity、game_id；同一输入顺序变化不能改变输出。cases.jsonl 每行保存 game_id、opponent、failure_class、metric_ids、evidence、state_summary、actual_action、expected_action、expected_reason、trace_path。

- [ ] **Step 3: 接入 TraceStore**

只把 selected_game_ids 复制到 reports/evaluation/run_id/traces/；未选完整 trace 在当前 run 临时目录内删除。没有符合条件的 case 时允许 traces 为空，不制造虚假 case。

~~~bash
python3 -m pytest tests/test_evaluation_cases.py -q
~~~

预期：所有 case 选择结果最多 3 场且可解释，长期目录不会保留未选完整 trace。

## Task 11: 实现统一 report data、Markdown 和 HTML

**Files:**

- Create: evaluation/reporting/__init__.py
- Create: evaluation/reporting/models.py
- Create: evaluation/reporting/markdown.py
- Create: evaluation/reporting/html.py
- Modify: evaluation/runner/batch.py
- Create: tests/test_evaluation_reporting.py

**Interfaces:**

~~~python
@dataclass(frozen=True)
class ReportData:
    manifest: dict[str, object]
    summary: dict[str, object]
    games: tuple
    metrics: dict[str, object]
    cases: tuple

def render_markdown(data: ReportData) -> str:
    raise NotImplementedError
def render_html(data: ReportData) -> str:
    raise NotImplementedError
def write_report(data: ReportData, output_dir: Path) -> None:
    raise NotImplementedError
~~~

- [ ] **Step 1: 固化 summary 和 metrics 聚合结构**

summary.json 至少包含 total_games、wins、losses、draws、errors、unfinished、completed_games、win_rate、completion_rate、by_opponent；metrics.json 以 metric_id 为 key，保存 numerator、denominator、value、by_opponent、diagnostics。聚合只能运行一次，Markdown/HTML 都接收同一个 ReportData。

- [ ] **Step 2: 实现 Markdown**

report.md 必须包含总体 W/L/D、完成率、错误数；17 个 opponent 的 W/L/D 表；每个指标的 numerator/denominator/value；failure_class 分布；最多三场 case 及 evidence step/trace 路径；control（如果存在）只展示差异，不出现 promotion/reject/revert 决策。

- [ ] **Step 3: 实现无外部依赖 HTML**

report.html 使用内嵌 CSS 和 JSON 数据，不依赖 web server、CDN 或前端框架；包含总体结果卡片、17 opponent 胜率矩阵、横向 matchup 图、指标分子/分母/达成率、失败分布、case 卡片和 trace 路径。所有用户可见文本使用中文或明确的 metric_id。

- [ ] **Step 4: 测试同源和安全输出**

fixture ReportData 渲染 Markdown/HTML，断言两个输出都包含相同的 total_games、metric_id、17 个 opponent 名称和 case；插入含 HTML 特殊字符的 opponent 名称，断言 HTML 转义。运行：

~~~bash
python3 -m pytest tests/test_evaluation_reporting.py -q
~~~

预期：两种报告没有各自重新计算数字，输出可独立打开。

## Task 12: 接入 CLI、control 和旧脚本兼容入口

**Files:**

- Modify: evaluation/__main__.py
- Modify: evaluation/cli.py
- Modify: scripts/alakazam_auto_iter.py
- Modify: evaluation/README.md
- Create: tests/test_evaluation_cli.py

**Interfaces:**

~~~bash
python3 -m evaluation list-opponents
python3 -m evaluation validate evaluation/opponents/kiyotah_dragapult
python3 -m evaluation run \
  --candidate submission/alakazam_v8 \
  --opponents all \
  --games 30 \
  --output reports/evaluation
~~~

- [ ] **Step 1: 实现 list-opponents 和 validate**

list-opponents 读取唯一 catalog；validate 接受 package 目录并打印 name、deck_hash、cg tree_hash、60-card valid。预检输出错误时退出码非 0，不能开始对局。

- [ ] **Step 2: 实现 run 参数和输出**

CLI 支持 --candidate、--opponents all 或逗号列表、--games、--output、--control、--no-visualize、--keep-temp、--max-steps、--metric-module。默认对每个 opponent 交替先后手，并写入 reports/evaluation/run_id/manifest.json、summary.json、games.jsonl、metrics.json、cases.jsonl、report.md、report.html、traces/。

- [ ] **Step 3: 保留旧命令的分析和比较能力**

scripts/alakazam_auto_iter.py 的 analyze 和 compare 子命令继续读取新 report 产物；保留 EvaluationMetrics、AnalysisResult、write_analysis 和 compare_reports 的 import 兼容。删除 build_replay_command 对隔壁 eval/alakazam_replay.py 的调用；legacy run 接受原有参数但把 --agent 的 parent 解析为 candidate package，--evaluator-root 只作为已废弃参数记录，不能被 import 或用于路径查找；--cg-path 若不是 candidate/cg 则返回可解释错误。

~~~python
def legacy_candidate_root(agent_path: Path) -> Path:
    root = agent_path.resolve().parent if agent_path.name == "main.py" else agent_path.resolve()
    if not (root / "main.py").is_file() or not (root / "deck.csv").is_file():
        raise ValueError("legacy --agent must point to a standard submission package")
    return root
~~~

- [ ] **Step 4: 更新 evaluation/README.md 并测试 CLI**

README 说明标准 package、目录 catalog、临时 trace 生命周期、三场保留规则、指标插件接口、报告位置和不支持完全复现。测试使用 fake runner，不启动真实 engine，覆盖 list、validate、参数错误、control 仅出现在 report data。

~~~bash
python3 -m pytest tests/test_evaluation_cli.py -q
python3 -m evaluation list-opponents
~~~

预期：list 输出 17 项，CLI 单元测试通过，旧脚本不再需要隔壁 repo 才能分析或运行。

## Task 13: 更新项目文档和验证入口

**Files:**

- Modify: AGENTS.md
- Modify: evaluation/README.md
- Modify: docs/replay-visualization.md if replay path/format needs clarification
- Modify: scripts/check_assets.py
- Create: tests/test_evaluation_assets.py

**Interfaces:**

- 文档必须明确 evaluation/opponents 下每个 opponent 都是标准 package，而不是 adapter。
- 文档必须明确官方 engine 是运行时来源，evaluation 不修改 engine/source。
- 文档必须明确完整 trace 只在当前 run 临时保留，长期最多三场。

- [ ] **Step 1: 增加 evaluation asset 检查**

tests/test_evaluation_assets.py 读取 catalog，断言精确 17 个 name、每个目录有 main.py/deck.csv/cg、每个 deck 60 行、每个 cg tree_hash 与基线一致、main.py 不含隔壁 repo绝对路径。

~~~bash
python3 -m pytest tests/test_evaluation_assets.py -q
python3 scripts/check_assets.py
~~~

- [ ] **Step 2: 更新 AGENTS.md**

把 evaluation 的 CLI、报告目录、opponent package contract、worker 隔离、trace 保留和新增 opponent 的步骤加入项目结构/测试指南；保留 submission 的 Kaggle 打包规则，不把 evaluation/opponents 当作正式 Kaggle submission 输出目录。

- [ ] **Step 3: 做静态依赖检查**

~~~bash
if rg -n "ptcg-agent-kaggle|opponents\\.registry|eval/alakazam_replay" evaluation scripts/alakazam_auto_iter.py; then
  exit 1
fi
~~~

预期：evaluation 运行代码和兼容脚本不再包含隔壁 repo runtime 依赖；历史说明中若需要引用调研来源，放在 docs 中而不是运行路径。

## Task 14: 集成测试与真实官方 engine smoke test

**Files:**

- Modify: tests/test_evaluation_batch.py
- Modify: tests/test_evaluation_worker.py
- Create: tests/fixtures/evaluation/
- Modify: evaluation/README.md if smoke-test command differs by platform

**Interfaces:**

- Integration fixture must run at least two fake opponents、both candidate_first states、one worker error、one selected case, and verify all report files.
- Real smoke test uses one migrated opponent and one existing candidate package, writes output under /tmp, and never commits generated traces.

- [ ] **Step 1: 运行纯 Python 全套测试**

~~~bash
python3 -m pytest tests/test_evaluation_*.py -q
python3 -m compileall -q evaluation scripts
python3 -m pytest -q
python3 -m pip check
~~~

预期：新增测试和原有测试全部通过；如果仓库没有 pytest 安装，使用项目当前测试 runner 报告缺失，而不是引入运行时依赖。

- [ ] **Step 2: 运行资产、hash 和 diff 检查**

~~~bash
python3 scripts/check_assets.py
python3 -m evaluation validate submission/alakazam_v8
python3 -m evaluation validate evaluation/opponents/yanxiaohan
git diff --check
git status --short
~~~

预期：所有 package valid，17 个 opponent 的 cg hash 一致，没有 whitespace error；临时 trace 不出现在 git status。

- [ ] **Step 3: 运行单 opponent 官方 engine smoke test**

使用兼容的 candidate 和一个迁移后的 opponent，执行 1 场：

~~~bash
python3 -m evaluation run \
  --candidate submission/alakazam_v8 \
  --opponents sue_alakazam \
  --games 1 \
  --output /tmp/pokemon-tcg-evaluation-smoke \
  --no-visualize \
  --max-steps 10000
~~~

验证 manifest、summary、games.jsonl、metrics.json、cases.jsonl、report.md、report.html 均存在；若有 case，traces/ 不超过 3 个；若 engine/native library 在当前平台不可加载，只记录真实错误和阻塞位置，不修改官方 engine。

- [ ] **Step 4: 运行完整 17 opponent 小批次**

在 smoke test 通过后运行每 opponent 2 场交替先后手，输出仍放在 /tmp；检查总对局数为 34，单局错误数量与 games.jsonl 一致，任何一个 opponent 的错误都没有阻断其他 opponent，最终完整 trace 不超过 3 个。

~~~bash
python3 -m evaluation run \
  --candidate submission/alakazam_v8 \
  --opponents all \
  --games 2 \
  --output /tmp/pokemon-tcg-evaluation-full-smoke \
  --no-visualize \
  --max-steps 10000
~~~

## 并行执行安排与合并闸门

实现时使用 subagent-driven-development；每个 subagent 只修改其任务列出的文件，并返回测试命令、实际变更和未解决问题。

~~~text
串行 Gate 1: Task 1 契约和 loader
             ├── 并行 Agent A: Task 2 opponent pool A
             ├── 并行 Agent B: Task 3 opponent pool B
             └── 并行 Agent C: Task 4 opponent pool C/yanxiaohan
串行 Gate 2: Task 5 catalog + 17 package/hash 全池预检
串行 Gate 3: Task 6 worker
             └── Task 7 batch/trace store
串行 Gate 4: Task 8 metric base
             ├── 并行 Metric Agent A: Powerful Hand + Rare Candy
             ├── 并行 Metric Agent B: post-KO + Run Away
             └── 并行 Metric Agent C: library pressure + correctness
串行 Gate 5: Task 10 case selector
             └── Task 11 reporting
串行 Gate 6: Task 12 CLI/legacy
             └── Task 13 docs/assets
串行 Gate 7: Task 14 full verification
~~~

合并每个并行任务后必须先执行对应分组测试，再由主 agent 执行全套测试、hash 检查和官方 engine smoke test。任何 subagent 若修改了不在任务范围内的文件，先停在合并闸门检查 diff，不直接接受。

## 完成定义

只有以下条件全部满足，才可声称实现完成：

- 17 个 opponent 均存在完整、可独立加载的 main.py/deck.csv/cg package，且 deck 顺序和旧模块逐项一致。
- evaluation 的 runner、worker、trace store、metrics、CaseSelector、Markdown/HTML reporting 和 CLI 都通过测试。
- candidate 与 opponent 经过同一个 loader，cg hash 不兼容时在开局前失败。
- 单局完整临时 JSON 可供所有指标读取，最终长期最多三份完整 trace。
- games.jsonl、metrics.json、cases.jsonl、report.md、report.html 使用一致的 run 数据。
- python3 -m evaluation run 不需要隔壁 repo；旧脚本兼容入口也不再启动隔壁 evaluator。
- 真实 engine smoke test 已运行并有实际输出，或明确记录平台 native library 阻塞。
- 未执行用户未要求的 commit/push。
