# 官方 PTCGEngine 源码快照整理记录

## 源码快照

- 原始压缩包：`ptcg_engine.zip`（仓库根目录，Git 忽略）
- 已整理源码：`engine/source/ptcgProgram 22/`（纳入 Git）
- SHA-256：`d1a824b2740e9447acb988cc54f9d5de70af77f2be4f54b891cfa2d74e2a3802`
- 压缩包：47 个条目，约 1.4 MB 展开内容
- 根目录：`ptcgProgram 22/`
- 构建方式：C++20、header-only 风格、Visual Studio 2022 solution、无第三方依赖

主要内容是 `CardImpl.h`（卡牌和效果实现）、`State.h`/`GameProc.h`（状态机和流程）、`Api.h`/`Export.cpp`（外部 API）、`Search.h`（搜索状态管理），以及官方许可证和 REUSE 元数据。源码表包含 1–1267 的官方卡牌 ID，与 `data/official/EN_Card_Data.csv` 的 ID 上界一致。

## 与本框架的边界

| 官方内容 | 本仓库位置 | 用途 |
|---|---|---|
| C++ 源码 | `engine/source/`（版本化） | 构建和理解引擎，不打包 |
| 源码构建产物 | `engine/build/`（本地、忽略） | ABI/规则回归，不直接替换提交运行时 |
| 预编译 `cg` 运行时 | `submission/cg/` | 继续作为提交包中的运行时 |
| agent 和卡组 | `submission/main.py`、`submission/deck.csv` | 我们维护的提交逻辑 |

源码的 C ABI 导出包括 `GameInitialize`、`BattleStart`、`AgentStart`、`GetBattleData`、`Select`、`VisualizeData`、`SearchBegin/Step/End/Release`、`AllCard` 和 `AllAttack`。现有 `submission/cg/sim.py` 已经覆盖这些入口；源码的内部 enum 在 JSON API 层会减去 `None` 的偏移，因此与现有 Python `SelectType`/`SelectContext` 数值保持兼容。

## 官方 discussion 对本次更新的说明

来源：[June 30 Update: Updated Simulation Environment, gameplay increases](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/discussion/716045)

官方说明的是 Data 页 `sample_submission.zip` 和线上 simulation environment 的更新：

- `cg` 新增 macOS `libcg.dylib` 和 Linux ARM64 `libcg-arm64.so`；
- `main.py`、`deck.csv` 和其余 `cg API` 没有变化；
- step limit 提高，死循环应最终因 timeout 失败，不再因步数上限结束为 draw；
- 后续提高每个 submission 的目标对局量，并加入随机对手匹配概率。

所以这次源码包不是新的 agent 接口，也不需要改写 `main.py`。它的合理角色是本地 engine SDK/source snapshot；提交包仍只从 `submission/` 打包。

## 使用流程

在满足 C++20 的 Linux/macOS 工具链环境中：

```bash
./scripts/build_official_engine.sh
PTCG_CG_LIBRARY="$PWD/engine/build/libcg.so" ./scripts/run_local_battle.sh
```

macOS 构建产物对应 `engine/build/libcg.dylib`。当前这台 Ubuntu 20.04 环境的 GCC 9.3 不提供 `<ranges>`，且系统 `libstdc++` 最高为 `GLIBCXX_3.4.28`，因此不能在这里直接编译/加载这个快照；这属于本机工具链限制，不是源码或 Python wrapper 的 API 失败。

在用源码构建产物替换 `submission/cg/` 之前，应至少完成：

1. `GameInitialize`、`AllCard` 和 C ABI 符号检查；
2. 当前 `submission/deck.csv` 的一局 AI vs AI smoke test；
3. observation/select 字段和 replay 结果的回归比较；
4. 再决定是否更新提交目录中的预编译运行时。
