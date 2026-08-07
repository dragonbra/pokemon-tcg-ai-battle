# 官方引擎源码与构建边界

这里定义官方 `PTCGEngine.zip` 在本仓库中的职责边界：

- `engine/source/`：由 `ptcg_engine.zip` 解压出的官方 C++ 源码，作为仓库中的版本化构建输入。
- `engine/build/`：从源码构建出的本地产物，只作 ABI、规则回归和可重复本地评测。
- `submission/cg/`：提交包实际携带的官方预编译运行时；不要用源码或本地 build 产物自动覆盖它。

只有 `engine/build/` 加入 `.gitignore`。官方源码自带的 README、许可证和 REUSE 元数据必须原样保留；该源码仅限本比赛期间用于参赛和测试，不进入提交包。

旧的一次性解压/构建 wrapper 已从根目录 `scripts/` 移除。需要重新构建时，直接按照
`engine/source/ptcgProgram 22/README.md` 的官方说明操作，并把产物限制在 `engine/build/`；
不要覆盖任何 submission package 中的 `cg/`。

## 本地派生产物

`engine/build/0031/full_engine_prototype_export` 是只读卡牌原型表导出程序。它由 0031 的
`tools/full_engine_prototype_export.cpp` 编译，读取官方 `All.h` 中的 Card、Skill 和 Attack 表并
输出 JSON；它不是对战 engine 动态库，也没有改变随机数或对局状态转移。旧 0025 导出产物已
删除。

可重复本地对局使用 `evaluation/runtime/seeded_engine.cpp` 这一仓库外层 translation unit。
它编译时复用官方 `Export.cpp`，但不编辑、格式化或覆盖 `engine/source/`。构建入口
`python3 -m evaluation.runtime.seeded` 会把 official source hash、adapter hash 和编译器身份写入
manifest 校验，并将第一个明确版本原子生成到：

```text
engine/build/seeded_official/0001/libcg.so
engine/build/seeded_official/0001/manifest.json
```

`0001` 是 seeded local runtime 的显式版本号，不是内容 hash；完整 hash 仍保存在 manifest。
这个派生 ABI 暴露独立的 battle seed 与 Search seed，强制 `deviceRand=false`。它主要用于本地
evaluation 固定对局，也供 RL rollout 构造 paired-seat Episodes，不进入 Kaggle package。
Evaluation 的 `run_batch` 和 0034 的 `build_jobs` 都先调用统一构建入口，并把返回的
`engine_library` 绝对路径传给 worker，因此不会静默回退到 package 内的未 seeded 动态库。
