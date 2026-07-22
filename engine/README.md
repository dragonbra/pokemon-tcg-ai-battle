# 官方引擎源码与构建边界

这里定义官方 `PTCGEngine.zip` 在本仓库中的职责边界：

- `engine/source/`：由 `ptcg_engine.zip` 解压出的官方 C++ 源码，作为仓库中的版本化构建输入。
- `engine/build/`：从源码构建出的本地动态库，只作 ABI 和规则回归测试。
- `submission/cg/`：提交包实际携带的官方预编译运行时；不要用源码或本地 build 产物自动覆盖它。

只有 `engine/build/` 加入 `.gitignore`。官方源码自带的 README、许可证和 REUSE 元数据必须原样保留；该源码仅限本比赛期间用于参赛和测试，不进入提交包。

旧的一次性解压/构建 wrapper 已从根目录 `scripts/` 移除。需要重新构建时，直接按照
`engine/source/ptcgProgram 22/README.md` 的官方说明操作，并把产物限制在 `engine/build/`；
不要覆盖任何 submission package 中的 `cg/`。
