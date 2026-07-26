# Alakazam Reward-Weighted BC 新机器训练环境设计

## 目标

在全新 WSL2 Ubuntu 22.04 机器上，为 `train/0011_alakazam_sota_reward_weighted_bc` 配置当前真实系统 Python 环境，并验证 CPU 与 NVIDIA GPU 执行链路。真实 0010 基础语料与 0009 reward sidecar 由另一个会话准备；本阶段不启动全量训练，不执行 Kaggle 提交。

## 已确认的机器条件

- Python 3.11.15，`uv` 0.11.32。
- NVIDIA GeForce RTX 5080，约 16 GiB 显存，驱动 596.49。
- `nvidia-smi` 可用；其 CUDA 13.2 表示驱动兼容能力。
- 本机没有 `nvcc` 或 `/usr/local/cuda`，但训练代码不包含自定义 CUDA 扩展，因此不需要先安装完整 CUDA Toolkit。
- 内存约 23 GiB；WSL ext4 文件系统约 949 GiB 可用。
- 当前系统 Python 未安装 PyTorch，也没有激活 venv。

## 环境架构

所有 Python 包直接安装到当前真实解释器 `/usr/local/bin/python` 对应的系统级 site-packages，不创建 `.venv`、Conda 环境或其他额外 Python 环境。安装和验证命令统一使用 `/usr/local/bin/python -m pip` 与 `/usr/local/bin/python`，避免 `pip` 和解释器错配。

这是一项有意接受的机器级状态变更：依赖可能升级、降级或替换该解释器中未来安装的同名包，回滚也不具备删除虚拟环境那样的原子性。安装前必须先记录 `/usr/local/bin/python -m pip freeze`、解释器路径和 site-packages 路径；安装后再次记录版本快照及 `pip check` 结果。若安装失败，基于前后快照识别变化，不自动大范围卸载或回滚未知的既有包。

依赖分两层安装：

1. 向真实系统 Python 安装支持 RTX 5080/Blackwell 的官方 PyTorch CUDA wheel。版本必须来自 PyTorch 官方索引，并通过实际 CUDA kernel 验证，不仅检查 `torch.cuda.is_available()`。
2. 向同一解释器以 editable 模式安装仓库 RL extra：`-e '.[rl]'`。仓库声明 Python `>=3.11`，RL extra 包含 `numpy<2`、`torch>=2.2`、`tensorboard>=2.14`。editable 安装会在系统 site-packages 中写入指向当前仓库的项目元数据，因此移动或删除仓库会使该安装失效。

安装完成后记录 Python、PyTorch、PyTorch CUDA runtime、NumPy、TensorBoard 和 GPU compute capability，作为该机器真实环境的可审计版本快照。仓库没有 lockfile，因此本次记录证明实际可运行组合，不声称还原历史机器的逐包版本。

## 验证流程

### 1. 基础导入

验证项目包、NumPy、PyTorch 和 TensorBoard 均可导入，并确认解释器严格为 `/usr/local/bin/python`。运行 `/usr/local/bin/python -m pip check`，确认共享系统环境中不存在已知依赖冲突。

### 2. GPU 执行验证

必须完成以下真实 GPU 操作：

- `torch.cuda.is_available()` 为真；
- 设备名为 RTX 5080，并读取 compute capability；
- 创建 CUDA tensor 并执行矩阵运算；
- 在 autocast 下执行 forward；
- 执行 backward 和 optimizer step；
- 必要时同步 GPU，以确保异步 kernel 错误不会被漏掉。

若 GPU 可见但出现 `no kernel image`、架构不支持或动态库错误，则视为 PyTorch wheel 不兼容，进入根因调查后更换官方兼容 wheel。不会通过安装任意 CUDA Toolkit 来掩盖 wheel 架构问题。

### 3. 项目 smoke test

运行：

```bash
/usr/local/bin/python -m unittest -v tests.test_alakazam_sota_reward_weighted_bc
/usr/local/bin/python -m compileall -q rl_environment train
```

专用测试使用临时合成数据和小型 CPU 模型，覆盖 dataset join、reward 合成、weighted token cross entropy、三轮训练以及 best checkpoint 生成。它证明代码与依赖链路可运行，但不代表真实 221,289 条训练语料已经复现。

## 数据交接合同

另一个会话负责准备以下 Git-ignored 资产：

```text
rl_runs/dataset/0010-alakazam_sota_model/
  train.jsonl.gz
  validation.jsonl.gz
  test.jsonl.gz
  dataset_audit.json

rl_runs/dataset/0009-reward_weighted_bc/
  reward_sidecar.train.jsonl
  reward_sidecar.validation.jsonl
  reward_sidecar.test.jsonl
  reward_sidecar.jsonl.reward_audit.json
```

资产到位后，构建真实 0011 merged dataset：

```bash
/usr/local/bin/python -m train.alakazam_sota_reward_weighted_bc build-dataset \
  --base-root rl_runs/dataset/0010-alakazam_sota_model \
  --reward-root rl_runs/dataset/0009-reward_weighted_bc \
  --output rl_runs/dataset/0011-alakazam_sota_reward_weighted_bc
```

构建器会按 `(episode_id, player_index, episode_step)` 连接两类记录，并记录上游 audit 哈希、输出数量和输出哈希。

## 正式训练边界

真实训练应基于推荐的 V2 reward setting，但必须复制成新的配置并使用新的运行版本，例如 `V6_<tag>`。不得修改历史 V1–V5 配置或覆盖已有 artifact/checkpoint/TensorBoard 路径。

正式运行形状为：

```bash
/usr/local/bin/python -m train.alakazam_sota_reward_weighted_bc train \
  --dataset-root rl_runs/dataset/0011-alakazam_sota_reward_weighted_bc \
  --output rl_runs/artifact/0011-alakazam_sota_reward_weighted_bc/V6_<tag> \
  --config <new-config-path>
```

新配置保留 CUDA、AMP、batch size 96 和 V2 reward 参数，按新机器路径设置 `storage_path`。若 16 GiB 显存不足，只在新配置中降低 batch size，并以独立新版本记录；不改写历史实验。

## 错误处理与安全边界

- 安装失败：保留完整命令与错误输出，识别索引、Python ABI、CUDA wheel 或网络根因后再调整；由于目标是共享系统环境，不自动卸载安装前已存在的未知包。
- 环境冲突：以安装前后的 `pip freeze` 和 `pip check` 为证据，明确列出被替换的版本及残留冲突；除非用户再次授权，不对无关系统包进行清理。
- GPU 失败：区分驱动不可见、wheel 无 Blackwell kernel、CUDA runtime 动态库和显存不足，不盲目安装系统 CUDA。
- 测试失败：使用系统化调试定位根因，不修改测试以迁就环境。
- 数据缺失：明确报告为外部资产阻塞，不使用合成 smoke 结果替代真实训练结论。
- 不安装或要求 Anthropic/Claude SDK、API key 或模型；训练代码无此运行时依赖。
- 不执行 Kaggle 提交、数据上传或其他外部可见操作。

## 验收标准

本阶段完成必须同时满足：

1. 不创建 `.venv`、Conda 或其他额外 Python 环境；所有 pip 包安装到 `/usr/local/bin/python` 对应的真实系统 site-packages。
2. 安装前后均保存依赖版本快照，且最终 `/usr/local/bin/python -m pip check` 通过。
3. 仓库以 editable RL extra 成功安装到该真实解释器。
4. PyTorch 在 RTX 5080 上完成 CUDA tensor、AMP、反向传播和 optimizer step。
5. 专用 reward-weighted BC 单元测试通过。
6. `compileall` 通过。
7. 输出精确环境版本与尚缺真实数据的状态。
8. 未修改历史训练配置、artifact、checkpoint 或提交资产。

全量训练的完成标准另行定义：真实数据到位、0011 dataset audit 通过、使用 V6+ 新配置启动并生成 checkpoint/metrics/TensorBoard；这不属于本阶段。
