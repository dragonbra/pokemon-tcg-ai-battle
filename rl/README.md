# RL 训练实验框架

这里保存训练研究代码，不是正式 Kaggle submission。正式 submission 仍然必须经过
`work/<name>/`、官方 `cg/` runtime 和 `evaluation` 验收。

## 当前目标

第一版固定 Alakazam 卡组，先验证一个通用的 candidate policy/value 模型：

```text
observation + 当前合法 options
    -> 状态/动作编码
    -> CandidatePolicyValueNet
    -> masked policy logits + value
```

模型不生成 simulator 不允许的动作。候选动作由官方 observation 的 `select.option`
提供，mask 只保留当前合法候选。

## 目录

- `core/model.py`：结构化状态、候选动作打分和 value head。
- `core/losses.py`：合法候选交叉熵和带 padding mask 的 Huber loss。
- `core/reward.py`：可版本化 reward profile 和透明 reward breakdown。
- `core/logging.py`：JSONL 原始日志，TensorBoard 可选镜像。
- `core/checkpoint.py`：保存模型、optimizer、随机状态和实验 metadata。
- `core/promotion.py`：checkpoint 晋级护栏。
- `core/storage.py`：WSL C 盘空间护栏，默认低于 20 GiB 时阻止大任务。
- `ptcg/features.py`：官方 observation 的初版纯 Python 编码器。
- `ptcg/dataset.py`：把本地官方 battle trace 转成合法候选 BC JSONL。
- `ptcg/train_behavior_cloning.py`：从真实 PTCG trace 训练 policy checkpoint。
- `ptcg/build_research_candidate.py`：生成“模型主动作 + 规则效果 handler”的本地评测 candidate。
- `demo/`：没有 simulator 依赖的行为克隆 toy 演示。
- `references/`：Kaggle notebook/report 参考和分析。
- `DESIGN.md`：当前设计、训练路线和评测协议。

## 安装和 toy 演示

项目要求 Python 3.11+。训练依赖是可选的：

```bash
python3.11 -m pip install -e '.[rl]'
PYTHONPATH=. python3.11 -m rl.demo.train_behavior_cloning \
  --output rl/runs/toy_behavior_cloning
```

在本机 RTX 5080 环境中，真实 Python 3.11 已验证使用 CUDA 版 PyTorch：

```bash
python3.11 -m pip install --user --break-system-packages \
  --upgrade 'torch==2.11.0+cu128' \
  --index-url https://download.pytorch.org/whl/cu128
python3.11 -m pip install --user --break-system-packages \
  --upgrade 'numpy<2' 'tensorboard>=2.14'
```

验证 GPU：

```bash
python3.11 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

当前实测输出为 `True NVIDIA GeForce RTX 5080`。`cu128` 是 PyTorch wheel 自带的
CUDA 12.8 runtime，驱动报告 CUDA 13.2，二者可以兼容。

如果官方 `cg` 提示缺少 `GLIBCXX_3.4.29`，准备一个只提供 C++ runtime 的 conda
prefix，并通过环境变量交给本地 battle runner：

```bash
conda create -y -p "$HOME/.local/ptcg-cxx-runtime" \
  -c conda-forge 'libstdcxx-ng>=13'
PTCG_CXX_RUNTIME="$HOME/.local/ptcg-cxx-runtime" \
  ./scripts/run_local_battle.sh --agent0 alakazam_v9 --agent1 alakazam_v9 \
  --output rl/runs/ptcg_bc/battle.json
```

验证日志：

```bash
python3.11 -m unittest -v rl.tests.test_rl_framework
tensorboard --logdir rl/runs/toy_behavior_cloning/tensorboard
```

启动后打开 `http://localhost:6006`，在 Scalars 页面选择：
`train/bc_loss`、`train/action_accuracy` 和 `train/legal_action_rate`。如果希望同时
比较多个实验，可以直接指定整个目录：

```bash
tensorboard --logdir rl/runs
```

`rl/runs/` 专门保存本地训练生成的 metrics、TensorBoard event、checkpoint 和实验数据，
已加入根目录 `.gitignore`，不会进入提交或同步。

## 存储空间护栏

WSL 下优先检查挂载到 `/mnt/c` 的 C 盘，而不是只看 Linux 根盘。dataset builder 和
训练器默认都会在开始前检查至少 20 GiB 可用空间：

```bash
python3.11 -m rl.core.storage
df -h /mnt/c /
```

如果低于警戒线，程序会停止并提示清理旧的 `rl/runs` 实验；不会自动删除文件。

## 评测层级

1–2 局/对手只用于快速探索，不能据此晋级模型。任何 promotion、正式仓库 evaluation
或提交前结论，必须使用全部 17 个 opponent、每个至少 10 局（最低 170 局），并保留
先手/后手轮换；如果胜率差异接近或方差较大，再提高到每个 opponent 20–30 局。

toy 演示只证明 model forward、合法候选 mask、交叉熵行为克隆、JSONL 和 checkpoint
能协同工作。它不是 Pokémon TCG 强度实验。

## 第一条真实数据链

先用现有规则策略作为 teacher 运行本地官方 simulator。输出 trace 只写到 `/tmp`：

```bash
./scripts/run_local_battle.sh \
  --agent0 alakazam_v9 --agent1 alakazam_v9 \
  --output /tmp/ptcg-bc-game.json

PYTHONPATH=. python3.11 -m rl.ptcg.build_bc_dataset \
  /tmp/ptcg-bc-game.json \
  --output rl/runs/ptcg_bc/dataset.jsonl

PYTHONPATH=. python3.11 -m rl.ptcg.train_behavior_cloning \
  rl/runs/ptcg_bc/dataset.jsonl \
  --output rl/runs/ptcg_bc/run
```

这里默认只收集 `select.type=0, context=0` 的主动作，并且只保留 teacher 返回单个
option index 的决策；卡牌效果的多选仍由规则 handler 负责。`select.option` 本身就是
官方 simulator 给出的合法动作集合，所以模型不会被训练成生成一个 simulator 不接受的
全局动作编号。`checkpoints/best_validation.pt` 的 metadata 会保存模型和特征 schema，
可直接交给 `PTCGCandidatePolicy.from_checkpoint()` 恢复推理结构。

终局 reward 过渡实验可以在不改变输入输出 contract 的前提下开启：

```bash
python3.11 -m rl.ptcg.train_behavior_cloning \
  rl/runs/datasets/alakazam_v9_teacher_v4_reward.jsonl \
  --output rl/runs/training/alakazam_terminal_reward \
  --outcome-weight 0.5 --value-loss-weight 0.1 --device auto
```

TensorBoard 会额外记录 `train/policy_loss` 和 `train/value_loss`；如果实战 evaluation
没有改善，这个 profile 不应晋级。

### 本地 checkpoint 评测 candidate

BC checkpoint 可以生成一个只用于研究评测的 hybrid candidate：主动作使用模型，卡牌效果
选择暂时复用 `work/alakazam_v9`。它不会进入正式 Kaggle submission，生成目录同样在
`rl/runs/`：

```bash
python3.11 -m rl.ptcg.build_research_candidate \
  --output rl/runs/research_candidates/alakazam_bc_v2

PTCG_RL_CHECKPOINT="$PWD/rl/runs/training/alakazam_bc_v2/checkpoints/best_validation.pt" \
PTCG_RL_CONFIDENCE_THRESHOLD=0.8 \
LD_LIBRARY_PATH="$HOME/.local/ptcg-cxx-runtime/lib" \
LD_PRELOAD="$HOME/.local/ptcg-cxx-runtime/lib/libstdc++.so.6" \
python3.11 -m evaluation run \
  --candidate rl/runs/research_candidates/alakazam_bc_v2 \
  --opponents all --games 2 --no-visualize \
  --metric-profile auto_iteration_v8_setup_relay \
  --output rl/runs/evaluation/alakazam_bc_v2
```

## 训练边界

批量训练可以使用经过验证的快速环境，但官方 `cg` simulator 是规则真相来源。任何
训练模型必须在固定 seed、先后手和 opponent catalog 上经过 `evaluation` 冻结评测，
再考虑复制到 `work/<name>/` 作为可打包策略。
