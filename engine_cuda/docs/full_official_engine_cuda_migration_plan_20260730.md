# 官方引擎全量 CUDA 迁移与验收计划（2026-07-30）

## 0. 本计划的结论

目标在工程上可行，但不能把官方 C++ 源码原样加上 `__device__` 后直接编译。
正确路线是：

```text
只读官方源码
  -> host extractor（初始化并遍历官方数值表）
  -> private typed IR / rule pack
  -> 共用合同的 CPU POD reference + CUDA VM
  -> device codec + resident BC pool + learner
  -> 全 GPU rollout / PPO
```

这里的 CPU POD 只用于开发期 differential test，不进入生产 rollout，也不是运行时
fallback。最终热路径必须是：

```text
GPU battle state
  -> GPU advance-to-decision
  -> GPU codec
  -> GPU frozen BC / learner inference
  -> GPU action normalize
  -> GPU apply-action
  -> GPU rollout buffer / PPO update
```

CPU 仅负责进程启动、一次性加载、checkpoint、低频指标和显式 debug 抽样。任何未支持语义
都必须返回稳定错误码并终止对应环境，禁止静默切回 CPU。

“完美复刻”应限定为一个冻结的官方引擎版本。可交付的严格定义是：该版本所有可达规则
分支已纳入覆盖矩阵，所有晋级卡组的 seeded CPU/CUDA paired replay 在逐决策状态、选项、
RNG 和终局上零差异。未来官方引擎更新必须作为新 oracle 版本重新晋级，不能自动宣称等价。

截至本计划执行当日，MainSelect 已有 35 个 official CPU/POD 场景和 35 个 CPU POD/CUDA
完整状态场景通过，覆盖 Play、Attach、Evolve、Ability、Discard、Retreat、Attack、End
的代表路径，并已修正 Main context 与 `turnActionCount` 的 step 语义。ABI v6 通用
`State -> OfficialStatePod` bridge 已在 10,000 个真实 setup 后 Main 状态通过，并成为
Main/Attack oracle 的基础状态来源；116 个 continuation ID 已冻结，opcode 47
`MainSelect` 与 opcode 79 `SelectedMain` 已形成 CPU/CUDA action 闭环：47 产出 79，79
消费 action 后由下一次 Main decision 重新产生。全卡牌、全 skill 和
全条件分支仍未迁移，因此该进度不改变下文 M3/M4/M5 的退出条件，也不能据此启动正式
full-CUDA RL。

额外的 100,000-seed setup replay 已比较 428,336 个 setup decisions，逐字段和 RNG
零差异。该结果只证明 setup 子集的 decision contract，不能替代完整战斗阶段的
100,000-decision corpus。

另外，1,100 seeds 的真实 setup-to-terminal end-turn-only replay 已比较 103,400 decisions：
official CPU/CPU POD canonical state 零差异，CPU POD/CUDA raw state 与 status 零差异。该
策略只在 Main 选择 End，覆盖 turn end、Pokemon Checkup、refresh、turn start draw、MT19937
状态和 deck-out terminal；它不是 Play/Attack/card/effect 的全动作 100,000-decision gate。

## 1. 已完成的事实审计

### 1.1 源码和许可边界

比赛数据包位于：

```text
data/simulation/pokemon-tcg-ai-battle.zip
SHA256 4D12DBCC46079A7ABAF9AEB0918EF1E2C116C2D190376776003C30A85FB604B5
```

完整源码快照位于：

```text
engine/source/ptcgProgram 22/
Git commit 3aaeaa92c9eff5272d0c28582a1f64a4cdc2f772
CardImpl.h SHA256 286A51820D36F9B60B5B13C58BC2EDFF352EB4050581DDADF1960F13FD6F21A9
Types.h SHA256 C5A9AD65B1221FC9B6ED09E5F56C6BD7EC325117CC2247592164ABE698BAD0A8
```

它虽然托管在 GitHub，但许可证明确写明它不是 open-source software，只允许比赛期间用于
参赛和测试，禁止再分发，并要求比赛结束后删除。实施时必须遵守以下边界：

- 官方源码始终只读，不修改、格式化或覆盖。
- 官方源码、卡文和可逆还原源码的生成物不进入公开仓库、公开 artifact 或 W&B。
- extractor、通用 VM、测试框架可以版本化；生成的完整 rule pack 放在
  `engine_cuda/generated/private/<oracle-id>/` 并强制 Git ignore。
- private rule pack 的 manifest 只记录 hash、数量、schema 和覆盖结果，不记录卡文。
- 当前私人 GitHub 是否允许保存衍生实现仍以比赛规则为准；不能因为仓库是 private 就跳过
  许可检查。

官方源码已在 Ubuntu 24.04 容器中以 C++20 只读构建成功，因此从源码初始化内部表、再由
host exporter 遍历这些表的方案可实施。

### 1.2 官方实现规模

本次静态审计得到：

| 项目 | 数量/规模 |
|---|---:|
| 卡牌定义 | 1,267 |
| 攻击定义 | 1,556 |
| `EffectType` | 245 |
| `TargetType` | 102 |
| `ConditionType` | 24 |
| `TriggerType` | 21 |
| `EffectSelectType` | 11 |
| `SelectType` | 12 |
| `SelectContext` | 50 |
| `SelectOptionType` | 17 |
| continuation function table | 116 |
| `CardImpl.h` | 13,735 行 |
| `State.h` | 1,782 行 |
| `EffectInstant.h` | 2,048 行 |
| `EffectProc.h` | 1,225 行 |
| `GameProc.h` | 997 行 |

直接 CUDA 编译困难的原因不是卡牌数量本身，而是当前 host 实现包含：

- `std::vector`、`unordered_map`、`unordered_set` 和字符串；
- host 函数指针表和动态 continuation stack；
- 异常、断言和动态容量；
- 递归式/栈式 effect、trigger、KO、Prize、选择恢复流程；
- `std::mt19937`、`std::shuffle` 及严格的 RNG 消耗顺序；
- continual effect 全场刷新和大量按顺序扫描；
- 观察可见性、反面牌、临时 looking 区域和多玩家选择。

因此必须把控制流状态机化、容器定长化、文本/指针 ID 化，而不是逐文件机械改 CUDA 标记。

### 1.3 本机 Docker/CUDA 结果

本机环境：

| 项目 | 结果 |
|---|---|
| GPU | RTX 3060 Laptop, 6,144 MiB, SM 8.6 |
| Docker Desktop | 4.63.0, WSL2 backend |
| Docker Engine | 29.2.1 |
| NVIDIA runtime | 已注册并可用 |
| CUDA image | `nvidia/cuda:13.0.0-devel-ubuntu24.04` |
| host `nvcc` | 未安装，不需要；由容器提供 |

实测命令：

```powershell
docker run --rm --gpus all `
  nvidia/cuda:13.0.0-devel-ubuntu24.04 `
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

容器成功识别 GPU。现有 prototype 在容器中由 `nvcc` 直接编译和运行，结果为：

| 指标 | 本机结果 |
|---|---:|
| environments | 4,096 |
| policies | 12 |
| iterations | 1,000 |
| prototype env-steps/s | 2,567,847 |
| arena allocation | 108.219 MiB |
| routed environments | 4,096 |
| first state error | 0 |

这证明普通 Docker CUDA 编译、执行和本地 paired test 可行。该吞吐只包含 prototype opcode、
codec-shaped buffer 和 routing，不是完整引擎或端到端 PPO 的速度。

### 1.4 OpenMP 与 Compute Sanitizer

宿主 Python 的 `libiomp5md.dll already initialized` 是同一 Windows 进程重复加载 Intel
OpenMP runtime。它可能影响宿主 PyTorch/NumPy 进程，但与 Docker 能否使用 CUDA 无关。
不要用 `KMP_DUPLICATE_LIB_OK=TRUE` 作为正式修复；训练镜像要统一 PyTorch、NumPy/MKL 和
OpenMP 来源。

Compute Sanitizer 在本机 Docker 中仍会遇到：

```text
Failed to initialize WDDM debugger interface.
Please run EnableDebuggerInterface.bat as an administrator.
Device not supported.
```

原因是 Docker Desktop/WSL2 共享 Windows WDDM 驱动，容器不会把它变成原生 Linux/TCC。
这不是少装 CUDA Toolkit、NVIDIA Container Toolkit 或 OpenMP。正式 sanitizer gate 留在
原生 Linux NVIDIA 主机；本机执行编译、普通 CUDA test、canary、assert 和 paired replay。

按项目要求，本里程碑不做 24 小时 soak。后续使用有上限的高步数 stress test；24 小时 soak
只作为将来可选的生产稳定性检查，不阻塞当前 RL 启动。

## 2. 当前 prototype 的边界

现有 `engine_cuda/` 是正确的垂直切片，可以复用：

- 固定布局 `BattleState`；
- one-thread-per-environment 的初始执行模型；
- opcode/rule-pack ABI；
- device routing 和零拷贝 PyTorch binding；
- policy pool adapter；
- memory estimator；
- 官方 setup/RNG 和 status-core 的局部 fixture；
- 明确 error code 和已知问题 quarantine。

但它目前不是完整引擎：

- 主解释器只有少量 smoke opcode；
- 主 `BattleState` 仍用 64-bit xorshift RNG，而不是全量官方 MT19937 状态；
- device codec 只是 PolicyCodecV1-shaped smoke，不是完整 codec；
- options 是全局 action descriptors，不是每个环境真实生成的官方合法选项；
- setup/status-core benchmark 是专门内核，尚未并入通用 VM；
- 未覆盖完整选择、effect、trigger、continual refresh、KO/Prize 和全部终局。

后续必须演进这套 ABI，不能把已有局部 benchmark 解释成全卡牌 parity。

## 3. 目标合同

### 3.1 功能合同

1. 引擎与牌组无关。换牌组只改变 reset 数据和可达规则，不重新编译 CUDA kernel。
2. 支持冻结 oracle 快照中的全部 1,267 张运行时卡牌，而不是只支持当前六套牌。
3. 精确迁移 setup、回合流程、卡牌移动、效果、选择、trigger、持续效果、特殊状态、
   KO/Prize、RNG 和终局。
4. 保持官方 option 的数量、顺序、字段、最小/最大选择数和选择玩家。
5. 保持隐藏信息边界；codec 不能读取当前 actor 在官方 observation 中看不到的数据。
6. 每个环境独立 RNG、stack、scratch、error 和 episode identity。
7. 所有容量溢出、未知 opcode、非法动作和已知差异 fail closed。

### 3.2 训练合同

1. 默认正式 opponent pool 是当前确认的六个纯 BC：
   Pure Lucario、Cynthia、Kangaskhan-Crustle、Marnie Prize Control v4、
   Yushin Alakazam、Dragapult large-data 2-layer。
2. 规则 Lucario、规则 Crustal、搜索/规则胡地、旧 Marnie 概率合并模型不进入正式池。
3. 引擎和 policy router 的容量必须支持至少 10 个 frozen BC 加 1 个 learner 同时常驻；
   容量测试可加载额外归档 BC，但不因此改变六对手的正式采样 catalog。
4. frozen policy 不创建梯度或 optimizer state；只有 learner 可训练。
5. learner 可采用 encoder 冻结、decoder 解冻的参数组，但这属于 policy optimizer 合同，
   不得让 engine 对某个模型结构产生依赖。
6. steady-state decision hot path 内不允许逐步 H2D、D2H、`.cpu()`、`.item()`、JSON、
   Python action list 或 host synchronization。

### 3.3 不接受的捷径

- 只实现六套牌当前能触发的卡牌并宣称“通用引擎”；
- 遇到未知卡牌时中途调用 CPU official engine；
- 用相同最终胜负替代逐步状态/RNG parity；
- 用 `option_equiv` 掩盖 engine option 顺序或语义差异；
- 用当前 2.57M prototype env-steps/s 作为完整引擎加速比；
- 修改冻结 CPU oracle 来让 CUDA 更容易匹配；
- 将 CPU/GPU 神经网络 argmax 数值差异归咎于引擎。

## 4. 目录和产物设计

建议扩展为：

```text
engine_cuda/
  extractor/                    # 我方 host-only extractor 源码
  generated/private/<oracle>/   # 完整 private IR/rule pack，永不提交
  include/ptcg_cuda/
    abi/                        # versioned POD/IR ABI
    cpu/                        # CPU POD reference declarations
    device/                     # CUDA declarations
  src/
    common/                     # host/device 共用纯函数
    cpu/                        # CPU POD interpreter
    device/                     # CUDA kernels/interpreter
    torch/                      # zero-copy custom op
  codecs/                       # device codec registry
  policies/                     # device adapter registry
  tests/
    fixtures/private/           # 受限 paired fixtures，永不提交
    manifests/                  # 可提交的 hash/coverage summary
  tools/                        # extract/capture/compare/profile CLI
  docs/
```

`generated/private/`、private fixture、official source、完整 traces、checkpoint、Nsight 和
sanitizer 大 artifact 都必须由 `.gitignore` 和打包测试双重阻止上传。

## 5. 核心架构

### 5.1 Source extractor

不要正则解析 `CardImpl.h` 作为权威抽取方式。正确方式是编译一个 host-only exporter：

1. include 官方 `All.h`，调用其初始化入口；
2. 遍历已初始化的 Card/Attack/Skill 表；
3. 将指针关系转换为稳定 ID 和 offset；
4. 将字符串条件解析为 card/name-set ID，rule pack 中不保留卡文；
5. 将 effect、target、condition、trigger、energy requirement 和所有 flag 写入 typed IR；
6. 将 116 个 continuation 注册为显式 enum；
7. 为每一行记录来源类型、owner card/skill/attack ID 和稳定 branch ID；
8. 输出 schema-versioned 二进制及一个不含受限内容的 hash manifest。

必须同时生成：

| 表 | 关键字段 |
|---|---|
| `CardMasterRow` | card ID、类型、HP、进化、弱点/抗性、flags、skill/attack offsets |
| `AttackRow` | attack ID、能量、伤害、flags、pre/post effect range |
| `SkillRow` | skill ID、类型、区域、flags、trigger/effect range |
| `EffectRow` | effect type、select contract、values、condition、target、control flags |
| `TargetRow` | player mask、area range、condition range、not/skip flags |
| `TriggerRow` | trigger type、subject target、priority/order metadata |
| `NameSetRow` | 仅数值 ID 集合，不保留卡文 |
| `ContinuationRow` | continuation opcode 和固定参数类型 |

extractor 还要输出 unsupported report。任何官方字段组合没有 IR 表达时，抽取必须失败，
不能丢字段后继续。

### 5.2 Versioned IR/rule pack

rule pack header 至少包含：

```text
magic
schema_version
endianness
official_source_hash
official_binary_hash
extractor_git_hash
card_data_hash
table counts and byte ranges
payload SHA256
known-divergence policy
```

加载时逐项检查 bounds、alignment、referential integrity 和 hash。训练 run 必须记录 rule-pack
ID；不同 rule-pack 不得无记录续训或直接比较。

### 5.3 CPU POD reference

CPU POD 和 CUDA 使用相同的：

- struct layout；
- enum/opcode；
- rule pack；
- RNG 实现；
- option builder；
- target/condition 纯函数；
- continuation frame 格式；
- error code。

CPU POD 的价值是可以逐字段调试并与官方 oracle 比较。它不能调用官方 effect 函数完成一半
逻辑，否则会隐藏 IR 缺陷。完成后它必须是一个独立、定长、无 heap-after-reset 的解释器。

### 5.4 CUDA execution model

第一版沿用 one-thread-per-environment，因为分支复杂且最容易保持 CPU POD 同序执行。后续只在
Nsight 证明扫描成本成为瓶颈时，对 target scan、option emission、continual refresh 使用
warp-per-environment 协作。优化前后都必须保持稳定 env ID 和完全相同的语义顺序。

建议 kernel 边界：

1. `reset_from_device`
2. `advance_to_decision`
3. `encode_and_route`
4. policy adapters / learner inference
5. `normalize_action`
6. `apply_action_and_advance`
7. `write_rollout_and_reset_done`

生产循环不得每一步由 host 读取 ready count。初版可为每个 policy 启动固定容量 cohort，之后
用 CUDA Graphs 或 grouped execution 降低 launch overhead。

## 6. 状态布局与容量

### 6.1 必须显式保存的状态

- 两个玩家的 deck/hand/prize/trash/active/bench/energy/tool；
- 128 个官方 card instance 及 `moveCounter`/attach identity；
- 每卡 turn/next-turn/continual/KO/ability-used 状态；
- stadium、looking、playing、selected/each/check/target/pre-target/KO list；
- delay/temporary/active trigger stacks；
- continuation stack 和 effect stack；
- turn history、action counters、selection contract；
- special conditions、poison/burn；
- full MT19937 words、cursor 和可审计 RNG call counter；
- terminal result、finish reason 和稳定 error payload。

### 6.2 容量确定方法

不要只根据现有六副牌设容量。先在官方 oracle 上进行全卡表构造、随机合法动作和定向 fixture，
记录所有 dynamic container 的 high-water mark，再用规则上界证明并固定容量。候选初值：

| buffer | 候选容量 | 晋级要求 |
|---|---:|---|
| card instances | 128 | 与官方 `allCard` 容量一致 |
| codec entities | 192 | 覆盖 PolicyCodecV1 既有合同 |
| legal options | 128 | oracle high-water + 规则证明 |
| effect/continuation frames | 256 | 全分支 high-water 后固定 |
| delayed/trigger frames | 128 | 全分支 high-water 后固定 |
| target/list entries | 128 | 每个 list 单独 canary/high-water |

任何真实路径超过容量必须返回命名 overflow error，并让 gate 失败；禁止截断。

### 6.3 AoS/SoA 选择

以 AoS per environment 作为 correctness baseline，便于单环境复制和 digest。卡表/rule pack 使用
只读 SoA，以便 coalesced access。若 profile 显示卡实例扫描占主导，再将高频字段拆为 SoA；
这一变更必须重新跑完整 paired corpus。

## 7. RNG 精确性

### 7.1 生产 RNG 合同

官方路径使用 `std::mt19937`，每环境需要 624 个 32-bit word 加 cursor，约 2.44 KiB。
CUDA 不能用 cuRAND 或 prototype xorshift 替代。需要迁移：

- seed 初始化/seed folding；
- MT twist 和 temper；
- coin 的 `rng() % 2` 语义；
- shuffle 的 exact bounded sampling；
- random target selection；
- 每一处 RNG 调用的顺序。

`std::shuffle` 的具体抽样序列受标准库实现影响；只冻结“C++ 标准算法名称”不够。oracle
manifest 必须记录 binary、compiler 和 standard library，CUDA 实现以该冻结 binary 的输出序列
为准。

### 7.2 RNG 验收字段

每个 paired frame 比较：

- 624 个 word 的 hash；
- cursor；
- 累计 raw draws；
- 当前 coin-head count；
- setup/mulligan/shuffle 结果；
- random-select 结果。

出现 state 相同但 RNG cursor 不同也算失败，因为它会在后续随机事件中造成分叉。

## 8. continuation、选择和效果状态机

### 8.1 Continuation VM

把 host 的函数指针栈替换为：

```text
ContinuationFrame {
  opcode;
  arg0/arg1/arg2;
  arg_type;
  call_count;
  called_count;
}
```

116 个 continuation 都要有稳定 opcode 和单元 fixture。VM 一直推进到以下三种边界之一：

- 需要玩家/策略选择；
- terminal；
- named error。

interpreter budget 只防止死循环。预算耗尽必须报告当前 opcode、PC、stack high-water 和 env ID，
不能把未完成状态当作正常 decision。

### 8.2 Selection contract

每个选择点必须保存并比较：

- `selectPlayer`、`selectType`、`selectContext`；
- `selectMin`、`selectMax`；
- options 数量、顺序和全部参数；
- option 的实体 identity / move counter；
- multi-select 唯一性和 cardinality；
- `option_equiv` 仅作为 policy label equivalence，不改变 engine raw option。

重点覆盖 setup active/bench、能量分配、伤害指示物任意移动、效果二选一、trigger 顺序、
Prize、换位、进化/退化、复制攻击、重复选择和 0..N 选择。

### 8.3 Effect family 实施顺序

按语义依赖实施，而不是按卡组实施：

1. zone/card movement、draw、shuffle、attach、evolve/devolve；
2. target/condition 与 option builder；
3. damage/heal/weakness/resistance/energy requirement；
4. status、checkup、coin；
5. KO、Prize、active replacement、simultaneous terminal；
6. instant effects 和 effect control flow；
7. continual refresh 和全场 modifier；
8. trigger collection、priority、order selection、nested depth；
9. delayed effect、copied attack/skill；
10. rare transforms、arbitrary counter movement 和特殊 win effects。

每个 family 先过 official CPU vs CPU POD micro-fixture，再允许进入 CUDA。

## 9. Codec 和 policy pool

### 9.1 Device codec registry

至少实现并冻结：

- PolicyCodecV1；
- `idonly_codec_v1`；
- `marnie_prize_codec_v4`；
- 当前所有仍需加载的 legacy codec；
- action decode、multi-select stop/unique/min/max/equivalence。

先按 `codec_id` 和静态 shape 路由，再写入可复用 cohort buffer，不能给每个 env 永久分配所有
codec 的最大 tensor。

### 9.2 Policy numerical semantics

现有 127,205-decision policy replay 中，CPU/GPU 神经网络有 12 次 raw argmax flip，其中 8 次
跨 `option_equiv`。这属于 BC 浮点边界，不属于 engine 或 codec 错误。

因此分开两个 gate：

- engine parity：给 CPU/CUDA 完全相同 raw action，要求状态/RNG/终局零差异；
- policy promotion：冻结 GPU policy semantics，使用大规模 battle A/B 评估行为风险。

禁止要求 CUDA engine 伪造 CPU argmax 来掩盖神经网络数值差异。

### 9.3 10+ resident BC

正式采样仍使用六个确认对手，但 residency/profile gate 加载至少 10 个 frozen checkpoint 和一个
learner。每个 policy 必须有：

- checkpoint SHA256；
- deck SHA256；
- adapter/codec/version；
- dtype 和 parameter count；
- frozen flag；
- promotion provenance。

同一决策 range 内用 Nsight 证明 engine、codec、全部 policy forward、decode 和 apply action
之间没有 H2D/D2H/host sync。

## 10. PPO 集成

### 10.1 Device rollout

GPU 常驻：observation、legal mask、action、logprob、value、reward、done、opponent ID、seat、
episode ID、RNG/state digest 和 recurrent state。只把 learner seat 的 transition 写入优化 batch；
frozen opponent 的 inference 不进入 learner optimizer。

done 环境通过 device reset queue 异步重置。opponent sampling、seat、seed 和 curriculum stage 在
GPU 或 reset-time 一次性生成，并写入 run manifest。

### 10.2 Learner 参数组

若本轮实验采用 decoder-only RL：

- encoder 所有参数 `requires_grad=False`；
- optimizer 只接收 decoder/value head 明确列出的参数；
- checkpoint 记录 frozen/trainable 参数名和 hash；
- 每次 update 后确认 encoder hash 不变且 decoder 至少一个参数变化；
- frozen BC checkpoint 前后 SHA256 必须完全相同。

### 10.3 Resume 合同

checkpoint 必须包含：

- learner/optimizer/scheduler/scaler；
- learner RNG、env RNG、episode counters；
- rule-pack/oracle/codec hashes；
- opponent catalog 和 sampling weights；
- rollout position；
- stable W&B run ID。

resume 后下一段 rollout digest 必须与不中断控制组一致。

## 11. 分阶段实施计划

### M0：冻结基线和边界

任务：

- 冻结 source、binary、card data、compiler、stdlib、deck、codec 和 checkpoint hash；
- 固化 Docker image digest；
- 将官方源码和 private generated 路径加入泄漏测试；
- 记录 tuned 15-core CPU baseline 和现有 GPU-policy baseline；
- 保留 `KNOWN_DIVERGENCE_660_1207`。

退出条件：manifest 在干净环境可验证；任何 hash 变化会 fail。

### M1：extractor 和 schema inventory

任务：

- 实现 host exporter 和 versioned IR；
- 导出全部 card/attack/skill/effect/target/trigger/continuation；
- 生成 unsupported-field report、branch IDs 和 referential-integrity test；
- 生成 dynamic container high-water instrumentation build。

退出条件：1,267 张运行时卡牌和 1,556 个攻击全部出现；官方字段零静默丢失；IR 重复生成 bitwise
一致；private payload 不被 Git/打包收集。

### M2：CPU POD core + exact RNG/setup

任务：

- 固定 BattleState/zone/card identity；
- 迁移 exact MT19937、shuffle、coin 和 random select；
- 完成 setup/mulligan/first player/Prize；
- 完成 continuation VM 和 basic selection。

退出条件：至少 10,000 seeds 的两玩家完整 deck order、setup decision、mulligan、Prize、RNG
cursor 零差异；无 heap-after-reset。

### M3：CPU POD 全规则

任务：

- 按第 8.3 节完成全部 effect family；
- 建立每 card/attack/skill/branch coverage matrix；
- 用 oracle state clone 生成 rare-branch micro-fixture；
- 完成 full state serializer/digest。

退出条件：CPU official vs CPU POD 至少 100,000 个全动作 battle decisions 零差异；所有语义 branch 为
`covered`、`unreachable-with-proof` 或命名 quarantine，不能是未知状态。

### M4：CUDA VM parity

任务：

- 将同一 POD contract 编译/实现到 CUDA；
- 加入 per-buffer canary、高水位、error payload；
- 合并 setup/status-core 到通用 VM；
- 批量 replay M2/M3 corpus。

退出条件：CPU POD vs CUDA 全 corpus 在 decision-before、action-after、RNG 和 terminal 零差异；
本机 4,096 env stress 无 error；原生 Linux sanitizer 四个 tool 通过。

### M5：全卡池和晋级卡组 paired replay

任务：

- 运行全 branch fixture；
- 对六个晋级牌组进行所有 seat/matchup 的完整 seeded paired battle；
- 使用相同 raw action trace 或确定性合法 action driver，隔离 policy 数值差异；
- 对错误 seed 自动最小化到首个分歧 frame。

退出条件：六牌组完整游戏零 engine/state/RNG/terminal mismatch，零 unsupported event；全卡池
coverage matrix 无未解释空洞。

### M6：device codec + 10+ policy pool

任务：

- 实现每个 codec 的 device parity；
- 实现 route/gather/forward/decode/scatter；
- 加载 10+ frozen BC + learner；
- 冻结 GPU policy semantics 并复用现有 battle A/B 方法。

退出条件：PolicyCodecV1 127,205 corpus tensor element 零差异；每个 legacy codec 有冻结 corpus
并零差异；非法 action 为 0；Nsight hot path 无 host transfer/sync。

### M7：PPO smoke、容量和性能晋级

任务：

- 全 CUDA engine 上做 fresh + resume PPO smoke；
- 验证 decoder-only 参数组（若该实验启用）；
- 实测 local 6 GiB 小 cohort 和 32 GiB 训练配置；
- 用相同硬件、相同牌组/seed/策略比较 tuned CPU pipeline；
- profile 并只优化 top bottleneck。

退出条件：resume digest 连续、frozen hashes 不变、无内存增长；端到端 games/s 至少为 tuned
CPU pipeline 的 3 倍；p99 decision latency 更低；GPU steady-state utilization 达到 profile 预期。

### M8：版本冻结和 RL 启动

任务：

- 冻结 engine/rule/codec/policy catalog manifest；
- 归档小型 machine-readable acceptance summary；
- 将大 trace、Nsight、sanitizer 保留在受限 artifact store；
- 建立 W&B private run 并记录所有 hash。

退出条件：下文所有强制验收项为 PASS，已知 quarantine 不在训练 deck 可达路径中。

## 12. 完整验收方案

### 12.1 每一步 paired frame 比较内容

在选择前比较：

1. turn、phase、actor、first player；
2. select player/type/context/min/max；
3. option 数量、顺序、类型和全部参数；
4. 两玩家所有 zone 的 card ID + instance/move identity + 顺序；
5. damage、attached cards、status、turn/continual flags；
6. effect/continuation/trigger/delay stacks；
7. selected/target/check/KO/looking lists；
8. RNG words hash、cursor、draw count；
9. full hidden-state digest；
10. actor-visible observation digest。

应用完全相同 raw action 后再次比较所有状态。终局额外比较 result、winner、finish reason 和
final RNG state。

### 12.2 强制 gate

| Gate | 最低要求 | 失败定义 |
|---|---|---|
| Extractor | 全表计数、引用和字段覆盖一致 | 丢字段、未知组合、非确定生成 |
| CPU POD | >=100,000 decisions，零差异 | 任一 state/option/RNG/terminal mismatch |
| RNG/setup | >=10,000 seeds，完整 deck/Prize/RNG 零差异 | 仅首手相同但 cursor 不同也失败 |
| PolicyCodecV1 | 复用 127,205 decisions，逐 tensor 零差异 | 任一元素、mask、entity/equiv 不同 |
| Legacy codecs | 每种 codec 独立冻结 corpus | 没语料不能标 PASS |
| 全卡覆盖 | 每 card/attack/skill/branch 有状态 | 仅“卡牌加载成功”不算覆盖 |
| 六牌组 paired | 全 matchup/seat 完整游戏零 engine mismatch | winner 相同但中途不同也失败 |
| Unsupported | 晋级 corpus 为 0 | silent fallback 直接失败 |
| Residency | 10+ frozen BC + 1 learner 同时常驻 | 逐 policy unload/reload 不算常驻 |
| Hot path | H2D=0、D2H=0、host sync=0 | 任一 decision-loop transfer/sync |
| Sanitizer | Linux 上 mem/race/init/sync 全通过 | WDDM 普通运行不算 sanitizer pass |
| PPO smoke | fresh+resume、optimizer/RNG 恢复 | resume 断链或 frozen hash 变化 |
| 性能 | 相同硬件端到端 >=3x tuned CPU | kernel-only speedup 不算 |
| 稳定性 | 有界 stress，无 OOM/增长/error | 当前不要求 24 小时 soak |

### 12.3 六牌组 paired 规模

建议两级：

- 本地/每 PR：六牌组 36 个有向 matchup，每个 16 seeds，固定 seat，快速 fail-fast；
- 正式晋级：36 个有向 matchup，每个至少 1,024 seeds；完整游戏、零差异。

自对局用于覆盖相同规则交互，不计入对手强度。正式报告按 deck pair、seat、seed、first mismatch
branch 聚合。任何 mismatch 都先修正确性，不能用比例阈值放行。

### 12.4 全卡牌覆盖定义

coverage manifest 每张卡至少列出：

```text
card_id
all attack_ids
all skill_ids
pre/post/continual/delayed effects
all target and condition branches
all selection contexts emitted
all trigger types emitted/resolved
fixture IDs
CPU official/POD/CUDA result
quarantine reason, if any
```

有条件分支必须覆盖 true/false；随机分支必须用可控 seed 覆盖两侧；选择分支必须覆盖最小、最大
和允许 0 选。不可通过普通对局自然触发的路径用 cloned-state micro-fixture，不得标记为“应该没
问题”。

### 12.5 计划中的验收 CLI 合同

以下命令是 Terra 实现时应提供的稳定入口；当前 prototype 尚未全部具备：

```bash
# 只读抽取和 schema 校验
python engine_cuda/tools/extract_official_rules.py \
  --source "<private-source>/ptcgProgram 22" \
  --oracle <private-libcg> \
  --out engine_cuda/generated/private/<oracle-id>
python engine_cuda/tools/verify_rule_pack.py --manifest <manifest>

# CPU official -> CPU POD
python engine_cuda/tools/run_paired_replay.py \
  --left official --right cpu-pod --corpus <corpus> --strict

# CPU POD -> CUDA
python engine_cuda/tools/run_paired_replay.py \
  --left cpu-pod --right cuda --corpus <corpus> --strict

# 全卡 coverage 和六牌组完整游戏
python engine_cuda/tools/check_card_branch_coverage.py --require-complete
python engine_cuda/tools/run_promoted_deck_matrix.py \
  --seeds 1024 --both-seats --strict --output <private-report>

# codec
python engine_cuda/tools/replay_device_codec.py \
  --codec policy_codec_v1 --corpus <127205-corpus> --strict
python engine_cuda/tools/replay_all_legacy_codecs.py --strict

# PPO fresh/resume
python engine_cuda/tools/run_full_cuda_ppo_smoke.py --fresh <run-dir>
python engine_cuda/tools/run_full_cuda_ppo_smoke.py --resume <run-dir>
python engine_cuda/tools/verify_full_cuda_ppo_smoke.py <run-dir>
```

原生 Linux server 的 sanitizer/profile：

```bash
compute-sanitizer --tool memcheck   <full-engine-test>
compute-sanitizer --tool racecheck  <full-engine-test>
compute-sanitizer --tool initcheck  <full-engine-test>
compute-sanitizer --tool synccheck  <full-engine-test>

nsys profile --trace=cuda,nvtx,osrt \
  --output <private-report> \
  <full-cuda-rollout-benchmark>
python engine_cuda/tools/analyze_nsys_hot_range.py <report>
```

## 13. 显存预算

### 13.1 Engine-only 估算

当前规划估算按每环境：

| 部分 | 预算 |
|---|---:|
| full fixed state | 119,936 B（ABI v6 实测） |
| largest active codec cohort allocation | 22,224 B/env equivalent |
| external runtime scratch reserve（state 内部 stack 之外） | 8 KiB |
| routing/control | 2 KiB |
| fixed rule + graph/allocator reserve | 96 MiB |
| CUDA device stack | 至少 32 KiB/thread；本机 WDDM 固定保留约 1.47 GB |
| safety factor | 1.20x |

得到：

| environments | engine operational estimate |
|---:|---:|
| 4,096 | 约 2.1792 GiB（含本机 stack reservation） |
| 16,384 | 约 4.2721 GiB（含本机 stack reservation） |

`OfficialStatePod ABI v6` 已固定为 119,936 B，其中包含全量 MT19937、effect/continuation/
trigger stack 和 effect scratch。attack-resume fixture 在 1/2/4 KiB device stack 下失败、
8 KiB 通过；正式 runtime 使用 32 KiB 下限。4,096-env `OfficialDeviceArena` 的数组账面分配
实测为 470.4 MiB，`cudaMemGetInfo` free-memory delta 为 1,868.0 MiB；后者还包含本机约
1.47 GB 的 WDDM stack reservation、CUDA context、module 和 allocator 粒度。上述
operational estimate 额外保留 codec、外部 scratch/control、固定 rule/graph reserve 和
1.2x headroom。stack reservation 随 GPU 硬件 residency 变化，最终仍以每台目标 GPU 的
完整 rollout Torch peak 实测为准。

### 13.2 模型和 learner

现有估算中，10 frozen BC + 1 learner（BF16 frozen、Adam-like learner、复用 inference
workspace，不含长 rollout）为：

| environments | engine + model estimate |
|---:|---:|
| 4,096 | 约 2.5611 GiB（含本机 stack reservation） |
| 16,384 | 约 4.9949 GiB（含本机 stack reservation） |

真实 policy-only server profile 的峰值为约 2.546 GiB allocated / 3.982 GiB reserved，说明
activation、allocator 和实际 adapter 必须以 measurement 覆盖简化估算。

更重要的是，现有 CPU-engine/GPU-policy PPO iteration 曾达到 17.425 GiB allocated / 23.402
GiB reserved。由此可见，最终显存主项很可能是 learner activation 和 rollout retention，而不是
CUDA battle engine。

### 13.3 Rollout 敏感性

| 每环境保留 rollout | 4,096 env 总估算 | 16,384 env 总估算 |
|---:|---:|---:|
| 0 KiB | 约 2.5611 GiB | 约 4.9949 GiB |
| 256 KiB | 约 3.5611 GiB | 约 8.9949 GiB |
| 1,024 KiB | 约 6.5611 GiB | 约 20.9949 GiB |
| 2,048 KiB | 约 10.5611 GiB | 约 36.9949 GiB |

因此不能预先承诺 16,384-env PPO。正确容量流程是 512 -> 1,024 -> 2,048 -> 4,096 逐级实测，
把 `max_memory_allocated`、`max_memory_reserved`、largest free block 和 OOM headroom 写入报告。

### 13.4 设备结论

- 本机 6 GiB：足够 CUDA 编译、全规则 micro-test、小/中 batch paired replay、10+ frozen
  model 的低 cohort residency 和小规模 inference；不作为完整 PPO 容量证明。
- 32 GiB 4080 实验机：足够完整引擎开发、全量 paired replay 和现有规模 PPO smoke；正式
  batch 必须通过逐级容量扫描确定。
- 5090 训练机：性能和容量仍需重新 profile；不能直接沿用 4080 的最佳 batch/graph。

## 14. 性能验收方法

### 14.1 公平 baseline

CPU baseline 使用实际 15-core 配额下已经调优的配置，而不是单 worker。现有有效参考是
8 engine threads + 10 opponent workers、512 concurrent envs，长跑约 17.916 games/s 和
1,216.878 decisions/s。

GPU 比较必须保持：

- 同一硬件；
- 同一 deck pair、seed、seat、action/policy semantics；
- 同样完整游戏数；
- 包含 engine、codec、policy、rollout write 和 reset；
- 排除初始化和 checkpoint I/O，但不能排除真实 policy inference。

### 14.2 必报指标

- games/s、decisions/s、env-steps/s；
- p50/p95/p99 decision latency；
- setup、engine、codec、route、policy、decode、rollout 各阶段时间；
- GPU utilization、SM occupancy、memory bandwidth、kernel launch count；
- H2D/D2H/D2D、host sync 数量；
- peak allocated/reserved VRAM；
- engine error、unsupported、overflow、max-step 数量；
- 每 opponent/seat 的吞吐和对局长度。

正式晋级要求同硬件端到端至少 3x tuned CPU pipeline。达不到时先依据 Nsight 的最大占比
优化，不能为了速度改变 RNG/option/trigger 顺序。

## 15. 已知差异：Ninetales 660 / Amarys 1207

保留稳定标识：

```text
KNOWN_DIVERGENCE_660_1207
```

冻结本地旧 oracle 在 Ninetales 660 的复制攻击借用 Amarys 1207 延迟效果时可能访问错误的
delay skill 并崩溃。处理规则：

1. 不修改或覆盖 `tmp/seeded_cpp_shim_pure_v1/libcg_seeded.so`；
2. CUDA/CPU POD 遇到精确触发路径时返回 quarantine error，不猜测修复语义；
3. 当前训练/晋级 deck 不得包含可达的该组合；
4. 相关 error 不计为 policy 正常败局；
5. 获得官方修复版后创建新 oracle ID；
6. 同时保留 old-crash fixture 和 new-fixed fixture，再决定 CUDA 新版本语义；
7. 历史结果不回写。

这是一项已知 quarantine，不允许扩展成“其他未实现卡牌也先忽略”的通用豁免。

## 16. 风险登记

| 风险 | 后果 | 控制 |
|---|---|---|
| 许可/衍生数据泄漏 | 合规失败 | private output、ignore、package leak test、W&B 禁传 |
| `std::shuffle` 实现差异 | 同 seed 不同牌序 | 冻结 binary/compiler/stdlib，按 oracle 输出实现 |
| continuation/trigger 顺序错 | 后续状态完全分叉 | 显式 frame、逐 stack digest、micro-fixture |
| 定长容量过小 | rare card 截断 | high-water + 规则上界 + fail-closed canary |
| GPU warp divergence | 利用率低 | parity 后 profile，再按 opcode/cohort 分组 |
| policy argmax 数值 flip | battle winner 偶发翻转 | engine/policy gate 分离，冻结 GPU semantics 做 A/B |
| codec 隐藏信息泄漏 | 训练不合法 | actor-visible digest、逐 tensor corpus parity |
| WDDM sanitizer 不可用 | 本机缺少正式内存证明 | 原生 Linux sanitizer gate |
| 只覆盖常见牌组 | 全卡牌声明失真 | card/skill/branch coverage matrix |
| 官方更新 | 历史不可比 | oracle-versioned rule pack，重新晋级 |

## 17. 用户最终验收清单

只有下列项目均有 machine-readable evidence，才可称“完整 CUDA 引擎已完成”：

- [ ] oracle/source/card data/compiler/stdlib/rule pack/checkpoint/deck 全部有 hash；
- [ ] 官方源码和 private rule pack 未进入不允许的 Git/W&B/package；
- [ ] extractor 全表、全字段、全引用校验通过；
- [ ] CPU official vs CPU POD >=100,000 个全动作 battle decisions 零差异（已另行通过
  103,400-decision end-turn-only 子集，但不据此勾选本项）；
- [ ] >=10,000 setup seeds 的完整 deck/Prize/RNG 零差异；
- [ ] CPU POD vs CUDA 全冻结 corpus 零差异；
- [ ] PolicyCodecV1 127,205 rows 逐 tensor 零差异；
- [ ] 每种 legacy codec 有独立冻结 corpus 并零差异；
- [ ] 1,267 张运行时卡牌的 attack/skill/branch coverage matrix 无未知项；
- [ ] 六个晋级牌组 36 个有向 matchup x 1,024 seeds 完整 paired game 零 engine mismatch；
- [ ] 晋级 corpus 中 unsupported/overflow/silent fallback 为 0；
- [ ] `660/1207` 仍为精确 quarantine，未污染普通统计；
- [ ] 10+ frozen BC + 1 learner 同时常驻；
- [ ] Nsight decision hot path H2D=0、D2H=0、host sync=0；
- [ ] 原生 Linux Compute Sanitizer 四个 tool 全通过；
- [ ] full-CUDA PPO fresh/resume smoke 通过，frozen hashes 不变；
- [ ] 若使用 decoder-only RL，encoder hash 不变、decoder 发生更新；
- [ ] 相同硬件端到端 >=3x tuned 15-core CPU baseline；
- [ ] 本机 6 GiB 和目标 32 GiB 都有实测容量报告；
- [ ] W&B 只记录允许的标量/config/hash，不上传受限源码、rule pack、trace 或 checkpoint。

## 18. Go/No-Go 原则

可以分阶段开始 RL，但标签必须准确：

- 仅 M0-M4 通过：叫“CUDA engine development build”，不能跑正式 league RL；
- M5 对六牌组通过但全卡 coverage 未完成：可叫“six-deck promoted subset”，不能叫全卡引擎；
- M5 全卡 coverage、M6、M7 全通过：可以冻结为正式 full-CUDA RL environment；
- 任一新 deck 含未覆盖 branch：先补 fixture/parity，再加入 opponent/learner catalog；
- 出现首个 state/RNG mismatch：停止性能测试，先修语义；
- 只有 policy argmax flip、engine parity 仍为零：按 GPU policy A/B 流程处理，不回退 engine。

按这个定义，最终可以实现纯 CUDA 训练热路径，同时保留足够严格的 CPU 证据链来说明它复刻
的是哪一个官方引擎版本、复刻到了什么覆盖范围，以及任何保留问题在哪里。
