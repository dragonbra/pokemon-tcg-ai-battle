# OFFICIAL CPU SEARCH — PHASE 2
# DECISION BOUNDARY & DETERMINISM AUDIT

本报告只以未修改的官方 CPU engine `engine/source/ptcgProgram 22/` 及直接编译该源码的 runtime probe 为依据。未使用 CUDA engine、RL/inference wrapper，也未修改 production inference 或官方引擎。

## 1. SearchStep stopping rule

源码链为：

```text
Export.cpp:134 SearchStep
  -> Api.h:162 ApiSearchStep
  -> Search.h:166 Search::step
       clone source State
       selected = caller vector
       State::checkPlayerSelect
       State::step
         -> callFunction / functionStack
         -> 直到 terminal 或遇到 selectType != None
         -> 用 options.size() 下调 selectMax，再下调 selectMin
       while !terminal && selectMax == 0:
         selected.clear()
         State::step
       return cloned branch
```

核心证据是 [`Search.h:166-193`](../../engine/source/ptcgProgram%2022/Search.h) 和 [`State.h:1735-1779`](../../engine/source/ptcgProgram%2022/State.h)。精确语义如下：

- `State::step()` 自动执行 `functionStack` 中所有不要求 selection 的工作；遇到 `selectType != None` 才返回，或在 `gameResult != None` 时终止。
- 返回 selection 前，它执行 `selectMax=min(selectMax, options.size())`，再执行 `selectMin=min(selectMin,selectMax)`。
- `Search::step()` 在第一次 `State::step()` 后，只要非 terminal 且 `selectMax==0`，就把空 vector 当作唯一可提交结果继续 `State::step()`。
- 因此 public `SearchStep` 的正常非 terminal 返回边界是 `selectMax>0`；terminal 返回边界是 `State::isFinish()==true`。它不会因为 option 只有一个而继续。
- `selectMin==0` 表示空 vector 合法，selection 是 optional；只要 `selectMax>0`，Search 仍会停止，因为还有非空选择可能性。
- `selectMin>0` 表示至少必须提交该数量的不同 option index。
- `selectMax==0` 表示只允许空 vector；这是 Search 自动穿过的 engine no-op/empty selection boundary。
- `selectMax>0` 表示存在允许提交的非空 vector；不代表一定有战略分支。
- `options.empty()` 时 `State::step()` 会把 min/max 都压到 0，Search 随后自动继续。反向并不成立：代码也允许 options 非空但显式 max 为 0。

`clearSelect()` 只清除 `selectType/options/selected/contextCard/selectDeck`，不重置 min/max/player（[`State.h:448-455`](../../engine/source/ptcgProgram%2022/State.h)）。所以只有在一个真实返回 decision 上解释 min/max/player 才可靠；不能读取 `selectType=None` 时的残留字段。

Terminal 由 `GameResult::{None,Player0Win,Player1Win,Draw}` 表示；`isFinish()` 即 `gameResult!=None`（[`State.h:16-31`](../../engine/source/ptcgProgram%2022/State.h), [`State.h:345-398`](../../engine/source/ptcgProgram%2022/State.h)）。`FinishReason` 为 `Prize0/Deck0/NoActivePokemon/Effect/Other`。Prize/无场上 Pokémon 在 `finishCheck()` 判断，回合开始空牌库在 [`GameProc.h:989-994`](../../engine/source/ptcgProgram%2022/GameProc.h) 判断。Terminal 时 `selectPlayer` 可能是旧值，不再具有 actor 语义。

特殊 decision 结论：

- `selectMax>0` 但非战略选择：存在，例如 singleton Energy payment、只剩 `End` 的 Main、manual coin result。
- 只有一个 option：存在，Retreat probe 已实测。
- confirmation/acknowledgement：没有独立 Ack 类型；`YesNo` 承载 Activate、FirstEffect、Mulligan 和 manual CoinHead。前三者是二选一语义，CoinHead 是外部提供随机结果而非策略。
- UI-oriented selection：没有独立 UI-only primitive；setup、manual coin 等仍通过同一 selection contract。
- 空 options：会被 clamp 为 min=max=0，并由 Search 自动穿过，不作为 public 非 terminal boundary 返回。

## 2. Decision representation

字段定义在 [`State.h:171-181`](../../engine/source/ptcgProgram%2022/State.h)，枚举在 [`ApiType.h:9-163`](../../engine/source/ptcgProgram%2022/ApiType.h)：

- `selectPlayer`：当前必须提交 selection 的 player index；由每个 caller 调用 `setSelect(..., player, min, max)` 写入。
- `selectType`：option payload/handler 的 engine primitive。
- `selectContext`：用途提示；源码自身注明它“并不严格”，必须与 type/option 一起检查，不能单独作为安全判据。
- `selectMin/selectMax`：提交 vector 的允许长度区间。
- `options`：`SelectOption` 数组；payload 可为 number、card position、attached-card index、energy count、attack id、skill serial 等，定义见 [`AddOption.h:11-117`](../../engine/source/ptcgProgram%2022/AddOption.h)。
- `selected`：调用者提交的 option index vector。

`checkPlayerSelect()` 只做三件事：拒绝重复 index（error 6）、拒绝越界 index（error 5）、拒绝长度不在 `[min,max]`（error 4）。它不检查排序，也不 canonicalize。因此 `[0,1]` 和 `[1,0]` 均合法；`SkillOrder` 的 handler 还明确按提交顺序重排 trigger（[`EffectProc.h:970-1003`](../../engine/source/ptcgProgram%2022/EffectProc.h)）。

Public Search JSON 暴露 `type/context/minCount/maxCount/options`（[`ToJson.h:195-243`](../../engine/source/ptcgProgram%2022/ToJson.h)）。`current.yourIndex` 等于该返回 State 的 `selectPlayer`；而且 Search observation 会从该 player 的视角序列化（[`ToJson.h:285-315`](../../engine/source/ptcgProgram%2022/ToJson.h)）。调用方必须自己保留初始 focal player id，不能把每次返回的 `yourIndex` 自动理解为 focal。

## 3. Decision type taxonomy

| Decision type | Actor | Selection rule / payload | Can be forced? | True branch? | Representative example / source |
|---|---|---|---:|---:|---|
| `None` | none | 无 selection；继续 function stack | N/A | no | `State::step`, `State.h:1735` |
| `Main` | turn player | 通常 1/1；Play/Attach/Evolve/Ability/Discard/Retreat/Attack/End | yes | usually | `MainSelect`, `GameProc.h:768-947` |
| `Card` | caller-selected | Card position；用途由 context 区分 setup/switch/zone move/target/damage | yes | yes | `SelectSwitchPokemon`, `SelectProc.h:113-125`; generic effect `EffectProc.h:641-711` |
| `AttachedCard` | effect player/selected actor | ToolCard 或 EnergyCard payload，通常 1..N | yes | yes | `EffectProc.h:520-597` |
| `CardOrAttachedCard` | effect player/selected actor | Card/ToolCard/EnergyCard 混合 payload | yes | yes | `EffectProc.h:598-640` |
| `Energy` | paying player | 每次选择一张 attached Energy，max=1；可能 min=0 | yes | yes | Retreat/effect payment, `SelectProc.h:256-300` |
| `Skill` | active/opposite player per trigger phase | 必须选全部 option；vector 顺序决定 trigger order | no when N>1 | yes | `ResolveTriggerStack`, `EffectProc.h:982-1003` |
| `Attack` | caller/turn player | Attack id；可为 0..1 或 1/1 | yes | yes | copied/disabled attack selection, `GameProc.h:241-554`; `EffectInstant.h:1914-1925` |
| `Evolve` | effect-selected actor | Evolve payload：来源卡与进化目标 | yes | yes | `EffectProc.h:476-496` |
| `Count` | caller-selected | Number payload，通常 1/1 | yes | yes | mulligan draw count `SetupProc.h:70-89`; damage-counter counts |
| `YesNo` | caller-selected | Yes/No，固定两个 option | no at vector level | yes or non-strategic | Activate/FirstEffect/Mulligan/manual CoinHead |
| `SpecialCondition` | effect player | condition enum，1/1 | yes | yes | affect/recover condition, `EffectInstant.h:1565-1585,2028-2042` |

`SelectContext` 的完整实际集合是 `Main`、setup、Switch/zone movement、damage/heal、evolution、attach/detach/energy movement、SkillOrder/Attack、Count、IsFirst/Mulligan/Activate/FirstEffect/MoreDevolve/CoinHead、special condition 等，枚举见 [`ApiType.h:39-143`](../../engine/source/ptcgProgram%2022/ApiType.h)。它是 secondary semantic tag，不是 legality rule。

按 engine boundary 分类：

- `TRUE_BRANCH`：accepted selection vectors 在语义上产生多个不同后继，例如多目标 Switch、Card target、Main、SkillOrder。
- `FORCED_SELECTION`：Search 必须再调用一次，但只有一个 accepted vector，例如 options=1,min=max=1 的 Retreat payment。
- `NON_INTERACTIVE`：`selectType=None` 的 function-stack work，或 max=0 的 empty selection；Search 自动执行。

## 4. Forced-selection semantics

若严格按 `checkPlayerSelect()` 的 API contract 枚举，令 `n=options.size()`，则合法有序 vector 数为：

```text
number_of_accepted_vectors = Σ P(n,k),  k = selectMin..selectMax
P(n,k) = n! / (n-k)!
```

原因是 index 不能重复但顺序保留。Probe 明确验证 `n=2,min=max=2` 时 `[0,1]`、`[1,0]` 均通过，而 `[0,0]` 被 error 6 拒绝；所以题目中的“只有 `[A,B]`”并不是官方 checker 的真实 API 语义。对 `SkillOrder`，两个排列还确实有不同执行顺序。

结论：

- 可以仅用 metadata 完整枚举 checker 接受的 vectors；probe 中的 research helper 已实现有序枚举（带防爆上限）、饱和计数和 classification。
- `len(options)==1` 不足够：若 min=0,max=1，则 `{}` 与 `{0}` 都合法；反之 n>1,min=max=n 仍有 `n!` 个合法有序 vector。
- “只有一个 accepted vector”可可靠识别 engine-level forced submission；它不等于“安全、确定、可自动穿过”。
- 反例 1：Main 只剩 End 时 accepted vector 唯一，但 End 可能进入 Asleep/Burned checkup 并消耗 RNG。
- 反例 2：一个唯一 card/energy selection 后续可进入 `ShuffleDeck`、coin、`randomSelect` 或 opponent decision。
- 反例 3：唯一 vector 的效果仍可能改变隐藏区或暴露新信息；唯一性只描述当前输入，不描述下游 transition。

因此 forced 识别需要拆成两层：metadata 可以证明“当前提交唯一”；但 RNG-free、隐藏信息和后续 ownership 不能由当前 min/max/options 证明。

## 5. Actor ownership

`activePlayerIndex()` 从 `turn/firstPlayer` 计算 turn player（[`State.h:1217-1219`](../../engine/source/ptcgProgram%2022/State.h)）；`effectPlayerIndex()` 是当前 ability 的 `usePlayerIndex`（[`State.h:1591-1606`](../../engine/source/ptcgProgram%2022/State.h)）；`selectPlayer` 则是当前 decision owner，三者不是同义字段。

Generic effect 默认 `selectPlayer=usePlayerIndex`；若 effect 设置 `enemySelect`，则翻转为 `1-usePlayerIndex`（[`EffectProc.h:450-453`](../../engine/source/ptcgProgram%2022/EffectProc.h)）。Main 明确设置为 turn player。Trigger ordering 还会按 phase/trigger type 选择 active 或 opposite player（`EffectProc.h:982-1003`）。

真实 opponent-owned testcase 使用 Hippopotas 的 Push Down（[`CardImpl.h:101-108`](../../engine/source/ptcgProgram%2022/CardImpl.h)）：

```text
focal player 1 Main -> Attack(3)
  -> damage/post-effect automatic work
  -> effectSwitch(Enemy).enemySelect()
  -> Card/Switch, selectPlayer=0, options=2
```

Runtime 得到 `activePlayer=1, selectPlayer=0, min=max=1, options=2`，证明 selection owner 可以不是 turn player/attacker。

同一 chain 可以 opponent -> focal：Hand Trimmer 先以 `.enemySelect()` 让对手弃牌，再执行未标 enemySelect 的己方弃牌（[`CardImpl.h:3338-3345`](../../engine/source/ptcgProgram%2022/CardImpl.h)）。引擎没有 simultaneous submission；它用全局单一 `selectPlayer` 和 function stack 串行化 ordered multi-player decisions。Terminal 时不应读取 ownership。

所以原则“只展开 focal-owned forced decision，遇 opponent-owned discretionary decision 停止”可用 `selectPlayer` 与持久化的 focal id 正确判断当前边界；但它只判断当前 decision，不预测后续链会交给谁。

## 6. RNG audit

全源码搜索到的 transition RNG call sites 只有以下五类；初始化 seed 另列在表后。

| Engine operation | RNG call site | Consumes RNG? | Conditions | Notes |
|---|---|---:|---|---|
| Deck shuffle | `CardMove.h:258-270` | conditional yes | deck nonempty；多于一张时 shuffle 实际推进 URBG | `deviceRand=false` 用共享 `Game::rng`，否则 `random_device` |
| Automatic coin | `SelectProc.h:50-68` | yes per flip | `manualCoin=false` | manual coin 改为 YesNo decision，不访问 RNG |
| Coin-until-tail | `SelectProc.h:86-103` | yes, variable count | `manualCoin=false` | 至少取一次 RNG；直到 tail |
| Random effect target | `EffectProc.h:696-699` | conditional yes | `effect.randomSelect` | shuffle `targetList` 后 resize；小于两项时可能不推进 |
| Hidden bottom ordering | `EffectInstant.h:582-589` | yes when size>=2 | `ToDeckBottomClose`, targetList>=2 | 隐藏顺序随机化 |
| Explicit research API shuffle | `Search.h:196-210` | conditional yes | 调用 `Search::shuffle` | 不属于 `SearchStep` 自动行为，但共享同一个 search `Game::rng` |

RNG 初始化而非 transition：`ApiBattleStart` 在 [`Api.h:25-80`](../../engine/source/ptcgProgram%2022/Api.h) 使用 `random_device/seed_seq`；`ApiAgentStart` 在 `Api.h:84-92` 随机 seed；`Game::init` 在 seed=0 时使用 `random_device`（[`Game.h:79-85`](../../engine/source/ptcgProgram%2022/Game.h)）。

操作级结论：

| Engine operation | RNG-free by itself? | Ground truth |
|---|---:|---|
| Shuffle | no | `ShuffleDeck` 是真实 RNG 入口 |
| Draw | yes | `Draw` 从已经固定的 deck 尾部移动卡，`CardMove.h:285-296` |
| Initial Prize placement | yes after setup shuffle | `DeckToPrize` 从固定 deck 尾部移动，`CardMove.h:298-307` |
| Taking Prize | yes unless explicit coin-prize modifier | `SelectPrize2` 是 Card selection；特殊 `coinPrizePlus` 才先 coin |
| Coin flip | no, unless `manualCoin=true` | automatic mode 直接推进 RNG；manual mode 形成 YesNo boundary |
| Random discard/card selection | no | generic `effect.randomSelect` shuffle target list |
| Player deck search selection | yes for selection; normally no for full search-and-shuffle chain | 搜索 target decision 不随机；随后卡文的 shuffle 才取 RNG |
| Ordinary damage | yes | damage math本身无 RNG；coin-dependent damage 通过 `SelectCoin` |
| KO processing | yes by itself | KO/bench/prize movement不直接取 RNG；触发能力或 coin-prize 可另行取 RNG |
| Turn transition and Draw | yes by themselves | TurnEnd/TurnStart/Draw 无直接 RNG |
| Pokémon Checkup | conditional | Poison/Paralyze 无 coin；Asleep 和 Burned 调 `SelectCoin`，`EffectProc.h:31-70` |
| Automatic effects | conditional | 只有走到上述 shuffle/coin/randomSelect/hidden-order call sites 才取 RNG |

Public Search runtime 还证明：Hoothoot Triple Stab（[`CardImpl.h:1615-1622`](../../engine/source/ptcgProgram%2022/CardImpl.h)）的一个 `SearchStep(Attack)` 内执行三次 coin、解析伤害/回合结束/抽牌并到下一 Main，`Game::rng` hash 从 `b125ed87eee2ffd1` 变为 `7389aebcd8cc31c5`。

## 7. Determinism definitions

- **Transition deterministic**：给定完整 internal `State`、selection，以及所有外部随机源状态，next internal State 唯一。`deviceRand=false` 时还必须包含 `Game::rng` 状态；仅复制 `State` 不够，因为 branches 共享 `Game*`。`deviceRand=true` 路径调用 `random_device`，不能仅由 reconstructed State/Game rng 重放。
- **RNG-free**：执行路径完全不访问/推进 RNG。Play Basic、普通 attach、Retreat payment/switch 的 probe chain 属于此类。RNG-free 强于固定 seed 下可重放。
- **Agent-observation deterministic**：对 Agent 可见的 serialized observation 唯一。它不蕴含 internal State 唯一；不同隐藏 deck/hand 可以投影成同一 observation。反过来，内部 RNG 引发的差异也可能暂时只落在隐藏顺序而观察相同。

因此不能用一个 `deterministic` 布尔值混写三者。对当前 Value Search 最直接可审计的是 transition determinism 与 RNG-free；observation determinism 还需要跨 hidden reconstruction/determinization 验证。

## 8. Runtime probe results

Probe：[`tests/official_cpu_search_decision_boundary_probe.cpp`](../../tests/official_cpu_search_decision_boundary_probe.cpp)，build driver：[`tests/official_cpu_search_decision_boundary_probe.py`](../../tests/official_cpu_search_decision_boundary_probe.py)。全部直接编译 unmodified official `Export.cpp`。

运行命令：

```bash
python3 tests/official_cpu_search_decision_boundary_probe.py
```

结果：

```text
TEST 1 PASS
  Play Basic -> automatic handler/Refresh/ToMain -> Main
  returned: selectPlayer=0 type=Main min=1 max=1 options=17

TEST 2 PASS
  Retreat -> DiscardEnergy
  returned: selectPlayer=0 type=Energy min=1 max=1 options=1
  acceptedVectors=1 class=FORCED_SINGLE_SELECTION

TEST 3 PASS
  pay unique Energy -> Switch
  returned: selectPlayer=0 type=Card min=1 max=1 options=3
  acceptedVectors=3 class=FOCAL_BRANCHING_DECISION

TEST 4 PASS
  focal player 1 Push Down -> opponent Switch
  returned: activePlayer=1 selectPlayer=0 type=Card context=Switch
  min=1 max=1 options=2 acceptedVectors=2
  class=OPPONENT_BRANCHING_DECISION

TEST 5 PASS
  Play -> Attach -> Retreat -> pay -> Switch -> Main
  RNG before=a62b3b93a61d5c90 after=a62b3b93a61d5c90

TEST 6 PASS
  focal Hoothoot Triple Stab -> 3 automatic coins -> damage -> turn end
  -> next player's Main
  RNG before=b125ed87eee2ffd1 after=7389aebcd8cc31c5

RESULT PASS
```

额外 internal contract probe（不冒充真实对局 testcase）验证：`options=2,min=max=2` 时 `[0,1]` 和 `[1,0]` 都通过 checker，重复 `[0,0]` 被拒绝。

## 9. Forced-step recognizer feasibility

Research prototype 已能返回：`TERMINAL / NO_SELECTION / FORCED_SINGLE_SELECTION / FOCAL_BRANCHING_DECISION / OPPONENT_BRANCHING_DECISION / UNKNOWN`，并按官方 checker 计算 accepted ordered vectors。

仅凭 public Search observation/metadata，能够可靠得到：

- 当前是否 terminal；
- 当前 accepted vector 是否唯一；
- 当前 owner 是 focal 还是 opponent，前提是调用方持久化 focal player id；
- type/context/options payload。

但下面这个 production 条件目前**不能只靠当前 observation 安全证明**：

```text
while unique forced
  AND focal-owned
  AND entire continuation is deterministic and RNG-free:
    SearchStep(unique vector)
```

卡点是第三项：public metadata 不暴露 pending `functionStack`、完整 effect flags（如 `randomSelect/enemySelect`）、未来 trigger stack 或 `Game::rng` 状态；当前唯一 selection 的后续可自动进入 shuffle/coin/隐藏变化/opponent decision。Internal `State` 能看 pending machinery，但要静态证明任意 continuation RNG-free 仍需沿实际 effect/function path 分析；更稳健的研究方法是执行后同时检查 RNG state、owner 和返回边界，不过 branches 共享 RNG，执行本身会污染其他 branch 的 search-side RNG 顺序。

所以当前已经具备“唯一 accepted selection”和“当前 ownership”识别能力；尚不具备仅凭 public metadata 预判完整 path deterministic/RNG-free 的充分条件。不能把 unique 当成 safe auto-advance。

## 10. One-paragraph conclusion

在官方 CPU engine 中，一个 `SearchStep` 的 decision boundary 是：提交并校验当前有序 selection vector 后，自动执行 function stack，穿过所有 `selectType=None` 以及 clamp 后 `selectMax==0` 的空选择，直到 terminal 或下一个 `selectMax>0` 的 selection；forced decision 可以通过完整计算 `checkPlayerSelect()` 接受的有序 vectors 是否恰为一个来识别，不能用 `len(options)==1` 代替；player ownership 由非 terminal State 的 `selectPlayer` 判断，并需与持久化 focal id 比较；RNG 风险只来自实际走到 deck shuffle、automatic coin、random target、hidden-order shuffle 或显式 Search shuffle 的路径。因而目前已经具备 current-boundary forced/owner 分类，但要定义可安全连续穿越的 deterministic semantic afterstate，仍缺少对 unique selection 后续 function/effect/trigger path 是否 RNG-free、是否改变隐藏状态以及是否转交 opponent 的可靠前瞻证明。
