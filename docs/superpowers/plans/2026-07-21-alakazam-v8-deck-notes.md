# Alakazam V8 DECK_NOTES Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完善 V8 实际 60 张卡组的逐卡使用笔记，清楚区分 V7 AutoIter 已确认的策略语义与 V8 后续候选建议。

**Architecture:** 只修改 `submission/alakazam_v8/DECK_NOTES.md` 的人工说明，不修改 `main.py`、`deck.csv` 或运行时行为。文档保留现有卡表作为唯一逐卡索引，在表格前记录规则与资源总则，在表格后记录 V8 构筑差异。

**Tech Stack:** Markdown；Shell 校验命令；仓库现有 `scripts/check_assets.py` 资产校验。

## Global Constraints

- 以 `submission/alakazam_v8/deck.csv` 为准，覆盖 22 种卡牌、60 张总数。
- 卡面事实以 `data/official/EN_Card_Data.csv` 为准。
- 策略背景以 V7 AutoIter 的 iter-46 语义和 V8 当前说明为准。
- 使用中文说明，保留英文卡名、ID、攻击名和 Ability 名以便对照代码与 replay。
- 不预设 opponent ID 或完整对手构筑；需要实际观察的效果必须写明触发条件。
- 明确区分“V7 已实现/已验证”和“V8 建议”，不能把尚未落地的建议写成自动行为。
- 不改变 `deck.csv`、运行时 Python、打包脚本或用户已有工作树修改；除非用户另行要求，不执行 git commit。

---

### Task 1: 编写 V8 逐卡策略笔记

**Files:**
- Modify: `submission/alakazam_v8/DECK_NOTES.md`

**Interfaces:**
- Consumes: V8 `deck.csv` 的卡牌种类、ID、数量；官方 CSV 卡面文本；V7 AutoIter 的进化、附能、过牌、恢复、干扰和攻击边界。
- Produces: 一份供人工复核和后续策略实现使用的 Markdown 卡组笔记；不被当前 agent 自动读取。

- [ ] **Step 1: 在表格前加入阅读约定与卡组级边界**

保留现有文件的用途说明，并补充以下明确内容：卡面事实是规则底座；“V7 已实现/已验证”表示当前继承策略已经关注的行为；“V8 建议”表示还要经过针对性回放或测试的候选语义。说明攻击一旦提交就结束本回合；每回合 Supporter、手填 Basic/Special Energy、Retreat 各一次；本回合刚放下的 Basic 不能进化；牌库 10 张进入保护线，只有能闭合最后 Prize 的终局路线可以越过保护线。

- [ ] **Step 2: 按现有表格顺序填写 Pokémon 行**

逐行将状态改为 `已填写`，并在备注中写清以下内容：

| 卡牌 | 必须覆盖的内容 |
|---|---|
| Abra | 是 Basic 和 Poffin 目标；带 Psychic 时优先保留给可验证的 Rare Candy + Alakazam 直通路线；没有直通资源时再考虑自然进化；不使用 `Teleportation Attack`，因为换上的宝可梦通常没有能量且会破坏攻击节奏。 |
| Kadabra | 自然进化后一次性触发 `Psychic Draw` 抽 2；有能量的 Bench Abra 走直通攻击时，先让其它无能量、合法入场的 Abra 进化并过牌；Active Kadabra 有 Psychic 且手里有 Alakazam 时先自然进化 Active；非 KO 的 `Super Psy Bolt` 只作最后手段。 |
| Alakazam | Stage 2 进化触发 `Psychic Draw` 抽 3；`Powerful Hand` 每张手牌提供 20 点伤害，必须在所有非终局过牌、进化和干扰动作完成后提交；只需一张 Psychic，`Enriching Energy` 不能替代它；不为了无即时收益连续消耗 Rare Candy 做多只 Alakazam。 |
| Dunsparce | 70 HP Basic，可由 Poffin 找到，是低 Prize 的铺场和 Dudunsparce 过牌底座；`Trading Places` 是攻击提交而不是普通换位，V7 禁用；只有实际能 KO 或终局闭环等明确收益时才考虑其它攻击。 |
| Dudunsparce | Stage 1 的 `Run Away Draw` 抽 3 后把自身、进化堆叠和所有附着卡洗回牌库；有 Enriching Energy 时要按抽牌与洗回资源计算牌库净变化；只有存在合法接班者或终局换位链时才使用，Bench 为空时不把唯一 Active 洗回去；V8 只有 2 张，不能像 V7 三张构筑一样无成本循环。 |
| Fezandipiti ex | Rule Box、两 Prize 的 Basic，不能由 Poffin 或 Poké Pad 搜索；对手上一回合击倒我方 Pokémon 后，用 `Flip the Script` 抽 3，且每回合最多一次；手牌不足且没有立即 KO 时优先考虑，但最后 Prize 或已有足够 Powerful Hand 时不能让 Ability 抢在攻击前。 |
| Shaymin | Basic、非 Rule Box、低 Prize 的备用 Active；`Flower Curtain` 只保护 Bench 上没有 Rule Box 的 Pokémon，不保护 Active，也不保护 Fezandipiti ex；80 HP 不能被 Buddy-Buddy Poffin 找到；通常用于起手或承受换位，不应消耗 Psychic 去做无明确收益的攻击。 |

- [ ] **Step 3: 按现有表格顺序填写 Energy 行**

逐行将状态改为 `已填写`，并覆盖以下边界：

| 卡牌 | 必须覆盖的内容 |
|---|---|
| Basic {P} Energy | Abra/Kadabra/Alakazam 每只攻击线只需要一张 Psychic；每回合手填只有一次，当前 Active 的确定攻击路线优先，其次为可见接力 Abra；可由 Hilda、Lana's Aid 或 Night Stretcher 回收，不能把手填机会浪费在已有能量的同一只 Pokémon 上。 |
| Enriching Energy | ACE SPEC、只提供无色，不能支付 Abra 线的 Psychic 攻击；从手牌附到 Dunsparce/Dudunsparce 可触发抽 4，再配合 `Run Away Draw` 抽 3；只有 Active Alakazam 已能攻击且过牌路线不会牺牲即时 Prize 时优先给 Dunsparce 线；这是 V8 唯一一张，不能把它当普通能量消耗。 |
| Telepath Psychic Energy | 附着到 Psychic Pokémon 时提供 Psychic，并按卡面触发从牌库检索 Basic Psychic 的效果；适合在建立 Abra 接力线时同时完成附能与能量来源准备；牌库低于保护线时要把检索和后续抽牌一起计入预算；不能把它当作 Enriching Energy 的抽牌替代。 |

- [ ] **Step 4: 按现有表格顺序填写 Trainer 行**

逐行将状态改为 `已填写`，并覆盖以下边界：

| 卡牌 | 必须覆盖的内容 |
|---|---|
| Rare Candy | Item，只能让合法在场且满足回合进化时机的 Basic 直接进化 Stage 2；本回合刚放下的 Abra 不能使用；优先服务带 Psychic、手里已有 Alakazam 且本回合能形成攻击的 primary attacker，不能连续制造没有即时收益的第二只 Alakazam；Item Lock 时不可用。 |
| Enhanced Hammer | Item，只能弃掉对手 Pokémon 身上的 Special Energy，不能处理 Basic Energy；V8 有 4 张，实际看到 Mist、Telepath 或其它阻挡 KO 的特殊能量时价值提高；应先判断移除后是否改变 Powerful Hand/K.O.，无目标或对手只有基础能量时保留；Item Lock 时不可用。 |
| Buddy-Buddy Poffin | Item，从牌库找最多两只 HP 70 或以下的 Basic Pokémon；可找 Abra、Dunsparce，不能找 80 HP Shaymin 或 Fezandipiti ex；优先补足三只 Abra 攻击线并建立 Dunsparce 底座，同时计入牌库消耗；Bench 满或没有具体后续路线时不盲目使用。 |
| Night Stretcher | Item，从弃牌区拿一只 Pokémon 或一张 Basic Energy 到手牌，不能拿 Special Energy；只有能补齐现实的下一只攻击者或 Psychic 能量路线时才使用；它是 Item，可与另一个 Supporter 同回合使用，但 Item Lock 时不可用。 |
| Sacred Ash | Item，把弃牌区最多五只 Pokémon 洗回牌库，不会直接拿到手里；适合在多只 Abra 线被击倒后恢复进化链和未来搜索密度，不适合替代能立即完成攻击的回收；洗回后仍需通过 Poffin、Poké Pad、Hilda 或 Dawn 重新找出资源。 |
| Poké Pad | Item，只能从牌库找没有 Rule Box 的 Pokémon；可找 Abra、Kadabra、Alakazam、Dunsparce、Dudunsparce、Shaymin，不能找 Fezandipiti ex；优先找能完成当前进化或下一次接力的具体阶段，盲目搜索会消耗牌库且不能拿 Energy；Item Lock 时不可用。 |
| Boss's Orders | Supporter，每回合唯一 Supporter 机会之一，把对手 Bench Pokémon 换到 Active；当 Active 不能 KO、Bench 有确定 KO 目标时优先使用，多个确定目标优先剩余 HP 较高者；不能只为换位消耗 Supporter，也要比较 Hilda/Dawn/Lana/Xerosic 的即时收益。 |
| Lana's Aid | Supporter，从弃牌区最多拿三张无 Rule Box Pokémon 和 Basic Energy 的组合；需要同时恢复 Pokémon 与 Psychic 时优先于 Night Stretcher；不能回收 Fezandipiti ex 或 Special Energy；使用前要确认拿回的资源能构成当前或下一次攻击，且会占用本回合唯一 Supporter。 |
| Xerosic's Machinations | Supporter，让对手弃牌直到手牌剩 3；Active Alakazam 已能攻击、对手 Active 已受伤但不能 KO、对手手牌至少 6 张且没有 Boss KO 时，才在攻击前使用；对手手牌较少或本回合有更高价值的进化/恢复/终局路线时保留。 |
| Hilda | Supporter，从牌库找一只 Evolution Pokémon 和一张 Energy；优先服务 Rare Candy + Alakazam、自然进化 Kadabra 或 Dudunsparce + Enriching Energy 的完整路线；没有 Abra 底座时不要只找孤立 Alakazam；Item Lock 下用它找 Kadabra 和 Enriching Energy，改走自然进化与过牌。 |
| Dawn | Supporter，从牌库各找一只 Basic、Stage 1、Stage 2；适合一次补齐 Abra/Kadabra/Alakazam 进化链，也可按实际路线组合 Dunsparce 线；不能找 Energy，且会消耗唯一 Supporter，需与 Hilda、Boss、Lana、Xerosic 比较后续收益。 |
| Nighttime Mine | Stadium，使场上双方每只 Tera Pokémon 的攻击费用增加一张无色；只在观察到实际 Tera Pokémon、且多一张无色确实能改变对手攻击节奏时优先使用；它不是通用增伤或锁攻卡，非 Tera 对局即时收益很低，且不应预设对手构筑。 |

- [ ] **Step 5: 在表格后加入 V8 构筑差异与待验证清单**

说明 V8 相比 V7 的五项变化：Enhanced Hammer 2→4、Nighttime Mine 0→2、Dudunsparce 3→2、移除 Wondrous Patch 1、移除 Battle Cage 2。明确这些变化意味着特殊能量干扰密度提高，但 Dudunsparce 循环次数、弃牌区 Psychic 回收和后场伤害保护下降；不要把移除的两类卡加入逐卡表。最后列出后续需要通过 replay/测试确认的方向：四张 Hammer 是否改善实际 KO 路线、Nighttime Mine 的 Tera 触发价值、两张 Dudunsparce 对接力断档和牌库保护的影响。

### Task 2: 运行文档与资产校验

**Files:**
- Read: `submission/alakazam_v8/deck.csv`
- Read: `submission/alakazam_v8/DECK_NOTES.md`
- Read: `data/official/EN_Card_Data.csv`
- Run: `scripts/check_assets.py`

**Interfaces:**
- Consumes: Task 1 的 Markdown 表格与 V8 卡表。
- Produces: 可复核的覆盖、数量、Markdown 格式和资产校验结果；不产生运行时变更。

- [ ] **Step 1: 校验卡表总数与唯一卡牌覆盖**

运行：

```bash
python3 - <<'PY'
from collections import Counter
from pathlib import Path

deck = [int(line) for line in Path("submission/alakazam_v8/deck.csv").read_text().split()]
assert len(deck) == 60, len(deck)
assert len(Counter(deck)) == 22, len(Counter(deck))
print(f"deck_total={len(deck)} unique_cards={len(Counter(deck))}")
PY
```

预期：输出 `deck_total=60 unique_cards=22`。

- [ ] **Step 2: 校验每个 V8 ID 都有已填写备注**

运行：

```bash
python3 - <<'PY'
import re
from pathlib import Path

notes = Path("submission/alakazam_v8/DECK_NOTES.md").read_text()
deck_ids = {int(line) for line in Path("submission/alakazam_v8/deck.csv").read_text().split()}
rows = re.findall(r"\| (?:Pokémon|Energy|Trainer) \|.*?\| (\d+) \|.*?\| 已填写 \| (.+?) \|", notes)
note_ids = {int(card_id) for card_id, note in rows if note.strip()}
assert deck_ids == note_ids, (sorted(deck_ids - note_ids), sorted(note_ids - deck_ids))
print(f"noted_ids={len(note_ids)}")
PY
```

预期：输出 `noted_ids=22`。

- [ ] **Step 3: 检查 Markdown、资产和差异范围**

运行：

```bash
git diff --check -- submission/alakazam_v8/DECK_NOTES.md
python3 scripts/check_assets.py
```

预期：`git diff --check` 无输出且退出码为 0；`check_assets.py` 以退出码 0 完成。最后用 `git diff -- submission/alakazam_v8/DECK_NOTES.md` 确认本任务只改动目标笔记，设计记录和计划文件属于本次流程文件，用户原有的 `scripts/package_submission.sh` 修改及 V8 产物不被覆盖。
