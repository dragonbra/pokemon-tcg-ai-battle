# Limitless Card Image UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Limitless 环境报告升级为环境日报式卡图体验，让牌型、正文卡牌引用、热力图和 exact 60-card 构筑都能直接识别并预览卡牌。

**Architecture:** 在 `render.py` 内建立由冻结代表构筑派生的卡牌注册表和显式牌型代表图映射；所有 HTML 组件只通过统一 helper 渲染卡图、牌型视觉标签与完整构筑卡片。统计快照保持不变，交互由单例 hover preview 和现有 `dialog` 承担。

**Tech Stack:** Python 3.11+、标准库 `unittest`、静态 HTML/CSS/JavaScript、Playwright 浏览器回归。

## Global Constraints

- 不修改 Tournament、pairing、胜率、名称映射或 matchup 证据分级。
- 19 份代表构筑必须继续保持 exact 60 cards。
- 正文以文字为主，小卡图提供识别；完整构筑展示 Pokémon、Trainer、Energy 全量卡图。
- 390px 移动端不得出现页面级横向溢出，宽表仅在自身容器滚动。
- 不执行 `git commit` 或 `git push`。

---

### Task 1: 卡牌注册表与视觉 helper

**Files:**
- Modify: `data/processed/environment_limitless/render.py`
- Test: `tests/test_environment_limitless.py`

**Interfaces:**
- Produces: `_build_card_registry(decks) -> dict[str, dict[str, Any]]`
- Produces: `_card_ref(card, *, compact=False, label=True) -> str`
- Produces: `_archetype_visual(name, registry, *, compact=False) -> str`

- [ ] **Step 1: 写失败测试**

```python
def test_card_registry_and_archetype_visuals_are_complete(self):
    registry = render._build_card_registry(self.snapshot["representative_decklists"])
    self.assertEqual(registry["dragapult ex"]["set"], "TWM")
    visual = render._archetype_visual("Dragapult Dusknoir", registry)
    self.assertIn("Dragapult ex", visual)
    self.assertIn("Dusknoir", visual)
    self.assertIn("data-card-preview", visual)
```

- [ ] **Step 2: 运行并确认失败**

Run: `python3 -m unittest -v tests.test_environment_limitless.ReportContractTests.test_card_registry_and_archetype_visuals_are_complete`

Expected: `AttributeError`，因为 helper 尚未存在。

- [ ] **Step 3: 实现注册表与显式映射**

```python
ARCHETYPE_CARD_NAMES = {
    "Dragapult Dusknoir": ("Dragapult ex", "Dusknoir"),
    "Dragapult Dudunsparce": ("Dragapult ex", "Dudunsparce"),
    "Marnie's Grimmsnarl": ("Marnie's Grimmsnarl ex", "Froslass"),
}

def _build_card_registry(decks):
    registry = {}
    for deck in decks:
        for card in deck["cards"]:
            registry.setdefault(card["name"].casefold(), card)
    return registry
```

补齐报告出现的全部 primary/variant 名称映射；映射引用不存在或不唯一时抛出 `ValueError`，不猜测卡图。

- [ ] **Step 4: 运行测试确认通过**

Run: `python3 -m unittest -v tests.test_environment_limitless.ReportContractTests.test_card_registry_and_archetype_visuals_are_complete`

Expected: `OK`。

### Task 2: 表格、热力图与正文的紧凑卡图

**Files:**
- Modify: `data/processed/environment_limitless/render.py`
- Test: `tests/test_environment_limitless.py`

**Interfaces:**
- Consumes: `_archetype_visual(name, registry, compact=True)`
- Produces: 所有牌型数据单元格与热力图行列表头的 `.archetype-visual`
- Produces: 关键结论中的 `.inline-card-ref`

- [ ] **Step 1: 写失败渲染合同测试**

```python
def test_report_adds_card_visuals_to_analysis_surfaces(self):
    html = render_report(self.snapshot)
    self.assertIn('id="primary-heatmap"', html)
    self.assertGreaterEqual(html.count('class="archetype-visual'), 100)
    self.assertIn('class="inline-card-ref', html)
    self.assertIn('data-card-preview=', html)
```

- [ ] **Step 2: 运行并确认失败**

Run: `python3 -m unittest -v tests.test_environment_limitless.ReportContractTests.test_report_adds_card_visuals_to_analysis_surfaces`

Expected: FAIL，现有页面没有这些组件。

- [ ] **Step 3: 将统一视觉标签接入所有牌型表面**

```python
def deck_label(name: str, *, compact: bool = True) -> str:
    return _archetype_visual(name, card_registry, compact=compact)
```

替换 Points Share、Labs 参赛分布、多龙细分、全变种、代表玩家、Kaggle 差异、一级/细分热力图和克制表的纯文本牌型名称；在成因与训练区为 Dragapult、Dusknoir、Dudunsparce、Blaziken、Froslass、Grimmsnarl 和 Alakazam 等具体卡牌加入内联引用。

- [ ] **Step 4: 加入稳定尺寸 CSS**

```css
.archetype-visual{display:inline-flex;align-items:center;gap:7px;min-width:0}
.archetype-thumbs{display:inline-flex;flex:0 0 auto}
.card-thumb{position:relative;display:inline-grid;width:28px;height:39px}
.inline-card-ref{display:inline-flex;align-items:center;vertical-align:middle}
.matrix .archetype-label{writing-mode:vertical-rl;transform:rotate(180deg)}
```

- [ ] **Step 5: 运行合同测试**

Run: `python3 -m unittest -v tests.test_environment_limitless.ReportContractTests.test_report_adds_card_visuals_to_analysis_surfaces`

Expected: `OK`。

### Task 3: Exact 60-card 全卡图构筑

**Files:**
- Modify: `data/processed/environment_limitless/render.py`
- Test: `tests/test_environment_limitless.py`

**Interfaces:**
- Consumes: `_card_ref(card, label=True)` 和 `_card_image(card, size)`
- Produces: `.deck-card-grid`、`.deck-card-tile`、`.deck-count`

- [ ] **Step 1: 写失败测试**

```python
def test_every_exact_deck_card_type_has_an_image_tile(self):
    html = render_report(self.snapshot)
    expected = sum(len(deck["cards"]) for deck in self.snapshot["representative_decklists"])
    self.assertEqual(html.count('class="deck-card-tile"'), expected)
    self.assertIn('data-card-group="trainer"', html)
    self.assertIn('data-card-group="energy"', html)
```

- [ ] **Step 2: 运行并确认失败**

Run: `python3 -m unittest -v tests.test_environment_limitless.ReportContractTests.test_every_exact_deck_card_type_has_an_image_tile`

Expected: FAIL，因为 Trainer/Energy 当前为纯文本。

- [ ] **Step 3: 统一渲染三个分组**

```python
for group in ("pokemon", "trainer", "energy"):
    tiles = "".join(_deck_card_tile(card) for card in groups[group])
    sections.append(f'<section data-card-group="{group}"><div class="deck-card-grid">{tiles}</div></section>')
```

每个 tile 保留数量、名称、系列、编号、小图、大图 URL 和失败占位，使用固定 `aspect-ratio: 2.5 / 3.5`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python3 -m unittest -v tests.test_environment_limitless.ReportContractTests.test_every_exact_deck_card_type_has_an_image_tile`

Expected: `OK`。

### Task 4: Hover preview 与可访问高清弹窗

**Files:**
- Modify: `data/processed/environment_limitless/render.py`
- Modify: `.tmp/environment_limitless/review/interaction.spec.js`
- Test: `tests/test_environment_limitless.py`

**Interfaces:**
- Consumes: `[data-card-preview]` 元素
- Produces: `#card-hover-preview` 和增强后的 `#card-modal`

- [ ] **Step 1: 写失败浏览器断言**

```javascript
const ref = page.locator('.inline-card-ref').first();
await ref.hover();
await expect(page.locator('#card-hover-preview')).toBeVisible();
await ref.click();
await expect(page.locator('#card-modal')).toHaveJSProperty('open', true);
await page.keyboard.press('Escape');
await expect(page.locator('#card-modal')).toHaveJSProperty('open', false);
```

- [ ] **Step 2: 实现单例 hover 与 modal**

通过事件委托监听 `pointerenter`、`pointerleave`、`focusin`、`focusout` 和 `click`；浮层位置限制在视口内，`dialog` 关闭后恢复触发元素焦点。移动端不依赖 hover。

- [ ] **Step 3: 校验 JavaScript 语法**

Run: `node -e "const fs=require('fs'),vm=require('vm');const h=fs.readFileSync('docs/environment/limitless.html','utf8');[...h.matchAll(/<script>([\\s\\S]*?)<\\/script>/g)].forEach(m=>new vm.Script(m[1]))"`

Expected: exit 0。

### Task 5: 生成、完整验证与视觉复核

**Files:**
- Regenerate: `data/processed/environment_limitless/snapshot.json`
- Regenerate: `docs/environment/limitless.html`
- Test: `tests/test_environment_limitless.py`

**Interfaces:**
- Produces: 最终可审计 HTML 和与其 SHA-256 对应的 snapshot。

- [ ] **Step 1: 重新生成**

Run: `python3 -m data.processed.environment_limitless.generate`

Expected: 8 events、10,923 players、41,663 matches、19 representative decks。

- [ ] **Step 2: 运行聚焦测试**

Run: `python3 -m unittest -v tests.test_environment_limitless`

Expected: 全部通过。

- [ ] **Step 3: 运行浏览器测试**

Run: `NODE_PATH=/Users/hejinyu/.npm/_npx/420ff84f11983ee5/node_modules /Users/hejinyu/.npm/_npx/420ff84f11983ee5/node_modules/.bin/playwright test --config .tmp/environment_limitless/review/playwright.config.js`

Expected: 桌面与移动端全部通过。

- [ ] **Step 4: 视觉与结构检查**

生成桌面首页、正文 hover、热力图、完整构筑和 390px 移动端截图；确认卡图可辨识、文字仍为主体、表格无重叠、页面级 `scrollWidth == clientWidth`。

- [ ] **Step 5: 最终一致性检查**

Run: `python3 -m compileall -q data/processed/environment_limitless tests/test_environment_limitless.py && git diff --check`

Expected: exit 0；snapshot `report_sha256` 与 HTML SHA-256 一致。
