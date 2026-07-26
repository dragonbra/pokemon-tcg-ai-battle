# PI Codex Model Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Codex model and reasoning effort available through the existing CC-SWITCH-backed PI provider.

**Architecture:** Retain the existing `my-codex` provider, local CC-SWITCH URL, and authenticated key command. Expand only the provider model catalog so PI can select both Codex models; map PI's six enabled thinking choices one-to-one to Codex `low` through `ultra`. The mapping marks PI's non-Codex `off` level unsupported so requests omit reasoning rather than send an invented effort token.

**Tech Stack:** PI Coding Agent 0.82.1, JSON model catalog, CC-SWITCH local OpenAI Responses proxy.

## Global Constraints

- Do not alter the existing CC-SWITCH base URL, authentication command, or default model.
- Register only models exposed to this Codex runtime: `gpt-5.6-terra` and `gpt-5.6-sol`.
- PI 0.82.1 accepts exactly `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, and `max`; map its six enabled levels to Codex `low`, `medium`, `high`, `xhigh`, `max`, and `ultra` in order.
- Preserve valid JSON and avoid exposing the resolved API key.

---

### Task 1: Extend the PI model catalog

**Files:**
- Modify: `/home/cyd/.pi/agent/models.json`

**Interfaces:**
- Consumes: existing `providers.my-codex` transport and authentication fields.
- Produces: two `reasoning: true` Codex model entries selectable as `my-codex/gpt-5.6-terra` and `my-codex/gpt-5.6-sol`.

- [ ] **Step 1: Preserve the provider transport and existing Terra entry**

Keep `baseUrl`, `api`, `apiKey`, and `authHeader` byte-for-byte unchanged. Retain Terra's text/image modality and `128000`/`16384` limits.

- [ ] **Step 2: Add the Sol entry**

Add an adjacent `gpt-5.6-sol` model object with a distinct display name, `reasoning: true`, text/image input support, and the same validated context and output limits. Add this `thinkingLevelMap` to both model objects:

```json
{
  "off": null,
  "minimal": "low",
  "low": "medium",
  "medium": "high",
  "high": "xhigh",
  "xhigh": "max",
  "max": "ultra"
}
```

- [ ] **Step 3: Validate JSON and discoverability**

Run:

```bash
jq empty /home/cyd/.pi/agent/models.json
pi --list-models my-codex
```

Expected: JSON parses and PI lists both models as reasoning-capable.

### Task 2: Validate selectable reasoning levels

**Files:**
- Test: PI CLI runtime configuration (no file changes)

**Interfaces:**
- Consumes: registered model catalog and PI's supported `--thinking` parser.
- Produces: proof that both Codex models can be selected with all seven PI-supported effort levels.

- [ ] **Step 1: Enumerate the supported levels**

Use the PI CLI's declared set: `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`. The map exposes its six enabled levels, which correspond respectively to Codex `low`, `medium`, `high`, `xhigh`, `max`, `ultra`; `off` is intentionally unavailable for these reasoning-only Codex models.

- [ ] **Step 2: Parse every model-level selector offline**

Run one no-network empty-prompt initialization for each `my-codex/<model>:<level>` selector, covering two models times six enabled levels. A zero exit status demonstrates PI accepts the selector without sending a model request.

- [ ] **Step 3: Report the compatibility boundary**

State the PI-to-Codex mapping so users can select Codex `ultra` as PI's `max`.
