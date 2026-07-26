# Visibility Audit — Official Engine, 2026-07-26

This audit is read-only evidence for `causal_knowledge_v1`. Source paths below are under
`engine/source/ptcgProgram 22/`; no engine source was modified.

## View classification

| Actor-visible payload | Classification | Evidence |
|---|---|---|
| non-null `select.deck` | `ordered_view` | `ToJson.h:221-226` serializes the selecting player's entire current deck; `EffectProc.h:405-447` sets `selectDeck`; eligible cards are separately represented by options at `EffectProc.h:701-709` |
| legal deck options | `eligible_subset` | `EffectProc.h:701-709` |
| normal own deck | `none` unless privileged `sendDeck` | `ToJson.h:124-127` |
| own hand | `ordered_view` | `ToJson.h:115-122` |
| opponent hand | identity `none`, count public | `ToJson.h:115-122` |
| discard | `ordered_view` and public | `ToJson.h:109-110` |
| Prize | identity `none`, opaque slots/count only | `ToJson.h:11-15,112-114` |
| authorized Looking | `ordered_view` | `ToJson.h:159-177` |
| unauthorized Looking | identity `none` | `ToJson.h:159-177` |

`select.context` is a hint, not visibility authority; `State.h:171-175` explicitly warns it may
not be strict. Presence and shape of actor-visible fields determine knowledge.

## Causal event rules

- Move visibility is actor-relative. `CardMove.h:48-49` defines `openType`; `ApiJson.h:208-229`
  emits identity-bearing `MoveCard` only to authorized actors and otherwise emits redacted
  movement.
- Search-to-hand may be public (`EffectInstant.h:459-466`) or owner-only
  (`EffectInstant.h:471-477`). The emitted actor-local log, not the context name, is authoritative.
- Draw identity is visible to the drawer, redacted to the opponent (`AddLog.h:39-52`,
  `ApiJson.h:181-207`). Hidden draws create opponent unknown slots.
- Shuffle reveals no permutation (`AddLog.h:10-14`, `CardMove.h:256-269`). It invalidates deck
  order while preserving independently established membership.
- Prize placement from deck is hidden to both (`CardMove.h:297-306`). Ordinary Prize take is
  owner-only identity visibility (`SelectProc.h:329-332`). No atomic Prize-swap primitive was
  established; reconstruct only explicit constituent movements.
- `serial` is battle-local physical identity (`Card.h:11-26`, `ToJson.h:17-20`) and remains stable
  across moves (`State.h:712-759`). It is cleared on new game; never joined across episodes.

## Timing contract

- `turn=0` is setup; `1` and `2` are the first and second player's first turns
  (`State.h:113-118`).
- Turn start increments turn, resets action count, logs, draws, then exposes the decision
  (`GameProc.h:956-997`).
- `turnActionCount` increments immediately before each decision observation, including setup and
  effect choices (`State.h:1733-1751`); it is not a completed-action ordinal.
- Logs are delivered with independent actor cursors (`State.h:425-429`, `Api.h:99-113`). A causal
  ledger consumes each log when that actor receives it and never backfills future disclosure.

## Fail-closed rules

Unknown event/log shapes are rejected until classified. Opaque null cards reveal no identity.
Opponent hidden draw, hidden Prize movement, and post-shuffle order are never inferred. A non-null
`select.deck` is usable at the current decision, but the selected action's consequences become
ledger facts only on a later actor-visible observation/reconciliation.
