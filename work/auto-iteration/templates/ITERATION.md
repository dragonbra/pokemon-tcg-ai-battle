# Iteration {{iteration_id}}

> 本文件是记录模板。复制后放入对应的 `history_iterations/` iteration 目录，并将所有
> `{{...}}` 字段替换为实际内容。

## Context

- `iteration_id`: `{{iteration_id}}`
- iteration type: `{{strategy | deck | measurement | evaluation}}`
- control: `{{control_ref}}`
- candidate: `{{candidate_ref}}`
- deck state/hash: `{{deck_ref}}`
- metric profile: `{{metrics/<profile>.md}}`
- metric priority order: `{{high priority -> lower priority}}`
- evaluator/analyzer version: `{{version}}`
- evaluation matrix: `{{opponents x games, turn-order policy}}`

## Long-term intent

- goal level: `{{G0 | G1 | G2 | G3}}`
- long-term goal: `{{goal}}`
- current gap: `{{gap}}`
- supporting rules/design sources: `{{paths}}`

## Diagnosis

- primary failure class: `{{rule_error | strategy_miss | unavailable | unavoidable | measurement_error | outcome_only}}`
- observed facts: `{{facts}}`
- interpretation: `{{interpretation}}`
- unresolved ambiguity: `{{ambiguity}}`

## Hypothesis

- primary hypothesis: `{{single falsifiable hypothesis}}`
- candidate change: `{{one behavior or boundary}}`
- mechanism: `{{why it should help}}`
- alternative path: `{{alternative}}`
- expected target signal: `{{signal}}`
- health risks: `{{risks}}`
- falsification condition: `{{condition}}`

## Evaluation plan

### Correctness gate

- [ ] legal action contract
- [ ] no agent error
- [ ] no cross-game state leakage
- [ ] evaluator event/denominator attribution verified
- [ ] packaging/runtime contract verified

### Focused evaluation

- scope: `{{cases or scenarios}}`
- target event: `{{event}}`
- numerator: `{{numerator}}`
- denominator: `{{denominator}}`
- unavailable/exclusions: `{{rules}}`

### Full comparison

- control/candidate comparability: `{{yes/no and reason}}`
- opponents and games: `{{scope}}`
- first/second-turn sample counts: `{{counts}}`
- repeat/randomness policy: `{{policy}}`
- outcome guardrails: `{{win rate and health signals}}`

### External validation

- official Episode/replay scope: `{{scope or not yet}}`
- expected evidence: `{{evidence}}`

## Results

### Priority review

1. highest-priority result and core ability: `{{...}}`
2. next-priority process ability: `{{...}}`
3. lower-priority penalty or diagnostic signal: `{{...}}`

低优先级指标的改善不能抵消高优先级指标的未解释回退。

### Correctness

`{{gate result}}`

### Target signal

`{{control vs candidate, counts, denominators, direction}}`

### Health and outcome

`{{health signals and overall/first/second win rate}}`

### Case classification

| Class | Count | Interpretation |
| --- | ---: | --- |
| reachable and completed | `{{n}}` | `{{...}}` |
| reachable but missed | `{{n}}` | `{{...}}` |
| unavailable/unavoidable | `{{n}}` | `{{...}}` |
| measurement unknown | `{{n}}` | `{{...}}` |

## Interpretation

- hypothesis status: `{{supported | unverified | refuted}}`
- what improved: `{{...}}`
- what regressed: `{{...}}`
- what cannot be attributed: `{{...}}`

## Decision

- status: `{{promote | observe | reject | measurement_only}}`
- reason: `{{...}}`
- update control/best: `{{yes/no}}`
- next minimum research question: `{{...}}`
