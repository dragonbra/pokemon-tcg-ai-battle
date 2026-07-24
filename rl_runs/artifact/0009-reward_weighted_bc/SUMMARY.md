# 0009 Reward-weighted BC Summary

## Final 0009 deliverable

The previous `V1_control_finetune` final label has been retracted. The long-horizon correction
ran the V3 early-setup setting for eight complete epochs before the user stopped epoch 9.
V7 peaks at epoch 2 with validation exact action rate `0.8085779575`; its full-train exact
continues from `0.8859365` to `0.9133667` while validation exact declines to `0.8048866` and
validation loss rises to `0.6476776` by epoch 8.

At the user's direction, V7 epoch 2 is the final 0009 deliverable. Its official-engine report
completed 230/230 games with zero errors and zero unfinished games: 174 wins, 56 losses,
91/115 wins when first and 83/115 when second. The validated archive was submitted to Kaggle
once as ref `54954628` and was `PENDING` immediately after upload.

V1's earlier independent run had 185 wins. V7 is therefore not presented as a strategy SOTA
or a causal improvement over V1; it is selected because the user explicitly chose the
long-curve-confirmed epoch-2 checkpoint and it passed the correctness gate.

## Experiment table

| Version | Main variable | Validation exact | Offline decision | Official gate |
| --- | --- | ---: | --- | --- |
| V1 | low-LR pure BC continuation | 80.817% | control / selected | 230/230, 0 errors, 0 unfinished |
| V2 | 9-class card category embedding | 80.793% | neutral, not carried | not promoted |
| V3 | opening Abra + second-turn setup | 80.858% | small positive | rejected: 1 game error |
| V4 | V3 + strict Dunsparce bridge | 80.828% | rejected; bridge slices regressed | not promoted |
| V5 | V3 + attack quality | 80.858% | Prize slice +20 exact | rejected: 1 unfinished, attack quality regressed in engine |
| V6 | V5 + Post-KO relay | 80.846% | rejected; relay slices unchanged | not promoted |
| V7 | V3 rerun for 8 complete epochs | 80.858% at epoch 2 | selected after long-curve confirmation | 230/230, 0 errors, 174 wins; submitted |

## Reward findings

- Early setup produced a small offline improvement with healthy ESS (`0.950`) but did not
  pass the official G0 gate in its sampled report.
- Strict bridge weighting did not improve either bridge execution slice and was removed.
- Attack quality increased Prize-positive validation exact from `585/768` to `605/768`, but
  its official report had a higher non-Prize attack rate and one step-limit unfinished game.
- Relay annotations were too sparse: only 199 positive train records and 18 positive
  validation records. V6 did not change the relay slices.
- Category embedding is implemented and data-free at input time, but this one-seed ablation
  was neutral and was not combined with rewards.

## Audited identities

- Dataset SHA-256: `36415a61b93fdcdb2e1b17c0fbaebb9432f8d94ed0c203365b36cd3011b09bdf`
- Reward sidecar SHA-256: `97a43da7fef29c7d9b7045cb7a7ae1c4d1efeaeb797238807b5aa5ddce59c2bd`
- Source checkpoint SHA-256: `8141faf9ad65bd3fd75e53b489bb794a03e9636b79b222fc537daba8e0e6cd7c`
- Earlier V1 checkpoint SHA-256: `b9522b808d547a6080c2bd11d6920603b44c8f0a3567c30134258199bb0a244b`
- Earlier V1 inference model SHA-256: `68995af2ff4ec2ff30f1b8b7292ea2b273cdb6f958da5d32fb5f33a8fa958403`
- V7 selected checkpoint SHA-256: `93b7795070f7bcc0ffedd061b66def609d5825e6f9bcc95e201d6dfcca5ca17f`
- V7 inference model SHA-256: `c7b806346a29ce58ea1d514c3790a824e648be82ffe7e239b170b6fe108c0a1a`
- V7 submission archive SHA-256: `f78872528213c79df5b3da5cff01ec3db36608aca1d918f69dbf0a4a04f55776`

The validation reward-slice JSON for V1, V3, V4, V5, and V6 is stored beside each version's
training summary. Full official reports remain under the corresponding
`rl_runs/evaluation/0009-reward_weighted_bc/V<n>_<tag>/run-*/report.html` directory.
