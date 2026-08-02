# V2 Stop And Formal Frozen Evaluation

The user stopped V2 while update 10 rollout collection was in progress. No update 10 checkpoint
exists; update 9 is the last completed PPO update. The version remains preserved as interrupted.

Checkpoint selection used the independent deterministic 102-game Frozen suite, not sampled
rollout win rate. Update 5 was the best observed point at 32-69-1 (31.37%), so
`update-0005.pt` was exported as the evaluation candidate.

The unmodified official engine completed 510 games against the 51-opponent frozen catalog:
156-354-0, 30.59% win rate, zero errors and zero unfinished games. The 0025 V4 best-greedy
reference was 153-357-0 (30.00%) under its 510-game Frozen evaluation. Three additional wins are
not meaningful evidence of decoder-only PPO improvement. The formal 510-game result supersedes
the smaller 102-game estimate for policy-strength reporting.
