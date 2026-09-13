# Final system provenance

## Why the public core is the 0045 system

The final visible package family and its deployment code are all 0045 artifacts:

- five standalone `.tar.gz` packages;
- `final-dance-0816.zip`;
- `v2-final-dance-0817.zip`;
- the compact public Meta router and exact-deck router implementations;
- complete-delta checkpoint recovery and strict materialization tests;
- the final audited semantic BC backbone plus PPO/LoRA adaptation stack.

The later-numbered 0047 directory is not treated as the release model. Its design says V11–V15 failed or were stopped and V16 was a corrected continuation. Its large local Policy-0814 files are ignored rather than committed, and the repository contains no committed final 0047 model-only checkpoint. Presenting it as the finished model would overstate the evidence.

## Model lineage

The public implementation is self-contained at runtime, while retaining immutable provenance strings and content hashes for the research lineage:

```text
official-engine semantic prototypes
        -> full-action behavior cloning foundation
        -> replay-pretrained multi-task critic/value initialization
        -> policy/value ownership split
        -> deck-specific PPO specialist
        -> policy-side Option LoRA and shared State adaptation
        -> deployment-effective FP16 artifact, strict-loaded as FP32
        -> public-information checkpoint router for the final package family
```

The write-up must describe this as an engineering lineage, not as a clean single-variable ablation. The repository's retrospective audit documents where historical evidence is strong, partial, invalid, or unavailable.

## Immutable archive boundary

- Commit: `f6e16cae` (`chore: freeze final competition repository`)
- Tag: `archive/final-competition-repo-2026-09-13`
- Git LFS integrity: `git lfs fsck` passed at freeze time.
- Package checksums: `archive/submission/dist/2026-09-13-final-packages.sha256`
- Official engine source was not modified by the archive or cleanup operation.

## Evidence language

Public claims must distinguish:

1. offline exact-action or legal-action imitation metrics;
2. sampled on-policy rollout diagnostics;
3. fixed-seed greedy official-engine evaluation;
4. package smoke tests;
5. actual externally scored competition results, which require a retained receipt or public record.

No result may be called Promote evidence unless both candidate and opponent identity audits pass the canonical policy protocol. Historical Hybrid-0806 results remain invalid as Frozen-0806 evidence.
