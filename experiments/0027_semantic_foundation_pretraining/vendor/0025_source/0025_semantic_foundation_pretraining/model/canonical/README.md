# Canonical Semantic Policy

This package is the post-V3 0025 model. It does not consume `IDOnlyCodec` tensors.

Data flow:

1. `PrototypeBank` encodes official Card, Attack, Skill, and Effect prototypes with typed categorical embeddings and numeric projections.
2. `CanonicalStateEncoder` joins the global turn contract, visible card instances, the exact-deck causal ledger, and typed recent events.
3. `CanonicalOptionEncoder` binds every legal option to source/target instances and its explicit Card → Attack/Skill → Effect relations, then cross-attends to the full state memory.
4. `OrderedOptionDecoder` predicts the full ordered option sequence and STOP under simulator `minCount/maxCount` bounds.
5. `CanonicalSemanticPolicy` validates the exact tensor contract and only orchestrates these modules.

The compiler owns deterministic facts. The model owns policy value. Unknown or currently unresolved facts remain explicitly typed rather than being guessed or encoded as numeric zero.
