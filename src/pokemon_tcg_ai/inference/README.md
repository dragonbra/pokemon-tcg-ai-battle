# Inference

The deployable policy uses FP16 artifacts for storage and strict-loads FP32 tensors for runtime.
`PortableSemanticPolicy` serves a single complete policy identity. The final public-information
router is `PublicDeckV3RoutedCompoundPolicy`; it may observe only public, certain opponent Pokémon
identities and never receives an exact-deck label or Critic output.

Final self-contained runtime bundles are in `archive/submission/dist/`.
