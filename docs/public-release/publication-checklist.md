# Publication checklist

The code has been structurally curated, but publishing a repository is also a rights and privacy decision. Complete these checks before changing GitHub visibility:

- [ ] Choose and add a license for code owned by the project authors.
- [ ] Confirm whether the official engine source/binaries may be redistributed publicly; otherwise replace them with documented download/build instructions.
- [ ] Confirm redistribution terms for Pokemon card data, names, and remotely linked imagery.
- [ ] Confirm competition rules permit redistribution of packaged model/runtime artifacts.
- [ ] Scan the complete Git history, not only the current tree, for credentials, tokens, private email addresses, personal filesystem paths, and large unintended blobs.
- [ ] Decide whether the public remote should contain the full historical object graph or a new history rooted at this curated tree. The archive tag should remain in a private/archive remote if history contains material that cannot be published.
- [ ] Run `git lfs fsck`, package checksums, the unit suite, and a fresh-extraction official-engine smoke.
- [ ] Verify every numeric write-up claim against `docs/writeup/evidence-map.md`.

No license has been invented during cleanup. Until the rights review is complete, the absence of a license means public viewers do not automatically receive open-source reuse rights.
