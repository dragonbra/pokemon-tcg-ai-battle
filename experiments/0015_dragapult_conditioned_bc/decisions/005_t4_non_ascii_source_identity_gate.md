# Decision 005: repair non-ASCII source identity before any T4 materialization

Date: 2026-07-27

## Finding

The frozen V1 core index uses an ASCII slug for `source_key`. When a team name has no ASCII
transliteration, the current helper falls back to the single value `non_ascii_source`.

This does not affect V1–V5: their four actual sources are `third_ptcg_club`, `lumenliquidity`,
`oshbocker`, and `flg`, all independently represented. It does affect the future unmaterialized T4
pool. The collapsed T4 bucket contains 191 trajectories / 188 episodes from seven distinct team
names and three registered deck hashes:

- やる気元気ミワハルキ: 1 trajectory;
- カントー地方マスター: 29;
- ペンギン: 1;
- 今井大登: 10;
- 懒惰的金枪鱼: 88;
- 西松大祐: 58;
- 서주영: 4.

Using that bucket as one persona would violate the multi-team BC source-conditioning contract.

## Gate

Do not materialize or train a Marnie/Munkidori arm from the current T4 candidate index. First create
a versioned source-index schema that uses a readable slug when available and a stable hash suffix
derived from the normalized full Unicode team name for empty slugs and collisions. Audit uniqueness
against exact team names before assigning source IDs.

Do not rewrite the V1 core index or renumber the source vocabulary used by V1–V5. The corrected T4
dataset must be a new immutable dataset generation and any training must receive a new repository
version. This preserves existing checkpoint reproducibility while satisfying the future source
identity contract.
