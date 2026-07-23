# agent_kazuki_marnie_grimmsnarl_proxy_v0

Rule proxy for the kazuki0123 Marnie's Grimmsnarl ex / Darkness / Dudunsparce deck.

This version is intentionally a readable first imitation, not a tuned final submit:

- Build Marnie's Impidimp into Marnie's Grimmsnarl ex as the main pressure line.
- Use Dawn, Spikemuth Gym, Poke Pad, Buddy-Buddy Poffin, and Rare Candy to assemble the line.
- Prefer Shadow Bullet once available, with bench-damage targets selected by low HP and engine value.
- Use Xerosic's Machinations only after a pressure board exists and the opponent has a large hand.
- Attach Darkness Energy first to Grimmsnarl ex, then backup Grimmsnarl/Morpeko, with Munkidori as a utility target.
- Keep Energy Recycler and Night Stretcher for long resource games.
- Against Archaludon-style boards, prefer putting Shadow Bullet / damage-counter pressure on Duraludon rather than chasing Relicanth only because it has lower HP.
- After Grimmsnarl is online, prioritize getting one Darkness Energy onto Munkidori so Adrena-Brain can convert tank damage into bench pressure.

Local notes:

- First smoke passed with 0 import errors / illegal actions / max-step games.
- Initial 10-agent pressure pool was 15-21; after Duraludon targeting, Munkidori energy, and one-card setup bench discipline, 10-agent 8x confirm was 37-35.
- Strong into the current local Alakazam/Starmie/Dragapult references, but still not enough into the top Archaludon/RL family.
- A go-second probe was worse. A stricter Archaludon Xerosic gate was also worse. Over-spreading Punk Up energy to backup Impidimp/Morgrem was worse.

Known risk: public replay evidence showed Crustle/control as the largest visible hole, and local validation still shows a hole into `agent_rl_ranker_v0`.

Kaggle probe submission:

- Competition: `pokemon-tcg-ai-battle`
- Ref: `54223790`
- Submitted: `2026-07-01 06:50:46.947000` UTC
- Message: `kazuki_marnie_grimmsnarl_proxy_v0: rule proxy probe from public replay style`
- Status at submit check: `PENDING`
