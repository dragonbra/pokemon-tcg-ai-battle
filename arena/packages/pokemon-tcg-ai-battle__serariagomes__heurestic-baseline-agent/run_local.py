"""Local evaluation: heuristic agent vs the engine's random agent.

Usage:
    python run_local.py [n_games]

Also writes result.html (visual replay of the last game).
"""

import sys
import random

from kaggle_environments import make
from kaggle_environments.envs.cabt.cabt import deck as SAMPLE_DECK

from main import agent as my_agent, DECK


def random_agent(obs):
    if obs["select"] is None:
        return list(DECK)
    return random.sample(
        list(range(len(obs["select"]["option"]))), obs["select"]["maxCount"]
    )


def winner(env):
    r0, r1 = env.state[0].reward, env.state[1].reward
    if r0 is None or r1 is None or r0 == r1:
        return None
    return 0 if r0 > r1 else 1


def main(n_games=10):
    wins = losses = draws = 0
    env = None
    for g in range(n_games):
        env = make("cabt", debug=True)
        # alternate seats so first/second player advantage cancels out
        if g % 2 == 0:
            env.run([my_agent, random_agent])
            mine = 0
        else:
            env.run([random_agent, my_agent])
            mine = 1
        w = winner(env)
        if w is None:
            draws += 1
            tag = "draw"
        elif w == mine:
            wins += 1
            tag = "WIN"
        else:
            losses += 1
            tag = "loss"
        print(f"game {g + 1:>3}: {tag}  (my seat: player {mine})")

    print(f"\nvs random agent over {n_games} games: "
          f"{wins} wins / {losses} losses / {draws} draws "
          f"({100 * wins / max(1, n_games):.0f}% winrate)")

    if env is not None:
        try:
            html = env.render(mode="html")
            with open("result.html", "w") as f:
                f.write(html)
            print("last game replay written to result.html")
        except Exception as e:
            # Older kaggle-environments builds lack the cabt visualizer assets
            # (cabt.js). Purely cosmetic - the evaluation above is unaffected.
            print(f"replay render skipped ({type(e).__name__}); "
                  "try: pip install -U kaggle-environments")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
