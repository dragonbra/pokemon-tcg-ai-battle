from agent import choose_action, deck_list

def agent(observation):
    if not isinstance(observation, dict) or observation.get("select") is None:
        return deck_list()
    return choose_action(observation)
