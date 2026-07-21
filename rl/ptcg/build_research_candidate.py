from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


MAIN_TEMPLATE = '''
from __future__ import annotations

import copy
import importlib.util
import os
import random
from dataclasses import asdict
from pathlib import Path

from rl.ptcg.inference import PTCGCandidatePolicy
from rl.ptcg.rewards import observation_potential


TEACHER_ROOT = Path(__TEACHER_ROOT__)
CHECKPOINT = os.environ.get("PTCG_RL_CHECKPOINT")
if not CHECKPOINT:
    raise RuntimeError("PTCG_RL_CHECKPOINT must point to a trained checkpoint")


def _load_teacher():
    path = TEACHER_ROOT / "main.py"
    spec = importlib.util.spec_from_file_location("rl_research_teacher", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load teacher module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TEACHER = _load_teacher()
_POLICY = PTCGCandidatePolicy.from_checkpoint(CHECKPOINT, map_location="cpu")
CONFIDENCE_THRESHOLD = float(os.environ.get("PTCG_RL_CONFIDENCE_THRESHOLD", "1.1"))
SEARCH_ENABLED = os.environ.get("PTCG_RL_SEARCH", "0") == "1"
SEARCH_BUDGET = max(1, int(os.environ.get("PTCG_RL_SEARCH_BUDGET", "4")))
SEARCH_RNG = random.Random(int(os.environ.get("PTCG_RL_SEARCH_SEED", "7")))
DECK = _TEACHER.read_deck_csv()
_MODEL_HISTORY: list[dict[str, int]] = []


def _model_observation(obs_dict: dict) -> dict:
    enriched = dict(obs_dict)
    enriched["rl_history"] = list(_MODEL_HISTORY)
    return enriched


def _remember_action(obs_dict: dict, action: list[int]) -> None:
    select = obs_dict.get("select") or {}
    options = select.get("option") or []
    if len(action) != 1 or not 0 <= int(action[0]) < len(options):
        return
    option = options[int(action[0])]
    _MODEL_HISTORY.append(
        {
            "type": int(option.get("type", 0) or 0),
            "cardId": int(option.get("cardId", 0) or 0),
            "attackId": int(option.get("attackId", 0) or 0),
        }
    )
    del _MODEL_HISTORY[:-32]


def _record_model_main_action(obs_dict: dict, option_index: int) -> None:
    """Keep the stateful rule effect handler aligned with a model action."""
    current, player = _TEACHER._your_state(obs_dict)
    select = obs_dict["select"]
    options = select.get("option") or []
    _TEACHER._TURN_MEMORY.sync(current, player, logs=obs_dict.get("logs") or [])
    _TEACHER._TURN_MEMORY.last_main_options = list(options)
    _TEACHER._TURN_MEMORY.record_main_action(options[option_index], current, player)


def _first_legal_selection(observation: dict) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    minimum = max(1, int(select.get("minCount", 1) or 1))
    return list(range(minimum)) if len(options) >= minimum else []


def _teacher_search_selection(observation: dict) -> list[int]:
    try:
        action = _TEACHER.agent(observation)
        select = observation.get("select") or {}
        options = select.get("option") or []
        if isinstance(action, list) and 1 <= len(action) <= int(select.get("maxCount", 1) or 1):
            indices = [int(index) for index in action]
            if len(set(indices)) == len(indices) and all(0 <= index < len(options) for index in indices):
                return indices
    except Exception:
        pass
    return _first_legal_selection(observation)


def _search_score(observation: object, your_index: int) -> float:
    state = getattr(observation, "current", None)
    if state is None:
        return -1000.0
    result = getattr(state, "result", -1)
    if result == your_index:
        return 100.0
    if result in (0, 1):
        return -100.0
    observation_dict = asdict(observation)
    value = observation_potential(observation_dict).total
    if getattr(state, "yourIndex", -1) == your_index:
        try:
            _, model_value = _POLICY.select(observation_dict)
            value += 0.25 * model_value
        except Exception:
            pass
    return value


def _search_main_action(obs_dict: dict, model_index: int) -> int | None:
    """Run a small optional forward search over model and teacher first actions.

    The search is deliberately fail-closed. It is an experiment switch, not a
    replacement for the teacher's normal effect handler or the legal-option
    contract.
    """
    if not SEARCH_ENABLED:
        return None
    select = obs_dict.get("select") or {}
    options = select.get("option") or []
    if not options or not obs_dict.get("search_begin_input"):
        return None
    try:
        from cg.api import search_begin, search_end, search_step, to_observation_class
    except Exception:
        return None

    current = obs_dict.get("current") or {}
    your_index = int(current.get("yourIndex", 0) or 0)
    players = current.get("players") or []
    opponent = players[1 - your_index] if len(players) == 2 else {}
    own_deck_count = int((players[your_index] if len(players) == 2 else {}).get("deckCount", 0) or 0)
    opponent_deck_count = int(opponent.get("deckCount", 0) or 0)
    opponent_active = opponent.get("active") or []
    memory_state = copy.deepcopy(getattr(_TEACHER._TURN_MEMORY, "__dict__", {}))
    effect_progress = dict(getattr(_TEACHER, "_EFFECT_PROGRESS", {}))
    first_actions = [model_index]
    teacher_action = _teacher_search_selection(obs_dict)
    if len(teacher_action) == 1 and teacher_action[0] not in first_actions:
        first_actions.append(teacher_action[0])
    first_actions = [index for index in first_actions if 0 <= index < len(options)]
    if not first_actions:
        return None

    best_index: int | None = None
    best_score = float("-inf")
    try:
        for first_index in first_actions[:SEARCH_BUDGET]:
            search_id = None
            try:
                root = search_begin(
                    to_observation_class(obs_dict),
                    your_deck=SEARCH_RNG.sample(DECK, min(own_deck_count, len(DECK))),
                    your_prize=SEARCH_RNG.sample(DECK, min(len((players[your_index] if len(players) == 2 else {}).get("prize") or []), len(DECK))),
                    opponent_deck=[1072] * opponent_deck_count,
                    opponent_prize=[1] * len(opponent.get("prize") or []),
                    opponent_hand=[1] * int(opponent.get("handCount", 0) or 0),
                    opponent_active=[1072] if opponent_active and opponent_active[0] is None else [],
                )
                search_id = root.searchId
                current_state = root.observation
                selection = [first_index]
                for _ in range(64):
                    next_state = search_step(search_id, selection)
                    current_state = next_state.observation
                    state = getattr(current_state, "current", None)
                    if state is None or getattr(state, "result", -1) >= 0:
                        break
                    if getattr(state, "yourIndex", -1) != your_index:
                        break
                    current_dict = asdict(current_state)
                    if current_dict.get("select") is None:
                        break
                    selection = _teacher_search_selection(current_dict)
                    if not selection:
                        break
                score = _search_score(current_state, your_index)
                if score > best_score:
                    best_score = score
                    best_index = first_index
            except Exception:
                continue
            finally:
                if search_id is not None:
                    try:
                        search_end()
                    except Exception:
                        pass
    finally:
        _TEACHER._TURN_MEMORY.__dict__.clear()
        _TEACHER._TURN_MEMORY.__dict__.update(memory_state)
        _TEACHER._EFFECT_PROGRESS.clear()
        _TEACHER._EFFECT_PROGRESS.update(effect_progress)
    return best_index


def agent(obs_dict: dict):
    if obs_dict.get("select") is None:
        _MODEL_HISTORY.clear()
        return _TEACHER.agent(obs_dict)
    select = obs_dict.get("select") or {}
    if int(select.get("type", 0)) == 0 and int(select.get("context", 0)) == 0:
        option_index, _value, confidence = _POLICY.select_with_confidence(
            _model_observation(obs_dict)
        )
        if confidence >= CONFIDENCE_THRESHOLD:
            searched_index = _search_main_action(obs_dict, option_index)
            if searched_index is not None:
                option_index = searched_index
            _record_model_main_action(obs_dict, option_index)
            _remember_action(obs_dict, [option_index])
            return [option_index]
    action = _TEACHER.agent(obs_dict)
    if int(select.get("type", 0)) == 0 and int(select.get("context", 0)) == 0:
        _remember_action(obs_dict, action)
    return action
'''


def build(output: Path, teacher: Path) -> Path:
    output = output.resolve()
    teacher = teacher.resolve()
    if output.exists():
        raise FileExistsError(f"research candidate already exists: {output}")
    if not (teacher / "main.py").is_file():
        raise FileNotFoundError(f"teacher main.py does not exist: {teacher / 'main.py'}")
    output.mkdir(parents=True)
    shutil.copy2(teacher / "deck.csv", output / "deck.csv")
    shutil.copytree(teacher / "cg", output / "cg")
    main = MAIN_TEMPLATE.replace("__TEACHER_ROOT__", repr(str(teacher)))
    (output / "main.py").write_text(main.lstrip(), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "rl"
        / "runs"
        / "research_candidates"
        / "alakazam_bc",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "work" / "alakazam_v9",
    )
    args = parser.parse_args()
    print(build(args.output, args.teacher))


if __name__ == "__main__":
    main()
