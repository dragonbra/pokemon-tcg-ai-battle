from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path


MAIN_TEMPLATE = '''
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from rl.ptcg.inference import PTCGCandidatePolicy


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
DECK = _TEACHER.read_deck_csv()


def _record_model_main_action(obs_dict: dict, option_index: int) -> None:
    """Keep the stateful rule effect handler aligned with a model action."""
    current, player = _TEACHER._your_state(obs_dict)
    select = obs_dict["select"]
    options = select.get("option") or []
    _TEACHER._TURN_MEMORY.sync(current, player, logs=obs_dict.get("logs") or [])
    _TEACHER._TURN_MEMORY.last_main_options = list(options)
    _TEACHER._TURN_MEMORY.record_main_action(options[option_index], current, player)


def agent(obs_dict: dict):
    if obs_dict.get("select") is None:
        return DECK
    select = obs_dict.get("select") or {}
    if int(select.get("type", 0)) == 0 and int(select.get("context", 0)) == 0:
        option_index, _value, confidence = _POLICY.select_with_confidence(obs_dict)
        if confidence >= CONFIDENCE_THRESHOLD:
            _record_model_main_action(obs_dict, option_index)
            return [option_index]
    return _TEACHER.agent(obs_dict)
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
