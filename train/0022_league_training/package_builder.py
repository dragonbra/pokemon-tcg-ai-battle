"""Export audited 0022 decoder checkpoints as self-contained Arena candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from .decks import load_deck_plugins
from .policy import load_league_actor_critic


ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
CG_SOURCE = ROOT / "evaluation/arena/opponents/dragapult_ex_03_v20260729_rl/cg"
MODEL_SOURCE = PROJECT / "foundation/model_source"
ONTOLOGY = PROJECT / "assets/card_ontology.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_runtime(target: Path) -> None:
    strategy = target / "strategy"
    shutil.copytree(MODEL_SOURCE, strategy / "model_source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(PROJECT / "policy/online_runtime.py", strategy / "online_runtime_source.py")
    text = (strategy / "online_runtime_source.py").read_text(encoding="utf-8")
    text = text.replace("from ..foundation.model_source", "from .model_source")
    (strategy / "online_runtime.py").write_text(text, encoding="utf-8")
    (strategy / "online_runtime_source.py").unlink()
    shutil.copy2(ONTOLOGY, strategy / "card_ontology.json")


def _write_entrypoint(target: Path) -> None:
    (target / "main.py").write_text(
        '''"""0022 exported Arena candidate entrypoint."""
import os
from pathlib import Path
import sys
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(globals().get("__file__", Path.cwd())).resolve()
if ROOT.is_file():
    ROOT = ROOT.parent
if not (ROOT / "deck.csv").is_file() and Path("/kaggle_simulations/agent/deck.csv").is_file():
    ROOT = Path("/kaggle_simulations/agent")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from strategy.portable_inference import PortablePolicy
DECK = [int(x) for x in (ROOT / "deck.csv").read_text().splitlines() if x.strip()]
POLICY = PortablePolicy.from_checkpoint(ROOT / "strategy/model.bin", ROOT / "strategy/card_ontology.json", DECK)
def read_deck_csv():
    return list(DECK)
def agent(observation):
    if observation.get("select") is None:
        POLICY.reset()
        return list(DECK)
    return POLICY.select(observation)
''',
        encoding="utf-8",
    )


def _write_portable(target: Path) -> None:
    (target / "strategy/portable_inference.py").write_text(
        '''from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import torch
from .model_source.base_model import IDOnlyConfig
from .model_source.r15_model import R15ModelConfig
from .model_source.ac_model import ACModelConfig
from .model_source.source_model import SourceModelConfig
from .model_source.source_r15_model import SourceConditionedR15Policy
from .online_runtime import OnlineCausalEncoder

def legal_fallback(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option")
    if not isinstance(options, list): raise ValueError("select.option is not a list")
    minimum, maximum = int(select.get("minCount", 0)), int(select.get("maxCount", len(options)))
    if not 0 <= minimum <= maximum <= len(options): raise ValueError("invalid selection bounds")
    return list(range(minimum))

def _config():
    base = IDOnlyConfig(max_card_id=2048, d_model=320, heads=8, encoder_layers=4, ffn_multiplier=3, dropout=0.1, max_entities=192, max_options=128, max_action_steps=64)
    return R15ModelConfig(ac=ACModelConfig(base=base, event_layers=1, auxiliary_ffn_multiplier=2, goal_roles=4), scenario_layers=2, scenario_ffn_multiplier=3, scale_gate_ffn_multiplier=2, option_initial_scale=0.35), SourceModelConfig(vocabulary_size=509, initial_scale=0.05)

class PortablePolicy:
    def __init__(self, actor: SourceConditionedR15Policy, deck: Sequence[int]):
        torch.set_num_threads(1); self.model = actor.eval(); self.actor = self.model; self.config = actor.config; self.deck = tuple(int(x) for x in deck); self.encoder = None
    @classmethod
    def from_checkpoint(cls, checkpoint: Path, ontology: Path, deck: Sequence[int]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True); config, source = _config()
        actor = SourceConditionedR15Policy(config, source, ontology_path=ontology); actor.load_state_dict(payload["actor"], strict=True)
        return cls(actor, deck)
    def reset(self): self.encoder = None
    def select(self, observation: dict[str, Any]) -> list[int]:
        actor_index = (observation.get("current") or {}).get("yourIndex")
        if actor_index not in (0, 1): raise ValueError("observation has no valid actor")
        if self.encoder is None or self.encoder.actor != actor_index: self.encoder = OnlineCausalEncoder(actor_index, self.deck, self.actor.config.ac.base)
        options = (observation.get("select") or {}).get("option") or []; minimum = int((observation.get("select") or {}).get("minCount", 0))
        if len(options) > self.actor.config.ac.base.max_options or minimum > self.actor.config.ac.base.max_action_steps: return legal_fallback(observation)
        try:
            batch = self.encoder.encode(observation); batch["source_id"] = torch.zeros(1, dtype=torch.long)
            with torch.inference_mode(): return self.actor.greedy_action(batch)
        except (IndexError, RuntimeError, ValueError): self.encoder = None; return legal_fallback(observation)
''', encoding="utf-8")
    portable = target / "strategy/portable_inference.py"
    portable.write_text(
        portable.read_text(encoding="utf-8")
        .replace("self.actor.config.ac.base", "self.actor.config"),
        encoding="utf-8",
    )
    (target / "strategy/inference.py").write_text(
        '"""Shared-evaluation inference contract for exported 0022 policies."""\n'
        "from .portable_inference import legal_fallback\n"
        "__all__ = [\"legal_fallback\"]\n",
        encoding="utf-8",
    )


def build(checkpoint: Path, target_name: str, *, overwrite: bool = False) -> Path:
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_payload, dict):
        raise ValueError("decoder checkpoint payload must be an object")
    deck_id = checkpoint_payload.get("deck_id")
    if not isinstance(deck_id, str) or not deck_id:
        raise ValueError("decoder checkpoint has no valid deck_id")
    plugins = {plugin.deck_id: plugin for plugin in load_deck_plugins(PROJECT / "deck")}
    if deck_id not in plugins:
        raise ValueError(f"decoder checkpoint references unknown League deck: {deck_id}")
    plugin = plugins[deck_id]
    target = ROOT / "evaluation/arena/candidates" / target_name
    if target.exists():
        if not overwrite:
            raise FileExistsError(target)
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(plugin.root / "deck.csv", target / "deck.csv")
    shutil.copytree(CG_SOURCE, target / "cg", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _copy_runtime(target)
    model, identity = load_league_actor_critic("cpu", decoder_checkpoint=checkpoint, deck_id=deck_id, deck_sha256=plugin.deck_sha256)
    payload = {
        "schema_version": "0022_arena_actor_model_only_v1",
        "update": int(checkpoint_payload["update"]),
        "foundation_sha256": identity.weights_sha256,
        "decoder_sha256": sha256(checkpoint),
        "deck_id": deck_id,
        "deck_sha256": plugin.deck_sha256,
        "actor": {key: value.detach().cpu() for key, value in model.actor.state_dict().items()},
    }
    torch.save(payload, target / "strategy/model.bin")
    _write_portable(target)
    _write_entrypoint(target)
    manifest = {
        "schema_version": "0022_arena_candidate_v1",
        "candidate": target_name,
        "deck_id": deck_id,
        "deck_sha256": plugin.deck_sha256,
        "foundation_sha256": identity.weights_sha256,
        "decoder_checkpoint": str(checkpoint),
        "decoder_sha256": sha256(checkpoint),
        "update": payload["update"],
        "policy_role": "live_snapshot",
        "model_only": True,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(build(args.checkpoint, args.target, overwrite=args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
