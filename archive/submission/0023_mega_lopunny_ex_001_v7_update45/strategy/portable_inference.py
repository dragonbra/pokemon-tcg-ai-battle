from __future__ import annotations
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
        if self.encoder is None or self.encoder.actor != actor_index: self.encoder = OnlineCausalEncoder(actor_index, self.deck, self.actor.config)
        options = (observation.get("select") or {}).get("option") or []; minimum = int((observation.get("select") or {}).get("minCount", 0))
        if len(options) > self.actor.config.max_options or minimum > self.actor.config.max_action_steps: return legal_fallback(observation)
        try:
            batch = self.encoder.encode(observation); batch["source_id"] = torch.zeros(1, dtype=torch.long)
            with torch.inference_mode(): return self.actor.greedy_action(batch)
        except (IndexError, RuntimeError, ValueError): self.encoder = None; return legal_fallback(observation)
