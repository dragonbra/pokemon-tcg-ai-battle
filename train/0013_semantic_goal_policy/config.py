"""Frozen project identity and approved source contract for the scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from . import PROJECT_ID


@dataclass(frozen=True)
class ProjectConfig:
    """Auditable source boundary before protocol and dataset construction."""

    project_id: str
    deck: str
    expert_name: str
    expert_name_normalization: str
    source_dates: tuple[str, str]
    winner_only: bool
    submission_id_unavailable: bool
    source_manifest: str
    patch_path: str
    patch_precedence: str
    deck_conditioning: str
    action_contract: str
    visibility_contract: str


PROJECT_CONFIG = ProjectConfig(
    project_id=PROJECT_ID,
    deck="alakazam_dudunsparce",
    expert_name="Yushin Ito",
    expert_name_normalization="casefold_whitespace_collapse_exact",
    source_dates=("2026-07-18", "2026-07-25"),
    winner_only=True,
    submission_id_unavailable=True,
    source_manifest="data/raw/episodes/source_manifest.json",
    patch_path="data/raw/episodes/patches/2026-07-24/episode-87841523-replay.json",
    patch_precedence="patch_replaces_same_episode_from_archive",
    deck_conditioning="actual_registered_deck_per_episode_player",
    action_contract="ordered_full_action",
    visibility_contract="causal_actor_visible_only",
)
