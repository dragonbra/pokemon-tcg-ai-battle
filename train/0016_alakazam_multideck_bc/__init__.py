"""0016 multi-deck Alakazam behavioral-cloning experiment."""

from pathlib import Path


PROJECT_ID = "0016_alakazam_multideck_bc"
PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_DECK_PATH = PROJECT_ROOT / "deck.csv"
TARGET_DECK_MANIFEST_PATH = PROJECT_ROOT / "deck_manifest.json"

__all__ = [
    "PROJECT_ID",
    "PROJECT_ROOT",
    "TARGET_DECK_MANIFEST_PATH",
    "TARGET_DECK_PATH",
]
