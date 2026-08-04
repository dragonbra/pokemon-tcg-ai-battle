"""Join 0031 epoch metrics to exact frozen decks and freeze the V2 choice."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


FOCAL_DECK_ID = "mega_lopunny_ex_mega_froslass_ex_001"


def count_manifest_sha256(card_ids: Iterable[int]) -> str:
    cards = [int(card) for card in card_ids]
    if len(cards) != 60:
        raise ValueError("exact deck must contain 60 cards")
    counts = [list(item) for item in sorted(Counter(cards).items())]
    encoded = json.dumps(counts, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_deck(path: Path) -> list[int]:
    cards = [int(line) for line in path.read_text(encoding="ascii").splitlines() if line]
    if len(cards) != 60:
        raise ValueError(f"{path} is not an exact 60-card deck")
    return cards


def build_selection_audit(
    *,
    validation_breakdown: Path,
    winner_catalog: Path,
    frozen_root: Path,
    cuda_support: Path,
    selected_deck_id: str = FOCAL_DECK_ID,
) -> dict[str, Any]:
    lines = [json.loads(line) for line in validation_breakdown.read_text().splitlines()]
    epoch_two = [row for row in lines if int(row.get("trainer/epoch", -1)) == 2]
    if len(epoch_two) != 1:
        raise ValueError(f"expected one epoch-2 breakdown, got {len(epoch_two)}")
    by_deck: Mapping[str, Mapping[str, Any]] = epoch_two[0]["arms"][
        "rule_faithful_semantic"
    ]["by_deck"]

    support = json.loads(cuda_support.read_text(encoding="utf-8"))
    supported = {
        row["deck_id"]: row
        for row in support["decks"]
        if row.get("status") == "supported"
    }
    episodes = json.loads(winner_catalog.read_text(encoding="utf-8"))["episodes"]
    coverage: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"train_episodes": 0, "validation_episodes": 0, "source_ids": set()}
    )
    for episode in episodes:
        bucket = coverage[str(episode["deck_sha256"])]
        bucket[f"{episode['split']}_episodes"] += 1
        bucket["source_ids"].add(int(episode["source_id"]))

    ranked = []
    for deck_id in sorted(supported):
        deck_path = frozen_root / deck_id / "deck.csv"
        dataset_hash = count_manifest_sha256(_read_deck(deck_path))
        metrics = by_deck.get(dataset_hash)
        bucket = coverage[dataset_hash]
        ranked.append(
            {
                "deck_id": deck_id,
                "deck_path": str(deck_path),
                "deck_sha256": supported[deck_id]["deck_sha256"],
                "dataset_deck_sha256": dataset_hash,
                "cuda_supported": True,
                "validation": dict(metrics) if metrics is not None else None,
                "train_episodes": bucket["train_episodes"],
                "validation_episodes": bucket["validation_episodes"],
                "source_count": len(bucket["source_ids"]),
            }
        )
    ranked.sort(
        key=lambda row: (
            float((row["validation"] or {}).get("exact_action", -1.0)),
            int((row["validation"] or {}).get("decisions", 0)),
            row["deck_id"],
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked, start=1):
        row["offline_rank"] = index

    selected = next((row for row in ranked if row["deck_id"] == selected_deck_id), None)
    if selected is None or selected["validation"] is None:
        raise ValueError("selected deck lacks epoch-2 validation evidence")
    return {
        "schema": "0032_focal_selection_audit_v1",
        "selection_rule": "user_business_value_override_after_epoch2_evidence_audit",
        "selected_deck_id": selected_deck_id,
        "selected": selected,
        "offline_metric_leader": ranked[0],
        "business_rationale": (
            "Mega Lopunny ex / Mega Froslass ex has submission and product-value "
            "potential; the user explicitly selected it over the offline metric leader."
        ),
        "ranked_cuda_supported_decks": ranked,
    }


def write_audit(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


__all__ = ["FOCAL_DECK_ID", "build_selection_audit", "count_manifest_sha256", "write_audit"]
