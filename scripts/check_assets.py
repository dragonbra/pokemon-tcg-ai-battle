from __future__ import annotations

from pathlib import Path
import csv
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "official"
SUBMISSION = ROOT / "submission"


def check_cards_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        first = next(reader, None)
    if not header or not first:
        raise SystemExit(f"{path.name}: empty or malformed")
    print(f"OK {path.name}: {len(header)} columns, first data row card id={first[0]}")


def check_submission() -> None:
    main_py = SUBMISSION / "main.py"
    deck_csv = SUBMISSION / "deck.csv"
    if not main_py.exists():
        raise SystemExit("submission/main.py missing")
    if not deck_csv.exists():
        raise SystemExit("submission/deck.csv missing")
    deck = [line.strip() for line in deck_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise SystemExit(f"deck.csv should have 60 non-empty lines, got {len(deck)}")
    try:
        card_ids = [int(card_id) for card_id in deck]
    except ValueError as exc:
        raise SystemExit(f"deck.csv contains a non-integer card ID: {exc}") from exc
    if any(card_id < 1 for card_id in card_ids):
        raise SystemExit("deck.csv contains a non-positive card ID")

    data_ids: set[int] = set()
    with (DATA / "EN_Card_Data.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if row and row[0] != "Card ID":
                data_ids.add(int(row[0].split(":")[-1]))
    unknown = sorted(set(card_ids) - data_ids)
    if unknown:
        raise SystemExit(f"deck.csv contains IDs absent from EN_Card_Data.csv: {unknown}")
    print(f"OK submission: main.py present, 60 valid cards, counts={dict(Counter(card_ids))}")


def check_simulator() -> None:
    cg = SUBMISSION / "cg"
    if not (cg / "libcg.so").exists():
        raise SystemExit("submission/cg/libcg.so missing")
    if not (cg / "cg.dll").exists():
        raise SystemExit("submission/cg/cg.dll missing")
    print("OK simulator: Linux libcg.so and Windows cg.dll present")


def main() -> None:
    check_cards_csv(DATA / "EN_Card_Data.csv")
    check_cards_csv(DATA / "JP_Card_Data.csv")
    check_submission()
    check_simulator()


if __name__ == "__main__":
    main()
