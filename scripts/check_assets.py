from __future__ import annotations

from pathlib import Path
import csv
import sys
from collections import Counter

try:
    from .submission_paths import historical_submission_dirs, work_submission_dirs
except ImportError:
    from submission_paths import historical_submission_dirs, work_submission_dirs


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "official"
EVALUATION_CG_BASELINE = ROOT / "submission" / "alakazam_v8" / "cg"
EXPECTED_EVALUATION_OPPONENT_NAMES = (
    "romanrozen_v9",
    "pilkwang_v2",
    "kokinn_search",
    "penguin_915",
    "crustle_wall",
    "crustle_v1",
    "kiyotah_lucario",
    "kiyotah_dragapult",
    "kiyotah_iono",
    "kiyotah_abomasnow",
    "kacchan_anti_wall",
    "nursrijan_lucario",
    "yakitori_raging_bolt",
    "zoli_dragapult",
    "sue_alakazam",
    "maktha_1084",
    "yanxiaohan",
)
FOREIGN_EVALUATION_REPOSITORY = "/Users/hejinyu/Documents/repos/ptcg-agent-kaggle"


def submission_dirs() -> list[Path]:
    return historical_submission_dirs(ROOT) + work_submission_dirs(ROOT)


def check_cards_csv(path: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        first = next(reader, None)
    if not header or not first:
        raise SystemExit(f"{path.name}: empty or malformed")
    print(f"OK {path.name}: {len(header)} columns, first data row card id={first[0]}")


def check_submission(submission: Path) -> None:
    main_py = submission / "main.py"
    deck_csv = submission / "deck.csv"
    if not main_py.exists():
        raise SystemExit(f"{submission}/main.py missing")
    if not deck_csv.exists():
        raise SystemExit(f"{submission}/deck.csv missing")
    deck = [line.strip() for line in deck_csv.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise SystemExit(f"{submission.name}/deck.csv should have 60 non-empty lines, got {len(deck)}")
    try:
        card_ids = [int(card_id) for card_id in deck]
    except ValueError as exc:
        raise SystemExit(f"{submission.name}/deck.csv contains a non-integer card ID: {exc}") from exc
    if any(card_id < 1 for card_id in card_ids):
        raise SystemExit(f"{submission.name}/deck.csv contains a non-positive card ID")

    data_ids: set[int] = set()
    with (DATA / "EN_Card_Data.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if row and row[0] != "Card ID":
                data_ids.add(int(row[0].split(":")[-1]))
    unknown = sorted(set(card_ids) - data_ids)
    if unknown:
        raise SystemExit(f"{submission.name}/deck.csv contains IDs absent from EN_Card_Data.csv: {unknown}")
    print(f"OK submission/{submission.name}: main.py present, 60 valid cards, counts={dict(Counter(card_ids))}")


def official_card_ids() -> set[int]:
    ids: set[int] = set()
    with (DATA / "EN_Card_Data.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            if row and row[0] != "Card ID":
                ids.add(int(row[0].split(":")[-1]))
    return ids


def check_evaluation_catalog() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from evaluation.cli import validate_catalog
    from evaluation.runtime.loader import compute_cg_manifest

    catalog = ROOT / "evaluation" / "configs" / "opponents.json"
    packages = validate_catalog(catalog, official_card_ids())
    names = tuple(package.name for package in packages)
    if names != EXPECTED_EVALUATION_OPPONENT_NAMES:
        raise SystemExit(
            "evaluation catalog must contain the exact enabled opponent set: "
            f"{', '.join(EXPECTED_EVALUATION_OPPONENT_NAMES)}"
        )

    baseline_hash = compute_cg_manifest(EVALUATION_CG_BASELINE)["tree_hash"]
    opponents_root = ROOT / "evaluation" / "opponents"
    for package in packages:
        expected_root = opponents_root / package.name
        if package.root != expected_root:
            raise SystemExit(f"evaluation/{package.name} must be a direct opponent package")
        if package.cg_manifest["tree_hash"] != baseline_hash:
            raise SystemExit(f"evaluation/{package.name}/cg hash does not match the baseline")
        main_source = package.entrypoint.read_text(encoding="utf-8")
        if FOREIGN_EVALUATION_REPOSITORY in main_source:
            raise SystemExit(f"evaluation/{package.name}/main.py references an external repository")
    print(f"OK evaluation catalog: {len(packages)} standard opponents, matching cg baseline")


def check_simulator(submission: Path) -> None:
    cg = submission / "cg"
    if not (cg / "libcg.so").exists():
        raise SystemExit(f"submission/{submission.name}/cg/libcg.so missing")
    if not (cg / "cg.dll").exists():
        raise SystemExit(f"submission/{submission.name}/cg/cg.dll missing")
    print(f"OK simulator/{submission.name}: Linux libcg.so and Windows cg.dll present")


def main() -> None:
    check_cards_csv(DATA / "EN_Card_Data.csv")
    check_cards_csv(DATA / "JP_Card_Data.csv")
    submissions = submission_dirs()
    if not submissions:
        raise SystemExit("no complete submission directories found under submission/")
    for submission in submissions:
        check_submission(submission)
        check_simulator(submission)
    check_evaluation_catalog()


if __name__ == "__main__":
    main()
