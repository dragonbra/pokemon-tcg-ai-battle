from __future__ import annotations

from pathlib import Path


PROJECT_ID = "0017_dragapult_terminal_rl"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT = (
    REPOSITORY_ROOT
    / "rl_runs/0015_dragapult_conditioned_bc/versions/"
    "V2_t1_plus_pure_dragapult/checkpoint/epoch-0010-7f75894270984186.pt"
)
SOURCE_CHECKPOINT_SHA256 = (
    "7f7589427098418682f3a53e0763f3fd24850d641a0a664955f6094f90530532"
)
ONTOLOGY_PATH = Path(__file__).resolve().parent / "assets/card_ontology.json"
TARGET_SOURCE_ID = 1
TARGET_DECK_HASH = "213d75c4498e480647d755c1999e2090bcb94580a4a15f9997e431032a822892"
TARGET_DECK_COUNTS = (
    (2, 3),
    (5, 3),
    (7, 2),
    (112, 2),
    (119, 4),
    (120, 4),
    (121, 3),
    (131, 2),
    (132, 1),
    (133, 1),
    (140, 1),
    (184, 1),
    (235, 2),
    (1071, 2),
    (1079, 2),
    (1080, 1),
    (1086, 4),
    (1097, 2),
    (1121, 4),
    (1152, 4),
    (1182, 3),
    (1198, 2),
    (1227, 4),
    (1240, 1),
    (1256, 2),
)
TARGET_DECK = tuple(
    card_id for card_id, count in TARGET_DECK_COUNTS for _ in range(count)
)

# The 20-package arena snapshot immediately before the 2026-07-28 promotion.
# Every additional enabled package remains unseen during the initial PPO phase.
INITIAL_TRAIN_OPPONENTS = (
    "alakazam_dudunsparce_01",
    "alakazam_dudunsparce_02",
    "alakazam_dudunsparce_03_bc",
    "crustle_01",
    "crustle_02",
    "dragapult_ex_01",
    "dragapult_ex_02",
    "ionos_bellibolt_ex_kilowattrel_01",
    "marnies_grimmsnarl_ex_dudunsparce_01",
    "marnies_grimmsnarl_ex_froslass_01",
    "marnies_grimmsnarl_ex_froslass_02",
    "marnies_grimmsnarl_ex_froslass_03_bc",
    "mega_abomasnow_ex_kyogre_01",
    "mega_lucario_ex_solrock_01",
    "mega_lucario_ex_solrock_02",
    "mega_lucario_ex_solrock_03",
    "mega_lucario_ex_solrock_04",
    "mega_lucario_ex_solrock_05",
    "mega_lucario_ex_solrock_06",
    "team_rockets_mewtwo_ex_spidops_01_bc",
)

if len(TARGET_DECK) != 60:
    raise RuntimeError("the frozen THIRD PTCG Club deck must contain 60 cards")
