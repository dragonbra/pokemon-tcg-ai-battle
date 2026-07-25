from __future__ import annotations

from collections import Counter


CATEGORY_NAMES = (
    "Basic Pokémon",
    "Stage 1 Pokémon",
    "Stage 2 Pokémon",
    "Item",
    "Supporter",
    "Pokémon Tool",
    "Stadium",
    "Special Energy",
    "Basic Energy",
)
CATEGORY_TO_ID = {name: index + 1 for index, name in enumerate(CATEGORY_NAMES)}

# Frozen from data/official/EN_Card_Data.csv on 2026-07-24. The official source
# contains repeated printing rows, but all 1,267 unique card IDs agree on category.
SOURCE_CSV_SHA256 = "a0ea63cf7adcb65d35436ce0eb390de6e2e35654a7c67c065a45f4abaa00f373"
STATIC_MAPPING_SHA256 = "46cd549b46bfed98dc4afff0466912ab67edf65d1e8689f50a98678c7a4b3de9"
STATIC_MAPPING_VERSION = "en_card_data_20260724_v1"

# IDs 21-1076 are Pokemon. Each character is the exact category ID for the
# corresponding official card ID; the surrounding ranges are uniform categories.
_POKEMON_CATEGORY_SEQUENCE = (
    "2121121122121211111311111112312211211112111112112312121112311222121112312211131112121221"
    "1211123111123111221232123211211111111231131231321211212112311211211111112311111232212211"
    "1112113121211121131111212123211231131112122221223212111123121211231212112121212311211231"
    "2121231121231311231112121211112211121211231112111111111212112123112123121111212112121112"
    "1121111231212111211211211211212312112123111221211212112312111123123123121211212312312312"
    "1211211121211231211231212111211212312121232311211112311121231212112123112112312312212111"
    "2312311212312121211231211123112123112121121121231212312312112121212123121231112123112112"
    "1112312121231123121123121211231112111211121211123211112122111232121121212311211212111121"
    "2123121123112112123112321211211111212121111121231211112311231121231121112121231121121231"
    "2121121121231212112312312112121211212123231121212123121311211212112112121112312123122112"
    "3123111212312311123121231123112312121232123111112111131112121212121211111111111123121111"
    "1121123121212112311211121123123123231211212231123112112121212112231112312111123112111211"
)

if len(_POKEMON_CATEGORY_SEQUENCE) != 1056:
    raise RuntimeError("static Pokemon category sequence has an unexpected length")

OFFICIAL_CARD_CATEGORY_BY_ID: dict[int, int] = {
    **{card_id: 9 for card_id in range(1, 9)},
    **{card_id: 8 for card_id in range(9, 21)},
    **{
        card_id: int(category_id)
        for card_id, category_id in enumerate(_POKEMON_CATEGORY_SEQUENCE, start=21)
    },
    **{card_id: 4 for card_id in range(1077, 1154)},
    **{card_id: 6 for card_id in range(1154, 1181)},
    **{card_id: 5 for card_id in range(1181, 1242)},
    **{card_id: 7 for card_id in range(1242, 1268)},
}

if len(OFFICIAL_CARD_CATEGORY_BY_ID) != 1267:
    raise RuntimeError("static official card category mapping is incomplete")

_counts = Counter(OFFICIAL_CARD_CATEGORY_BY_ID.values())
CATEGORY_COUNTS = {
    name: _counts[category_id] for name, category_id in CATEGORY_TO_ID.items()
}


def build_card_category_lookup(*, card_vocab_size: int = 4096) -> list[int]:
    """Build the padded model-ID lookup from the frozen in-memory mapping."""
    if card_vocab_size < 1:
        raise ValueError("card_vocab_size must be positive")
    lookup = [0] * (card_vocab_size + 1)
    for raw_card_id, category_id in OFFICIAL_CARD_CATEGORY_BY_ID.items():
        encoded_id = raw_card_id + 1
        if encoded_id <= card_vocab_size:
            lookup[encoded_id] = category_id
    return lookup
