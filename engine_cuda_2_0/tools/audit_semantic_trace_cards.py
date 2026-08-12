"""Summarize card-specific events in official semantic JSONL traces."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("card_id", type=int, nargs="+")
    args = parser.parse_args()
    wanted = set(args.card_id)
    events = {card_id: Counter() for card_id in wanted}
    seeds = {card_id: set() for card_id in wanted}
    stadium_ability_seeds = {card_id: set() for card_id in wanted}
    stadium_ability_choices = Counter()
    for raw in args.trace.open(encoding="utf-8"):
        record = json.loads(raw)
        observation = record.get("actor_observation") or {}
        for item in objects(observation):
            card_id = item.get("cardId")
            if card_id not in wanted:
                continue
            key = (
                item.get("type"), item.get("fromArea"), item.get("toArea"),
                item.get("area"),
            )
            events[card_id][key] += 1
            seeds[card_id].add(record["seed"])

        # OptionType.ABILITY is 10 and AreaType.STADIUM is 7.  Resolve the
        # selected option back through current.stadium so this distinguishes
        # merely seeing a Stadium from actually choosing its Ability.
        select = observation.get("select") or {}
        options = select.get("option") or []
        stadium = (observation.get("current") or {}).get("stadium") or []
        for selected_index in record.get("ordered_action") or []:
            if not isinstance(selected_index, int) or not 0 <= selected_index < len(options):
                continue
            option = options[selected_index]
            if option.get("type") != 10 or option.get("area") != 7:
                continue
            stadium_index = option.get("index")
            if not isinstance(stadium_index, int) or not 0 <= stadium_index < len(stadium):
                continue
            card_id = stadium[stadium_index].get("id")
            if card_id in wanted:
                stadium_ability_choices[card_id] += 1
                stadium_ability_seeds[card_id].add(record["seed"])
    print(json.dumps({
        str(card_id): {
            "seeds_observed": len(seeds[card_id]),
            "stadium_ability_choices": stadium_ability_choices[card_id],
            "stadium_ability_seeds": len(stadium_ability_seeds[card_id]),
            "events": [
                {"type": key[0], "from_area": key[1], "to_area": key[2],
                 "area": key[3], "count": count}
                for key, count in events[card_id].most_common()
            ],
        }
        for card_id in sorted(wanted)
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
