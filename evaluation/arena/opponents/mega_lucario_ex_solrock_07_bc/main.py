import os
import sys
import random
from collections import defaultdict
from pathlib import Path

from cg.api import (
    AreaType, CardType, EnergyType, Observation, SelectContext, SelectType,
    OptionType, LogType, Card, Pokemon, all_card_data, all_attack,
    to_observation_class,
)

# --- Optional forward-search API (engine exposes a forward model). ----------
# We import defensively: if the names differ in the deployed SDK, USE_SEARCH
# simply stays disabled and we fall back to the heuristic. Nothing crashes.
_SEARCH_OK = False
try:
    from cg.api import search_begin, search_step, search_end, search_release  # type: ignore
    _SEARCH_OK = True
except Exception:
    _SEARCH_OK = False

# ============================================================================
# CONFIG
# ============================================================================
# Keep this False for your first safe submissions. It ships the proven
# heuristic, hardened against crashes. Turn it on ONLY after you have verified
# locally (see local_harness.py -> smoke_test_search) that search_begin works
# in your SDK build and stays within the per-turn time budget. The agent always
# falls back to the heuristic on any search error, so it can never crash even
# if you enable it before fully validating.
USE_SEARCH = False
SEARCH_TIME_BUDGET = 1.5     # seconds, soft cap per decision when searching
SEARCH_MAX_CANDIDATES = 6    # how many first-actions to roll out

# ============================================================================
# DECK
# ============================================================================
file_path = Path(__file__).resolve().with_name("deck.csv")
with file_path.open("r", encoding="utf-8") as f:
    _csv = f.read().split("\n")
my_deck = [int(_csv[i]) for i in range(60)]

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
attack_table = {a.attackId: a for a in all_attack()}

# Decklist IDs (used by the rule-based policy)
Makuhita = 673
Hariyama = 674
Lunatone = 675
Solrock = 676
Riolu = 677
Mega_Lucario_ex = 678
Dusk_Ball = 1102
Switch = 1123
Premium_Power_Pro = 1141
Fighting_Gong = 1142
Poke_Pad = 1152
Hero_Cape = 1159
Boss_Orders = 1182
Carmine = 1192
Explorer_Guidance = 1185
Xerosic_Machinations = 1197
Lillie_Determination = 1227
Full_Metal_Lab = 1244
Gravity_Mountain = 1252
Neutralization_Zone = 1247
Basic_Fighting_Energy = 6
Rock_Fighting_Energy = 20

# Deck-out guard: below this many cards left, stop firing draw-heavy actions
# (Carmine / Lillie / Lunatone's draw ability) so we don't deck ourselves out.
LOW_DECK_COUNT = 8
GREAT_TUSK_MILL_PER_TURN = 4
TURN_DRAW_PER_TURN = 1
GREAT_TUSK_DECK_BUFFER = 10
STALL_CONTROL_DECK_FLOOR = 24
BENCH_INSURANCE_MIN_TURN = 5
BENCH_INSURANCE_ENABLED = True
HAMMER_BASIC_ENERGY_BONUS = 40
HAMMER_SPECIAL_ENERGY_PENALTY = 40

# --- Meta tech: the Day-1 #1 deck is a Crustle wall. -----------------------
# Crustle (345) ability "Mysterious Rock Inn" negates ALL damage from the
# opponent's Pokemon ex. Mega Lucario ex is a mega-ex, so swinging it into
# Crustle does ZERO damage. The deck's answer is Hariyama (non-ex, 210): the
# ability does not stop non-ex attackers. The policy below routes around the
# wall instead of whiffing ex attacks into it.
Crustle = 345
Great_Tusk = 58
Relicanth = 57
Dwebble = 344
Duraludon = 169
Archaludon_ex = 190
Cinderace = 666
Abra = 741
Kadabra = 742
Alakazam = 743
Munkidori = 112
Hops_Trevenant = 879
Ionos_Bellibolt_ex = 269
Cubchoo = 506
Enhanced_Hammer = 1081
Crushing_Hammer = 1120
Gravity_Gemstone = 1166
Metal_Defender = 253
ARCHALUDON_MATCHUP_MARKERS = {Duraludon, Archaludon_ex}
STALL_CONTROL_MARKERS = {Cubchoo, Crushing_Hammer, Gravity_Gemstone}
META_ENGINE_TARGET_BONUS = {
    Alakazam: 420,
    Kadabra: 220,
    Abra: 120,
    Munkidori: 260,
    Crustle: 300,
    Dwebble: 160,
    Hops_Trevenant: 240,
    Ionos_Bellibolt_ex: 260,
}
CRUSTLE_AWARE = True  # set False to reproduce the old "ex into the wall" behavior


class AttackPlan:
    attacker = -1
    target = -1
    attack_index = -1
    remain_hp = -1
    energy = False
    premium_needed = 0
    target_is_ko = False
    target_prize = 0
    wins_game = False


plan = AttackPlan()
pre_turn = 0
ability_used = False
metal_defender_serial = None
premium_power_serials = set()


# ============================================================================
# HELPERS (from the proven sample agent)
# ============================================================================
def get_card(obs, area, index, player_index):
    """Safely pull a Card/Pokemon from a zone. Returns None on anything odd."""
    try:
        ps = obs.current.players[player_index]
        if area == AreaType.DECK:
            return obs.select.deck[index]
        if area == AreaType.HAND:
            return ps.hand[index]
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index]
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index]
    except Exception:
        return None
    return None


def prize_count(pokemon):
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:  # Legacy Energy
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:  # Lillie's Pearl
            count -= 1
    return max(0, count)


def pokemon_score(pokemon):
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130

    pid = pokemon.id
    # De-prioritise low-value supports (Squawkabilly ex, Noctowl, Fan Rotom, Archaludon ex)
    if pid == 144 or pid == 322 or pid == 323 or pid == 337:
        score -= 200
    if pid == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


def has_rule_box(card_data):
    return bool(card_data and (card_data.ex or getattr(card_data, "megaEx", False)))


def attack_damage(
    attacker,
    defender,
    attack_id,
    *,
    stadium_id=0,
    premium_bonus=0,
    weakness_disabled_serial=None,
):
    """Damage visible before attacking, including known temporary modifiers."""
    attack = attack_table.get(attack_id)
    attacker_data = card_table.get(attacker.id)
    defender_data = card_table.get(defender.id)
    if attack is None or attacker_data is None or defender_data is None:
        return 0

    damage = max(0, int(attack.damage))
    if attacker_data.energyType == EnergyType.FIGHTING:
        damage += max(0, int(premium_bonus))

    ignores_weakness = "isn't affected by weakness or resistance" in (
        attack.text or ""
    ).lower()
    weakness_disabled = (
        weakness_disabled_serial is not None
        and defender.serial == weakness_disabled_serial
    )
    if not ignores_weakness:
        if (
            not weakness_disabled
            and defender_data.weakness == attacker_data.energyType
        ):
            damage *= 2
        elif defender_data.resistance == attacker_data.energyType:
            damage = max(0, damage - 30)

    if stadium_id == Full_Metal_Lab and defender_data.energyType == EnergyType.METAL:
        damage = max(0, damage - 30)
    return damage


def premium_ko_plan(
    attacker,
    defender,
    attack_id,
    *,
    stadium_id=0,
    current_bonus=0,
    available_count=0,
    weakness_disabled_serial=None,
):
    """Return current damage, projected KO damage, and exact cards required."""
    damage_now = attack_damage(
        attacker,
        defender,
        attack_id,
        stadium_id=stadium_id,
        premium_bonus=current_bonus,
        weakness_disabled_serial=weakness_disabled_serial,
    )
    if defender.hp <= damage_now:
        return damage_now, damage_now, 0

    for premium_count in range(1, max(0, int(available_count)) + 1):
        candidate_damage = attack_damage(
            attacker,
            defender,
            attack_id,
            stadium_id=stadium_id,
            premium_bonus=current_bonus + 30 * premium_count,
            weakness_disabled_serial=weakness_disabled_serial,
        )
        if defender.hp <= candidate_damage:
            return damage_now, candidate_damage, premium_count
    return damage_now, damage_now, 0


def priority_target_bonus(pokemon):
    # Great Tusk LO is Lucario's current worst matchup. Remove the mill engine
    # and its wall setup pieces when they are genuinely damageable.
    if pokemon.id == Great_Tusk:
        return 700 + 120 * len(pokemon.energies)
    bonus = META_ENGINE_TARGET_BONUS.get(pokemon.id, 0)
    if pokemon.id == Dwebble:
        bonus = max(bonus, 220)
    if pokemon.id == Crustle:
        bonus = max(bonus, 300)
    data = card_table[pokemon.id]
    if data.stage2:
        bonus += 80
    if len(pokemon.energies) >= 2:
        bonus += 80
    return bonus


def archaludon_target_bonus(
    pokemon,
    *,
    facing_archaludon,
    target_is_ko,
    is_active,
    active_arch_is_ko,
):
    """Prefer a removable Relicanth engine without passing a two-prize KO."""
    if not facing_archaludon or not target_is_ko:
        return 0
    if is_active and pokemon.id == Archaludon_ex:
        return 1800
    if pokemon.id == Relicanth and not active_arch_is_ko:
        return 1600
    return 0


def boss_orders_score(attack_plan):
    """Boss is worthwhile only when the planned bench target is a KO."""
    if attack_plan.target < 1 or not attack_plan.target_is_ko:
        return -1
    return 50000 if attack_plan.wins_game else 40000


def estimated_next_attack_damage(attacker, defender, opponent_hand_count):
    """Conservative public-information KO check for emergency benching.

    Count one possible manual attachment on the opponent's next turn.  This is
    intentionally only a gate for board insurance, not a full opponent model.
    """
    if attacker is None or defender is None:
        return 0
    attacker_data = card_table.get(attacker.id)
    defender_data = card_table.get(defender.id)
    if attacker_data is None or defender_data is None:
        return 0
    available_energy = len(attacker.energies) + 1
    best = 0
    for attack_id in getattr(attacker_data, "attacks", []):
        attack = attack_table.get(attack_id)
        if attack is None or len(attack.energies) > available_energy:
            continue
        damage = max(0, int(attack.damage))
        # Alakazam's Powerful Hand uses counters rather than the damage field.
        if attacker.id == Alakazam and attack_id == 1072:
            damage = max(damage, 20 * max(0, opponent_hand_count))
        text = (attack.text or "").lower()
        if "isn't affected by weakness or resistance" not in text:
            if defender_data.weakness == attacker_data.energyType:
                damage *= 2
            elif defender_data.resistance == attacker_data.energyType:
                damage = max(0, damage - 30)
        best = max(best, damage)
    return best


# ============================================================================
# HEURISTIC POLICY  (organizers' tuned logic, kept intact)
# ============================================================================
def heuristic_agent(obs):
    """Returns the option indices for the current selection (descending score)."""
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize = len(my_state.prize)
    # deckCount may be absent in odd states; default high so the guard is off.
    deck_count = getattr(my_state, "deckCount", 999)
    deck_floor = LOW_DECK_COUNT
    low_deck = deck_count <= deck_floor

    global plan, pre_turn, ability_used
    global metal_defender_serial, premium_power_serials
    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()
        ability_used = False
        metal_defender_serial = None
        premium_power_serials = set()

    for log in obs.logs or []:
        if (
            log.type == LogType.ATTACK
            and log.playerIndex != my_index
            and log.attackId == Metal_Defender
        ):
            metal_defender_serial = log.serial
        elif (
            log.type == LogType.PLAY
            and log.playerIndex == my_index
            and log.cardId == Premium_Power_Pro
        ):
            premium_power_serials.add(log.serial)
    premium_power_bonus = 30 * len(premium_power_serials)

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    attacker1 = False
    attacker2 = False
    for card in my_state.active + my_state.bench:
        if card is None:
            continue
        field_counts[card.id] += 1
        if card.id == Makuhita or card.id == Hariyama:
            if len(card.energies) >= 3:
                attacker2 = True
        elif card.id == Riolu or card.id == Mega_Lucario_ex:
            if len(card.energies) >= 2:
                attacker1 = True

    for card in my_state.hand:
        hand_counts[card.id] += 1
    for card in my_state.discard:
        discard_counts[card.id] += 1

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    # Great Tusk wins by milling four cards per attack, so the generic
    # eight-card guard fires far too late.  Preserve enough deck for roughly
    # one attack cycle per remaining prize (opponent mill + our turn draw).
    # This is matchup-gated so normal tempo matchups keep the original draw
    # policy.
    op_visible_pokemon = [
        card for card in op_state.active + op_state.bench if card is not None
    ]
    op_visible_ids = {
        card.id for card in op_visible_pokemon + list(op_state.discard)
        if card is not None
    }
    # Attached tools are public information too.  Gravity Gemstone normally
    # sits on the Active Pokemon, so looking only at top-level Pokemon/discard
    # IDs would miss one of the explicit stall-control markers.
    for pokemon in op_visible_pokemon:
        op_visible_ids.update(
            tool.id for tool in getattr(pokemon, "tools", []) if tool is not None
        )
    facing_great_tusk = bool(
        Great_Tusk in op_visible_ids
        or (
            Explorer_Guidance in op_visible_ids
            and bool(op_visible_ids & {Dwebble, Crustle})
        )
    )
    facing_archaludon = bool(
        op_visible_ids & ARCHALUDON_MATCHUP_MARKERS
    )
    facing_stall_control = bool(op_visible_ids & STALL_CONTROL_MARKERS)
    facing_enhanced_hammer = Enhanced_Hammer in op_visible_ids
    prefer_basic_against_hammer_control = (
        facing_stall_control
        and (
            facing_enhanced_hammer
            or Cubchoo in op_visible_ids
            or Crushing_Hammer in op_visible_ids
        )
    )
    if facing_great_tusk:
        great_tusk_deck_floor = (
            GREAT_TUSK_DECK_BUFFER
            + (GREAT_TUSK_MILL_PER_TURN + TURN_DRAW_PER_TURN) * my_prize
        )
        deck_floor = max(deck_floor, great_tusk_deck_floor)
    if facing_stall_control:
        # Cubchoo/Hammer wins by preventing productive attacks until natural
        # draws and our own draw engine exhaust the deck.  Preserve a much
        # longer runway as soon as a dedicated lock marker is visible.
        deck_floor = max(deck_floor, STALL_CONTROL_DECK_FLOOR)
    low_deck = deck_count <= deck_floor

    board_count = sum(field_counts.values())
    opponent_prize_count = len(op_state.prize)
    my_active = my_state.active[0] if my_state.active else None
    opponent_active = op_state.active[0] if op_state.active else None
    opponent_hand_count = max(0, int(getattr(op_state, "handCount", 0) or 0))
    imminent_active_ko = (
        my_active is not None
        and estimated_next_attack_damage(
            opponent_active,
            my_active,
            opponent_hand_count,
        ) >= my_active.hp
    )
    needs_bench_insurance = (
        BENCH_INSURANCE_ENABLED
        and state.turn >= BENCH_INSURANCE_MIN_TURN
        and board_count <= 2
        and 2 <= opponent_prize_count
        and (
            (board_count == 1 and imminent_active_ko)
            or (board_count == 2 and opponent_prize_count <= 2)
        )
    )
    lillie_draw_count = 8 if my_prize == 6 else 6
    # Lillie removes itself, shuffles the rest of the hand into the deck, then
    # draws.  In a stall game it is a true recycler only when that operation
    # grows the deck rather than shrinking it further.
    lillie_is_net_recycler = (
        facing_stall_control
        and (len(my_state.hand) - 1) > lillie_draw_count
    )

    can_attack = False
    if context == SelectContext.MAIN:
        can_switch = False
        can_op_switch = False
        can_use_mega_brave = False
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card and card.id == Switch:
                    can_switch = True
                elif card and card.id == Boss_Orders:
                    can_op_switch = True
            elif o.type == OptionType.EVOLVE:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card and card.id == Hariyama:
                    can_op_switch = True
            elif o.type == OptionType.RETREAT:
                can_switch = True
            elif o.type == OptionType.ATTACK:
                can_attack = True
                if o.attackId == 983:  # Mega Brave
                    can_use_mega_brave = True

        my_cards = [my_state.active[0]] + list(my_state.bench)
        op_cards = [op_state.active[0]] + list(op_state.bench)

        if state.turn >= 2:
            best_score = -1
            for i, my_pokemon in enumerate(my_cards):
                if my_pokemon is None:
                    continue
                if i != 0 and not can_switch:
                    break
                for a in range(2):
                    attack_id = 0
                    energy_required = 0
                    base_damage = 0
                    base_score = 0
                    if my_pokemon.id == Mega_Lucario_ex:
                        if a == 0:
                            attack_id = 982
                            energy_required = 1
                            base_damage = 130
                            base_score += 60 * min(3, discard_counts[Basic_Fighting_Energy])
                        else:
                            attack_id = 983
                            energy_required = 2
                            base_damage = 270
                        if my_prize == 2 or my_prize == 3:
                            base_score -= 500
                    elif a == 1:
                        break
                    elif my_pokemon.id == Hariyama:
                        attack_id = card_table[Hariyama].attacks[0]
                        energy_required = 3
                        base_damage = 210
                    elif my_pokemon.id == Makuhita:
                        for o in select.option:
                            if o.type == OptionType.EVOLVE:
                                index = o.inPlayIndex
                                if o.inPlayArea == AreaType.BENCH:
                                    index += 1
                                if index == i:
                                    break
                        else:
                            break
                        base_score -= 100
                        attack_id = card_table[Hariyama].attacks[0]
                        energy_required = 3
                        base_damage = 210
                    elif my_pokemon.id == Solrock:
                        if field_counts[Lunatone] >= 1:
                            attack_id = 980
                            energy_required = 1
                            base_damage = 70

                    if base_damage <= 0:
                        continue

                    more_energy = False
                    energy_count = len(my_pokemon.energies)
                    if a == 1 and i == 0 and energy_count >= 2 and not can_use_mega_brave:
                        break
                    if energy_count < energy_required:
                        if hand_counts[Basic_Fighting_Energy] >= 1 and not state.energyAttached:
                            energy_count += 1
                            if energy_count < energy_required:
                                continue
                            else:
                                more_energy = True
                        else:
                            continue

                    active_arch_is_ko = False
                    if (
                        facing_archaludon
                        and
                        opponent_active is not None
                        and opponent_active.id == Archaludon_ex
                    ):
                        _, active_arch_damage, _ = premium_ko_plan(
                            my_pokemon,
                            opponent_active,
                            attack_id,
                            stadium_id=stadium_id,
                            current_bonus=premium_power_bonus,
                            available_count=hand_counts[Premium_Power_Pro],
                            weakness_disabled_serial=metal_defender_serial,
                        )
                        active_arch_is_ko = (
                            opponent_active.hp <= active_arch_damage
                        )

                    for j, op_pokemon in enumerate(op_cards):
                        if op_pokemon is None:
                            continue
                        if j != 0 and not can_op_switch:
                            break
                        data = card_table[op_pokemon.id]
                        if facing_archaludon:
                            damage_now, damage, premium_needed = premium_ko_plan(
                                my_pokemon,
                                op_pokemon,
                                attack_id,
                                stadium_id=stadium_id,
                                current_bonus=premium_power_bonus,
                                available_count=hand_counts[Premium_Power_Pro],
                                weakness_disabled_serial=metal_defender_serial,
                            )
                        else:
                            # Preserve the submitted champion's exact planner
                            # semantics outside the scoped Archaludon matchup.
                            damage = base_damage
                            if data.weakness == EnergyType.FIGHTING:
                                damage *= 2
                            elif data.resistance == EnergyType.FIGHTING:
                                damage -= 30
                            damage_now = damage
                            premium_needed = 0
                        # Crustle wall and Neutralization Zone both void damage
                        # from our ex / mega-ex attackers into non-rule-box
                        # targets. Treat those attacks as whiffs so the policy
                        # does not select a high-paper-damage zero-damage line.
                        my_data = card_table[my_pokemon.id]
                        crustle_immune = (
                            CRUSTLE_AWARE
                            and op_pokemon.id == Crustle
                            and (my_data.ex or my_data.megaEx)
                        )
                        neutral_zone_immune = (
                            stadium_id == Neutralization_Zone
                            and (my_data.ex or my_data.megaEx)
                            and not has_rule_box(data)
                        )
                        if crustle_immune or neutral_zone_immune:
                            damage_now = 0
                            damage = 0
                            premium_needed = 0
                        target_is_ko = op_pokemon.hp <= damage
                        prize = 0
                        score = pokemon_score(op_pokemon)
                        if target_is_ko:
                            prize = prize_count(op_pokemon)
                        else:
                            score *= damage / op_pokemon.hp
                        score += base_score
                        if damage > 0:
                            score += priority_target_bonus(op_pokemon)
                        score += archaludon_target_bonus(
                            op_pokemon,
                            facing_archaludon=facing_archaludon,
                            target_is_ko=target_is_ko,
                            is_active=(j == 0),
                            active_arch_is_ko=active_arch_is_ko,
                        )

                        wins_game = my_prize <= prize
                        terminal_target = (
                            wins_game
                            if facing_archaludon
                            else len(op_state.prize) <= prize
                        )
                        if terminal_target:
                            score = 50000

                        if crustle_immune or neutral_zone_immune:
                            # Never choose a guaranteed-zero swing into a wall.
                            score = -10000

                        if i == 0:
                            score += 220
                        if j == 0:
                            score += 300
                        score += energy_count
                        if best_score < score:
                            best_score = score
                            plan.attacker = i
                            plan.target = j
                            plan.attack_index = a
                            plan.remain_hp = op_pokemon.hp - damage_now
                            plan.energy = more_energy
                            plan.premium_needed = premium_needed
                            plan.target_is_ko = target_is_ko
                            plan.target_prize = prize
                            plan.wins_game = wins_game

    def energy_score(pokemon, active):
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id == Makuhita or pokemon.id == Hariyama:
            if pokemon.id == Hariyama:
                score += 1
            if energy_count < 3:
                score += 100
            if attacker2:
                score -= 50
        elif pokemon.id == Lunatone:
            score -= 100
        elif pokemon.id == Solrock:
            if energy_count < 1:
                score += 20
            else:
                score -= 100
        elif pokemon.id == Riolu or pokemon.id == Mega_Lucario_ex:
            if pokemon.id == Mega_Lucario_ex:
                score += 1
            if energy_count < 2:
                score += 100
            if attacker1:
                score -= 50
        return score

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 100 if context == SelectContext.IS_FIRST else 1
        elif o.type == OptionType.NO:
            score = 0
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = len(card.energies) if isinstance(card, Pokemon) else 0
                if context == SelectContext.SWITCH or context == SelectContext.TO_ACTIVE:
                    if o.playerIndex == my_index:
                        score += energy_count * 2
                        if o.index == plan.attacker - 1:
                            score += 100
                        if card.id == Mega_Lucario_ex:
                            score += 8 if (my_prize == 2 or my_prize == 3) else 20
                        elif card.id == Hariyama and energy_count >= 2:
                            score += 15
                        elif card.id == Makuhita and energy_count >= 2:
                            score += 10
                        elif card.id == Solrock:
                            score += 5
                        elif card.id == Riolu:
                            score += 4
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == Solrock:
                        score = 2 if state.firstPlayer == my_index else 4
                    elif card.id == Riolu:
                        score = 3
                    elif card.id == Makuhita:
                        score = 1
                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Makuhita:
                        score += -10 if field_counts[card.id] >= 1 else 10
                    elif card.id == Hariyama:
                        score += 20 if field_counts[Makuhita] >= 1 else -20
                    elif card.id == Lunatone:
                        if field_counts[card.id] >= 1:
                            score += -20 if needs_bench_insurance else -250
                        else:
                            score += 60
                    elif card.id == Solrock:
                        if field_counts[card.id] >= 1:
                            score += -20 if needs_bench_insurance else -250
                        else:
                            score += 50
                    elif card.id == Riolu:
                        if field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                            score += 20 if needs_bench_insurance else -150
                        elif field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 1:
                            score -= 3
                        else:
                            score += 40
                    elif card.id == Mega_Lucario_ex:
                        score += 40 if field_counts[Riolu] >= 1 else -15
                    elif card.id == Basic_Fighting_Energy:
                        score += 30 if (not ability_used or not state.energyAttached) else -1
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
                elif (context == SelectContext.SETUP_BENCH_POKEMON
                      or context == SelectContext.TO_BENCH):
                    # Bench the Lucario line (Riolu) first, then the draw engine.
                    data = card_table.get(card.id)
                    if data is not None and data.cardType == CardType.POKEMON:
                        if card.id == Riolu:
                            score = 120 - 25 * field_counts[Riolu]
                        elif card.id == Solrock:
                            score = 90 if field_counts[Solrock] == 0 else (35 if needs_bench_insurance else -1)
                        elif card.id == Lunatone:
                            score = 80 if field_counts[Lunatone] == 0 else (30 if needs_bench_insurance else -1)
                        elif card.id == Makuhita:
                            score = 65 if field_counts[Makuhita] == 0 else 10
                elif context == SelectContext.DISCARD:
                    # Pitch redundant/dead cards; protect key pieces.
                    cid = card.id
                    if cid == Basic_Fighting_Energy:
                        score = 45 if hand_counts[cid] >= 2 else 5
                        if plan.energy and not state.energyAttached:
                            score -= 200
                    elif hand_counts[cid] >= 2:
                        score = 70
                    elif (cid == Lunatone or cid == Solrock) and field_counts[cid] >= 1:
                        score = 55
                    elif cid == Gravity_Mountain and stadium_id == Gravity_Mountain:
                        score = 50
                    elif (cid == Carmine or cid == Lillie_Determination) and state.supporterPlayed:
                        score = 30
                    elif cid == Mega_Lucario_ex and field_counts[Riolu] == 0:
                        score = -80
                    elif cid == Hariyama and field_counts[Makuhita] == 0:
                        score = -50
                    elif cid == Xerosic_Machinations and not state.supporterPlayed:
                        score = -45
                    elif cid == Switch and facing_stall_control:
                        score = -90
                    elif cid in (Riolu, Makuhita, Boss_Orders, Hero_Cape):
                        score = -40
                elif (context == SelectContext.DAMAGE_COUNTER
                      or context == SelectContext.DAMAGE_COUNTER_ANY):
                    if isinstance(card, Pokemon):
                        if o.playerIndex != my_index:
                            score = 10000 + prize_count(card) * 1000 - getattr(card, "hp", 0)
                        else:
                            score = -pokemon_score(card)
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            if card is None:
                scores.append(0)
                continue
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id == Lunatone or card.id == Solrock:
                    if field_counts[card.id] >= 1 and not needs_bench_insurance:
                        score = -1
                elif card.id == Riolu:
                    if (
                        field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2
                        and not needs_bench_insurance
                    ):
                        score = -1
            else:
                score = 10000
                if card.id == Switch:
                    score = -1 if plan.attacker <= 0 else 6000
                elif card.id == Premium_Power_Pro:
                    if facing_archaludon:
                        if plan.premium_needed > 0 and can_attack:
                            score = 18000
                        else:
                            score = -1
                    elif state.supporterPlayed and plan.remain_hp <= 0:
                        score = -1
                    elif not can_attack:
                        if (
                            not state.supporterPlayed
                            and hand_counts[Carmine] > 0
                            and hand_counts[Lillie_Determination] == 0
                        ):
                            score = 3050
                        else:
                            score = -1
                    else:
                        score = 5000
                elif card.id == Boss_Orders:
                    if facing_archaludon:
                        score = boss_orders_score(plan)
                    else:
                        score = 3200 if plan.target >= 1 else -1
                elif card.id == Carmine:
                    score = -1 if low_deck else 3000
                elif card.id == Lillie_Determination:
                    score = 3100 if (not low_deck or lillie_is_net_recycler) else -1
                elif card.id == Gravity_Mountain:
                    if stadium_id == 0:
                        score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card is None or pokemon is None:
                scores.append(0)
                continue
            if card.id == Hero_Cape:
                score = 7000
                if pokemon.id == Riolu:
                    score += 100
                elif pokemon.id == Mega_Lucario_ex:
                    score += 200
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                # Enhanced Hammer alone also appears in tempo decks such as
                # Alakazam, where Rock Fighting Energy's attack-effect shield
                # is valuable.  Apply this bias only after the broader
                # Cubchoo/Crushing/Gravity stall shell is confirmed.
                if prefer_basic_against_hammer_control:
                    if card.id == Basic_Fighting_Energy:
                        score += HAMMER_BASIC_ENERGY_BONUS
                    elif card.id == Rock_Fighting_Energy:
                        score -= HAMMER_SPECIAL_ENERGY_PENALTY
                if o.inPlayArea == AreaType.ACTIVE:
                    if plan.attacker == 0 and plan.energy:
                        score += 200
                else:
                    if plan.attacker == 1 + o.inPlayIndex and plan.energy:
                        score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if pokemon is None:
                scores.append(0)
                continue
            score = 9000 + len(pokemon.energies)
            if pokemon.id == Makuhita and plan.target == 0:
                score = -1
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None and card.id == 1267:  # Lumiose City
                score = 1
            elif card is not None and card.id == Lunatone and low_deck:
                score = -1  # Lunar Cycle draws 3 -> don't deck ourselves out
            else:
                score = 30000
        elif o.type == OptionType.RETREAT:
            score = 2000 if plan.attacker >= 1 else -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if plan.attack_index == 1:
                if o.attackId == 983:
                    score += 100
            else:
                if o.attackId != 983:
                    score += 100
        scores.append(score)

    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    if context == SelectContext.MAIN:
        o = select.option[desc_indices[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card is not None and card.id == Lunatone:
                ability_used = True
    return desc_indices


# ============================================================================
# STATE EVALUATION  (used only by the optional forward search)
# ============================================================================
def evaluate_state(obs):
    """Heuristic value of a board from our perspective. Higher = better."""
    st = obs.current
    if st is None:
        return 0.0
    me = st.players[st.yourIndex]
    op = st.players[1 - st.yourIndex]

    # Terminal: prizes are the win condition (6 -> 0).
    val = 0.0
    val += (len(op.prize) - len(me.prize)) * 10000.0  # prize race dominates

    # Reward having set-up attackers and energy in play.
    for p in [me.active[0] if me.active else None] + list(me.bench):
        if p is None:
            continue
        val += len(p.energies) * 120.0
        if p.id == Mega_Lucario_ex:
            val += 400.0
        if p.id == Hariyama:
            val += 200.0
    # Penalise our active being low / opponent active being healthy.
    if me.active and me.active[0] is not None:
        val += me.active[0].hp * 1.0
    if op.active and op.active[0] is not None:
        val -= op.active[0].hp * 1.5  # pressure on their active is good
    # Card advantage (rough).
    val += me.handCount * 5.0
    return val


def _legal_fallback(select):
    """A structurally-legal selection: the first minCount distinct option indices."""
    n = len(select.option)
    k = max(1, select.minCount) if n else 0
    k = min(k, n)
    return list(range(k))


def search_plan(obs_dict, obs):
    """Optional single-turn forward search using the engine's forward model.

    Strategy (kept simple and robust): for the top few first-actions ranked by
    the heuristic, force each one, then GREEDILY complete the rest of our turn
    inside the simulator, and score the resulting board with evaluate_state().
    Pick the first-action whose rollout yields the best board. This turns the
    pure 1-ply greedy policy into "1-step lookahead + greedy completion".

    Returns a list[int] selection, or None to defer to the heuristic.

    NOTE: search_begin's exact input semantics depend on your SDK build. We pass
    obs.search_begin_input when present. Everything is wrapped so any failure
    cleanly defers to the heuristic -- the agent can never crash from search.
    """
    import time
    if not (_SEARCH_OK and USE_SEARCH):
        return None
    select = obs.select
    if select is None or select.context != SelectContext.MAIN:
        return None

    t0 = time.time()
    sbi = getattr(obs, "search_begin_input", None) or obs_dict.get("search_begin_input")
    if sbi is None:
        return None

    base_order = heuristic_agent(obs)  # heuristic ranking of first actions
    candidates = base_order[:SEARCH_MAX_CANDIDATES]

    best_idx, best_val = None, float("-inf")
    for first in candidates:
        if time.time() - t0 > SEARCH_TIME_BUDGET:
            break
        sid = None
        try:
            # Begin a fresh determinized search rooted at the current state.
            res = search_begin(sbi)  # SDK-specific; see local_harness smoke test
            if getattr(res, "error", 0) != 0 or res.state is None:
                return None
            sid = res.state.searchId
            cur = res.state.observation

            # Apply our chosen first action, then greedily finish the turn.
            sel = [first]
            steps = 0
            while steps < 40:
                ar = search_step(sid, sel)
                if getattr(ar, "error", 0) != 0 or ar.state is None:
                    break
                cur = ar.state.observation
                # Stop when the turn is no longer ours, or game ended.
                if cur.select is None or cur.current is None:
                    break
                if cur.current.result is not None and cur.current.result != -1:
                    break
                if cur.current.yourIndex != obs.current.yourIndex:
                    break
                if cur.select.context != SelectContext.MAIN:
                    # sub-selection: greedily resolve it
                    sub = heuristic_agent(cur)
                    sel = sub[: max(1, cur.select.minCount)]
                    steps += 1
                    continue
                # MAIN again -> greedily pick best; if best is END, finish.
                nxt = heuristic_agent(cur)
                sel = [nxt[0]]
                steps += 1
                if cur.select.option[nxt[0]].type == OptionType.END:
                    ar = search_step(sid, sel)
                    if ar.state is not None:
                        cur = ar.state.observation
                    break

            val = evaluate_state(cur)
            if val > best_val:
                best_val, best_idx = val, first
        except Exception:
            return None
        finally:
            try:
                if sid is not None:
                    search_release(sid)
            except Exception:
                pass

    if best_idx is None:
        return None
    # Put the chosen first action at the front; rest in heuristic order.
    rest = [i for i in base_order if i != best_idx]
    return [best_idx] + rest


# ============================================================================
# TOP-LEVEL AGENT  (crash-safe wrapper)
# ============================================================================
def agent(obs_dict):
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        # Cannot even parse -> if this is deck selection, return the deck.
        if obs_dict.get("select") is None:
            return my_deck
        return [0]

    # Initial deck selection.
    if obs.select is None:
        return my_deck

    select = obs.select
    try:
        # Optional lookahead (off by default; safe fallback inside).
        ordered = None
        if USE_SEARCH:
            ordered = search_plan(obs_dict, obs)
        if ordered is None:
            ordered = heuristic_agent(obs)

        # Respect minCount/maxCount and option bounds.
        n = len(select.option)
        ordered = [i for i in ordered if 0 <= i < n]
        if not ordered:
            return _legal_fallback(select)
        k = min(select.maxCount, n)
        k = max(k, min(max(1, select.minCount), n))
        return ordered[:k]
    except Exception:
        # Anything unexpected -> never crash, never forfeit on an exception.
        return _legal_fallback(select)
