#pragma once

#include <cstddef>
#include <cstdint>

#include "ptcg_cuda/official_rule_layout.cuh"
#include "ptcg_cuda/official_state_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_CORE_HD __host__ __device__
#else
#define PTCG_OFFICIAL_CORE_HD
#endif

namespace ptcg::cuda_engine {

enum OfficialCardRuntimeFlag : std::uint64_t {
    kCardAppear = 1ULL << 0,
    kCardEvolved = 1ULL << 1,
    kCardBenchToActive = 1ULL << 2,
    kCardKo = 1ULL << 3,
    kCardKoAttackDamage = 1ULL << 4,
    kCardKoEnemyAttackDamage = 1ULL << 5,
    kCardKoEnemyAttackDamageActive = 1ULL << 6,
    kCardKoEnemyExAttackDamage = 1ULL << 7,
    kCardKoEnemyTerastalAttackDamage = 1ULL << 8,
    kCardKoEnemyNAttackDamage = 1ULL << 9,
    kCardKoFull = 1ULL << 10,
    kCardKoPrizePlus1 = 1ULL << 11,
    kCardKoPrizeDecreaseOnce = 1ULL << 12,
    kCardKoPrizeZero = 1ULL << 13,
    kCardKoByDamageToHand = 1ULL << 14,
    kCardNoSpecialCondition = 1ULL << 15,
    kCardNoSleepParalyzeConfuse = 1ULL << 16,
    kCardNoSleep = 1ULL << 17,
    kCardKoNoDamageAndEffectAttackNextEnemyTurn = 1ULL << 18,
};

PTCG_OFFICIAL_CORE_HD inline void official_pod_set_card_runtime_flag(
    OfficialCardStatePod* card,
    std::uint64_t flag) {
    if (card == nullptr) return;
    card->runtime_flags |= flag;
    if (flag <= kCardKoEnemyExAttackDamage) {
        card->turn_state[1] |= static_cast<std::uint32_t>(flag) << 24U;
    } else if (flag >= kCardKoEnemyTerastalAttackDamage
        && flag <= kCardKoPrizeZero) {
        card->turn_state[2] |= static_cast<std::uint32_t>(flag >> 8U);
    } else if (flag == kCardKoNoDamageAndEffectAttackNextEnemyTurn) {
        card->turn_state[2] |= 1U << 6U;
    }
}

constexpr std::uint8_t kOfficialChangedFlag = 1U << 0;
constexpr std::uint64_t kOfficialCardNoPrizeFlag = 1ULL << 9;

PTCG_OFFICIAL_CORE_HD inline bool official_pod_clear_special_conditions(
    OfficialStatePod* state,
    std::int32_t player);

PTCG_OFFICIAL_CORE_HD inline bool official_pod_ok(const OfficialStatePod* state) {
    return state->error == static_cast<std::int32_t>(OfficialPodError::kNone);
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_fail(
    OfficialStatePod* state,
    OfficialPodError error,
    std::int32_t detail = 0) {
    if (official_pod_ok(state)) {
        state->error = static_cast<std::int32_t>(error);
        state->error_detail = detail;
    }
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_mark_changed(OfficialStatePod* state) {
    state->control_flags |= kOfficialChangedFlag;
}

template <typename T, std::size_t Capacity>
PTCG_OFFICIAL_CORE_HD inline bool official_pod_push(
    OfficialStatePod* state,
    OfficialPodList<T, Capacity>* list,
    const T& value,
    OfficialPodError overflow = OfficialPodError::kZoneOverflow) {
    if (list->count >= Capacity) {
        official_pod_fail(state, overflow, static_cast<std::int32_t>(Capacity));
        return false;
    }
    list->values[list->count++] = value;
    return true;
}

template <typename T, std::size_t Capacity>
PTCG_OFFICIAL_CORE_HD inline bool official_pod_push_front(
    OfficialStatePod* state,
    OfficialPodList<T, Capacity>* list,
    const T& value,
    OfficialPodError overflow = OfficialPodError::kZoneOverflow) {
    if (list->count >= Capacity) {
        official_pod_fail(state, overflow, static_cast<std::int32_t>(Capacity));
        return false;
    }
    for (std::uint16_t i = list->count; i > 0; --i) {
        list->values[i] = list->values[i - 1];
    }
    list->values[0] = value;
    ++list->count;
    return true;
}

template <typename T, std::size_t Capacity>
PTCG_OFFICIAL_CORE_HD inline T official_pod_remove(
    OfficialStatePod* state,
    OfficialPodList<T, Capacity>* list,
    std::uint16_t index) {
    if (index >= list->count) {
        official_pod_fail(state, OfficialPodError::kInvalidAreaIndex, index);
        return {};
    }
    const T result = list->values[index];
    for (std::uint16_t i = index + 1; i < list->count; ++i) {
        list->values[i - 1] = list->values[i];
    }
    --list->count;
    return result;
}

template <typename T, std::size_t Capacity>
PTCG_OFFICIAL_CORE_HD inline T official_pod_pop_back(
    OfficialStatePod* state,
    OfficialPodList<T, Capacity>* list) {
    if (list->count == 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAreaIndex, -1);
        return {};
    }
    return list->values[--list->count];
}

PTCG_OFFICIAL_CORE_HD inline OfficialCardStatePod* official_pod_card(
    OfficialStatePod* state,
    OfficialCardRefPod ref) {
    if (ref.index == 0 || ref.index >= kOfficialCardCapacity) {
        official_pod_fail(state, OfficialPodError::kInvalidCardRef, ref.index);
        return nullptr;
    }
    return &state->cards[ref.index];
}

PTCG_OFFICIAL_CORE_HD inline const OfficialCardStatePod* official_pod_card(
    const OfficialStatePod* state,
    OfficialCardRefPod ref) {
    if (ref.index == 0 || ref.index >= kOfficialCardCapacity) {
        return nullptr;
    }
    return &state->cards[ref.index];
}

PTCG_OFFICIAL_CORE_HD inline std::int32_t official_pod_card_id_or_zero(
    const OfficialStatePod* state,
    OfficialCardRefPod ref) {
    const OfficialCardStatePod* card = official_pod_card(state, ref);
    return card == nullptr ? 0 : card->card_id;
}

PTCG_OFFICIAL_CORE_HD inline OfficialAreaRefPod official_pod_area_ref(
    OfficialStatePod* state,
    OfficialCardRefPod ref) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    return card == nullptr ? OfficialAreaRefPod{} : OfficialAreaRefPod{ref, 0, card->move_counter};
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_area_ref_valid(
    const OfficialStatePod* state,
    OfficialAreaRefPod ref) {
    const OfficialCardStatePod* card = official_pod_card(state, ref.card);
    return card != nullptr && card->move_counter == ref.move_counter;
}

template <std::size_t Capacity>
PTCG_OFFICIAL_CORE_HD inline OfficialCardRefPod official_pod_zone_get(
    OfficialStatePod* state,
    const OfficialPodList<OfficialCardRefPod, Capacity>* list,
    std::uint16_t index) {
    if (index >= list->count) {
        official_pod_fail(state, OfficialPodError::kInvalidAreaIndex, index);
        return {};
    }
    return list->values[index];
}

PTCG_OFFICIAL_CORE_HD inline OfficialCardRefPod official_pod_get_zone_card(
    OfficialStatePod* state,
    std::int32_t player,
    OfficialArea area,
    std::uint16_t index) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return {};
    }
    OfficialPlayerStatePod* ps = &state->players[player];
    switch (area) {
        case OfficialArea::kDeck: return official_pod_zone_get(state, &ps->deck, index);
        case OfficialArea::kHand: return official_pod_zone_get(state, &ps->hand, index);
        case OfficialArea::kTrash: return official_pod_zone_get(state, &ps->trash, index);
        case OfficialArea::kActive: return official_pod_zone_get(state, &ps->active, index);
        case OfficialArea::kBench: return official_pod_zone_get(state, &ps->bench, index);
        case OfficialArea::kPrize: return official_pod_zone_get(state, &ps->prize, index);
        case OfficialArea::kEnergy: return official_pod_zone_get(state, &ps->energy, index);
        case OfficialArea::kTool: return official_pod_zone_get(state, &ps->tool, index);
        case OfficialArea::kPreEvolution:
            return official_pod_zone_get(state, &ps->pre_evolution, index);
        case OfficialArea::kTemporary:
            return official_pod_zone_get(state, &ps->temporary, index);
        case OfficialArea::kStadium:
            return official_pod_zone_get(state, &state->stadium, index);
        case OfficialArea::kLooking:
            return official_pod_zone_get(state, &state->looking, index);
        case OfficialArea::kPlaying:
            return official_pod_zone_get(state, &state->playing, index);
        default:
            official_pod_fail(state, OfficialPodError::kInvalidArea, static_cast<int>(area));
            return {};
    }
}

PTCG_OFFICIAL_CORE_HD inline OfficialCardRefPod official_pod_remove_zone_card(
    OfficialStatePod* state,
    std::int32_t player,
    OfficialArea area,
    std::uint16_t index) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return {};
    }
    OfficialPlayerStatePod* ps = &state->players[player];
    switch (area) {
        case OfficialArea::kDeck: return official_pod_remove(state, &ps->deck, index);
        case OfficialArea::kHand: return official_pod_remove(state, &ps->hand, index);
        case OfficialArea::kTrash: return official_pod_remove(state, &ps->trash, index);
        case OfficialArea::kActive: return official_pod_remove(state, &ps->active, index);
        case OfficialArea::kBench: return official_pod_remove(state, &ps->bench, index);
        case OfficialArea::kPrize: return official_pod_remove(state, &ps->prize, index);
        case OfficialArea::kEnergy: return official_pod_remove(state, &ps->energy, index);
        case OfficialArea::kTool: return official_pod_remove(state, &ps->tool, index);
        case OfficialArea::kPreEvolution:
            return official_pod_remove(state, &ps->pre_evolution, index);
        case OfficialArea::kTemporary:
            return official_pod_remove(state, &ps->temporary, index);
        case OfficialArea::kStadium:
            return official_pod_remove(state, &state->stadium, index);
        case OfficialArea::kLooking:
            return official_pod_remove(state, &state->looking, index);
        case OfficialArea::kPlaying:
            return official_pod_remove(state, &state->playing, index);
        default:
            official_pod_fail(state, OfficialPodError::kInvalidArea, static_cast<int>(area));
            return {};
    }
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_push_zone_card(
    OfficialStatePod* state,
    std::int32_t player,
    OfficialArea area,
    OfficialCardRefPod ref) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return false;
    }
    OfficialPlayerStatePod* ps = &state->players[player];
    switch (area) {
        case OfficialArea::kDeck: return official_pod_push(state, &ps->deck, ref);
        case OfficialArea::kDeckBottom: return official_pod_push_front(state, &ps->deck, ref);
        case OfficialArea::kHand: return official_pod_push(state, &ps->hand, ref);
        case OfficialArea::kTrash: return official_pod_push(state, &ps->trash, ref);
        case OfficialArea::kActive: return official_pod_push(state, &ps->active, ref);
        case OfficialArea::kBench: return official_pod_push(state, &ps->bench, ref);
        case OfficialArea::kPrize: return official_pod_push(state, &ps->prize, ref);
        case OfficialArea::kEnergy: return official_pod_push(state, &ps->energy, ref);
        case OfficialArea::kTool: return official_pod_push(state, &ps->tool, ref);
        case OfficialArea::kPreEvolution:
            return official_pod_push(state, &ps->pre_evolution, ref);
        case OfficialArea::kTemporary:
            return official_pod_push(state, &ps->temporary, ref);
        case OfficialArea::kStadium:
            return official_pod_push(state, &state->stadium, ref);
        case OfficialArea::kLooking:
            return official_pod_push(state, &state->looking, ref);
        case OfficialArea::kPlaying:
            return official_pod_push(state, &state->playing, ref);
        default:
            official_pod_fail(state, OfficialPodError::kInvalidArea, static_cast<int>(area));
            return false;
    }
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_is_in_play(OfficialArea area) {
    return area == OfficialArea::kActive || area == OfficialArea::kBench;
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_clear_card_state(
    OfficialCardStatePod* card) {
    card->damage = 0;
    card->skill_order = 0;
    card->take_attack_damage_this_turn = 0;
    card->take_attack_damage_pre_turn = 0;
    card->ability_used_count = 0;
    for (int i = 0; i < 8; ++i) card->ability_used[i] = 0;
    card->next_enemy_turn_end_battlefield = 0;
    card->next_enemy_turn_end = 0;
    for (int i = 0; i < 3; ++i) card->turn_state[i] = 0;
    for (int i = 0; i < 5; ++i) card->continual_state[i] = 0;
    card->runtime_flags = 0;
    card->ko_prize_change_always = 0;
    card->ko_prize_change = 0;
    card->hp_change = 0;
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_clear_next_turn_state(
    OfficialCardStatePod* card,
    bool preserve_active_to_bench_attack_limit) {
    constexpr std::uint32_t kCannotAttackLessEqualEnergy2 = 1U << 3U;
    const std::uint32_t this_turn_preserved = preserve_active_to_bench_attack_limit
        ? card->this_turn[3] & kCannotAttackLessEqualEnergy2
        : 0;
    const std::uint32_t next_turn_preserved = preserve_active_to_bench_attack_limit
        ? card->next_turn[3] & kCannotAttackLessEqualEnergy2
        : 0;
    for (int i = 0; i < 4; ++i) {
        card->this_turn[i] = 0;
        card->next_turn[i] = 0;
    }
    card->this_turn[3] = this_turn_preserved;
    card->next_turn[3] = next_turn_preserved;
    card->this_turn_enemy = 0;
    card->next_turn_enemy = 0;
    card->next_enemy_turn_end = 0;
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_card_moved(
    OfficialStatePod* state,
    OfficialCardRefPod ref,
    OfficialArea new_area,
    bool reverse = false) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr || card->area == static_cast<std::uint8_t>(new_area)) return;
    const OfficialArea old_area = static_cast<OfficialArea>(card->area);
    if (old_area == OfficialArea::kActive && card->player >= 0 && card->player <= 1) {
        state->players[card->player].active_state = 0;
        for (std::int32_t index =
                 static_cast<std::int32_t>(state->delay_triggers.count) - 1;
             index >= 0;
             --index) {
            const OfficialCardStatePod* subject = official_pod_card(
                state, state->delay_triggers.values[index].trigger.subject.card);
            if (subject != nullptr && subject->player == card->player) {
                official_pod_remove(
                    state,
                    &state->delay_triggers,
                    static_cast<std::uint16_t>(index));
            }
        }
    }
    if (official_pod_is_in_play(old_area) && official_pod_is_in_play(new_area)) {
        // Active/bench switches preserve damage, attachments, and effects.
    } else {
        official_pod_clear_card_state(card);
        card->move_counter = state->move_counter++;
        if (official_pod_is_in_play(new_area)) {
            official_pod_set_card_runtime_flag(card, kCardAppear);
        }
    }
    if (new_area != OfficialArea::kActive) {
        official_pod_clear_next_turn_state(
            card,
            old_area == OfficialArea::kActive
                && new_area == OfficialArea::kBench);
        card->cannot_use_attack_id_non_active = 0;
    }
    card->pre_area = card->area;
    card->area = static_cast<std::uint8_t>(new_area);
    card->reverse = reverse ? 1 : 0;
    card->attach_move_counter = 0;
}

PTCG_OFFICIAL_CORE_HD inline OfficialCardRefPod official_pod_move_card(
    OfficialStatePod* state,
    std::int32_t player,
    OfficialArea from_area,
    std::uint16_t from_index,
    OfficialArea to_area,
    bool reverse = false,
    bool log_semantic_move = true,
    std::int32_t open_type = 0) {
    OfficialCardRefPod ref = official_pod_remove_zone_card(state, player, from_area, from_index);
    if (!official_pod_ok(state)) return {};
    if (!official_pod_push_zone_card(state, player, to_area, ref)) return {};
    const OfficialArea stored_area = to_area == OfficialArea::kDeckBottom
        ? OfficialArea::kDeck
        : to_area;
    official_pod_card_moved(state, ref, stored_area, reverse);
    if (log_semantic_move) {
        const OfficialCardStatePod* moved = official_pod_card(state, ref);
        if (reverse || moved == nullptr || moved->card_id == 0) {
            official_semantic_history_append(
                state,
                OfficialSemanticLogType::kMoveCardReverse,
                player,
                moved == nullptr ? 0 : moved->card_id,
                ref.index,
                static_cast<std::int32_t>(from_area),
                static_cast<std::int32_t>(stored_area));
        } else {
            official_semantic_history_append(
                state,
                OfficialSemanticLogType::kMoveCard,
                player,
                moved->card_id,
                ref.index,
                static_cast<std::int32_t>(from_area),
                static_cast<std::int32_t>(stored_area),
                open_type);
        }
    }
    if (from_area == OfficialArea::kStadium) {
        state->last_stadium_player = static_cast<std::int8_t>(player);
    } else if (from_area == OfficialArea::kPrize
        && (stored_area == OfficialArea::kHand || stored_area == OfficialArea::kBench)
        && state->phase == static_cast<std::uint8_t>(OfficialGamePhase::kMain)
        && (((state->turn + 1) ^ state->first_player) & 1) == player) {
        ++state->turn_histories[0].take_prize_count;
    }
    return ref;
}

template <std::size_t Capacity>
PTCG_OFFICIAL_CORE_HD inline std::int32_t official_pod_find_in_list(
    const OfficialPodList<OfficialCardRefPod, Capacity>& list,
    OfficialCardRefPod ref) {
    for (std::uint16_t index = 0; index < list.count; ++index) {
        if (list.values[index] == ref) return index;
    }
    return -1;
}

PTCG_OFFICIAL_CORE_HD inline std::int32_t official_pod_find_zone_index(
    OfficialStatePod* state,
    std::int32_t player,
    OfficialArea area,
    OfficialCardRefPod ref) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return -1;
    }
    const OfficialPlayerStatePod& ps = state->players[player];
    std::int32_t result = -1;
    switch (area) {
        case OfficialArea::kDeck: result = official_pod_find_in_list(ps.deck, ref); break;
        case OfficialArea::kHand: result = official_pod_find_in_list(ps.hand, ref); break;
        case OfficialArea::kTrash: result = official_pod_find_in_list(ps.trash, ref); break;
        case OfficialArea::kActive: result = official_pod_find_in_list(ps.active, ref); break;
        case OfficialArea::kBench: result = official_pod_find_in_list(ps.bench, ref); break;
        case OfficialArea::kPrize: result = official_pod_find_in_list(ps.prize, ref); break;
        case OfficialArea::kEnergy: result = official_pod_find_in_list(ps.energy, ref); break;
        case OfficialArea::kTool: result = official_pod_find_in_list(ps.tool, ref); break;
        case OfficialArea::kPreEvolution:
            result = official_pod_find_in_list(ps.pre_evolution, ref); break;
        case OfficialArea::kTemporary:
            result = official_pod_find_in_list(ps.temporary, ref); break;
        case OfficialArea::kStadium:
            result = official_pod_find_in_list(state->stadium, ref); break;
        case OfficialArea::kLooking:
            result = official_pod_find_in_list(state->looking, ref); break;
        case OfficialArea::kPlaying:
            result = official_pod_find_in_list(state->playing, ref); break;
        default:
            official_pod_fail(state, OfficialPodError::kInvalidArea, static_cast<int>(area));
            return -1;
    }
    if (result < 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAreaIndex, ref.index);
    }
    return result;
}

PTCG_OFFICIAL_CORE_HD inline OfficialCardRefPod official_pod_move_ref(
    OfficialStatePod* state,
    OfficialCardRefPod ref,
    OfficialArea to_area,
    bool reverse = false,
    bool log_semantic_move = true,
    std::int32_t open_type = 0) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr) return {};
    const std::int32_t player = card->player;
    const OfficialArea from_area = static_cast<OfficialArea>(card->area);
    const std::int32_t index = official_pod_find_zone_index(
        state, player, from_area, ref);
    if (index < 0) return {};
    return official_pod_move_card(
        state,
        player,
        from_area,
        static_cast<std::uint16_t>(index),
        to_area,
        reverse,
        log_semantic_move,
        open_type);
}

template <std::size_t Capacity>
PTCG_OFFICIAL_CORE_HD inline void official_pod_move_matching_attachments(
    OfficialStatePod* state,
    std::int32_t player,
    OfficialArea from_area,
    OfficialPodList<OfficialCardRefPod, Capacity>* list,
    std::int32_t attach_move_counter,
    OfficialArea to_area) {
    for (std::int32_t index = static_cast<std::int32_t>(list->count) - 1;
         index >= 0 && official_pod_ok(state);
         --index) {
        const OfficialCardRefPod ref = list->values[index];
        OfficialCardStatePod* attached = official_pod_card(state, ref);
        if (attached != nullptr && attached->attach_move_counter == attach_move_counter) {
            official_pod_move_card(
                state,
                player,
                from_area,
                static_cast<std::uint16_t>(index),
                to_area,
                false);
        }
    }
}

PTCG_OFFICIAL_CORE_HD inline OfficialCardRefPod official_pod_move_ref_complete(
    OfficialStatePod* state,
    OfficialCardRefPod ref,
    OfficialArea to_area,
    bool reverse = false,
    bool with_attachments = false,
    std::int32_t open_type = 0) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr) return {};
    const OfficialArea from_area = static_cast<OfficialArea>(card->area);
    const std::int32_t player = card->player;
    const std::int32_t attach_move_counter = card->move_counter;
    const bool leaves_in_play =
        (from_area == OfficialArea::kActive && to_area != OfficialArea::kBench)
        || (from_area == OfficialArea::kBench && to_area != OfficialArea::kActive);
    const OfficialCardRefPod moved = official_pod_move_ref(
        state, ref, to_area, reverse, true, open_type);
    if (!official_pod_ok(state) || !leaves_in_play) return moved;

    OfficialPlayerStatePod* ps = &state->players[player];
    const OfficialArea pre_evolution_destination = with_attachments
        ? to_area
        : (to_area == OfficialArea::kHand ? OfficialArea::kHand : OfficialArea::kTrash);
    const OfficialArea attached_destination = with_attachments
        ? to_area
        : OfficialArea::kTrash;
    official_pod_move_matching_attachments(
        state,
        player,
        OfficialArea::kPreEvolution,
        &ps->pre_evolution,
        attach_move_counter,
        pre_evolution_destination);
    official_pod_move_matching_attachments(
        state,
        player,
        OfficialArea::kEnergy,
        &ps->energy,
        attach_move_counter,
        attached_destination);
    official_pod_move_matching_attachments(
        state,
        player,
        OfficialArea::kTool,
        &ps->tool,
        attach_move_counter,
        attached_destination);
    return moved;
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_switch_active(
    OfficialStatePod* state,
    std::int32_t player,
    std::uint16_t bench_index) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return false;
    }
    OfficialPlayerStatePod* ps = &state->players[player];
    if (bench_index >= ps->bench.count) {
        official_pod_fail(state, OfficialPodError::kInvalidAreaIndex, bench_index);
        return false;
    }
    if (ps->active.count == 0) {
        const OfficialCardRefPod replacement = ps->bench.values[bench_index];
        official_pod_move_card(
            state, player, OfficialArea::kBench, bench_index, OfficialArea::kActive);
        OfficialCardStatePod* active_card = official_pod_card(state, replacement);
        if (active_card != nullptr
            && state->phase == static_cast<std::uint8_t>(
                OfficialGamePhase::kMain)) {
            official_pod_set_card_runtime_flag(active_card, kCardBenchToActive);
        }
        return official_pod_ok(state);
    }
    const OfficialCardRefPod active = ps->active.values[0];
    const OfficialCardRefPod bench = ps->bench.values[bench_index];
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kSwitch,
        player,
        official_pod_card_id_or_zero(state, active),
        active.index,
        official_pod_card_id_or_zero(state, bench),
        bench.index);
    ps->active.values[0] = bench;
    ps->bench.values[bench_index] = active;
    official_pod_clear_special_conditions(state, player);
    official_pod_card_moved(state, bench, OfficialArea::kActive);
    official_pod_card_moved(state, active, OfficialArea::kBench);
    OfficialCardStatePod* active_card = official_pod_card(state, bench);
    if (active_card != nullptr
        && state->phase == static_cast<std::uint8_t>(OfficialGamePhase::kMain)) {
        official_pod_set_card_runtime_flag(active_card, kCardBenchToActive);
    }
    return official_pod_ok(state);
}

template <typename T>
PTCG_OFFICIAL_CORE_HD inline void official_pod_swap(T* left, T* right) {
    const T temporary = *left;
    *left = *right;
    *right = temporary;
}

template <typename T>
PTCG_OFFICIAL_CORE_HD inline void official_pod_shuffle(
    T* values,
    std::uint32_t count,
    OfficialMt19937* rng) {
    if (count < 2) return;
    std::uint32_t i = 1;
    if ((count & 1U) == 0U) {
        const std::uint32_t position = official_uniform_below(rng, 2);
        official_pod_swap(&values[i++], &values[position]);
    }
    while (i < count) {
        const std::uint32_t first_range = i + 1;
        const std::uint32_t combined = official_uniform_below(
            rng, first_range * (first_range + 1));
        const std::uint32_t first = combined / (first_range + 1);
        const std::uint32_t second = combined % (first_range + 1);
        official_pod_swap(&values[i++], &values[first]);
        official_pod_swap(&values[i++], &values[second]);
    }
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_shuffle_deck(
    OfficialStatePod* state,
    std::int32_t player,
    bool log_semantic_shuffle = true) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return;
    }
    OfficialPlayerStatePod* ps = &state->players[player];
    official_pod_shuffle(ps->deck.values, ps->deck.count, &state->rng);
    if (log_semantic_shuffle) {
        official_semantic_history_append(
            state,
            OfficialSemanticLogType::kShuffle,
            player);
    }
    if (ps->deck.count > 0) official_pod_mark_changed(state);
}

PTCG_OFFICIAL_CORE_HD inline std::int32_t official_pod_draw(
    OfficialStatePod* state,
    std::int32_t player,
    std::int32_t requested) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return 0;
    }
    OfficialPlayerStatePod* ps = &state->players[player];
    std::int32_t drawn = 0;
    while (drawn < requested && ps->deck.count > 0 && official_pod_ok(state)) {
        const std::uint16_t top = static_cast<std::uint16_t>(ps->deck.count - 1);
        const OfficialCardRefPod ref = official_pod_move_card(
            state,
            player,
            OfficialArea::kDeck,
            top,
            OfficialArea::kHand,
            false,
            false);
        if (official_pod_ok(state)) {
            official_semantic_history_append(
                state,
                OfficialSemanticLogType::kDraw,
                player,
                official_pod_card_id_or_zero(state, ref),
                ref.index);
        }
        ++drawn;
    }
    if (drawn > 0) official_pod_mark_changed(state);
    return drawn;
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_coin(
    OfficialStatePod* state,
    std::int32_t player) {
    const bool head = (official_mt19937_next(&state->rng) % 2U) == 0U;
    if (head) ++state->coin_head_count;
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kCoin,
        player,
        head ? 1 : 0);
    return head;
}

PTCG_OFFICIAL_CORE_HD inline std::int32_t official_pod_max_hp(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr) return 0;
    const OfficialCardRule* master = official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    if (master == nullptr) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, card->card_id);
        return 0;
    }
    const std::int32_t hp = master->values[kCardHp] + card->hp_change;
    return hp > 0 ? hp : 0;
}

PTCG_OFFICIAL_CORE_HD inline std::int32_t official_pod_add_damage(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    std::int32_t damage,
    bool put_damage_counter = false) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr) return 0;
    if (damage <= 0) {
        official_semantic_history_append(
            state,
            OfficialSemanticLogType::kHpChange,
            card->player,
            card->card_id,
            ref.index,
            0,
            put_damage_counter ? 1 : 0);
        return 0;
    }
    const std::int32_t before = card->damage;
    const std::int32_t max_hp = official_pod_max_hp(state, rules, ref);
    card->damage += damage;
    if (before < max_hp && card->damage >= max_hp && max_hp > 0) {
        card->damage = max_hp;
        official_pod_set_card_runtime_flag(card, kCardKo);
    }
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kHpChange,
        card->player,
        card->card_id,
        ref.index,
        -damage,
        put_damage_counter ? 1 : 0);
    return card->damage - before;
}

PTCG_OFFICIAL_CORE_HD inline std::int32_t official_pod_heal(
    OfficialStatePod* state,
    OfficialCardRefPod ref,
    std::int32_t amount,
    bool record_turn_heal = false) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr || amount < 0) return 0;
    const std::int32_t healed = amount < card->damage ? amount : card->damage;
    card->damage -= healed;
    if (card->damage == 0) card->runtime_flags &= ~static_cast<std::uint64_t>(kCardKo);
    if (healed > 0) {
        official_pod_mark_changed(state);
        if (record_turn_heal) {
            official_pod_push(
                state,
                &state->turn_heal,
                ref,
                OfficialPodError::kTurnRecordOverflow);
        }
    }
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kHpChange,
        card->player,
        card->card_id,
        ref.index,
        healed,
        0);
    return healed;
}

PTCG_OFFICIAL_CORE_HD inline std::int8_t official_pod_poison_counter(
    const OfficialPlayerStatePod& player) {
    return static_cast<std::int8_t>(player.active_state & 0xffU);
}

PTCG_OFFICIAL_CORE_HD inline OfficialBadStatus official_pod_bad_status(
    const OfficialPlayerStatePod& player) {
    return static_cast<OfficialBadStatus>((player.active_state >> 8U) & 0xffU);
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_burned(
    const OfficialPlayerStatePod& player) {
    return ((player.active_state >> 16U) & 0xffU) != 0U;
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_set_poison_counter(
    OfficialPlayerStatePod* player,
    std::int8_t value) {
    player->active_state = (player->active_state & 0xffffff00U)
        | static_cast<std::uint8_t>(value);
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_set_bad_status(
    OfficialPlayerStatePod* player,
    OfficialBadStatus value) {
    player->active_state = (player->active_state & 0xffff00ffU)
        | (static_cast<std::uint32_t>(value) << 8U);
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_set_burned(
    OfficialPlayerStatePod* player,
    bool value) {
    player->active_state = (player->active_state & 0xff00ffffU)
        | (static_cast<std::uint32_t>(value ? 1U : 0U) << 16U);
}

PTCG_OFFICIAL_CORE_HD inline OfficialCardStatePod* official_pod_active_card(
    OfficialStatePod* state,
    std::int32_t player) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return nullptr;
    }
    if (state->players[player].active.count == 0) return nullptr;
    return official_pod_card(state, state->players[player].active.values[0]);
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_blocks_special_condition(
    const OfficialCardStatePod& card,
    OfficialBadStatus status) {
    if ((card.runtime_flags & kCardNoSpecialCondition) != 0) return true;
    if (status != OfficialBadStatus::kNone
        && (card.runtime_flags & kCardNoSleepParalyzeConfuse) != 0) return true;
    return status == OfficialBadStatus::kAsleep
        && (card.runtime_flags & kCardNoSleep) != 0;
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_poison(
    OfficialStatePod* state,
    std::int32_t player,
    std::int8_t counters = 1) {
    OfficialCardStatePod* active = official_pod_active_card(state, player);
    if (active == nullptr || official_pod_blocks_special_condition(
            *active, OfficialBadStatus::kNone)) return false;
    OfficialPlayerStatePod* ps = &state->players[player];
    if (official_pod_poison_counter(*ps) == counters) return false;
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kPoisoned,
        player,
        0,
        active->card_id,
        state->players[player].active.values[0].index);
    official_pod_set_poison_counter(ps, counters);
    official_pod_mark_changed(state);
    return true;
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_burn(
    OfficialStatePod* state,
    std::int32_t player) {
    OfficialCardStatePod* active = official_pod_active_card(state, player);
    if (active == nullptr || official_pod_blocks_special_condition(
            *active, OfficialBadStatus::kNone)) return false;
    OfficialPlayerStatePod* ps = &state->players[player];
    if (official_pod_burned(*ps)) return false;
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kBurned,
        player,
        0,
        active->card_id,
        state->players[player].active.values[0].index);
    official_pod_set_burned(ps, true);
    official_pod_mark_changed(state);
    return true;
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_set_status(
    OfficialStatePod* state,
    std::int32_t player,
    OfficialBadStatus status) {
    OfficialCardStatePod* active = official_pod_active_card(state, player);
    if (active == nullptr || official_pod_blocks_special_condition(*active, status)) return false;
    OfficialPlayerStatePod* ps = &state->players[player];
    if (official_pod_bad_status(*ps) == status) return false;
    OfficialSemanticLogType log_type = OfficialSemanticLogType::kAsleep;
    if (status == OfficialBadStatus::kParalyzed) {
        log_type = OfficialSemanticLogType::kParalyzed;
    } else if (status == OfficialBadStatus::kConfused) {
        log_type = OfficialSemanticLogType::kConfused;
    }
    official_semantic_history_append(
        state,
        log_type,
        player,
        0,
        active->card_id,
        state->players[player].active.values[0].index);
    official_pod_set_bad_status(ps, status);
    official_pod_mark_changed(state);
    return true;
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_clear_special_conditions(
    OfficialStatePod* state,
    std::int32_t player) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return false;
    }
    OfficialPlayerStatePod* ps = &state->players[player];
    const bool changed = ps->active_state != 0;
    const OfficialCardRefPod active_ref = ps->active.count > 0
        ? ps->active.values[0]
        : OfficialCardRefPod{};
    const OfficialCardStatePod* active = ps->active.count > 0
        ? official_pod_card(state, active_ref)
        : nullptr;
    if (active != nullptr) {
        const OfficialBadStatus status = official_pod_bad_status(*ps);
        if (status != OfficialBadStatus::kNone) {
            OfficialSemanticLogType log_type = OfficialSemanticLogType::kAsleep;
            if (status == OfficialBadStatus::kParalyzed) {
                log_type = OfficialSemanticLogType::kParalyzed;
            } else if (status == OfficialBadStatus::kConfused) {
                log_type = OfficialSemanticLogType::kConfused;
            }
            official_semantic_history_append(
                state, log_type, player, 1, active->card_id, active_ref.index);
        }
        if (official_pod_poison_counter(*ps) > 0) {
            official_semantic_history_append(
                state,
                OfficialSemanticLogType::kPoisoned,
                player,
                1,
                active->card_id,
                active_ref.index);
        }
        if (official_pod_burned(*ps)) {
            official_semantic_history_append(
                state,
                OfficialSemanticLogType::kBurned,
                player,
                1,
                active->card_id,
                active_ref.index);
        }
    }
    ps->active_state = 0;
    // Official ClearSpecialCondition mutates the player's status fields without
    // touching State::changed. Effect callers that expose this as an instant
    // effect mark changed themselves; switching and continual refresh do not.
    return changed;
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_set_result(
    OfficialStatePod* state,
    std::int32_t losing_player,
    OfficialFinishReason reason) {
    if (losing_player == 0) {
        state->game_result = static_cast<std::uint8_t>(OfficialGameResult::kPlayer1Win);
    } else if (losing_player == 1) {
        state->game_result = static_cast<std::uint8_t>(OfficialGameResult::kPlayer0Win);
    } else {
        state->game_result = static_cast<std::uint8_t>(OfficialGameResult::kDraw);
    }
    state->finish_reason = static_cast<std::uint8_t>(reason);
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kResult,
        state->game_result,
        static_cast<std::int32_t>(reason));
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_finish_check(OfficialStatePod* state) {
    if (state->game_result != static_cast<std::uint8_t>(OfficialGameResult::kNone)) return true;
    OfficialFinishReason reason = OfficialFinishReason::kNone;
    std::int32_t score[2]{};
    const std::int32_t order[2] = {state->first_player, 1 - state->first_player};
    for (int index = 0; index < 2; ++index) {
        const int player = order[index];
        if (player < 0 || player > 1) continue;
        const OfficialPlayerStatePod& ps = state->players[player];
        if (ps.prize.count == 0) {
            ++score[player];
            reason = OfficialFinishReason::kPrize;
        }
        if (ps.active.count == 0 && ps.bench.count == 0) {
            ++score[1 - player];
            reason = OfficialFinishReason::kNoActivePokemon;
        }
    }
    if (reason != OfficialFinishReason::kNone) {
        int losing_player = 2;
        if (score[0] < score[1]) losing_player = 0;
        else if (score[0] > score[1]) losing_player = 1;
        official_pod_set_result(state, losing_player, reason);
    }
    return state->game_result != static_cast<std::uint8_t>(OfficialGameResult::kNone);
}

PTCG_OFFICIAL_CORE_HD inline bool official_pod_turn_start_draw(
    OfficialStatePod* state,
    std::int32_t player) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return false;
    }
    if (state->players[player].deck.count == 0) {
        official_pod_set_result(state, player, OfficialFinishReason::kDeck);
        return false;
    }
    return official_pod_draw(state, player, 1) == 1;
}

PTCG_OFFICIAL_CORE_HD inline std::int32_t official_pod_prize_count(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr) return 0;
    if ((card->runtime_flags & kCardKoPrizeZero) != 0) return 0;
    const OfficialCardRule* master = official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    if (master == nullptr) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, card->card_id);
        return 0;
    }
    std::int32_t count = 1;
    const std::int32_t pokemon_type = master->values[kCardPokemonType];
    if (pokemon_type == 3) count = 2;
    else if (pokemon_type == 4) count = 3;
    count += card->ko_prize_change_always;
    if ((card->runtime_flags & kCardKoEnemyAttackDamage) != 0) {
        count += card->ko_prize_change;
        if ((card->runtime_flags & kCardKoPrizeDecreaseOnce) != 0) --count;
    }
    if ((card->runtime_flags & kCardKoPrizePlus1) != 0) ++count;
    return count > 0 ? count : 0;
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_queue_ko_prizes(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr || card->player < 0 || card->player > 1) return;
    const OfficialCardRule* master = official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    if (master == nullptr) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, card->card_id);
        return;
    }
    if ((master->flags & kOfficialCardNoPrizeFlag) == 0) {
        state->pending_prize_count[1 - card->player] += static_cast<std::int16_t>(
            official_pod_prize_count(state, rules, ref));
    }
}

PTCG_OFFICIAL_CORE_HD inline void official_pod_reset(
    OfficialStatePod* state,
    std::uint64_t episode_id,
    std::uint64_t seed) {
    *state = OfficialStatePod{};
    state->episode_id = episode_id;
    state->first_player = -1;
    state->looking_player = -1;
    state->select_player = -1;
    state->last_stadium_player = 0;
    state->players[0].player = 0;
    state->players[1].player = 1;
    state->move_counter = 1;
    official_seed_mt19937(&state->rng, seed);
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_CORE_HD
