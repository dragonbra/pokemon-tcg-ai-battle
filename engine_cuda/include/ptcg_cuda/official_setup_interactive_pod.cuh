#pragma once

#include <cstdint>

#include "ptcg_cuda/official_continuation_pod.cuh"
#include "ptcg_cuda/official_seeded_setup_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_SETUP_VM_HD __host__ __device__
#else
#define PTCG_OFFICIAL_SETUP_VM_HD
#endif

namespace ptcg::cuda_engine {

constexpr std::uint8_t kOfficialSetupContextActive = 2;
constexpr std::uint8_t kOfficialSetupContextBench = 3;
constexpr std::uint8_t kOfficialSetupContextDrawCount = 39;
constexpr std::uint8_t kOfficialSetupContextIsFirst = 42;
constexpr std::uint8_t kOfficialSetupContextMulligan = 43;
constexpr std::int32_t kOfficialSetupBenchCapacity = 5;
constexpr std::int32_t kOfficialSetupVmBudget = 512;

template <typename T, std::size_t Capacity>
PTCG_OFFICIAL_SETUP_VM_HD inline void official_setup_clear_tail(
    OfficialPodList<T, Capacity>* list) {
    for (std::size_t index = list->count; index < Capacity; ++index) {
        list->values[index] = T{};
    }
}

PTCG_OFFICIAL_SETUP_VM_HD inline void official_setup_normalize_boundary(
    OfficialStatePod* state) {
    for (std::int32_t player = 0; player < 2; ++player) {
        OfficialPlayerStatePod& ps = state->players[player];
        official_setup_clear_tail(&ps.active);
        official_setup_clear_tail(&ps.bench);
        official_setup_clear_tail(&ps.prize);
        official_setup_clear_tail(&ps.hand);
        official_setup_clear_tail(&ps.deck);
        official_setup_clear_tail(&ps.trash);
        official_setup_clear_tail(&ps.energy);
        official_setup_clear_tail(&ps.tool);
        official_setup_clear_tail(&ps.pre_evolution);
        official_setup_clear_tail(&ps.temporary);
    }
    official_setup_clear_tail(&state->options);
    official_setup_clear_tail(&state->selected);
    official_setup_clear_tail(&state->pre_targets);
    official_setup_clear_tail(&state->targets);
    official_setup_clear_tail(&state->ko_list);
    official_setup_clear_tail(&state->effect_stack);
    official_setup_clear_tail(&state->continuations);
    official_setup_clear_tail(&state->effect_ref_scratch);
}

PTCG_OFFICIAL_SETUP_VM_HD inline void official_setup_clear_selection(
    OfficialStatePod* state) {
    state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kNone);
    state->select_deck = 0;
    state->context_card = {};
    state->options.count = 0;
    state->selected.count = 0;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_begin_selection(
    OfficialStatePod* state,
    OfficialSelectTypeId type,
    std::uint8_t context,
    std::int32_t player,
    std::int32_t minimum = 1,
    std::int32_t maximum = 1) {
    if (state == nullptr || player < 0 || player > 1
        || minimum < 0 || maximum < minimum
        || minimum > 0x7fff || maximum > 0x7fff) {
        if (state != nullptr) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, player);
        }
        return false;
    }
    state->select_type = static_cast<std::uint8_t>(type);
    state->select_context = context;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = static_cast<std::int16_t>(minimum);
    state->select_max = static_cast<std::int16_t>(maximum);
    state->options.count = 0;
    state->selected.count = 0;
    return true;
}

PTCG_OFFICIAL_SETUP_VM_HD inline OfficialSelectOptionPod*
official_setup_add_option(
    OfficialStatePod* state,
    OfficialSelectOptionTypeId type,
    OfficialCardRefPod resolved = {}) {
    if (state->options.count >= state->options.capacity()) {
        official_pod_fail(
            state, OfficialPodError::kOptionOverflow, state->options.count);
        return nullptr;
    }
    OfficialSelectOptionPod& option = state->options.values[state->options.count++];
    option = {};
    option.type = static_cast<std::uint8_t>(type);
    option.resolved_card = resolved.index;
    if (resolved.index != 0) {
        const OfficialCardStatePod* card = official_pod_card(state, resolved);
        option.option_equiv = card == nullptr
            ? resolved.index
            : static_cast<std::uint16_t>(card->card_id);
    }
    return &option;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_add_yes_no(
    OfficialStatePod* state) {
    return official_setup_add_option(
               state, OfficialSelectOptionTypeId::kYes) != nullptr
        && official_setup_add_option(
               state, OfficialSelectOptionTypeId::kNo) != nullptr;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_add_number(
    OfficialStatePod* state,
    std::int32_t number) {
    OfficialSelectOptionPod* option = official_setup_add_option(
        state, OfficialSelectOptionTypeId::kNumber);
    if (option == nullptr || number < -0x8000 || number > 0x7fff) return false;
    option->params[0] = static_cast<std::int16_t>(number);
    return true;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_add_hand_card(
    OfficialStatePod* state,
    std::int32_t player,
    std::uint16_t hand_index) {
    if (player < 0 || player > 1
        || hand_index >= state->players[player].hand.count) {
        official_pod_fail(
            state, OfficialPodError::kInvalidAreaIndex, hand_index);
        return false;
    }
    const OfficialCardRefPod ref = state->players[player].hand.values[hand_index];
    OfficialSelectOptionPod* option = official_setup_add_option(
        state, OfficialSelectOptionTypeId::kCard, ref);
    if (option == nullptr) return false;
    option->params[0] = static_cast<std::int16_t>(OfficialArea::kHand);
    option->params[1] = static_cast<std::int16_t>(hand_index);
    option->params[2] = static_cast<std::int16_t>(player);
    return true;
}

PTCG_OFFICIAL_SETUP_VM_HD inline OfficialFlowStatus
official_setup_yield(OfficialStatePod* state) {
    const std::int32_t option_count = state->options.count;
    if (state->select_max > option_count) {
        state->select_max = static_cast<std::int16_t>(option_count);
    }
    if (state->select_min > state->select_max) {
        state->select_min = state->select_max;
    }
    ++state->turn_action_count;
    official_setup_normalize_boundary(state);
    return OfficialFlowStatus::kNeedsAction;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_validate_action(
    OfficialStatePod* state,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (option_indices == nullptr
        || count < static_cast<std::uint16_t>(state->select_min)
        || count > static_cast<std::uint16_t>(state->select_max)
        || count > kOfficialOptionCapacity) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        return false;
    }
    state->selected.count = 0;
    for (std::uint16_t index = 0; index < count; ++index) {
        const std::uint16_t selected = option_indices[index];
        if (selected >= state->options.count) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, selected);
            return false;
        }
        for (std::uint16_t previous = 0; previous < index; ++previous) {
            if (option_indices[previous] == selected) {
                official_pod_fail(
                    state, OfficialPodError::kInvalidAction, selected);
                return false;
            }
        }
        state->selected.values[state->selected.count++] = selected;
    }
    return true;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_selected_yes(
    OfficialStatePod* state,
    bool* result) {
    if (state->selected.count != 1 || result == nullptr) return false;
    const OfficialSelectOptionPod& option = state->options.values[
        state->selected.values[0]];
    if (option.type == static_cast<std::uint8_t>(
            OfficialSelectOptionTypeId::kYes)) {
        *result = true;
        return true;
    }
    if (option.type == static_cast<std::uint8_t>(
            OfficialSelectOptionTypeId::kNo)) {
        *result = false;
        return true;
    }
    return false;
}

PTCG_OFFICIAL_SETUP_VM_HD inline void official_setup_set_mulligan(
    OfficialStatePod* state,
    std::int32_t player,
    bool value) {
    const std::uint8_t bit = static_cast<std::uint8_t>(1U << player);
    if (value) {
        state->mulligan_mask |= bit;
    } else {
        state->mulligan_mask &= static_cast<std::uint8_t>(~bit);
    }
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_deck_has_basic(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    const auto& deck = state->players[player].deck;
    for (std::uint16_t index = 0; index < deck.count; ++index) {
        const OfficialCardStatePod* card = official_pod_card(
            state, deck.values[index]);
        const OfficialCardRule* master = card == nullptr ? nullptr
            : official_card_rule(rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) return false;
        if (master->values[kCardType] == 0
            && master->values[kCardEvolutionType] == 1) return true;
    }
    return false;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_pre_active(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    const OfficialSeededSetupPresence presence = official_seeded_setup_presence(
        state, rules, player);
    if (!official_pod_ok(state)) return false;
    if (presence.basic) {
        official_setup_set_mulligan(state, player, false);
        return true;
    }
    if (!presence.doll) {
        official_setup_set_mulligan(state, player, true);
        return true;
    }
    if (!official_setup_begin_selection(
            state,
            OfficialSelectTypeId::kYesNo,
            kOfficialSetupContextMulligan,
            player)
        || !official_setup_add_yes_no(state)
        || !official_push_continuation(
            state,
            OfficialContinuationId::kSelectedMulligan,
            1,
            player)) {
        return false;
    }
    return true;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_active(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    const OfficialSeededSetupPresence presence = official_seeded_setup_presence(
        state, rules, player);
    if (!official_pod_ok(state)) return false;
    if (!presence.basic && !presence.doll) {
        official_setup_set_mulligan(state, player, true);
    }
    if ((state->mulligan_mask & (1U << player)) != 0) {
        if (!presence.basic && !official_setup_deck_has_basic(
                state, rules, player)) {
            official_pod_fail(
                state,
                OfficialPodError::kInvalidAction,
                static_cast<std::int32_t>(
                    OfficialSeededSetupError::kNoBasicPokemon));
            return false;
        }
        return true;
    }
    if (!official_setup_begin_selection(
            state,
            OfficialSelectTypeId::kCard,
            kOfficialSetupContextActive,
            player)) {
        return false;
    }
    const auto& hand = state->players[player].hand;
    for (std::uint16_t index = 0; index < hand.count; ++index) {
        const OfficialCardStatePod* card = official_pod_card(
            state, hand.values[index]);
        const OfficialCardRule* master = card == nullptr ? nullptr
            : official_card_rule(rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) return false;
        if (official_seeded_card_can_setup_active(*master)
            && !official_setup_add_hand_card(state, player, index)) {
            return false;
        }
    }
    return state->options.count > 0
        && official_push_continuation(
            state,
            OfficialContinuationId::kSelectedSetupActivePokemon,
            1,
            player);
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_resetup_active(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    if (state->mulligan_count[player]
        < kOfficialSeededDeckSize
            - kOfficialSeededFirstHand
            - kOfficialSeededPrizeSize) {
        ++state->mulligan_count[player];
    }
    if (!official_seeded_return_hand_and_draw(state, player)
        || !official_push_continuation(
            state,
            OfficialContinuationId::kAfterResetupActivePokemon,
            1,
            player)
        || !official_push_continuation(
            state,
            OfficialContinuationId::kSetupActivePokemon,
            1,
            player)) {
        return false;
    }
    return official_setup_pre_active(state, rules, player);
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_both_active(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    const std::int32_t first = state->first_player;
    if (first < 0 || first > 1
        || !official_push_continuation(
            state, OfficialContinuationId::kAfterSetupActivePokemon)
        || !official_push_continuation(
            state,
            OfficialContinuationId::kSetupActivePokemon,
            1,
            1 - first)
        || !official_push_continuation(
            state,
            OfficialContinuationId::kSetupActivePokemon,
            1,
            first)
        || !official_push_continuation(
            state,
            OfficialContinuationId::kPreSetupActivePokemon,
            1,
            1 - first)) {
        return false;
    }
    return official_setup_pre_active(state, rules, first);
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_after_active(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    const bool done0 = (state->setup_done_mask & 1U) != 0;
    const bool done1 = (state->setup_done_mask & 2U) != 0;
    if (done0 && done1) {
        const std::int32_t order[2] = {
            state->first_player, 1 - state->first_player};
        return official_seeded_setup_prizes(state, order[0])
            && official_seeded_setup_prizes(state, order[1]);
    }
    if (done0) {
        return official_seeded_setup_prizes(state, 0)
            && official_setup_resetup_active(state, rules, 1);
    }
    if (done1) {
        return official_seeded_setup_prizes(state, 1)
            && official_setup_resetup_active(state, rules, 0);
    }
    const std::int32_t order[2] = {
        state->first_player, 1 - state->first_player};
    if (!official_seeded_return_hand_and_draw(state, order[0])
        || !official_seeded_return_hand_and_draw(state, order[1])) {
        return false;
    }
    return official_setup_both_active(state, rules);
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_start_bench(
    OfficialStatePod* state) {
    const std::int32_t first = state->first_player;
    if (!official_push_continuation(
            state,
            OfficialContinuationId::kSetupBenchPokemon,
            1,
            1 - first)
        || !official_push_continuation(
            state,
            OfficialContinuationId::kSetupBenchPokemon,
            1,
            first)) {
        return false;
    }
    const std::int32_t order[2] = {first, 1 - first};
    for (std::int32_t ordinal = 0; ordinal < 2; ++ordinal) {
        const std::int32_t mulligan_player = order[ordinal];
        const std::int32_t count = state->mulligan_count[mulligan_player];
        if (count <= 0) continue;
        const std::int32_t select_player = 1 - mulligan_player;
        if (!official_setup_begin_selection(
                state,
                OfficialSelectTypeId::kCount,
                kOfficialSetupContextDrawCount,
                select_player)) {
            return false;
        }
        for (std::int32_t value = 0; value <= count; ++value) {
            if (!official_setup_add_number(state, value)) return false;
        }
        if (!official_push_continuation(
                state,
                OfficialContinuationId::kSelectedDrawCount,
                1,
                select_player)) {
            return false;
        }
    }
    return true;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_begin_bench(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    if (!official_setup_begin_selection(
            state,
            OfficialSelectTypeId::kCard,
            kOfficialSetupContextBench,
            player,
            0,
            0)) {
        return false;
    }
    const auto& hand = state->players[player].hand;
    for (std::uint16_t index = 0; index < hand.count; ++index) {
        const OfficialCardStatePod* card = official_pod_card(
            state, hand.values[index]);
        const OfficialCardRule* master = card == nullptr ? nullptr
            : official_card_rule(rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) return false;
        if (official_seeded_card_can_setup_bench(*master)
            && !official_setup_add_hand_card(state, player, index)) {
            return false;
        }
    }
    const std::int32_t remaining = kOfficialSetupBenchCapacity
        - state->players[player].bench.count;
    const std::int32_t maximum = state->options.count < remaining
        ? state->options.count : remaining;
    state->select_max = static_cast<std::int16_t>(maximum);
    return official_push_continuation(
        state,
        OfficialContinuationId::kSelectedSetupBenchPokemon,
        1,
        player);
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_selected_targets(
    OfficialStatePod* state) {
    state->targets.count = 0;
    for (std::uint16_t index = 0; index < state->selected.count; ++index) {
        const OfficialSelectOptionPod& option = state->options.values[
            state->selected.values[index]];
        if (option.type != static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kCard)
            || option.resolved_card == 0
            || state->targets.count >= state->targets.capacity()) {
            return false;
        }
        const OfficialCardRefPod ref{option.resolved_card};
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        if (card == nullptr) return false;
        OfficialAreaRefPod& target = state->targets.values[state->targets.count++];
        target = {};
        target.card = ref;
        target.move_counter = card->move_counter;
    }
    official_setup_clear_selection(state);
    return true;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_move_targets_to_bench(
    OfficialStatePod* state,
    std::int32_t player) {
    for (std::uint16_t index = 0; index < state->targets.count; ++index) {
        const OfficialCardRefPod ref = state->targets.values[index].card;
        const std::int32_t hand_index = official_pod_find_in_list(
            state->players[player].hand, ref);
        if (hand_index < 0) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAreaIndex, ref.index);
            return false;
        }
        official_pod_move_card(
            state,
            player,
            OfficialArea::kHand,
            static_cast<std::uint16_t>(hand_index),
            OfficialArea::kBench,
            true);
        if (!official_pod_ok(state)) return false;
    }
    return true;
}

PTCG_OFFICIAL_SETUP_VM_HD inline bool official_setup_finish_bench(
    OfficialStatePod* state,
    std::int32_t player) {
    if (player == state->first_player) {
        return official_setup_selected_targets(state);
    }
    if (!official_setup_move_targets_to_bench(state, state->first_player)
        || !official_setup_selected_targets(state)
        || !official_setup_move_targets_to_bench(state, player)) {
        return false;
    }
    state->setup_done_mask = 0;
    const std::int32_t order[2] = {
        state->first_player, 1 - state->first_player};
    for (std::int32_t ordinal = 0; ordinal < 2; ++ordinal) {
        OfficialPlayerStatePod& ps = state->players[order[ordinal]];
        for (std::uint16_t index = 0; index < ps.active.count; ++index) {
            OfficialCardStatePod* card = official_pod_card(
                state, ps.active.values[index]);
            if (card == nullptr) return false;
            card->reverse = 0;
        }
        for (std::uint16_t index = 0; index < ps.bench.count; ++index) {
            OfficialCardStatePod* card = official_pod_card(
                state, ps.bench.values[index]);
            if (card == nullptr) return false;
            card->reverse = 0;
        }
    }
    OfficialCardStatePod* first_active = official_pod_card(
        state, state->players[state->first_player].active.values[0]);
    OfficialCardStatePod* other_active = official_pod_card(
        state, state->players[1 - state->first_player].active.values[0]);
    if (first_active == nullptr || other_active == nullptr) return false;
    if (first_active->move_counter > other_active->move_counter) {
        official_pod_swap(
            &first_active->move_counter, &other_active->move_counter);
    }
    return official_finish_turn_start(state) == OfficialTurnFlowResult::kComplete
        && official_pod_ok(state);
}

PTCG_OFFICIAL_SETUP_VM_HD inline OfficialFlowStatus
official_setup_dispatch_no_action(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    const OfficialContinuationCall call = official_begin_continuation_call(state);
    if (!call.valid) return OfficialFlowStatus::kError;
    bool complete = false;
    switch (static_cast<OfficialContinuationId>(call.frame.opcode)) {
        case OfficialContinuationId::kPreSetupActivePokemon:
            complete = official_setup_pre_active(
                state, rules, call.frame.args[0]);
            break;
        case OfficialContinuationId::kSetupActivePokemon:
            complete = official_setup_active(
                state, rules, call.frame.args[0]);
            break;
        case OfficialContinuationId::kAfterSetupActivePokemon:
            complete = official_setup_after_active(state, rules);
            break;
        case OfficialContinuationId::kAfterResetupActivePokemon:
            complete = (state->setup_done_mask & (1U << call.frame.args[0])) != 0
                ? official_seeded_setup_prizes(state, call.frame.args[0])
                : official_setup_resetup_active(
                    state, rules, call.frame.args[0]);
            break;
        case OfficialContinuationId::kBothSetupActivePokemon:
            complete = official_setup_both_active(state, rules);
            break;
        case OfficialContinuationId::kStartSetupBench:
            complete = official_setup_start_bench(state);
            break;
        case OfficialContinuationId::kSetupBenchPokemon:
            complete = official_setup_begin_bench(
                state, rules, call.frame.args[0]);
            break;
        default:
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                call.frame.opcode);
            return OfficialFlowStatus::kError;
    }
    if (!complete || !official_finish_continuation_call(state, call)) {
        return OfficialFlowStatus::kError;
    }
    if (state->select_type != static_cast<std::uint8_t>(
            OfficialSelectTypeId::kNone)) {
        return official_setup_yield(state);
    }
    return OfficialFlowStatus::kIdle;
}

PTCG_OFFICIAL_SETUP_VM_HD inline OfficialFlowStatus official_setup_apply_action(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count);

PTCG_OFFICIAL_SETUP_VM_HD inline OfficialFlowStatus official_setup_advance(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    for (std::int32_t step = 0; step < kOfficialSetupVmBudget; ++step) {
        if (!official_pod_ok(state)) return OfficialFlowStatus::kError;
        if (state->phase != static_cast<std::uint8_t>(OfficialGamePhase::kSetup)) {
            const OfficialMainResult main = official_main_begin_refresh(state, rules);
            if (main == OfficialMainResult::kError) {
                return OfficialFlowStatus::kError;
            }
            if (main == OfficialMainResult::kNeedsAction) {
                if (state->select_type == static_cast<std::uint8_t>(
                        OfficialSelectTypeId::kMain)
                    && !official_push_continuation(
                        state, OfficialContinuationId::kSelectedMain)) {
                    return OfficialFlowStatus::kError;
                }
                state->continual_refresh_passes = 0;
                if (state->trigger_resolver.active == 0) {
                    state->trigger_resolver = OfficialTriggerResolverPod{};
                }
                if (state->effect_interpreter.active == 0) {
                    state->effect_interpreter = OfficialEffectInterpreterPod{};
                }
                if (state->trigger_resolver.active == 0
                    && state->effect_interpreter.active == 0) {
                    state->effect_state = OfficialEffectStatePod{};
                    state->trigger_info = OfficialTriggerInfoPod{};
                }
                ++state->turn_action_count;
                official_setup_normalize_boundary(state);
                return OfficialFlowStatus::kNeedsAction;
            }
            official_setup_normalize_boundary(state);
            return OfficialFlowStatus::kIdle;
        }
        if (state->select_type != static_cast<std::uint8_t>(
                OfficialSelectTypeId::kNone)) {
            return official_setup_yield(state);
        }
        if (state->continuations.count == 0) {
            official_pod_fail(
                state, OfficialPodError::kUnsupportedContinuation, -1001);
            return OfficialFlowStatus::kError;
        }
        const OfficialFlowStatus status = official_setup_dispatch_no_action(
            state, rules);
        if (status == OfficialFlowStatus::kNeedsAction
            && state->select_max == 0) {
            // ApiSelect automatically advances zero-cardinality selections;
            // they increment turnActionCount inside State::step but are never
            // surfaced to the policy.
            const std::uint16_t unused = 0;
            return official_setup_apply_action(
                state, rules, &unused, 0);
        }
        if (status != OfficialFlowStatus::kIdle) return status;
    }
    official_pod_fail(
        state, OfficialPodError::kInterpreterBudget, kOfficialSetupVmBudget);
    return OfficialFlowStatus::kError;
}

template <typename DeckT>
PTCG_OFFICIAL_SETUP_VM_HD inline OfficialFlowStatus
official_seeded_setup_interactive_state(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const DeckT* input_decks,
    std::uint64_t episode_id,
    std::uint64_t seed) {
    official_seeded_initialize_decks(
        state, rules, input_decks, episode_id, seed);
    if (!official_pod_ok(state)) return OfficialFlowStatus::kError;
    // A value-initialized official State exposes lookingPlayer=0 during setup.
    state->looking_player = 0;
    // ShuffleDeck marks the official state as changed.
    state->control_flags |= 1U;
    if (!official_setup_begin_selection(
            state,
            OfficialSelectTypeId::kYesNo,
            kOfficialSetupContextIsFirst,
            0)
        || !official_setup_add_yes_no(state)
        || !official_push_continuation(
            state, OfficialContinuationId::kSelectedIsFirst)) {
        return OfficialFlowStatus::kError;
    }
    return official_setup_yield(state);
}

PTCG_OFFICIAL_SETUP_VM_HD inline OfficialFlowStatus official_setup_apply_action(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (state == nullptr || !official_pod_ok(state)
        || state->phase != static_cast<std::uint8_t>(OfficialGamePhase::kSetup)
        || !official_setup_validate_action(state, option_indices, count)) {
        return OfficialFlowStatus::kError;
    }
    const OfficialContinuationCall call = official_begin_continuation_call(state);
    if (!call.valid) return OfficialFlowStatus::kError;
    bool complete = false;
    switch (static_cast<OfficialContinuationId>(call.frame.opcode)) {
        case OfficialContinuationId::kSelectedIsFirst: {
            bool yes = false;
            if (!official_setup_selected_yes(state, &yes)) break;
            const std::int32_t select_player = state->select_player;
            state->first_player = static_cast<std::int8_t>(
                yes ? select_player : 1 - select_player);
            official_setup_clear_selection(state);
            const std::int32_t order[2] = {
                state->first_player, 1 - state->first_player};
            if (official_pod_draw(
                    state, order[0], kOfficialSeededFirstHand)
                    != kOfficialSeededFirstHand
                || official_pod_draw(
                    state, order[1], kOfficialSeededFirstHand)
                    != kOfficialSeededFirstHand
                || !official_push_continuation(
                    state, OfficialContinuationId::kStartSetupBench)) {
                break;
            }
            complete = official_setup_both_active(state, rules);
            break;
        }
        case OfficialContinuationId::kSelectedMulligan: {
            bool yes = false;
            if (!official_setup_selected_yes(state, &yes)) break;
            official_setup_set_mulligan(state, call.frame.args[0], yes);
            official_setup_clear_selection(state);
            complete = true;
            break;
        }
        case OfficialContinuationId::kSelectedSetupActivePokemon: {
            if (state->selected.count != 1) break;
            const OfficialSelectOptionPod option = state->options.values[
                state->selected.values[0]];
            if (option.type != static_cast<std::uint8_t>(
                    OfficialSelectOptionTypeId::kCard)
                || option.params[0] != static_cast<std::int16_t>(
                    OfficialArea::kHand)
                || option.params[2] != call.frame.args[0]
                || option.params[1] < 0
                || option.params[1]
                    >= state->players[call.frame.args[0]].hand.count) {
                break;
            }
            official_setup_clear_selection(state);
            official_pod_move_card(
                state,
                call.frame.args[0],
                OfficialArea::kHand,
                static_cast<std::uint16_t>(option.params[1]),
                OfficialArea::kActive,
                true);
            if (!official_pod_ok(state)) break;
            state->setup_done_mask |= static_cast<std::uint8_t>(
                1U << call.frame.args[0]);
            complete = true;
            break;
        }
        case OfficialContinuationId::kSelectedDrawCount: {
            if (state->selected.count != 1) break;
            const OfficialSelectOptionPod option = state->options.values[
                state->selected.values[0]];
            if (option.type != static_cast<std::uint8_t>(
                    OfficialSelectOptionTypeId::kNumber)
                || option.params[0] < 0) {
                break;
            }
            const std::int32_t draw_count = option.params[0];
            official_setup_clear_selection(state);
            complete = official_pod_draw(
                    state, call.frame.args[0], draw_count)
                == draw_count;
            break;
        }
        case OfficialContinuationId::kSelectedSetupBenchPokemon:
            complete = official_setup_finish_bench(
                state, call.frame.args[0]);
            break;
        default:
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                call.frame.opcode);
            return OfficialFlowStatus::kError;
    }
    if (!complete || !official_finish_continuation_call(state, call)) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, call.frame.opcode);
        }
        return OfficialFlowStatus::kError;
    }
    return official_setup_advance(state, rules);
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_SETUP_VM_HD
