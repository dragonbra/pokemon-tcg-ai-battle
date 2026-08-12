#pragma once

#include <cstdint>

#include "ptcg_cuda/official_continual_refresh_pod.cuh"
#include "ptcg_cuda/official_knockout_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_TURN_HD __host__ __device__
#else
#define PTCG_OFFICIAL_TURN_HD
#endif

namespace ptcg::cuda_engine {

enum class OfficialTurnFlowResult : std::int32_t {
    kComplete = 0,
    kNeedsAction = 1,
    kError = 2,
};

enum class OfficialTurnFlowStage : std::uint8_t {
    kIdle = 0,
    kTurnEndTriggers = 1,
    kTurnEndRefresh = 2,
    kCheckupRefresh = 3,
    kCheckupTriggers = 4,
    kCheckupEndRefresh = 5,
    kTurnStart = 6,
};

enum class OfficialRefreshFlowStage : std::uint8_t {
    kIdle = 0,
    kStart = 1,
    kBenchOverflow = 2,
    kToolOverflow = 3,
    kKnockout = 4,
};

constexpr std::uint8_t kOfficialTriggerTurnEnd = 1;
constexpr std::uint8_t kOfficialTriggerPokemonCheckup = 2;
constexpr std::uint8_t kOfficialSelectContextDiscard = 9;
constexpr std::uint8_t kOfficialSelectContextDiscardTool = 28;
constexpr std::uint8_t kOfficialRefreshSkipContinualAfterBenchOverflowFlag = 1U << 3U;
constexpr std::uint64_t kOfficialCardTrashMyTurnEndFlag = 1ULL << 1;
constexpr std::uint64_t kOfficialCardOnlyTeamRocketFlag = 1ULL << 10;
constexpr std::uint64_t kOfficialCardTeamRocketFlag = 1ULL << 25;
constexpr std::uint32_t kOfficialTool2ContinualBit = 34;
constexpr std::uint32_t kOfficialTool4ContinualBit = 35;
constexpr std::int32_t kOfficialDefaultBenchSize = 5;

PTCG_OFFICIAL_TURN_HD inline OfficialTurnFlowResult official_advance_turn_flow(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

PTCG_OFFICIAL_TURN_HD inline OfficialTurnFlowResult official_advance_refresh(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

PTCG_OFFICIAL_TURN_HD inline std::int32_t official_bench_capacity(
    const OfficialPlayerStatePod& player) {
    const std::int32_t value = static_cast<std::int32_t>(
        (player.continual_state >> 40U) & 0xfULL);
    return value == 0 ? kOfficialDefaultBenchSize : value;
}

PTCG_OFFICIAL_TURN_HD inline std::int32_t official_tool_capacity(
    const OfficialCardStatePod& card) {
    if (official_continual_flag(card, kOfficialTool4ContinualBit)) return 4;
    if (official_continual_flag(card, kOfficialTool2ContinualBit)) return 2;
    return 1;
}

PTCG_OFFICIAL_TURN_HD inline std::int32_t official_attached_tool_count(
    const OfficialStatePod& state,
    const OfficialCardStatePod& pokemon) {
    if (pokemon.player < 0 || pokemon.player > 1) return 0;
    std::int32_t count = 0;
    const auto& tools = state.players[pokemon.player].tool;
    for (std::uint16_t index = 0; index < tools.count; ++index) {
        const OfficialCardStatePod* tool = official_pod_card(&state, tools.values[index]);
        if (tool != nullptr && tool->attach_move_counter == pokemon.move_counter) ++count;
    }
    return count;
}

PTCG_OFFICIAL_TURN_HD inline void official_prepare_flow_selection(
    OfficialStatePod* state,
    OfficialSelectTypeId type,
    std::uint8_t context,
    std::int32_t player,
    std::int32_t count) {
    official_clear_effect_selection(state);
    state->select_type = static_cast<std::uint8_t>(type);
    state->select_context = context;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = static_cast<std::int16_t>(count);
    state->select_max = static_cast<std::int16_t>(count);
}

PTCG_OFFICIAL_TURN_HD inline bool official_prepare_bench_overflow(
    OfficialStatePod* state) {
    std::int32_t order[2]{};
    if (state->last_stadium_player == 0 || state->last_stadium_player == 1) {
        order[0] = state->last_stadium_player;
        order[1] = 1 ^ state->last_stadium_player;
    } else {
        official_pod_fail(
            state, OfficialPodError::kInvalidAction, state->last_stadium_player);
        return false;
    }
    for (int ordinal = 0; ordinal < 2; ++ordinal) {
        const std::int32_t player = order[ordinal];
        if (player < 0 || player > 1) continue;
        const std::int32_t excess = static_cast<std::int32_t>(
            state->players[player].bench.count) - official_bench_capacity(
                state->players[player]);
        if (excess <= 0) continue;
        official_prepare_flow_selection(
            state, OfficialSelectTypeId::kCard, kOfficialSelectContextDiscard,
            player, excess);
        const auto& bench = state->players[player].bench;
        for (std::uint16_t index = 0; index < bench.count; ++index) {
            OfficialSelectOptionPod option{};
            option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kCard);
            option.params[0] = static_cast<std::int16_t>(OfficialArea::kBench);
            option.params[1] = static_cast<std::int16_t>(index);
            option.params[2] = static_cast<std::int16_t>(player);
            option.resolved_card = bench.values[index].index;
            const OfficialCardStatePod* card = official_pod_card(
                state, bench.values[index]);
            option.option_equiv = static_cast<std::uint16_t>(
                card == nullptr ? 0 : card->card_id);
            if (!official_pod_push(
                    state, &state->options, option, OfficialPodError::kOptionOverflow)) {
                return false;
            }
        }
        state->refresh_flow_stage = static_cast<std::uint8_t>(
            OfficialRefreshFlowStage::kBenchOverflow);
        state->control_flags |= 1U << 4U;
        return true;
    }
    return false;
}

template <std::size_t Capacity>
PTCG_OFFICIAL_TURN_HD inline bool official_prepare_tool_overflow_zone(
    OfficialStatePod* state,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone,
    std::int32_t player) {
    for (std::uint16_t pokemon_index = 0; pokemon_index < zone.count; ++pokemon_index) {
        const OfficialCardRefPod pokemon_ref = zone.values[pokemon_index];
        const OfficialCardStatePod* pokemon = official_pod_card(state, pokemon_ref);
        if (pokemon == nullptr) return false;
        const OfficialArea pokemon_area = static_cast<OfficialArea>(pokemon->area);
        const std::int32_t current_pokemon_index = pokemon_area == OfficialArea::kActive
            ? official_pod_find_in_list(state->players[player].active, pokemon_ref)
            : official_pod_find_in_list(state->players[player].bench, pokemon_ref);
        if (current_pokemon_index < 0) return false;
        const std::int32_t excess = official_attached_tool_count(*state, *pokemon)
            - official_tool_capacity(*pokemon);
        if (excess <= 0) continue;
        official_prepare_flow_selection(
            state, OfficialSelectTypeId::kAttachedCard,
            kOfficialSelectContextDiscardTool, player, excess);
        state->context_card = pokemon_ref;
        const auto& tools = state->players[player].tool;
        std::int16_t attached_index = 0;
        for (std::uint16_t index = 0; index < tools.count; ++index) {
            const OfficialCardRefPod tool_ref = tools.values[index];
            const OfficialCardStatePod* tool = official_pod_card(state, tool_ref);
            if (tool == nullptr || tool->attach_move_counter != pokemon->move_counter) continue;
            OfficialSelectOptionPod option{};
            option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kToolCard);
            option.params[0] = static_cast<std::int16_t>(pokemon_area);
            option.params[1] = static_cast<std::int16_t>(current_pokemon_index);
            option.params[2] = static_cast<std::int16_t>(player);
            option.params[3] = attached_index++;
            option.resolved_card = tool_ref.index;
            option.option_equiv = static_cast<std::uint16_t>(tool->card_id);
            if (!official_pod_push(
                    state, &state->options, option, OfficialPodError::kOptionOverflow)) {
                return false;
            }
        }
        state->refresh_flow_stage = static_cast<std::uint8_t>(
            OfficialRefreshFlowStage::kToolOverflow);
        state->control_flags |= 1U << 4U;
        return true;
    }
    return false;
}

PTCG_OFFICIAL_TURN_HD inline bool official_prepare_tool_overflow(
    OfficialStatePod* state) {
    const std::int32_t order[2] = {1 - state->first_player, state->first_player};
    for (int ordinal = 0; ordinal < 2; ++ordinal) {
        const std::int32_t player = order[ordinal];
        if (player < 0 || player > 1) continue;
        const auto& ps = state->players[player];
        OfficialPodList<OfficialCardRefPod, kOfficialBenchCapacity> reverse_bench{};
        for (std::int32_t index = static_cast<std::int32_t>(ps.bench.count) - 1;
             index >= 0;
             --index) {
            reverse_bench.values[reverse_bench.count++] = ps.bench.values[index];
        }
        if (official_prepare_tool_overflow_zone(state, reverse_bench, player)) return true;
        if (!official_pod_ok(state)) return false;
        if (official_prepare_tool_overflow_zone(state, ps.active, player)) return true;
        if (!official_pod_ok(state)) return false;
    }
    return false;
}

template <std::size_t Capacity>
PTCG_OFFICIAL_TURN_HD inline bool official_attached_to_team_rocket_zone(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone,
    std::int32_t attach_move_counter) {
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        const OfficialCardStatePod* pokemon = official_pod_card(state, zone.values[index]);
        if (pokemon == nullptr || pokemon->move_counter != attach_move_counter) continue;
        const OfficialCardRule* master = official_card_rule(
            rules, static_cast<std::uint32_t>(pokemon->card_id));
        return master != nullptr
            && (master->flags & kOfficialCardTeamRocketFlag) != 0;
    }
    return false;
}

PTCG_OFFICIAL_TURN_HD inline bool official_cleanup_team_rocket_energy(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    bool changed = false;
    for (int player = 0; player < 2; ++player) {
        auto& energy = state->players[player].energy;
        for (std::int32_t index = static_cast<std::int32_t>(energy.count) - 1;
             index >= 0 && official_pod_ok(state);
             --index) {
            const OfficialCardRefPod energy_ref = energy.values[index];
            const OfficialCardStatePod* card = official_pod_card(state, energy_ref);
            const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
                rules, static_cast<std::uint32_t>(card->card_id));
            if (master == nullptr) {
                official_pod_fail(
                    state, OfficialPodError::kRulePackBounds,
                    card == nullptr ? 0 : card->card_id);
                return false;
            }
            if ((master->flags & kOfficialCardOnlyTeamRocketFlag) == 0) continue;
            bool team_rocket = official_attached_to_team_rocket_zone(
                state, rules, state->players[player].active, card->attach_move_counter);
            if (!team_rocket) {
                team_rocket = official_attached_to_team_rocket_zone(
                    state, rules, state->players[player].bench,
                    card->attach_move_counter);
            }
            if (!team_rocket) {
                official_pod_move_card(
                    state, player, OfficialArea::kEnergy,
                    static_cast<std::uint16_t>(index), OfficialArea::kTrash, false);
                changed = true;
            }
        }
    }
    return changed;
}

template <std::size_t Capacity>
PTCG_OFFICIAL_TURN_HD inline bool official_zone_needs_knockout(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone) {
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        OfficialCardStatePod* card = official_pod_card(state, zone.values[index]);
        if (card == nullptr) return false;
        if ((card->runtime_flags & kCardKo) != 0) return true;
        const std::int32_t max_hp = official_pod_max_hp(state, rules, zone.values[index]);
        if (max_hp > 0 && card->damage >= max_hp) return true;
    }
    return false;
}

PTCG_OFFICIAL_TURN_HD inline bool official_state_needs_knockout_flow(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    for (int player = 0; player < 2; ++player) {
        const auto& ps = state->players[player];
        if (ps.active.count == 0 && ps.bench.count > 0) return true;
        if (official_zone_needs_knockout(state, rules, ps.active)
            || official_zone_needs_knockout(state, rules, ps.bench)) return true;
        if (!official_pod_ok(state)) return false;
    }
    return false;
}

PTCG_OFFICIAL_TURN_HD inline OfficialTurnFlowResult official_advance_refresh(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pod_ok(state)) return OfficialTurnFlowResult::kError;
    for (std::int32_t guard = 0; guard < 32; ++guard) {
        const auto stage = static_cast<OfficialRefreshFlowStage>(
            state->refresh_flow_stage);
        if (stage == OfficialRefreshFlowStage::kBenchOverflow
            || stage == OfficialRefreshFlowStage::kToolOverflow) {
            return OfficialTurnFlowResult::kNeedsAction;
        }
        if (stage == OfficialRefreshFlowStage::kKnockout) {
            if (state->reserved_flow != 0 || state->trigger_resolver.active != 0) {
                return OfficialTurnFlowResult::kNeedsAction;
            }
            // State::step stops as soon as ActiveCheck establishes a result;
            // pending stateChanged work is not allowed to start another
            // Refresh pass after the terminal boundary.
            if (state->game_result != static_cast<std::uint8_t>(
                    OfficialGameResult::kNone)) {
                state->refresh_flow_stage = 0;
                return OfficialTurnFlowResult::kComplete;
            }
            // KOProc2 returns to AfterRefresh.  Only KO or replacement work
            // marked stateChanged starts the next Refresh pass; the ordinary
            // no-KO path returns directly to the parent flow.
            if ((state->control_flags & (1U << 4U)) != 0) {
                state->control_flags &= static_cast<std::uint8_t>(~(1U << 4U));
                state->refresh_flow_stage = static_cast<std::uint8_t>(
                    OfficialRefreshFlowStage::kStart);
                continue;
            }
            official_pod_finish_check(state);
            state->refresh_flow_stage = 0;
            return official_pod_ok(state)
                ? OfficialTurnFlowResult::kComplete
                : OfficialTurnFlowResult::kError;
        }
        if (stage != OfficialRefreshFlowStage::kStart) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, state->refresh_flow_stage);
            return OfficialTurnFlowResult::kError;
        }
        if ((state->flow_flags
                & kOfficialRefreshSkipContinualAfterBenchOverflowFlag) != 0) {
            state->flow_flags &= static_cast<std::uint8_t>(
                ~kOfficialRefreshSkipContinualAfterBenchOverflowFlag);
        } else {
            const OfficialContinualRefreshResult continual =
                official_refresh_continual_effects(state, rules);
            if (continual != OfficialContinualRefreshResult::kApplied) {
                if (continual == OfficialContinualRefreshResult::kDepthLimit) {
                    official_pod_fail(state, OfficialPodError::kInterpreterBudget, 10);
                }
                return OfficialTurnFlowResult::kError;
            }
            if (official_prepare_bench_overflow(state)) {
                return official_pod_ok(state)
                    ? OfficialTurnFlowResult::kNeedsAction
                    : OfficialTurnFlowResult::kError;
            }
            if (!official_pod_ok(state)) return OfficialTurnFlowResult::kError;
        }
        if (official_prepare_tool_overflow(state)) {
            return official_pod_ok(state)
                ? OfficialTurnFlowResult::kNeedsAction
                : OfficialTurnFlowResult::kError;
        }
        if (!official_pod_ok(state)) return OfficialTurnFlowResult::kError;
        if (official_cleanup_team_rocket_energy(state, rules)) continue;
        if (!official_pod_ok(state)) return OfficialTurnFlowResult::kError;
        if (state->phase == static_cast<std::uint8_t>(
                OfficialGamePhase::kPokemonCheckup)) {
            state->refresh_flow_stage = 0;
            return OfficialTurnFlowResult::kComplete;
        }
        // Refresh always enters ToolCountProc -> KOProc on the official
        // engine, including when no Pokemon is currently knocked out.  That
        // KOProc resolves pending depth-1 triggers and prize selections before
        // AfterRefresh performs the finish check.  Checking for a winner here
        // is observably too early when an attack both removes its user from the
        // Active Spot and knocks out the opposing Active Pokemon.
        state->refresh_flow_stage = static_cast<std::uint8_t>(
            OfficialRefreshFlowStage::kKnockout);
        const OfficialKnockoutResult knockout = official_begin_knockout(state, rules);
        if (knockout == OfficialKnockoutResult::kError) {
            return OfficialTurnFlowResult::kError;
        }
        if (knockout == OfficialKnockoutResult::kNeedsAction) {
            return OfficialTurnFlowResult::kNeedsAction;
        }
    }
    official_pod_fail(state, OfficialPodError::kInterpreterBudget, 32);
    return OfficialTurnFlowResult::kError;
}

PTCG_OFFICIAL_TURN_HD inline OfficialTurnFlowResult official_begin_refresh(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pod_ok(state) || state->refresh_flow_stage != 0
        || state->reserved_flow != 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, state->refresh_flow_stage);
        }
        return OfficialTurnFlowResult::kError;
    }
    state->control_flags &= static_cast<std::uint8_t>(~(1U << 4U));
    state->refresh_flow_stage = static_cast<std::uint8_t>(
        OfficialRefreshFlowStage::kStart);
    return official_advance_refresh(state, rules);
}

PTCG_OFFICIAL_TURN_HD inline bool official_validate_flow_action(
    OfficialStatePod* state,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (count < state->select_min || count > state->select_max) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        return false;
    }
    for (std::uint16_t index = 0; index < count; ++index) {
        if (option_indices[index] >= state->options.count) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, option_indices[index]);
            return false;
        }
        for (std::uint16_t prior = 0; prior < index; ++prior) {
            if (option_indices[prior] == option_indices[index]) {
                official_pod_fail(
                    state, OfficialPodError::kInvalidAction, option_indices[index]);
                return false;
            }
        }
    }
    return true;
}

PTCG_OFFICIAL_TURN_HD inline OfficialTurnFlowResult official_resume_refresh(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (!official_pod_ok(state) || state->refresh_flow_stage == 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        }
        return OfficialTurnFlowResult::kError;
    }
    const auto stage = static_cast<OfficialRefreshFlowStage>(
        state->refresh_flow_stage);
    if (stage == OfficialRefreshFlowStage::kKnockout) {
        const OfficialKnockoutResult knockout = official_resume_knockout(
            state, rules, option_indices, count);
        if (knockout == OfficialKnockoutResult::kError) {
            return OfficialTurnFlowResult::kError;
        }
        if (knockout == OfficialKnockoutResult::kNeedsAction) {
            return OfficialTurnFlowResult::kNeedsAction;
        }
        return official_advance_refresh(state, rules);
    }
    if ((stage != OfficialRefreshFlowStage::kBenchOverflow
            && stage != OfficialRefreshFlowStage::kToolOverflow)
        || !official_validate_flow_action(state, option_indices, count)) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, state->refresh_flow_stage);
        }
        return OfficialTurnFlowResult::kError;
    }
    const std::int32_t player = state->select_player;
    for (std::uint16_t index = 0; index < count; ++index) {
        const OfficialCardRefPod ref{
            state->options.values[option_indices[index]].resolved_card};
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        const OfficialArea expected = stage == OfficialRefreshFlowStage::kBenchOverflow
            ? OfficialArea::kBench : OfficialArea::kTool;
        if (card == nullptr || card->player != player
            || card->area != static_cast<std::uint8_t>(expected)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, ref.index);
            return OfficialTurnFlowResult::kError;
        }
        if (stage == OfficialRefreshFlowStage::kToolOverflow) {
            const OfficialCardStatePod* pokemon = official_pod_card(
                state, state->context_card);
            if (pokemon == nullptr
                || card->attach_move_counter != pokemon->move_counter) {
                official_pod_fail(state, OfficialPodError::kInvalidAction, ref.index);
                return OfficialTurnFlowResult::kError;
            }
        }
    }
    state->targets.count = 0;
    for (std::uint16_t index = 0; index < count; ++index) {
        const OfficialCardRefPod ref{
            state->options.values[option_indices[index]].resolved_card};
        if (!official_pod_push(
                state,
                &state->targets,
                official_pod_area_ref(state, ref),
                OfficialPodError::kSelectionOverflow)) {
            return OfficialTurnFlowResult::kError;
        }
    }
    for (std::uint16_t index = 0; index < count; ++index) {
        const OfficialCardRefPod ref{
            state->options.values[option_indices[index]].resolved_card};
        official_pod_move_ref_complete(
            state, ref, OfficialArea::kTrash, false, false);
        if (!official_pod_ok(state)) return OfficialTurnFlowResult::kError;
    }
    official_clear_effect_selection(state);
    if (stage == OfficialRefreshFlowStage::kBenchOverflow) {
        // BenchCheck queues every overflowing player after one continual
        // refresh. SelectedBenchMaxTrash does not refresh effects between
        // those sibling selections, so preserve the current continual state
        // until the whole queued batch has been consumed.
        if (official_prepare_bench_overflow(state)) {
            return official_pod_ok(state)
                ? OfficialTurnFlowResult::kNeedsAction
                : OfficialTurnFlowResult::kError;
        }
        if (!official_pod_ok(state)) return OfficialTurnFlowResult::kError;
        state->flow_flags |= kOfficialRefreshSkipContinualAfterBenchOverflowFlag;
    }
    state->refresh_flow_stage = static_cast<std::uint8_t>(
        OfficialRefreshFlowStage::kStart);
    return official_advance_refresh(state, rules);
}

template <std::size_t Capacity>
PTCG_OFFICIAL_TURN_HD inline void official_card_turn_end_zone(
    OfficialStatePod* state,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone,
    std::int32_t active_player) {
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        OfficialCardStatePod* card = official_pod_card(state, zone.values[index]);
        if (card == nullptr) continue;
        for (int word = 0; word < 3; ++word) card->turn_state[word] = 0;
        // These runtime mirrors represent the official Card.turnState union;
        // they must not survive the turn boundary. Continual flags are kept.
        constexpr std::uint64_t kTurnScopedRuntimeFlags =
            kCardAppear
            | kCardEvolved
            | kCardBenchToActive
            | kCardKo
            | kCardKoAttackDamage
            | kCardKoEnemyAttackDamage
            | kCardKoEnemyAttackDamageActive
            | kCardKoEnemyExAttackDamage
            | kCardKoEnemyTerastalAttackDamage
            | kCardKoEnemyNAttackDamage
            | kCardKoFull
            | kCardKoPrizePlus1
            | kCardKoPrizeDecreaseOnce
            | kCardKoPrizeZero
            | kCardKoByDamageToHand;
        card->runtime_flags &= ~kTurnScopedRuntimeFlags;
        card->ability_used_count = 0;
        for (int index = 0; index < 8; ++index) card->ability_used[index] = 0;
        for (int word = 0; word < 4; ++word) card->this_turn[word] = 0;
        card->this_turn_enemy = 0;
        if (card->player != active_player) {
            card->next_enemy_turn_end = 0;
            card->next_enemy_turn_end_battlefield = 0;
        }
    }
}

PTCG_OFFICIAL_TURN_HD inline bool official_turn_end_cleanup(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t active_player) {
    state->turn_state = 0;
    official_card_turn_end_zone(state, state->stadium, active_player);
    const std::int32_t order[2] = {state->first_player, 1 - state->first_player};
    for (int ordinal = 0; ordinal < 2; ++ordinal) {
        const std::int32_t player = order[ordinal];
        if (player < 0 || player > 1) continue;
        auto& ps = state->players[player];
        ps.turn_state = 0;
        ps.this_turn = 0;
        official_card_turn_end_zone(state, ps.active, active_player);
        official_card_turn_end_zone(state, ps.bench, active_player);
    }
    auto& ps = state->players[active_player];
    if ((state->continual_state & 1U) == 0) {
        for (std::int32_t index = static_cast<std::int32_t>(ps.tool.count) - 1;
             index >= 0 && official_pod_ok(state);
             --index) {
            const OfficialCardRefPod ref = ps.tool.values[index];
            const OfficialCardStatePod* card = official_pod_card(state, ref);
            const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
                rules, static_cast<std::uint32_t>(card->card_id));
            if (master == nullptr) return false;
            if ((master->flags & kOfficialCardTrashMyTurnEndFlag) != 0) {
                official_pod_move_card(
                    state, active_player, OfficialArea::kTool,
                    static_cast<std::uint16_t>(index), OfficialArea::kTrash, false);
            }
        }
    }
    for (std::int32_t index = static_cast<std::int32_t>(ps.energy.count) - 1;
         index >= 0 && official_pod_ok(state);
         --index) {
        const OfficialCardRefPod ref = ps.energy.values[index];
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
            rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) return false;
        if ((master->flags & kOfficialCardTrashMyTurnEndFlag) != 0) {
            official_pod_move_card(
                state, active_player, OfficialArea::kEnergy,
                static_cast<std::uint16_t>(index), OfficialArea::kTrash, false);
        }
    }
    return official_pod_ok(state);
}

template <std::size_t Capacity>
PTCG_OFFICIAL_TURN_HD inline void official_card_turn_start_zone(
    OfficialStatePod* state,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone,
    std::int32_t active_player) {
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        OfficialCardStatePod* card = official_pod_card(state, zone.values[index]);
        if (card == nullptr) continue;
        card->take_attack_damage_pre_turn = card->take_attack_damage_this_turn;
        card->take_attack_damage_this_turn = 0;
        if (card->player == active_player) {
            for (int word = 0; word < 4; ++word) {
                card->this_turn[word] = card->next_turn[word];
                card->next_turn[word] = 0;
            }
        } else {
            card->this_turn_enemy = card->next_turn_enemy;
            card->next_turn_enemy = 0;
        }
    }
}

PTCG_OFFICIAL_TURN_HD inline OfficialTurnFlowResult official_finish_turn_start(
    OfficialStatePod* state) {
    if (official_pod_finish_check(state)) return OfficialTurnFlowResult::kComplete;
    ++state->turn;
    state->turn_action_count = 0;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    state->turn_used_skills.count = 0;
    state->turn_play.count = 0;
    state->turn_heal.count = 0;
    state->turn_evolve.count = 0;
    state->turn_histories[2] = state->turn_histories[1];
    state->turn_histories[1] = state->turn_histories[0];
    state->turn_histories[0] = OfficialTurnHistoryPod{};
    const std::int32_t active_player = official_active_player(*state);
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kTurnStart,
        active_player);
    official_card_turn_start_zone(state, state->stadium, active_player);
    const std::int32_t order[2] = {state->first_player, 1 - state->first_player};
    for (int ordinal = 0; ordinal < 2; ++ordinal) {
        const std::int32_t player = order[ordinal];
        if (player < 0 || player > 1) continue;
        auto& ps = state->players[player];
        if (player == active_player) {
            ps.this_turn = ps.next_turn;
            ps.next_turn = 0;
        }
        official_card_turn_start_zone(state, ps.active, active_player);
        official_card_turn_start_zone(state, ps.bench, active_player);
    }
    if (!official_pod_turn_start_draw(state, active_player)) {
        state->turn_flow_stage = 0;
        return official_pod_ok(state)
            ? OfficialTurnFlowResult::kComplete
            : OfficialTurnFlowResult::kError;
    }
    state->turn_flow_stage = 0;
    return OfficialTurnFlowResult::kComplete;
}

PTCG_OFFICIAL_TURN_HD inline bool official_queue_checkup_triggers(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pull_trigger(
            state, rules, kOfficialTriggerPokemonCheckup, OfficialCardRefPod{})) {
        return false;
    }
    for (int player = 0; player < 2; ++player) {
        const auto& ps = state->players[player];
        if (official_pod_bad_status(ps) == OfficialBadStatus::kAsleep
            || official_pod_bad_status(ps) == OfficialBadStatus::kParalyzed
            || official_pod_poison_counter(ps) > 0
            || official_pod_burned(ps)) {
            OfficialTriggeredAbilityPod special{};
            special.activate.is_special_condition = 1;
            return official_pod_push(
                state, &state->temporary_triggers, special,
                OfficialPodError::kTriggerStackOverflow);
        }
    }
    return true;
}

PTCG_OFFICIAL_TURN_HD inline OfficialTurnFlowResult official_advance_turn_flow(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pod_ok(state)) return OfficialTurnFlowResult::kError;
    while (true) {
        const auto stage = static_cast<OfficialTurnFlowStage>(state->turn_flow_stage);
        if (stage == OfficialTurnFlowStage::kTurnEndTriggers) {
            const std::int32_t active_player = official_active_player(*state);
            if (!official_turn_end_cleanup(state, rules, active_player)) {
                return OfficialTurnFlowResult::kError;
            }
            state->turn_flow_stage = static_cast<std::uint8_t>(
                OfficialTurnFlowStage::kTurnEndRefresh);
            const OfficialTurnFlowResult refresh = official_begin_refresh(state, rules);
            if (refresh != OfficialTurnFlowResult::kComplete) return refresh;
            continue;
        }
        if (stage == OfficialTurnFlowStage::kTurnEndRefresh) {
            state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kPokemonCheckup);
            if (official_pod_finish_check(state)) {
                state->turn_flow_stage = 0;
                return OfficialTurnFlowResult::kComplete;
            }
            if (!official_queue_checkup_triggers(state, rules)) {
                return OfficialTurnFlowResult::kError;
            }
            state->turn_flow_stage = static_cast<std::uint8_t>(
                OfficialTurnFlowStage::kCheckupRefresh);
            const OfficialTurnFlowResult refresh = official_begin_refresh(state, rules);
            if (refresh != OfficialTurnFlowResult::kComplete) return refresh;
            continue;
        }
        if (stage == OfficialTurnFlowStage::kCheckupRefresh) {
            state->turn_flow_stage = static_cast<std::uint8_t>(
                OfficialTurnFlowStage::kCheckupTriggers);
            const OfficialTriggerResolverResult trigger =
                official_begin_trigger_resolution(state, rules, 0);
            if (trigger == OfficialTriggerResolverResult::kError) {
                return OfficialTurnFlowResult::kError;
            }
            if (trigger == OfficialTriggerResolverResult::kNeedsAction) {
                return OfficialTurnFlowResult::kNeedsAction;
            }
            continue;
        }
        if (stage == OfficialTurnFlowStage::kCheckupTriggers) {
            state->phase = static_cast<std::uint8_t>(
                OfficialGamePhase::kPokemonCheckupEnd);
            state->turn_flow_stage = static_cast<std::uint8_t>(
                OfficialTurnFlowStage::kCheckupEndRefresh);
            const OfficialTurnFlowResult refresh = official_begin_refresh(state, rules);
            if (refresh != OfficialTurnFlowResult::kComplete) return refresh;
            continue;
        }
        if (stage == OfficialTurnFlowStage::kCheckupEndRefresh) {
            state->turn_flow_stage = static_cast<std::uint8_t>(
                OfficialTurnFlowStage::kTurnStart);
            continue;
        }
        if (stage == OfficialTurnFlowStage::kTurnStart) {
            return official_finish_turn_start(state);
        }
        official_pod_fail(
            state, OfficialPodError::kInvalidAction, state->turn_flow_stage);
        return OfficialTurnFlowResult::kError;
    }
}

PTCG_OFFICIAL_TURN_HD inline bool official_queue_turn_end_triggers(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t active_player) {
    const OfficialCardRefPod player_ref{
        static_cast<std::uint16_t>(1 + active_player)};
    for (std::int32_t index = static_cast<std::int32_t>(state->delay_triggers.count) - 1;
         index >= 0;
         --index) {
        const OfficialTriggeredAbilityPod pending = state->delay_triggers.values[index];
        const OfficialCardStatePod* effect_card = official_pod_card(
            state, pending.activate.effect_card.card);
        if (effect_card == nullptr) return false;
        const bool move = pending.trigger.type == kOfficialTriggerTurnEnd
            && (effect_card->player != active_player
                || pending.trigger.subject.card == player_ref);
        if (!move) continue;
        if (!official_pod_push(
                state, &state->temporary_triggers, pending,
                OfficialPodError::kTriggerStackOverflow)) return false;
        official_pod_remove(
            state, &state->delay_triggers, static_cast<std::uint16_t>(index));
    }
    return official_pull_trigger(
        state, rules, kOfficialTriggerTurnEnd, player_ref);
}

PTCG_OFFICIAL_TURN_HD inline OfficialTurnFlowResult official_begin_turn_end(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pod_ok(state) || state->turn_flow_stage != 0
        || state->refresh_flow_stage != 0 || state->reserved_flow != 0
        || state->trigger_resolver.active != 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, state->turn_flow_stage);
        }
        return OfficialTurnFlowResult::kError;
    }
    if (official_pod_finish_check(state)) return OfficialTurnFlowResult::kComplete;
    const std::int32_t active_player = official_active_player(*state);
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kTurnEnd,
        active_player);
    if (!official_queue_turn_end_triggers(state, rules, active_player)) {
        return OfficialTurnFlowResult::kError;
    }
    state->turn_flow_stage = static_cast<std::uint8_t>(
        OfficialTurnFlowStage::kTurnEndTriggers);
    const OfficialTriggerResolverResult trigger = official_begin_trigger_resolution(
        state, rules, 0);
    if (trigger == OfficialTriggerResolverResult::kError) {
        return OfficialTurnFlowResult::kError;
    }
    if (trigger == OfficialTriggerResolverResult::kNeedsAction) {
        return OfficialTurnFlowResult::kNeedsAction;
    }
    return official_advance_turn_flow(state, rules);
}

PTCG_OFFICIAL_TURN_HD inline OfficialTurnFlowResult official_resume_turn_flow(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (!official_pod_ok(state) || state->turn_flow_stage == 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        }
        return OfficialTurnFlowResult::kError;
    }
    if (state->refresh_flow_stage != 0) {
        const OfficialTurnFlowResult refresh = official_resume_refresh(
            state, rules, option_indices, count);
        if (refresh != OfficialTurnFlowResult::kComplete) return refresh;
        return official_advance_turn_flow(state, rules);
    }
    if (state->trigger_resolver.active != 0) {
        const OfficialTriggerResolverResult trigger = official_resume_trigger_resolution(
            state, rules, option_indices, count);
        if (trigger == OfficialTriggerResolverResult::kError) {
            return OfficialTurnFlowResult::kError;
        }
        if (trigger == OfficialTriggerResolverResult::kNeedsAction) {
            return OfficialTurnFlowResult::kNeedsAction;
        }
        return official_advance_turn_flow(state, rules);
    }
    official_pod_fail(state, OfficialPodError::kInvalidAction, count);
    return OfficialTurnFlowResult::kError;
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_TURN_HD
