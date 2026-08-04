#pragma once

#include <cstdint>

#include "ptcg_cuda/official_conditions_pod.cuh"
#include "ptcg_cuda/official_continuation_ids.cuh"
#include "ptcg_cuda/official_continual_refresh_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_INTERPRETER_HD __host__ __device__
#else
#define PTCG_OFFICIAL_INTERPRETER_HD
#endif

namespace ptcg::cuda_engine {

enum class OfficialEffectInterpreterResult : std::int32_t {
    kComplete = 0,
    kNeedsAction = 1,
    kError = 2,
};

enum class OfficialSelectTypeId : std::uint8_t {
    kNone = 0,
    kMain = 1,
    kCard = 2,
    kAttachedCard = 3,
    kCardOrAttachedCard = 4,
    kEnergy = 5,
    kSkill = 6,
    kAttack = 7,
    kEvolve = 8,
    kCount = 9,
    kYesNo = 10,
    kSpecialCondition = 11,
};

enum class OfficialSelectOptionTypeId : std::uint8_t {
    kNumber = 0,
    kYes = 1,
    kNo = 2,
    kCard = 3,
    kToolCard = 4,
    kEnergyCard = 5,
    kEnergy = 6,
    kPlay = 7,
    kAttach = 8,
    kEvolve = 9,
    kAbility = 10,
    kDiscard = 11,
    kRetreat = 12,
    kAttack = 13,
    kEnd = 14,
    kSkill = 15,
    kSpecialCondition = 16,
};

enum class OfficialEffectSelectTypeId : std::uint8_t {
    kAll = 0,
    kCardCount = 1,
    kMaxCardCount = 2,
    kCardUntil = 3,
    kMaxCardUntil = 4,
    kEnergy = 5,
    kMaxEnergyCard = 6,
    kToolCard = 7,
    kCardOrAttachedCardCount = 8,
    kEvolve = 9,
    kEvolve2 = 10,
};

enum class OfficialEffectResumeKind : std::uint8_t {
    kNone = 0,
    kApplyPrimitive = 1,
    kEvolve = 2,
    kEnergyLoop = 3,
    kRemoveDamageCounter = 4,
    kSelectActivate = 5,
    kSelectEffect = 6,
    kDamageCounterAny = 7,
};

constexpr std::uint64_t kOfficialEffectEnemySelect = 1ULL << 1;
constexpr std::uint64_t kOfficialEffectRandomSelect = 1ULL << 2;
constexpr std::uint64_t kOfficialEffectEachSelectedList = 1ULL << 3;
constexpr std::uint64_t kOfficialEffectEachList = 1ULL << 4;
constexpr std::uint64_t kOfficialEffectAddCheckList = 1ULL << 5;
constexpr std::uint64_t kOfficialEffectNotClearSelectedList = 1ULL << 6;
constexpr std::uint64_t kOfficialEffectNotPreTarget = 1ULL << 7;
constexpr std::uint64_t kOfficialEffectNotUpdateTarget = 1ULL << 8;
constexpr std::uint64_t kOfficialEffectCanNoSelect = 1ULL << 11;
constexpr std::uint64_t kOfficialEffectCanNoSelectIfPreTarget = 1ULL << 12;
constexpr std::uint64_t kOfficialEffectCannotNoSelect = 1ULL << 13;
constexpr std::uint64_t kOfficialEffectEnergyMaxSelect = 1ULL << 14;
constexpr std::uint64_t kOfficialEffectSelectTargetCount = 1ULL << 15;
constexpr std::uint64_t kOfficialEffectSelectCoinHeadCount = 1ULL << 16;
constexpr std::uint64_t kOfficialEffectSelectCoinHeadCount2 = 1ULL << 17;
constexpr std::uint64_t kOfficialEffectSelectEnemyEnergyCount = 1ULL << 18;
constexpr std::uint64_t kOfficialEffectSkipNoTarget = 1ULL << 19;
constexpr std::uint64_t kOfficialEffectSeeingDeck = 1ULL << 25;
constexpr std::uint64_t kOfficialEffectSeparator = 1ULL << 26;
constexpr std::uint64_t kOfficialEffectIsNotOpenSelectNoCondition = 1ULL << 29;

constexpr std::uint8_t kOfficialInterpreterLoopStopFlag = 1U << 2;
constexpr std::uint32_t kOfficialDefaultEffectStepBudget = 4096;
constexpr std::uint8_t kOfficialSelectContextEvolve = 38;
constexpr std::uint8_t kOfficialSelectContextDamageCounterAny = 15;
constexpr std::uint8_t kOfficialSelectContextRemoveDamageCounterCount = 41;
constexpr std::uint8_t kOfficialEffectSelectContextActivate = 44;
constexpr std::uint8_t kOfficialEffectSelectContextFirstEffect = 45;
constexpr std::uint8_t kOfficialEffectContextToHand = 8;
constexpr std::uint8_t kOfficialEffectContextDiscard = 9;
constexpr std::uint8_t kOfficialEffectContextToDeck = 10;
constexpr std::uint8_t kOfficialEffectContextDiscardEnergyCard = 27;
constexpr std::uint8_t kOfficialEffectContextDiscardToolCard = 28;
constexpr std::uint8_t kOfficialEffectContextDiscardEnergy = 31;
constexpr std::uint8_t kOfficialEffectContextToHandEnergy = 32;
constexpr std::uint8_t kOfficialEffectContextToDeckEnergy = 33;
constexpr std::int32_t kOfficialInterpreterDefaultBenchCapacity = 5;
constexpr std::uint64_t kOfficialInterpreterTransformOnlyFlag = 1ULL << 4U;

PTCG_OFFICIAL_INTERPRETER_HD inline std::int32_t
official_interpreter_bench_capacity(const OfficialPlayerStatePod& player) {
    const std::int32_t value = static_cast<std::int32_t>(
        (player.continual_state >> 40U) & 0xfULL);
    return value == 0 ? kOfficialInterpreterDefaultBenchCapacity : value;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_interpreter_has_flag(
    const OfficialEffectRule& effect,
    std::uint64_t flag) {
    return (effect.flags & flag) != 0;
}

PTCG_OFFICIAL_INTERPRETER_HD inline void official_clear_effect_selection(
    OfficialStatePod* state) {
    state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kNone);
    state->select_deck = 0;
    state->context_card = {};
    state->options.count = 0;
    state->selected.count = 0;
}

PTCG_OFFICIAL_INTERPRETER_HD inline void
official_record_auto_empty_selection_boundary(OfficialStatePod* state) {
    // A Main-returning ability, trigger, or attack effect selection is first
    // exposed through State::step even when it has zero options. ApiSelect
    // auto-advances it, but the hidden boundary remains observable in
    // turnActionCount. Direct item empties stay in the current step.
    constexpr std::uint8_t refresh_trigger_return_to_main_flag = 1U << 2U;
    constexpr std::uint8_t ability_return_to_main_flag = 1U << 4U;
    constexpr std::uint8_t counted_hidden_boundary_flags =
        refresh_trigger_return_to_main_flag | ability_return_to_main_flag;
    const bool refresh_knockout_trigger =
        state->trigger_resolver.active != 0
        && state->refresh_flow_stage != 0
        && state->reserved_flow != 0;
    const bool trigger_hidden_boundary =
        state->trigger_resolver.active != 0
        && ((state->flow_flags & counted_hidden_boundary_flags) != 0
            || refresh_knockout_trigger);
    const bool attack_hidden_boundary = state->attack_flow_stage != 0;
    if (trigger_hidden_boundary || attack_hidden_boundary) {
        ++state->turn_action_count;
    }
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_begin_effect_selection(
    OfficialStatePod* state,
    OfficialSelectTypeId type,
    std::uint8_t context,
    std::int32_t player,
    std::int32_t minimum,
    std::int32_t maximum,
    OfficialEffectResumeKind resume_kind) {
    if (player < 0 || player > 1 || minimum < 0 || maximum < minimum) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, player);
        return false;
    }
    state->select_type = static_cast<std::uint8_t>(type);
    state->select_context = context;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = static_cast<std::int16_t>(minimum);
    state->select_max = static_cast<std::int16_t>(maximum);
    state->options.count = 0;
    state->selected.count = 0;
    state->effect_interpreter.awaiting_selection = 1;
    state->effect_interpreter.resume_kind = static_cast<std::uint8_t>(resume_kind);
    return true;
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialSelectOptionPod* official_add_effect_option(
    OfficialStatePod* state,
    OfficialSelectOptionTypeId type,
    OfficialCardRefPod resolved_card = {}) {
    if (state->options.count >= state->options.capacity()) {
        official_pod_fail(state, OfficialPodError::kOptionOverflow, state->options.count);
        return nullptr;
    }
    OfficialSelectOptionPod* option = &state->options.values[state->options.count++];
    *option = OfficialSelectOptionPod{};
    option->type = static_cast<std::uint8_t>(type);
    option->resolved_card = resolved_card.index;
    if (resolved_card.index != 0) {
        const OfficialCardStatePod* card = official_pod_card(state, resolved_card);
        option->option_equiv = card == nullptr
            ? resolved_card.index
            : static_cast<std::uint16_t>(card->card_id);
    }
    return option;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_add_card_effect_option(
    OfficialStatePod* state,
    OfficialCardRefPod ref,
    OfficialSelectOptionTypeId type = OfficialSelectOptionTypeId::kCard) {
    const OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr || card->player < 0 || card->player > 1) return false;
    const OfficialArea area = static_cast<OfficialArea>(card->area);
    const std::int32_t area_index = official_pod_find_zone_index(
        state, card->player, area, ref);
    if (area_index < 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAreaIndex, ref.index);
        return false;
    }
    OfficialSelectOptionPod* option = official_add_effect_option(state, type, ref);
    if (option == nullptr) return false;
    option->params[0] = static_cast<std::int16_t>(area);
    option->params[1] = static_cast<std::int16_t>(area_index);
    option->params[2] = static_cast<std::int16_t>(card->player);
    return true;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_attached_ref_from_option(
    OfficialStatePod* state,
    const OfficialSelectOptionPod& option,
    OfficialCardRefPod* attached_ref) {
    const auto option_type = static_cast<OfficialSelectOptionTypeId>(option.type);
    const bool tool = option_type == OfficialSelectOptionTypeId::kToolCard;
    const bool energy = option_type == OfficialSelectOptionTypeId::kEnergy
        || option_type == OfficialSelectOptionTypeId::kEnergyCard;
    if (attached_ref == nullptr || (!tool && !energy)
        || option.params[2] < 0 || option.params[2] > 1
        || option.params[1] < 0 || option.params[3] < 0) {
        return false;
    }
    const std::int32_t player_index = option.params[2];
    const OfficialArea pokemon_area = static_cast<OfficialArea>(option.params[0]);
    if (pokemon_area != OfficialArea::kActive
        && pokemon_area != OfficialArea::kBench) {
        return false;
    }
    const OfficialPlayerStatePod& player = state->players[player_index];
    OfficialCardRefPod pokemon{};
    if (pokemon_area == OfficialArea::kActive) {
        if (option.params[1] >= player.active.count) return false;
        pokemon = player.active.values[option.params[1]];
    } else {
        if (option.params[1] >= player.bench.count) return false;
        pokemon = player.bench.values[option.params[1]];
    }
    if (option.resolved_card != pokemon.index) return false;
    const OfficialCardStatePod* pokemon_card = official_pod_card(state, pokemon);
    if (pokemon_card == nullptr) return false;
    std::int32_t ordinal = 0;
    const auto& zone = tool ? player.tool : player.energy;
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        const OfficialCardRefPod candidate = zone.values[index];
        const OfficialCardStatePod* attached = official_pod_card(state, candidate);
        if (attached == nullptr
            || attached->attach_move_counter != pokemon_card->move_counter) {
            continue;
        }
        if (ordinal++ == option.params[3]) {
            *attached_ref = candidate;
            return true;
        }
    }
    return false;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_energy_ref_from_option(
    OfficialStatePod* state,
    const OfficialSelectOptionPod& option,
    OfficialCardRefPod* energy_ref) {
    return option.type == static_cast<std::uint8_t>(
            OfficialSelectOptionTypeId::kEnergy)
        && official_attached_ref_from_option(state, option, energy_ref);
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_find_attachment_position(
    const OfficialStatePod& state,
    OfficialCardRefPod attached_ref,
    OfficialCardRefPod* pokemon_ref,
    std::int32_t* attached_index) {
    const OfficialCardStatePod* attached = official_pod_card(&state, attached_ref);
    if (attached == nullptr || attached->player < 0 || attached->player > 1) return false;
    const OfficialPlayerStatePod& player = state.players[attached->player];
    OfficialCardRefPod pokemon{};
    if (player.active.count > 0) {
        const OfficialCardStatePod* active = official_pod_card(&state, player.active.values[0]);
        if (active != nullptr && active->move_counter == attached->attach_move_counter) {
            pokemon = player.active.values[0];
        }
    }
    if (pokemon.index == 0) {
        for (std::uint16_t index = 0; index < player.bench.count; ++index) {
            const OfficialCardStatePod* bench = official_pod_card(&state, player.bench.values[index]);
            if (bench != nullptr && bench->move_counter == attached->attach_move_counter) {
                pokemon = player.bench.values[index];
                break;
            }
        }
    }
    if (pokemon.index == 0) return false;
    const OfficialArea area = static_cast<OfficialArea>(attached->area);
    const auto& zone = area == OfficialArea::kTool ? player.tool : player.energy;
    std::int32_t ordinal = 0;
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        const OfficialCardStatePod* candidate = official_pod_card(&state, zone.values[index]);
        if (candidate == nullptr
            || candidate->attach_move_counter != attached->attach_move_counter) continue;
        if (zone.values[index] == attached_ref) {
            *pokemon_ref = pokemon;
            *attached_index = ordinal;
            return true;
        }
        ++ordinal;
    }
    return false;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_add_attached_effect_option(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    OfficialSelectOptionTypeId type) {
    OfficialCardRefPod pokemon{};
    std::int32_t attached_index = 0;
    if (!official_find_attachment_position(*state, ref, &pokemon, &attached_index)) return true;
    const OfficialCardStatePod* pokemon_card = official_pod_card(state, pokemon);
    const OfficialCardStatePod* attached_card = official_pod_card(state, ref);
    if (pokemon_card == nullptr || attached_card == nullptr) return false;
    const OfficialArea pokemon_area = static_cast<OfficialArea>(pokemon_card->area);
    const std::int32_t pokemon_index = official_pod_find_zone_index(
        state, pokemon_card->player, pokemon_area, pokemon);
    if (pokemon_index < 0) return false;
    // Official attached-card options identify the attached Pokemon in their
    // card position/equivalence fields; params[3] identifies the attachment.
    OfficialSelectOptionPod* option = official_add_effect_option(
        state, type, pokemon);
    if (option == nullptr) return false;
    option->params[0] = static_cast<std::int16_t>(pokemon_area);
    option->params[1] = static_cast<std::int16_t>(pokemon_index);
    option->params[2] = static_cast<std::int16_t>(pokemon_card->player);
    option->params[3] = static_cast<std::int16_t>(attached_index);
    if (type == OfficialSelectOptionTypeId::kEnergy) {
        const OfficialEnergyInfoPod info = official_energy_info(*state, rules, ref, pokemon);
        option->params[4] = static_cast<std::int16_t>(info.count);
    }
    return true;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_effect_target_is_deck(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect) {
    const OfficialTargetRule* target = official_effect_target_rule(state, rules, effect);
    if (target == nullptr) return false;
    const std::int32_t count = target->values[kTargetAreaCount];
    for (std::int32_t index = 0; index < count; ++index) {
        if (target->values[kTargetArea0 + index]
            == static_cast<std::int32_t>(OfficialArea::kDeck)) return true;
    }
    return false;
}

PTCG_OFFICIAL_INTERPRETER_HD inline void official_remove_pre_targets(
    OfficialStatePod* state) {
    for (std::uint16_t index = 0; index < state->targets.count;) {
        bool found = false;
        for (std::uint16_t previous = 0; previous < state->pre_targets.count; ++previous) {
            if (state->targets.values[index].card == state->pre_targets.values[previous].card) {
                found = true;
                break;
            }
        }
        if (!found) {
            ++index;
            continue;
        }
        for (std::uint16_t move = index + 1; move < state->targets.count; ++move) {
            state->targets.values[move - 1] = state->targets.values[move];
        }
        --state->targets.count;
    }
}

PTCG_OFFICIAL_INTERPRETER_HD inline std::int32_t official_effect_select_count(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect) {
    std::int32_t count = effect.values[kEffectSelectCount];
    if (official_interpreter_has_flag(effect, kOfficialEffectSelectTargetCount)) {
        count = state->targets.count;
    } else if (official_interpreter_has_flag(effect, kOfficialEffectSelectCoinHeadCount)) {
        count = state->coin_head_count;
    } else if (official_interpreter_has_flag(effect, kOfficialEffectSelectCoinHeadCount2)) {
        count = state->coin_head_count * 2;
    } else if (official_interpreter_has_flag(effect, kOfficialEffectSelectEnemyEnergyCount)) {
        count = official_condition_energy_count(
            *state, rules, 1 - state->effect_interpreter.effect_owner);
    }
    return count < 0 ? 0 : count;
}

PTCG_OFFICIAL_INTERPRETER_HD inline std::int32_t official_effect_select_player(
    const OfficialStatePod& state,
    const OfficialEffectRule& effect) {
    const std::int32_t owner = state.effect_interpreter.effect_owner;
    return official_interpreter_has_flag(effect, kOfficialEffectEnemySelect)
        ? 1 - owner : owner;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_prepare_card_options(
    OfficialStatePod* state,
    OfficialSelectTypeId select_type,
    OfficialSelectOptionTypeId option_type) {
    for (std::uint16_t index = 0; index < state->targets.count; ++index) {
        if (!official_add_card_effect_option(
                state, state->targets.values[index].card, option_type)) return false;
    }
    state->select_type = static_cast<std::uint8_t>(select_type);
    return official_pod_ok(state);
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_prepare_attached_options(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialSelectOptionTypeId option_type) {
    for (std::uint16_t index = 0; index < state->targets.count; ++index) {
        const OfficialCardRefPod ref = state->targets.values[index].card;
        // SelectPokemonEnergyLoop excludes an attached Energy when its host
        // Pokemon is protected from the current effect.  The official check
        // treats effects on attached cards as effects on that Pokemon.
        if (option_type == OfficialSelectOptionTypeId::kEnergy
            && official_effect_blocks_target_effect(state, rules, ref)) {
            continue;
        }
        if (!official_add_attached_effect_option(
                state, rules, ref, option_type)) return false;
    }
    return official_pod_ok(state);
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_prepare_evolve_options(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t select_player,
    bool stage2_shortcut) {
    const OfficialPlayerStatePod& player = state->players[select_player];
    for (std::uint16_t target_index = 0; target_index < state->targets.count; ++target_index) {
        const OfficialCardRefPod evolution_ref = state->targets.values[target_index].card;
        const OfficialCardStatePod* evolution = official_pod_card(state, evolution_ref);
        const OfficialCardRule* evolution_master = evolution == nullptr ? nullptr
            : official_card_rule(rules, static_cast<std::uint32_t>(evolution->card_id));
        if (evolution_master == nullptr
            || evolution_master->values[kCardType] != 0
            || (evolution_master->flags
                & kOfficialInterpreterTransformOnlyFlag) != 0) continue;
        for (int zone = 0; zone < 2; ++zone) {
            const OfficialCardRefPod* values = zone == 0
                ? player.active.values : player.bench.values;
            const std::uint16_t count = zone == 0
                ? player.active.count : player.bench.count;
            for (std::uint16_t index = 0; index < count; ++index) {
                const OfficialCardStatePod* base = official_pod_card(state, values[index]);
                const OfficialCardRule* base_master = base == nullptr ? nullptr
                    : official_card_rule(rules, static_cast<std::uint32_t>(base->card_id));
                if (base_master == nullptr
                    || (stage2_shortcut
                        && (base->runtime_flags & kCardAppear) != 0)
                    || !official_evolves_from(*evolution_master, *base_master, stage2_shortcut)) {
                    continue;
                }
                OfficialSelectOptionPod* option = official_add_effect_option(
                    state, OfficialSelectOptionTypeId::kEvolve, evolution_ref);
                if (option == nullptr) return false;
                option->params[0] = static_cast<std::int16_t>(evolution->area);
                option->params[1] = static_cast<std::int16_t>(official_pod_find_zone_index(
                    state,
                    evolution->player,
                    static_cast<OfficialArea>(evolution->area),
                    evolution_ref));
                option->params[2] = static_cast<std::int16_t>(
                    zone == 0 ? OfficialArea::kActive : OfficialArea::kBench);
                option->params[3] = static_cast<std::int16_t>(index);
                option->option_equiv = static_cast<std::uint16_t>(
                    (evolution->card_id * 131 + base->card_id) & 0xffff);
            }
        }
    }
    return official_pod_ok(state);
}

PTCG_OFFICIAL_INTERPRETER_HD inline void official_finish_effect_iteration(
    OfficialStatePod* state) {
    OfficialEffectInterpreterPod& interpreter = state->effect_interpreter;
    state->effect_state.on_effect = 0;
    interpreter.damage_counter_any_total = 0;
    ++interpreter.repeat_index;
    if (interpreter.repeat_index < interpreter.repeat_count) return;
    interpreter.repeat_index = 0;
    interpreter.repeat_count = 0;
    ++interpreter.effect_index;
    if (interpreter.separator_pending != 0) {
        interpreter.separator_pending = 0;
        if ((state->control_flags & kOfficialChangedFlag) == 0) {
            interpreter.active = 0;
        }
    }
    if ((state->control_flags & kOfficialBreakEffectFlag) != 0) {
        state->control_flags &= static_cast<std::uint8_t>(~kOfficialBreakEffectFlag);
        interpreter.active = 0;
    }
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_advance_effect_interpreter(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

PTCG_OFFICIAL_INTERPRETER_HD inline void
official_complete_energy_selection_scratch(OfficialStatePod* state);

constexpr std::uint64_t kOfficialEffectSetTargetSwitchBench = 1ULL << 21U;
constexpr std::uint64_t kOfficialEffectTargetActive = 1ULL << 22U;
constexpr std::uint64_t kOfficialEffectTargetBench = 1ULL << 23U;

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectApplyResult
official_apply_switch_effect(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect) {
    for (std::uint16_t index = 0; index < state->targets.count; ++index) {
        const OfficialAreaRefPod target = state->targets.values[index];
        if (!official_pod_area_ref_valid(state, target)) continue;
        OfficialCardStatePod* card = official_pod_card(state, target.card);
        if (card == nullptr || card->player < 0 || card->player > 1) continue;
        const OfficialPlayerStatePod& player = state->players[card->player];
        if (player.active.count == 0) continue;

        const OfficialCardRefPod active_ref = player.active.values[0];
        OfficialCardRefPod effected_ref = target.card;
        if ((effect.flags & kOfficialEffectTargetBench) == 0
            && ((effect.flags & kOfficialEffectTargetActive) != 0
                || (state->effect_state.ability.use_player != card->player
                    && state->select_player != card->player))) {
            effected_ref = active_ref;
        }
        if (official_effect_blocks_target_effect(state, rules, effected_ref)) {
            if (!official_pod_ok(state)) return OfficialEffectApplyResult::kError;
            continue;
        }
        if (card->area != static_cast<std::uint8_t>(OfficialArea::kBench)) continue;
        const std::int32_t bench_index = official_pod_find_zone_index(
            state, card->player, OfficialArea::kBench, target.card);
        if (bench_index < 0) return OfficialEffectApplyResult::kError;

        official_pod_mark_changed(state);
        if (!official_switch_pokemon(
                state,
                rules,
                card->player,
                static_cast<std::uint16_t>(bench_index))) {
            return OfficialEffectApplyResult::kError;
        }
        if ((effect.flags & kOfficialEffectSetTargetSwitchBench) != 0) {
            state->targets.count = 0;
            if (!official_pod_push(
                    state,
                    &state->targets,
                    official_pod_area_ref(state, active_ref),
                    OfficialPodError::kSelectionOverflow)) {
                return OfficialEffectApplyResult::kError;
            }
            break;
        }
    }
    return official_pod_ok(state)
        ? OfficialEffectApplyResult::kApplied
        : OfficialEffectApplyResult::kError;
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_apply_effect_and_finish(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect) {
    const OfficialEffectApplyResult result = effect.values[kEffectType]
            == static_cast<std::int32_t>(OfficialEffectTypeId::kSwitch)
        ? official_apply_switch_effect(state, rules, effect)
        : official_apply_effect_primitive(state, rules, effect);
    if (result == OfficialEffectApplyResult::kApplied) {
        if (official_interpreter_has_flag(effect, kOfficialEffectAddCheckList)) {
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                official_pod_push(
                    state,
                    &state->check_list,
                    state->targets.values[index].card,
                    OfficialPodError::kSelectionOverflow);
            }
        }
        official_finish_effect_iteration(state);
        return official_pod_ok(state)
            ? OfficialEffectInterpreterResult::kComplete
            : OfficialEffectInterpreterResult::kError;
    }
    if (result == OfficialEffectApplyResult::kNeedsSelection) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedEffect,
            effect.values[kEffectType]);
    }
    return OfficialEffectInterpreterResult::kError;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_prepare_effect_repeat(
    OfficialStatePod* state,
    const OfficialEffectRule& effect) {
    OfficialEffectInterpreterPod& interpreter = state->effect_interpreter;
    if (interpreter.repeat_count == 0) {
        if (official_interpreter_has_flag(effect, kOfficialEffectEachSelectedList)) {
            interpreter.repeat_count = state->selected_list.count;
            interpreter.repeat_continuation = static_cast<std::uint16_t>(
                OfficialContinuationId::kActivateEffectEachSelected);
        } else if (official_interpreter_has_flag(effect, kOfficialEffectEachList)) {
            interpreter.repeat_count = state->each_list.count;
            interpreter.repeat_continuation = static_cast<std::uint16_t>(
                OfficialContinuationId::kActivateEffectForEach);
        } else if (effect.values[kEffectLoopCount] > 0) {
            interpreter.repeat_count = static_cast<std::uint16_t>(effect.values[kEffectLoopCount]);
            interpreter.repeat_continuation = static_cast<std::uint16_t>(
                OfficialContinuationId::kActivateEffectMultiple);
        } else {
            interpreter.repeat_count = 1;
            interpreter.repeat_continuation = 0;
        }
        interpreter.repeat_index = 0;
    }
    if (interpreter.repeat_count == 0) {
        interpreter.repeat_continuation = 0;
        ++interpreter.effect_index;
        return false;
    }
    if (official_interpreter_has_flag(effect, kOfficialEffectEachSelectedList)) {
        state->context_card = state->selected_list.values[interpreter.repeat_index];
        state->effect_state.selected_list_index = static_cast<std::int8_t>(
            interpreter.repeat_index);
    } else if (official_interpreter_has_flag(effect, kOfficialEffectEachList)) {
        state->context_card = state->each_list.values[interpreter.repeat_index];
        state->effect_state.each_list_index = static_cast<std::int8_t>(
            interpreter.repeat_index);
    } else if (effect.values[kEffectLoopCount] > 0) {
        state->effect_state.selected_list_index = static_cast<std::int8_t>(
            interpreter.repeat_index);
    }
    return true;
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_prepare_effect_selection(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect,
    OfficialEffectSelectTypeId select_type,
    std::int32_t select_count) {
    const std::int32_t player = official_effect_select_player(*state, effect);
    std::uint8_t context = static_cast<std::uint8_t>(
        effect.values[kEffectSelectContext]);
    const std::int32_t target_count = state->targets.count;

    // ActivateEffect rewrites generic move contexts to the selection-specific
    // contexts exposed by the official API.
    if (select_type == OfficialEffectSelectTypeId::kEnergy) {
        if (context == kOfficialEffectContextDiscard) {
            context = kOfficialEffectContextDiscardEnergy;
        } else if (context == kOfficialEffectContextToHand) {
            context = kOfficialEffectContextToHandEnergy;
        } else if (context == kOfficialEffectContextToDeck) {
            context = kOfficialEffectContextToDeckEnergy;
        }
    } else if (select_type == OfficialEffectSelectTypeId::kMaxEnergyCard
        && context == kOfficialEffectContextDiscard) {
        context = kOfficialEffectContextDiscardEnergyCard;
    } else if (select_type == OfficialEffectSelectTypeId::kToolCard
        && context == kOfficialEffectContextDiscard) {
        context = kOfficialEffectContextDiscardToolCard;
    }

    if (select_type == OfficialEffectSelectTypeId::kEvolve
        || select_type == OfficialEffectSelectTypeId::kEvolve2) {
        if (!official_begin_effect_selection(
                state,
                OfficialSelectTypeId::kEvolve,
                kOfficialSelectContextEvolve,
                player,
                1,
                1,
                OfficialEffectResumeKind::kEvolve)
            || !official_prepare_evolve_options(
                state,
                rules,
                player,
                select_type == OfficialEffectSelectTypeId::kEvolve2)) {
            return OfficialEffectInterpreterResult::kError;
        }
        if (state->options.count == 0) {
            official_record_auto_empty_selection_boundary(state);
            official_clear_effect_selection(state);
            state->effect_interpreter.awaiting_selection = 0;
            official_finish_effect_iteration(state);
            return OfficialEffectInterpreterResult::kComplete;
        }
        return OfficialEffectInterpreterResult::kNeedsAction;
    }

    if (select_type == OfficialEffectSelectTypeId::kEnergy) {
        // ActivateEffect only calls SelectPokemonEnergy after at least one
        // target establishes the attached-card owner. With no targets it
        // leaves the existing selection metadata untouched.
        if (state->targets.count == 0) {
            official_finish_effect_iteration(state);
            return OfficialEffectInterpreterResult::kComplete;
        }
        state->energy_cost = select_count;
        state->remain_energy_cost = select_count;
        state->selected_energy_card_count = 0;
        state->selected_list.count = 0;
        if (state->targets.count > 0) {
            // Official SelectPokemonEnergy marks changed when its incoming
            // AttachedCardList is non-empty, before filtering individual
            // options for prevented effects.
            official_pod_mark_changed(state);
            std::int32_t energy_sum = 0;
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialCardRefPod ref = state->targets.values[index].card;
                OfficialCardRefPod pokemon{};
                std::int32_t attached_index = 0;
                if (official_find_attachment_position(
                        *state, ref, &pokemon, &attached_index)) {
                    energy_sum += official_energy_info(
                        *state, rules, ref, pokemon).count;
                }
            }
            if (state->energy_cost > energy_sum) {
                state->energy_cost = energy_sum;
                state->remain_energy_cost = energy_sum;
            }
        }
        if (!official_begin_effect_selection(
                state,
                OfficialSelectTypeId::kEnergy,
                context,
                player,
                state->remain_energy_cost <= 0 ? 0 : 1,
                1,
                OfficialEffectResumeKind::kEnergyLoop)
            || !official_prepare_attached_options(
                state, rules, OfficialSelectOptionTypeId::kEnergy)) {
            return OfficialEffectInterpreterResult::kError;
        }
        if (state->options.count == 0) {
            official_clear_effect_selection(state);
            state->effect_interpreter.awaiting_selection = 0;
            official_complete_energy_selection_scratch(state);
            official_finish_effect_iteration(state);
            return OfficialEffectInterpreterResult::kComplete;
        }
        return OfficialEffectInterpreterResult::kNeedsAction;
    }

    std::int32_t maximum = select_count < target_count ? select_count : target_count;
    std::int32_t minimum = maximum;
    if (select_type == OfficialEffectSelectTypeId::kCardUntil
        || select_type == OfficialEffectSelectTypeId::kMaxCardUntil) {
        maximum = target_count - effect.values[kEffectSelectCount];
        if (maximum < 0) maximum = 0;
        minimum = maximum;
        if (select_type == OfficialEffectSelectTypeId::kMaxCardUntil) {
            maximum = target_count;
        }
    }
    if (select_type == OfficialEffectSelectTypeId::kMaxCardCount) {
        minimum = maximum > 0 ? 1 : 0;
        if (state->current_attack_id != 0) minimum = 0;
    }
    // Tool/attached-card selections require one card when the target list is
    // non-empty. Empty target lists take the official automatic no-op path.
    if ((select_type == OfficialEffectSelectTypeId::kMaxEnergyCard
            || select_type == OfficialEffectSelectTypeId::kToolCard
            || select_type == OfficialEffectSelectTypeId::kCardOrAttachedCardCount)
        && target_count > 0) {
        minimum = 1;
        if (select_type == OfficialEffectSelectTypeId::kMaxEnergyCard
            && state->current_attack_id != 0) {
            minimum = 0;
        }
    }
    if (official_interpreter_has_flag(effect, kOfficialEffectCanNoSelect)
        || (official_interpreter_has_flag(effect, kOfficialEffectCanNoSelectIfPreTarget)
            && state->pre_targets.count > 0)
        || official_interpreter_has_flag(
            effect, kOfficialEffectIsNotOpenSelectNoCondition)) {
        minimum = 0;
    }
    if (maximum > 0
        && official_interpreter_has_flag(effect, kOfficialEffectCannotNoSelect)) {
        minimum = 1;
    }

    // The official interpreter constrains effects such as Buddy-Buddy Poffin
    // by the target player's current (continual-effect-adjusted) Bench
    // capacity before exposing the selection.  The physical POD Bench array
    // is deliberately larger than the rules default, so its storage capacity
    // cannot be used as the gameplay limit.
    if (effect.values[kEffectType]
        == static_cast<std::int32_t>(OfficialEffectTypeId::kToBench)) {
        const std::uint8_t player_mask = official_effect_player_mask(
            state, rules, effect);
        const std::int32_t target_player = official_effect_target_player(
            *state, player_mask, 0);
        if (target_player >= 0 && target_player < 2) {
            const std::int32_t remaining = official_interpreter_bench_capacity(
                state->players[target_player])
                - static_cast<std::int32_t>(
                    state->players[target_player].bench.count);
            if (remaining <= 0) {
                state->targets.count = 0;
                official_finish_effect_iteration(state);
                return OfficialEffectInterpreterResult::kComplete;
            }
            if (maximum > remaining) maximum = remaining;
            if (minimum > maximum) minimum = maximum;
        }
    }

    OfficialSelectTypeId state_select_type = OfficialSelectTypeId::kCard;
    OfficialSelectOptionTypeId option_type = OfficialSelectOptionTypeId::kCard;
    if (select_type == OfficialEffectSelectTypeId::kMaxEnergyCard) {
        state_select_type = OfficialSelectTypeId::kAttachedCard;
        option_type = OfficialSelectOptionTypeId::kEnergyCard;
    } else if (select_type == OfficialEffectSelectTypeId::kToolCard) {
        state_select_type = OfficialSelectTypeId::kAttachedCard;
        option_type = OfficialSelectOptionTypeId::kToolCard;
    } else if (select_type == OfficialEffectSelectTypeId::kCardOrAttachedCardCount) {
        state_select_type = OfficialSelectTypeId::kCardOrAttachedCard;
    }
    if (!official_begin_effect_selection(
            state,
            state_select_type,
            context,
            player,
            minimum,
            maximum,
            OfficialEffectResumeKind::kApplyPrimitive)) {
        return OfficialEffectInterpreterResult::kError;
    }
    if (select_type == OfficialEffectSelectTypeId::kMaxEnergyCard
        || select_type == OfficialEffectSelectTypeId::kToolCard) {
        if (!official_prepare_attached_options(state, rules, option_type)) {
            return OfficialEffectInterpreterResult::kError;
        }
    } else if (select_type == OfficialEffectSelectTypeId::kCardOrAttachedCardCount) {
        for (std::uint16_t index = 0; index < state->targets.count; ++index) {
            const OfficialCardRefPod ref = state->targets.values[index].card;
            const OfficialCardStatePod* card = official_pod_card(state, ref);
            if (card == nullptr) continue;
            const OfficialArea area = static_cast<OfficialArea>(card->area);
            if (area == OfficialArea::kEnergy) {
                if (!official_add_attached_effect_option(
                        state, rules, ref, OfficialSelectOptionTypeId::kEnergyCard)) {
                    return OfficialEffectInterpreterResult::kError;
                }
            } else if (area == OfficialArea::kTool) {
                if (!official_add_attached_effect_option(
                        state, rules, ref, OfficialSelectOptionTypeId::kToolCard)) {
                    return OfficialEffectInterpreterResult::kError;
                }
            } else if (!official_add_card_effect_option(state, ref)) {
                return OfficialEffectInterpreterResult::kError;
            }
        }
    } else if (!official_prepare_card_options(state, state_select_type, option_type)) {
        return OfficialEffectInterpreterResult::kError;
    }
    if (state->select_max == 0
        && select_type == OfficialEffectSelectTypeId::kCardUntil) {
        // ActivateEffect returns without applying the primitive when the
        // target zone is already at or below the requested limit.
        state->effect_interpreter.awaiting_selection = 0;
        official_clear_effect_selection(state);
        official_finish_effect_iteration(state);
        return OfficialEffectInterpreterResult::kComplete;
    }
    if (state->options.count == 0) {
        official_record_auto_empty_selection_boundary(state);
        constexpr std::uint8_t play_effect_return_to_main_flag = 1U << 5U;
        const bool played_card_has_following_effect =
            (state->flow_flags & play_effect_return_to_main_flag) != 0
            && state->effect_interpreter.repeat_count == 1
            && state->effect_interpreter.effect_index + 1
                < state->effect_interpreter.effect_count;
        const bool played_card_has_later_empty_repeat =
            (state->flow_flags & play_effect_return_to_main_flag) != 0
            && state->effect_interpreter.repeat_count > 1
            && state->effect_interpreter.repeat_index > 0
            && state->effect_interpreter.repeat_continuation
                == static_cast<std::uint16_t>(
                    OfficialContinuationId::kActivateEffectEachSelected);
        if (played_card_has_following_effect
            || played_card_has_later_empty_repeat) {
            // ApiSelect counts this zero-cardinality boundary before it
            // auto-advances into the played card's following effects. It also
            // counts a later empty iteration after an earlier repeated
            // selection was exposed to the player.
            ++state->turn_action_count;
        }
        state->effect_interpreter.awaiting_selection = 0;
        official_clear_effect_selection(state);
        state->targets.count = 0;
        return official_apply_effect_and_finish(state, rules, effect);
    }
    if (maximum > state->options.count) state->select_max = state->options.count;
    if (state->select_min > state->select_max) state->select_min = state->select_max;
    return OfficialEffectInterpreterResult::kNeedsAction;
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_prepare_direct_yes_no_effect_selection(
    OfficialStatePod* state,
    const OfficialEffectRule& effect) {
    const auto type = static_cast<OfficialEffectTypeId>(
        effect.values[kEffectType]);
    const bool select_activate = type == OfficialEffectTypeId::kSelectActivate;
    if (!select_activate && type != OfficialEffectTypeId::kSelectEffect) {
        official_pod_fail(
            state, OfficialPodError::kUnsupportedEffect,
            effect.values[kEffectType]);
        return OfficialEffectInterpreterResult::kError;
    }
    if (!official_begin_effect_selection(
            state,
            OfficialSelectTypeId::kYesNo,
            select_activate
                ? kOfficialEffectSelectContextActivate
                : kOfficialEffectSelectContextFirstEffect,
            state->effect_interpreter.effect_owner,
            1,
            1,
            select_activate
                ? OfficialEffectResumeKind::kSelectActivate
                : OfficialEffectResumeKind::kSelectEffect)) {
        return OfficialEffectInterpreterResult::kError;
    }
    if (official_add_effect_option(
            state, OfficialSelectOptionTypeId::kYes) == nullptr
        || official_add_effect_option(
            state, OfficialSelectOptionTypeId::kNo) == nullptr) {
        return OfficialEffectInterpreterResult::kError;
    }
    return OfficialEffectInterpreterResult::kNeedsAction;
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_prepare_damage_counter_any_selection(OfficialStatePod* state) {
    if (state->remain_damage_counter <= 0 || state->targets.count == 0) {
        state->remain_damage_counter = 0;
        official_finish_effect_iteration(state);
        return OfficialEffectInterpreterResult::kComplete;
    }
    if (!official_begin_effect_selection(
            state,
            OfficialSelectTypeId::kCard,
            kOfficialSelectContextDamageCounterAny,
            state->effect_interpreter.effect_owner,
            1,
            1,
            OfficialEffectResumeKind::kDamageCounterAny)
        || !official_prepare_card_options(
            state,
            OfficialSelectTypeId::kCard,
            OfficialSelectOptionTypeId::kCard)) {
        return OfficialEffectInterpreterResult::kError;
    }
    if (state->options.count == 0) {
        official_record_auto_empty_selection_boundary(state);
        official_clear_effect_selection(state);
        state->effect_interpreter.awaiting_selection = 0;
        state->remain_damage_counter = 0;
        official_finish_effect_iteration(state);
        return OfficialEffectInterpreterResult::kComplete;
    }
    return OfficialEffectInterpreterResult::kNeedsAction;
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_advance_effect_interpreter(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    OfficialEffectInterpreterPod& interpreter = state->effect_interpreter;
    if (!official_pod_ok(state)) return OfficialEffectInterpreterResult::kError;
    if (interpreter.awaiting_selection != 0) {
        return OfficialEffectInterpreterResult::kNeedsAction;
    }
    while (interpreter.active != 0) {
        if (interpreter.step_budget == 0) {
            official_pod_fail(
                state,
                OfficialPodError::kInterpreterBudget,
                static_cast<std::int32_t>(interpreter.effect_index));
            return OfficialEffectInterpreterResult::kError;
        }
        --interpreter.step_budget;
        ++state->interpreter_steps;
        if (interpreter.effect_index >= interpreter.effect_count) {
            interpreter.active = 0;
            break;
        }
        if (state->effect_jump > 0) {
            --state->effect_jump;
            ++interpreter.effect_index;
            continue;
        }
        const std::uint64_t absolute = static_cast<std::uint64_t>(
            interpreter.effect_offset) + interpreter.effect_index;
        const std::uint32_t total = rules.header->counts[
            static_cast<std::uint32_t>(OfficialRuleSection::kEffects)];
        if (absolute >= total) {
            official_pod_fail(
                state,
                OfficialPodError::kRulePackBounds,
                static_cast<std::int32_t>(absolute));
            return OfficialEffectInterpreterResult::kError;
        }
        const OfficialEffectRule& effect = rules.effects[absolute];
        state->effect_state.effect_index = static_cast<std::int8_t>(
            interpreter.effect_index);
        state->effect_state.on_effect = 1;

        if ((effect.flags & kOfficialEffectIsCondition) != 0) {
            // Skill first conditions are evaluated while the action is made
            // available. ActivateSkillEffect subsequently visits their frames
            // without evaluating them a second time.
            if (interpreter.effect_index < interpreter.first_condition_count) {
                state->effect_state.on_effect = 0;
                ++interpreter.effect_index;
                continue;
            }
            const OfficialConditionResult condition = official_satisfy_condition(
                state,
                rules,
                rules.effects + interpreter.effect_offset,
                interpreter.effect_count,
                interpreter.effect_index,
                interpreter.effect_card,
                interpreter.effect_owner);
            state->effect_state.on_effect = 0;
            if (condition == OfficialConditionResult::kTrue) {
                ++interpreter.effect_index;
                continue;
            }
            if (condition != OfficialConditionResult::kFalse) {
                return OfficialEffectInterpreterResult::kError;
            }
            const std::int32_t fail_skip = effect.values[kEffectFailSkip];
            if (fail_skip > 0) {
                interpreter.effect_index = static_cast<std::uint16_t>(
                    interpreter.effect_index + fail_skip + 1);
                continue;
            }
            interpreter.active = 0;
            break;
        }
#if defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
        official_mark_effect_reached(state, static_cast<std::uint32_t>(absolute));
        official_mark_effect_applied(state, static_cast<std::uint32_t>(absolute));
#endif
        if (!official_prepare_effect_repeat(state, effect)) continue;
        interpreter.separator_pending = official_interpreter_has_flag(
            effect, kOfficialEffectSeparator) ? 1 : 0;

        const OfficialTargetRule* target = official_effect_target_rule(state, rules, effect);
        if (target == nullptr) return OfficialEffectInterpreterResult::kError;
        if (!official_interpreter_has_flag(effect, kOfficialEffectNotUpdateTarget)) {
            if (target->values[kTargetAreaCount] == 0
                || target->values[kTargetArea0]
                    != static_cast<std::int32_t>(OfficialArea::kEffectedPreTarget)) {
                state->pre_targets = state->targets;
            }
            if (!official_build_target_list(
                    state,
                    rules,
                    *target,
                    &state->targets,
                    interpreter.effect_card,
                    interpreter.effect_owner)) {
                return OfficialEffectInterpreterResult::kError;
            }
        }
        if (official_interpreter_has_flag(effect, kOfficialEffectNotPreTarget)) {
            official_remove_pre_targets(state);
        }
        if (official_interpreter_has_flag(effect, kOfficialEffectSkipNoTarget)
            && state->targets.count == 0) {
            official_finish_effect_iteration(state);
            continue;
        }
        state->select_deck = static_cast<std::uint8_t>(
            official_effect_target_is_deck(state, rules, effect)
            || official_interpreter_has_flag(effect, kOfficialEffectSeeingDeck));

        const auto effect_type = static_cast<OfficialEffectTypeId>(
            effect.values[kEffectType]);
        if (effect_type == OfficialEffectTypeId::kSelectActivate
            || effect_type == OfficialEffectTypeId::kSelectEffect) {
            return official_prepare_direct_yes_no_effect_selection(
                state, effect);
        }
        if (effect_type == OfficialEffectTypeId::kDamageCounterAny) {
            const std::int32_t damage_counter_count =
                official_effect_value(*state, effect, 0);
            state->effect_interpreter.damage_counter_any_total =
                damage_counter_count > 0
                ? static_cast<std::uint32_t>(damage_counter_count)
                : 0U;
            state->remain_damage_counter = damage_counter_count;
            const OfficialEffectInterpreterResult selection =
                official_prepare_damage_counter_any_selection(state);
            if (selection != OfficialEffectInterpreterResult::kComplete) return selection;
            continue;
        }

        const OfficialEffectSelectTypeId select_type =
            static_cast<OfficialEffectSelectTypeId>(effect.values[kEffectSelectType]);
        if (select_type != OfficialEffectSelectTypeId::kAll
            && state->selected_list.count == 0
            && (effect_type == OfficialEffectTypeId::kAttachSelectedCard
                || effect_type == OfficialEffectTypeId::kSwitchSelectedCard)) {
            // ActivateEffect returns before target selection when an earlier
            // effect produced no selected attached cards.
            official_finish_effect_iteration(state);
            continue;
        }
        const std::int32_t select_count = official_effect_select_count(state, rules, effect);
        if (select_type == OfficialEffectSelectTypeId::kAll) {
            const OfficialEffectInterpreterResult applied = official_apply_effect_and_finish(
                state, rules, effect);
            if (applied == OfficialEffectInterpreterResult::kError) return applied;
            continue;
        }
        if (select_count == 0
            && (select_type == OfficialEffectSelectTypeId::kCardCount
                || select_type == OfficialEffectSelectTypeId::kCardUntil)) {
            official_finish_effect_iteration(state);
            continue;
        }
        if (official_interpreter_has_flag(effect, kOfficialEffectRandomSelect)) {
            official_pod_shuffle(
                state->targets.values,
                state->targets.count,
                &state->rng);
            const std::int32_t maximum = select_count < state->targets.count
                ? select_count : state->targets.count;
            state->targets.count = static_cast<std::uint16_t>(maximum);
            const OfficialEffectInterpreterResult applied = official_apply_effect_and_finish(
                state, rules, effect);
            if (applied == OfficialEffectInterpreterResult::kError) return applied;
            continue;
        }
        const OfficialEffectInterpreterResult selection = official_prepare_effect_selection(
            state, rules, effect, select_type, select_count);
        if (selection != OfficialEffectInterpreterResult::kComplete) return selection;
    }
    return official_pod_ok(state)
        ? OfficialEffectInterpreterResult::kComplete
        : OfficialEffectInterpreterResult::kError;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_selection_contains(
    const std::uint16_t* indices,
    std::uint16_t count,
    std::uint16_t value) {
    for (std::uint16_t index = 0; index < count; ++index) {
        if (indices[index] == value) return true;
    }
    return false;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_validate_effect_action(
    OfficialStatePod* state,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (state->effect_interpreter.awaiting_selection == 0
        || count < state->select_min
        || count > state->select_max) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        return false;
    }
    for (std::uint16_t index = 0; index < count; ++index) {
        if (option_indices[index] >= state->options.count
            || official_selection_contains(option_indices, index, option_indices[index])) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, option_indices[index]);
            return false;
        }
    }
    return true;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool official_move_selected_energy(
    OfficialStatePod* state,
    OfficialCardRefPod ref) {
    const OfficialArea destination = state->select_context == 31
        ? OfficialArea::kTrash
        : (state->select_context == 32
            ? OfficialArea::kHand
            : (state->select_context == 33 ? OfficialArea::kDeck : OfficialArea::kEnergy));
    if (destination != OfficialArea::kEnergy) {
        return official_pod_move_ref(state, ref, destination, false).index != 0;
    }
    return true;
}

PTCG_OFFICIAL_INTERPRETER_HD inline void
official_complete_energy_selection_scratch(OfficialStatePod* state) {
    // Official SelectedPokemonEnergy runs after the final loop iteration.
    state->energy_cost = 0;
    state->remain_energy_cost = 0;
    state->selected_energy_card_count = 0;
    state->selecting_energy_pokemon = {};
    state->targets.count = 0;
}

PTCG_OFFICIAL_INTERPRETER_HD inline bool
official_card_cannot_move_damage_counter(const OfficialCardStatePod& card) {
    constexpr std::uint32_t bit = static_cast<std::uint32_t>(
        OfficialEffectTypeId::kCannotMoveDamageCounter)
        - static_cast<std::uint32_t>(OfficialEffectTypeId::kNoAbility);
    return official_continual_flag(card, bit);
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_prepare_remove_damage_counter(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect) {
    bool changed = false;
    for (std::uint16_t index = 0; index < state->targets.count; ++index) {
        const OfficialCardRefPod ref = state->targets.values[index].card;
        OfficialCardStatePod* card = official_pod_card(state, ref);
        if (card == nullptr) continue;
        if (official_effect_blocks_damage_counter(state, rules, ref)
            || official_card_cannot_move_damage_counter(*card)) {
            continue;
        }
        std::int32_t maximum = official_effect_value(*state, effect, 0);
        const std::int32_t damage_counters = card->damage / 10;
        if (maximum > damage_counters) maximum = damage_counters;
        if (maximum >= 2) {
            changed = true;
            official_pod_mark_changed(state);
            state->context_card = ref;
            if (!official_begin_effect_selection(
                    state,
                    OfficialSelectTypeId::kCount,
                    kOfficialSelectContextRemoveDamageCounterCount,
                    state->effect_interpreter.effect_owner,
                    1,
                    1,
                    OfficialEffectResumeKind::kRemoveDamageCounter)) {
                return OfficialEffectInterpreterResult::kError;
            }
            for (std::int32_t count = 1; count <= maximum; ++count) {
                OfficialSelectOptionPod option{};
                option.type = static_cast<std::uint8_t>(
                    OfficialSelectOptionTypeId::kNumber);
                option.params[0] = static_cast<std::int16_t>(count);
                if (!official_pod_push(
                        state,
                        &state->options,
                        option,
                        OfficialPodError::kOptionOverflow)) {
                    return OfficialEffectInterpreterResult::kError;
                }
            }
            return OfficialEffectInterpreterResult::kNeedsAction;
        }
        if (maximum == 1) {
            changed = true;
            official_pod_mark_changed(state);
            official_pod_heal(state, ref, 10);
            state->removed_damage_counter = 1;
        }
    }
    if (!changed) state->control_flags |= kOfficialBreakEffectFlag;
    official_finish_effect_iteration(state);
    return official_advance_effect_interpreter(state, rules);
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_apply_effect_action(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (!official_validate_effect_action(state, option_indices, count)) {
        return OfficialEffectInterpreterResult::kError;
    }
    const OfficialEffectResumeKind resume = static_cast<OfficialEffectResumeKind>(
        state->effect_interpreter.resume_kind);
    const std::uint64_t absolute = static_cast<std::uint64_t>(
        state->effect_interpreter.effect_offset) + state->effect_interpreter.effect_index;
    const std::uint32_t total = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kEffects)];
    if (absolute >= total) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, absolute);
        return OfficialEffectInterpreterResult::kError;
    }
    const OfficialEffectRule& effect = rules.effects[absolute];

    if (resume == OfficialEffectResumeKind::kSelectActivate
        || resume == OfficialEffectResumeKind::kSelectEffect) {
        if (count != 1) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
            return OfficialEffectInterpreterResult::kError;
        }
        const std::uint8_t option_type = state->options.values[
            option_indices[0]].type;
        if (option_type != static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kYes)
            && option_type != static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kNo)) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, option_type);
            return OfficialEffectInterpreterResult::kError;
        }
        const bool selected_yes = option_type == static_cast<std::uint8_t>(
            OfficialSelectOptionTypeId::kYes);
        official_clear_effect_selection(state);
        state->effect_interpreter.awaiting_selection = 0;
        if (!selected_yes) {
            if (resume == OfficialEffectResumeKind::kSelectActivate) {
                state->control_flags |= kOfficialBreakEffectFlag;
            } else {
                state->effect_jump = official_effect_value(*state, effect, 0);
            }
        }
        official_finish_effect_iteration(state);
        return official_advance_effect_interpreter(state, rules);
    }

    if (resume == OfficialEffectResumeKind::kRemoveDamageCounter) {
        if (count != 1) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
            return OfficialEffectInterpreterResult::kError;
        }
        std::int32_t removed = state->options.values[option_indices[0]].params[0];
        if (removed < 1) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, removed);
            return OfficialEffectInterpreterResult::kError;
        }
        const OfficialCardRefPod target = state->context_card;
        if (official_effect_blocks_damage_counter(state, rules, target)) removed = 0;
        official_clear_effect_selection(state);
        state->effect_interpreter.awaiting_selection = 0;
        official_pod_heal(state, target, removed * 10);
        state->removed_damage_counter = removed;
        official_finish_effect_iteration(state);
        return official_advance_effect_interpreter(state, rules);
    }

    if (resume == OfficialEffectResumeKind::kDamageCounterAny) {
        if (count != 1) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
            return OfficialEffectInterpreterResult::kError;
        }
        const OfficialSelectOptionPod option = state->options.values[option_indices[0]];
        if (option.type != static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kCard)
            || option.resolved_card == 0) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, option_indices[0]);
            return OfficialEffectInterpreterResult::kError;
        }
        const OfficialCardRefPod target{option.resolved_card};
        std::int32_t damage = 10;
        if (official_effect_blocks_damage_counter(state, rules, target)) damage = 0;
        official_clear_effect_selection(state);
        state->effect_interpreter.awaiting_selection = 0;
        if (damage > 0) {
            official_pod_mark_changed(state);
            OfficialCardStatePod* card = official_pod_card(state, target);
            if (card != nullptr) card->damage += damage;
        }
        --state->remain_damage_counter;
        if (state->remain_damage_counter <= 0) {
            state->remain_damage_counter = 0;
            official_finish_effect_iteration(state);
            return official_advance_effect_interpreter(state, rules);
        }
        const OfficialEffectInterpreterResult selection =
            official_prepare_damage_counter_any_selection(state);
        if (selection == OfficialEffectInterpreterResult::kComplete) {
            return official_advance_effect_interpreter(state, rules);
        }
        return selection;
    }

    if (resume == OfficialEffectResumeKind::kEnergyLoop) {
        if (count == 0) {
            official_clear_effect_selection(state);
            state->effect_interpreter.awaiting_selection = 0;
            official_complete_energy_selection_scratch(state);
            official_finish_effect_iteration(state);
            return official_advance_effect_interpreter(state, rules);
        }
        const OfficialSelectOptionPod option = state->options.values[option_indices[0]];
        OfficialCardRefPod ref{};
        if (!official_energy_ref_from_option(state, option, &ref)) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, option_indices[0]);
            return OfficialEffectInterpreterResult::kError;
        }
        official_pod_push(
            state,
            &state->selected_list,
            ref,
            OfficialPodError::kSelectionOverflow);
        if (!official_move_selected_energy(state, ref)) {
            return OfficialEffectInterpreterResult::kError;
        }
        state->remain_energy_cost -= option.params[4];
        ++state->selected_energy_card_count;
        const bool maximum_reached = state->selected_energy_card_count >= state->energy_cost;
        const bool may_stop = state->remain_energy_cost <= 0
            || (official_interpreter_has_flag(effect, kOfficialEffectEnergyMaxSelect)
                && state->selected_energy_card_count > 0);
        if (maximum_reached || (may_stop && !official_interpreter_has_flag(
                effect, kOfficialEffectEnergyMaxSelect))) {
            official_clear_effect_selection(state);
            state->effect_interpreter.awaiting_selection = 0;
            official_complete_energy_selection_scratch(state);
            official_finish_effect_iteration(state);
            return official_advance_effect_interpreter(state, rules);
        }
        const std::uint8_t context = state->select_context;
        const std::int8_t player = state->select_player;
        official_begin_effect_selection(
            state,
            OfficialSelectTypeId::kEnergy,
            context,
            player,
            may_stop ? 0 : 1,
            1,
            OfficialEffectResumeKind::kEnergyLoop);
        if (!official_prepare_attached_options(
                state, rules, OfficialSelectOptionTypeId::kEnergy)) {
            return OfficialEffectInterpreterResult::kError;
        }
        for (std::uint16_t index = 0; index < state->options.count;) {
            bool selected_before = false;
            OfficialCardRefPod candidate{};
            if (!official_energy_ref_from_option(
                    state, state->options.values[index], &candidate)) {
                official_pod_fail(
                    state, OfficialPodError::kInvalidAction, index);
                return OfficialEffectInterpreterResult::kError;
            }
            for (std::uint16_t prior = 0; prior < state->selected_list.count; ++prior) {
                if (candidate == state->selected_list.values[prior]) {
                    selected_before = true;
                    break;
                }
            }
            if (!selected_before) {
                ++index;
                continue;
            }
            for (std::uint16_t move = index + 1; move < state->options.count; ++move) {
                state->options.values[move - 1] = state->options.values[move];
            }
            --state->options.count;
        }
        if (state->options.count == 0) {
            official_clear_effect_selection(state);
            state->effect_interpreter.awaiting_selection = 0;
            official_complete_energy_selection_scratch(state);
            official_finish_effect_iteration(state);
            return official_advance_effect_interpreter(state, rules);
        }
        return OfficialEffectInterpreterResult::kNeedsAction;
    }

    if (resume == OfficialEffectResumeKind::kEvolve) {
        if (count > 0) {
            const OfficialSelectOptionPod option = state->options.values[option_indices[0]];
            const OfficialArea target_area = static_cast<OfficialArea>(option.params[2]);
            if (state->select_player < 0 || state->select_player > 1
                || (target_area != OfficialArea::kActive
                    && target_area != OfficialArea::kBench)
                || option.params[3] < 0) {
                official_pod_fail(
                    state, OfficialPodError::kInvalidAction, option_indices[0]);
                return OfficialEffectInterpreterResult::kError;
            }
            const OfficialCardRefPod target_ref = official_pod_get_zone_card(
                state,
                state->select_player,
                target_area,
                static_cast<std::uint16_t>(option.params[3]));
            if (!official_pod_ok(state)) {
                return OfficialEffectInterpreterResult::kError;
            }
            if (!official_effect_evolve_card(
                    state,
                    rules,
                    OfficialCardRefPod{option.resolved_card},
                    target_ref)) {
                official_pod_fail(state, OfficialPodError::kInvalidAction, option_indices[0]);
                return OfficialEffectInterpreterResult::kError;
            }
            official_pod_mark_changed(state);
        }
        official_clear_effect_selection(state);
        state->effect_interpreter.awaiting_selection = 0;
        official_finish_effect_iteration(state);
        return official_advance_effect_interpreter(state, rules);
    }

    state->targets.count = 0;
    for (std::uint16_t index = 0; index < count; ++index) {
        const OfficialSelectOptionPod& option = state->options.values[option_indices[index]];
        OfficialCardRefPod selected_ref{option.resolved_card};
        const auto option_type = static_cast<OfficialSelectOptionTypeId>(option.type);
        if ((option_type == OfficialSelectOptionTypeId::kEnergyCard
                || option_type == OfficialSelectOptionTypeId::kToolCard)
            && !official_attached_ref_from_option(state, option, &selected_ref)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, option_indices[index]);
            return OfficialEffectInterpreterResult::kError;
        }
        if (selected_ref.index == 0
            || !official_pod_push(
                state,
                &state->targets,
                official_pod_area_ref(state, selected_ref),
                OfficialPodError::kSelectionOverflow)) {
            return OfficialEffectInterpreterResult::kError;
        }
    }
    official_clear_effect_selection(state);
    state->effect_interpreter.awaiting_selection = 0;
    if (resume != OfficialEffectResumeKind::kApplyPrimitive) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, static_cast<int>(resume));
        return OfficialEffectInterpreterResult::kError;
    }
    if (count == 0 && effect.values[kEffectLoopCount] > 0) {
        // Official ActivateEffect2 stops a loopCount effect when the selected
        // target list is empty. This terminates the current repeated effect,
        // but the enclosing skill chain continues to its following effects
        // (for example, post-search deck shuffle).
        // The effect-level changed bit remains set on this path in the official
        // state snapshot.
        official_pod_mark_changed(state);
        if (state->effect_interpreter.repeat_count > 0) {
            state->effect_interpreter.repeat_index =
                static_cast<std::uint16_t>(
                    state->effect_interpreter.repeat_count - 1U);
        }
        state->control_flags &= static_cast<std::uint8_t>(
            ~kOfficialBreakEffectFlag);
        official_finish_effect_iteration(state);
        return official_advance_effect_interpreter(state, rules);
    }
    if (effect.values[kEffectType]
        == static_cast<std::int32_t>(OfficialEffectTypeId::kRemoveDamageCounter)) {
        return official_prepare_remove_damage_counter(state, rules, effect);
    }
    const OfficialEffectInterpreterResult applied = official_apply_effect_and_finish(
        state, rules, effect);
    if (applied == OfficialEffectInterpreterResult::kError) return applied;
    return official_advance_effect_interpreter(state, rules);
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_begin_effect_range(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint32_t effect_offset,
    std::uint16_t effect_count,
    OfficialAreaRefPod effect_card,
    std::int32_t effect_owner,
    std::uint8_t first_condition_count = 0,
    std::uint32_t step_budget = kOfficialDefaultEffectStepBudget,
    std::uint16_t start_index = 0,
    bool clear_target_lists = true) {
    const std::uint32_t total = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kEffects)];
    if (effect_owner < 0 || effect_owner > 1
        || start_index > effect_count
        || static_cast<std::uint64_t>(effect_offset) + effect_count > total) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, effect_offset);
        return OfficialEffectInterpreterResult::kError;
    }
    official_clear_effect_selection(state);
    state->effect_interpreter = OfficialEffectInterpreterPod{};
    state->effect_interpreter.effect_offset = effect_offset;
    state->effect_interpreter.effect_count = effect_count;
    state->effect_interpreter.effect_index = start_index;
    state->effect_interpreter.first_condition_count = first_condition_count;
    state->effect_interpreter.active = start_index == effect_count ? 0 : 1;
    state->effect_interpreter.effect_owner = static_cast<std::int8_t>(effect_owner);
    state->effect_interpreter.effect_card = effect_card;
    state->effect_interpreter.step_budget = step_budget;
    state->effect_state.ability.effect_card = effect_card;
    state->effect_state.ability.use_player = static_cast<std::int8_t>(effect_owner);
    state->effect_state.effect_rate = 1;
    state->effect_state.on_effect = 0;
    state->effect_jump = 0;
    if (clear_target_lists) {
        state->targets.count = 0;
    }
    state->control_flags &= static_cast<std::uint8_t>(
        ~(kOfficialChangedFlag | kOfficialBreakEffectFlag | kOfficialInterpreterLoopStopFlag));
    return official_advance_effect_interpreter(state, rules);
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_begin_skill_effects_at(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t skill_id,
    OfficialAreaRefPod effect_card,
    std::int32_t use_player,
    std::uint16_t start_index,
    std::uint32_t step_budget = kOfficialDefaultEffectStepBudget) {
    const OfficialSkillRule* skill = official_skill_rule(
        rules, static_cast<std::uint32_t>(skill_id));
    if (skill == nullptr) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, skill_id);
        return OfficialEffectInterpreterResult::kError;
    }
    state->effect_state.ability.skill_id = skill_id;
    return official_begin_effect_range(
        state,
        rules,
        static_cast<std::uint32_t>(skill->values[kSkillEffectOffset]),
        static_cast<std::uint16_t>(skill->values[kSkillEffectCount]),
        effect_card,
        use_player,
        static_cast<std::uint8_t>(skill->values[kSkillFirstConditionCount]),
        step_budget,
        start_index);
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_begin_skill_effects(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t skill_id,
    OfficialAreaRefPod effect_card,
    std::int32_t use_player,
    std::uint32_t step_budget = kOfficialDefaultEffectStepBudget) {
    return official_begin_skill_effects_at(
        state, rules, skill_id, effect_card, use_player, 0, step_budget);
}

PTCG_OFFICIAL_INTERPRETER_HD inline OfficialEffectInterpreterResult
official_begin_attack_effects(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t attack_id,
    bool post_effect,
    OfficialAreaRefPod attacker,
    std::int32_t use_player,
    std::uint32_t step_budget = kOfficialDefaultEffectStepBudget) {
    const OfficialAttackRule* attack = official_attack_rule(
        rules, static_cast<std::uint32_t>(attack_id));
    if (attack == nullptr) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, attack_id);
        return OfficialEffectInterpreterResult::kError;
    }
    state->current_attack_id = attack_id;
    state->attacker = attacker.card;
    state->post_attack_effect = post_effect ? 1 : 0;
    state->effect_state.ability = OfficialActivateAbilityPod{};
    return official_begin_effect_range(
        state,
        rules,
        static_cast<std::uint32_t>(attack->values[
            post_effect ? kAttackPostEffectOffset : kAttackPreEffectOffset]),
        static_cast<std::uint16_t>(attack->values[
            post_effect ? kAttackPostEffectCount : kAttackPreEffectCount]),
        attacker,
        use_player,
        0,
        step_budget,
        0,
        false);
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_INTERPRETER_HD
