#pragma once

#include <cstdint>

#include "ptcg_cuda/official_turn_flow_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_ATTACK_HD __host__ __device__
#else
#define PTCG_OFFICIAL_ATTACK_HD
#endif

namespace ptcg::cuda_engine {

enum class OfficialAttackResult : std::int32_t {
    kComplete = 0,
    kNeedsAction = 1,
    kError = 2,
};

enum class OfficialAttackStage : std::uint8_t {
    kIdle = 0,
    kCopySelection = 1,
    kPreEffects = 2,
    kPostEffects = 3,
    kSupporterEffects = 4,
    kAfterAttackTriggers = 5,
    kRefresh = 6,
    kSecondAttackSelection = 7,
    kTurnEnd = 8,
    kActiveReplacement = 9,
};

constexpr std::uint8_t kOfficialSelectContextAttack = 36;
constexpr std::uint64_t kAttackCopyEnemy = 1ULL << 0U;
constexpr std::uint64_t kAttackCopyEnemyTerastal = 1ULL << 1U;
constexpr std::uint64_t kAttackCopyEnemyCoin = 1ULL << 2U;
constexpr std::uint64_t kAttackCopyBenchN = 1ULL << 3U;
constexpr std::uint64_t kAttackCopyEnemyDeckTop10 = 1ULL << 4U;
constexpr std::uint64_t kAttackNoTargetEffect = 1ULL << 5U;
constexpr std::uint64_t kAttackCannotUseFirstTurn = 1ULL << 8U;
constexpr std::uint64_t kAttackCannotUseSameNamePreTurn = 1ULL << 9U;
constexpr std::uint64_t kAttackCanOnlyUseEnemyPrizeOne = 1ULL << 10U;
constexpr std::uint64_t kAttackDeckTopAttack = 1ULL << 11U;
constexpr std::uint64_t kAttackDeckTopSupporter = 1ULL << 12U;
constexpr std::uint64_t kAttackCanUseBench = 1ULL << 17U;
constexpr std::uint64_t kAttackCanUseFirst = 1ULL << 18U;
constexpr std::uint64_t kAttackNoCheckCondition = 1ULL << 19U;

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_attack_after_body(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_attack_enter_selected_body(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

PTCG_OFFICIAL_ATTACK_HD inline std::uint8_t official_attack_this_turn_flags(
    const OfficialCardStatePod& card) {
    return static_cast<std::uint8_t>(card.this_turn[3] & 0xffU);
}

PTCG_OFFICIAL_ATTACK_HD inline bool official_attack_card_can_attack(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod attacker_ref) {
    const OfficialCardStatePod* attacker = official_pod_card(state, attacker_ref);
    if (attacker == nullptr || attacker->player < 0 || attacker->player > 1) return false;
    const std::uint8_t turn_flags = official_attack_this_turn_flags(*attacker);
    if (official_continual_flag(*attacker, 24) || (turn_flags & (1U << 2U)) != 0) {
        return false;
    }
    if ((turn_flags & (1U << 3U)) != 0
        && official_attached_energy_cards(
            *state, attacker_ref, -1, false, rules) <= 2) {
        return false;
    }
    return official_pod_ok(state);
}

PTCG_OFFICIAL_ATTACK_HD inline bool official_attack_state_condition(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialCardStatePod& attacker,
    const OfficialAttackRule& attack,
    std::int32_t source_attack_id) {
    const std::int32_t attack_id = attack.values[kAttackId];
    const std::int16_t cannot_use_1 = static_cast<std::int16_t>(
        attacker.this_turn[0] & 0xffffU);
    const std::int16_t cannot_use_2 = static_cast<std::int16_t>(
        (attacker.this_turn[0] >> 16U) & 0xffffU);
    if ((cannot_use_1 == attack_id
            || cannot_use_2 == attack_id
            || attacker.cannot_use_attack_id_non_active == attack_id)
        && (source_attack_id == 0 || source_attack_id == attack_id)) {
        return false;
    }
    if ((attack.flags & kAttackCannotUseFirstTurn) != 0 && state->turn <= 2) {
        return false;
    }
    if ((attack.flags & kAttackCannotUseSameNamePreTurn) != 0
        && state->turn_histories[2].attack_id > 0) {
        const OfficialAttackRule* previous = official_attack_rule(
            rules, static_cast<std::uint32_t>(state->turn_histories[2].attack_id));
        if (previous == nullptr) {
            official_pod_fail(
                state, OfficialPodError::kRulePackBounds,
                state->turn_histories[2].attack_id);
            return false;
        }
        if (previous->values[kAttackNameId] == attack.values[kAttackNameId]) return false;
    }
    if ((attack.flags & kAttackCanOnlyUseEnemyPrizeOne) != 0
        && state->players[1 - attacker.player].prize.count != 1) {
        return false;
    }
    return true;
}

PTCG_OFFICIAL_ATTACK_HD inline bool official_attack_first_conditions(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod attacker_ref,
    const OfficialAttackRule& attack) {
    if (attack.values[kAttackDamage] != 0
        || (attack.flags & kAttackNoCheckCondition) != 0) {
        return true;
    }
    const std::int32_t pre_count = attack.values[kAttackPreEffectCount];
    const std::int32_t post_count = attack.values[kAttackPostEffectCount];
    if (pre_count > 0 && post_count > 0) return true;
    const std::int32_t offset = attack.values[kAttackPostEffectOffset];
    const std::uint32_t total = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kEffects)];
    if (offset < 0 || post_count < 0
        || static_cast<std::uint64_t>(offset) + post_count > total) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, offset);
        return false;
    }
    const OfficialCardStatePod* attacker = official_pod_card(state, attacker_ref);
    if (attacker == nullptr) return false;
    for (std::int32_t index = 0; index < post_count; ++index) {
        const OfficialEffectRule& effect = rules.effects[offset + index];
        if ((effect.flags & kOfficialEffectIsCondition) == 0
            || effect.values[kEffectFailSkip] != 0) {
            break;
        }
        const OfficialConditionResult result = official_satisfy_condition(
            state,
            rules,
            rules.effects + offset,
            post_count,
            index,
            official_pod_area_ref(state, attacker_ref),
            attacker->player);
        if (result == OfficialConditionResult::kFalse) return false;
        if (result != OfficialConditionResult::kTrue) return false;
    }
    return official_pod_ok(state);
}

PTCG_OFFICIAL_ATTACK_HD inline bool official_attack_initial_action_is_legal(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod attacker_ref,
    const OfficialAttackRule& attack,
    std::int32_t source_attack_id) {
    const OfficialCardStatePod* attacker = official_pod_card(state, attacker_ref);
    if (attacker == nullptr || attacker->player != official_active_player(*state)) {
        return false;
    }
    const OfficialArea area = static_cast<OfficialArea>(attacker->area);
    if (area == OfficialArea::kActive) {
        if (state->players[attacker->player].active.count == 0
            || state->players[attacker->player].active.values[0] != attacker_ref) {
            return false;
        }
    } else if (area == OfficialArea::kBench) {
        if ((attack.flags & kAttackCanUseBench) == 0 || state->turn < 2) return false;
    } else {
        return false;
    }
    const OfficialBadStatus status = official_pod_bad_status(
        state->players[attacker->player]);
    if (status == OfficialBadStatus::kAsleep
        || status == OfficialBadStatus::kParalyzed) {
        return false;
    }
    if (!official_attack_card_can_attack(state, rules, attacker_ref)) return false;
    if (state->turn < 2
        && (attack.flags & kAttackCanUseFirst) == 0
        && !official_continual_flag(*attacker, 22)) {
        return false;
    }
    if (!official_attack_first_conditions(state, rules, attacker_ref, attack)
        || !official_attack_state_condition(
            state, rules, *attacker, attack, source_attack_id)) {
        return false;
    }
    const OfficialAttackRule* energy_attack = &attack;
    if (source_attack_id > 0
        && source_attack_id != attack.values[kAttackId]) {
        energy_attack = official_attack_rule(
            rules, static_cast<std::uint32_t>(source_attack_id));
        if (energy_attack == nullptr
            || !official_attack_state_condition(
                state, rules, *attacker, *energy_attack, source_attack_id)) {
            return false;
        }
    }
    return official_attack_energy_extra(
        state, rules, *attacker, attacker_ref, *energy_attack) >= 0
        && official_pod_ok(state);
}

PTCG_OFFICIAL_ATTACK_HD inline void official_attack_prepare_selection(
    OfficialStatePod* state,
    std::int32_t player,
    std::int32_t minimum) {
    official_clear_effect_selection(state);
    state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kAttack);
    state->select_context = kOfficialSelectContextAttack;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = static_cast<std::int16_t>(minimum);
    state->select_max = 1;
}

PTCG_OFFICIAL_ATTACK_HD inline bool official_attack_add_option(
    OfficialStatePod* state,
    std::int32_t attack_id,
    std::int32_t source_attack_id,
    std::int32_t bench_index = -1) {
    OfficialSelectOptionPod option{};
    option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kAttack);
    option.params[0] = static_cast<std::int16_t>(attack_id);
    option.params[1] = static_cast<std::int16_t>(source_attack_id);
    option.params[2] = static_cast<std::int16_t>(bench_index);
    option.option_equiv = static_cast<std::uint16_t>(attack_id);
    return official_pod_push(
        state, &state->options, option, OfficialPodError::kOptionOverflow);
}

PTCG_OFFICIAL_ATTACK_HD inline bool official_attack_card_attack_range(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialCardRule& card,
    std::int32_t* offset,
    std::int32_t* count) {
    *offset = card.values[kCardAttackOffset];
    *count = card.values[kCardAttackCount];
    const std::uint32_t total = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kCardAttackIds)];
    if (*offset < 0 || *count < 0
        || static_cast<std::uint64_t>(*offset) + *count > total) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, *offset);
        return false;
    }
    return true;
}

PTCG_OFFICIAL_ATTACK_HD inline bool official_attack_add_card_attacks(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod card_ref,
    std::int32_t source_attack_id,
    bool skip_copy_enemy) {
    const OfficialCardStatePod* card = official_pod_card(state, card_ref);
    const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    if (master == nullptr) return false;
    std::int32_t offset = 0;
    std::int32_t count = 0;
    if (!official_attack_card_attack_range(
            state, rules, *master, &offset, &count)) return false;
    for (std::int32_t index = 0; index < count; ++index) {
        const std::int32_t attack_id = static_cast<std::int32_t>(
            rules.card_attack_ids[offset + index]);
        const OfficialAttackRule* attack = official_attack_rule(
            rules, static_cast<std::uint32_t>(attack_id));
        if (attack == nullptr) {
            official_pod_fail(state, OfficialPodError::kRulePackBounds, attack_id);
            return false;
        }
        if (skip_copy_enemy && (attack->flags & kAttackCopyEnemy) != 0) continue;
        if (!official_attack_add_option(
                state, attack_id, source_attack_id)) return false;
    }
    return true;
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_attack_begin_turn_end(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    state->current_attack_id = 0;
    state->attacker = {};
    state->attack_flow_stage = static_cast<std::uint8_t>(OfficialAttackStage::kTurnEnd);
    const OfficialTurnFlowResult turn = official_begin_turn_end(state, rules);
    if (turn == OfficialTurnFlowResult::kError) return OfficialAttackResult::kError;
    if (turn == OfficialTurnFlowResult::kNeedsAction) {
        return OfficialAttackResult::kNeedsAction;
    }
    state->attack_flow_stage = 0;
    return OfficialAttackResult::kComplete;
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult
official_attack_prepare_second_or_turn_end(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (state->second_attack != 0) {
        state->second_attack = 0;
        return official_attack_begin_turn_end(state, rules);
    }
    const std::int32_t player = official_active_player(*state);
    const OfficialCardStatePod* attacker = official_pod_card(state, state->attacker);
    if (attacker == nullptr || player < 0 || player > 1
        || state->players[player].active.count == 0
        || state->players[player].active.values[0] != state->attacker
        || !official_continual_flag(*attacker, 33)
        || official_pod_bad_status(state->players[player]) == OfficialBadStatus::kAsleep
        || official_pod_bad_status(state->players[player]) == OfficialBadStatus::kParalyzed) {
        return official_attack_begin_turn_end(state, rules);
    }
    const OfficialCardRule* master = official_card_rule(
        rules, static_cast<std::uint32_t>(attacker->card_id));
    if (master == nullptr
        || !official_card_has_attack(
            rules, *master, state->current_attack_id)
        || !official_attack_card_can_attack(state, rules, state->attacker)) {
        return official_attack_begin_turn_end(state, rules);
    }
    official_attack_prepare_selection(state, player, 0);
    std::int32_t offset = 0;
    std::int32_t count = 0;
    if (!official_attack_card_attack_range(
            state, rules, *master, &offset, &count)) {
        return OfficialAttackResult::kError;
    }
    for (std::int32_t index = 0; index < count && official_pod_ok(state); ++index) {
        const std::int32_t attack_id = static_cast<std::int32_t>(
            rules.card_attack_ids[offset + index]);
        const OfficialAttackRule* candidate = official_attack_rule(
            rules, static_cast<std::uint32_t>(attack_id));
        if (candidate == nullptr) {
            official_pod_fail(state, OfficialPodError::kRulePackBounds, attack_id);
            break;
        }
        if (official_attack_energy_extra(
                state, rules, *attacker, state->attacker, *candidate) < 0
            || !official_attack_first_conditions(
                state, rules, state->attacker, *candidate)
            || !official_attack_state_condition(
                state, rules, *attacker, *candidate, 0)) {
            continue;
        }
        if (!official_attack_add_option(state, attack_id, 0)) break;
    }
    if (!official_pod_ok(state)) return OfficialAttackResult::kError;
    if (state->options.count == 0) {
        official_clear_effect_selection(state);
        return official_attack_begin_turn_end(state, rules);
    }
    state->attack_flow_stage = static_cast<std::uint8_t>(
        OfficialAttackStage::kSecondAttackSelection);
    return OfficialAttackResult::kNeedsAction;
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult
official_attack_begin_refresh_after_active_check(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    state->attack_flow_stage = static_cast<std::uint8_t>(OfficialAttackStage::kRefresh);
    const OfficialTurnFlowResult refresh = official_begin_refresh(state, rules);
    if (refresh == OfficialTurnFlowResult::kError) return OfficialAttackResult::kError;
    if (refresh == OfficialTurnFlowResult::kNeedsAction) {
        return OfficialAttackResult::kNeedsAction;
    }
    if (state->game_result != static_cast<std::uint8_t>(
            OfficialGameResult::kNone)) {
        // The official State::step stops as soon as prize/KO refresh reaches a
        // terminal result.  Its pending attack continuations are never run, so
        // currentAttackId and attacker remain observable in the terminal state.
        state->attack_flow_stage = 0;
        return OfficialAttackResult::kComplete;
    }
    return official_attack_prepare_second_or_turn_end(state, rules);
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult
official_attack_begin_refresh_phase(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    const OfficialContinualRefreshResult continual =
        official_refresh_continual_effects(state, rules);
    if (continual != OfficialContinualRefreshResult::kApplied) {
        if (continual == OfficialContinualRefreshResult::kDepthLimit) {
            official_pod_fail(state, OfficialPodError::kInterpreterBudget, 10);
        } else if (official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kInvalidAction,
                -464);
        }
        return OfficialAttackResult::kError;
    }
    official_rebuild_active_replacement_mask(state);
    official_mark_active_check_changed(state);
    if (state->pending_active_replacement_mask == 0) {
        return official_attack_begin_refresh_after_active_check(state, rules);
    }
    state->attack_flow_stage = static_cast<std::uint8_t>(
        OfficialAttackStage::kActiveReplacement);
    state->reserved_flow = static_cast<std::uint8_t>(
        OfficialKnockoutStage::kActiveReplacement);
    const std::int32_t active = official_active_player(*state);
    const std::int32_t order[2] = {1 - active, active};
    for (int index = 0; index < 2; ++index) {
        const std::int32_t player = order[index];
        if ((state->pending_active_replacement_mask & (1U << player)) == 0) {
            continue;
        }
        const OfficialKnockoutResult replacement =
            official_begin_active_replacement(state, player);
        if (replacement == OfficialKnockoutResult::kError) {
            return OfficialAttackResult::kError;
        }
        if (replacement == OfficialKnockoutResult::kNeedsAction) {
            return OfficialAttackResult::kNeedsAction;
        }
    }
    state->reserved_flow = 0;
    return official_attack_begin_refresh_after_active_check(state, rules);
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult
official_attack_resume_active_replacement(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (count != 1 || option_indices == nullptr
        || option_indices[0] >= state->options.count) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        return OfficialAttackResult::kError;
    }
    const std::int32_t player = state->select_player;
    const OfficialCardRefPod ref{
        state->options.values[option_indices[0]].resolved_card};
    const std::int32_t bench_index = official_pod_find_in_list(
        state->players[player].bench, ref);
    if (bench_index < 0
        || !official_switch_pokemon(
            state, rules, player, static_cast<std::uint16_t>(bench_index))) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, ref.index);
        return OfficialAttackResult::kError;
    }
    state->pending_active_replacement_mask &= static_cast<std::uint16_t>(
        ~(1U << player));
    official_clear_effect_selection(state);

    const std::int32_t active = official_active_player(*state);
    const std::int32_t order[2] = {1 - active, active};
    for (int index = 0; index < 2; ++index) {
        const std::int32_t next_player = order[index];
        if ((state->pending_active_replacement_mask
                & (1U << next_player)) == 0) {
            continue;
        }
        const OfficialKnockoutResult replacement =
            official_begin_active_replacement(state, next_player);
        if (replacement == OfficialKnockoutResult::kError) {
            return OfficialAttackResult::kError;
        }
        if (replacement == OfficialKnockoutResult::kNeedsAction) {
            return OfficialAttackResult::kNeedsAction;
        }
    }
    state->reserved_flow = 0;
    return official_attack_begin_refresh_after_active_check(state, rules);
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_attack_after_body(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    state->attack_damage_change = 0;
    state->post_attack_effect = 0;
    state->post_effect_activate = 0;
    state->fail_attack = 0;
    official_clear_resolved_ability(state);
    if (state->triggers.count > 0 || state->temporary_triggers.count > 0) {
        if (official_refresh_continual_effects(state, rules)
            != OfficialContinualRefreshResult::kApplied) {
            if (official_pod_ok(state)) {
                official_pod_fail(
                    state,
                    OfficialPodError::kInvalidAction,
                    -567);
            }
            return OfficialAttackResult::kError;
        }
        state->attack_flow_stage = static_cast<std::uint8_t>(
            OfficialAttackStage::kAfterAttackTriggers);
        const OfficialTriggerResolverResult trigger = official_begin_trigger_resolution(
            state, rules, 0);
        if (trigger == OfficialTriggerResolverResult::kError) {
            if (official_pod_ok(state)) {
                official_pod_fail(
                    state,
                    OfficialPodError::kInvalidAction,
                    -589);
            }
            return OfficialAttackResult::kError;
        }
        if (trigger == OfficialTriggerResolverResult::kNeedsAction) {
            return OfficialAttackResult::kNeedsAction;
        }
    }
    return official_attack_begin_refresh_phase(state, rules);
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_attack_begin_post_effects(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (state->fail_attack != 0) return official_attack_after_body(state, rules);
    state->attack_flow_stage = static_cast<std::uint8_t>(
        OfficialAttackStage::kPostEffects);
    const OfficialEffectInterpreterResult post = official_begin_attack_effects(
        state,
        rules,
        state->current_attack_id,
        true,
        official_pod_area_ref(state, state->attacker),
        official_active_player(*state));
    if (post == OfficialEffectInterpreterResult::kError) {
        return OfficialAttackResult::kError;
    }
    if (post == OfficialEffectInterpreterResult::kNeedsAction) {
        return OfficialAttackResult::kNeedsAction;
    }
    return official_attack_after_body(state, rules);
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_attack_after_pre_effects(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (state->fail_attack != 0) return official_attack_after_body(state, rules);
    const OfficialContinualRefreshResult refreshed = official_refresh_continual_effects(
        state, rules);
    if (refreshed != OfficialContinualRefreshResult::kApplied) {
        return OfficialAttackResult::kError;
    }
    const std::int32_t opponent = 1 - official_active_player(*state);
    if (state->players[opponent].active.count > 0) {
        const OfficialAttackRule* attack = official_attack_rule(
            rules, static_cast<std::uint32_t>(state->current_attack_id));
        OfficialCardStatePod* attacker = official_pod_card(state, state->attacker);
        if (attack == nullptr || attacker == nullptr) {
            official_pod_fail(
                state, OfficialPodError::kRulePackBounds, state->current_attack_id);
            return OfficialAttackResult::kError;
        }
        std::int32_t base_damage = attack->values[kAttackDamage]
            + state->attack_damage_change;
        const OfficialCardRule* master = official_card_rule(
            rules, static_cast<std::uint32_t>(attacker->card_id));
        if (master != nullptr
            && official_card_has_attack(
                rules, *master, state->current_attack_id)) {
            base_damage += official_packed_i16(attacker->this_turn, 8);
        }
        state->last_attack_damage = 0;
        const OfficialCardRefPod target_ref = state->players[opponent].active.values[0];
        official_apply_attack_damage_to_card(
            state, rules, target_ref, base_damage, true);
        if (!official_pod_ok(state)) return OfficialAttackResult::kError;
    }
    return official_attack_begin_post_effects(state, rules);
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_attack_begin_pre_effects(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    state->turn_histories[0].attack_id = static_cast<std::int16_t>(
        state->source_attack_id);
    state->turn_histories[0].attack_card = state->attacker;
    state->effect_state = OfficialEffectStatePod{};
    state->attack_flow_stage = static_cast<std::uint8_t>(
        OfficialAttackStage::kPreEffects);
    const OfficialEffectInterpreterResult pre = official_begin_attack_effects(
        state,
        rules,
        state->current_attack_id,
        false,
        official_pod_area_ref(state, state->attacker),
        official_active_player(*state));
    if (pre == OfficialEffectInterpreterResult::kError) {
        return OfficialAttackResult::kError;
    }
    if (pre == OfficialEffectInterpreterResult::kNeedsAction) {
        return OfficialAttackResult::kNeedsAction;
    }
    return official_attack_after_pre_effects(state, rules);
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_attack_prepare_special(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    bool regular_copy_selection = false) {
    const OfficialAttackRule* attack = official_attack_rule(
        rules, static_cast<std::uint32_t>(state->current_attack_id));
    const OfficialCardStatePod* attacker = official_pod_card(state, state->attacker);
    if (attack == nullptr || attacker == nullptr) {
        official_pod_fail(
            state, OfficialPodError::kRulePackBounds, state->current_attack_id);
        return OfficialAttackResult::kError;
    }
    const std::int32_t player = official_active_player(*state);
    if ((attack->flags & kAttackDeckTopAttack) != 0) {
        auto& deck = state->players[player].deck;
        if (deck.count == 0) return official_attack_after_body(state, rules);
        const OfficialCardRefPod top = official_pod_move_card(
            state,
            player,
            OfficialArea::kDeck,
            static_cast<std::uint16_t>(deck.count - 1),
            OfficialArea::kTrash,
            false);
        const OfficialCardStatePod* card = official_pod_card(state, top);
        const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
            rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) return OfficialAttackResult::kError;
        if (master->values[kCardType] != 0
            || master->values[kCardPokemonType] != 1
            || master->values[kCardAttackCount] == 0) {
            return official_attack_after_body(state, rules);
        }
        official_attack_prepare_selection(state, player, 1);
        if (!official_attack_add_card_attacks(
                state, rules, top, attack->values[kAttackId], false)) {
            return OfficialAttackResult::kError;
        }
    } else if ((attack->flags & kAttackCopyBenchN) != 0) {
        official_attack_prepare_selection(state, player, 1);
        const auto& bench = state->players[player].bench;
        for (std::uint16_t index = 0; index < bench.count; ++index) {
            const OfficialCardStatePod* card = official_pod_card(
                state, bench.values[index]);
            const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
                rules, static_cast<std::uint32_t>(card->card_id));
            if (master == nullptr) return OfficialAttackResult::kError;
            if ((master->flags & (1ULL << 16U)) == 0) continue;
            const std::uint16_t before = state->options.count;
            if (!official_attack_add_card_attacks(
                    state, rules, bench.values[index], attack->values[kAttackId], false)) {
                return OfficialAttackResult::kError;
            }
            std::uint16_t option = before;
            while (option < state->options.count) {
                const OfficialAttackRule* candidate = official_attack_rule(
                    rules,
                    static_cast<std::uint32_t>(state->options.values[option].params[0]));
                if (candidate != nullptr && (candidate->flags & kAttackCopyBenchN) != 0) {
                    official_pod_remove(state, &state->options, option);
                } else {
                    ++option;
                }
            }
        }
    } else if ((attack->flags
            & (kAttackCopyEnemy | kAttackCopyEnemyCoin | kAttackCopyEnemyTerastal)) != 0) {
        const std::int32_t enemy = 1 - player;
        if (state->players[enemy].active.count == 0) {
            return official_attack_after_body(state, rules);
        }
        const OfficialCardRefPod enemy_active = state->players[enemy].active.values[0];
        const OfficialCardStatePod* enemy_card = official_pod_card(state, enemy_active);
        const OfficialCardRule* enemy_master = enemy_card == nullptr ? nullptr
            : official_card_rule(rules, static_cast<std::uint32_t>(enemy_card->card_id));
        if (enemy_master == nullptr) return OfficialAttackResult::kError;
        if ((attack->flags & kAttackCopyEnemyTerastal) != 0
            && (enemy_master->flags & 1ULL) == 0) {
            return official_attack_after_body(state, rules);
        }
        official_attack_prepare_selection(state, player, 1);
        if (!official_attack_add_card_attacks(
                state,
                rules,
                enemy_active,
                attack->values[kAttackId],
                (attack->flags & kAttackCopyEnemy) != 0)) {
            return OfficialAttackResult::kError;
        }
    } else if ((attack->flags & kAttackCopyEnemyDeckTop10) != 0) {
        const std::int32_t enemy = 1 - player;
        auto& deck = state->players[enemy].deck;
        official_attack_prepare_selection(state, player, 0);
        for (std::int32_t draw = 0; draw < 10 && deck.count > 0; ++draw) {
            const OfficialCardRefPod top = official_pod_move_card(
                state,
                enemy,
                OfficialArea::kDeck,
                static_cast<std::uint16_t>(deck.count - 1),
                OfficialArea::kLooking,
                false);
            const OfficialCardStatePod* card = official_pod_card(state, top);
            const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
                rules, static_cast<std::uint32_t>(card->card_id));
            if (master == nullptr) return OfficialAttackResult::kError;
            if (master->values[kCardType] != 0) continue;
            std::int32_t offset = 0;
            std::int32_t count = 0;
            if (!official_attack_card_attack_range(
                    state, rules, *master, &offset, &count)) {
                return OfficialAttackResult::kError;
            }
            for (std::int32_t index = 0; index < count; ++index) {
                const std::int32_t attack_id = static_cast<std::int32_t>(
                    rules.card_attack_ids[offset + index]);
                if (official_attack_rule(
                        rules, static_cast<std::uint32_t>(attack_id)) == nullptr) {
                    official_pod_fail(
                        state, OfficialPodError::kRulePackBounds, attack_id);
                    return OfficialAttackResult::kError;
                }
                bool duplicate = false;
                for (std::uint16_t option = 0; option < state->options.count; ++option) {
                    if (state->options.values[option].params[0] == attack_id) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate && !official_attack_add_option(
                        state, attack_id, attack->values[kAttackId])) {
                    return OfficialAttackResult::kError;
                }
            }
        }
        while (state->looking.count > 0 && official_pod_ok(state)) {
            official_pod_move_card(
                state,
                enemy,
                OfficialArea::kLooking,
                0,
                OfficialArea::kDeck,
                false);
        }
        official_pod_shuffle_deck(state, enemy);
    } else if ((attack->flags & kAttackDeckTopSupporter) != 0) {
        auto& deck = state->players[player].deck;
        if (deck.count == 0) return official_attack_after_body(state, rules);
        const OfficialCardRefPod top = official_pod_move_card(
            state,
            player,
            OfficialArea::kDeck,
            static_cast<std::uint16_t>(deck.count - 1),
            OfficialArea::kTrash,
            false);
        const OfficialCardStatePod* card = official_pod_card(state, top);
        const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
            rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) return OfficialAttackResult::kError;
        if (master->values[kCardType] != 3 || master->values[kCardPlayId] <= 0) {
            return official_attack_after_body(state, rules);
        }
        state->attack_flow_stage = static_cast<std::uint8_t>(
            OfficialAttackStage::kSupporterEffects);
        const OfficialEffectInterpreterResult effect = official_begin_skill_effects(
            state,
            rules,
            master->values[kCardPlayId],
            official_pod_area_ref(state, state->attacker),
            player);
        if (effect == OfficialEffectInterpreterResult::kError) {
            return OfficialAttackResult::kError;
        }
        if (effect == OfficialEffectInterpreterResult::kNeedsAction) {
            return OfficialAttackResult::kNeedsAction;
        }
        return official_attack_after_body(state, rules);
    } else {
        return official_attack_begin_pre_effects(state, rules);
    }
    if (!official_pod_ok(state)) return OfficialAttackResult::kError;
    if (state->options.count == 0) {
        official_clear_effect_selection(state);
        return official_attack_after_body(state, rules);
    }
    if (regular_copy_selection) {
        // State::SpecialAttackProc exposes this as a normal Attack selection
        // while the parent attack frame is still the caller's stack frame.
        // The copied body is resumed through SelectedAttackId, rather than
        // through the CUDA CopySelection attack-flow stage.
        return OfficialAttackResult::kNeedsAction;
    }
    state->attack_flow_stage = static_cast<std::uint8_t>(
        OfficialAttackStage::kCopySelection);
    return OfficialAttackResult::kNeedsAction;
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_attack_enter_selected_body(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    ++state->turn_attack_count;
    if (state->turn_attack_count > 10000) {
        return official_attack_after_body(state, rules);
    }
    const OfficialAttackRule* attack = official_attack_rule(
        rules, static_cast<std::uint32_t>(state->current_attack_id));
    if (attack == nullptr) {
        official_pod_fail(
            state, OfficialPodError::kRulePackBounds, state->current_attack_id);
        return OfficialAttackResult::kError;
    }
    if ((attack->flags & kAttackCopyEnemyCoin) != 0) {
        state->coin_head_count = 0;
        if (!official_pod_coin(state)) return official_attack_after_body(state, rules);
    }
    return official_attack_prepare_special(state, rules);
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_begin_attack(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod attacker_ref,
    std::int32_t attack_id,
    std::int32_t source_attack_id = 0) {
    if (!official_pod_ok(state) || state->attack_flow_stage != 0
        || state->turn_flow_stage != 0 || state->refresh_flow_stage != 0
        || state->trigger_resolver.active != 0 || state->effect_interpreter.active != 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, attack_id);
        }
        return OfficialAttackResult::kError;
    }
    const OfficialAttackRule* attack = official_attack_rule(
        rules, static_cast<std::uint32_t>(attack_id));
    if (attack == nullptr
        || !official_attack_initial_action_is_legal(
            state, rules, attacker_ref, *attack, source_attack_id)) {
        if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, attack_id);
        }
        return OfficialAttackResult::kError;
    }
    state->current_attack_id = attack_id;
    state->source_attack_id = source_attack_id == 0 ? attack_id : source_attack_id;
    state->attacker = attacker_ref;
    state->post_effect_activate = 0;
    state->fail_attack = 0;
    state->last_attack_damage = 0;
    state->turn_attack_count = 0;
    state->second_attack = 0;
    state->attack_flow_flags = 0;

    const OfficialCardStatePod* attacker = official_pod_card(state, attacker_ref);
    const std::uint8_t turn_flags = official_attack_this_turn_flags(*attacker);
    std::int32_t required_heads = 0;
    if ((turn_flags & (1U << 5U)) != 0) required_heads = 2;
    else if ((turn_flags & (1U << 4U)) != 0) required_heads = 1;
    if (required_heads > 0) {
        state->coin_head_count = 0;
        for (std::int32_t index = 0; index < required_heads; ++index) {
            official_pod_coin(state);
        }
        if (state->coin_head_count < required_heads) {
            return official_attack_after_body(state, rules);
        }
    }
    if (official_pod_bad_status(state->players[attacker->player])
            == OfficialBadStatus::kConfused
        && state->players[attacker->player].active.count > 0
        && state->players[attacker->player].active.values[0] == attacker_ref) {
        state->coin_head_count = 0;
        if (!official_pod_coin(state)) {
            official_pod_add_damage(state, rules, attacker_ref, 30);
            return official_attack_after_body(state, rules);
        }
    }
    if (source_attack_id == attack_id
        && (attack->flags & kAttackCopyBenchN) != 0) {
        ++state->turn_attack_count;
        if (state->turn_attack_count > 10000) {
            return official_attack_after_body(state, rules);
        }
        const std::int32_t saved_attack_id = state->current_attack_id;
        const std::int32_t saved_source_attack_id = state->source_attack_id;
        const OfficialAttackResult prepared = official_attack_prepare_special(
            state, rules, true);
        if (prepared == OfficialAttackResult::kNeedsAction) {
            // CPU leaves the resumable parent attack fields untouched while
            // SelectedAttackId is exposed; the source attack is carried by
            // the continuation argument instead.
            state->current_attack_id = saved_attack_id;
            state->source_attack_id = saved_source_attack_id;
        }
        return prepared;
    }
    return official_attack_enter_selected_body(state, rules);
}

PTCG_OFFICIAL_ATTACK_HD inline bool official_attack_validate_selection(
    OfficialStatePod* state,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (count < state->select_min || count > state->select_max) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        return false;
    }
    if (count == 0) return true;
    if (count != 1 || option_indices[0] >= state->options.count
        || state->options.values[option_indices[0]].type
            != static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kAttack)) {
        official_pod_fail(
            state,
            OfficialPodError::kInvalidAction,
            count == 0 ? 0 : option_indices[0]);
        return false;
    }
    return true;
}

PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult official_resume_attack(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (!official_pod_ok(state) || state->attack_flow_stage == 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        }
        return OfficialAttackResult::kError;
    }
    const OfficialAttackStage stage = static_cast<OfficialAttackStage>(
        state->attack_flow_stage);
    if (stage == OfficialAttackStage::kActiveReplacement) {
        return official_attack_resume_active_replacement(
            state, rules, option_indices, count);
    }
    // Refresh and turn-end own nested KO/Prize/trigger/effect resumptions.
    // Delegate to those sub-state-machines before interpreting an active
    // trigger resolver as the attack's AfterAttackTriggers stage.
    if (stage == OfficialAttackStage::kRefresh) {
        const OfficialTurnFlowResult refresh = official_resume_refresh(
            state, rules, option_indices, count);
        if (refresh == OfficialTurnFlowResult::kError) {
            return OfficialAttackResult::kError;
        }
        if (refresh == OfficialTurnFlowResult::kNeedsAction) {
            return OfficialAttackResult::kNeedsAction;
        }
        if (state->game_result != static_cast<std::uint8_t>(
                OfficialGameResult::kNone)) {
            state->attack_flow_stage = 0;
            return OfficialAttackResult::kComplete;
        }
        return official_attack_prepare_second_or_turn_end(state, rules);
    }
    if (stage == OfficialAttackStage::kTurnEnd) {
        const OfficialTurnFlowResult turn = official_resume_turn_flow(
            state, rules, option_indices, count);
        if (turn == OfficialTurnFlowResult::kError) return OfficialAttackResult::kError;
        if (turn == OfficialTurnFlowResult::kNeedsAction) {
            return OfficialAttackResult::kNeedsAction;
        }
        state->attack_flow_stage = 0;
        return OfficialAttackResult::kComplete;
    }
    if (stage == OfficialAttackStage::kAfterAttackTriggers
        && state->trigger_resolver.active != 0) {
        const OfficialTriggerResolverResult trigger = official_resume_trigger_resolution(
            state, rules, option_indices, count);
        if (trigger == OfficialTriggerResolverResult::kError) {
            return OfficialAttackResult::kError;
        }
        if (trigger == OfficialTriggerResolverResult::kNeedsAction) {
            return OfficialAttackResult::kNeedsAction;
        }
        return official_attack_begin_refresh_phase(state, rules);
    }
    if (state->effect_interpreter.awaiting_selection != 0) {
        const OfficialEffectInterpreterResult effect = official_apply_effect_action(
            state, rules, option_indices, count);
        if (effect == OfficialEffectInterpreterResult::kError) {
            return OfficialAttackResult::kError;
        }
        if (effect == OfficialEffectInterpreterResult::kNeedsAction) {
            return OfficialAttackResult::kNeedsAction;
        }
        if (stage == OfficialAttackStage::kPreEffects) {
            return official_attack_after_pre_effects(state, rules);
        }
        if (stage == OfficialAttackStage::kPostEffects
            || stage == OfficialAttackStage::kSupporterEffects) {
            return official_attack_after_body(state, rules);
        }
        official_pod_fail(state, OfficialPodError::kInvalidAction, state->attack_flow_stage);
        return OfficialAttackResult::kError;
    }
    if (state->trigger_resolver.active != 0) {
        official_pod_fail(
            state, OfficialPodError::kInvalidAction, state->attack_flow_stage);
        return OfficialAttackResult::kError;
    }
    if (stage != OfficialAttackStage::kCopySelection
        && stage != OfficialAttackStage::kSecondAttackSelection) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, state->attack_flow_stage);
        return OfficialAttackResult::kError;
    }
    if (!official_attack_validate_selection(state, option_indices, count)) {
        return OfficialAttackResult::kError;
    }
    if (count == 0) {
        official_clear_effect_selection(state);
        return stage == OfficialAttackStage::kSecondAttackSelection
            ? official_attack_begin_turn_end(state, rules)
            : official_attack_after_body(state, rules);
    }
    const OfficialSelectOptionPod selected = state->options.values[option_indices[0]];
    const std::int32_t selected_attack_id = selected.params[0];
    const std::int32_t immediate_source_id = selected.params[1];
    const OfficialAttackRule* selected_attack = official_attack_rule(
        rules, static_cast<std::uint32_t>(selected_attack_id));
    const OfficialCardStatePod* attacker = official_pod_card(state, state->attacker);
    official_clear_effect_selection(state);
    if (selected_attack == nullptr || attacker == nullptr) {
        official_pod_fail(
            state, OfficialPodError::kRulePackBounds, selected_attack_id);
        return OfficialAttackResult::kError;
    }
    state->current_attack_id = selected_attack_id;
    if (!official_attack_state_condition(
            state, rules, *attacker, *selected_attack, immediate_source_id)) {
        return official_attack_after_body(state, rules);
    }
    if (stage == OfficialAttackStage::kSecondAttackSelection) {
        state->second_attack = 1;
    }
    return official_attack_enter_selected_body(state, rules);
}

// CPU SelectedAttackId resumes a copy attack's nested choice without using
// the parent CopySelection function frame.  This is observable for an N
// Zoroark ex copied attack whose selected body has the same id as the source.
PTCG_OFFICIAL_ATTACK_HD inline OfficialAttackResult
official_resume_regular_attack_selection(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (state == nullptr || !official_pod_ok(state)
        || state->attack_flow_stage != static_cast<std::uint8_t>(OfficialAttackStage::kIdle)
        || state->select_type != static_cast<std::uint8_t>(OfficialSelectTypeId::kAttack)
        || !official_attack_validate_selection(state, option_indices, count)) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        }
        return OfficialAttackResult::kError;
    }
    if (count == 0) {
        official_clear_effect_selection(state);
        return official_attack_after_body(state, rules);
    }
    const OfficialSelectOptionPod selected = state->options.values[option_indices[0]];
    const OfficialAttackRule* attack = official_attack_rule(
        rules, static_cast<std::uint32_t>(selected.params[0]));
    const OfficialCardStatePod* attacker = official_pod_card(state, state->attacker);
    if (attack == nullptr || attacker == nullptr || selected.params[1] <= 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, selected.params[0]);
        return OfficialAttackResult::kError;
    }
    const std::int32_t source_attack_id = selected.params[1];
    official_clear_effect_selection(state);
    state->current_attack_id = selected.params[0];
    state->source_attack_id = source_attack_id;
    if (!official_attack_state_condition(
            state, rules, *attacker, *attack, source_attack_id)) {
        return official_attack_after_body(state, rules);
    }
    return official_attack_enter_selected_body(state, rules);
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_ATTACK_HD
