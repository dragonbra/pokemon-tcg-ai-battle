#pragma once

#include <cstdint>

#include "ptcg_cuda/official_setup_interactive_pod.cuh"
#include "ptcg_cuda/official_flow_status.h"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_FLOW_HD __host__ __device__
#else
#define PTCG_OFFICIAL_FLOW_HD
#endif

namespace ptcg::cuda_engine {

PTCG_OFFICIAL_FLOW_HD inline void official_normalize_boundary_scratch(
    OfficialStatePod* state) {
    // Refresh pass count is diagnostic scratch, not persistent official State.
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
}

PTCG_OFFICIAL_FLOW_HD inline bool official_attack_effects_are_damage_change_only(
    const OfficialRulePackView& rules,
    const OfficialAttackRule& attack) {
    bool saw_effect = false;
    const std::uint32_t ranges[2][2] = {
        {
            static_cast<std::uint32_t>(attack.values[kAttackPreEffectOffset]),
            static_cast<std::uint32_t>(attack.values[kAttackPreEffectCount]),
        },
        {
            static_cast<std::uint32_t>(attack.values[kAttackPostEffectOffset]),
            static_cast<std::uint32_t>(attack.values[kAttackPostEffectCount]),
        },
    };
    for (std::uint32_t range = 0; range < 2; ++range) {
        const std::uint32_t offset = ranges[range][0];
        const std::uint32_t count = ranges[range][1];
        for (std::uint32_t index = 0; index < count; ++index) {
            const OfficialEffectRule& effect = rules.effects[offset + index];
            if ((effect.flags & kOfficialEffectIsCondition) != 0) continue;
            saw_effect = true;
            const auto type = static_cast<OfficialEffectTypeId>(
                effect.values[kEffectType]);
            if (type < OfficialEffectTypeId::kAttackDamageChange
                || type > OfficialEffectTypeId::kAttackDamageChangePreTurnTakePrizeCount) {
                return false;
            }
        }
    }
    return saw_effect;
}

PTCG_OFFICIAL_FLOW_HD inline OfficialFlowStatus official_yield_decision_impl(
    OfficialStatePod* state,
    bool count_turn_action) {
    official_normalize_boundary_scratch(state);
    if (state->attack_flow_stage == static_cast<std::uint8_t>(
            OfficialAttackStage::kTurnEnd)) {
        // State::AttackEnd clears the active attack before turn-end effects
        // expose any selections.  Nested POD flow can otherwise leave the
        // last selected (including a second/copied) attack observable.
        state->current_attack_id = 0;
        state->attacker = {};
    }
    if (state->select_type
            == static_cast<std::uint8_t>(OfficialSelectTypeId::kMain)) {
        const bool selected_main_ready = state->continuations.count > 0
            && state->continuations.values[state->continuations.count - 1].opcode
                == static_cast<std::uint16_t>(
                    OfficialContinuationId::kSelectedMain);
        if (!selected_main_ready
            && !official_push_continuation(
                state, OfficialContinuationId::kSelectedMain)) {
            return OfficialFlowStatus::kError;
        }
    } else if (state->trigger_resolver.active != 0
        && state->trigger_resolver.awaiting_activation != 0) {
        if (!official_build_trigger_activation_continuation_mirror(state)) {
            return OfficialFlowStatus::kError;
        }
    } else if (official_trigger_order_awaiting_action(*state)) {
        if (!official_build_trigger_order_continuation_mirror(state)) {
            return OfficialFlowStatus::kError;
        }
    } else if (official_prize_selection_awaiting_action(*state)) {
        if (!official_build_prize_selection_continuation_mirror(state)) {
            return OfficialFlowStatus::kError;
        }
    } else if (official_active_replacement_awaiting_action(*state)) {
        if (!official_build_active_replacement_continuation_mirror(state)) {
            return OfficialFlowStatus::kError;
        }
    } else if (official_refresh_overflow_awaiting_action(*state)) {
        if (!official_build_refresh_overflow_continuation_mirror(state)) {
            return OfficialFlowStatus::kError;
        }
    } else if (official_attack_selection_awaiting_action(*state)) {
        if (!official_build_attack_selection_continuation_mirror(state)) {
            return OfficialFlowStatus::kError;
        }
    } else if (state->effect_interpreter.awaiting_selection != 0) {
        if (!official_build_effect_selection_continuation_mirror(state)) {
            return OfficialFlowStatus::kError;
        }
    } else if ((state->turn_state & (1U << 3U)) != 0
        && ((state->select_type
                    == static_cast<std::uint8_t>(OfficialSelectTypeId::kEnergy)
                && state->select_context == 31)
            || (state->select_type
                    == static_cast<std::uint8_t>(OfficialSelectTypeId::kCard)
                && state->select_context == 4))) {
        if (!official_build_retreat_selection_continuation_mirror(state)) {
            return OfficialFlowStatus::kError;
        }
    }
    if (count_turn_action) ++state->turn_action_count;
    return OfficialFlowStatus::kNeedsAction;
}

PTCG_OFFICIAL_FLOW_HD inline OfficialFlowStatus official_yield_decision(
    OfficialStatePod* state) {
    return official_yield_decision_impl(state, true);
}

PTCG_OFFICIAL_FLOW_HD inline OfficialFlowStatus
official_yield_decision_without_turn_action_count(OfficialStatePod* state) {
    return official_yield_decision_impl(state, false);
}

PTCG_OFFICIAL_FLOW_HD inline OfficialFlowStatus official_flow_status(
    const OfficialStatePod& state) {
    if (state.error != 0) return OfficialFlowStatus::kError;
    if (state.game_result != static_cast<std::uint8_t>(OfficialGameResult::kNone)) {
        return OfficialFlowStatus::kTerminal;
    }
    if (state.select_type != static_cast<std::uint8_t>(OfficialSelectTypeId::kNone)
        && (state.options.count > 0
            || state.phase == static_cast<std::uint8_t>(
                OfficialGamePhase::kSetup))) {
        return OfficialFlowStatus::kNeedsAction;
    }
    return OfficialFlowStatus::kIdle;
}

PTCG_OFFICIAL_FLOW_HD inline OfficialFlowStatus official_boundary_status(
    OfficialStatePod* state) {
    const OfficialFlowStatus status = official_flow_status(*state);
    if (status == OfficialFlowStatus::kIdle
        || status == OfficialFlowStatus::kTerminal) {
        official_normalize_boundary_scratch(state);
    }
    return status;
}

PTCG_OFFICIAL_FLOW_HD inline OfficialFlowStatus official_apply_pending_action(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (state == nullptr || !official_pod_ok(state)) {
        return OfficialFlowStatus::kError;
    }
    if (state->game_result
        != static_cast<std::uint8_t>(OfficialGameResult::kNone)) {
        return official_boundary_status(state);
    }
    if (state->phase == static_cast<std::uint8_t>(OfficialGamePhase::kSetup)) {
        return official_setup_apply_action(
            state, rules, option_indices, count);
    }
    if (official_prize_selection_awaiting_action(*state)
        && !official_consume_prize_selection_continuation_mirror(state)) {
        return OfficialFlowStatus::kError;
    }
    if (official_active_replacement_awaiting_action(*state)
        && !official_consume_active_replacement_continuation_mirror(state)) {
        return OfficialFlowStatus::kError;
    }
    if (official_refresh_overflow_awaiting_action(*state)
        && !official_consume_refresh_overflow_continuation_mirror(state)) {
        return OfficialFlowStatus::kError;
    }
    if (official_trigger_order_awaiting_action(*state)
        && !official_consume_trigger_order_continuation_mirror(state)) {
        return OfficialFlowStatus::kError;
    }
    // Trigger activation can be nested under refresh/KO, attack, or turn flow.
    // Consume its selection callback before dispatching to the enclosing flow,
    // matching State::callFunction before the selected ability is resumed.
    if (state->trigger_resolver.active != 0
        && state->trigger_resolver.awaiting_activation != 0) {
        if (!official_consume_trigger_activation_continuation_mirror(state)) {
            return OfficialFlowStatus::kError;
        }
    }
    if (state->effect_interpreter.awaiting_selection != 0
        && !official_consume_effect_selection_continuation_mirror(state)) {
        return OfficialFlowStatus::kError;
    }
    if (official_attack_selection_awaiting_action(*state)
        && !official_consume_attack_selection_continuation_mirror(state)) {
        return OfficialFlowStatus::kError;
    }

    if (state->attack_flow_stage == static_cast<std::uint8_t>(OfficialAttackStage::kIdle)
        && state->select_type == static_cast<std::uint8_t>(OfficialSelectTypeId::kAttack)
        && state->options.count > 0
        && state->options.values[0].type
            == static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kAttack)) {
        const bool return_to_main =
            (state->attack_flow_flags & kOfficialAttackReturnToMainFlag) != 0;
        const OfficialAttackResult result =
            official_resume_regular_attack_selection(
                state, rules, option_indices, count);
        if (result == OfficialAttackResult::kError) {
            return OfficialFlowStatus::kError;
        }
        if (result == OfficialAttackResult::kNeedsAction) {
            return official_yield_decision(state);
        }
        if (return_to_main) {
            state->attack_flow_flags &= static_cast<std::uint8_t>(
                ~kOfficialAttackReturnToMainFlag);
            const OfficialMainResult main = official_main_after_flow(state, rules);
            if (main == OfficialMainResult::kError) {
                return OfficialFlowStatus::kError;
            }
            if (main == OfficialMainResult::kNeedsAction) {
                return official_yield_decision(state);
            }
        }
        return official_boundary_status(state);
    }

    if (state->attack_flow_stage != 0) {
        const bool return_to_main =
            (state->attack_flow_flags & kOfficialAttackReturnToMainFlag) != 0;
        const OfficialAttackResult result = official_resume_attack(
            state, rules, option_indices, count);
        if (result == OfficialAttackResult::kError) {
            return OfficialFlowStatus::kError;
        }
        if (result == OfficialAttackResult::kNeedsAction) {
            return official_yield_decision(state);
        }
        if (return_to_main) {
            state->attack_flow_flags &= static_cast<std::uint8_t>(
                ~kOfficialAttackReturnToMainFlag);
            const OfficialMainResult main = official_main_after_flow(state, rules);
            if (main == OfficialMainResult::kError) return OfficialFlowStatus::kError;
            if (main == OfficialMainResult::kNeedsAction) {
                return official_yield_decision(state);
            }
        }
        return official_boundary_status(state);
    }
    if (state->turn_flow_stage != 0) {
        const bool return_to_main =
            (state->flow_flags & kOfficialTurnReturnToMainFlag) != 0;
        const OfficialTurnFlowResult result = official_resume_turn_flow(
            state, rules, option_indices, count);
        if (result == OfficialTurnFlowResult::kError) {
            return OfficialFlowStatus::kError;
        }
        if (result == OfficialTurnFlowResult::kNeedsAction) {
            return official_yield_decision(state);
        }
        if (return_to_main) {
            state->flow_flags &= static_cast<std::uint8_t>(
                ~kOfficialTurnReturnToMainFlag);
            const OfficialMainResult main = official_main_after_flow(state, rules);
            if (main == OfficialMainResult::kError) return OfficialFlowStatus::kError;
            if (main == OfficialMainResult::kNeedsAction) {
                return official_yield_decision(state);
            }
        }
        return official_boundary_status(state);
    }
    if (state->refresh_flow_stage != 0) {
        const bool selected_trigger_activation =
            count == 1 && option_indices != nullptr
            && option_indices[0] < state->options.count
            && state->options.values[option_indices[0]].type
                == static_cast<std::uint8_t>(
                    OfficialSelectOptionTypeId::kYes);
        const bool refresh_knockout_trigger_activation_accept =
            state->trigger_resolver.active != 0
            && state->trigger_resolver.awaiting_activation != 0
            && state->trigger_resolver.current.activate.skill_id == 26
            && state->refresh_flow_stage == static_cast<std::uint8_t>(
                OfficialRefreshFlowStage::kKnockout)
            && (state->flow_flags & kOfficialRefreshReturnToMainFlag) != 0
            && selected_trigger_activation;
        const bool retreat_pending =
            (state->flow_flags & kOfficialRetreatRefreshPendingFlag) != 0;
        const bool return_to_main =
            (state->flow_flags & kOfficialRefreshReturnToMainFlag) != 0;
        const OfficialTurnFlowResult result = official_resume_refresh(
            state, rules, option_indices, count);
        if (result == OfficialTurnFlowResult::kError) {
            return OfficialFlowStatus::kError;
        }
        if (result == OfficialTurnFlowResult::kNeedsAction) {
            return official_yield_decision(state);
        }
        if (retreat_pending) {
            const OfficialMainResult main =
                official_main_continue_retreat_after_refresh(state, rules);
            if (main == OfficialMainResult::kError) return OfficialFlowStatus::kError;
            if (main == OfficialMainResult::kNeedsAction) {
                return official_yield_decision(state);
            }
            return official_boundary_status(state);
        }
        if (return_to_main) {
            state->flow_flags &= static_cast<std::uint8_t>(
                ~kOfficialRefreshReturnToMainFlag);
            const OfficialMainResult main = official_main_after_refresh(state, rules);
            if (main == OfficialMainResult::kError) return OfficialFlowStatus::kError;
            if (main == OfficialMainResult::kNeedsAction) {
                if (refresh_knockout_trigger_activation_accept) {
                    return official_yield_decision_without_turn_action_count(state);
                }
                return official_yield_decision(state);
            }
        }
        return official_boundary_status(state);
    }
    if (state->trigger_resolver.active != 0) {
        const bool retreat_pending =
            (state->flow_flags & kOfficialRetreatTriggerPendingFlag) != 0;
        const bool refresh_return_to_main =
            (state->flow_flags & kOfficialRefreshTriggerReturnToMainFlag) != 0;
        const bool ability_return_to_main =
            (state->flow_flags & kOfficialAbilityReturnToMainFlag) != 0;
        const OfficialTriggerResolverResult result =
            official_resume_trigger_resolution(
                state, rules, option_indices, count);
        if (result == OfficialTriggerResolverResult::kError) {
            return OfficialFlowStatus::kError;
        }
        if (result == OfficialTriggerResolverResult::kNeedsAction) {
            return official_yield_decision(state);
        }
        if (retreat_pending) {
            const OfficialMainResult main =
                official_main_continue_retreat_after_triggers(state, rules);
            if (main == OfficialMainResult::kError) return OfficialFlowStatus::kError;
            if (main == OfficialMainResult::kNeedsAction) {
                return official_yield_decision(state);
            }
            return official_boundary_status(state);
        }
        if (refresh_return_to_main) {
            state->flow_flags &= static_cast<std::uint8_t>(
                ~kOfficialRefreshTriggerReturnToMainFlag);
            const OfficialMainResult main = official_main_begin_refresh(state, rules);
            if (main == OfficialMainResult::kError) return OfficialFlowStatus::kError;
            if (main == OfficialMainResult::kNeedsAction) {
                return official_yield_decision(state);
            }
            return official_boundary_status(state);
        }
        if (ability_return_to_main) {
            state->flow_flags &= static_cast<std::uint8_t>(
                ~kOfficialAbilityReturnToMainFlag);
            const OfficialMainResult main = official_main_begin_refresh(state, rules);
            if (main == OfficialMainResult::kError) return OfficialFlowStatus::kError;
            if (main == OfficialMainResult::kNeedsAction) {
                return official_yield_decision(state);
            }
        }
        return official_boundary_status(state);
    }
    if (state->effect_interpreter.awaiting_selection != 0) {
        const bool return_to_main =
            (state->flow_flags & kOfficialPlayEffectReturnToMainFlag) != 0;
        const OfficialEffectInterpreterResult result = official_apply_effect_action(
            state, rules, option_indices, count);
        if (result == OfficialEffectInterpreterResult::kError) {
            return OfficialFlowStatus::kError;
        }
        if (result == OfficialEffectInterpreterResult::kNeedsAction) {
            return official_yield_decision(state);
        }
        if (return_to_main) {
            const OfficialMainResult main = official_main_after_play_effect(
                state, rules);
            if (main == OfficialMainResult::kError) return OfficialFlowStatus::kError;
            if (main == OfficialMainResult::kNeedsAction) {
                return official_yield_decision(state);
            }
            if (official_flow_status(*state) == OfficialFlowStatus::kNeedsAction) {
                return official_yield_decision(state);
            }
            return official_boundary_status(state);
        }
        return official_boundary_status(state);
    }
    if ((state->turn_state & (1U << 3U)) != 0
        && ((state->select_type
                    == static_cast<std::uint8_t>(OfficialSelectTypeId::kEnergy)
                && state->select_context == 31)
            || (state->select_type
                    == static_cast<std::uint8_t>(OfficialSelectTypeId::kCard)
                && state->select_context == 4))) {
        if (!official_consume_retreat_selection_continuation_mirror(state)) {
            return OfficialFlowStatus::kError;
        }
        const OfficialMainResult result = official_main_resume_retreat_action(
            state, rules, option_indices, count);
        if (result == OfficialMainResult::kError) return OfficialFlowStatus::kError;
        if (result == OfficialMainResult::kNeedsAction) {
            return official_yield_decision(state);
        }
        return official_boundary_status(state);
    }
    if (state->reserved_flow != 0) {
        const OfficialKnockoutResult result = official_resume_knockout(
            state, rules, option_indices, count);
        if (result == OfficialKnockoutResult::kError) {
            return OfficialFlowStatus::kError;
        }
        if (result == OfficialKnockoutResult::kNeedsAction) {
            return official_yield_decision(state);
        }
        return official_boundary_status(state);
    }
    if (state->select_type
            == static_cast<std::uint8_t>(OfficialSelectTypeId::kMain)) {
        const OfficialSelectOptionPod* main_option =
            count == 1
            && option_indices != nullptr
            && option_indices[0] < state->options.count
                ? &state->options.values[option_indices[0]]
                : nullptr;
        const bool main_attack_action = main_option != nullptr
            && main_option->type
                == static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kAttack);
        const std::int32_t previous_source_attack_id = state->source_attack_id;
        const std::uint16_t previous_turn_used_skill_count =
            state->turn_used_skills.count;
        bool main_attack_damage_change_only_action = false;
        bool main_attack_stale_source_no_effect_action = false;
        if (main_attack_action) {
            const OfficialAttackRule* selected_attack = official_attack_rule(
                rules, static_cast<std::uint32_t>(main_option->params[0]));
            if (selected_attack != nullptr) {
                const bool has_attack_effects =
                    selected_attack->values[kAttackPreEffectCount] > 0
                    || selected_attack->values[kAttackPostEffectCount] > 0;
                main_attack_damage_change_only_action =
                    official_attack_effects_are_damage_change_only(
                        rules, *selected_attack);
                if (!has_attack_effects
                    && main_option->params[1] > 0
                    && previous_source_attack_id > 0
                    && previous_source_attack_id != main_option->params[0]) {
                    const OfficialAttackRule* previous_source_attack =
                        official_attack_rule(
                            rules,
                            static_cast<std::uint32_t>(previous_source_attack_id));
                    if (previous_source_attack != nullptr) {
                        const bool previous_has_attack_effects =
                            previous_source_attack->values[kAttackPreEffectCount] > 0
                            || previous_source_attack->values[
                                kAttackPostEffectCount] > 0;
                        main_attack_stale_source_no_effect_action =
                            previous_has_attack_effects
                            && !official_attack_effects_are_damage_change_only(
                                rules, *previous_source_attack);
                    }
                }
            }
        }
        if (!official_consume_selected_main_continuation(state)) {
            return OfficialFlowStatus::kError;
        }
        const OfficialMainResult result = official_apply_main_action(
            state, rules, option_indices, count);
        if (result == OfficialMainResult::kError) return OfficialFlowStatus::kError;
        if (result == OfficialMainResult::kNeedsAction) {
            const OfficialFlowStatus yielded = official_yield_decision(state);
            const bool after_attack4_changed_boundary =
                (main_attack_damage_change_only_action
                    || main_attack_stale_source_no_effect_action)
                && state->turn_attack_count > 0
                && state->turn_used_skills.count
                    == previous_turn_used_skill_count
                && (state->control_flags & (1U << 4U)) != 0
                && state->continuations.count > 0
                && state->continuations.values[0].opcode
                    == static_cast<std::uint16_t>(
                        OfficialContinuationId::kAfterAttack4);
            if (yielded == OfficialFlowStatus::kNeedsAction
                && after_attack4_changed_boundary) {
                state->control_flags &= static_cast<std::uint8_t>(
                    ~kOfficialChangedFlag);
            }
            return yielded;
        }
        const OfficialFlowStatus boundary = official_boundary_status(state);
        if (main_attack_damage_change_only_action
            && state->turn_attack_count > 0
            && (state->finish_reason == static_cast<std::uint8_t>(
                    OfficialFinishReason::kPrize)
                || state->finish_reason == static_cast<std::uint8_t>(
                    OfficialFinishReason::kNoActivePokemon))
            && boundary == OfficialFlowStatus::kTerminal) {
            state->control_flags &= static_cast<std::uint8_t>(
                ~kOfficialChangedFlag);
        }
        return boundary;
    }

    official_pod_fail(state, OfficialPodError::kInvalidAction, count);
    return OfficialFlowStatus::kError;
}

PTCG_OFFICIAL_FLOW_HD inline OfficialFlowStatus
official_advance_idle_state_to_decision(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (state == nullptr) return OfficialFlowStatus::kError;
    const OfficialFlowStatus status = official_flow_status(*state);
    if (status != OfficialFlowStatus::kIdle) return status;
    if (state->phase == static_cast<std::uint8_t>(OfficialGamePhase::kSetup)) {
        return official_setup_advance(state, rules);
    }
    if (state->continuations.count > 0) {
        return official_dispatch_no_action_continuation(state, rules);
    }
    if (state->phase != static_cast<std::uint8_t>(OfficialGamePhase::kMain)) {
        return status;
    }
    const OfficialMainResult result = official_prepare_main_options(state, rules);
    if (result == OfficialMainResult::kError) return OfficialFlowStatus::kError;
    if (result == OfficialMainResult::kNeedsAction) {
        return official_yield_decision(state);
    }
    return official_boundary_status(state);
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_FLOW_HD
