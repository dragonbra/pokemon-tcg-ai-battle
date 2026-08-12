#pragma once

#include <cstdint>

#include "ptcg_cuda/official_continuation_ids.cuh"
#include "ptcg_cuda/official_flow_status.h"
#include "ptcg_cuda/official_main_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_CONT_HD __host__ __device__
#else
#define PTCG_OFFICIAL_CONT_HD
#endif

namespace ptcg::cuda_engine {

struct OfficialContinuationCall {
    OfficialContinuationPod frame{};
    std::uint16_t stack_index = 0;
    bool valid = false;
};

PTCG_OFFICIAL_CONT_HD inline bool official_push_continuation(
    OfficialStatePod* state,
    OfficialContinuationId opcode,
    std::uint8_t arg_type = 0,
    std::int32_t arg0 = 0,
    std::int32_t arg1 = 0,
    std::int32_t arg2 = 0,
    std::uint8_t call_count = 1) {
    if (state == nullptr || !official_pod_ok(state)) return false;
    if (state->continuations.count >= kOfficialContinuationCapacity) {
        official_pod_fail(
            state,
            OfficialPodError::kContinuationStackOverflow,
            state->continuations.count);
        return false;
    }
    if (call_count == 0 || arg_type > 4
        || static_cast<std::uint16_t>(opcode) >= kOfficialContinuationCount) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            static_cast<std::int32_t>(opcode));
        return false;
    }
    OfficialContinuationPod& frame =
        state->continuations.values[state->continuations.count++];
    frame = {};
    frame.args[0] = arg0;
    frame.args[1] = arg1;
    frame.args[2] = arg2;
    frame.opcode = static_cast<std::uint16_t>(opcode);
    frame.arg_type = arg_type;
    frame.call_count = call_count;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline OfficialContinuationCall
official_begin_continuation_call(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || state->continuations.count == 0) {
        if (state != nullptr) {
            official_pod_fail(
                state, OfficialPodError::kUnsupportedContinuation, -1);
        }
        return {};
    }
    const std::uint16_t index = state->continuations.count - 1;
    return {state->continuations.values[index], index, true};
}

PTCG_OFFICIAL_CONT_HD inline bool official_finish_continuation_call(
    OfficialStatePod* state,
    const OfficialContinuationCall& call,
    bool break_call = false) {
    if (state == nullptr || !call.valid
        || call.stack_index >= state->continuations.count) {
        if (state != nullptr) {
            official_pod_fail(
                state, OfficialPodError::kUnsupportedContinuation, -2);
        }
        return false;
    }
    OfficialContinuationPod& current =
        state->continuations.values[call.stack_index];
    if (current.opcode != call.frame.opcode
        || current.called_count != call.frame.called_count) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            current.opcode);
        return false;
    }
    ++current.called_count;
    if (!break_call && current.called_count < current.call_count) return true;
    for (std::uint16_t index = call.stack_index + 1;
         index < state->continuations.count;
         ++index) {
        state->continuations.values[index - 1] =
            state->continuations.values[index];
    }
    --state->continuations.count;
    return true;
}

// A selection callback is a one-shot continuation.  Consume it before the
// selected action mutates the state so a resumed action cannot be replayed if
// it yields another decision or enters a nested flow.
PTCG_OFFICIAL_CONT_HD inline bool official_consume_selected_main_continuation(
    OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)) return false;
    if (state->continuations.count == 0) return true;
    OfficialContinuationPod& frame = state->continuations.values[
        state->continuations.count - 1];
    if (frame.opcode != static_cast<std::uint16_t>(
            OfficialContinuationId::kSelectedMain)) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            frame.opcode);
        return false;
    }
    --state->continuations.count;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_build_retreat_selection_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || (state->turn_state & (1U << 3U)) == 0) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state, OfficialPodError::kUnsupportedContinuation, -10);
        }
        return false;
    }
    const std::int32_t player = official_active_player(*state);
    state->continuations.count = 0;
    if (!official_push_continuation(state, OfficialContinuationId::kToMain)
        || !official_push_continuation(state, OfficialContinuationId::kAfterRetreat)) {
        return false;
    }
    if (state->select_type == static_cast<std::uint8_t>(
            OfficialSelectTypeId::kEnergy)
        && state->select_context == 31) {
        return official_push_continuation(
                state,
                OfficialContinuationId::kSelectSwitchPokemon,
                3,
                player,
                player)
            && official_push_continuation(
                state, OfficialContinuationId::kSelectedPokemonEnergy)
            && official_push_continuation(
                state,
                OfficialContinuationId::kSelectedPokemonEnergyLoop,
                1,
                player);
    }
    if (state->select_type == static_cast<std::uint8_t>(
            OfficialSelectTypeId::kCard)
        && state->select_context == 4) {
        return official_push_continuation(
            state, OfficialContinuationId::kSelectedSwitchPokemon);
    }
    official_pod_fail(
        state, OfficialPodError::kUnsupportedContinuation, state->select_context);
    return false;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_consume_retreat_selection_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)) return false;
    const bool energy = state->select_type == static_cast<std::uint8_t>(
            OfficialSelectTypeId::kEnergy)
        && state->select_context == 31;
    const bool switching = state->select_type == static_cast<std::uint8_t>(
            OfficialSelectTypeId::kCard)
        && state->select_context == 4;
    const std::uint16_t expected_count = energy ? 5 : (switching ? 3 : 0);
    const OfficialContinuationId expected_top = energy
        ? OfficialContinuationId::kSelectedPokemonEnergyLoop
        : OfficialContinuationId::kSelectedSwitchPokemon;
    if (expected_count == 0
        || state->continuations.count != expected_count
        || state->continuations.values[0].opcode != static_cast<std::uint16_t>(
            OfficialContinuationId::kToMain)
        || state->continuations.values[1].opcode != static_cast<std::uint16_t>(
            OfficialContinuationId::kAfterRetreat)
        || state->continuations.values[expected_count - 1].opcode
            != static_cast<std::uint16_t>(expected_top)) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            state->continuations.count);
        return false;
    }
    state->continuations.count = 0;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_attack_selection_awaiting_action(const OfficialStatePod& state) {
    const auto stage = static_cast<OfficialAttackStage>(state.attack_flow_stage);
    if (state.select_type != static_cast<std::uint8_t>(
            OfficialSelectTypeId::kAttack)) {
        return false;
    }
    if (stage == OfficialAttackStage::kCopySelection
        || stage == OfficialAttackStage::kSecondAttackSelection) {
        return true;
    }
    // N Zoroark's self-copy candidate is exposed by the CPU as a regular
    // SelectedAttackId callback while the parent attack fields remain at
    // their previous boundary values.
    return stage == OfficialAttackStage::kIdle
        && state.options.count > 0
        && state.options.values[0].type
            == static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kAttack)
        && state.options.values[0].params[1] > 0;
}

PTCG_OFFICIAL_CONT_HD inline OfficialContinuationId
official_attack_selection_callback(const OfficialStatePod& state) {
    return static_cast<OfficialAttackStage>(state.attack_flow_stage)
            == OfficialAttackStage::kSecondAttackSelection
        ? OfficialContinuationId::kSelectedSecondAttack
        : OfficialContinuationId::kSelectedAttackId;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_build_attack_selection_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || !official_attack_selection_awaiting_action(*state)) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->attack_flow_stage);
        }
        return false;
    }
    const auto stage = static_cast<OfficialAttackStage>(state->attack_flow_stage);
    state->continuations.count = 0;
    if (stage == OfficialAttackStage::kCopySelection
        || stage == OfficialAttackStage::kIdle) {
        const std::int32_t source_attack_id = state->options.count > 0
            ? state->options.values[0].params[1] : 0;
        return official_push_continuation(
            state,
            OfficialContinuationId::kSelectedAttackId,
            1,
            source_attack_id);
    }
    return official_push_continuation(
        state, official_attack_selection_callback(*state));
}

PTCG_OFFICIAL_CONT_HD inline bool
official_consume_attack_selection_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || !official_attack_selection_awaiting_action(*state)
        || state->continuations.count == 0
        || state->continuations.values[state->continuations.count - 1].opcode
            != static_cast<std::uint16_t>(
                official_attack_selection_callback(*state))) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.count);
        }
        return false;
    }
    const auto stage = static_cast<OfficialAttackStage>(state->attack_flow_stage);
    if (stage == OfficialAttackStage::kCopySelection
        || stage == OfficialAttackStage::kIdle) {
        if (state->continuations.count != 1
            || state->continuations.values[0].arg_type != 1
            || state->continuations.values[0].args[0] <= 0
            || state->continuations.values[0].args[1] != 0) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.count);
            return false;
        }
    } else if (state->continuations.count != 1) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            state->continuations.count);
        return false;
    }
    state->continuations.count = 0;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline OfficialContinuationId
official_effect_selection_callback(const OfficialStatePod& state) {
    const auto resume = static_cast<OfficialEffectResumeKind>(
        state.effect_interpreter.resume_kind);
    if (resume == OfficialEffectResumeKind::kEvolve) {
        return OfficialContinuationId::kSelectedEvolveTarget;
    }
    if (resume == OfficialEffectResumeKind::kApplyPrimitive) {
        if (state.select_type == static_cast<std::uint8_t>(
                OfficialSelectTypeId::kAttachedCard)
            && state.options.count > 0
            && state.options.values[0].type == static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kEnergyCard)) {
            return OfficialContinuationId::kSelectedEnergyTarget;
        }
        if (state.select_type == static_cast<std::uint8_t>(
                OfficialSelectTypeId::kAttachedCard)
            && state.options.count > 0
            && state.options.values[0].type == static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kToolCard)) {
            return OfficialContinuationId::kSelectedToolTarget;
        }
        if (state.select_type == static_cast<std::uint8_t>(
                OfficialSelectTypeId::kCardOrAttachedCard)) {
            return OfficialContinuationId::kSelectedCardOrAttachedCard;
        }
        return OfficialContinuationId::kSelectedEffectTarget;
    }
    if (resume == OfficialEffectResumeKind::kRemoveDamageCounter) {
        return OfficialContinuationId::kSelectedRemoveDamageCounter;
    }
    if (resume == OfficialEffectResumeKind::kDamageCounterAny) {
        return OfficialContinuationId::kSelectedDamageCounterAny;
    }
    if (resume == OfficialEffectResumeKind::kSelectActivate) {
        return OfficialContinuationId::kSelectedActivate;
    }
    if (resume == OfficialEffectResumeKind::kSkillChooseEffect) {
        return OfficialContinuationId::kSelectedWhichEffect;
    }
    if (resume == OfficialEffectResumeKind::kEnemySkillChooseEffect) {
        return OfficialContinuationId::kEnemySelectedWhichEffect;
    }
    if (resume == OfficialEffectResumeKind::kSelectEffect) {
        return OfficialContinuationId::kSelectedFirstEffect;
    }
    return OfficialContinuationId::kSelectedPokemonEnergyLoop;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_trigger_inside_refresh_knockout(const OfficialStatePod& state) {
    return state.trigger_resolver.active != 0
        && state.refresh_flow_stage == static_cast<std::uint8_t>(
            OfficialRefreshFlowStage::kKnockout);
}

PTCG_OFFICIAL_CONT_HD inline bool
official_trigger_returns_to_ko_proc2(const OfficialStatePod& state) {
    return official_trigger_inside_refresh_knockout(state)
        && state.trigger_resolver.depth == 1
        && state.reserved_flow == static_cast<std::uint8_t>(
            OfficialKnockoutStage::kPreKoTriggers);
}

PTCG_OFFICIAL_CONT_HD inline bool
official_trigger_returns_to_ko_proc3(const OfficialStatePod& state) {
    return official_trigger_inside_refresh_knockout(state)
        && state.trigger_resolver.depth == 1
        && state.reserved_flow == static_cast<std::uint8_t>(
            OfficialKnockoutStage::kPostKoTriggers);
}

PTCG_OFFICIAL_CONT_HD inline bool
official_trigger_returns_to_knockout_proc(const OfficialStatePod& state) {
    return official_trigger_returns_to_ko_proc2(state)
        || official_trigger_returns_to_ko_proc3(state);
}

PTCG_OFFICIAL_CONT_HD inline OfficialContinuationId
official_trigger_knockout_return_continuation(const OfficialStatePod& state) {
    return official_trigger_returns_to_ko_proc3(state)
        ? OfficialContinuationId::kKoProc3
        : OfficialContinuationId::kKoProc2;
}

PTCG_OFFICIAL_CONT_HD inline bool official_trigger_return_continuations(
    OfficialStatePod* state,
    OfficialContinuationId* parent,
    OfficialContinuationId* after_resolver);

PTCG_OFFICIAL_CONT_HD inline bool
official_build_effect_selection_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || state->effect_interpreter.awaiting_selection == 0) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state, OfficialPodError::kUnsupportedContinuation, -11);
        }
        return false;
    }
    const bool play_effect =
        (state->flow_flags & kOfficialPlayEffectReturnToMainFlag) != 0;
    const bool ability_effect = state->trigger_resolver.active != 0
        && (state->flow_flags & kOfficialAbilityReturnToMainFlag) != 0;
    const bool trigger_effect = state->trigger_resolver.active != 0
        && (state->flow_flags & kOfficialRefreshTriggerReturnToMainFlag) != 0;
    const bool refresh_knockout_trigger_effect =
        official_trigger_inside_refresh_knockout(*state);
    const bool refresh_knockout_return_frame =
        official_trigger_returns_to_knockout_proc(*state);
    OfficialContinuationId refresh_knockout_parent =
        OfficialContinuationId::kMainSelect;
    OfficialContinuationId refresh_knockout_after =
        OfficialContinuationId::kAfterRefresh;
    if (refresh_knockout_trigger_effect
        && !official_trigger_return_continuations(
            state,
            &refresh_knockout_parent,
            &refresh_knockout_after)) {
        return false;
    }
    const bool attack_trigger_effect = state->trigger_resolver.active != 0
        && state->attack_flow_stage == static_cast<std::uint8_t>(
            OfficialAttackStage::kAfterAttackTriggers);
    const bool pre_attack_effect = state->attack_flow_stage
        == static_cast<std::uint8_t>(OfficialAttackStage::kPreEffects);
    const bool post_attack_effect = state->attack_flow_stage
        == static_cast<std::uint8_t>(OfficialAttackStage::kPostEffects);
    if (!play_effect && !ability_effect && !trigger_effect
        && !refresh_knockout_trigger_effect
        && !attack_trigger_effect && !pre_attack_effect
        && !post_attack_effect) {
        official_pod_fail(
            state, OfficialPodError::kUnsupportedContinuation, -11);
        return false;
    }
    if (trigger_effect || refresh_knockout_trigger_effect || attack_trigger_effect) {
        // AfterMoveTriggerStack marks stateChanged before the official engine
        // enters an unconditionally activated trigger ability.
        state->control_flags |= 1U << 4U;
    }
    state->continuations.count = 0;
    if (play_effect) {
        if (!official_push_continuation(state, OfficialContinuationId::kToMain)
            || !official_push_continuation(
                state, OfficialContinuationId::kAfterPlay)) {
            return false;
        }
    } else if (ability_effect) {
        if (!official_push_continuation(state, OfficialContinuationId::kToMain)
            || !official_push_continuation(
                state, OfficialContinuationId::kAfterAbility)) {
            return false;
        }
    } else if (pre_attack_effect) {
        if (!official_push_continuation(
                state,
                OfficialContinuationId::kAttackEffects,
                2,
                0)
            || !official_push_continuation(
                state, OfficialContinuationId::kAttackDamage)) {
            return false;
        }
    } else if (post_attack_effect) {
        if (!official_push_continuation(
                state, OfficialContinuationId::kAfterAttack)) {
            return false;
        }
    } else if (attack_trigger_effect) {
        if (!official_push_continuation(
                state, OfficialContinuationId::kAfterAttack2)
            || !official_push_continuation(
                state, OfficialContinuationId::kAfterAttackTrigger)
            || !official_push_continuation(
                state,
                OfficialContinuationId::kAfterTriggerAbility,
                1,
                state->trigger_resolver.depth)) {
            return false;
        }
    } else if (refresh_knockout_trigger_effect) {
        if (!official_push_continuation(
                state, refresh_knockout_parent)
            || !official_push_continuation(
                state, refresh_knockout_after)
            || (refresh_knockout_return_frame && !official_push_continuation(
                state,
                official_trigger_knockout_return_continuation(*state)))
            || !official_push_continuation(
                state,
                OfficialContinuationId::kAfterTriggerAbility,
                1,
                state->trigger_resolver.depth)) {
            return false;
        }
    } else if (!official_push_continuation(
                   state, OfficialContinuationId::kMainSelect)
        || !official_push_continuation(
            state, OfficialContinuationId::kAfterRefresh)
        || !official_push_continuation(
            state,
            OfficialContinuationId::kAfterTriggerAbility,
            1,
            state->trigger_resolver.depth)) {
        return false;
    }
    const auto resume = static_cast<OfficialEffectResumeKind>(
        state->effect_interpreter.resume_kind);
    if (resume == OfficialEffectResumeKind::kSkillChooseEffect) {
        return official_push_continuation(
            state, OfficialContinuationId::kSelectedWhichEffect);
    }
    if (resume == OfficialEffectResumeKind::kEnemySkillChooseEffect) {
        return official_push_continuation(
            state, OfficialContinuationId::kEnemySelectedWhichEffect);
    }
    const bool retain_effect_frame =
        resume != OfficialEffectResumeKind::kSkillChooseEffect
        && resume != OfficialEffectResumeKind::kEnemySkillChooseEffect
        && state->effect_interpreter.effect_index + 1
            < state->effect_interpreter.effect_count;
    if (retain_effect_frame) {
        if (!official_push_continuation(
                state,
                (pre_attack_effect || post_attack_effect)
                    ? OfficialContinuationId::kAttackEffect
                    : OfficialContinuationId::kActivateSkillEffect,
                0,
                0,
                0,
                0,
                static_cast<std::uint8_t>(state->effect_interpreter.effect_count))) {
            return false;
        }
        state->continuations.values[state->continuations.count - 1].called_count =
            static_cast<std::uint8_t>(state->effect_interpreter.effect_index + 1);
    }
    const bool retain_repeat_frame =
        state->effect_interpreter.repeat_continuation != 0
        && state->effect_interpreter.repeat_index + 1
            < state->effect_interpreter.repeat_count;
    if (retain_repeat_frame) {
        const auto repeat_continuation = static_cast<OfficialContinuationId>(
            state->effect_interpreter.repeat_continuation);
        if (!official_push_continuation(
                state,
                repeat_continuation,
                1,
                state->effect_interpreter.effect_index,
                0,
                0,
                static_cast<std::uint8_t>(
                    state->effect_interpreter.repeat_count))) {
            return false;
        }
        state->continuations.values[state->continuations.count - 1].called_count =
            static_cast<std::uint8_t>(
                state->effect_interpreter.repeat_index + 1);
    }
    if (state->effect_interpreter.separator_pending != 0
        && !official_push_continuation(
            state, OfficialContinuationId::kSeparatorProc)) {
        return false;
    }
    if (resume == OfficialEffectResumeKind::kApplyPrimitive
        && state->effect_interpreter.reserved2[2] != 0) {
        const std::uint32_t total = state->effect_interpreter.reserved2[0];
        const std::uint32_t completed = state->effect_interpreter.reserved2[1];
        if (total == 0 || total > 0xffU || completed >= total) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                static_cast<std::int32_t>(total));
            return false;
        }
        // State::step executes the first SelectDamageMulti frame before the
        // first decision is exposed, so only total-completed-1 frames remain
        // below the SelectedDamageMulti callback.
        const std::uint32_t remaining = total - completed - 1U;
        if (!official_push_continuation(
                state, OfficialContinuationId::kAfterEffect)
            || !official_push_continuation(
                state, OfficialContinuationId::kSelectedDamageMultiAll)) {
            return false;
        }
        for (std::uint32_t index = 0; index < remaining; ++index) {
            if (!official_push_continuation(
                    state, OfficialContinuationId::kSelectDamageMulti)) {
                return false;
            }
        }
        return official_push_continuation(
            state, OfficialContinuationId::kSelectedDamageMulti);
    }
    if (resume == OfficialEffectResumeKind::kEnergyLoop) {
        return official_push_continuation(
                state, OfficialContinuationId::kSelectedPokemonEnergy)
            && official_push_continuation(
                state,
                OfficialContinuationId::kSelectedPokemonEnergyLoop,
                1,
                state->select_player);
    }
    if (resume == OfficialEffectResumeKind::kRemoveDamageCounter) {
        return official_push_continuation(
                state, OfficialContinuationId::kAfterEffect)
            && official_push_continuation(
                state, OfficialContinuationId::kSelectedRemoveDamageCounter);
    }
    if (resume == OfficialEffectResumeKind::kSelectActivate
        || resume == OfficialEffectResumeKind::kSelectEffect) {
        return official_push_continuation(
                state, OfficialContinuationId::kAfterEffect)
            && official_push_continuation(
                state, official_effect_selection_callback(*state));
    }
    if (resume == OfficialEffectResumeKind::kDamageCounterAny) {
        const std::uint32_t total =
            state->effect_interpreter.damage_counter_any_total;
        const std::int32_t remaining = state->remain_damage_counter;
        if (total == 0 || total > 0xffU || remaining <= 0
            || static_cast<std::uint32_t>(remaining) > total) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                static_cast<std::int32_t>(total));
            return false;
        }
        const std::uint8_t called_count = static_cast<std::uint8_t>(
            total - static_cast<std::uint32_t>(remaining) + 1U);
        if (!official_push_continuation(
                state, OfficialContinuationId::kAfterEffect)) {
            return false;
        }
        if (called_count < total && !official_push_continuation(
                state,
                OfficialContinuationId::kSelectDamageCounterAny,
                0,
                0,
                0,
                0,
                static_cast<std::uint8_t>(total))) {
            return false;
        }
        if (called_count < total) {
            state->continuations.values[
                state->continuations.count - 1].called_count = called_count;
        }
        return official_push_continuation(
            state, OfficialContinuationId::kSelectedDamageCounterAny);
    }
    return official_push_continuation(
        state, official_effect_selection_callback(*state));
}

PTCG_OFFICIAL_CONT_HD inline bool
official_consume_effect_selection_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)) return false;
    const auto resume = static_cast<OfficialEffectResumeKind>(
        state->effect_interpreter.resume_kind);
    const bool play_effect =
        (state->flow_flags & kOfficialPlayEffectReturnToMainFlag) != 0;
    const bool ability_effect = state->trigger_resolver.active != 0
        && (state->flow_flags & kOfficialAbilityReturnToMainFlag) != 0;
    const bool trigger_effect = state->trigger_resolver.active != 0
        && (state->flow_flags & kOfficialRefreshTriggerReturnToMainFlag) != 0;
    const bool refresh_knockout_trigger_effect =
        official_trigger_inside_refresh_knockout(*state);
    const bool refresh_knockout_return_frame =
        official_trigger_returns_to_knockout_proc(*state);
    OfficialContinuationId refresh_knockout_parent =
        OfficialContinuationId::kMainSelect;
    OfficialContinuationId refresh_knockout_after =
        OfficialContinuationId::kAfterRefresh;
    if (refresh_knockout_trigger_effect
        && !official_trigger_return_continuations(
            state,
            &refresh_knockout_parent,
            &refresh_knockout_after)) {
        return false;
    }
    const bool attack_trigger_effect = state->trigger_resolver.active != 0
        && state->attack_flow_stage == static_cast<std::uint8_t>(
            OfficialAttackStage::kAfterAttackTriggers);
    const bool pre_attack_effect = state->attack_flow_stage
        == static_cast<std::uint8_t>(OfficialAttackStage::kPreEffects);
    const bool post_attack_effect = state->attack_flow_stage
        == static_cast<std::uint8_t>(OfficialAttackStage::kPostEffects);
    const std::uint16_t base_prefix_count = play_effect || ability_effect
        ? 2
        : ((trigger_effect || attack_trigger_effect)
                ? 3
                : (refresh_knockout_trigger_effect
                    ? (refresh_knockout_return_frame ? 4 : 3)
                    : (pre_attack_effect
                        ? 2
                        : (post_attack_effect ? 1 : 0))));
    const bool retain_effect_frame =
        resume != OfficialEffectResumeKind::kSkillChooseEffect
        && resume != OfficialEffectResumeKind::kEnemySkillChooseEffect
        && state->effect_interpreter.effect_index + 1
            < state->effect_interpreter.effect_count;
    const bool retain_repeat_frame =
        state->effect_interpreter.repeat_continuation != 0
        && state->effect_interpreter.repeat_index + 1
            < state->effect_interpreter.repeat_count;
    const bool retain_separator_frame =
        state->effect_interpreter.separator_pending != 0;
    const std::uint16_t prefix_count = static_cast<std::uint16_t>(
        base_prefix_count + (retain_effect_frame ? 1 : 0)
            + (retain_repeat_frame ? 1 : 0)
            + (retain_separator_frame ? 1 : 0));
    if (resume == OfficialEffectResumeKind::kApplyPrimitive
        && state->effect_interpreter.reserved2[2] != 0) {
        const std::uint32_t total = state->effect_interpreter.reserved2[0];
        const std::uint32_t completed = state->effect_interpreter.reserved2[1];
        if (total == 0 || total > 0xffU || completed >= total) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                static_cast<std::int32_t>(total));
            return false;
        }
        const std::uint32_t remaining = total - completed - 1U;
        const std::uint16_t expected_count = static_cast<std::uint16_t>(
            prefix_count + 3U + remaining);
        const std::uint16_t selected_all_index = prefix_count + 1U;
        const std::uint16_t select_begin = selected_all_index + 1U;
        bool matches = base_prefix_count > 0
            && state->continuations.count == expected_count
            && state->continuations.values[prefix_count].opcode
                == static_cast<std::uint16_t>(OfficialContinuationId::kAfterEffect)
            && state->continuations.values[selected_all_index].opcode
                == static_cast<std::uint16_t>(
                    OfficialContinuationId::kSelectedDamageMultiAll)
            && state->continuations.values[expected_count - 1].opcode
                == static_cast<std::uint16_t>(
                    OfficialContinuationId::kSelectedDamageMulti);
        for (std::uint32_t index = 0; matches && index < remaining; ++index) {
            matches = state->continuations.values[select_begin + index].opcode
                == static_cast<std::uint16_t>(
                    OfficialContinuationId::kSelectDamageMulti);
        }
        if (!matches) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.count);
            return false;
        }
        state->continuations.count = 0;
        return true;
    }
    const bool after_effect_frame =
        resume == OfficialEffectResumeKind::kRemoveDamageCounter
        || resume == OfficialEffectResumeKind::kSelectActivate
        || resume == OfficialEffectResumeKind::kSelectEffect
        || resume == OfficialEffectResumeKind::kDamageCounterAny;
    const std::uint32_t damage_counter_total =
        state->effect_interpreter.damage_counter_any_total;
    const std::int32_t damage_counter_remaining = state->remain_damage_counter;
    const std::uint32_t damage_counter_called =
        damage_counter_remaining > 0
            && static_cast<std::uint32_t>(damage_counter_remaining)
                <= damage_counter_total
        ? damage_counter_total
            - static_cast<std::uint32_t>(damage_counter_remaining) + 1U
        : 0U;
    const bool damage_counter_repeat_frame =
        resume == OfficialEffectResumeKind::kDamageCounterAny
        && damage_counter_called < damage_counter_total;
    const std::uint16_t expected_count = prefix_count
        + (resume == OfficialEffectResumeKind::kEnergyLoop
            ? 2
            : (damage_counter_repeat_frame ? 3
                : (after_effect_frame ? 2 : 1)));
    const OfficialContinuationId expected_top =
        official_effect_selection_callback(*state);
    const bool prefix_matches = play_effect
        ? state->continuations.values[0].opcode == static_cast<std::uint16_t>(
              OfficialContinuationId::kToMain)
            && state->continuations.values[1].opcode
                == static_cast<std::uint16_t>(OfficialContinuationId::kAfterPlay)
        : ability_effect
            ? state->continuations.values[0].opcode == static_cast<std::uint16_t>(
                  OfficialContinuationId::kToMain)
                && state->continuations.values[1].opcode
                    == static_cast<std::uint16_t>(
                        OfficialContinuationId::kAfterAbility)
        : pre_attack_effect
            ? state->continuations.values[0].opcode
                    == static_cast<std::uint16_t>(
                        OfficialContinuationId::kAttackEffects)
                && state->continuations.values[0].arg_type == 2
                && state->continuations.values[0].args[0] == 0
                && state->continuations.values[1].opcode
                    == static_cast<std::uint16_t>(
                        OfficialContinuationId::kAttackDamage)
        : post_attack_effect
            ? state->continuations.values[0].opcode == static_cast<std::uint16_t>(
                  OfficialContinuationId::kAfterAttack)
        : attack_trigger_effect
            ? state->continuations.values[0].opcode
                    == static_cast<std::uint16_t>(
                        OfficialContinuationId::kAfterAttack2)
                && state->continuations.values[1].opcode
                    == static_cast<std::uint16_t>(
                        OfficialContinuationId::kAfterAttackTrigger)
                && state->continuations.values[2].opcode
                    == static_cast<std::uint16_t>(
                        OfficialContinuationId::kAfterTriggerAbility)
                && state->continuations.values[2].arg_type == 1
                && state->continuations.values[2].args[0]
                    == state->trigger_resolver.depth
        : trigger_effect
            ? state->continuations.values[0].opcode
                == static_cast<std::uint16_t>(OfficialContinuationId::kMainSelect)
            && state->continuations.values[1].opcode
                == static_cast<std::uint16_t>(OfficialContinuationId::kAfterRefresh)
            && state->continuations.values[2].opcode
                == static_cast<std::uint16_t>(
                    OfficialContinuationId::kAfterTriggerAbility)
            && state->continuations.values[2].arg_type == 1
            && state->continuations.values[2].args[0]
                == state->trigger_resolver.depth
        : refresh_knockout_trigger_effect
            && state->continuations.values[0].opcode
                == static_cast<std::uint16_t>(refresh_knockout_parent)
            && state->continuations.values[1].opcode
                == static_cast<std::uint16_t>(refresh_knockout_after)
            && (!refresh_knockout_return_frame
                || state->continuations.values[2].opcode
                    == static_cast<std::uint16_t>(
                        official_trigger_knockout_return_continuation(*state)))
            && state->continuations.values[
                    refresh_knockout_return_frame ? 3 : 2].opcode
                == static_cast<std::uint16_t>(
                    OfficialContinuationId::kAfterTriggerAbility)
            && state->continuations.values[
                    refresh_knockout_return_frame ? 3 : 2].arg_type == 1
            && state->continuations.values[
                    refresh_knockout_return_frame ? 3 : 2].args[0]
                == state->trigger_resolver.depth;
    const bool effect_frame_matches = !retain_effect_frame
        || state->continuations.values[base_prefix_count].opcode
            == static_cast<std::uint16_t>(
                (pre_attack_effect || post_attack_effect)
                    ? OfficialContinuationId::kAttackEffect
                    : OfficialContinuationId::kActivateSkillEffect);
    const std::uint16_t repeat_frame_index = static_cast<std::uint16_t>(
        base_prefix_count + (retain_effect_frame ? 1 : 0));
    const bool repeat_frame_matches = !retain_repeat_frame
        || (state->continuations.values[repeat_frame_index].opcode
                == state->effect_interpreter.repeat_continuation
            && state->continuations.values[repeat_frame_index].arg_type == 1
            && state->continuations.values[repeat_frame_index].args[0]
                == state->effect_interpreter.effect_index
            && state->continuations.values[repeat_frame_index].call_count
                == state->effect_interpreter.repeat_count
            && state->continuations.values[repeat_frame_index].called_count
                == state->effect_interpreter.repeat_index + 1);
    const std::uint16_t separator_frame_index = static_cast<std::uint16_t>(
        repeat_frame_index + (retain_repeat_frame ? 1 : 0));
    const bool separator_frame_matches = !retain_separator_frame
        || state->continuations.values[separator_frame_index].opcode
            == static_cast<std::uint16_t>(
                OfficialContinuationId::kSeparatorProc);
    const bool after_effect_frame_matches = !after_effect_frame
        || state->continuations.values[
                expected_count - (damage_counter_repeat_frame ? 3 : 2)].opcode
            == static_cast<std::uint16_t>(
                OfficialContinuationId::kAfterEffect);
    const OfficialContinuationPod& damage_counter_frame =
        state->continuations.values[expected_count - 2];
    const bool damage_counter_repeat_frame_matches =
        resume != OfficialEffectResumeKind::kDamageCounterAny
        || (damage_counter_total > 0 && damage_counter_total <= 0xffU
            && damage_counter_called > 0
            && damage_counter_called <= damage_counter_total
            && (!damage_counter_repeat_frame
                || (damage_counter_frame.opcode == static_cast<std::uint16_t>(
                        OfficialContinuationId::kSelectDamageCounterAny)
                    && damage_counter_frame.arg_type == 0
                    && damage_counter_frame.call_count == damage_counter_total
                    && damage_counter_frame.called_count
                        == damage_counter_called)));
    const bool energy_cleanup_frame_matches =
        resume != OfficialEffectResumeKind::kEnergyLoop
        || state->continuations.values[expected_count - 2].opcode
            == static_cast<std::uint16_t>(
                OfficialContinuationId::kSelectedPokemonEnergy);
    if (base_prefix_count == 0
        || state->continuations.count != expected_count
        || !prefix_matches
        || !effect_frame_matches
        || !repeat_frame_matches
        || !separator_frame_matches
        || !after_effect_frame_matches
        || !damage_counter_repeat_frame_matches
        || !energy_cleanup_frame_matches
        || state->continuations.values[expected_count - 1].opcode
            != static_cast<std::uint16_t>(expected_top)) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            state->continuations.count);
        return false;
    }
    state->continuations.count = 0;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_trigger_return_continuations(
    OfficialStatePod* state,
    OfficialContinuationId* parent,
    OfficialContinuationId* after_resolver) {
    if (state->attack_flow_stage == static_cast<std::uint8_t>(
            OfficialAttackStage::kAfterAttackTriggers)) {
        *parent = OfficialContinuationId::kAfterAttack2;
        *after_resolver = OfficialContinuationId::kAfterAttackTrigger;
        return true;
    }
    if (state->attack_flow_stage == static_cast<std::uint8_t>(
            OfficialAttackStage::kRefresh)) {
        *parent = OfficialContinuationId::kAfterAttack4;
        *after_resolver = OfficialContinuationId::kAfterRefresh;
        return true;
    }
    *after_resolver = OfficialContinuationId::kAfterRefresh;
    if ((state->flow_flags & kOfficialRefreshTriggerReturnToMainFlag) != 0
        || (state->flow_flags & kOfficialRefreshReturnToMainFlag) != 0) {
        *parent = OfficialContinuationId::kMainSelect;
        return true;
    }
    switch (static_cast<OfficialTurnFlowStage>(state->turn_flow_stage)) {
        case OfficialTurnFlowStage::kTurnEndTriggers:
        case OfficialTurnFlowStage::kTurnEndRefresh:
            *parent = OfficialContinuationId::kTurnEnd2;
            return true;
        case OfficialTurnFlowStage::kCheckupRefresh:
            *parent = OfficialContinuationId::kPokemonCheckup;
            return true;
        case OfficialTurnFlowStage::kCheckupTriggers:
        case OfficialTurnFlowStage::kCheckupEndRefresh:
            *parent = OfficialContinuationId::kPokemonCheckupEnd;
            return true;
        default:
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->turn_flow_stage);
            return false;
    }
}

PTCG_OFFICIAL_CONT_HD inline bool
official_build_trigger_activation_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || state->trigger_resolver.active == 0
        || state->trigger_resolver.awaiting_activation == 0) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->trigger_resolver.activation_kind);
        }
        return false;
    }
    OfficialContinuationId parent{};
    OfficialContinuationId after_resolver{};
    if (!official_trigger_return_continuations(
            state, &parent, &after_resolver)) return false;
    const bool refresh_knockout =
        official_trigger_returns_to_knockout_proc(*state);
    state->continuations.count = 0;
    if (!official_push_continuation(state, parent)
        || !official_push_continuation(state, after_resolver)
        || (refresh_knockout && !official_push_continuation(
            state, official_trigger_knockout_return_continuation(*state)))
        || !official_push_continuation(
            state,
            OfficialContinuationId::kAfterTriggerAbility,
            1,
            state->trigger_resolver.depth)) {
        return false;
    }
    const auto kind = static_cast<OfficialTriggerActivationKind>(
        state->trigger_resolver.activation_kind);
    OfficialContinuationId selected = OfficialContinuationId::kSelectedActivateAbility;
    if (kind == OfficialTriggerActivationKind::kChooseEffect) {
        selected = OfficialContinuationId::kSelectedWhichEffect;
    } else if (kind == OfficialTriggerActivationKind::kEnemyChooseEffect) {
        selected = OfficialContinuationId::kEnemySelectedWhichEffect;
    } else if (kind != OfficialTriggerActivationKind::kOptional) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            state->trigger_resolver.activation_kind);
        return false;
    }
    if (!official_push_continuation(state, selected)) return false;
    // Official ActivateAbility clears `changed`, while pulling the trigger marks
    // the surrounding refresh as stateChanged.
    state->control_flags &= static_cast<std::uint8_t>(~kOfficialChangedFlag);
    state->control_flags |= 1U << 4U;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_consume_trigger_activation_continuation_mirror(OfficialStatePod* state) {
    const bool refresh_knockout = state != nullptr
        && official_trigger_returns_to_knockout_proc(*state);
    const std::uint16_t after_trigger_index = static_cast<std::uint16_t>(
        2 + (refresh_knockout ? 1 : 0));
    if (state == nullptr || !official_pod_ok(state)
        || state->continuations.count != after_trigger_index + 2
        || state->continuations.values[after_trigger_index].opcode
            != static_cast<std::uint16_t>(
            OfficialContinuationId::kAfterTriggerAbility)) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.count);
        }
        return false;
    }
    OfficialContinuationId parent{};
    OfficialContinuationId after_resolver{};
    if (!official_trigger_return_continuations(
            state, &parent, &after_resolver)
        || state->continuations.values[0].opcode
            != static_cast<std::uint16_t>(parent)
        || state->continuations.values[1].opcode
            != static_cast<std::uint16_t>(after_resolver)) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.values[0].opcode);
        }
        return false;
    }
    if (refresh_knockout && state->continuations.values[2].opcode
            != static_cast<std::uint16_t>(
                official_trigger_knockout_return_continuation(*state))) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            state->continuations.values[2].opcode);
        return false;
    }
    const std::uint16_t selected = state->continuations.values[
        after_trigger_index + 1].opcode;
    if (selected != static_cast<std::uint16_t>(
            OfficialContinuationId::kSelectedActivateAbility)
        && selected != static_cast<std::uint16_t>(
            OfficialContinuationId::kSelectedWhichEffect)
        && selected != static_cast<std::uint16_t>(
            OfficialContinuationId::kEnemySelectedWhichEffect)) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            selected);
        return false;
    }
    state->continuations.count = 0;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_trigger_order_awaiting_action(const OfficialStatePod& state) {
    return state.trigger_resolver.active != 0
        && state.trigger_resolver.awaiting_order != 0
        && state.select_type
            == static_cast<std::uint8_t>(OfficialSelectTypeId::kSkill)
        && state.select_context == kOfficialSelectContextSkillOrder;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_build_trigger_order_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || !official_trigger_order_awaiting_action(*state)) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->turn_flow_stage);
        }
        return false;
    }
    OfficialContinuationId parent{};
    OfficialContinuationId after_resolver{};
    if (!official_trigger_return_continuations(
            state, &parent, &after_resolver)) return false;
    const bool refresh_knockout =
        official_trigger_returns_to_knockout_proc(*state);
    state->continuations.count = 0;
    return official_push_continuation(state, parent)
        && official_push_continuation(state, after_resolver)
        && (!refresh_knockout || official_push_continuation(
            state, official_trigger_knockout_return_continuation(*state)))
        && official_push_continuation(
            state,
            OfficialContinuationId::kSelectedSkillOrder,
            1,
            state->trigger_resolver.depth);
}

PTCG_OFFICIAL_CONT_HD inline bool
official_consume_trigger_order_continuation_mirror(OfficialStatePod* state) {
    const bool refresh_knockout = state != nullptr
        && official_trigger_returns_to_knockout_proc(*state);
    const std::uint16_t selected_index = static_cast<std::uint16_t>(
        2 + (refresh_knockout ? 1 : 0));
    if (state == nullptr || !official_pod_ok(state)
        || !official_trigger_order_awaiting_action(*state)
        || state->continuations.count != selected_index + 1
        || state->continuations.values[selected_index].opcode
            != static_cast<std::uint16_t>(
                OfficialContinuationId::kSelectedSkillOrder)
        || state->continuations.values[selected_index].arg_type != 1
        || state->continuations.values[selected_index].args[0]
            != state->trigger_resolver.depth) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.count);
        }
        return false;
    }
    OfficialContinuationId parent{};
    OfficialContinuationId after_resolver{};
    if (!official_trigger_return_continuations(
            state, &parent, &after_resolver)
        || state->continuations.values[0].opcode
            != static_cast<std::uint16_t>(parent)
        || state->continuations.values[1].opcode
            != static_cast<std::uint16_t>(after_resolver)) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.values[0].opcode);
        }
        return false;
    }
    if (refresh_knockout && state->continuations.values[2].opcode
            != static_cast<std::uint16_t>(
                official_trigger_knockout_return_continuation(*state))) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            state->continuations.values[2].opcode);
        return false;
    }
    state->continuations.count = 0;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_prize_selection_awaiting_action(const OfficialStatePod& state) {
    return state.select_type
            == static_cast<std::uint8_t>(OfficialSelectTypeId::kCard)
        && state.select_context == kOfficialSelectContextToHand
        && state.refresh_flow_stage == static_cast<std::uint8_t>(
            OfficialRefreshFlowStage::kKnockout)
        && state.reserved_flow == static_cast<std::uint8_t>(
            OfficialKnockoutStage::kPrizeSelection);
}

PTCG_OFFICIAL_CONT_HD inline bool official_refresh_parent_continuation(
    OfficialStatePod* state,
    OfficialContinuationId* parent) {
    if ((state->flow_flags & kOfficialRefreshReturnToMainFlag) != 0) {
        *parent = OfficialContinuationId::kMainSelect;
        return true;
    }
    if (state->attack_flow_stage == static_cast<std::uint8_t>(
            OfficialAttackStage::kRefresh)) {
        *parent = OfficialContinuationId::kAfterAttack4;
        return true;
    }
    switch (static_cast<OfficialTurnFlowStage>(state->turn_flow_stage)) {
        case OfficialTurnFlowStage::kTurnEndRefresh:
            *parent = OfficialContinuationId::kTurnEnd2;
            return true;
        case OfficialTurnFlowStage::kCheckupRefresh:
            *parent = OfficialContinuationId::kPokemonCheckup;
            return true;
        case OfficialTurnFlowStage::kCheckupEndRefresh:
            *parent = OfficialContinuationId::kTurnStart;
            return true;
        default:
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->turn_flow_stage);
            return false;
    }
}

PTCG_OFFICIAL_CONT_HD inline bool
official_refresh_overflow_awaiting_action(const OfficialStatePod& state) {
    const auto stage = static_cast<OfficialRefreshFlowStage>(
        state.refresh_flow_stage);
    return (stage == OfficialRefreshFlowStage::kBenchOverflow
            && state.select_type == static_cast<std::uint8_t>(
                OfficialSelectTypeId::kCard)
            && state.select_context == kOfficialSelectContextDiscard)
        || (stage == OfficialRefreshFlowStage::kToolOverflow
            && state.select_type == static_cast<std::uint8_t>(
                OfficialSelectTypeId::kAttachedCard)
            && state.select_context == kOfficialSelectContextDiscardTool);
}

PTCG_OFFICIAL_CONT_HD inline bool
official_append_pending_bench_overflow_continuations(OfficialStatePod* state) {
    if (state->last_stadium_player < 0 || state->last_stadium_player > 1) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            state->last_stadium_player);
        return false;
    }
    const std::int32_t order[2] = {
        1 ^ state->last_stadium_player,
        state->last_stadium_player,
    };
    for (std::int32_t ordinal = 0; ordinal < 2; ++ordinal) {
        const std::int32_t player = order[ordinal];
        const std::int32_t excess = static_cast<std::int32_t>(
            state->players[player].bench.count)
            - official_bench_capacity(state->players[player]);
        if (excess <= 0 || player == state->select_player) continue;
        if (!official_push_continuation(
                state,
                OfficialContinuationId::kSelectBenchMaxTrash,
                1,
                player)) {
            return false;
        }
    }
    return true;
}

template <std::size_t Capacity>
PTCG_OFFICIAL_CONT_HD inline bool
official_append_pending_tool_overflow_continuations(
    OfficialStatePod* state,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone,
    std::int32_t player) {
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        const OfficialCardRefPod ref = zone.values[index];
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        if (card == nullptr) return false;
        const std::int32_t excess = official_attached_tool_count(*state, *card)
            - official_tool_capacity(*card);
        if (excess <= 0 || ref == state->context_card) continue;
        if (!official_push_continuation(
                state,
                OfficialContinuationId::kSelectToolMaxTrash,
                3,
                player,
                ref.index)) {
            return false;
        }
    }
    return true;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_validate_pending_bench_overflow_continuations(
    OfficialStatePod* state,
    std::uint16_t* continuation_index) {
    if (state->last_stadium_player < 0 || state->last_stadium_player > 1) {
        return false;
    }
    const std::int32_t order[2] = {
        1 ^ state->last_stadium_player,
        state->last_stadium_player,
    };
    for (std::int32_t ordinal = 0; ordinal < 2; ++ordinal) {
        const std::int32_t player = order[ordinal];
        const std::int32_t excess = static_cast<std::int32_t>(
            state->players[player].bench.count)
            - official_bench_capacity(state->players[player]);
        if (excess <= 0 || player == state->select_player) continue;
        if (*continuation_index >= state->continuations.count) return false;
        const OfficialContinuationPod& frame =
            state->continuations.values[(*continuation_index)++];
        if (frame.opcode != static_cast<std::uint16_t>(
                OfficialContinuationId::kSelectBenchMaxTrash)
            || frame.arg_type != 1
            || frame.args[0] != player) {
            return false;
        }
    }
    return true;
}

template <std::size_t Capacity>
PTCG_OFFICIAL_CONT_HD inline bool
official_validate_pending_tool_overflow_continuations(
    OfficialStatePod* state,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone,
    std::int32_t player,
    std::uint16_t* continuation_index) {
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        const OfficialCardRefPod ref = zone.values[index];
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        if (card == nullptr) return false;
        const std::int32_t excess = official_attached_tool_count(*state, *card)
            - official_tool_capacity(*card);
        if (excess <= 0 || ref == state->context_card) continue;
        if (*continuation_index >= state->continuations.count) return false;
        const OfficialContinuationPod& frame =
            state->continuations.values[(*continuation_index)++];
        if (frame.opcode != static_cast<std::uint16_t>(
                OfficialContinuationId::kSelectToolMaxTrash)
            || frame.arg_type != 3
            || frame.args[0] != player
            || frame.args[1] != ref.index) {
            return false;
        }
    }
    return true;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_build_refresh_overflow_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || !official_refresh_overflow_awaiting_action(*state)) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state, OfficialPodError::kUnsupportedContinuation,
                state->refresh_flow_stage);
        }
        return false;
    }
    OfficialContinuationId parent{};
    if (!official_refresh_parent_continuation(state, &parent)) return false;
    const bool tool = state->refresh_flow_stage == static_cast<std::uint8_t>(
        OfficialRefreshFlowStage::kToolOverflow);
    state->continuations.count = 0;
    if (!official_push_continuation(state, parent)
        || !official_push_continuation(
            state, OfficialContinuationId::kAfterRefresh)
        || !official_push_continuation(
            state, OfficialContinuationId::kToolCountProc)) {
        return false;
    }
    if (tool) {
        if (!official_push_continuation(state, OfficialContinuationId::kKoProc)) {
            return false;
        }
        const std::int32_t order[2] = {
            state->first_player,
            1 - state->first_player,
        };
        for (std::int32_t ordinal = 0; ordinal < 2; ++ordinal) {
            const std::int32_t player = order[ordinal];
            const OfficialPlayerStatePod& ps = state->players[player];
            if (!official_append_pending_tool_overflow_continuations(
                    state, ps.active, player)
                || !official_append_pending_tool_overflow_continuations(
                    state, ps.bench, player)) {
                return false;
            }
        }
    } else if (!official_append_pending_bench_overflow_continuations(state)) {
        return false;
    }
    return official_push_continuation(
        state,
        tool ? OfficialContinuationId::kSelectedToolMaxTrash
             : OfficialContinuationId::kSelectedBenchMaxTrash);
}

PTCG_OFFICIAL_CONT_HD inline bool
official_consume_refresh_overflow_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || !official_refresh_overflow_awaiting_action(*state)) {
        return false;
    }
    OfficialContinuationId parent{};
    if (!official_refresh_parent_continuation(state, &parent)) return false;
    const bool tool = state->refresh_flow_stage == static_cast<std::uint8_t>(
        OfficialRefreshFlowStage::kToolOverflow);
    const OfficialContinuationId selected = tool
        ? OfficialContinuationId::kSelectedToolMaxTrash
        : OfficialContinuationId::kSelectedBenchMaxTrash;
    if (state->continuations.count < (tool ? 5 : 4)
        || state->continuations.values[0].opcode != static_cast<std::uint16_t>(
            parent)
        || state->continuations.values[1].opcode != static_cast<std::uint16_t>(
            OfficialContinuationId::kAfterRefresh)
        || state->continuations.values[2].opcode != static_cast<std::uint16_t>(
            OfficialContinuationId::kToolCountProc)) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            state->continuations.count);
        return false;
    }
    std::uint16_t continuation_index = 3;
    bool pending_matches = true;
    if (tool) {
        if (state->continuations.values[continuation_index].opcode
            != static_cast<std::uint16_t>(OfficialContinuationId::kKoProc)) {
            pending_matches = false;
        } else {
            ++continuation_index;
            const std::int32_t order[2] = {
                state->first_player,
                1 - state->first_player,
            };
            for (std::int32_t ordinal = 0;
                 ordinal < 2 && pending_matches;
                 ++ordinal) {
                const std::int32_t player = order[ordinal];
                const OfficialPlayerStatePod& ps = state->players[player];
                pending_matches =
                    official_validate_pending_tool_overflow_continuations(
                        state, ps.active, player, &continuation_index)
                    && official_validate_pending_tool_overflow_continuations(
                        state, ps.bench, player, &continuation_index);
            }
        }
    } else {
        pending_matches = official_validate_pending_bench_overflow_continuations(
            state, &continuation_index);
    }
    if (!pending_matches
        || continuation_index + 1 != state->continuations.count
        || state->continuations.values[continuation_index].opcode
            != static_cast<std::uint16_t>(selected)) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            state->continuations.count);
        return false;
    }
    state->continuations.count = 0;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_build_prize_selection_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || !official_prize_selection_awaiting_action(*state)) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->reserved_flow);
        }
        return false;
    }
    OfficialContinuationId parent{};
    if (!official_refresh_parent_continuation(state, &parent)) return false;
    state->continuations.count = 0;
    if (!official_push_continuation(state, parent)
        || !official_push_continuation(
            state, OfficialContinuationId::kAfterRefresh)) {
        return false;
    }
    for (std::uint16_t index = 0; index < state->prize_requests.count; ++index) {
        const OfficialPrizeRequestPod request = state->prize_requests.values[index];
        if (!official_push_continuation(
                state,
                OfficialContinuationId::kSelectPrize,
                4,
                request.player,
                request.count,
                0)) {
            return false;
        }
    }
    return official_push_continuation(
        state, OfficialContinuationId::kSelectedPrize);
}

PTCG_OFFICIAL_CONT_HD inline bool
official_consume_prize_selection_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || !official_prize_selection_awaiting_action(*state)
        || state->continuations.count
            != static_cast<std::uint16_t>(3 + state->prize_requests.count)
        || state->continuations.values[1].opcode != static_cast<std::uint16_t>(
            OfficialContinuationId::kAfterRefresh)
        || state->continuations.values[state->continuations.count - 1].opcode
            != static_cast<std::uint16_t>(
                OfficialContinuationId::kSelectedPrize)) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.count);
        }
        return false;
    }
    OfficialContinuationId parent{};
    if (!official_refresh_parent_continuation(state, &parent)
        || state->continuations.values[0].opcode
            != static_cast<std::uint16_t>(parent)) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.values[0].opcode);
        }
        return false;
    }
    for (std::uint16_t index = 0; index < state->prize_requests.count; ++index) {
        const OfficialPrizeRequestPod request = state->prize_requests.values[index];
        const OfficialContinuationPod& frame = state->continuations.values[index + 2];
        if (frame.opcode != static_cast<std::uint16_t>(
                OfficialContinuationId::kSelectPrize)
            || frame.arg_type != 4
            || frame.args[0] != request.player
            || frame.args[1] != request.count
            || frame.args[2] != 0) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                frame.opcode);
            return false;
        }
    }
    state->continuations.count = 0;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline bool
official_active_replacement_awaiting_action(const OfficialStatePod& state) {
    const bool replacement_selection = state.select_type
            == static_cast<std::uint8_t>(OfficialSelectTypeId::kCard)
        && state.select_context == kOfficialSelectContextToActive
        && state.reserved_flow == static_cast<std::uint8_t>(
            OfficialKnockoutStage::kActiveReplacement);
    return replacement_selection
        && (state.refresh_flow_stage == static_cast<std::uint8_t>(
                OfficialRefreshFlowStage::kKnockout)
            || state.attack_flow_stage == static_cast<std::uint8_t>(
                OfficialAttackStage::kActiveReplacement));
}

PTCG_OFFICIAL_CONT_HD inline bool
official_build_active_replacement_continuation_mirror(OfficialStatePod* state) {
    if (state == nullptr || !official_pod_ok(state)
        || !official_active_replacement_awaiting_action(*state)) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->reserved_flow);
        }
        return false;
    }
    const bool attack_pre_refresh = state->attack_flow_stage
        == static_cast<std::uint8_t>(
            OfficialAttackStage::kActiveReplacement);
    state->continuations.count = 0;
    if (attack_pre_refresh) {
        if (!official_push_continuation(
                state, OfficialContinuationId::kAfterAttack3)) {
            return false;
        }
    } else {
        OfficialContinuationId parent{};
        if (!official_refresh_parent_continuation(state, &parent)
            || !official_push_continuation(state, parent)
            || !official_push_continuation(
                state, OfficialContinuationId::kAfterRefresh)
            || !official_push_continuation(
                state, OfficialContinuationId::kResolveTriggerStack, 1, 0)) {
            return false;
        }
    }
    const std::int32_t active = official_active_player(*state);
    const std::int32_t order[2] = {active, 1 - active};
    for (int ordinal = 0; ordinal < 2; ++ordinal) {
        const std::int32_t player = order[ordinal];
        if (player == state->select_player
            || (state->pending_active_replacement_mask & (1U << player)) == 0) {
            continue;
        }
        if (!official_push_continuation(
                state,
                OfficialContinuationId::kSelectActivePokemon,
                1,
                player)) {
            return false;
        }
    }
    return official_push_continuation(
        state, OfficialContinuationId::kSelectedSwitchPokemon);
}

PTCG_OFFICIAL_CONT_HD inline bool
official_consume_active_replacement_continuation_mirror(OfficialStatePod* state) {
    const bool attack_pre_refresh = state != nullptr
        && state->attack_flow_stage == static_cast<std::uint8_t>(
            OfficialAttackStage::kActiveReplacement);
    if (state == nullptr || !official_pod_ok(state)
        || !official_active_replacement_awaiting_action(*state)
        || state->continuations.count == 0
        || state->continuations.values[state->continuations.count - 1].opcode
            != static_cast<std::uint16_t>(
                OfficialContinuationId::kSelectedSwitchPokemon)) {
        if (state != nullptr && official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.count);
        }
        return false;
    }
    if (attack_pre_refresh) {
        if (state->continuations.count < 2
            || state->continuations.values[0].opcode
                != static_cast<std::uint16_t>(
                    OfficialContinuationId::kAfterAttack3)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.count);
            return false;
        }
        state->continuations.count = 0;
        return true;
    }
    if (state->continuations.count < 4
        || state->continuations.values[1].opcode != static_cast<std::uint16_t>(
            OfficialContinuationId::kAfterRefresh)
        || state->continuations.values[2].opcode != static_cast<std::uint16_t>(
            OfficialContinuationId::kResolveTriggerStack)
        || state->continuations.values[2].arg_type != 1
        || state->continuations.values[2].args[0] != 0) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedContinuation,
            state->continuations.count);
        return false;
    }
    OfficialContinuationId parent{};
    if (!official_refresh_parent_continuation(state, &parent)
        || state->continuations.values[0].opcode
            != static_cast<std::uint16_t>(parent)) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                state->continuations.values[0].opcode);
        }
        return false;
    }
    state->continuations.count = 0;
    return true;
}

PTCG_OFFICIAL_CONT_HD inline OfficialFlowStatus
official_dispatch_no_action_continuation(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    const OfficialContinuationCall call = official_begin_continuation_call(state);
    if (!call.valid) return OfficialFlowStatus::kError;
    switch (static_cast<OfficialContinuationId>(call.frame.opcode)) {
        case OfficialContinuationId::kMainSelect: {
            const OfficialMainResult result = official_prepare_main_options(state, rules);
            if (result == OfficialMainResult::kError) {
                return OfficialFlowStatus::kError;
            }
            if (result == OfficialMainResult::kNeedsAction) {
                if (!official_push_continuation(
                        state, OfficialContinuationId::kSelectedMain)) {
                    return OfficialFlowStatus::kError;
                }
            }
            if (!official_finish_continuation_call(state, call)) {
                return OfficialFlowStatus::kError;
            }
            if (result == OfficialMainResult::kNeedsAction) {
                ++state->turn_action_count;
                return OfficialFlowStatus::kNeedsAction;
            }
            return OfficialFlowStatus::kIdle;
        }
        default:
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedContinuation,
                call.frame.opcode);
            return OfficialFlowStatus::kError;
    }
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_CONT_HD
