#pragma once

#include <cstdint>

#include "ptcg_cuda/official_conditions_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_TRIGGER_HD __host__ __device__
#else
#define PTCG_OFFICIAL_TRIGGER_HD
#endif

namespace ptcg::cuda_engine {

constexpr std::uint64_t kOfficialSkillOnceTurnFlag = 1ULL << 1;
constexpr std::uint64_t kOfficialSkillTriggerNotStackFlag = 1ULL << 3;
constexpr std::uint8_t kOfficialTriggerEvolveFromHand = 8;

PTCG_OFFICIAL_TRIGGER_HD inline bool official_trigger_area_ref_equal(
    OfficialAreaRefPod left,
    OfficialAreaRefPod right) {
    return left.card == right.card && left.move_counter == right.move_counter;
}

PTCG_OFFICIAL_TRIGGER_HD inline const OfficialTargetRule* official_trigger_target_rule(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialTriggerRule& trigger) {
    if (trigger.target_index < 0
        || static_cast<std::uint32_t>(trigger.target_index)
            >= rules.header->counts[static_cast<std::uint32_t>(
                OfficialRuleSection::kTargets)]) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, trigger.target_index);
        return nullptr;
    }
    return &rules.targets[trigger.target_index];
}

PTCG_OFFICIAL_TRIGGER_HD inline bool official_is_trigger_target(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod subject,
    const OfficialTargetRule& target,
    OfficialCardRefPod effect_card_ref) {
    if (target.values[kTargetPlayer] == 0) return true;
    const OfficialCardStatePod* effect_card = official_pod_card(state, effect_card_ref);
    const OfficialCardStatePod* card = official_pod_card(state, subject);
    if (effect_card == nullptr || card == nullptr
        || !official_target_player_matches(
            effect_card->player, card->player, target.values[kTargetPlayer])) {
        return false;
    }
    const std::int32_t area_count = target.values[kTargetAreaCount];
    if (area_count < 0 || area_count > 4) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, area_count);
        return false;
    }
    bool area_match = false;
    for (std::int32_t index = 0; index < area_count; ++index) {
        const OfficialArea area = static_cast<OfficialArea>(
            target.values[kTargetArea0 + index]);
        if (area == OfficialArea::kMe) {
            area_match = card->move_counter == effect_card->move_counter;
        } else if (area == OfficialArea::kAttach) {
            area_match = effect_card->attach_move_counter > 0
                && card->move_counter == effect_card->attach_move_counter;
        } else if (area == OfficialArea::kAll
            || area == static_cast<OfficialArea>(card->area)) {
            area_match = true;
        }
        if (area_match) break;
    }
    if (!area_match) return false;
    return official_match_target(
        state,
        rules,
        subject,
        target,
        official_pod_area_ref(state, effect_card_ref))
        == OfficialTargetMatchResult::kMatch;
}

PTCG_OFFICIAL_TRIGGER_HD inline bool official_trigger_first_condition_is_eager(
    const OfficialEffectRule& effect) {
    const auto type = static_cast<OfficialConditionTypeId>(
        effect.values[kEffectConditionType]);
    return type == OfficialConditionTypeId::kMyTurn
        || type == OfficialConditionTypeId::kTurn
        || type == OfficialConditionTypeId::kKoPreEnemyTurn
        || type == OfficialConditionTypeId::kKoAttackDamagePreEnemyTurn;
}

PTCG_OFFICIAL_TRIGGER_HD inline bool official_trigger_once_turn_exists(
    const OfficialStatePod& state,
    std::int32_t skill_id,
    OfficialAreaRefPod effect_card) {
    for (std::uint16_t index = 0; index < state.triggers.count; ++index) {
        const OfficialTriggeredAbilityPod& ability = state.triggers.values[index];
        if (ability.activate.skill_id == skill_id
            && official_trigger_area_ref_equal(
                ability.activate.effect_card, effect_card)) return true;
    }
    for (std::uint16_t index = 0; index < state.temporary_triggers.count; ++index) {
        const OfficialTriggeredAbilityPod& ability = state.temporary_triggers.values[index];
        if (ability.activate.skill_id == skill_id
            && official_trigger_area_ref_equal(
                ability.activate.effect_card, effect_card)) return true;
    }
    return false;
}

PTCG_OFFICIAL_TRIGGER_HD inline bool official_trigger_not_stack_exists(
    const OfficialStatePod& state,
    std::int32_t skill_id,
    OfficialCardRefPod subject) {
    for (std::uint16_t index = 0; index < state.temporary_triggers.count; ++index) {
        const OfficialTriggeredAbilityPod& ability = state.temporary_triggers.values[index];
        if (ability.activate.skill_id == skill_id
            && ability.trigger.subject.card == subject) return true;
    }
    return false;
}

PTCG_OFFICIAL_TRIGGER_HD inline bool official_trigger_skill_area_matches(
    const OfficialSkillRule& skill,
    OfficialArea area) {
    const std::int32_t count = skill.values[kSkillAreaCount];
    if (count < 0 || count > 3) return false;
    for (std::int32_t index = 0; index < count; ++index) {
        if (skill.values[kSkillArea0 + index] == static_cast<std::int32_t>(area)) {
            return true;
        }
    }
    return false;
}

PTCG_OFFICIAL_TRIGGER_HD inline bool official_trigger_list(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint8_t trigger_type,
    OfficialCardRefPod subject,
    OfficialCardRefPod object,
    OfficialCardRefPod effect_card_ref,
    std::int8_t depth) {
    const OfficialCardStatePod* effect_card = official_pod_card(state, effect_card_ref);
    if (effect_card != nullptr && official_continual_flag(*effect_card, 0)) return true;
    const OfficialCardRule* master = effect_card == nullptr ? nullptr : official_card_rule(
        rules, static_cast<std::uint32_t>(effect_card->card_id));
    if (master == nullptr || master->values[kCardAbilityId] == 0) return true;
    const OfficialSkillRule* skill = official_skill_rule(
        rules, static_cast<std::uint32_t>(master->values[kCardAbilityId]));
    if (skill == nullptr
        || !official_trigger_skill_area_matches(
            *skill, static_cast<OfficialArea>(effect_card->area))) return true;
    const std::int32_t trigger_offset = skill->values[kSkillTriggerOffset];
    const std::int32_t trigger_count = skill->values[kSkillTriggerCount];
    const std::uint32_t total_triggers = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kTriggers)];
    if (trigger_offset < 0 || trigger_count < 0
        || static_cast<std::uint64_t>(trigger_offset) + trigger_count > total_triggers) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, trigger_offset);
        return false;
    }
    for (std::int32_t index = 0; index < trigger_count; ++index) {
        const OfficialTriggerRule& trigger = rules.triggers[trigger_offset + index];
        if (trigger.type == 0 || trigger.type != trigger_type) continue;
        const OfficialTargetRule* target = official_trigger_target_rule(
            state, rules, trigger);
        if (target == nullptr
            || !official_is_trigger_target(
                state, rules, subject, *target, effect_card_ref)) continue;

        OfficialTriggeredAbilityPod pending{};
        pending.trigger.type = trigger_type;
        pending.trigger.depth = depth;
        if (subject.index != 0) {
            pending.trigger.subject = official_pod_area_ref(state, subject);
        }
        if (object.index != 0) {
            pending.trigger.object = official_pod_area_ref(state, object);
        }

        const std::int32_t effect_offset = skill->values[kSkillEffectOffset];
        const std::int32_t effect_count = skill->values[kSkillEffectCount];
        if (effect_count > 0) {
            const OfficialEffectRule& first = rules.effects[effect_offset];
            if ((first.flags & kOfficialEffectIsCondition) != 0
                && official_trigger_first_condition_is_eager(first)) {
                const OfficialTriggerInfoPod previous = state->trigger_info;
                state->trigger_info = pending.trigger;
                const OfficialConditionResult result = official_satisfy_condition(
                    state,
                    rules,
                    rules.effects + effect_offset,
                    effect_count,
                    0,
                    official_pod_area_ref(state, effect_card_ref),
                    effect_card->player);
                state->trigger_info = previous;
                if (result == OfficialConditionResult::kFalse) break;
                if (result != OfficialConditionResult::kTrue) return false;
            }
        }

        const OfficialAreaRefPod effect_card_area = official_pod_area_ref(
            state, effect_card_ref);
        if ((skill->flags & kOfficialSkillOnceTurnFlag) != 0
            && official_trigger_once_turn_exists(
                *state, skill->values[kSkillId], effect_card_area)) break;
        pending.activate.skill_id = skill->values[kSkillId];
        pending.activate.effect_card = effect_card_area;
        pending.activate.use_player = effect_card->player;
        if ((skill->flags & kOfficialSkillTriggerNotStackFlag) != 0
            && official_trigger_not_stack_exists(
                *state, pending.activate.skill_id, subject)) break;
        return official_pod_push(
            state,
            &state->temporary_triggers,
            pending,
            OfficialPodError::kTriggerStackOverflow);
    }
    return official_pod_ok(state);
}

template <std::size_t Capacity>
PTCG_OFFICIAL_TRIGGER_HD inline bool official_pull_trigger_zone(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone,
    std::uint8_t trigger_type,
    OfficialCardRefPod subject,
    OfficialCardRefPod object,
    std::int8_t depth) {
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        if (!official_trigger_list(
                state,
                rules,
                trigger_type,
                subject,
                object,
                zone.values[index],
                depth)) return false;
    }
    return true;
}

PTCG_OFFICIAL_TRIGGER_HD inline bool official_pull_trigger(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint8_t trigger_type,
    OfficialCardRefPod subject,
    OfficialCardRefPod object = {},
    std::int8_t depth = 0) {
    for (int player = 0; player < 2; ++player) {
        const OfficialPlayerStatePod& ps = state->players[player];
        if (!official_pull_trigger_zone(
                state, rules, ps.active, trigger_type, subject, object, depth)
            || !official_pull_trigger_zone(
                state, rules, ps.bench, trigger_type, subject, object, depth)) {
            return false;
        }
        if ((state->continual_state & 1U) == 0
            && !official_pull_trigger_zone(
                state, rules, ps.tool, trigger_type, subject, object, depth)) {
            return false;
        }
        if (!official_pull_trigger_zone(
                state, rules, ps.energy, trigger_type, subject, object, depth)
            || !official_pull_trigger_zone(
                state, rules, ps.trash, trigger_type, subject, object, depth)) {
            return false;
        }
    }
    return official_pull_trigger_zone(
        state, rules, state->stadium, trigger_type, subject, object, depth);
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_TRIGGER_HD
