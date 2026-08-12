#pragma once

#include <cstdint>

#include "ptcg_cuda/official_continual_refresh_pod.cuh"
#include "ptcg_cuda/official_effect_interpreter_pod.cuh"
#include "ptcg_cuda/official_trigger_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_RESOLVER_HD __host__ __device__
#else
#define PTCG_OFFICIAL_RESOLVER_HD
#endif

namespace ptcg::cuda_engine {

enum class OfficialTriggerResolverResult : std::int32_t {
    kComplete = 0,
    kNeedsAction = 1,
    kError = 2,
};

enum class OfficialTriggerActivationKind : std::uint8_t {
    kNone = 0,
    kOptional = 1,
    kChooseEffect = 2,
    kEnemyChooseEffect = 3,
};

constexpr std::uint64_t kOfficialSkillCanSelectActivateFlag = 1ULL << 2;
constexpr std::uint64_t kOfficialSkillCanActivateTrashFlag = 1ULL << 4;
constexpr std::uint8_t kOfficialSelectContextSkillOrder = 35;
constexpr std::uint8_t kOfficialSelectContextActivate = 44;
constexpr std::uint8_t kOfficialSelectContextFirstEffect = 45;
constexpr std::uint8_t kOfficialTriggerDamagedEnemyAttack = 10;
constexpr std::uint8_t kOfficialTriggerDamagedEnemyAttackActive = 11;

PTCG_OFFICIAL_RESOLVER_HD inline OfficialTriggerResolverResult
official_advance_trigger_resolver(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

PTCG_OFFICIAL_RESOLVER_HD inline std::int32_t official_player_continual_i16(
    const OfficialPlayerStatePod& player,
    std::uint32_t shift) {
    return static_cast<std::int16_t>((player.continual_state >> shift) & 0xffffU);
}

PTCG_OFFICIAL_RESOLVER_HD inline std::int32_t official_player_continual_i8(
    const OfficialPlayerStatePod& player,
    std::uint32_t shift) {
    return static_cast<std::int8_t>((player.continual_state >> shift) & 0xffU);
}

PTCG_OFFICIAL_RESOLVER_HD inline bool official_apply_special_condition_checkup(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    const std::int32_t basic_order[2] = {
        state->first_player, 1 - state->first_player};
    for (int index = 0; index < 2; ++index) {
        const int player = basic_order[index];
        if (player < 0 || player > 1
            || state->players[player].active.count == 0) continue;
        OfficialPlayerStatePod& ps = state->players[player];
        const std::int32_t poison = official_pod_poison_counter(ps);
        if (poison <= 0) continue;
        const OfficialCardRefPod ref = ps.active.values[0];
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
            rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) {
            official_pod_fail(
                state,
                OfficialPodError::kRulePackBounds,
                card == nullptr ? 0 : card->card_id);
            return false;
        }
        std::int32_t damage = poison * 10
            + official_player_continual_i16(ps, 0);
        if (!official_energy_type_matches(
                official_effective_pokemon_type(*card, *master), 1 << 6)) {
            damage += official_player_continual_i8(ps, 32);
        }
        if (damage > 0) official_pod_add_damage(state, rules, ref, damage);
    }

    for (int index = 0; index < 2; ++index) {
        const int player = basic_order[index];
        if (player < 0 || player > 1
            || state->players[player].active.count == 0) continue;
        OfficialPlayerStatePod& ps = state->players[player];
        if (!official_pod_burned(ps)) continue;
        const std::int32_t damage = 20 + official_player_continual_i16(ps, 16);
        if (damage > 0) {
            official_pod_add_damage(state, rules, ps.active.values[0], damage);
        }
        state->coin_head_count = 0;
        if (official_pod_coin(state, player)) {
            const OfficialCardRefPod active_ref = ps.active.values[0];
            const OfficialCardStatePod* active_card = official_pod_card(
                state, active_ref);
            if (active_card != nullptr) {
                official_semantic_history_append(
                    state,
                    OfficialSemanticLogType::kBurned,
                    player,
                    1,
                    active_card->card_id,
                    active_ref.index);
            }
            official_pod_set_burned(&ps, false);
            official_pod_mark_changed(state);
        }
    }

    for (int index = 0; index < 2; ++index) {
        const int player = basic_order[index];
        if (player < 0 || player > 1
            || state->players[player].active.count == 0) continue;
        OfficialPlayerStatePod& ps = state->players[player];
        if (official_pod_bad_status(ps) != OfficialBadStatus::kAsleep) continue;
        state->coin_head_count = 0;
        if (official_pod_coin(state, player)) {
            const OfficialCardRefPod active_ref = ps.active.values[0];
            const OfficialCardStatePod* active_card = official_pod_card(
                state, active_ref);
            if (active_card != nullptr) {
                official_semantic_history_append(
                    state,
                    OfficialSemanticLogType::kAsleep,
                    player,
                    1,
                    active_card->card_id,
                    active_ref.index);
            }
            official_pod_set_bad_status(&ps, OfficialBadStatus::kNone);
            official_pod_mark_changed(state);
        }
    }

    const std::int32_t active = official_active_player(*state);
    if (active >= 0 && active <= 1 && state->players[active].active.count > 0
        && official_pod_bad_status(state->players[active])
            == OfficialBadStatus::kParalyzed) {
        const OfficialCardRefPod active_ref = state->players[active].active.values[0];
        const OfficialCardStatePod* active_card = official_pod_card(state, active_ref);
        if (active_card != nullptr) {
            official_semantic_history_append(
                state,
                OfficialSemanticLogType::kParalyzed,
                active,
                1,
                active_card->card_id,
                active_ref.index);
        }
        official_pod_set_bad_status(
            &state->players[active], OfficialBadStatus::kNone);
        official_pod_mark_changed(state);
    }
    return official_pod_ok(state);
}

PTCG_OFFICIAL_RESOLVER_HD inline void official_clear_resolved_ability(
    OfficialStatePod* state) {
    official_clear_effect_selection(state);
    state->effect_state = OfficialEffectStatePod{};
    state->trigger_info = OfficialTriggerInfoPod{};
    state->effect_interpreter = OfficialEffectInterpreterPod{};
    state->context_card = OfficialCardRefPod{};
    state->targets.count = 0;
    state->pre_targets.count = 0;
    state->selected_list.count = 0;
    state->each_list.count = 0;
    state->check_list.count = 0;
    state->removed_damage_counter = 0;
    state->effect_jump = 0;
    state->attach_active = 0;
}

PTCG_OFFICIAL_RESOLVER_HD inline bool official_resolver_satisfy_skill_conditions(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialSkillRule& skill,
    std::uint16_t start_index,
    OfficialAreaRefPod effect_card,
    std::int32_t effect_owner,
    bool* satisfied) {
    *satisfied = false;
    const std::int32_t effect_offset = skill.values[kSkillEffectOffset];
    const std::int32_t effect_count = skill.values[kSkillEffectCount];
    const std::uint32_t total_effects = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kEffects)];
    if (effect_offset < 0 || effect_count < 0 || start_index > effect_count
        || static_cast<std::uint64_t>(effect_offset) + effect_count > total_effects) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, effect_offset);
        return false;
    }
    for (std::uint16_t index = start_index; index < effect_count; ++index) {
        const OfficialEffectRule& effect = rules.effects[effect_offset + index];
        if ((effect.flags & kOfficialEffectIsCondition) == 0) break;
        const OfficialConditionResult result = official_satisfy_condition(
            state,
            rules,
            rules.effects + effect_offset,
            effect_count,
            index,
            effect_card,
            effect_owner);
        if (result == OfficialConditionResult::kTrue) continue;
        if (result != OfficialConditionResult::kFalse) return false;
        if (effect.values[kEffectFailSkip] > 0) break;
        return true;
    }
    if ((skill.flags & kOfficialSkillOnceTurnFlag) != 0) {
        const OfficialCardStatePod* card = official_pod_card(state, effect_card.card);
        if (card == nullptr) return false;
        for (std::uint16_t index = 0; index < card->ability_used_count; ++index) {
            if (card->ability_used[index] == card->card_id) return true;
        }
    }
    *satisfied = true;
    return true;
}

PTCG_OFFICIAL_RESOLVER_HD inline bool official_record_resolved_skill(
    OfficialStatePod* state,
    const OfficialSkillRule& skill,
    OfficialAreaRefPod effect_card) {
    OfficialCardStatePod* card = official_pod_card(state, effect_card.card);
    if (card == nullptr) return false;
    if ((skill.flags & kOfficialSkillOnceTurnFlag) != 0) {
        for (std::uint16_t index = 0; index < card->ability_used_count; ++index) {
            if (card->ability_used[index] == card->card_id) return true;
        }
        if (card->ability_used_count == 8) {
            for (std::uint16_t index = 1; index < 8; ++index) {
                card->ability_used[index - 1] = card->ability_used[index];
            }
            card->ability_used_count = 7;
        }
        card->ability_used[card->ability_used_count++] = static_cast<std::int16_t>(
            card->card_id);
    }
    return official_pod_push(
        state,
        &state->turn_used_skills,
        skill.values[kSkillId],
        OfficialPodError::kTurnRecordOverflow);
}

PTCG_OFFICIAL_RESOLVER_HD inline OfficialTriggerResolverResult
official_finish_resolved_trigger(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    official_clear_resolved_ability(state);
    state->trigger_resolver.current_valid = 0;
    state->trigger_resolver.awaiting_activation = 0;
    state->trigger_resolver.activation_kind = 0;
    if (state->trigger_resolver.depth != 0) {
        const OfficialContinualRefreshResult refreshed =
            official_refresh_continual_effects(state, rules);
        if (refreshed != OfficialContinualRefreshResult::kApplied) {
            if (official_pod_ok(state)) {
                official_pod_fail(
                    state,
                    OfficialPodError::kInterpreterBudget,
                    static_cast<std::int32_t>(refreshed));
            }
            return OfficialTriggerResolverResult::kError;
        }
    }
    return OfficialTriggerResolverResult::kComplete;
}

PTCG_OFFICIAL_RESOLVER_HD inline bool official_add_trigger_yes_no_options(
    OfficialStatePod* state,
    std::uint8_t context,
    std::int32_t player,
    OfficialTriggerActivationKind kind) {
    official_clear_effect_selection(state);
    state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kYesNo);
    state->select_context = context;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = 1;
    state->select_max = 1;
    OfficialSelectOptionPod yes{};
    yes.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kYes);
    OfficialSelectOptionPod no{};
    no.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kNo);
    if (!official_pod_push(state, &state->options, yes, OfficialPodError::kOptionOverflow)
        || !official_pod_push(state, &state->options, no, OfficialPodError::kOptionOverflow)) {
        return false;
    }
    state->context_card = state->trigger_resolver.current.activate.effect_card.card;
    state->trigger_resolver.awaiting_activation = 1;
    state->trigger_resolver.activation_kind = static_cast<std::uint8_t>(kind);
    return true;
}

PTCG_OFFICIAL_RESOLVER_HD inline OfficialTriggerResolverResult
official_start_resolved_skill_effects(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialSkillRule& skill,
    std::uint16_t start_index) {
    if (!official_record_resolved_skill(
            state, skill, state->trigger_resolver.current.activate.effect_card)) {
        return OfficialTriggerResolverResult::kError;
    }
    state->effect_state.ability = state->trigger_resolver.current.activate;
    const OfficialEffectInterpreterResult result = official_begin_skill_effects_at(
        state,
        rules,
        skill.values[kSkillId],
        state->trigger_resolver.current.activate.effect_card,
        state->trigger_resolver.current.activate.use_player,
        start_index);
    if (result == OfficialEffectInterpreterResult::kError) {
        return OfficialTriggerResolverResult::kError;
    }
    if (result == OfficialEffectInterpreterResult::kNeedsAction) {
        return OfficialTriggerResolverResult::kNeedsAction;
    }
    return OfficialTriggerResolverResult::kComplete;
}

PTCG_OFFICIAL_RESOLVER_HD inline OfficialTriggerResolverResult
official_prepare_resolved_activation(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialSkillRule& skill) {
    // Official ActivateAbility clears targetList and changed before it decides
    // whether to yield an optional/which-effect selection or start effects.
    state->targets.count = 0;
    state->control_flags &= static_cast<std::uint8_t>(~kOfficialChangedFlag);
    const std::int32_t owner = state->trigger_resolver.current.activate.use_player;
    const std::uint16_t trigger_start = static_cast<std::uint16_t>(
        skill.values[kSkillTriggerStart]);
    if ((skill.flags & kOfficialSkillCanSelectActivateFlag) != 0) {
        return official_add_trigger_yes_no_options(
                   state,
                   kOfficialSelectContextActivate,
                   owner,
                   OfficialTriggerActivationKind::kOptional)
            ? OfficialTriggerResolverResult::kNeedsAction
            : OfficialTriggerResolverResult::kError;
    }
    const std::int32_t second = skill.values[kSkillSecondEffectStart];
    if (second > 0) {
        bool first_satisfied = false;
        if (!official_resolver_satisfy_skill_conditions(
                state,
                rules,
                skill,
                0,
                state->trigger_resolver.current.activate.effect_card,
                owner,
                &first_satisfied)) {
            return OfficialTriggerResolverResult::kError;
        }
        if (!first_satisfied) {
            return official_start_resolved_skill_effects(
                state, rules, skill, static_cast<std::uint16_t>(second));
        }
        return official_add_trigger_yes_no_options(
                   state,
                   kOfficialSelectContextFirstEffect,
                   owner,
                   OfficialTriggerActivationKind::kChooseEffect)
            ? OfficialTriggerResolverResult::kNeedsAction
            : OfficialTriggerResolverResult::kError;
    }
    const std::int32_t enemy_second = skill.values[kSkillSecondEffectStartEnemy];
    if (enemy_second > 0) {
        return official_add_trigger_yes_no_options(
                   state,
                   kOfficialSelectContextActivate,
                   1 - owner,
                   OfficialTriggerActivationKind::kEnemyChooseEffect)
            ? OfficialTriggerResolverResult::kNeedsAction
            : OfficialTriggerResolverResult::kError;
    }
    return official_start_resolved_skill_effects(state, rules, skill, trigger_start);
}

PTCG_OFFICIAL_RESOLVER_HD inline OfficialTriggerResolverResult
official_group_temporary_triggers(
    OfficialStatePod* state,
    const OfficialRulePackView&) {
    OfficialTriggerResolverPod& resolver = state->trigger_resolver;
    std::uint16_t first = 0;
    bool found = false;
    for (std::uint16_t index = 0; index < state->temporary_triggers.count; ++index) {
        if (state->temporary_triggers.values[index].trigger.depth == resolver.depth) {
            first = index;
            found = true;
            break;
        }
    }
    if (!found) first = 0;
    const std::uint16_t count = static_cast<std::uint16_t>(
        state->temporary_triggers.count - first);
    resolver.temporary_first = first;
    resolver.temporary_count = count;
    resolver.grouped = 1;
    if (count == 0) return OfficialTriggerResolverResult::kComplete;
    if (count == 1) {
        if (!official_pod_push(
                state,
                &state->triggers,
                state->temporary_triggers.values[first],
                OfficialPodError::kTriggerStackOverflow)) {
            return OfficialTriggerResolverResult::kError;
        }
        state->temporary_triggers.count = first;
        return OfficialTriggerResolverResult::kComplete;
    }

    const std::uint8_t last_type = state->temporary_triggers.values[
        state->temporary_triggers.count - 1].trigger.type;
    std::int32_t player = official_active_player(*state);
    if (state->phase != static_cast<std::uint8_t>(OfficialGamePhase::kMain)
        || last_type == kOfficialTriggerDamagedEnemyAttack
        || last_type == kOfficialTriggerDamagedEnemyAttackActive) {
        player = 1 - player;
    }
    // ResolveTriggerStack has no active Ability while it asks the player to
    // order a same-depth group. The next selected trigger rebuilds these
    // fields before condition/effect evaluation.
    state->effect_state = OfficialEffectStatePod{};
    state->trigger_info = OfficialTriggerInfoPod{};
    state->effect_interpreter = OfficialEffectInterpreterPod{};
    official_clear_effect_selection(state);
    state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kSkill);
    state->select_context = kOfficialSelectContextSkillOrder;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = static_cast<std::int16_t>(count);
    state->select_max = static_cast<std::int16_t>(count);
    for (std::uint16_t index = 0; index < count; ++index) {
        const OfficialTriggeredAbilityPod& ability =
            state->temporary_triggers.values[first + index];
        OfficialSelectOptionPod option{};
        option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kSkill);
        // Special-condition triggers intentionally carry a null effect card.
        // The official AddOptionSkillOrder emits a zero-valued Skill option
        // for them instead of dereferencing the card.
        const OfficialCardStatePod* card = official_pod_card(
            static_cast<const OfficialStatePod*>(state),
            ability.activate.effect_card.card);
        if (card != nullptr) {
            option.params[0] = static_cast<std::int16_t>(card->card_id);
            option.params[1] = static_cast<std::int16_t>(
                ability.activate.effect_card.card.index);
            option.resolved_card = ability.activate.effect_card.card.index;
            option.option_equiv = static_cast<std::uint16_t>(card->card_id);
        }
        if (!official_pod_push(
                state, &state->options, option, OfficialPodError::kOptionOverflow)) {
            return OfficialTriggerResolverResult::kError;
        }
    }
    resolver.awaiting_order = 1;
    return OfficialTriggerResolverResult::kNeedsAction;
}

PTCG_OFFICIAL_RESOLVER_HD inline OfficialTriggerResolverResult
official_advance_trigger_resolver(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    OfficialTriggerResolverPod& resolver = state->trigger_resolver;
    if (!official_pod_ok(state)) return OfficialTriggerResolverResult::kError;
    if (resolver.active == 0) return OfficialTriggerResolverResult::kComplete;
    if (resolver.awaiting_order != 0 || resolver.awaiting_activation != 0
        || state->effect_interpreter.awaiting_selection != 0) {
        return OfficialTriggerResolverResult::kNeedsAction;
    }
    if (state->effect_interpreter.active != 0) {
        const OfficialEffectInterpreterResult effect =
            official_advance_effect_interpreter(state, rules);
        if (effect == OfficialEffectInterpreterResult::kError) {
            if (official_pod_ok(state)) {
                official_pod_fail(
                    state,
                    OfficialPodError::kInvalidAction,
                    -441);
            }
            return OfficialTriggerResolverResult::kError;
        }
        if (effect == OfficialEffectInterpreterResult::kNeedsAction) {
            return OfficialTriggerResolverResult::kNeedsAction;
        }
        const OfficialTriggerResolverResult finished =
            official_finish_resolved_trigger(state, rules);
        if (finished == OfficialTriggerResolverResult::kError) return finished;
        return official_advance_trigger_resolver(state, rules);
    }
    if (resolver.grouped == 0) {
        const OfficialTriggerResolverResult grouped =
            official_group_temporary_triggers(state, rules);
        if (grouped != OfficialTriggerResolverResult::kComplete) return grouped;
    }

    while (state->triggers.count > 0) {
        const OfficialTriggeredAbilityPod next =
            state->triggers.values[state->triggers.count - 1];
        // AfterMoveTriggerStack marks the surrounding Refresh as changed even
        // when a shallower trigger makes this resolver return without popping.
        state->control_flags |= 1U << 4U;
        if (next.trigger.depth < resolver.depth) break;
        --state->triggers.count;
        resolver.current = next;
        resolver.current_valid = 1;
        if (next.activate.is_special_condition != 0) {
            if (!official_apply_special_condition_checkup(state, rules)) {
                return OfficialTriggerResolverResult::kError;
            }
            const OfficialTriggerResolverResult finished =
                official_finish_resolved_trigger(state, rules);
            if (finished == OfficialTriggerResolverResult::kError) return finished;
            continue;
        }

        const bool source_valid = official_pod_area_ref_valid(
            state, next.activate.effect_card);
        const OfficialCardStatePod* card = source_valid
            ? official_pod_card(state, next.activate.effect_card.card)
            : nullptr;
        const OfficialSkillRule* skill = official_skill_rule(
            rules, static_cast<std::uint32_t>(next.activate.skill_id));
        if (skill == nullptr) {
            official_pod_fail(
                state, OfficialPodError::kRulePackBounds, next.activate.skill_id);
            return OfficialTriggerResolverResult::kError;
        }
        bool allowed_area = card != nullptr;
        if (allowed_area) {
            const OfficialArea area = static_cast<OfficialArea>(card->area);
            if (area == OfficialArea::kTrash) {
                allowed_area = (skill->flags & kOfficialSkillCanActivateTrashFlag) != 0;
            } else if (area == OfficialArea::kDeck || area == OfficialArea::kHand
                || area == OfficialArea::kPrize) {
                allowed_area = false;
            }
        }
        if (!allowed_area || (card != nullptr && official_continual_flag(*card, 0))) {
            const OfficialTriggerResolverResult skipped =
                official_finish_resolved_trigger(state, rules);
            if (skipped == OfficialTriggerResolverResult::kError) return skipped;
            continue;
        }

        state->effect_state = OfficialEffectStatePod{};
        state->effect_state.ability = next.activate;
        state->effect_state.effect_rate = 1;
        state->trigger_info = next.trigger;
        bool first_satisfied = false;
        if (!official_resolver_satisfy_skill_conditions(
                state,
                rules,
                *skill,
                static_cast<std::uint16_t>(skill->values[kSkillTriggerStart]),
                next.activate.effect_card,
                next.activate.use_player,
                &first_satisfied)) {
            return OfficialTriggerResolverResult::kError;
        }
        if (!first_satisfied) {
            const OfficialTriggerResolverResult skipped =
                official_finish_resolved_trigger(state, rules);
            if (skipped == OfficialTriggerResolverResult::kError) return skipped;
            continue;
        }
        const OfficialTriggerResolverResult prepared =
            official_prepare_resolved_activation(state, rules, *skill);
        if (prepared != OfficialTriggerResolverResult::kComplete) return prepared;
        if (state->effect_interpreter.active != 0) {
            const OfficialEffectInterpreterResult effect =
                official_advance_effect_interpreter(state, rules);
            if (effect == OfficialEffectInterpreterResult::kError) {
                if (official_pod_ok(state)) {
                    official_pod_fail(
                        state,
                        OfficialPodError::kInvalidAction,
                        -535);
                }
                return OfficialTriggerResolverResult::kError;
            }
            if (effect == OfficialEffectInterpreterResult::kNeedsAction) {
                return OfficialTriggerResolverResult::kNeedsAction;
            }
        }
        const OfficialTriggerResolverResult finished =
            official_finish_resolved_trigger(state, rules);
        if (finished == OfficialTriggerResolverResult::kError) return finished;
    }
    resolver.active = 0;
    resolver.current_valid = 0;
    return OfficialTriggerResolverResult::kComplete;
}

PTCG_OFFICIAL_RESOLVER_HD inline OfficialTriggerResolverResult
official_begin_trigger_resolution(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int8_t depth) {
    if (!official_pod_ok(state) || state->trigger_resolver.active != 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, depth);
        }
        return OfficialTriggerResolverResult::kError;
    }
    state->trigger_resolver = OfficialTriggerResolverPod{};
    state->trigger_resolver.active = 1;
    state->trigger_resolver.depth = depth;
    return official_advance_trigger_resolver(state, rules);
}

PTCG_OFFICIAL_RESOLVER_HD inline OfficialTriggerResolverResult
official_resume_trigger_resolution(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    OfficialTriggerResolverPod& resolver = state->trigger_resolver;
    if (!official_pod_ok(state) || resolver.active == 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        }
        return OfficialTriggerResolverResult::kError;
    }
    if (state->effect_interpreter.awaiting_selection != 0) {
        const OfficialEffectInterpreterResult resumed = official_apply_effect_action(
            state, rules, option_indices, count);
        if (resumed == OfficialEffectInterpreterResult::kError) {
            return OfficialTriggerResolverResult::kError;
        }
        if (resumed == OfficialEffectInterpreterResult::kNeedsAction) {
            return OfficialTriggerResolverResult::kNeedsAction;
        }
        const OfficialTriggerResolverResult finished =
            official_finish_resolved_trigger(state, rules);
        if (finished == OfficialTriggerResolverResult::kError) return finished;
        return official_advance_trigger_resolver(state, rules);
    }
    if (resolver.awaiting_order != 0) {
        if (count != resolver.temporary_count
            || count != state->options.count
            || static_cast<std::size_t>(state->triggers.count) + count
                > kOfficialTriggerCapacity) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
            return OfficialTriggerResolverResult::kError;
        }
        for (std::uint16_t index = 0; index < count; ++index) {
            if (option_indices[index] >= count) {
                official_pod_fail(
                    state, OfficialPodError::kInvalidAction, option_indices[index]);
                return OfficialTriggerResolverResult::kError;
            }
            for (std::uint16_t prior = 0; prior < index; ++prior) {
                if (option_indices[prior] == option_indices[index]) {
                    official_pod_fail(
                        state, OfficialPodError::kInvalidAction, option_indices[index]);
                    return OfficialTriggerResolverResult::kError;
                }
            }
        }
        for (std::uint16_t reverse = count; reverse > 0; --reverse) {
            const std::uint16_t selected = option_indices[reverse - 1];
            if (!official_pod_push(
                    state,
                    &state->triggers,
                    state->temporary_triggers.values[
                        resolver.temporary_first + selected],
                    OfficialPodError::kTriggerStackOverflow)) {
                return OfficialTriggerResolverResult::kError;
            }
        }
        state->temporary_triggers.count = resolver.temporary_first;
        resolver.awaiting_order = 0;
        official_clear_effect_selection(state);
        // ResolveTriggerStack is entered after the preceding ability has
        // completed.  The official State has already discarded its effect
        // target scratch at that boundary; do the same before processing the
        // newly ordered trigger stack.
        state->targets.count = 0;
        state->pre_targets.count = 0;
        state->selected_list.count = 0;
        state->each_list.count = 0;
        return official_advance_trigger_resolver(state, rules);
    }
    if (resolver.awaiting_activation != 0) {
        if (count != 1 || option_indices[0] >= state->options.count) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
            return OfficialTriggerResolverResult::kError;
        }
        const OfficialSelectOptionPod chosen = state->options.values[option_indices[0]];
        const bool yes = chosen.type
            == static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kYes);
        const bool no = chosen.type
            == static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kNo);
        if (!yes && !no) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, chosen.type);
            return OfficialTriggerResolverResult::kError;
        }
        const OfficialTriggerActivationKind kind =
            static_cast<OfficialTriggerActivationKind>(resolver.activation_kind);
        resolver.awaiting_activation = 0;
        resolver.activation_kind = 0;
        official_clear_effect_selection(state);
        const OfficialSkillRule* skill = official_skill_rule(
            rules, static_cast<std::uint32_t>(resolver.current.activate.skill_id));
        if (skill == nullptr) {
            official_pod_fail(
                state,
                OfficialPodError::kRulePackBounds,
                resolver.current.activate.skill_id);
            return OfficialTriggerResolverResult::kError;
        }
        if (kind == OfficialTriggerActivationKind::kOptional && no) {
            const OfficialTriggerResolverResult finished =
                official_finish_resolved_trigger(state, rules);
            if (finished == OfficialTriggerResolverResult::kError) return finished;
            return official_advance_trigger_resolver(state, rules);
        }
        std::uint16_t start = static_cast<std::uint16_t>(
            skill->values[kSkillTriggerStart]);
        if (kind == OfficialTriggerActivationKind::kChooseEffect && no) {
            start = static_cast<std::uint16_t>(skill->values[kSkillSecondEffectStart]);
        } else if (kind == OfficialTriggerActivationKind::kEnemyChooseEffect && no) {
            start = static_cast<std::uint16_t>(
                skill->values[kSkillSecondEffectStartEnemy]);
        }
        const OfficialTriggerResolverResult started =
            official_start_resolved_skill_effects(state, rules, *skill, start);
        if (started != OfficialTriggerResolverResult::kComplete) return started;
        const OfficialTriggerResolverResult finished =
            official_finish_resolved_trigger(state, rules);
        if (finished == OfficialTriggerResolverResult::kError) return finished;
        return official_advance_trigger_resolver(state, rules);
    }
    official_pod_fail(state, OfficialPodError::kInvalidAction, count);
    return OfficialTriggerResolverResult::kError;
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_RESOLVER_HD
