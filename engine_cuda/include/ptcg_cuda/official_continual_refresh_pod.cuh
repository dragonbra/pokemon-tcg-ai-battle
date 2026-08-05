#pragma once

#include <cstdint>

#include "ptcg_cuda/official_conditions_pod.cuh"
#include "ptcg_cuda/official_continual_effects_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_REFRESH_HD __host__ __device__
#else
#define PTCG_OFFICIAL_REFRESH_HD
#endif

namespace ptcg::cuda_engine {

constexpr std::size_t kOfficialContinualSourceCapacity = kOfficialCardCapacity;
constexpr std::int32_t kOfficialSkillOrderUnset = 0x7fffffff;
constexpr std::uint64_t kOfficialContinualMainAbilityFlag = 1ULL << 0;
constexpr std::uint64_t kOfficialSkillNotStackFlag = 1ULL << 3;
constexpr std::uint64_t kOfficialContinualRuntimeMask =
    kCardNoSpecialCondition
    | kCardNoSleepParalyzeConfuse
    | kCardNoSleep
    | kCardKoByDamageToHand;

enum class OfficialContinualRefreshResult : std::int32_t {
    kApplied = 0,
    kDepthLimit = 1,
    kError = 2,
};

struct OfficialContinualSourcePod {
    OfficialCardRefPod ref{};
    std::uint16_t reserved = 0;
    std::int32_t skill_id = 0;
    std::int32_t priority = 0;
    std::int32_t skill_order = 0;
    std::int32_t move_counter = 0;
};
static_assert(sizeof(OfficialContinualSourcePod) == 20);

template <std::size_t Capacity>
PTCG_OFFICIAL_REFRESH_HD inline void official_clear_zone_continual(
    OfficialStatePod* state,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone) {
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        OfficialCardStatePod* card = official_pod_card(state, zone.values[index]);
        if (card == nullptr) continue;
        for (int word = 0; word < 5; ++word) card->continual_state[word] = 0;
        card->runtime_flags &= ~kOfficialContinualRuntimeMask;
        card->hp_change = 0;
    }
}

PTCG_OFFICIAL_REFRESH_HD inline void official_clear_continual_state(
    OfficialStatePod* state) {
    state->continual_state = 0;
    for (int player = 0; player < 2; ++player) {
        OfficialPlayerStatePod& ps = state->players[player];
        ps.continual_state = 0;
        official_clear_zone_continual(state, ps.active);
        official_clear_zone_continual(state, ps.bench);
        official_clear_zone_continual(state, ps.hand);
        official_clear_zone_continual(state, ps.tool);
        official_clear_zone_continual(state, ps.energy);
    }
    official_clear_zone_continual(state, state->stadium);
}

PTCG_OFFICIAL_REFRESH_HD inline bool official_skill_area_matches(
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

PTCG_OFFICIAL_REFRESH_HD inline bool official_skill_has_continual_effect(
    const OfficialSkillRule& skill) {
    // Mirrors Skill::hasContinual(). The official engine gathers every
    // non-main ability with no triggers, or with a continual prefix before
    // triggerStartIndex. Filtering by effect type changes refresh ordering.
    return (skill.values[kSkillTriggerCount] == 0
            || skill.values[kSkillTriggerStart] > 0)
        && (skill.flags & kOfficialContinualMainAbilityFlag) == 0;
}

PTCG_OFFICIAL_REFRESH_HD inline bool official_collect_continual_source(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    OfficialContinualSourcePod* sources,
    std::uint16_t* count) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    if (master == nullptr || master->values[kCardAbilityId] == 0) return true;
    const OfficialSkillRule* skill = official_skill_rule(
        rules, static_cast<std::uint32_t>(master->values[kCardAbilityId]));
    if (skill == nullptr
        || !official_skill_area_matches(*skill, static_cast<OfficialArea>(card->area))
        || !official_skill_has_continual_effect(*skill)) {
        return official_pod_ok(state);
    }
    if (*count >= kOfficialContinualSourceCapacity) {
        official_pod_fail(state, OfficialPodError::kEffectStackOverflow, *count);
        return false;
    }
    sources[(*count)++] = OfficialContinualSourcePod{
        ref,
        0,
        skill->values[kSkillId],
        skill->values[kSkillPriority],
        card->skill_order,
        card->move_counter,
    };
    return true;
}

template <std::size_t Capacity>
PTCG_OFFICIAL_REFRESH_HD inline bool official_collect_continual_zone(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone,
    OfficialContinualSourcePod* sources,
    std::uint16_t* count) {
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        if (!official_collect_continual_source(
                state, rules, zone.values[index], sources, count)) return false;
    }
    return true;
}

PTCG_OFFICIAL_REFRESH_HD inline bool official_continual_source_before(
    const OfficialContinualSourcePod& left,
    const OfficialContinualSourcePod& right) {
    if (left.priority != right.priority) return left.priority > right.priority;
    if (left.skill_order != right.skill_order) return left.skill_order < right.skill_order;
    return left.move_counter < right.move_counter;
}

PTCG_OFFICIAL_REFRESH_HD inline void official_sort_continual_sources(
    OfficialContinualSourcePod* sources,
    std::uint16_t count) {
    for (std::uint16_t index = 1; index < count; ++index) {
        const OfficialContinualSourcePod value = sources[index];
        std::uint16_t position = index;
        while (position > 0
            && official_continual_source_before(value, sources[position - 1])) {
            sources[position] = sources[position - 1];
            --position;
        }
        sources[position] = value;
    }
}

PTCG_OFFICIAL_REFRESH_HD inline void official_remove_no_enemy_ability_targets(
    OfficialStatePod* state,
    OfficialCardRefPod source_ref) {
    const OfficialCardStatePod* source = official_pod_card(state, source_ref);
    if (source == nullptr) return;
    for (std::uint16_t index = 0; index < state->targets.count;) {
        const OfficialCardStatePod* target = official_pod_card(
            state, state->targets.values[index].card);
        const bool blocked = target != nullptr
            && target->player != source->player
            && source->area != static_cast<std::uint8_t>(OfficialArea::kStadium)
            && official_continual_flag(*target, 12);
        if (!blocked) {
            ++index;
            continue;
        }
        for (std::uint16_t move = index + 1; move < state->targets.count; ++move) {
            state->targets.values[move - 1] = state->targets.values[move];
        }
        --state->targets.count;
    }
}

PTCG_OFFICIAL_REFRESH_HD inline bool official_skill_already_stacked(
    const std::int32_t used[2][kOfficialContinualSourceCapacity],
    const std::uint16_t used_count[2],
    std::int32_t player,
    std::int32_t skill_id) {
    for (std::uint16_t index = 0; index < used_count[player]; ++index) {
        if (used[player][index] == skill_id) return true;
    }
    return false;
}

PTCG_OFFICIAL_REFRESH_HD inline bool official_apply_static_continual_skill(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialContinualSourcePod* sources,
    std::uint16_t source_count,
    std::uint16_t source_index,
    std::int32_t used[2][kOfficialContinualSourceCapacity],
    std::uint16_t used_count[2],
    bool* update_order) {
    const OfficialContinualSourcePod& source = sources[source_index];
    OfficialCardStatePod* card = official_pod_card(state, source.ref);
    const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    const OfficialSkillRule* skill = official_skill_rule(
        rules, static_cast<std::uint32_t>(source.skill_id));
    if (card == nullptr || master == nullptr || skill == nullptr) return false;
    const std::int32_t saved_order = card->skill_order;
    // Official StaticEffect marks every collected source as unordered before
    // checking an ability disabled by an earlier source in this same pass,
    // disabled Tool effects, or an already-stacked notStack ability.
    // Those early returns deliberately leave INT_MAX serialized on the card.
    card->skill_order = kOfficialSkillOrderUnset;
    if (official_continual_flag(*card, 0)) return true;
    if ((state->continual_state & 1U) != 0
        && master->values[kCardType] == 2) return true;
    if ((skill->flags & kOfficialSkillNotStackFlag) != 0
        && official_skill_already_stacked(
            used, used_count, card->player, source.skill_id)) return true;

    const std::int32_t offset = skill->values[kSkillEffectOffset];
    const std::int32_t count = skill->values[kSkillEffectCount];
    bool registered = false;
    state->effect_state.ability.skill_id = source.skill_id;
    state->effect_state.ability.effect_card = official_pod_area_ref(state, source.ref);
    state->effect_state.ability.use_player = card->player;
    // Official StaticEffect clears game->targetList once per Ability. Effects
    // within that Ability may then deliberately chain through Effected.
    state->targets.count = 0;
    for (std::int32_t effect_index = 0; effect_index < count; ++effect_index) {
        const OfficialEffectRule& effect = rules.effects[offset + effect_index];
        if ((effect.flags & kOfficialEffectIsCondition) != 0) {
            const OfficialConditionResult condition = official_satisfy_condition(
                state,
                rules,
                rules.effects + offset,
                count,
                effect_index,
                official_pod_area_ref(state, source.ref),
                card->player);
            if (condition == OfficialConditionResult::kTrue) continue;
            if (condition != OfficialConditionResult::kFalse) return false;
            if (effect.values[kEffectFailSkip] > 0) {
                effect_index += effect.values[kEffectFailSkip];
                continue;
            }
            return true;
        }
#if defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
        official_mark_effect_reached(
            state, static_cast<std::uint32_t>(offset + effect_index));
        official_mark_effect_applied(
            state, static_cast<std::uint32_t>(offset + effect_index));
#endif
        if (saved_order == kOfficialSkillOrderUnset) {
            card->skill_order = state->current_skill_order++;
        } else {
            card->skill_order = saved_order;
        }
        if (!registered && (skill->flags & kOfficialSkillNotStackFlag) != 0) {
            if (used_count[card->player] >= kOfficialContinualSourceCapacity) {
                official_pod_fail(
                    state, OfficialPodError::kEffectStackOverflow, used_count[card->player]);
                return false;
            }
            used[card->player][used_count[card->player]++] = source.skill_id;
            registered = true;
        }
        const OfficialTargetRule* target = official_effect_target_rule(state, rules, effect);
        if (target == nullptr
            || !official_build_target_list(
                state,
                rules,
                *target,
                &state->targets,
                official_pod_area_ref(state, source.ref),
                card->player)) return false;
        official_remove_no_enemy_ability_targets(state, source.ref);
        const OfficialEffectApplyResult result = official_apply_continual_effect_primitive(
            state, rules, effect, card->player, source.ref);
        if (result != OfficialEffectApplyResult::kApplied) {
            if (result == OfficialEffectApplyResult::kNeedsSelection) {
                official_pod_fail(
                    state, OfficialPodError::kUnsupportedEffect, effect.values[kEffectType]);
            }
            return false;
        }
        if (effect.values[kEffectType]
            == static_cast<std::int32_t>(OfficialEffectTypeId::kNoAbility)) {
            for (std::uint16_t target_index = 0;
                 target_index < state->targets.count;
                 ++target_index) {
                const OfficialCardRefPod target_ref = state->targets.values[target_index].card;
                for (std::uint16_t prior = 0; prior < source_index; ++prior) {
                    if (sources[prior].ref != target_ref) continue;
                    OfficialCardStatePod* target_card = official_pod_card(state, target_ref);
                    if (target_card != nullptr) {
                        target_card->skill_order = card->skill_order + 1;
                        *update_order = true;
                    }
                    break;
                }
            }
        }
        if (*update_order) break;
    }
    (void)source_count;
    return official_pod_ok(state);
}

PTCG_OFFICIAL_REFRESH_HD inline void official_clear_blocked_special_conditions(
    OfficialStatePod* state) {
    for (int player = 0; player < 2; ++player) {
        if (state->players[player].active.count == 0) continue;
        const OfficialCardStatePod* active = official_pod_card(
            state, state->players[player].active.values[0]);
        if (active == nullptr) continue;
        if ((active->runtime_flags & kCardNoSpecialCondition) != 0) {
            official_pod_clear_special_conditions(state, player);
        } else if ((active->runtime_flags & kCardNoSleepParalyzeConfuse) != 0) {
            official_pod_set_bad_status(&state->players[player], OfficialBadStatus::kNone);
        } else if ((active->runtime_flags & kCardNoSleep) != 0
            && official_pod_bad_status(state->players[player]) == OfficialBadStatus::kAsleep) {
            official_pod_set_bad_status(&state->players[player], OfficialBadStatus::kNone);
        }
    }
}

PTCG_OFFICIAL_REFRESH_HD inline bool official_begin_continual_target_scratch(
    OfficialStatePod* state) {
    if (state->targets.count > kOfficialEffectRefScratchCapacity) {
        official_pod_fail(
            state,
            OfficialPodError::kEffectScratchOverflow,
            state->targets.count);
        return false;
    }
    state->effect_ref_scratch.count = state->targets.count;
    for (std::uint16_t index = 0; index < state->targets.count; ++index) {
        state->effect_ref_scratch.values[index] = state->targets.values[index];
    }
    return true;
}

PTCG_OFFICIAL_REFRESH_HD inline void official_end_continual_target_scratch(
    OfficialStatePod* state,
    const OfficialEffectStatePod& saved_effect_state) {
    state->targets.count = state->effect_ref_scratch.count;
    for (std::uint16_t index = 0;
         index < state->effect_ref_scratch.count;
         ++index) {
        state->targets.values[index] = state->effect_ref_scratch.values[index];
    }
    state->effect_ref_scratch.count = 0;
    // StaticEffect receives its source/owner as explicit arguments and does
    // not replace the currently resolving instant Ability.  POD continual
    // primitives temporarily use effect_state for shared condition helpers,
    // so keep that mutation local to RefreshEffect.
    state->effect_state = saved_effect_state;
}

PTCG_OFFICIAL_REFRESH_HD inline OfficialContinualRefreshResult
official_refresh_continual_effects(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_begin_continual_target_scratch(state)) {
        return OfficialContinualRefreshResult::kError;
    }
    const OfficialEffectStatePod saved_effect_state = state->effect_state;
    state->continual_refresh_passes = 0;
    for (std::int32_t depth = 0; depth <= 10; ++depth) {
        state->continual_refresh_passes = static_cast<std::uint8_t>(depth + 1);
        official_clear_continual_state(state);
        OfficialContinualSourcePod sources[kOfficialContinualSourceCapacity]{};
        std::uint16_t source_count = 0;
        for (int player = 0; player < 2; ++player) {
            const OfficialPlayerStatePod& ps = state->players[player];
            if (!official_collect_continual_zone(
                    state, rules, ps.active, sources, &source_count)
                || !official_collect_continual_zone(
                    state, rules, ps.bench, sources, &source_count)
                || !official_collect_continual_zone(
                    state, rules, ps.energy, sources, &source_count)
                || !official_collect_continual_zone(
                    state, rules, ps.tool, sources, &source_count)
                || !official_collect_continual_zone(
                    state, rules, ps.hand, sources, &source_count)) {
                official_end_continual_target_scratch(state, saved_effect_state);
                return OfficialContinualRefreshResult::kError;
            }
        }
        if (!official_collect_continual_zone(
                state, rules, state->stadium, sources, &source_count)) {
            official_end_continual_target_scratch(state, saved_effect_state);
            return OfficialContinualRefreshResult::kError;
        }
        official_sort_continual_sources(sources, source_count);

        std::int32_t used[2][kOfficialContinualSourceCapacity]{};
        std::uint16_t used_count[2]{};
        bool update_order = false;
        for (std::uint16_t index = 0; index < source_count; ++index) {
            state->current_card_effect_index = index;
            if (!official_apply_static_continual_skill(
                    state,
                    rules,
                    sources,
                    source_count,
                    index,
                    used,
                    used_count,
                    &update_order)) {
                official_end_continual_target_scratch(state, saved_effect_state);
                return OfficialContinualRefreshResult::kError;
            }
            if (update_order) break;
        }
        if (!update_order) {
            official_clear_blocked_special_conditions(state);
            const OfficialContinualRefreshResult result = official_pod_ok(state)
                ? OfficialContinualRefreshResult::kApplied
                : OfficialContinualRefreshResult::kError;
            official_end_continual_target_scratch(state, saved_effect_state);
            return result;
        }
        if (depth == 10) {
            official_end_continual_target_scratch(state, saved_effect_state);
            return OfficialContinualRefreshResult::kDepthLimit;
        }
    }
    official_end_continual_target_scratch(state, saved_effect_state);
    return OfficialContinualRefreshResult::kDepthLimit;
}

constexpr std::uint8_t kOfficialSwitchTriggerActiveToBench = 4;
constexpr std::uint8_t kOfficialSwitchTriggerBenchToActive = 6;

// Mirrors the official SwitchPokemon helper rather than only its zone swap:
// swap, RefreshEffect, ActiveToBench trigger, then BenchToActive trigger.
PTCG_OFFICIAL_REFRESH_HD inline bool official_switch_pokemon(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player,
    std::uint16_t bench_index) {
    if (player < 0 || player > 1
        || bench_index >= state->players[player].bench.count) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, bench_index);
        return false;
    }
    const bool had_active = state->players[player].active.count != 0;
    const OfficialCardRefPod old_active = had_active
        ? state->players[player].active.values[0]
        : OfficialCardRefPod{};
    const OfficialCardRefPod new_active =
        state->players[player].bench.values[bench_index];
    if (!official_pod_switch_active(state, player, bench_index)) return false;

    const OfficialContinualRefreshResult refreshed =
        official_refresh_continual_effects(state, rules);
    if (refreshed != OfficialContinualRefreshResult::kApplied) {
        if (refreshed == OfficialContinualRefreshResult::kDepthLimit) {
            official_pod_fail(state, OfficialPodError::kInterpreterBudget, 10);
        } else if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, -535);
        }
        return false;
    }
    if (had_active
        && !official_pull_trigger(
            state,
            rules,
            kOfficialSwitchTriggerActiveToBench,
            old_active)) {
        return false;
    }
    return official_pull_trigger(
        state,
        rules,
        kOfficialSwitchTriggerBenchToActive,
        new_active);
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_REFRESH_HD
