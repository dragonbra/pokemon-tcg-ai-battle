#pragma once

#include <cstdint>

#include "ptcg_cuda/official_targets_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_CONDITION_HD __host__ __device__
#else
#define PTCG_OFFICIAL_CONDITION_HD
#endif

namespace ptcg::cuda_engine {

enum class OfficialConditionResult : std::int32_t {
    kFalse = 0,
    kTrue = 1,
    kUnsupported = 2,
    kError = 3,
};

constexpr std::uint64_t kOfficialEffectIsCondition = 1ULL << 0;
constexpr std::uint8_t kHistoryKo = 1U << 0;
constexpr std::uint8_t kHistoryKoTeamRocket = 1U << 1;
constexpr std::uint8_t kHistoryKoAttackDamage = 1U << 2;
constexpr std::uint8_t kHistoryKoAttackDamageEthan = 1U << 3;
constexpr std::uint8_t kHistoryKoAttackDamageHop = 1U << 4;

PTCG_OFFICIAL_CONDITION_HD inline const OfficialTargetRule* official_condition_target(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect) {
    const std::int32_t index = effect.values[kEffectTargetIndex];
    const std::uint32_t count = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kTargets)];
    if (index < 0 || static_cast<std::uint32_t>(index) >= count) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, index);
        return nullptr;
    }
    return &rules.targets[index];
}

PTCG_OFFICIAL_CONDITION_HD inline OfficialConditionResult official_condition_bool(bool value) {
    return value ? OfficialConditionResult::kTrue : OfficialConditionResult::kFalse;
}

PTCG_OFFICIAL_CONDITION_HD inline bool official_build_condition_target_list(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialTargetRule& target,
    OfficialPodList<OfficialAreaRefPod, kOfficialListCapacity>* matches,
    OfficialAreaRefPod effect_card,
    std::int32_t effect_owner) {
    // State::targetList survives between official effects.  Conditions build
    // into a local scratch list so they do not mutate that serialized state,
    // but Effected targets still need the previous effect's target list as
    // their input.
    *matches = state->targets;
    // The official condition scratch list stores CardRef identity, then
    // TargetList rebuilds every AreaRef from the card's current move counter.
    // This matters when the preceding effect moved a selected card (for
    // example Crispin's first Basic Energy from Deck to Hand): the serialized
    // target list deliberately keeps the old snapshot, but Effected must still
    // see that card in its new area.
    if (target.values[kTargetAreaCount] > 0
        && target.values[kTargetArea0]
            == static_cast<std::int32_t>(OfficialArea::kEffected)) {
        for (std::uint16_t index = 0; index < matches->count; ++index) {
            matches->values[index] = official_pod_area_ref(
                state, matches->values[index].card);
            if (!official_pod_ok(state)) return false;
        }
    }
    return official_build_target_list(
        state, rules, target, matches, effect_card, effect_owner);
}

PTCG_OFFICIAL_CONDITION_HD inline std::int32_t official_condition_energy_count(
    const OfficialStatePod& state,
    const OfficialRulePackView& rules,
    std::int32_t player,
    std::int32_t required_type = -1) {
    const OfficialPlayerStatePod& ps = state.players[player];
    std::int32_t count = 0;
    for (int zone = 0; zone < 2; ++zone) {
        const OfficialCardRefPod* values = zone == 0 ? ps.active.values : ps.bench.values;
        const std::uint16_t size = zone == 0 ? ps.active.count : ps.bench.count;
        for (std::uint16_t index = 0; index < size; ++index) {
            count += official_attached_energy_cards(
                state, values[index], required_type, false, rules);
        }
    }
    return count;
}

PTCG_OFFICIAL_CONDITION_HD inline std::int32_t official_active_player(
    const OfficialStatePod& state) {
    return ((state.turn + 1) ^ state.first_player) & 1;
}

PTCG_OFFICIAL_CONDITION_HD inline std::int32_t official_energy_type_index(
    std::int32_t type) {
    switch (type) {
        case 0: return 0;
        case 1: return 1;
        case 2: return 2;
        case 4: return 3;
        case 8: return 4;
        case 16: return 5;
        case 32: return 6;
        case 64: return 7;
        case 128: return 8;
        case 256: return 9;
        default: return -1;
    }
}

PTCG_OFFICIAL_CONDITION_HD inline bool official_card_has_attack(
    const OfficialRulePackView& rules,
    const OfficialCardRule& card,
    std::int32_t attack_id) {
    const std::int32_t offset = card.values[kCardAttackOffset];
    const std::int32_t count = card.values[kCardAttackCount];
    const std::uint32_t table_count = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kCardAttackIds)];
    if (offset < 0 || count < 0
        || static_cast<std::uint64_t>(offset) + static_cast<std::uint64_t>(count)
            > table_count) return false;
    for (std::int32_t index = 0; index < count; ++index) {
        if (rules.card_attack_ids[offset + index] == static_cast<std::uint32_t>(attack_id)) {
            return true;
        }
    }
    return false;
}

PTCG_OFFICIAL_CONDITION_HD inline bool official_player_has_special_condition(
    const OfficialPlayerStatePod& player) {
    return official_pod_poison_counter(player) > 0
        || official_pod_burned(player)
        || official_pod_bad_status(player) != OfficialBadStatus::kNone;
}

PTCG_OFFICIAL_CONDITION_HD inline std::int32_t official_attack_energy_extra(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialCardStatePod& attacker,
    OfficialCardRefPod attacker_ref,
    const OfficialAttackRule& attack) {
    std::int32_t required[10]{};
    std::int32_t required_colorless = 0;
    std::int32_t required_sum = 0;
    const OfficialCardRule* attacker_master = official_card_rule(
        rules, static_cast<std::uint32_t>(attacker.card_id));
    if (attacker_master == nullptr) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, attacker.card_id);
        return 0;
    }

    const bool owns_attack = official_card_has_attack(
        rules, *attacker_master, attack.values[kAttackId]);
    const bool force_colorless = official_continual_flag(attacker, 27) && owns_attack;
    const bool force_psychic = official_continual_flag(attacker, 28) && owns_attack;
    const bool no_energy = (attack.flags & (1ULL << 13)) != 0
        && official_player_has_special_condition(state->players[attacker.player]);
    const bool darkness_if_damaged = (attack.flags & (1ULL << 14)) != 0
        && attacker.damage > 0;
    const bool fixed = no_energy || force_colorless || force_psychic || darkness_if_damaged;
    if (!no_energy) {
        if (force_colorless) {
            required_colorless = 1;
            required_sum = 1;
        } else if (force_psychic) {
            required[5] = 1;
            required_sum = 1;
        } else if (darkness_if_damaged) {
            required[7] = 1;
            required_sum = 1;
        } else {
            const std::int32_t energy_count = attack.values[kAttackEnergyCount];
            for (std::int32_t index = 0; index < energy_count; ++index) {
                const std::int32_t type = attack.values[kAttackEnergy0 + index];
                if (type == 0) {
                    ++required_colorless;
                } else {
                    const std::int32_t type_index = official_energy_type_index(type);
                    if (type_index < 0) {
                        official_pod_fail(state, OfficialPodError::kRulePackBounds, type);
                        return 0;
                    }
                    ++required[type_index];
                }
                ++required_sum;
            }
        }
    }

    std::int32_t all_count = 0;
    std::int32_t psychic_darkness_count = 0;
    std::int32_t colorless_count = 0;
    const OfficialPlayerStatePod& player = state->players[attacker.player];
    for (std::uint16_t index = 0; index < player.energy.count; ++index) {
        const OfficialCardRefPod energy_ref = player.energy.values[index];
        const OfficialCardStatePod* energy = official_pod_card(state, energy_ref);
        if (energy == nullptr || energy->attach_move_counter != attacker.move_counter) continue;
        const OfficialEnergyInfoPod info = official_energy_info(
            *state, rules, energy_ref, attacker_ref);
        for (std::int32_t unit = 0; unit < info.count; ++unit) {
            if (info.type == 511) {
                ++all_count;
            } else if (info.type == 0) {
                ++colorless_count;
            } else if (info.type == 80) {
                ++psychic_darkness_count;
            } else {
                const std::int32_t type_index = official_energy_type_index(info.type);
                if (type_index < 0) {
                    official_pod_fail(state, OfficialPodError::kRulePackBounds, info.type);
                    return 0;
                }
                if (required[type_index] > 0) {
                    --required[type_index];
                    --required_sum;
                } else {
                    ++colorless_count;
                }
            }
        }
    }

    if (!fixed) {
        all_count += official_continual_i8(attacker, 28);
        const std::int32_t change = official_continual_i8(attacker, 27)
            + official_this_turn_i8(attacker, 10);
        if (change < 0) {
            colorless_count -= change;
        } else {
            required_colorless += change;
            required_sum += change;
        }
        if (owns_attack) {
            colorless_count += official_continual_i8(attacker, 29);
        }
    }

    for (std::int32_t index = 0; index < psychic_darkness_count; ++index) {
        if (required[5] > 0) {
            --required[5];
            --required_sum;
        } else if (required[7] > 0) {
            --required[7];
            --required_sum;
        } else {
            ++colorless_count;
        }
    }
    const std::int32_t typed_required = required_sum - required_colorless;
    if (all_count < typed_required) return all_count - typed_required;
    all_count -= typed_required;
    return all_count + colorless_count - required_colorless;
}

PTCG_OFFICIAL_CONDITION_HD inline OfficialConditionResult official_satisfy_condition_impl(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule* effects,
    std::int32_t effect_count,
    std::int32_t effect_index,
    OfficialAreaRefPod effect_card,
    std::int32_t effect_owner) {
    if (effect_index < 0 || effect_index >= effect_count) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, effect_index);
        return OfficialConditionResult::kError;
    }
    const OfficialEffectRule& effect = effects[effect_index];
    const auto type = static_cast<OfficialConditionTypeId>(
        effect.values[kEffectConditionType]);
    const std::int32_t comparator = effect.values[kEffectComparator];
    const std::int32_t value0 = effect.values[kEffectValue0];
    const std::int32_t value1 = effect.values[kEffectValue1];
    const OfficialTargetRule* target = official_condition_target(state, rules, effect);
    if (target == nullptr) return OfficialConditionResult::kError;
    OfficialPodList<OfficialAreaRefPod, kOfficialListCapacity> matches{};
    switch (type) {
        case OfficialConditionTypeId::kAlways:
            return official_condition_bool(official_bool_compare(true, comparator));
        case OfficialConditionTypeId::kAnyTargetAfterEffect: {
            bool found = false;
            for (std::int32_t index = effect_index + 1; index < effect_count; ++index) {
                const OfficialEffectRule& candidate = effects[index];
                if ((candidate.flags & kOfficialEffectIsCondition) != 0) continue;
                const OfficialTargetRule* candidate_target = official_condition_target(
                    state, rules, candidate);
                if (candidate_target == nullptr) return OfficialConditionResult::kError;
                if (!official_build_condition_target_list(
                        state,
                        rules,
                        *candidate_target,
                        &matches,
                        effect_card,
                        effect_owner)) {
                    return official_pod_ok(state)
                        ? OfficialConditionResult::kUnsupported
                        : OfficialConditionResult::kError;
                }
                if (matches.count > 0) {
                    found = true;
                    break;
                }
            }
            return official_condition_bool(official_bool_compare(found, comparator));
        }
        case OfficialConditionTypeId::kCountTarget:
        case OfficialConditionTypeId::kCountTarget2:
        case OfficialConditionTypeId::kCountTargetMeOrEnemy:
        case OfficialConditionTypeId::kCompareCountTargetMeEnemy: {
            if (type == OfficialConditionTypeId::kCountTargetMeOrEnemy) {
                OfficialTargetRule one_side = *target;
                one_side.values[kTargetPlayer] = 1;
                if (!official_build_condition_target_list(
                        state, rules, one_side, &matches, effect_card, effect_owner)) {
                    return OfficialConditionResult::kError;
                }
                const std::int32_t me = matches.count;
                one_side.values[kTargetPlayer] = 2;
                if (!official_build_condition_target_list(
                        state, rules, one_side, &matches, effect_card, effect_owner)) {
                    return OfficialConditionResult::kError;
                }
                const std::int32_t enemy = matches.count;
                return official_condition_bool(
                    official_compare(me, value0, comparator)
                    || official_compare(enemy, value1, comparator));
            }
            if (!official_build_condition_target_list(
                    state, rules, *target, &matches, effect_card, effect_owner)) {
                return OfficialConditionResult::kError;
            }
            if (type == OfficialConditionTypeId::kCountTarget) {
                return official_condition_bool(
                    official_compare(matches.count, value0, comparator));
            }
            if (type == OfficialConditionTypeId::kCountTarget2) {
                return official_condition_bool(official_bool_compare(
                    matches.count == value0 || matches.count == value1, comparator));
            }
            std::int32_t counts[2]{};
            for (std::uint16_t index = 0; index < matches.count; ++index) {
                const OfficialCardStatePod* card = official_pod_card(
                    state, matches.values[index].card);
                if (card != nullptr && card->player >= 0 && card->player <= 1) {
                    ++counts[card->player];
                }
            }
            return official_condition_bool(
                official_compare(counts[effect_owner], counts[1 - effect_owner], comparator));
        }
        case OfficialConditionTypeId::kCountEnergy:
        case OfficialConditionTypeId::kCountEnergyType: {
            std::int32_t required_type = -1;
            if (type == OfficialConditionTypeId::kCountEnergyType) {
                const std::int32_t offset = target->values[kTargetConditionOffset];
                if (target->values[kTargetConditionCount] <= 0
                    || offset < 0
                    || static_cast<std::uint32_t>(offset)
                        >= rules.header->counts[static_cast<std::uint32_t>(
                            OfficialRuleSection::kConditions)]) {
                    official_pod_fail(state, OfficialPodError::kRulePackBounds, offset);
                    return OfficialConditionResult::kError;
                }
                required_type = rules.conditions[offset].values[kConditionValue];
            }
            std::int32_t count = 0;
            const std::int32_t mask = target->values[kTargetPlayer];
            for (int player = 0; player < 2; ++player) {
                if (official_target_player_matches(effect_owner, player, mask)) {
                    count += official_condition_energy_count(
                        *state, rules, player, required_type);
                }
            }
            return official_condition_bool(official_compare(count, value0, comparator));
        }
        case OfficialConditionTypeId::kCompareCountEnergyMeEnemy: {
            if (!official_build_condition_target_list(
                    state, rules, *target, &matches, effect_card, effect_owner)) {
                return OfficialConditionResult::kError;
            }
            std::int32_t counts[2]{};
            for (std::uint16_t index = 0; index < matches.count; ++index) {
                const OfficialCardStatePod* card = official_pod_card(
                    state, matches.values[index].card);
                if (card == nullptr || card->player < 0 || card->player > 1) continue;
                const OfficialCardRule* master = official_card_rule(
                    rules, static_cast<std::uint32_t>(card->card_id));
                if (master != nullptr) {
                    counts[card->player] += master->values[kCardEnergyCount] > 0
                        ? master->values[kCardEnergyCount]
                        : 1;
                }
            }
            return official_condition_bool(
                official_compare(counts[effect_owner], counts[1 - effect_owner], comparator));
        }
        case OfficialConditionTypeId::kNotFullBench: {
            const int player = target->values[kTargetPlayer] == 1
                ? effect_owner : 1 - effect_owner;
            const std::int32_t configured_capacity = static_cast<std::int32_t>(
                (state->players[player].continual_state >> 40U) & 0xfULL);
            const std::int32_t bench_capacity = configured_capacity == 0
                ? 5 : configured_capacity;
            return official_condition_bool(
                official_bool_compare(
                    state->players[player].bench.count < bench_capacity,
                    comparator));
        }
        case OfficialConditionTypeId::kMyTurn: {
            const OfficialCardStatePod* card = official_pod_card(state, effect_card.card);
            const bool my_turn = state->phase == static_cast<std::uint8_t>(
                    OfficialGamePhase::kMain)
                && card != nullptr
                && card->player == official_active_player(*state);
            return official_condition_bool(official_bool_compare(my_turn, comparator));
        }
        case OfficialConditionTypeId::kTurn:
            return official_condition_bool(official_compare(state->turn, value0, comparator));
        case OfficialConditionTypeId::kKoPreEnemyTurn:
        case OfficialConditionTypeId::kKoPreEnemyTurnTeamRocket:
        case OfficialConditionTypeId::kKoAttackDamagePreEnemyTurn:
        case OfficialConditionTypeId::kKoAttackDamageEthanPreEnemyTurn:
        case OfficialConditionTypeId::kKoAttackDamageHopPreEnemyTurn: {
            std::uint8_t flag = kHistoryKo;
            if (type == OfficialConditionTypeId::kKoPreEnemyTurnTeamRocket) {
                flag = kHistoryKoTeamRocket;
            } else if (type == OfficialConditionTypeId::kKoAttackDamagePreEnemyTurn) {
                flag = kHistoryKoAttackDamage;
            } else if (type == OfficialConditionTypeId::kKoAttackDamageEthanPreEnemyTurn) {
                flag = kHistoryKoAttackDamageEthan;
            } else if (type == OfficialConditionTypeId::kKoAttackDamageHopPreEnemyTurn) {
                flag = kHistoryKoAttackDamageHop;
            }
            return official_condition_bool(official_bool_compare(
                (state->turn_histories[1].flags & flag) != 0, comparator));
        }
        case OfficialConditionTypeId::kNoSameNameSkillThisTurn: {
            const std::int32_t skill_id = effect.values[kEffectSkillId];
            std::int32_t used_count = 0;
            for (std::uint16_t index = 0; index < state->turn_used_skills.count; ++index) {
                if (state->turn_used_skills.values[index] == skill_id) {
                    ++used_count;
                }
            }
            // The official engine records an activation before interpreting
            // its effects.  Do not let the current activation fail its own
            // no-duplicate condition; earlier uses must still reject it.
            if (used_count > 0
                && state->effect_interpreter.active != 0
                && state->effect_state.on_effect != 0
                && state->effect_state.ability.skill_id == skill_id) {
                --used_count;
            }
            return official_condition_bool(
                official_bool_compare(used_count == 0, comparator));
        }
        case OfficialConditionTypeId::kSameAttackPreMyTurn:
            return official_condition_bool(official_bool_compare(
                state->turn_histories[2].attack_id == state->current_attack_id
                    && state->turn_histories[2].attack_card == state->attacker,
                comparator));
        case OfficialConditionTypeId::kCoinHeadCount:
            return official_condition_bool(
                official_compare(state->coin_head_count, value0, comparator));
        case OfficialConditionTypeId::kAttachActive:
            return official_condition_bool(official_bool_compare(
                state->attach_active != 0, comparator));
        case OfficialConditionTypeId::kMysteryGarden: {
            const OfficialPlayerStatePod& ps = state->players[effect_owner];
            std::int32_t psychic = 0;
            for (int zone = 0; zone < 2; ++zone) {
                const OfficialCardRefPod* values = zone == 0 ? ps.active.values : ps.bench.values;
                const std::uint16_t count = zone == 0 ? ps.active.count : ps.bench.count;
                for (std::uint16_t index = 0; index < count; ++index) {
                    const OfficialCardStatePod* card = official_pod_card(state, values[index]);
                    const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
                        rules, static_cast<std::uint32_t>(card->card_id));
                    if (master != nullptr
                        && official_energy_type_matches(master->values[kCardEnergyType], 1 << 4)) {
                        ++psychic;
                    }
                }
            }
            return official_condition_bool(ps.hand.count <= psychic);
        }
        case OfficialConditionTypeId::kLoveBall: {
            const OfficialPlayerStatePod& me = state->players[effect_owner];
            const OfficialPlayerStatePod& enemy = state->players[1 - effect_owner];
            for (int enemy_zone = 0; enemy_zone < 2; ++enemy_zone) {
                const OfficialCardRefPod* enemy_values = enemy_zone == 0
                    ? enemy.active.values : enemy.bench.values;
                const std::uint16_t enemy_count = enemy_zone == 0
                    ? enemy.active.count : enemy.bench.count;
                for (std::uint16_t enemy_index = 0; enemy_index < enemy_count; ++enemy_index) {
                    const OfficialCardStatePod* enemy_card = official_pod_card(
                        state, enemy_values[enemy_index]);
                    const OfficialCardRule* enemy_master = enemy_card == nullptr ? nullptr
                        : official_card_rule(rules, static_cast<std::uint32_t>(enemy_card->card_id));
                    if (enemy_master == nullptr || enemy_master->values[kCardType] != 0) continue;
                    std::int32_t same_name = 0;
                    const std::int32_t name_id = enemy_master->values[kCardNameId];
                    for (int my_zone = 0; my_zone < 3; ++my_zone) {
                        const OfficialCardRefPod* values = my_zone == 0
                            ? me.active.values
                            : (my_zone == 1 ? me.bench.values : me.trash.values);
                        const std::uint16_t count = my_zone == 0
                            ? me.active.count
                            : (my_zone == 1 ? me.bench.count : me.trash.count);
                        for (std::uint16_t index = 0; index < count; ++index) {
                            const OfficialCardStatePod* card = official_pod_card(state, values[index]);
                            const OfficialCardRule* master = card == nullptr ? nullptr
                                : official_card_rule(rules, static_cast<std::uint32_t>(card->card_id));
                            if (master != nullptr && master->values[kCardNameId] == name_id) {
                                ++same_name;
                            }
                        }
                    }
                    if (same_name < 4) return OfficialConditionResult::kTrue;
                }
            }
            return OfficialConditionResult::kFalse;
        }
        case OfficialConditionTypeId::kAttackEnergyExtra: {
            const OfficialCardStatePod* attacker = official_pod_card(state, state->attacker);
            const OfficialAttackRule* attack = official_attack_rule(
                rules, static_cast<std::uint32_t>(state->source_attack_id));
            if (attacker == nullptr || attack == nullptr) {
                official_pod_fail(
                    state, OfficialPodError::kRulePackBounds, state->source_attack_id);
                return OfficialConditionResult::kError;
            }
            const std::int32_t extra = official_attack_energy_extra(
                state, rules, *attacker, state->attacker, *attack);
            if (!official_pod_ok(state)) return OfficialConditionResult::kError;
            return official_condition_bool(official_compare(extra, value0, comparator));
        }
        default:
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedCondition,
                effect.values[kEffectConditionType]);
            return OfficialConditionResult::kUnsupported;
    }
}

PTCG_OFFICIAL_CONDITION_HD inline OfficialConditionResult official_satisfy_condition(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule* effects,
    std::int32_t effect_count,
    std::int32_t effect_index,
    OfficialAreaRefPod effect_card,
    std::int32_t effect_owner) {
    const OfficialConditionResult result = official_satisfy_condition_impl(
        state,
        rules,
        effects,
        effect_count,
        effect_index,
        effect_card,
        effect_owner);
#if defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
    if (effect_index >= 0 && effect_index < effect_count) {
        const std::ptrdiff_t absolute = effects + effect_index - rules.effects;
        if (absolute >= 0
            && absolute < static_cast<std::ptrdiff_t>(
                kOfficialBranchCoverageEffectCapacity)) {
            if (result == OfficialConditionResult::kTrue
                || result == OfficialConditionResult::kFalse) {
                official_mark_effect_condition(
                    state,
                    static_cast<std::uint32_t>(absolute),
                    result == OfficialConditionResult::kTrue);
            }
        }
    }
#endif
    return result;
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_CONDITION_HD
