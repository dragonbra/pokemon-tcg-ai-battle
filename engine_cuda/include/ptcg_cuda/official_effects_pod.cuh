#pragma once

#include <cstdint>

#include "ptcg_cuda/official_semantic_ids.cuh"
#include "ptcg_cuda/official_targets_pod.cuh"
#include "ptcg_cuda/official_trigger_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_EFFECT_HD __host__ __device__
#else
#define PTCG_OFFICIAL_EFFECT_HD
#endif

namespace ptcg::cuda_engine {

enum class OfficialEffectApplyResult : std::int32_t {
    kApplied = 0,
    kNeedsSelection = 1,
    kUnsupported = 2,
    kError = 3,
};

// official_effects_pod.cuh is included while the continual-refresh header is
// still being defined.  EvolveProc nevertheless calls RefreshEffect before
// it exposes any following selection, so keep an opaque declaration here and
// use the definition completed by official_continual_refresh_pod.cuh.
enum class OfficialContinualRefreshResult : std::int32_t;

PTCG_OFFICIAL_EFFECT_HD inline OfficialContinualRefreshResult
official_refresh_continual_effects(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

constexpr std::uint64_t kEffectMultiplyPreTargetCount = 1ULL << 9;
constexpr std::uint64_t kEffectMultiplyCoinHeadCount = 1ULL << 10;
constexpr std::uint64_t kEffectNotClearSelectedList = 1ULL << 6;
constexpr std::uint64_t kEffectOpen = 1ULL << 20;
constexpr std::uint8_t kOfficialTurnEndFlag = 1U << 4;
constexpr std::uint8_t kOfficialBreakEffectFlag = 1U << 1;
constexpr std::int32_t kOfficialInitialPrizeCount = 6;

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_clamp_i32(
    std::int32_t value,
    std::int32_t low,
    std::int32_t high) {
    return value < low ? low : (value > high ? high : value);
}

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_packed_i16(
    const std::uint32_t* words,
    std::uint32_t byte_offset) {
    const std::uint32_t word = byte_offset / 4U;
    const std::uint32_t shift = (byte_offset % 4U) * 8U;
    const std::uint16_t raw = static_cast<std::uint16_t>(
        (words[word] >> shift) & 0xffffU);
    return static_cast<std::int16_t>(raw);
}

PTCG_OFFICIAL_EFFECT_HD inline void official_set_packed_i16(
    std::uint32_t* words,
    std::uint32_t byte_offset,
    std::int32_t value) {
    const std::uint32_t word = byte_offset / 4U;
    const std::uint32_t shift = (byte_offset % 4U) * 8U;
    const std::uint32_t mask = 0xffffU << shift;
    words[word] = (words[word] & ~mask)
        | (static_cast<std::uint32_t>(static_cast<std::uint16_t>(value)) << shift);
}

PTCG_OFFICIAL_EFFECT_HD inline void official_add_packed_i16(
    std::uint32_t* words,
    std::uint32_t byte_offset,
    std::int32_t value) {
    official_set_packed_i16(
        words,
        byte_offset,
        official_packed_i16(words, byte_offset) + value);
}

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_packed_i8(
    const std::uint32_t* words,
    std::uint32_t byte_offset) {
    const std::uint32_t word = byte_offset / 4U;
    const std::uint32_t shift = (byte_offset % 4U) * 8U;
    return static_cast<std::int8_t>((words[word] >> shift) & 0xffU);
}

PTCG_OFFICIAL_EFFECT_HD inline void official_set_packed_i8(
    std::uint32_t* words,
    std::uint32_t byte_offset,
    std::int32_t value) {
    const std::uint32_t word = byte_offset / 4U;
    const std::uint32_t shift = (byte_offset % 4U) * 8U;
    const std::uint32_t mask = 0xffU << shift;
    words[word] = (words[word] & ~mask)
        | (static_cast<std::uint32_t>(static_cast<std::uint8_t>(value)) << shift);
}

PTCG_OFFICIAL_EFFECT_HD inline void official_add_packed_i8(
    std::uint32_t* words,
    std::uint32_t byte_offset,
    std::int32_t value) {
    official_set_packed_i8(
        words,
        byte_offset,
        official_packed_i8(words, byte_offset) + value);
}

PTCG_OFFICIAL_EFFECT_HD inline void official_set_packed_flag(
    std::uint32_t* words,
    std::uint32_t bit) {
    words[bit / 32U] |= 1U << (bit % 32U);
}

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_popcount_u32(
    std::uint32_t value) {
    std::int32_t count = 0;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
}

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_effect_target_energy_count(
    const OfficialStatePod& state,
    const OfficialRulePackView& rules,
    std::int32_t required_type = -1) {
    std::int32_t count = 0;
    for (std::uint16_t index = 0; index < state.targets.count; ++index) {
        const OfficialAreaRefPod target = state.targets.values[index];
        if (official_pod_area_ref_valid(&state, target)) {
            count += official_attached_energy_cards(
                state, target.card, required_type, false, rules);
        }
    }
    return count;
}

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_effect_roll_coins(
    OfficialStatePod* state,
    std::int32_t count) {
    state->coin_head_count = 0;
    for (std::int32_t index = 0; index < count; ++index) {
        official_pod_coin(state, state->effect_state.ability.use_player);
    }
    return state->coin_head_count;
}

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_effect_roll_until_tail(
    OfficialStatePod* state) {
    state->coin_head_count = 0;
    while (state->coin_head_count < 4096
        && official_pod_coin(state, state->effect_state.ability.use_player)) {}
    if (state->coin_head_count == 4096) {
        official_pod_fail(state, OfficialPodError::kInterpreterBudget, 4096);
    }
    return state->coin_head_count;
}

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_player_turn_i16(
    const OfficialPlayerStatePod& player,
    std::uint32_t byte_offset) {
    return static_cast<std::int16_t>(
        (player.turn_state >> (byte_offset * 8U)) & 0xffffULL);
}

PTCG_OFFICIAL_EFFECT_HD inline bool official_effect_has_special_energy(
    const OfficialStatePod& state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod pokemon) {
    const OfficialCardStatePod* target = official_pod_card(&state, pokemon);
    if (target == nullptr || target->player < 0 || target->player > 1) return false;
    const auto& energies = state.players[target->player].energy;
    for (std::uint16_t index = 0; index < energies.count; ++index) {
        const OfficialCardStatePod* energy = official_pod_card(
            &state, energies.values[index]);
        if (energy == nullptr || energy->attach_move_counter != target->move_counter) continue;
        const OfficialCardRule* master = official_card_rule(
            rules, static_cast<std::uint32_t>(energy->card_id));
        if (master != nullptr && master->values[kCardType] == 6) return true;
    }
    return false;
}

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_calc_attack_damage(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t base_damage,
    OfficialCardRefPod target_ref,
    OfficialCardRefPod attacker_ref,
    bool calculate_weakness) {
    if (base_damage <= 0) return 0;
    OfficialCardStatePod* target = official_pod_card(state, target_ref);
    OfficialCardStatePod* attacker = official_pod_card(state, attacker_ref);
    if (target == nullptr || attacker == nullptr) return 0;
    const OfficialCardRule* target_master = official_card_rule(
        rules, static_cast<std::uint32_t>(target->card_id));
    const OfficialCardRule* attacker_master = official_card_rule(
        rules, static_cast<std::uint32_t>(attacker->card_id));
    const OfficialAttackRule* attack = official_attack_rule(
        rules, static_cast<std::uint32_t>(state->current_attack_id));
    if (target_master == nullptr || attacker_master == nullptr || attack == nullptr) {
        official_pod_fail(
            state,
            OfficialPodError::kRulePackBounds,
            attack == nullptr ? state->current_attack_id : target->card_id);
        return 0;
    }

    std::int32_t damage = base_damage;
    const std::int32_t attacker_type = official_effective_pokemon_type(
        *attacker, *attacker_master);
    damage += official_packed_i16(attacker->this_turn, 4);
    damage += official_continual_i16(*attacker, 2);
    const bool target_active = target->area == static_cast<std::uint8_t>(OfficialArea::kActive);
    if (target_active) damage += official_packed_i16(attacker->this_turn, 6);
    if (target_active && attacker->player != target->player) {
        const OfficialPlayerStatePod& player = state->players[attacker->player];
        damage += official_continual_i16(*attacker, 4);
        damage += official_packed_i16(attacker->turn_state, 0);
        damage += official_continual_i16(*attacker, 12)
            * (kOfficialInitialPrizeCount - state->players[1 - attacker->player].prize.count);
        damage += official_player_turn_i16(player, 0);
        const bool target_ex = target_master->values[kCardPokemonType] == 3
            || target_master->values[kCardPokemonType] == 4;
        if (target_ex) {
            damage += official_continual_i16(*attacker, 6);
            damage += official_packed_i16(attacker->turn_state, 2);
            damage += official_player_turn_i16(player, 2);
        }
        if ((attacker_type & 32) != 0) {
            damage += official_player_turn_i16(player, 4);
        }
        if (official_continual_i16(*attacker, 8) != 0
            && target_master->values[kCardAbilityId] != 0
            && !official_continual_flag(*target, 0)) {
            damage += official_continual_i16(*attacker, 8);
        }
        if (official_continual_i16(*attacker, 10) != 0
            && target_master->values[kCardEvolutionType] != 1) {
            damage += official_continual_i16(*attacker, 10);
        }
    }
    if (damage <= 0) return 0;

    bool calculate_resistance = calculate_weakness;
    const bool calculate_target_effect = (attack->flags & (1ULL << 5)) == 0;
    if ((attack->flags & (1ULL << 6)) != 0) calculate_weakness = false;
    if ((attack->flags & (1ULL << 7)) != 0) calculate_resistance = false;
    if (calculate_weakness && (target->next_enemy_turn_end & (1U << 30U)) == 0) {
        std::int32_t weakness = target_master->values[kCardWeakness];
        const std::int32_t weakness_index = official_continual_i8(*target, 31);
        if (weakness_index > 0) weakness = official_energy_type_by_index(weakness_index);
        if ((weakness & attacker_type) != 0) damage *= 2;
    }
    if (calculate_resistance
        && (target_master->values[kCardResistance] & attacker_type) != 0) {
        damage -= 30;
        if (damage <= 0) return 0;
    }

    if (calculate_target_effect) {
        damage += official_continual_i16(*target, 14);
        damage += static_cast<std::int16_t>(target->next_enemy_turn_end & 0xffffU);
        damage += static_cast<std::int16_t>(target->this_turn_enemy & 0xffffU);
        const bool enemy_attack = attacker->player != target->player;
        const bool attacker_has_ability = attacker_master->values[kCardAbilityId] != 0
            && !official_continual_flag(*attacker, 0);
        const bool attacker_ex = attacker_master->values[kCardPokemonType] == 3
            || attacker_master->values[kCardPokemonType] == 4;
        if (enemy_attack) {
            damage += official_continual_i16(*target, 16);
            if ((attacker_type & (2 | 4)) != 0) {
                damage += official_continual_i16(*target, 20);
            }
            if ((attacker_type & (1 | 2 | 4 | 8)) != 0) {
                damage += official_continual_i16(*target, 22);
            }
            if (attacker_has_ability) {
                damage += official_continual_i16(*target, 18);
            }
            const std::int32_t metal_change = static_cast<std::int16_t>(
                state->players[attacker->player].this_turn & 0xffffU);
            if (metal_change != 0
                && (official_effective_pokemon_type(*target, *target_master) & 128) != 0) {
                damage += metal_change;
            }
            if ((official_continual_flag(*target, 2) && attacker_has_ability)
                || (official_continual_flag(*target, 3) && attacker_ex)
                || (official_continual_flag(*target, 4)
                    && attacker_ex
                    && attacker_master->values[kCardEvolutionType] == 1)
                || (official_continual_flag(*target, 5)
                    && (attacker_master->flags & 1ULL) != 0)
                || (official_continual_flag(*target, 6)
                    && official_effect_has_special_energy(
                        *state, rules, attacker_ref))
                || ((target->next_enemy_turn_end_battlefield & 1U) != 0 && attacker_ex)
                || official_continual_flag(*target, 7)
                || (target->next_enemy_turn_end & (1U << 25U)) != 0) {
                damage = 0;
            }
        }
        if ((target->next_enemy_turn_end & ((1U << 24U) | (1U << 26U))) != 0) {
            damage = 0;
        }
        if ((target->next_enemy_turn_end & (1U << 27U)) != 0
            && attacker_master->values[kCardEvolutionType] == 1) damage = 0;
        if ((target->next_enemy_turn_end & (1U << 28U)) != 0
            && attacker_master->values[kCardEvolutionType] == 1
            && attacker_type != 0) damage = 0;
        if ((target->next_enemy_turn_end & (1U << 29U)) != 0
            && attacker_has_ability) damage = 0;
        const std::int32_t no_damage_less_equal = static_cast<std::uint8_t>(
            (target->next_enemy_turn_end >> 16U) & 0xffU);
        if (damage <= no_damage_less_equal) damage = 0;
        if (target->area == static_cast<std::uint8_t>(OfficialArea::kBench)
            && (target_master->flags & 1ULL) != 0) damage = 0;
        const std::int32_t no_damage_greater_equal =
            official_continual_i16(*target, 24);
        if (no_damage_greater_equal > 0 && damage >= no_damage_greater_equal) damage = 0;
    }
    if (damage <= 0) return 0;
    return damage > 100000000 ? 100000000 : damage;
}

PTCG_OFFICIAL_EFFECT_HD inline void official_attack_consume_damage_berries(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod target_ref) {
    OfficialCardStatePod* target = official_pod_card(state, target_ref);
    const OfficialCardStatePod* attacker = official_pod_card(state, state->attacker);
    if (target == nullptr || attacker == nullptr
        || !official_continual_flag(*target, 37)
        || target->player < 0 || target->player > 1) {
        return;
    }
    const OfficialCardRule* attacker_master = official_card_rule(
        rules, static_cast<std::uint32_t>(attacker->card_id));
    if (attacker_master == nullptr) return;
    const std::int32_t attacker_type = official_effective_pokemon_type(
        *attacker, *attacker_master);
    auto& tools = state->players[target->player].tool;
    for (std::int32_t index = static_cast<std::int32_t>(tools.count) - 1;
         index >= 0 && official_pod_ok(state);
         --index) {
        const OfficialCardRefPod tool_ref = tools.values[index];
        const OfficialCardStatePod* tool = official_pod_card(state, tool_ref);
        if (tool == nullptr || tool->attach_move_counter != target->move_counter) continue;
        const bool psychic_berry = tool->card_id == 1164 && (attacker_type & 16) != 0;
        const bool dragon_berry = tool->card_id == 1170 && (attacker_type & 256) != 0;
        if (psychic_berry || dragon_berry) {
            official_pod_move_card(
                state,
                target->player,
                OfficialArea::kTool,
                static_cast<std::uint16_t>(index),
                OfficialArea::kTrash,
                false);
        }
    }
}

PTCG_OFFICIAL_EFFECT_HD inline bool official_pull_attack_damage_triggers(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod target_ref,
    std::int32_t damage) {
    const OfficialCardStatePod* target = official_pod_card(state, target_ref);
    const OfficialCardStatePod* attacker = official_pod_card(state, state->attacker);
    if (target == nullptr || attacker == nullptr || damage <= 0
        || target->player == attacker->player) {
        return true;
    }
    if (!official_pull_trigger(
            state, rules, 10, target_ref, state->attacker, 1)) return false;
    if (target->area != static_cast<std::uint8_t>(OfficialArea::kActive)) return true;
    if (!official_pull_trigger(
            state, rules, 11, target_ref, state->attacker, 1)) return false;
    for (std::int32_t index = static_cast<std::int32_t>(state->delay_triggers.count) - 1;
         index >= 0;
         --index) {
        OfficialTriggeredAbilityPod pending = state->delay_triggers.values[index];
        if (pending.trigger.type != 11
            || pending.trigger.subject.card != target_ref) continue;
        pending.trigger.object = official_pod_area_ref(state, state->attacker);
        pending.trigger.depth = 1;
        pending.trigger.value = damage;
        if (!official_pod_push(
                state,
                &state->temporary_triggers,
                pending,
                OfficialPodError::kTriggerStackOverflow)) return false;
    }
    return true;
}

PTCG_OFFICIAL_EFFECT_HD inline void official_apply_attack_damage_to_card(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod target_ref,
    std::int32_t base_damage,
    bool record_last_attack_damage = false) {
    OfficialCardStatePod* target = official_pod_card(state, target_ref);
    OfficialCardStatePod* attacker = official_pod_card(state, state->attacker);
    if (target == nullptr || attacker == nullptr) return;
    const OfficialAttackRule* attack = official_attack_rule(
        rules, static_cast<std::uint32_t>(state->current_attack_id));
    if (attack == nullptr) {
        official_pod_fail(
            state, OfficialPodError::kRulePackBounds, state->current_attack_id);
        return;
    }
    const bool target_active = target->area == static_cast<std::uint8_t>(OfficialArea::kActive);
    const std::int32_t damage = official_calc_attack_damage(
        state, rules, base_damage, target_ref, state->attacker, target_active);
    if (!official_pod_ok(state)) return;
    if (official_continual_flag(*target, 30) && damage > 0) {
        const bool protection_head = official_effect_roll_coins(state, 1) != 0;
        if (protection_head && (attack->flags & (1ULL << 5U)) == 0) return;
    }
    if (record_last_attack_damage) state->last_attack_damage = damage;
    const std::int32_t before = target->damage;
    const std::int32_t max_hp = official_pod_max_hp(state, rules, target_ref);
    const bool enemy_attack = attacker->player != target->player;
    official_pod_add_damage(state, rules, target_ref, damage);
    if (damage > 0 && before < max_hp) {
        target->take_attack_damage_this_turn += damage;
        if (before == 0 && damage >= max_hp && enemy_attack) {
            official_pod_set_card_runtime_flag(target, kCardKoFull);
        }
        if (target->damage >= max_hp) {
            target->turn_state[1] = (target->turn_state[1] & ~0xffU)
                | static_cast<std::uint32_t>(state->attacker.index & 0xffU);
            official_pod_set_card_runtime_flag(target, kCardKoAttackDamage);
            if (enemy_attack) {
                official_pod_set_card_runtime_flag(
                    target, kCardKoEnemyAttackDamage);
                if (target_active) {
                    official_pod_set_card_runtime_flag(
                        target, kCardKoEnemyAttackDamageActive);
                }
                const OfficialCardRule* attacker_master = official_card_rule(
                    rules, static_cast<std::uint32_t>(attacker->card_id));
                if (attacker_master != nullptr) {
                    const bool attacker_ex = attacker_master->values[kCardPokemonType] == 3
                        || attacker_master->values[kCardPokemonType] == 4;
                    if (attacker_ex) {
                        official_pod_set_card_runtime_flag(
                            target, kCardKoEnemyExAttackDamage);
                    }
                    if ((attacker_master->flags & 1ULL) != 0) {
                        official_pod_set_card_runtime_flag(
                            target, kCardKoEnemyTerastalAttackDamage);
                    }
                    if ((attacker_master->flags & (1ULL << 16U)) != 0) {
                        official_pod_set_card_runtime_flag(
                            target, kCardKoEnemyNAttackDamage);
                    }
                    const OfficialCardRule* target_master = official_card_rule(
                        rules, static_cast<std::uint32_t>(target->card_id));
                    if (official_continual_flag(*attacker, 32)
                        && target_master != nullptr
                        && target_master->values[kCardEvolutionType] == 1) {
                        official_pod_set_card_runtime_flag(
                            target, kCardKoPrizePlus1);
                    }
                    if (official_continual_flag(*target, 17) && attacker_ex) {
                        official_pod_set_card_runtime_flag(
                            target, kCardKoPrizeZero);
                    }
                }
                if ((attack->flags & (1ULL << 15U)) != 0) {
                    official_pod_set_card_runtime_flag(target, kCardKoPrizePlus1);
                }
                if ((attack->flags & (1ULL << 16U)) != 0) {
                    official_pod_set_card_runtime_flag(
                        target, kCardKoNoDamageAndEffectAttackNextEnemyTurn);
                }
            }
        }
        if (!official_pull_attack_damage_triggers(
                state, rules, target_ref, damage)) return;
    }
    official_attack_consume_damage_berries(state, rules, target_ref);
}

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_effect_value(
    const OfficialStatePod& state,
    const OfficialEffectRule& effect,
    std::uint32_t index) {
    std::int32_t value = effect.values[kEffectValue0 + index];
    if ((effect.flags & kEffectMultiplyPreTargetCount) != 0) {
        value *= state.pre_targets.count;
    }
    if ((effect.flags & kEffectMultiplyCoinHeadCount) != 0) {
        value *= state.coin_head_count;
    }
    return value;
}

PTCG_OFFICIAL_EFFECT_HD inline const OfficialTargetRule* official_effect_target_rule(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect) {
    const std::int32_t index = effect.values[kEffectTargetIndex];
    if (index < 0
        || static_cast<std::uint32_t>(index) >= rules.header->counts[
            static_cast<std::uint32_t>(OfficialRuleSection::kTargets)]) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, index);
        return nullptr;
    }
    return &rules.targets[index];
}

PTCG_OFFICIAL_EFFECT_HD inline std::uint8_t official_effect_player_mask(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect) {
    const OfficialTargetRule* target = official_effect_target_rule(state, rules, effect);
    return target == nullptr ? 0 : static_cast<std::uint8_t>(target->values[kTargetPlayer]);
}

PTCG_OFFICIAL_EFFECT_HD inline std::int32_t official_effect_target_player(
    const OfficialStatePod& state,
    std::uint8_t mask,
    std::int32_t ordinal) {
    const std::int32_t user = state.effect_state.ability.use_player;
    if (mask == 1) return ordinal == 0 ? user : -1;
    if (mask == 2) return ordinal == 0 ? 1 - user : -1;
    if (mask == 3) return ordinal < 2 ? ordinal : -1;
    return -1;
}

PTCG_OFFICIAL_EFFECT_HD inline OfficialCardRefPod official_effect_target_pokemon_ref(
    const OfficialStatePod& state,
    OfficialCardRefPod target_ref) {
    const OfficialCardStatePod* target = official_pod_card(&state, target_ref);
    if (target == nullptr
        || (target->area != static_cast<std::uint8_t>(OfficialArea::kEnergy)
            && target->area != static_cast<std::uint8_t>(OfficialArea::kTool))
        || target->player < 0 || target->player > 1) {
        return target_ref;
    }
    const OfficialPlayerStatePod& player = state.players[target->player];
    for (int zone = 0; zone < 2; ++zone) {
        const OfficialCardRefPod* values = zone == 0
            ? player.active.values : player.bench.values;
        const std::uint16_t count = zone == 0
            ? player.active.count : player.bench.count;
        for (std::uint16_t index = 0; index < count; ++index) {
            const OfficialCardStatePod* pokemon = official_pod_card(
                &state, values[index]);
            if (pokemon != nullptr
                && pokemon->move_counter == target->attach_move_counter) {
                return values[index];
            }
        }
    }
    return target_ref;
}

// Mirrors State::isPreventEffect.  Attack-effect immunity is distinct from
// attack-damage immunity: Rock Fighting Energy, for example, blocks Alakazam's
// Powerful Hand damage counters but does not block ordinary attack damage.
PTCG_OFFICIAL_EFFECT_HD inline bool official_effect_blocks_target_effect(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod target_ref) {
    const OfficialCardRefPod pokemon_ref = official_effect_target_pokemon_ref(
        *state, target_ref);
    const OfficialCardStatePod* target = official_pod_card(state, pokemon_ref);
    if (target == nullptr) return false;

    bool attack_effect = state->current_attack_id > 0;
    if (attack_effect
        && state->effect_state.on_effect != 0
        && state->attacker
            != state->effect_state.ability.effect_card.card) {
        attack_effect = false;
    }
    if (attack_effect) {
        const OfficialCardStatePod* attacker = official_pod_card(
            state, state->attacker);
        if (attacker == nullptr) return false;
        const OfficialCardRule* attacker_master = official_card_rule(
            rules, static_cast<std::uint32_t>(attacker->card_id));
        if (attacker_master == nullptr) {
            official_pod_fail(
                state, OfficialPodError::kRulePackBounds, attacker->card_id);
            return true;
        }
        if (attacker->player != target->player) {
            const bool attacker_ex =
                attacker_master->values[kCardPokemonType] == 3
                || attacker_master->values[kCardPokemonType] == 4;
            if ((official_continual_flag(*target, 5)
                    && (attacker_master->flags & 1ULL) != 0)
                || (official_continual_flag(*target, 6)
                    && official_effect_has_special_energy(
                        *state, rules, state->attacker))
                || official_continual_flag(*target, 8)
                || (target->next_enemy_turn_end & (1U << 25U)) != 0
                || ((target->next_enemy_turn_end_battlefield & 1U) != 0
                    && attacker_ex)) {
                return true;
            }
        }
        return (target->next_enemy_turn_end & (1U << 24U)) != 0;
    }

    if (state->effect_state.on_effect == 0) return false;
    const OfficialCardStatePod* source = official_pod_card(
        state, state->effect_state.ability.effect_card.card);
    if (source == nullptr || source->player == target->player) return false;
    const OfficialCardRule* source_master = official_card_rule(
        rules, static_cast<std::uint32_t>(source->card_id));
    if (source_master == nullptr) {
        official_pod_fail(
            state, OfficialPodError::kRulePackBounds, source->card_id);
        return true;
    }
    const std::int32_t source_type = source_master->values[kCardType];
    if (official_continual_flag(*target, 9) && source_type == 1) return true;
    if (official_continual_flag(*target, 10) && source_type == 3) return true;
    return official_continual_flag(*target, 12)
        && (source->area == static_cast<std::uint8_t>(OfficialArea::kActive)
            || source->area == static_cast<std::uint8_t>(OfficialArea::kBench));
}

PTCG_OFFICIAL_EFFECT_HD inline bool official_effect_blocks_active_effect(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return true;
    }
    if (state->players[player].active.count == 0) return true;
    return official_effect_blocks_target_effect(
        state, rules, state->players[player].active.values[0]);
}

PTCG_OFFICIAL_EFFECT_HD inline bool official_effect_blocks_damage_counter(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod target_ref) {
    if (official_effect_blocks_target_effect(
            state, rules, target_ref)) return true;
    const OfficialCardRefPod pokemon_ref = official_effect_target_pokemon_ref(
        *state, target_ref);
    const OfficialCardStatePod* target = official_pod_card(state, pokemon_ref);
    const OfficialCardStatePod* source = official_pod_card(
        state, state->effect_state.ability.effect_card.card);
    if (target == nullptr || source == nullptr
        || target->player == source->player) return false;
    const OfficialCardRule* source_master = official_card_rule(
        rules, static_cast<std::uint32_t>(source->card_id));
    if (source_master == nullptr) {
        official_pod_fail(
            state, OfficialPodError::kRulePackBounds, source->card_id);
        return true;
    }
    const std::int32_t source_type = source_master->values[kCardType];
    return official_continual_flag(*target, 11) && source_type == 0;
}

PTCG_OFFICIAL_EFFECT_HD inline bool official_card_cannot_move_damage_counter(
    const OfficialCardStatePod& card) {
    constexpr std::uint32_t bit = static_cast<std::uint32_t>(
        OfficialEffectTypeId::kCannotMoveDamageCounter)
        - static_cast<std::uint32_t>(OfficialEffectTypeId::kNoAbility);
    return official_continual_flag(card, bit);
}

PTCG_OFFICIAL_EFFECT_HD inline void official_effect_move_targets(
    OfficialStatePod* state,
    OfficialArea destination,
    bool reverse,
    bool with_attachments,
    std::int32_t open_type = 0) {
    for (std::uint16_t index = 0; index < state->targets.count && official_pod_ok(state); ++index) {
        const OfficialAreaRefPod target = state->targets.values[index];
        if (!official_pod_area_ref_valid(state, target)) continue;
        official_pod_mark_changed(state);
        official_pod_move_ref_complete(
            state, target.card, destination, reverse, with_attachments, open_type);
    }
}

PTCG_OFFICIAL_EFFECT_HD inline bool
official_effect_move_targets_to_bench(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    official_effect_move_targets(state, OfficialArea::kBench, false, false);
    for (std::uint16_t index = 0;
         index < state->targets.count && official_pod_ok(state);
         ++index) {
        const OfficialAreaRefPod target = state->targets.values[index];
        OfficialCardStatePod* card = official_pod_card(state, target.card);
        if (card == nullptr
            || card->area != static_cast<std::uint8_t>(OfficialArea::kBench)) {
            continue;
        }
        const OfficialArea from_area = static_cast<OfficialArea>(card->pre_area);
        const std::int32_t active_player =
            ((state->turn + 1) ^ state->first_player) & 1;
        if (card->player == active_player && from_area != OfficialArea::kActive
            && !official_pull_trigger(state, rules, 5, target.card)) {
            return false;
        }
        if (from_area == OfficialArea::kHand
            && !official_pull_trigger(state, rules, 3, target.card)) {
            return false;
        }
        if (from_area == OfficialArea::kActive
            && !official_pull_trigger(state, rules, 4, target.card)) {
            return false;
        }
    }
    return official_pod_ok(state);
}

PTCG_OFFICIAL_EFFECT_HD inline bool official_effect_attach_card(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod attach_ref,
    OfficialCardRefPod target_ref) {
    OfficialCardStatePod* attached = official_pod_card(state, attach_ref);
    OfficialCardStatePod* target = official_pod_card(state, target_ref);
    if (attached == nullptr || target == nullptr) return false;
    const OfficialArea target_area = static_cast<OfficialArea>(target->area);
    if (target_area != OfficialArea::kActive && target_area != OfficialArea::kBench) {
        return false;
    }
    const OfficialCardRule* master = official_card_rule(
        rules, static_cast<std::uint32_t>(attached->card_id));
    if (master == nullptr) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, attached->card_id);
        return false;
    }
    const std::int32_t attach_counter = target->move_counter;
    const std::int32_t player = attached->player;
    const std::int32_t attached_card_id = attached->card_id;
    const std::int32_t target_card_id = target->card_id;
    const OfficialArea destination = master->values[kCardType] == 2
        ? OfficialArea::kTool : OfficialArea::kEnergy;
    official_pod_move_ref(state, attach_ref, destination, false, false);
    attached = official_pod_card(state, attach_ref);
    if (attached != nullptr && official_pod_ok(state)) {
        attached->attach_move_counter = attach_counter;
        official_semantic_history_append(
            state,
            OfficialSemanticLogType::kAttach,
            player,
            attached_card_id,
            attach_ref.index,
            target_card_id,
            target_ref.index);
        if (target_area == OfficialArea::kActive) state->attach_active = 1;
        return true;
    }
    return false;
}

PTCG_OFFICIAL_EFFECT_HD inline void official_effect_log_move_attached(
    OfficialStatePod* state,
    OfficialCardRefPod attached_ref,
    OfficialCardRefPod target_ref) {
    const OfficialCardStatePod* attached = official_pod_card(state, attached_ref);
    const OfficialCardStatePod* target = official_pod_card(state, target_ref);
    if (attached == nullptr || target == nullptr) return;
    OfficialCardRefPod before_ref{};
    official_find_attached_pokemon(*state, *attached, &before_ref);
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kMoveAttached,
        target->player,
        attached->card_id,
        attached_ref.index,
        official_pod_card_id_or_zero(state, before_ref),
        before_ref.index,
        target->card_id,
        target_ref.index);
}

PTCG_OFFICIAL_EFFECT_HD inline bool official_effect_evolve_card(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod evolution_ref,
    OfficialCardRefPod in_play_ref) {
    OfficialCardStatePod* evolution = official_pod_card(state, evolution_ref);
    OfficialCardStatePod* in_play = official_pod_card(state, in_play_ref);
    if (evolution == nullptr || in_play == nullptr
        || evolution->player != in_play->player) return false;
    const OfficialArea in_play_area = static_cast<OfficialArea>(in_play->area);
    if (in_play_area != OfficialArea::kActive && in_play_area != OfficialArea::kBench) {
        return false;
    }
    const OfficialArea evolution_area = static_cast<OfficialArea>(evolution->area);
    const std::int32_t evolution_index = official_pod_find_zone_index(
        state, evolution->player, evolution_area, evolution_ref);
    const std::int32_t in_play_index = official_pod_find_zone_index(
        state, in_play->player, in_play_area, in_play_ref);
    if (evolution_index < 0 || in_play_index < 0) return false;

    const std::int32_t move_counter = in_play->move_counter;
    const std::int32_t damage = in_play->damage;
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kEvolve,
        evolution->player,
        evolution->card_id,
        evolution_ref.index,
        in_play->card_id,
        in_play_ref.index);
    official_pod_remove_zone_card(
        state,
        evolution->player,
        evolution_area,
        static_cast<std::uint16_t>(evolution_index));
    official_pod_push_zone_card(
        state, in_play->player, OfficialArea::kPreEvolution, in_play_ref);
    official_pod_card_moved(state, in_play_ref, OfficialArea::kPreEvolution);
    in_play = official_pod_card(state, in_play_ref);
    in_play->attach_move_counter = move_counter;

    OfficialPlayerStatePod* player = &state->players[evolution->player];
    if (in_play_area == OfficialArea::kActive) {
        player->active.values[in_play_index] = evolution_ref;
        player->active_state = 0;
    } else {
        player->bench.values[in_play_index] = evolution_ref;
    }
    official_pod_card_moved(state, evolution_ref, in_play_area);
    evolution = official_pod_card(state, evolution_ref);
    evolution->move_counter = move_counter;
    evolution->damage = damage;
    official_pod_set_card_runtime_flag(evolution, kCardAppear);
    official_pod_push(
        state,
        &state->turn_evolve,
        OfficialEvolveRecordPod{in_play_ref, evolution_ref},
        OfficialPodError::kZoneOverflow);
    if (!official_pod_ok(state)) return false;
    const OfficialContinualRefreshResult refreshed =
        official_refresh_continual_effects(state, rules);
    if (static_cast<std::int32_t>(refreshed) != 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state,
                OfficialPodError::kInterpreterBudget,
                static_cast<std::int32_t>(refreshed));
        }
        return false;
    }
    if (official_pod_ok(state)
        && evolution_area == OfficialArea::kHand
        && !official_pull_trigger(
            state,
            rules,
            kOfficialTriggerEvolveFromHand,
            evolution_ref)) {
        return false;
    }
    return official_pod_ok(state);
}

PTCG_OFFICIAL_EFFECT_HD inline bool official_effect_devolve_card(
    OfficialStatePod* state,
    OfficialCardRefPod current_ref,
    OfficialArea destination) {
    OfficialCardStatePod* current = official_pod_card(state, current_ref);
    if (current == nullptr || current->player < 0 || current->player > 1) return false;
    const OfficialArea in_play_area = static_cast<OfficialArea>(current->area);
    if (in_play_area != OfficialArea::kActive && in_play_area != OfficialArea::kBench) {
        return false;
    }
    OfficialPlayerStatePod* player = &state->players[current->player];
    std::int32_t pre_index = -1;
    for (std::int32_t index = static_cast<std::int32_t>(player->pre_evolution.count) - 1;
         index >= 0;
         --index) {
        const OfficialCardStatePod* candidate = official_pod_card(
            state, player->pre_evolution.values[index]);
        if (candidate != nullptr
            && candidate->attach_move_counter == current->move_counter) {
            pre_index = index;
            break;
        }
    }
    if (pre_index < 0) return false;
    const OfficialCardRefPod pre_ref = player->pre_evolution.values[pre_index];
    const std::int32_t in_play_index = official_pod_find_zone_index(
        state, current->player, in_play_area, current_ref);
    if (in_play_index < 0) return false;
    const std::int32_t move_counter = current->move_counter;
    const std::int32_t damage = current->damage;
    const std::int32_t player_index = current->player;
    const OfficialCardStatePod* pre_card = official_pod_card(state, pre_ref);
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kDevolve,
        player_index,
        pre_card == nullptr ? 0 : pre_card->card_id,
        pre_ref.index,
        current->card_id,
        current_ref.index);

    official_pod_remove_zone_card(
        state,
        player_index,
        OfficialArea::kPreEvolution,
        static_cast<std::uint16_t>(pre_index));
    official_pod_remove_zone_card(
        state,
        player_index,
        in_play_area,
        static_cast<std::uint16_t>(in_play_index));
    official_pod_push_zone_card(state, player_index, destination, current_ref);
    official_pod_card_moved(state, current_ref, destination);
    official_pod_push_zone_card(state, player_index, in_play_area, pre_ref);
    if (in_play_area == OfficialArea::kBench
        && static_cast<std::uint16_t>(in_play_index) < player->bench.count - 1) {
        const std::uint16_t last = static_cast<std::uint16_t>(player->bench.count - 1);
        const OfficialCardRefPod inserted = player->bench.values[last];
        for (std::uint16_t index = last; index > in_play_index; --index) {
            player->bench.values[index] = player->bench.values[index - 1];
        }
        player->bench.values[in_play_index] = inserted;
    }
    official_pod_card_moved(state, pre_ref, in_play_area);
    OfficialCardStatePod* pre = official_pod_card(state, pre_ref);
    pre->move_counter = move_counter;
    pre->damage = damage;
    official_pod_set_card_runtime_flag(pre, kCardAppear);
    if (in_play_area == OfficialArea::kActive) player->active_state = 0;
    return official_pod_ok(state);
}

PTCG_OFFICIAL_EFFECT_HD inline bool official_effect_transform_card(
    OfficialStatePod* state,
    OfficialCardRefPod before_ref,
    OfficialCardRefPod after_ref,
    OfficialArea destination) {
    OfficialCardStatePod* before = official_pod_card(state, before_ref);
    OfficialCardStatePod* after = official_pod_card(state, after_ref);
    if (before == nullptr || after == nullptr || before->player != after->player) return false;
    const OfficialArea in_play_area = static_cast<OfficialArea>(before->area);
    if (in_play_area != OfficialArea::kActive && in_play_area != OfficialArea::kBench) {
        return false;
    }
    const OfficialArea after_area = static_cast<OfficialArea>(after->area);
    const std::int32_t after_index = official_pod_find_zone_index(
        state, after->player, after_area, after_ref);
    const std::int32_t before_index = official_pod_find_zone_index(
        state, before->player, in_play_area, before_ref);
    if (after_index < 0 || before_index < 0) return false;
    const std::uint32_t active_state = state->players[before->player].active_state;
    const std::int32_t after_card_id = after->card_id;
    official_semantic_history_append(
        state,
        OfficialSemanticLogType::kChange,
        before->player,
        before->card_id,
        before_ref.index,
        after_card_id,
        after_ref.index);
    official_pod_remove_zone_card(
        state, after->player, after_area, static_cast<std::uint16_t>(after_index));
    *after = *before;
    after->card_id = after_card_id;
    after->pre_area = static_cast<std::uint8_t>(after_area);
    after->area = static_cast<std::uint8_t>(in_play_area);
    after->reverse = 0;

    OfficialPlayerStatePod* player = &state->players[before->player];
    if (in_play_area == OfficialArea::kActive) {
        player->active.values[before_index] = after_ref;
    } else {
        player->bench.values[before_index] = after_ref;
    }
    official_pod_push_zone_card(state, before->player, destination, before_ref);
    official_pod_card_moved(state, before_ref, destination);
    player->active_state = active_state;
    for (std::uint16_t index = 0; index < state->triggers.count; ++index) {
        OfficialTriggerInfoPod* trigger = &state->triggers.values[index].trigger;
        if (trigger->subject.card.index == before_ref.index) {
            trigger->subject.card = after_ref;
        }
        if (trigger->object.card.index == before_ref.index) {
            trigger->object.card = after_ref;
        }
    }
    official_pod_mark_changed(state);
    return official_pod_ok(state);
}

PTCG_OFFICIAL_EFFECT_HD inline void official_effect_draw_players(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect,
    std::int32_t mode) {
    const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
    for (int ordinal = 0; ordinal < 2 && official_pod_ok(state); ++ordinal) {
        const int player = official_effect_target_player(*state, mask, ordinal);
        if (player < 0) break;
        std::int32_t count = 0;
        if (mode == 0) count = official_effect_value(*state, effect, 0);
        else if (mode == 1) count = state->targets.count;
        else if (mode == 2) count = state->players[player].prize.count;
        else if (mode == 3) {
            count = official_effect_value(*state, effect, 0) - state->players[player].hand.count;
        } else if (mode == 4) {
            count = state->players[1 - player].hand.count - state->players[player].hand.count;
        }
        if (count > 0) official_pod_draw(state, player, count);
    }
}

PTCG_OFFICIAL_EFFECT_HD inline OfficialEffectApplyResult official_apply_effect_primitive(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect) {
    if (!official_pod_ok(state)) return OfficialEffectApplyResult::kError;
    const auto type = static_cast<OfficialEffectTypeId>(effect.values[kEffectType]);
    switch (type) {
        case OfficialEffectTypeId::kNoEffect:
            return OfficialEffectApplyResult::kApplied;
        case OfficialEffectTypeId::kSelectCard:
        case OfficialEffectTypeId::kNotMove:
        case OfficialEffectTypeId::kSelectEvolvesFrom:
        case OfficialEffectTypeId::kSelectEvolvesTo:
        case OfficialEffectTypeId::kSelectAttachFrom:
        case OfficialEffectTypeId::kSelectAttachTo:
        case OfficialEffectTypeId::kSelectSwitchEnergyCard:
            if ((effect.flags & kEffectNotClearSelectedList) == 0
                && state->effect_state.selected_list_index <= 0) {
                state->selected_list.count = 0;
            }
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                if (!official_pod_area_ref_valid(state, state->targets.values[index])) {
                    continue;
                }
                if (!official_pod_push(
                        state,
                        &state->selected_list,
                        state->targets.values[index].card,
                        OfficialPodError::kSelectionOverflow)) {
                    break;
                }
            }
            break;
        case OfficialEffectTypeId::kForEach:
            state->each_list.count = 0;
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                if (!official_pod_push(
                        state,
                        &state->each_list,
                        state->targets.values[index].card,
                        OfficialPodError::kSelectionOverflow)) break;
            }
            break;
        case OfficialEffectTypeId::kKo:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* card = official_pod_card(state, target.card);
                card->damage = official_pod_max_hp(state, rules, target.card);
                official_pod_set_card_runtime_flag(card, kCardKo);
                official_pod_mark_changed(state);
            }
            break;
        case OfficialEffectTypeId::kToHand:
            official_effect_move_targets(state, OfficialArea::kHand, false, false);
            break;
        case OfficialEffectTypeId::kToHandReverse:
            official_effect_move_targets(state, OfficialArea::kHand, false, false, 1);
            break;
        case OfficialEffectTypeId::kToHandWithAttach:
            official_effect_move_targets(state, OfficialArea::kHand, false, true);
            break;
        case OfficialEffectTypeId::kToTrash:
            official_effect_move_targets(state, OfficialArea::kTrash, false, false);
            break;
        case OfficialEffectTypeId::kToDeck:
            official_effect_move_targets(state, OfficialArea::kDeck, false, false);
            break;
        case OfficialEffectTypeId::kToDeckWithAttach:
            official_effect_move_targets(state, OfficialArea::kDeck, false, true);
            break;
        case OfficialEffectTypeId::kToDeckReverse:
        case OfficialEffectTypeId::kLookToDeckReverse:
            official_effect_move_targets(
                state,
                OfficialArea::kDeck,
                false,
                false,
                type == OfficialEffectTypeId::kLookToDeckReverse
                    ? state->effect_state.ability.use_player + 3
                    : 1);
            break;
        case OfficialEffectTypeId::kToDeckAndShuffle:
        case OfficialEffectTypeId::kToDeckReverseAndShuffle: {
            official_effect_move_targets(
                state,
                OfficialArea::kDeck,
                false,
                false,
                type == OfficialEffectTypeId::kToDeckReverseAndShuffle ? 1 : 0);
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                official_pod_shuffle_deck(state, player);
            }
            break;
        }
        case OfficialEffectTypeId::kToDeckBottom:
            official_effect_move_targets(state, OfficialArea::kDeckBottom, false, false);
            break;
        case OfficialEffectTypeId::kToDeckBottomReverse:
            official_effect_move_targets(state, OfficialArea::kDeckBottom, false, false, 1);
            break;
        case OfficialEffectTypeId::kToDeckBottomClose:
            official_pod_shuffle(
                state->targets.values, state->targets.count, &state->rng);
            official_effect_move_targets(state, OfficialArea::kDeckBottom, false, false, 2);
            break;
        case OfficialEffectTypeId::kToActiveAndTrashActive:
            if (state->targets.count == 1
                && official_pod_area_ref_valid(state, state->targets.values[0])) {
                const OfficialCardRefPod ref = state->targets.values[0].card;
                OfficialCardStatePod* card = official_pod_card(state, ref);
                const OfficialCardRule* master = card == nullptr ? nullptr
                    : official_card_rule(
                        rules, static_cast<std::uint32_t>(card->card_id));
                if (card != nullptr && master != nullptr
                    && (master->flags & (1ULL << 4U)) == 0) {
                    const int player = card->player;
                    if (state->players[player].active.count > 0) {
                        official_pod_move_card(
                            state,
                            player,
                            OfficialArea::kActive,
                            0,
                            OfficialArea::kTrash);
                    }
                    official_pod_move_ref(state, ref, OfficialArea::kActive);
                }
            }
            break;
        case OfficialEffectTypeId::kToBench:
            if (!official_effect_move_targets_to_bench(state, rules)) {
                return OfficialEffectApplyResult::kError;
            }
            break;
        case OfficialEffectTypeId::kToPrize:
            official_effect_move_targets(state, OfficialArea::kPrize, true, false);
            break;
        case OfficialEffectTypeId::kToLooking:
            state->looking_player = (effect.flags & kEffectOpen) != 0
                ? 2
                : state->effect_state.ability.use_player;
            official_effect_move_targets(state, OfficialArea::kLooking, false, false);
            break;
        case OfficialEffectTypeId::kToPlayingFirst:
            if (state->targets.count > 0
                && official_pod_area_ref_valid(state, state->targets.values[0])) {
                official_pod_move_ref_complete(
                    state,
                    state->targets.values[0].card,
                    OfficialArea::kPlaying,
                    false,
                    false);
            }
            break;
        case OfficialEffectTypeId::kSwitch:
            // Switch needs continual refresh and movement triggers, which are
            // applied by official_apply_switch_effect in the interpreter.
            return OfficialEffectApplyResult::kUnsupported;
        case OfficialEffectTypeId::kSwitchDeck:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* card = official_pod_card(state, target.card);
                if (card == nullptr || state->players[card->player].deck.count == 0) continue;
                official_pod_mark_changed(state);
                official_pod_draw(state, card->player, 1);
                official_pod_move_ref(state, target.card, OfficialArea::kDeck, false);
            }
            break;
        case OfficialEffectTypeId::kLookDeck:
        case OfficialEffectTypeId::kLookDeckReverse:
        case OfficialEffectTypeId::kLookDeckBottom: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            const int player = official_effect_target_player(*state, mask, 0);
            if (player >= 0) {
                state->looking_player = (effect.flags & kEffectOpen) != 0
                    ? 2
                    : state->effect_state.ability.use_player;
                if (type == OfficialEffectTypeId::kLookDeckReverse) {
                    state->looking_player += 3;
                }
                const std::int32_t requested = official_effect_value(*state, effect, 0);
                for (std::int32_t moved = 0;
                     moved < requested && state->players[player].deck.count > 0;
                      ++moved) {
                    official_pod_mark_changed(state);
                    const std::uint16_t deck_index = type
                        == OfficialEffectTypeId::kLookDeckBottom
                        ? 0
                        : static_cast<std::uint16_t>(state->players[player].deck.count - 1);
                    const std::int32_t open_type =
                        type == OfficialEffectTypeId::kLookDeckReverse
                        ? 2
                        : (state->looking_player == 2
                            ? 0
                            : static_cast<std::int32_t>(state->looking_player) + 3);
                    official_pod_move_card(
                        state,
                        player,
                        OfficialArea::kDeck,
                        deck_index,
                        OfficialArea::kLooking,
                        type == OfficialEffectTypeId::kLookDeckReverse,
                        true,
                        open_type);
                }
            }
            break;
        }
        case OfficialEffectTypeId::kLookAndReturn:
            if (official_effect_value(*state, effect, 0) == 1) {
                for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                    const OfficialAreaRefPod target = state->targets.values[index];
                    if (official_pod_area_ref_valid(state, target)) {
                        official_pod_card(state, target.card)->reverse = 0;
                    }
                }
            }
            break;
        case OfficialEffectTypeId::kDamageCounter:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)
                    && !official_effect_blocks_damage_counter(
                        state, rules, target.card)) {
                    const std::int32_t damage =
                        official_effect_value(*state, effect, 0) * 10
                        + state->effect_state.damage_change;
                    if (damage > 0) official_pod_mark_changed(state);
                    official_pod_add_damage(
                        state,
                        rules,
                        target.card,
                        damage,
                        true);
                }
            }
            break;
        case OfficialEffectTypeId::kDamageCounterRemoved:
        case OfficialEffectTypeId::kDamageCounterDamaged:
        case OfficialEffectTypeId::kDamageCounterDouble:
        case OfficialEffectTypeId::kDamageCounterHp:
        case OfficialEffectTypeId::kDamageCounterTypeEnergyCountMe:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* card = official_pod_card(state, target.card);
                std::int32_t damage = 0;
                if (type == OfficialEffectTypeId::kDamageCounterRemoved) {
                    damage = state->removed_damage_counter * 10;
                } else if (type == OfficialEffectTypeId::kDamageCounterDamaged) {
                    damage = state->trigger_info.value;
                } else if (type == OfficialEffectTypeId::kDamageCounterDouble) {
                    damage = card->damage;
                } else if (type == OfficialEffectTypeId::kDamageCounterHp) {
                    damage = official_pod_max_hp(state, rules, target.card)
                        - official_effect_value(*state, effect, 0);
                } else {
                    const OfficialCardRefPod effect_card =
                        state->effect_state.ability.effect_card.card;
                    damage = official_attached_energy_cards(
                        *state,
                        effect_card,
                        official_effect_value(*state, effect, 0),
                        false,
                        rules)
                        * official_effect_value(*state, effect, 1) * 10;
                }
                if (!official_effect_blocks_damage_counter(
                        state, rules, target.card)) {
                    if (damage > 0) official_pod_mark_changed(state);
                    official_pod_add_damage(
                        state, rules, target.card, damage, true);
                }
            }
            break;
        case OfficialEffectTypeId::kAttackDamage:
            for (std::int32_t index = static_cast<std::int32_t>(state->targets.count) - 1;
                 index >= 0;
                 --index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)) {
                    // Official EffectAttackDamage marks the effect as changed
                    // immediately after CalcDamage, even when prevention (for
                    // example, a Benched Tera Pokemon) reduces the applied
                    // damage to zero.  This differs from the ordinary attack
                    // body, which does not touch State::changed.
                    official_pod_mark_changed(state);
                    official_apply_attack_damage_to_card(
                        state,
                        rules,
                        target.card,
                        official_effect_value(*state, effect, 0)
                            + state->effect_state.damage_change);
                }
            }
            break;
        case OfficialEffectTypeId::kAttackDamageCoin:
            for (std::int32_t index = static_cast<std::int32_t>(state->targets.count) - 1;
                 index >= 0;
                 --index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)
                    && official_effect_roll_coins(state, 1) > 0) {
                    official_pod_mark_changed(state);
                    official_apply_attack_damage_to_card(
                        state,
                        rules,
                        target.card,
                        official_effect_value(*state, effect, 0)
                            + state->effect_state.damage_change);
                }
            }
            break;
        case OfficialEffectTypeId::kRemoveDamageCounterAll: {
            bool changed = false;
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* card = official_pod_card(state, target.card);
                if (card == nullptr) continue;
                if (official_effect_blocks_damage_counter(state, rules, target.card)
                    || official_card_cannot_move_damage_counter(*card)) {
                    continue;
                }
                state->removed_damage_counter = card->damage / 10;
                if (state->removed_damage_counter <= 0) continue;
                changed = true;
                official_pod_heal(state, target.card, card->damage);
            }
            if (!changed) state->control_flags |= kOfficialBreakEffectFlag;
            break;
        }
        case OfficialEffectTypeId::kHeal:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)) {
                    official_pod_heal(
                        state,
                        target.card,
                        official_effect_value(*state, effect, 0),
                        true);
                }
            }
            break;
        case OfficialEffectTypeId::kHealAll:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)) {
                    OfficialCardStatePod* card = official_pod_card(state, target.card);
                    official_pod_heal(state, target.card, card->damage, true);
                }
            }
            break;
        case OfficialEffectTypeId::kHealSand:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                const OfficialCardStatePod* card = official_pod_card(state, target.card);
                const OfficialCardRule* master = official_card_rule(
                    rules, static_cast<std::uint32_t>(card->card_id));
                const std::int32_t heal = master != nullptr
                    && (master->flags & (1ULL << 20U)) != 0 ? 100 : 30;
                official_pod_heal(state, target.card, heal, true);
            }
            break;
        case OfficialEffectTypeId::kResetHp:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* card = official_pod_card(state, target.card);
                const std::int32_t desired_damage = official_pod_max_hp(
                    state, rules, target.card) - official_effect_value(*state, effect, 0);
                if (desired_damage < card->damage) {
                    card->damage = desired_damage > 0 ? desired_damage : 0;
                    card->runtime_flags &= ~static_cast<std::uint64_t>(kCardKo);
                    official_pod_mark_changed(state);
                }
            }
            break;
        case OfficialEffectTypeId::kDrain:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)) {
                    official_pod_heal(
                        state,
                        target.card,
                        state->last_attack_damage,
                        true);
                }
            }
            break;
        case OfficialEffectTypeId::kDevolve:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)) {
                    official_effect_devolve_card(
                        state,
                        target.card,
                        static_cast<OfficialArea>(
                            official_effect_value(*state, effect, 0)));
                }
            }
            break;
        case OfficialEffectTypeId::kDevolveAny:
            if (state->targets.count > 0
                && official_pod_area_ref_valid(state, state->targets.values[0])) {
                const OfficialCardRefPod target = state->targets.values[0].card;
                OfficialCardStatePod* before = official_pod_card(state, target);
                const std::int32_t player_index = before->player;
                const OfficialArea area = static_cast<OfficialArea>(before->area);
                const std::int32_t area_index = official_pod_find_zone_index(
                    state, player_index, area, target);
                official_effect_devolve_card(state, target, OfficialArea::kHand);
                OfficialCardRefPod current_ref{};
                if (area_index >= 0) {
                    current_ref = area == OfficialArea::kActive
                        ? state->players[player_index].active.values[area_index]
                        : state->players[player_index].bench.values[area_index];
                }
                OfficialCardStatePod* current = official_pod_card(state, current_ref);
                if (current != nullptr) {
                    const OfficialPlayerStatePod& player = state->players[current->player];
                    for (std::uint16_t index = 0;
                         index < player.pre_evolution.count;
                         ++index) {
                        const OfficialCardStatePod* pre = official_pod_card(
                            state, player.pre_evolution.values[index]);
                        if (pre != nullptr
                            && pre->attach_move_counter == current->move_counter) {
                            return OfficialEffectApplyResult::kNeedsSelection;
                        }
                    }
                }
            }
            break;
        case OfficialEffectTypeId::kTransformDeck:
        case OfficialEffectTypeId::kTransformTrash:
            if (state->selected_list.count > 0) {
                for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                    const OfficialAreaRefPod target = state->targets.values[index];
                    if (official_pod_area_ref_valid(state, target)) {
                        official_effect_transform_card(
                            state,
                            target.card,
                            state->selected_list.values[0],
                            type == OfficialEffectTypeId::kTransformDeck
                                ? OfficialArea::kDeck : OfficialArea::kTrash);
                        break;
                    }
                }
            }
            break;
        case OfficialEffectTypeId::kExchangeSelected:
            if (state->check_list.count > 0 && state->targets.count > 0
                && official_pod_area_ref_valid(state, state->targets.values[0])) {
                const OfficialCardRefPod prize_ref = state->check_list.values[0];
                const OfficialCardRefPod hand_ref = state->targets.values[0].card;
                OfficialCardStatePod* prize = official_pod_card(state, prize_ref);
                OfficialCardStatePod* hand = official_pod_card(state, hand_ref);
                if (prize != nullptr && hand != nullptr) {
                    const std::int32_t prize_index = official_pod_find_zone_index(
                        state, prize->player, OfficialArea::kPrize, prize_ref);
                    const std::int32_t hand_index = official_pod_find_zone_index(
                        state, hand->player, OfficialArea::kHand, hand_ref);
                    if (prize_index >= 0 && hand_index >= 0) {
                        state->players[prize->player].prize.values[prize_index] = hand_ref;
                        state->players[hand->player].hand.values[hand_index] = prize_ref;
                        official_pod_card_moved(state, prize_ref, OfficialArea::kHand);
                        official_pod_card_moved(state, hand_ref, OfficialArea::kPrize);
                        official_pod_card(state, prize_ref)->reverse = 0;
                        official_pod_card(state, hand_ref)->reverse = 0;
                    }
                }
            }
            break;
        case OfficialEffectTypeId::kKoPrizeChangeAlways:
        case OfficialEffectTypeId::kKoPrizeChange:
        case OfficialEffectTypeId::kKoPrizeDecreaseOnce:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* card = official_pod_card(state, target.card);
                if (type == OfficialEffectTypeId::kKoPrizeChangeAlways) {
                    card->ko_prize_change_always += static_cast<std::int16_t>(
                        official_effect_value(*state, effect, 0));
                } else if (type == OfficialEffectTypeId::kKoPrizeChange) {
                    card->ko_prize_change += static_cast<std::int16_t>(
                        official_effect_value(*state, effect, 0));
                } else if (!state->players[card->player].ko_prize_once_changed) {
                    state->players[card->player].ko_prize_once_changed = 1;
                    official_pod_set_card_runtime_flag(
                        card, kCardKoPrizeDecreaseOnce);
                }
            }
            break;
        case OfficialEffectTypeId::kEvolvesToEach:
            if (state->effect_state.selected_list_index >= 0
                && static_cast<std::uint16_t>(state->effect_state.selected_list_index)
                    < state->selected_list.count) {
                const OfficialCardRefPod in_play = state->selected_list.values[
                    state->effect_state.selected_list_index];
                for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                    const OfficialAreaRefPod evolution = state->targets.values[index];
                    if (official_pod_area_ref_valid(state, evolution)) {
                        if (official_effect_evolve_card(
                                state, rules, evolution.card, in_play)) {
                            official_pod_mark_changed(state);
                        }
                    }
                }
            }
            break;
        case OfficialEffectTypeId::kEvolvesFromEach:
            if (state->effect_state.selected_list_index >= 0
                && static_cast<std::uint16_t>(state->effect_state.selected_list_index)
                    < state->selected_list.count) {
                const OfficialCardRefPod evolution = state->selected_list.values[
                    state->effect_state.selected_list_index];
                for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                    const OfficialAreaRefPod in_play = state->targets.values[index];
                    if (official_pod_area_ref_valid(state, in_play)) {
                        if (official_effect_evolve_card(
                                state, rules, evolution, in_play.card)) {
                            official_pod_mark_changed(state);
                        }
                        break;
                    }
                }
            }
            break;
        case OfficialEffectTypeId::kAttachToEach:
            if (state->effect_state.selected_list_index >= 0
                && static_cast<std::uint16_t>(state->effect_state.selected_list_index)
                    < state->selected_list.count) {
                const OfficialCardRefPod target = state->selected_list.values[
                    state->effect_state.selected_list_index];
                for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                    const OfficialAreaRefPod attached = state->targets.values[index];
                    if (official_pod_area_ref_valid(state, attached)) {
                        if (official_effect_attach_card(
                                state, rules, attached.card, target)) {
                            official_pod_mark_changed(state);
                        }
                    }
                }
            }
            break;
        case OfficialEffectTypeId::kAttachEnergyMe: {
            const OfficialCardRefPod target = state->effect_state.ability.effect_card.card;
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod attached = state->targets.values[index];
                if (official_pod_area_ref_valid(state, attached)) {
                    if (official_effect_attach_card(
                            state, rules, attached.card, target)) {
                        official_pod_mark_changed(state);
                    }
                }
            }
            break;
        }
        case OfficialEffectTypeId::kAttachSelectedCard:
            for (std::uint16_t target_index = 0;
                 target_index < state->targets.count;
                 ++target_index) {
                const OfficialAreaRefPod target = state->targets.values[target_index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                for (std::uint16_t index = 0; index < state->selected_list.count; ++index) {
                    if (official_effect_attach_card(
                            state,
                            rules,
                            state->selected_list.values[index],
                            target.card)) {
                        official_pod_mark_changed(state);
                    }
                }
                break;
            }
            break;
        case OfficialEffectTypeId::kSwitchSelectedCard:
            for (std::uint16_t target_index = 0;
                 target_index < state->targets.count;
                 ++target_index) {
                const OfficialAreaRefPod target = state->targets.values[target_index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* target_card = official_pod_card(state, target.card);
                for (std::uint16_t index = 0; index < state->selected_list.count; ++index) {
                    OfficialCardStatePod* energy = official_pod_card(
                        state, state->selected_list.values[index]);
                    if (energy != nullptr) {
                        official_effect_log_move_attached(
                            state,
                            state->selected_list.values[index],
                            target.card);
                        energy->attach_move_counter = target_card->move_counter;
                        official_pod_mark_changed(state);
                    }
                }
                break;
            }
            break;
        case OfficialEffectTypeId::kAttachFromEach:
        case OfficialEffectTypeId::kEnergySwitchEach:
            if (state->effect_state.selected_list_index >= 0
                && static_cast<std::uint16_t>(state->effect_state.selected_list_index)
                    < state->selected_list.count) {
                const OfficialCardRefPod attached = state->selected_list.values[
                    state->effect_state.selected_list_index];
                for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                    const OfficialAreaRefPod target = state->targets.values[index];
                    if (!official_pod_area_ref_valid(state, target)) continue;
                    if (type == OfficialEffectTypeId::kAttachFromEach) {
                        if (official_effect_attach_card(
                                state, rules, attached, target.card)) {
                            official_pod_mark_changed(state);
                        }
                    } else {
                        OfficialCardStatePod* energy = official_pod_card(state, attached);
                        OfficialCardStatePod* target_card = official_pod_card(state, target.card);
                        if (energy != nullptr && target_card != nullptr) {
                            official_effect_log_move_attached(
                                state, attached, target.card);
                            energy->attach_move_counter = target_card->move_counter;
                            official_pod_mark_changed(state);
                        }
                    }
                }
            }
            break;
        case OfficialEffectTypeId::kDelayEffect: {
            std::int32_t owner_card_id = 0;
            if (state->current_attack_id != 0) {
                const OfficialAttackRule* attack = official_attack_rule(
                    rules, static_cast<std::uint32_t>(state->current_attack_id));
                if (attack != nullptr) owner_card_id = attack->values[kAttackCardId];
            }
            if (owner_card_id == 0) {
                const OfficialCardStatePod* effect_card = official_pod_card(
                    state, state->effect_state.ability.effect_card.card);
                if (effect_card != nullptr) owner_card_id = effect_card->card_id;
            }
            const OfficialCardRule* owner = official_card_rule(
                rules, static_cast<std::uint32_t>(owner_card_id));
            const OfficialSkillRule* delay = owner == nullptr ? nullptr
                : official_skill_rule(
                    rules, static_cast<std::uint32_t>(owner->values[kCardDelayId]));
            if (delay == nullptr || delay->values[kSkillTriggerCount] <= 0) {
                official_pod_fail(state, OfficialPodError::kRulePackBounds, owner_card_id);
                break;
            }
            const std::int32_t trigger_offset = delay->values[kSkillTriggerOffset];
            const OfficialTriggerRule& trigger_rule = rules.triggers[trigger_offset];
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialTriggeredAbilityPod delayed{};
                delayed.activate.skill_id = delay->values[kSkillId];
                delayed.activate.effect_card = state->effect_state.ability.effect_card;
                delayed.activate.use_player = state->effect_state.ability.use_player;
                delayed.trigger.type = static_cast<std::uint8_t>(
                    trigger_rule.type);
                delayed.trigger.subject = target;
                official_pod_push(
                    state,
                    &state->delay_triggers,
                    delayed,
                    OfficialPodError::kTriggerStackOverflow);
            }
            break;
        }
        case OfficialEffectTypeId::kCoin: {
            official_effect_roll_coins(
                state, official_effect_value(*state, effect, 0));
            break;
        }
        case OfficialEffectTypeId::kCoinUntilTail: {
            official_effect_roll_until_tail(state);
            if (official_effect_value(*state, effect, 0) == 1
                && state->coin_head_count == 0) {
                state->control_flags |= kOfficialBreakEffectFlag;
            }
            break;
        }
        case OfficialEffectTypeId::kAttackDamageChange:
            state->attack_damage_change = official_effect_value(*state, effect, 0);
            break;
        case OfficialEffectTypeId::kAttackDamageChangeTargetCount:
            state->attack_damage_change = official_effect_value(*state, effect, 0)
                * state->targets.count;
            break;
        case OfficialEffectTypeId::kEffectDamageChangeTargetCount:
            state->effect_state.damage_change = official_effect_value(*state, effect, 0)
                * state->targets.count;
            break;
        case OfficialEffectTypeId::kAttackDamageChangeEnergyCount:
            state->attack_damage_change = official_effect_value(*state, effect, 0)
                * official_effect_target_energy_count(*state, rules);
            break;
        case OfficialEffectTypeId::kEffectDamageChangeEnergyCount:
            state->effect_state.damage_change = official_effect_value(*state, effect, 0)
                * official_effect_target_energy_count(*state, rules);
            break;
        case OfficialEffectTypeId::kAttackDamageChangeTypeEnergyCount:
            state->attack_damage_change = official_effect_value(*state, effect, 1)
                * official_effect_target_energy_count(
                    *state, rules, official_effect_value(*state, effect, 0));
            break;
        case OfficialEffectTypeId::kEffectDamageChangeTypeEnergyCount:
            state->effect_state.damage_change = official_effect_value(*state, effect, 1)
                * official_effect_target_energy_count(
                    *state, rules, official_effect_value(*state, effect, 0));
            break;
        case OfficialEffectTypeId::kAttackDamageChangeEnergyCountCoin:
            state->attack_damage_change = official_effect_value(*state, effect, 0)
                * official_effect_roll_coins(
                    state, official_effect_target_energy_count(*state, rules));
            break;
        case OfficialEffectTypeId::kAttackDamageChangeTypeEnergyCountCoin:
            state->attack_damage_change = official_effect_value(*state, effect, 1)
                * official_effect_roll_coins(
                    state,
                    official_effect_target_energy_count(
                        *state, rules, official_effect_value(*state, effect, 0)));
            break;
        case OfficialEffectTypeId::kAttackDamageChangeCoin:
            state->attack_damage_change = official_effect_value(*state, effect, 1)
                * official_effect_roll_coins(
                    state, official_effect_value(*state, effect, 0));
            break;
        case OfficialEffectTypeId::kAttackDamageChangeCoinUntilTail:
            state->attack_damage_change = official_effect_value(*state, effect, 0)
                * official_effect_roll_until_tail(state);
            break;
        case OfficialEffectTypeId::kAttackDamageChangeTargetCountCoin:
            state->attack_damage_change = official_effect_value(*state, effect, 0)
                * official_effect_roll_coins(state, state->targets.count);
            break;
        case OfficialEffectTypeId::kAttackDamageChangeTargetCountEnemyCoin: {
            const std::int32_t target_count = state->targets.count;
            state->attack_damage_change = official_effect_value(*state, effect, 0)
                * (target_count - official_effect_roll_coins(state, target_count));
            break;
        }
        case OfficialEffectTypeId::kAttackDamageChangeTakenPrize:
        case OfficialEffectTypeId::kEffectDamageChangeTakenPrize: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            std::int32_t count = 0;
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                count += kOfficialInitialPrizeCount - state->players[player].prize.count;
            }
            const std::int32_t value = official_effect_value(*state, effect, 0) * count;
            if (type == OfficialEffectTypeId::kAttackDamageChangeTakenPrize) {
                state->attack_damage_change = value;
            } else {
                state->effect_state.damage_change = value;
            }
            break;
        }
        case OfficialEffectTypeId::kAttackDamageChangeDamageCounter:
        case OfficialEffectTypeId::kEffectDamageChangeDamageCounter: {
            std::int32_t value = 0;
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                value += official_effect_value(*state, effect, 0)
                    * official_pod_card(state, target.card)->damage / 10;
            }
            if (type == OfficialEffectTypeId::kAttackDamageChangeDamageCounter) {
                state->attack_damage_change = value;
            } else {
                state->effect_state.damage_change = value;
            }
            break;
        }
        case OfficialEffectTypeId::kAttackDamageChangeRetreatCost: {
            state->attack_damage_change = 0;
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                const OfficialCardStatePod* card = official_pod_card(state, target.card);
                const OfficialCardRule* master = official_card_rule(
                    rules, static_cast<std::uint32_t>(card->card_id));
                if (master == nullptr) continue;
                std::int32_t cost = master->values[kCardRetreatCost]
                    + official_continual_i8(*card, 26)
                    + official_this_turn_i8(*card, 11);
                if (official_continual_flag(*card, 16)
                    || master->values[kCardPokemonType] == 2) cost = 0;
                if (cost < 0) cost = 0;
                state->attack_damage_change += official_effect_value(*state, effect, 0)
                    * cost;
            }
            break;
        }
        case OfficialEffectTypeId::kAttackDamageChangeTypeCount: {
            std::uint32_t energy_types = 0;
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                const OfficialCardStatePod* card = official_pod_card(state, target.card);
                const OfficialCardRule* master = official_card_rule(
                    rules, static_cast<std::uint32_t>(card->card_id));
                if (master != nullptr) {
                    energy_types |= static_cast<std::uint32_t>(master->values[kCardEnergyType]);
                }
            }
            state->attack_damage_change = official_effect_value(*state, effect, 0)
                * official_popcount_u32(energy_types);
            break;
        }
        case OfficialEffectTypeId::kAttackDamageChangeSpecialConditionCount: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            std::int32_t count = 0;
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                const OfficialPlayerStatePod& ps = state->players[player];
                count += official_pod_poison_counter(ps) > 0 ? 1 : 0;
                count += official_pod_burned(ps) ? 1 : 0;
                count += official_pod_bad_status(ps) != OfficialBadStatus::kNone ? 1 : 0;
            }
            state->attack_damage_change = official_effect_value(*state, effect, 0) * count;
            break;
        }
        case OfficialEffectTypeId::kAttackDamageChangeTakeAttackDamagePreTurn:
            state->attack_damage_change = 0;
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)) {
                    state->attack_damage_change += official_pod_card(
                        state, target.card)->take_attack_damage_pre_turn;
                }
            }
            break;
        case OfficialEffectTypeId::kAttackDamageChangePreTurnTakePrizeCount:
            state->attack_damage_change = official_effect_value(*state, effect, 0)
                * state->turn_histories[1].take_prize_count;
            break;
        case OfficialEffectTypeId::kBurn:
        case OfficialEffectTypeId::kPoison:
        case OfficialEffectTypeId::kPoison8:
        case OfficialEffectTypeId::kPoison16:
        case OfficialEffectTypeId::kSleep: {
            int player = -1;
            if (state->targets.count > 0
                && official_pod_area_ref_valid(state, state->targets.values[0])) {
                player = official_pod_card(state, state->targets.values[0].card)->player;
            } else {
                const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
                player = official_effect_target_player(*state, mask, 0);
            }
            if (player >= 0
                && !official_effect_blocks_active_effect(state, rules, player)) {
                if (type == OfficialEffectTypeId::kBurn) official_pod_burn(state, player);
                else if (type == OfficialEffectTypeId::kPoison) official_pod_poison(state, player);
                else if (type == OfficialEffectTypeId::kPoison8) official_pod_poison(state, player, 8);
                else if (type == OfficialEffectTypeId::kPoison16) official_pod_poison(state, player, 16);
                else official_pod_set_status(state, player, OfficialBadStatus::kAsleep);
            }
            break;
        }
        case OfficialEffectTypeId::kConfuse:
        case OfficialEffectTypeId::kParalyze:
        case OfficialEffectTypeId::kRecoverSpecialCondition: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                if (type == OfficialEffectTypeId::kConfuse) {
                    if (!official_effect_blocks_active_effect(state, rules, player)) {
                        official_pod_set_status(
                            state, player, OfficialBadStatus::kConfused);
                    }
                } else if (type == OfficialEffectTypeId::kParalyze) {
                    if (!official_effect_blocks_active_effect(state, rules, player)) {
                        official_pod_set_status(
                            state, player, OfficialBadStatus::kParalyzed);
                    }
                } else {
                    official_pod_clear_special_conditions(state, player);
                }
            }
            break;
        }
        case OfficialEffectTypeId::kDraw:
            official_effect_draw_players(state, rules, effect, 0);
            break;
        case OfficialEffectTypeId::kDrawTargetCount:
            official_effect_draw_players(state, rules, effect, 1);
            break;
        case OfficialEffectTypeId::kDrawPrizeCount:
            official_effect_draw_players(state, rules, effect, 2);
            break;
        case OfficialEffectTypeId::kDrawUntil:
            official_effect_draw_players(state, rules, effect, 3);
            break;
        case OfficialEffectTypeId::kDrawUntilPsychic: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                const OfficialPlayerStatePod& ps = state->players[player];
                std::int32_t psychic_count = 0;
                for (int zone = 0; zone < 2; ++zone) {
                    const OfficialCardRefPod* values = zone == 0
                        ? ps.active.values : ps.bench.values;
                    const std::uint16_t count = zone == 0
                        ? ps.active.count : ps.bench.count;
                    for (std::uint16_t index = 0; index < count; ++index) {
                        const OfficialCardStatePod* card = official_pod_card(
                            state, values[index]);
                        const OfficialCardRule* master = card == nullptr ? nullptr
                            : official_card_rule(
                                rules, static_cast<std::uint32_t>(card->card_id));
                        if (master != nullptr
                            && (official_effective_pokemon_type(*card, *master) & 16) != 0) {
                            ++psychic_count;
                        }
                    }
                }
                const std::int32_t draw_count = psychic_count - ps.hand.count;
                if (draw_count > 0) official_pod_draw(state, player, draw_count);
            }
            break;
        }
        case OfficialEffectTypeId::kDrawMirror:
            official_effect_draw_players(state, rules, effect, 4);
            break;
        case OfficialEffectTypeId::kDeckToTrash:
        case OfficialEffectTypeId::kDeckBottomToTrash:
        case OfficialEffectTypeId::kDeckToPrize: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                const int count = official_effect_value(*state, effect, 0);
                for (int index = 0;
                     index < count && state->players[player].deck.count > 0;
                     ++index) {
                    const std::uint16_t deck_index = type == OfficialEffectTypeId::kDeckBottomToTrash
                        ? 0
                        : static_cast<std::uint16_t>(state->players[player].deck.count - 1);
                    const OfficialArea destination = type == OfficialEffectTypeId::kDeckToPrize
                        ? OfficialArea::kPrize
                        : OfficialArea::kTrash;
                    official_pod_move_card(
                        state,
                        player,
                        OfficialArea::kDeck,
                        deck_index,
                        destination,
                        destination == OfficialArea::kPrize);
                }
            }
            break;
        }
        case OfficialEffectTypeId::kDeckToTrashCoinUntilTail: {
            const std::int32_t count = official_effect_roll_until_tail(state);
            state->targets.count = 0;
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                for (std::int32_t index = 0;
                     index < count && state->players[player].deck.count > 0;
                     ++index) {
                    const OfficialCardRefPod moved = official_pod_move_card(
                        state,
                        player,
                        OfficialArea::kDeck,
                        static_cast<std::uint16_t>(state->players[player].deck.count - 1),
                        OfficialArea::kTrash);
                    official_pod_push(
                        state,
                        &state->targets,
                        official_pod_area_ref(state, moved),
                        OfficialPodError::kSelectionOverflow);
                }
            }
            break;
        }
        case OfficialEffectTypeId::kShuffle: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                official_pod_shuffle_deck(state, player);
            }
            break;
        }
        case OfficialEffectTypeId::kEffectWin: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            const int player = official_effect_target_player(*state, mask, 0);
            if (player >= 0) {
                official_pod_set_result(state, 1 - player, OfficialFinishReason::kEffect);
                official_pod_mark_changed(state);
            }
            break;
        }
        case OfficialEffectTypeId::kFailRetreat:
            state->control_flags |= 1U << 3;
            break;
        case OfficialEffectTypeId::kTurnEnd:
            state->turn_state |= kOfficialTurnEndFlag;
            official_pod_mark_changed(state);
            break;
        case OfficialEffectTypeId::kDamageChangeThisTurn:
        case OfficialEffectTypeId::kDamageChangeExThisTurn:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* card = official_pod_card(state, target.card);
                official_add_packed_i16(
                    card->turn_state,
                    type == OfficialEffectTypeId::kDamageChangeThisTurn ? 0 : 2,
                    official_effect_value(*state, effect, 0));
            }
            break;
        case OfficialEffectTypeId::kPlayerDamageChange:
        case OfficialEffectTypeId::kPlayerDamageChangeEx:
        case OfficialEffectTypeId::kPlayerDamageChangeMyFighting: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            const std::uint32_t offset = type == OfficialEffectTypeId::kPlayerDamageChange
                ? 0U
                : (type == OfficialEffectTypeId::kPlayerDamageChangeEx ? 2U : 4U);
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                auto* words = reinterpret_cast<std::uint32_t*>(
                    &state->players[player].turn_state);
                official_add_packed_i16(
                    words, offset, official_effect_value(*state, effect, 0));
            }
            break;
        }
        case OfficialEffectTypeId::kTakePrizeCountChangeTerastalAttackKoActive:
        case OfficialEffectTypeId::kTakePrizeCountChangeNAttackKoActive: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            const std::uint32_t offset = type
                == OfficialEffectTypeId::kTakePrizeCountChangeTerastalAttackKoActive
                ? 6U : 7U;
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                auto* words = reinterpret_cast<std::uint32_t*>(
                    &state->players[player].turn_state);
                official_add_packed_i8(
                    words, offset, official_effect_value(*state, effect, 0));
            }
            break;
        }
        case OfficialEffectTypeId::kTakeDamageChangeNextEnemyTurn:
        case OfficialEffectTypeId::kNoDamageLessEqualAttackNextEnemyTurn:
        case OfficialEffectTypeId::kNoDamageAndEffectAttackNextEnemyTurn:
        case OfficialEffectTypeId::kNoDamageAndEffectEnemyAttackNextEnemyTurn:
        case OfficialEffectTypeId::kNoDamageAndEffectEnemyExAttackNextEnemyTurn:
        case OfficialEffectTypeId::kNoDamageAttackNextEnemyTurn:
        case OfficialEffectTypeId::kNoDamageBasicAttackNextEnemyTurn:
        case OfficialEffectTypeId::kNoDamageBasicColorAttackNextEnemyTurn:
        case OfficialEffectTypeId::kNoDamageAbilityAttackNextEnemyTurn:
        case OfficialEffectTypeId::kNoWeaknessNextEnemyTurn:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* card = official_pod_card(state, target.card);
                if (type == OfficialEffectTypeId::kTakeDamageChangeNextEnemyTurn) {
                    const std::int16_t before = static_cast<std::int16_t>(
                        card->next_enemy_turn_end & 0xffffU);
                    card->next_enemy_turn_end = (card->next_enemy_turn_end & 0xffff0000U)
                        | static_cast<std::uint16_t>(
                            before + official_effect_value(*state, effect, 0));
                } else if (type == OfficialEffectTypeId::kNoDamageLessEqualAttackNextEnemyTurn) {
                    const std::uint32_t value = static_cast<std::uint8_t>(
                        official_effect_value(*state, effect, 0));
                    const std::uint32_t before = (card->next_enemy_turn_end >> 16U) & 0xffU;
                    if (before < value) {
                        card->next_enemy_turn_end = (card->next_enemy_turn_end & 0xff00ffffU)
                            | (value << 16U);
                    }
                } else if (type
                    == OfficialEffectTypeId::kNoDamageAndEffectEnemyExAttackNextEnemyTurn) {
                    card->next_enemy_turn_end_battlefield |= 1U;
                } else {
                    std::uint32_t bit = 0;
                    if (type == OfficialEffectTypeId::kNoDamageAndEffectAttackNextEnemyTurn) bit = 24;
                    else if (type == OfficialEffectTypeId::kNoDamageAndEffectEnemyAttackNextEnemyTurn) bit = 25;
                    else if (type == OfficialEffectTypeId::kNoDamageAttackNextEnemyTurn) bit = 26;
                    else if (type == OfficialEffectTypeId::kNoDamageBasicAttackNextEnemyTurn) bit = 27;
                    else if (type == OfficialEffectTypeId::kNoDamageBasicColorAttackNextEnemyTurn) bit = 28;
                    else if (type == OfficialEffectTypeId::kNoDamageAbilityAttackNextEnemyTurn) bit = 29;
                    else bit = 30;
                    card->next_enemy_turn_end |= 1U << bit;
                }
            }
            break;
        case OfficialEffectTypeId::kCannotUseThisAttackNextTurn:
        case OfficialEffectTypeId::kDamageChangeMyAttackNextTurn:
        case OfficialEffectTypeId::kDamageChangeActiveNextTurn:
        case OfficialEffectTypeId::kDamageChangeNextTurn:
        case OfficialEffectTypeId::kAttackCostChangeNextTurn:
        case OfficialEffectTypeId::kRetreatCostChangeNextTurn:
        case OfficialEffectTypeId::kCannotRetreatNextTurn:
        case OfficialEffectTypeId::kCannotHandAttachEnergyNextTurn:
        case OfficialEffectTypeId::kCannotAttackNextTurn:
        case OfficialEffectTypeId::kCannotAttackLessEqualEnergy2NextTurn:
        case OfficialEffectTypeId::kAttackCoinNextTurn:
        case OfficialEffectTypeId::kAttackCoin2NextTurn:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* card = official_pod_card(state, target.card);
                if (official_effect_blocks_target_effect(state, rules, target.card)) continue;
                if (type == OfficialEffectTypeId::kCannotUseThisAttackNextTurn) {
                    official_set_packed_i16(card->next_turn, 0, state->current_attack_id);
                } else if (type == OfficialEffectTypeId::kDamageChangeMyAttackNextTurn) {
                    official_add_packed_i16(
                        card->next_turn, 8, official_effect_value(*state, effect, 0));
                } else if (type == OfficialEffectTypeId::kDamageChangeActiveNextTurn) {
                    official_add_packed_i16(
                        card->next_turn, 6, official_effect_value(*state, effect, 0));
                } else if (type == OfficialEffectTypeId::kDamageChangeNextTurn) {
                    official_add_packed_i16(
                        card->next_turn, 4, official_effect_value(*state, effect, 0));
                } else if (type == OfficialEffectTypeId::kAttackCostChangeNextTurn) {
                    official_add_packed_i8(
                        card->next_turn, 10, official_effect_value(*state, effect, 0));
                } else if (type == OfficialEffectTypeId::kRetreatCostChangeNextTurn) {
                    official_add_packed_i8(
                        card->next_turn, 11, official_effect_value(*state, effect, 0));
                } else {
                    std::uint32_t bit = 96;
                    if (type == OfficialEffectTypeId::kCannotHandAttachEnergyNextTurn) bit = 97;
                    else if (type == OfficialEffectTypeId::kCannotAttackNextTurn) bit = 98;
                    else if (type == OfficialEffectTypeId::kCannotAttackLessEqualEnergy2NextTurn) bit = 99;
                    else if (type == OfficialEffectTypeId::kAttackCoinNextTurn) bit = 100;
                    else if (type == OfficialEffectTypeId::kAttackCoin2NextTurn) bit = 101;
                    official_set_packed_flag(card->next_turn, bit);
                }
            }
            break;
        case OfficialEffectTypeId::kMetalDamageChangeNextTurn:
        case OfficialEffectTypeId::kCannotAttackLessEqualEnergy2NextTurnPlayer:
        case OfficialEffectTypeId::kCannotPlayItemNextTurn:
        case OfficialEffectTypeId::kCannotPlaySupporterNextTurn:
        case OfficialEffectTypeId::kCannotPlayStadiumNextTurn:
        case OfficialEffectTypeId::kCannotPlaySpecialEnergyNextTurn:
        case OfficialEffectTypeId::kCannotEvolveNextTurn:
        case OfficialEffectTypeId::kCannotRetreatPoison: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_effect_target_player(*state, mask, ordinal);
                if (player < 0) break;
                auto* words = &state->players[player].next_turn;
                if (type == OfficialEffectTypeId::kMetalDamageChangeNextTurn) {
                    official_set_packed_i16(
                        words,
                        0,
                        official_clamp_i32(
                            official_packed_i16(words, 0)
                                + official_effect_value(*state, effect, 0),
                            -32768,
                            32767));
                } else {
                    std::uint32_t bit = 16;
                    if (type == OfficialEffectTypeId::kCannotPlayItemNextTurn) bit = 17;
                    else if (type == OfficialEffectTypeId::kCannotPlaySupporterNextTurn) bit = 18;
                    else if (type == OfficialEffectTypeId::kCannotPlayStadiumNextTurn) bit = 19;
                    else if (type == OfficialEffectTypeId::kCannotPlaySpecialEnergyNextTurn) bit = 20;
                    else if (type == OfficialEffectTypeId::kCannotEvolveNextTurn) bit = 21;
                    else if (type == OfficialEffectTypeId::kCannotRetreatPoison) bit = 22;
                    official_set_packed_flag(words, bit);
                }
            }
            break;
        }
        case OfficialEffectTypeId::kTakeDamageChangeNextMyTurnEnemy:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                official_set_packed_i16(
                    &official_pod_card(state, target.card)->next_turn_enemy,
                    0,
                    official_effect_value(*state, effect, 0));
            }
            break;
        case OfficialEffectTypeId::kCannotUseThisAttackNonActive:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)) {
                    official_pod_card(state, target.card)->cannot_use_attack_id_non_active =
                        static_cast<std::int16_t>(state->current_attack_id);
                }
            }
            break;
        case OfficialEffectTypeId::kFailAttack:
            state->fail_attack = 1;
            break;
        case OfficialEffectTypeId::kCancelFailAttack:
            state->fail_attack = 0;
            break;
        case OfficialEffectTypeId::kBreakIfCoinHead:
            if (official_effect_roll_coins(state, 1) != 0) {
                state->control_flags |= kOfficialBreakEffectFlag;
            }
            break;
        case OfficialEffectTypeId::kBreakIfCoinTail:
            if (official_effect_roll_coins(state, 1) == 0) {
                state->control_flags |= kOfficialBreakEffectFlag;
            }
            break;
        case OfficialEffectTypeId::kBreakIfCoinTailMulti: {
            const std::int32_t count = official_effect_value(*state, effect, 0);
            if (official_effect_roll_coins(state, count) < count) {
                state->control_flags |= kOfficialBreakEffectFlag;
            }
            break;
        }
        case OfficialEffectTypeId::kSkipIfCoinTail:
            if (official_effect_roll_coins(state, 1) == 0) {
                state->effect_jump = static_cast<std::uint8_t>(
                    official_effect_value(*state, effect, 0));
            }
            break;
        case OfficialEffectTypeId::kPostEffectActivate:
            state->post_effect_activate = 1;
            break;
        case OfficialEffectTypeId::kBreakIfNotPostEffectActivated:
            if (state->post_effect_activate == 0) {
                state->control_flags |= kOfficialBreakEffectFlag;
            }
            break;
        case OfficialEffectTypeId::kPrizeToHand:
        case OfficialEffectTypeId::kDamageCounterAny:
        case OfficialEffectTypeId::kDamageCounterSwitchAny:
        case OfficialEffectTypeId::kRemoveDamageCounter:
        case OfficialEffectTypeId::kRecoverSpecialConditionSingle:
        case OfficialEffectTypeId::kAttackDamageChangePutDamageCounter:
        case OfficialEffectTypeId::kAttackDamageMulti:
        case OfficialEffectTypeId::kSelectSwitchEnergy:
        case OfficialEffectTypeId::kCannotUseSelectedAttack:
        case OfficialEffectTypeId::kSelectActivate:
        case OfficialEffectTypeId::kSelectEffect:
        case OfficialEffectTypeId::kSelectPoisonBurnConfuse:
        case OfficialEffectTypeId::kSelectSpecialCondition:
            return OfficialEffectApplyResult::kNeedsSelection;
        default:
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedEffect,
                effect.values[kEffectType]);
            return OfficialEffectApplyResult::kUnsupported;
    }
    return official_pod_ok(state)
        ? OfficialEffectApplyResult::kApplied
        : OfficialEffectApplyResult::kError;
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_EFFECT_HD
