#pragma once

#include <cstdint>

#include "ptcg_cuda/official_core_pod.cuh"
#include "ptcg_cuda/official_semantic_ids.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_TARGET_HD __host__ __device__
#else
#define PTCG_OFFICIAL_TARGET_HD
#endif

namespace ptcg::cuda_engine {

enum class OfficialTargetMatchResult : std::int32_t {
    kNoMatch = 0,
    kMatch = 1,
    kUnsupported = 2,
    kError = 3,
};

PTCG_OFFICIAL_TARGET_HD inline bool official_compare(
    std::int32_t left,
    std::int32_t right,
    std::int32_t comparator) {
    switch (comparator) {
        case 0: return left == right;
        case 1: return left >= right;
        case 2: return left <= right;
        case 3: return left != right;
        case 4: return left > right;
        case 5: return left < right;
        default: return false;
    }
}

PTCG_OFFICIAL_TARGET_HD inline bool official_bool_compare(
    bool value,
    std::int32_t comparator) {
    return comparator == 0 ? value : !value;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_energy_type_matches(
    std::int32_t actual,
    std::int32_t expected) {
    return expected == 0 ? actual == 0 : (actual & expected) != 0;
}

PTCG_OFFICIAL_TARGET_HD inline const OfficialNameSetRule* official_name_set_rule(
    const OfficialRulePackView& rules,
    std::int32_t name_id) {
    if (name_id <= 0) return nullptr;
    const std::uint32_t index = static_cast<std::uint32_t>(name_id - 1);
    const std::uint32_t count = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kNameSets)];
    if (index >= count) return nullptr;
    const OfficialNameSetRule* row = &rules.name_sets[index];
    return row->values[kNameSetId] == name_id ? row : nullptr;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_name_set_contains(
    const OfficialRulePackView& rules,
    std::int32_t name_id,
    std::uint32_t offset_index,
    std::uint32_t count_index,
    std::int32_t card_id) {
    const OfficialNameSetRule* row = official_name_set_rule(rules, name_id);
    if (row == nullptr) return false;
    const std::int32_t offset = row->values[offset_index];
    const std::int32_t count = row->values[count_index];
    const std::uint32_t table_count = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kNameCardIds)];
    if (offset < 0 || count < 0
        || static_cast<std::uint64_t>(offset) + static_cast<std::uint64_t>(count)
            > table_count) return false;
    for (std::int32_t index = 0; index < count; ++index) {
        if (rules.name_card_ids[offset + index] == static_cast<std::uint32_t>(card_id)) {
            return true;
        }
    }
    return false;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_card_flag(
    const OfficialCardRule& master,
    std::uint32_t bit) {
    return (master.flags & (1ULL << bit)) != 0;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_is_energy_card(std::int32_t card_type) {
    return card_type == 5 || card_type == 6;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_is_trainer(std::int32_t card_type) {
    return card_type >= 1 && card_type <= 4;
}

struct OfficialEnergyInfoPod {
    std::int32_t type = 0;
    std::int32_t count = 0;
};

PTCG_OFFICIAL_TARGET_HD inline std::int32_t official_energy_type_by_index(
    std::int32_t index) {
    switch (index) {
        case 0: return 0;
        case 1: return 1;
        case 2: return 2;
        case 3: return 4;
        case 4: return 8;
        case 5: return 16;
        case 6: return 32;
        case 7: return 64;
        case 8: return 128;
        case 9: return 256;
        case 10: return 511;
        case 11: return 80;
        default: return 0;
    }
}

PTCG_OFFICIAL_TARGET_HD inline std::int32_t official_continual_i8(
    const OfficialCardStatePod& card,
    std::uint32_t byte_offset) {
    const std::uint32_t word = byte_offset / 8U;
    const std::uint32_t shift = (byte_offset % 8U) * 8U;
    const std::uint8_t raw = static_cast<std::uint8_t>(
        (card.continual_state[word] >> shift) & 0xffU);
    return static_cast<std::int8_t>(raw);
}

PTCG_OFFICIAL_TARGET_HD inline std::int32_t official_continual_i16(
    const OfficialCardStatePod& card,
    std::uint32_t byte_offset) {
    const std::uint32_t word = byte_offset / 8U;
    const std::uint32_t shift = (byte_offset % 8U) * 8U;
    const std::uint16_t raw = static_cast<std::uint16_t>(
        (card.continual_state[word] >> shift) & 0xffffU);
    return static_cast<std::int16_t>(raw);
}

PTCG_OFFICIAL_TARGET_HD inline std::int32_t official_this_turn_i8(
    const OfficialCardStatePod& card,
    std::uint32_t byte_offset) {
    const std::uint32_t word = byte_offset / 4U;
    const std::uint32_t shift = (byte_offset % 4U) * 8U;
    const std::uint8_t raw = static_cast<std::uint8_t>(
        (card.this_turn[word] >> shift) & 0xffU);
    return static_cast<std::int8_t>(raw);
}

PTCG_OFFICIAL_TARGET_HD inline bool official_continual_flag(
    const OfficialCardStatePod& card,
    std::uint32_t bit) {
    return (card.continual_state[4] & (1ULL << bit)) != 0;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_target_has_enabled_ability(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialCardStatePod& card,
    const OfficialCardRule& master) {
    const std::int32_t ability_id = master.values[kCardAbilityId];
    if (ability_id <= 0 || official_continual_flag(card, 0)) return false;
    const OfficialSkillRule* ability = official_skill_rule(
        rules, static_cast<std::uint32_t>(ability_id));
    if (ability == nullptr) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, ability_id);
        return false;
    }
    constexpr std::uint64_t kTargetSkillKoMeAbilityFlag = 1ULL << 6U;
    return !official_continual_flag(card, 1)
        || (ability->flags & kTargetSkillKoMeAbilityFlag) == 0;
}

PTCG_OFFICIAL_TARGET_HD inline std::int32_t official_effective_pokemon_type(
    const OfficialCardStatePod& card,
    const OfficialCardRule& master) {
    std::int32_t type = master.values[kCardEnergyType];
    const std::int32_t type_index = official_continual_i8(card, 30);
    if (type_index > 0) type |= official_energy_type_by_index(type_index);
    return type;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_find_attached_pokemon(
    const OfficialStatePod& state,
    const OfficialCardStatePod& attached,
    OfficialCardRefPod* output) {
    if (attached.player < 0 || attached.player > 1) return false;
    const OfficialPlayerStatePod& ps = state.players[attached.player];
    for (int zone = 0; zone < 2; ++zone) {
        const OfficialCardRefPod* values = zone == 0 ? ps.active.values : ps.bench.values;
        const std::uint16_t count = zone == 0 ? ps.active.count : ps.bench.count;
        for (std::uint16_t index = 0; index < count; ++index) {
            const OfficialCardStatePod* candidate = official_pod_card(&state, values[index]);
            if (candidate != nullptr
                && candidate->move_counter == attached.attach_move_counter) {
                *output = values[index];
                return true;
            }
        }
    }
    return false;
}

PTCG_OFFICIAL_TARGET_HD inline OfficialEnergyInfoPod official_energy_info(
    const OfficialStatePod& state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod energy_ref,
    OfficialCardRefPod pokemon_ref) {
    const OfficialCardStatePod* energy = official_pod_card(&state, energy_ref);
    const OfficialCardStatePod* pokemon = official_pod_card(&state, pokemon_ref);
    if (energy == nullptr || pokemon == nullptr) return {};
    const OfficialCardRule* energy_master = official_card_rule(
        rules, static_cast<std::uint32_t>(energy->card_id));
    const OfficialCardRule* pokemon_master = official_card_rule(
        rules, static_cast<std::uint32_t>(pokemon->card_id));
    if (energy_master == nullptr || pokemon_master == nullptr) return {};

    const std::int32_t card_id = energy_master->values[kCardId];
    const std::int32_t card_type = energy_master->values[kCardType];
    const std::int32_t evolution = pokemon_master->values[kCardEvolutionType];
    if (card_type == 6) {
        if (card_id == 10) {
            return evolution == 3 ? OfficialEnergyInfoPod{511, 2}
                                  : OfficialEnergyInfoPod{0, 1};
        }
        if (card_id == 16) {
            return evolution == 1 ? OfficialEnergyInfoPod{511, 1}
                                  : OfficialEnergyInfoPod{0, 1};
        }
        if (card_id == 17) {
            return evolution == 2 || evolution == 3
                ? OfficialEnergyInfoPod{0, 3} : OfficialEnergyInfoPod{0, 1};
        }
    } else if (card_id == 1 && official_continual_flag(*pokemon, 29)) {
        return OfficialEnergyInfoPod{1, 2};
    }
    return OfficialEnergyInfoPod{
        energy_master->values[kCardEnergyType],
        energy_master->values[kCardEnergyCount]};
}

PTCG_OFFICIAL_TARGET_HD inline bool official_card_ref_in_list(
    const OfficialCardRefPod* values,
    std::uint16_t count,
    OfficialCardRefPod ref) {
    for (std::uint16_t index = 0; index < count; ++index) {
        if (values[index] == ref) return true;
    }
    return false;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_card_id_in_list(
    const OfficialStatePod& state,
    const OfficialCardRefPod* values,
    std::uint16_t count,
    std::int32_t card_id) {
    for (std::uint16_t index = 0; index < count; ++index) {
        const OfficialCardStatePod* card = official_pod_card(&state, values[index]);
        if (card != nullptr && card->card_id == card_id) return true;
    }
    return false;
}

PTCG_OFFICIAL_TARGET_HD inline std::int32_t official_attached_energy_cards(
    const OfficialStatePod& state,
    OfficialCardRefPod pokemon,
    std::int32_t required_type,
    bool special_only,
    const OfficialRulePackView& rules) {
    const OfficialCardStatePod* target = official_pod_card(&state, pokemon);
    if (target == nullptr || target->player < 0 || target->player > 1) return 0;
    const auto& energy = state.players[target->player].energy;
    std::int32_t count = 0;
    for (std::uint16_t index = 0; index < energy.count; ++index) {
        const OfficialCardRefPod energy_ref = energy.values[index];
        const OfficialCardStatePod* card = official_pod_card(&state, energy_ref);
        if (card == nullptr || card->attach_move_counter != target->move_counter) continue;
        const OfficialCardRule* master = official_card_rule(
            rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) continue;
        if (special_only && master->values[kCardType] != 6) continue;
        const OfficialEnergyInfoPod info = official_energy_info(
            state, rules, energy_ref, pokemon);
        if (required_type != -1
            && !official_energy_type_matches(info.type, required_type)) {
            continue;
        }
        count += info.count;
    }
    return count;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_has_attached_energy_name(
    const OfficialStatePod& state,
    OfficialCardRefPod pokemon,
    const OfficialRulePackView& rules,
    std::int32_t name_id) {
    const OfficialCardStatePod* target = official_pod_card(&state, pokemon);
    if (target == nullptr || target->player < 0 || target->player > 1) return false;
    const auto& energy = state.players[target->player].energy;
    for (std::uint16_t index = 0; index < energy.count; ++index) {
        const OfficialCardStatePod* card = official_pod_card(&state, energy.values[index]);
        if (card != nullptr
            && card->attach_move_counter == target->move_counter
            && official_name_set_contains(
                rules, name_id, kNameEqualOffset, kNameEqualCount, card->card_id)) {
            return true;
        }
    }
    return false;
}

PTCG_OFFICIAL_TARGET_HD inline std::int32_t official_attached_tool_cards(
    const OfficialStatePod& state,
    OfficialCardRefPod pokemon,
    const OfficialRulePackView& rules,
    std::int32_t name_id = 0) {
    const OfficialCardStatePod* target = official_pod_card(&state, pokemon);
    if (target == nullptr || target->player < 0 || target->player > 1) return 0;
    const auto& tools = state.players[target->player].tool;
    std::int32_t count = 0;
    for (std::uint16_t index = 0; index < tools.count; ++index) {
        const OfficialCardStatePod* card = official_pod_card(&state, tools.values[index]);
        if (card == nullptr || card->attach_move_counter != target->move_counter) continue;
        if (name_id != 0 && !official_name_set_contains(
                rules,
                name_id,
                kNameEqualOffset,
                kNameEqualCount,
                card->card_id)) continue;
        ++count;
    }
    return count;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_evolves_from(
    const OfficialCardRule& evolution,
    const OfficialCardRule& base,
    bool stage2_shortcut = false) {
    const std::int32_t expected = evolution.values[
        stage2_shortcut ? kCardEvolvesFrom2NameId : kCardEvolvesFromNameId];
    return expected != 0 && expected == base.values[kCardNameId];
}

PTCG_OFFICIAL_TARGET_HD inline bool official_has_evolution_target(
    const OfficialStatePod& state,
    const OfficialRulePackView& rules,
    const OfficialCardRule& evolution,
    std::int32_t player,
    bool stage2_shortcut,
    bool reject_appeared) {
    const OfficialPlayerStatePod& ps = state.players[player];
    for (int zone = 0; zone < 2; ++zone) {
        const OfficialCardRefPod* values = zone == 0 ? ps.active.values : ps.bench.values;
        const std::uint16_t count = zone == 0 ? ps.active.count : ps.bench.count;
        for (std::uint16_t index = 0; index < count; ++index) {
            const OfficialCardStatePod* card = official_pod_card(&state, values[index]);
            if (card == nullptr || (reject_appeared && (card->runtime_flags & kCardAppear) != 0)) {
                continue;
            }
            const OfficialCardRule* master = official_card_rule(
                rules, static_cast<std::uint32_t>(card->card_id));
            if (master != nullptr
                && official_evolves_from(evolution, *master, stage2_shortcut)) return true;
        }
    }
    return false;
}

PTCG_OFFICIAL_TARGET_HD inline OfficialTargetMatchResult official_match_target_condition(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    const OfficialConditionRule& condition,
    OfficialAreaRefPod effect_card) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr) return OfficialTargetMatchResult::kError;
    const OfficialCardRule* master = official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    if (master == nullptr) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, card->card_id);
        return OfficialTargetMatchResult::kError;
    }
    const auto type = static_cast<OfficialTargetTypeId>(condition.values[kConditionType]);
    const std::int32_t comparator = condition.values[kConditionComparator];
    const std::int32_t value = condition.values[kConditionValue];
    const std::int32_t value2 = condition.values[kConditionValue2];
    const std::int32_t name_id = condition.values[kConditionNameId];
    const std::int32_t card_type = master->values[kCardType];
    const std::int32_t pokemon_type = master->values[kCardPokemonType];
    const std::int32_t evolution_type = master->values[kCardEvolutionType];
    bool match = false;
    switch (type) {
        case OfficialTargetTypeId::kAll: match = true; break;
        case OfficialTargetTypeId::kHp:
            match = official_compare(
                official_pod_max_hp(state, rules, ref) - card->damage, value, comparator);
            return match ? OfficialTargetMatchResult::kMatch : OfficialTargetMatchResult::kNoMatch;
        case OfficialTargetTypeId::kMaxHp:
            match = official_compare(master->values[kCardHp], value, comparator);
            return match ? OfficialTargetMatchResult::kMatch : OfficialTargetMatchResult::kNoMatch;
        case OfficialTargetTypeId::kRetreatCost:
            match = official_compare(master->values[kCardRetreatCost], value, comparator);
            return match ? OfficialTargetMatchResult::kMatch : OfficialTargetMatchResult::kNoMatch;
        case OfficialTargetTypeId::kEnergyType:
            match = official_energy_type_matches(master->values[kCardEnergyType], value); break;
        case OfficialTargetTypeId::kEnergyType2:
            match = official_energy_type_matches(master->values[kCardEnergyType], value)
                || official_energy_type_matches(master->values[kCardEnergyType], value2); break;
        case OfficialTargetTypeId::kResistance:
            match = official_energy_type_matches(master->values[kCardResistance], value); break;
        case OfficialTargetTypeId::kPokemonCard: match = card_type == 0; break;
        case OfficialTargetTypeId::kBasicPokemon:
            match = evolution_type == 1
                && (official_pod_is_in_play(static_cast<OfficialArea>(card->area)) || card_type == 0);
            break;
        case OfficialTargetTypeId::kEvolvedPokemon:
            match = evolution_type == 2 || evolution_type == 3; break;
        case OfficialTargetTypeId::kStage1: match = evolution_type == 2; break;
        case OfficialTargetTypeId::kStage2: match = evolution_type == 3; break;
        case OfficialTargetTypeId::kBasicEnergy: match = card_type == 5; break;
        case OfficialTargetTypeId::kSpecialEnergy: match = card_type == 6; break;
        case OfficialTargetTypeId::kEnergyCard: match = official_is_energy_card(card_type); break;
        case OfficialTargetTypeId::kItem: match = card_type == 1; break;
        case OfficialTargetTypeId::kTool: match = card_type == 2; break;
        case OfficialTargetTypeId::kSupporter: match = card_type == 3; break;
        case OfficialTargetTypeId::kStadium: match = card_type == 4; break;
        case OfficialTargetTypeId::kTrainer: match = official_is_trainer(card_type); break;
        case OfficialTargetTypeId::kCardId: match = card->card_id == value; break;
        case OfficialTargetTypeId::kPokemonOrBasicEnergy:
            match = card_type == 0 || card_type == 5; break;
        case OfficialTargetTypeId::kBasicPokemonOrBasicEnergy:
            match = (card_type == 0 && evolution_type == 1) || card_type == 5; break;
        case OfficialTargetTypeId::kNotRulePokemonCardOrBasicEnergy:
            match = pokemon_type == 1 || card_type == 5; break;
        case OfficialTargetTypeId::kEnergyTypePokemonOrStadium:
            match = card_type == 4
                || (card_type == 0
                    && official_energy_type_matches(master->values[kCardEnergyType], value));
            break;
        case OfficialTargetTypeId::kItemOrTool: match = card_type == 1 || card_type == 2; break;
        case OfficialTargetTypeId::kEnemyToolOrSpecialEnergyOrStadium: {
            const OfficialCardStatePod* source = official_pod_card(state, effect_card.card);
            if (card->area == static_cast<std::uint8_t>(OfficialArea::kTool)) {
                return source != nullptr && source->player != card->player
                    ? OfficialTargetMatchResult::kMatch : OfficialTargetMatchResult::kNoMatch;
            }
            if (card->area == static_cast<std::uint8_t>(OfficialArea::kEnergy)) {
                return source != nullptr && source->player != card->player && card_type == 6
                    ? OfficialTargetMatchResult::kMatch : OfficialTargetMatchResult::kNoMatch;
            }
            return OfficialTargetMatchResult::kMatch;
        }
        case OfficialTargetTypeId::kEthanPokemonOrBasicFireEnergy:
            match = official_card_flag(*master, 17) || card->card_id == 2; break;
        case OfficialTargetTypeId::kHasAbility:
            match = official_target_has_enabled_ability(
                state, rules, *card, *master);
            break;
        case OfficialTargetTypeId::kHasAbilityName:
            match = official_target_has_enabled_ability(
                    state, rules, *card, *master)
                && official_name_set_contains(
                    rules,
                    name_id,
                    kNameAbilityOffset,
                    kNameAbilityCount,
                    card->card_id);
            break;
        case OfficialTargetTypeId::kHasAttackName:
            match = official_name_set_contains(
                rules, name_id, kNameAttackOffset, kNameAttackCount, card->card_id); break;
        case OfficialTargetTypeId::kRulePokemon: match = official_card_flag(*master, 29); break;
        case OfficialTargetTypeId::kNotRulePokemon:
            match = !official_card_flag(*master, 29) && pokemon_type != 0; break;
        case OfficialTargetTypeId::kEx: match = pokemon_type == 3 || pokemon_type == 4; break;
        case OfficialTargetTypeId::kMegaEx: match = pokemon_type == 4; break;
        case OfficialTargetTypeId::kTerastal: match = official_card_flag(*master, 0); break;
        case OfficialTargetTypeId::kAncient: match = official_card_flag(*master, 11); break;
        case OfficialTargetTypeId::kFuture: match = official_card_flag(*master, 12); break;
        case OfficialTargetTypeId::kHop: match = official_card_flag(*master, 13); break;
        case OfficialTargetTypeId::kLillie: match = official_card_flag(*master, 14); break;
        case OfficialTargetTypeId::kIono: match = official_card_flag(*master, 15); break;
        case OfficialTargetTypeId::kN: match = official_card_flag(*master, 16); break;
        case OfficialTargetTypeId::kEthan: match = official_card_flag(*master, 17); break;
        case OfficialTargetTypeId::kCynthia: match = official_card_flag(*master, 18); break;
        case OfficialTargetTypeId::kMisty: match = official_card_flag(*master, 19); break;
        case OfficialTargetTypeId::kArven: match = official_card_flag(*master, 20); break;
        case OfficialTargetTypeId::kSteven: match = official_card_flag(*master, 21); break;
        case OfficialTargetTypeId::kMarnie: match = official_card_flag(*master, 22); break;
        case OfficialTargetTypeId::kErika: match = official_card_flag(*master, 23); break;
        case OfficialTargetTypeId::kLarry: match = official_card_flag(*master, 24); break;
        case OfficialTargetTypeId::kTeamRocket: match = official_card_flag(*master, 25); break;
        case OfficialTargetTypeId::kSilcoonOrCascoon:
        case OfficialTargetTypeId::kHonedgeOrDoubladeOrAegislash:
        case OfficialTargetTypeId::kName:
            match = official_name_set_contains(
                rules, name_id, kNameEqualOffset, kNameEqualCount, card->card_id); break;
        case OfficialTargetTypeId::kKoffingOrWeezing:
        case OfficialTargetTypeId::kNameContains:
            match = official_name_set_contains(
                rules, name_id, kNameContainsOffset, kNameContainsCount, card->card_id); break;
        case OfficialTargetTypeId::kCanEvolve:
        case OfficialTargetTypeId::kCanEvolveField:
        case OfficialTargetTypeId::kCanEvolveFieldNotAppearThisTurn:
            match = (evolution_type == 2 || evolution_type == 3)
                && official_has_evolution_target(
                    *state,
                    rules,
                    *master,
                    card->player,
                    false,
                    type == OfficialTargetTypeId::kCanEvolveFieldNotAppearThisTurn);
            break;
        case OfficialTargetTypeId::kCanEvolve2:
            match = evolution_type == 3
                && official_has_evolution_target(*state, rules, *master, card->player, true, true);
            break;
        case OfficialTargetTypeId::kCanEvolveMe: {
            const OfficialCardStatePod* source = official_pod_card(state, effect_card.card);
            const OfficialCardRule* source_master = source == nullptr ? nullptr : official_card_rule(
                rules, static_cast<std::uint32_t>(source->card_id));
            match = source_master != nullptr && official_evolves_from(*master, *source_master);
            break;
        }
        case OfficialTargetTypeId::kCanEvolveContextCard: {
            const OfficialCardStatePod* context = official_pod_card(state, state->context_card);
            const OfficialCardRule* context_master = context == nullptr ? nullptr : official_card_rule(
                rules, static_cast<std::uint32_t>(context->card_id));
            match = context_master != nullptr && official_evolves_from(*master, *context_master);
            break;
        }
        case OfficialTargetTypeId::kCanEvolvesToContextCard: {
            const OfficialCardStatePod* context = official_pod_card(state, state->context_card);
            const OfficialCardRule* context_master = context == nullptr ? nullptr : official_card_rule(
                rules, static_cast<std::uint32_t>(context->card_id));
            match = context_master != nullptr && official_evolves_from(*context_master, *master);
            break;
        }
        case OfficialTargetTypeId::kEvolved: {
            const auto& list = state->players[card->player].pre_evolution;
            for (std::uint16_t index = 0; index < list.count; ++index) {
                const OfficialCardStatePod* prior = official_pod_card(state, list.values[index]);
                if (prior != nullptr && prior->attach_move_counter == card->move_counter) {
                    match = true;
                    break;
                }
            }
            break;
        }
        case OfficialTargetTypeId::kEvolvedThisTurnName:
            for (std::uint16_t index = 0; index < state->turn_evolve.count; ++index) {
                const OfficialEvolveRecordPod record = state->turn_evolve.values[index];
                if (record.to != ref) continue;
                const OfficialCardStatePod* prior = official_pod_card(state, record.from);
                const OfficialCardRule* prior_master = prior == nullptr ? nullptr : official_card_rule(
                    rules, static_cast<std::uint32_t>(prior->card_id));
                if (prior_master != nullptr
                    && prior_master->values[kCardNameId] == name_id) match = true;
            }
            break;
        case OfficialTargetTypeId::kNotAppearThisTurn:
            match = (card->runtime_flags & kCardAppear) == 0; break;
        case OfficialTargetTypeId::kHealThisTurn:
            match = official_card_ref_in_list(
                state->turn_heal.values, state->turn_heal.count, ref); break;
        case OfficialTargetTypeId::kAttachedMe: {
            const OfficialCardStatePod* source = official_pod_card(state, effect_card.card);
            match = source != nullptr && card->attach_move_counter == source->move_counter;
            break;
        }
        case OfficialTargetTypeId::kAttachedEffected:
            for (std::uint16_t index = 0; index < state->pre_targets.count; ++index) {
                const OfficialCardStatePod* prior = official_pod_card(
                    state, state->pre_targets.values[index].card);
                if (prior != nullptr && card->attach_move_counter == prior->move_counter) match = true;
            }
            break;
        case OfficialTargetTypeId::kAttachedTriggerSubject: {
            const OfficialCardStatePod* subject = official_pod_card(state, state->trigger_info.subject.card);
            match = subject != nullptr && card->attach_move_counter == subject->move_counter;
            break;
        }
        case OfficialTargetTypeId::kAttachedTriggerObject: {
            const OfficialCardStatePod* object = official_pod_card(state, state->trigger_info.object.card);
            match = object != nullptr && card->attach_move_counter == object->move_counter;
            break;
        }
        case OfficialTargetTypeId::kAttachedContextCard: {
            const OfficialCardStatePod* context = official_pod_card(state, state->context_card);
            match = context != nullptr && card->attach_move_counter == context->move_counter;
            break;
        }
        case OfficialTargetTypeId::kAttachedActivePokemon: {
            const auto& active = state->players[card->player].active;
            const OfficialCardStatePod* target = active.count == 0
                ? nullptr : official_pod_card(state, active.values[0]);
            match = target != nullptr && card->attach_move_counter == target->move_counter;
            break;
        }
        case OfficialTargetTypeId::kAttachedBenchPokemon: {
            const auto& bench = state->players[card->player].bench;
            for (std::uint16_t index = 0; index < bench.count; ++index) {
                const OfficialCardStatePod* target = official_pod_card(state, bench.values[index]);
                if (target != nullptr && card->attach_move_counter == target->move_counter) match = true;
            }
            break;
        }
        case OfficialTargetTypeId::kIsAttachedEnergy:
            match = official_attached_energy_cards(*state, ref, -1, false, rules) > 0; break;
        case OfficialTargetTypeId::kAttachedEnergyCount:
            return official_compare(
                official_attached_energy_cards(*state, ref, -1, false, rules), value, comparator)
                ? OfficialTargetMatchResult::kMatch : OfficialTargetMatchResult::kNoMatch;
        case OfficialTargetTypeId::kIsAttachedSpecialEnergy:
            match = official_attached_energy_cards(*state, ref, -1, true, rules) > 0; break;
        case OfficialTargetTypeId::kIsAttachedEnergyType:
            match = official_attached_energy_cards(*state, ref, value, false, rules) > 0; break;
        case OfficialTargetTypeId::kIsAttachedEnergy2Type:
            match = official_attached_energy_cards(*state, ref, value, false, rules) >= 2; break;
        case OfficialTargetTypeId::kIsAttachedTool:
            match = official_attached_tool_cards(*state, ref, rules) > 0; break;
        case OfficialTargetTypeId::kIsAttachedToolName:
            match = official_attached_tool_cards(*state, ref, rules, name_id) > 0; break;
        case OfficialTargetTypeId::kIsAttachedToolOrSpecialEnergy:
            match = official_attached_tool_cards(*state, ref, rules) > 0
                || official_attached_energy_cards(*state, ref, -1, true, rules) > 0;
            break;
        case OfficialTargetTypeId::kNotContextCardAttachedPokemon: {
            const OfficialCardStatePod* context = official_pod_card(state, state->context_card);
            match = context == nullptr || context->attach_move_counter != card->move_counter;
            break;
        }
        case OfficialTargetTypeId::kNotSelectedListAttachedPokemon:
            match = true;
            for (std::uint16_t index = 0; index < state->selected_list.count; ++index) {
                const OfficialCardStatePod* selected = official_pod_card(
                    state, state->selected_list.values[index]);
                if (selected != nullptr && selected->attach_move_counter == card->move_counter) {
                    match = false;
                }
            }
            break;
        case OfficialTargetTypeId::kReverse:
            match = card->area == static_cast<std::uint8_t>(OfficialArea::kHand)
                || card->reverse != 0;
            break;
        case OfficialTargetTypeId::kArea:
            match = card->area == value; break;
        case OfficialTargetTypeId::kTriggerSubject:
            match = ref == state->trigger_info.subject.card; break;
        case OfficialTargetTypeId::kTriggerObject:
            match = ref == state->trigger_info.object.card; break;
        case OfficialTargetTypeId::kDamageCounter:
            return official_compare(card->damage / 10, value, comparator)
                ? OfficialTargetMatchResult::kMatch : OfficialTargetMatchResult::kNoMatch;
        case OfficialTargetTypeId::kMinHp: {
            std::int32_t minimum = 0x7fffffff;
            for (int player = 0; player < 2; ++player) {
                const OfficialPlayerStatePod& ps = state->players[player];
                for (int zone = 0; zone < 2; ++zone) {
                    const OfficialCardRefPod* values = zone == 0 ? ps.active.values : ps.bench.values;
                    const std::uint16_t count = zone == 0 ? ps.active.count : ps.bench.count;
                    for (std::uint16_t index = 0; index < count; ++index) {
                        if (values[index] == effect_card.card) continue;
                        const OfficialCardStatePod* candidate = official_pod_card(state, values[index]);
                        if (candidate == nullptr) continue;
                        const std::int32_t hp = official_pod_max_hp(state, rules, values[index])
                            - candidate->damage;
                        if (hp < minimum) minimum = hp;
                    }
                }
            }
            match = official_pod_max_hp(state, rules, ref) - card->damage == minimum;
            break;
        }
        case OfficialTargetTypeId::kSpecialCondition:
        case OfficialTargetTypeId::kSpecialConditionOrDamaged:
        case OfficialTargetTypeId::kPoison:
        case OfficialTargetTypeId::kBurn:
        case OfficialTargetTypeId::kConfuse:
        case OfficialTargetTypeId::kPoisonOrBurn: {
            if (card->area != static_cast<std::uint8_t>(OfficialArea::kActive)) {
                return OfficialTargetMatchResult::kNoMatch;
            }
            const OfficialPlayerStatePod& ps = state->players[card->player];
            const bool poisoned = official_pod_poison_counter(ps) > 0;
            const bool burned = official_pod_burned(ps);
            const bool confused = official_pod_bad_status(ps) == OfficialBadStatus::kConfused;
            if (type == OfficialTargetTypeId::kSpecialCondition) {
                match = poisoned || burned || official_pod_bad_status(ps) != OfficialBadStatus::kNone;
            } else if (type == OfficialTargetTypeId::kSpecialConditionOrDamaged) {
                match = poisoned || burned || official_pod_bad_status(ps) != OfficialBadStatus::kNone
                    || card->damage > 0;
            } else if (type == OfficialTargetTypeId::kPoison) match = poisoned;
            else if (type == OfficialTargetTypeId::kBurn) match = burned;
            else if (type == OfficialTargetTypeId::kConfuse) match = confused;
            else match = poisoned || burned;
            break;
        }
        case OfficialTargetTypeId::kBenchToActiveThisTurn:
            match = (card->runtime_flags & kCardBenchToActive) != 0; break;
        case OfficialTargetTypeId::kSameNameEnemyField: {
            const OfficialCardStatePod* source = official_pod_card(state, effect_card.card);
            if (source != nullptr) {
                const OfficialPlayerStatePod& enemy = state->players[1 - source->player];
                for (int zone = 0; zone < 2; ++zone) {
                    const OfficialCardRefPod* values = zone == 0 ? enemy.active.values : enemy.bench.values;
                    const std::uint16_t count = zone == 0 ? enemy.active.count : enemy.bench.count;
                    for (std::uint16_t index = 0; index < count; ++index) {
                        const OfficialCardStatePod* candidate = official_pod_card(state, values[index]);
                        const OfficialCardRule* candidate_master = candidate == nullptr ? nullptr
                            : official_card_rule(rules, static_cast<std::uint32_t>(candidate->card_id));
                        if (candidate_master != nullptr
                            && candidate_master->values[kCardNameId] == master->values[kCardNameId]) {
                            match = true;
                        }
                    }
                }
            }
            break;
        }
        case OfficialTargetTypeId::kNotChecked:
            match = !official_card_id_in_list(
                *state, state->check_list.values, state->check_list.count, card->card_id);
            break;
        case OfficialTargetTypeId::kIsAttachedEnergyName:
            match = official_has_attached_energy_name(*state, ref, rules, name_id);
            break;
        case OfficialTargetTypeId::kEnergyTypeAttached: {
            OfficialCardRefPod pokemon{};
            match = official_find_attached_pokemon(*state, *card, &pokemon)
                && official_energy_type_matches(
                    official_energy_info(*state, rules, ref, pokemon).type, value);
            break;
        }
        case OfficialTargetTypeId::kSameTypeEnemy: {
            std::int32_t enemy_types = 0;
            bool enemy_colorless = false;
            const OfficialPlayerStatePod& enemy = state->players[1 - card->player];
            for (int zone = 0; zone < 2; ++zone) {
                const OfficialCardRefPod* values = zone == 0
                    ? enemy.active.values : enemy.bench.values;
                const std::uint16_t count = zone == 0
                    ? enemy.active.count : enemy.bench.count;
                for (std::uint16_t index = 0; index < count; ++index) {
                    const OfficialCardStatePod* candidate = official_pod_card(
                        state, values[index]);
                    const OfficialCardRule* candidate_master = candidate == nullptr ? nullptr
                        : official_card_rule(
                            rules, static_cast<std::uint32_t>(candidate->card_id));
                    if (candidate_master == nullptr) continue;
                    const std::int32_t base_type = candidate_master->values[kCardEnergyType];
                    enemy_types |= base_type;
                    enemy_colorless = enemy_colorless || base_type == 0;
                    const std::int32_t type_index = official_continual_i8(*candidate, 30);
                    if (type_index > 0) {
                        const std::int32_t extra_type = official_energy_type_by_index(type_index);
                        enemy_types |= extra_type;
                        enemy_colorless = enemy_colorless || extra_type == 0;
                    }
                }
            }
            const OfficialPlayerStatePod& mine = state->players[card->player];
            for (int zone = 0; zone < 2 && !match; ++zone) {
                const OfficialCardRefPod* values = zone == 0
                    ? mine.active.values : mine.bench.values;
                const std::uint16_t count = zone == 0
                    ? mine.active.count : mine.bench.count;
                for (std::uint16_t index = 0; index < count; ++index) {
                    const OfficialCardStatePod* candidate = official_pod_card(
                        state, values[index]);
                    const OfficialCardRule* candidate_master = candidate == nullptr ? nullptr
                        : official_card_rule(
                            rules, static_cast<std::uint32_t>(candidate->card_id));
                    if (candidate_master == nullptr) continue;
                    const std::int32_t base_type = candidate_master->values[kCardEnergyType];
                    match = base_type == 0
                        ? enemy_colorless : (base_type & enemy_types) != 0;
                    const std::int32_t type_index = official_continual_i8(*candidate, 30);
                    if (!match && type_index > 0) {
                        const std::int32_t extra_type = official_energy_type_by_index(type_index);
                        match = extra_type == 0
                            ? enemy_colorless : (extra_type & enemy_types) != 0;
                    }
                    if (match) break;
                }
            }
            break;
        }
        default:
            official_pod_fail(
                state,
                OfficialPodError::kUnsupportedTarget,
                condition.values[kConditionType]);
            return OfficialTargetMatchResult::kUnsupported;
    }
    match = official_bool_compare(match, comparator);
    return match ? OfficialTargetMatchResult::kMatch : OfficialTargetMatchResult::kNoMatch;
}

PTCG_OFFICIAL_TARGET_HD inline OfficialTargetMatchResult official_match_target(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    const OfficialTargetRule& target,
    OfficialAreaRefPod effect_card) {
    const std::int32_t offset = target.values[kTargetConditionOffset];
    const std::int32_t count = target.values[kTargetConditionCount];
    const std::uint32_t condition_count = rules.header->counts[
        static_cast<std::uint32_t>(OfficialRuleSection::kConditions)];
    if (offset < 0 || count < 0
        || static_cast<std::uint64_t>(offset) + static_cast<std::uint64_t>(count)
            > condition_count) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, offset);
        return OfficialTargetMatchResult::kError;
    }
    for (std::int32_t index = 0; index < count; ++index) {
        const OfficialTargetMatchResult result = official_match_target_condition(
            state, rules, ref, rules.conditions[offset + index], effect_card);
        if (result != OfficialTargetMatchResult::kMatch) return result;
    }
    return OfficialTargetMatchResult::kMatch;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_target_player_matches(
    std::int32_t effect_owner,
    std::int32_t player,
    std::int32_t mask) {
    return ((1 + (effect_owner ^ player)) & mask) != 0;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_append_target(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialPodList<OfficialAreaRefPod, kOfficialListCapacity>* output,
    OfficialCardRefPod ref,
    const OfficialTargetRule& target,
    OfficialAreaRefPod effect_card,
    bool skip_conditions = false) {
    if (target.values[kTargetNotMe] != 0 && ref == effect_card.card) return true;
    if (!skip_conditions) {
        const OfficialTargetMatchResult result = official_match_target(
            state, rules, ref, target, effect_card);
        if (result == OfficialTargetMatchResult::kNoMatch) return true;
        if (result != OfficialTargetMatchResult::kMatch) return false;
    }
    return official_pod_push(
        state,
        output,
        official_pod_area_ref(state, ref),
        OfficialPodError::kSelectionOverflow);
}

template <std::size_t Capacity>
PTCG_OFFICIAL_TARGET_HD inline bool official_append_zone_targets(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialPodList<OfficialAreaRefPod, kOfficialListCapacity>* output,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone,
    const OfficialTargetRule& target,
    OfficialAreaRefPod effect_card,
    bool skip_conditions = false) {
    for (std::uint16_t index = 0; index < zone.count; ++index) {
        if (!official_append_target(
                state,
                rules,
                output,
                zone.values[index],
                target,
                effect_card,
                skip_conditions)) return false;
    }
    return true;
}

PTCG_OFFICIAL_TARGET_HD inline bool official_build_target_list(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialTargetRule& target,
    OfficialPodList<OfficialAreaRefPod, kOfficialListCapacity>* output,
    OfficialAreaRefPod effect_card,
    std::int32_t effect_owner) {
    const std::int32_t area_count = target.values[kTargetAreaCount];
    if (area_count < 0 || area_count > 4) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, area_count);
        return false;
    }
    if (area_count == 0) {
        output->count = 0;
        return true;
    }
    const OfficialArea first = static_cast<OfficialArea>(target.values[kTargetArea0]);
    OfficialPodList<OfficialAreaRefPod, kOfficialListCapacity> previous{};
    if (first == OfficialArea::kEffected) previous = *output;
    output->count = 0;

    if (first == OfficialArea::kMe) {
        return official_pod_area_ref_valid(state, effect_card)
            && official_append_target(state, rules, output, effect_card.card, target, effect_card);
    }
    if (first == OfficialArea::kEffected) {
        for (std::uint16_t index = 0; index < previous.count; ++index) {
            if (!official_append_target(
                    state,
                    rules,
                    output,
                    previous.values[index].card,
                    target,
                    effect_card)) return false;
        }
        return official_pod_ok(state);
    }
    if (first == OfficialArea::kEffectedPreTarget) {
        for (std::uint16_t index = 0; index < state->pre_targets.count; ++index) {
            if (!official_append_target(
                    state,
                    rules,
                    output,
                    state->pre_targets.values[index].card,
                    target,
                    effect_card)) return false;
        }
        return official_pod_ok(state);
    }
    if (first == OfficialArea::kSelectedList) {
        return official_append_zone_targets(
            state, rules, output, state->selected_list, target, effect_card);
    }
    if (first == OfficialArea::kTriggerSubject) {
        if (!official_pod_area_ref_valid(state, state->trigger_info.subject)) {
            output->count = 0;
            return true;
        }
        return official_append_target(
            state,
            rules,
            output,
            state->trigger_info.subject.card,
            target,
            effect_card);
    }
    if (first == OfficialArea::kTriggerObject) {
        if (!official_pod_area_ref_valid(state, state->trigger_info.object)) {
            output->count = 0;
            return true;
        }
        return official_append_target(
            state,
            rules,
            output,
            state->trigger_info.object.card,
            target,
            effect_card);
    }
    if (first == OfficialArea::kAttach) {
        const OfficialCardStatePod* attached = official_pod_card(state, effect_card.card);
        OfficialCardRefPod pokemon{};
        if (attached == nullptr
            || !official_find_attached_pokemon(*state, *attached, &pokemon)) {
            output->count = 0;
            return true;
        }
        return official_append_target(
            state, rules, output, pokemon, target, effect_card);
    }
    if (first == OfficialArea::kTurnPlay) {
        return official_append_zone_targets(
            state, rules, output, state->turn_play, target, effect_card);
    }
    if (first == OfficialArea::kAttackPreMyTurn) {
        const OfficialCardStatePod* source = official_pod_card(state, effect_card.card);
        const std::int32_t active_player = ((state->turn + 1) ^ state->first_player) & 1;
        const std::int32_t history_index = source != nullptr
                && source->player != active_player
            ? 1 : 2;
        const OfficialCardRefPod attacker = state->turn_histories[history_index].attack_card;
        if (attacker.index == 0) {
            output->count = 0;
            return true;
        }
        return official_append_target(
            state, rules, output, attacker, target, effect_card);
    }

    const std::int32_t player_mask = target.values[kTargetPlayer];
    for (int order = 0; order < 2; ++order) {
        const int player = order == 0 ? state->first_player : 1 - state->first_player;
        if (player < 0 || player > 1
            || !official_target_player_matches(effect_owner, player, player_mask)) continue;
        const OfficialPlayerStatePod& ps = state->players[player];
        for (std::int32_t area_index = 0; area_index < area_count; ++area_index) {
            const OfficialArea area = static_cast<OfficialArea>(
                target.values[kTargetArea0 + area_index]);
            const bool skip_enemy = target.values[kTargetSkipEnemy] != 0
                && area == OfficialArea::kHand
                && official_pod_card(state, effect_card.card) != nullptr
                && official_pod_card(state, effect_card.card)->player != player;
            bool ok = true;
            switch (area) {
                case OfficialArea::kDeck:
                    ok = official_append_zone_targets(
                        state, rules, output, ps.deck, target, effect_card); break;
                case OfficialArea::kHand:
                    ok = official_append_zone_targets(
                        state, rules, output, ps.hand, target, effect_card, skip_enemy); break;
                case OfficialArea::kTrash:
                    ok = official_append_zone_targets(
                        state, rules, output, ps.trash, target, effect_card); break;
                case OfficialArea::kActive:
                    ok = official_append_zone_targets(
                        state, rules, output, ps.active, target, effect_card); break;
                case OfficialArea::kBench:
                    ok = official_append_zone_targets(
                        state, rules, output, ps.bench, target, effect_card); break;
                case OfficialArea::kPrize:
                    ok = official_append_zone_targets(
                        state, rules, output, ps.prize, target, effect_card); break;
                case OfficialArea::kEnergy:
                    ok = official_append_zone_targets(
                        state, rules, output, ps.energy, target, effect_card); break;
                case OfficialArea::kTool:
                    ok = official_append_zone_targets(
                        state, rules, output, ps.tool, target, effect_card); break;
                case OfficialArea::kStadium:
                    for (std::uint16_t index = 0; index < state->stadium.count; ++index) {
                        const OfficialCardStatePod* card = official_pod_card(
                            state, state->stadium.values[index]);
                        if (card != nullptr && card->player == player) {
                            ok = official_append_target(
                                state,
                                rules,
                                output,
                                state->stadium.values[index],
                                target,
                                effect_card);
                        }
                    }
                    break;
                case OfficialArea::kLooking:
                    for (std::uint16_t index = 0; index < state->looking.count; ++index) {
                        const OfficialCardStatePod* card = official_pod_card(
                            state, state->looking.values[index]);
                        if (card != nullptr && card->player == player) {
                            ok = official_append_target(
                                state,
                                rules,
                                output,
                                state->looking.values[index],
                                target,
                                effect_card);
                        }
                    }
                    break;
                case OfficialArea::kPlayer:
                    ok = official_append_target(
                        state,
                        rules,
                        output,
                        OfficialCardRefPod{static_cast<std::uint16_t>(1 + player)},
                        target,
                        effect_card,
                        target.values[kTargetConditionCount] == 0);
                    break;
                default:
                    official_pod_fail(
                        state, OfficialPodError::kUnsupportedTarget, static_cast<int>(area));
                    return false;
            }
            if (!ok || !official_pod_ok(state)) return false;
        }
    }
    return official_pod_ok(state);
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_TARGET_HD
