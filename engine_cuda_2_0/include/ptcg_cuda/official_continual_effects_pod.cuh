#pragma once

#include <cstdint>

#include "ptcg_cuda/official_effects_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_CONTINUAL_HD __host__ __device__
#else
#define PTCG_OFFICIAL_CONTINUAL_HD
#endif

namespace ptcg::cuda_engine {

PTCG_OFFICIAL_CONTINUAL_HD inline std::int32_t official_continual_packed_i16(
    const OfficialCardStatePod& card,
    std::uint32_t byte_offset) {
    const std::uint32_t word = byte_offset / 8U;
    const std::uint32_t shift = (byte_offset % 8U) * 8U;
    return static_cast<std::int16_t>(
        (card.continual_state[word] >> shift) & 0xffffU);
}

PTCG_OFFICIAL_CONTINUAL_HD inline void official_continual_set_i16(
    OfficialCardStatePod* card,
    std::uint32_t byte_offset,
    std::int32_t value) {
    const std::uint32_t word = byte_offset / 8U;
    const std::uint32_t shift = (byte_offset % 8U) * 8U;
    const std::uint64_t mask = 0xffffULL << shift;
    card->continual_state[word] = (card->continual_state[word] & ~mask)
        | (static_cast<std::uint64_t>(static_cast<std::uint16_t>(value)) << shift);
}

PTCG_OFFICIAL_CONTINUAL_HD inline void official_continual_add_i16(
    OfficialCardStatePod* card,
    std::uint32_t byte_offset,
    std::int32_t value) {
    official_continual_set_i16(
        card,
        byte_offset,
        official_clamp_i32(
            official_continual_packed_i16(*card, byte_offset) + value,
            -30000,
            30000));
}

PTCG_OFFICIAL_CONTINUAL_HD inline void official_continual_set_i8(
    OfficialCardStatePod* card,
    std::uint32_t byte_offset,
    std::int32_t value) {
    const std::uint32_t word = byte_offset / 8U;
    const std::uint32_t shift = (byte_offset % 8U) * 8U;
    const std::uint64_t mask = 0xffULL << shift;
    card->continual_state[word] = (card->continual_state[word] & ~mask)
        | (static_cast<std::uint64_t>(static_cast<std::uint8_t>(value)) << shift);
}

PTCG_OFFICIAL_CONTINUAL_HD inline void official_continual_add_i8(
    OfficialCardStatePod* card,
    std::uint32_t byte_offset,
    std::int32_t value,
    std::int32_t low,
    std::int32_t high) {
    official_continual_set_i8(
        card,
        byte_offset,
        official_clamp_i32(
            official_continual_i8(*card, byte_offset) + value, low, high));
}

PTCG_OFFICIAL_CONTINUAL_HD inline void official_continual_set_flag(
    OfficialCardStatePod* card,
    std::uint32_t bit) {
    card->continual_state[4] |= 1ULL << bit;
}

PTCG_OFFICIAL_CONTINUAL_HD inline std::int32_t official_continual_player(
    std::int32_t effect_player,
    std::uint8_t mask,
    std::int32_t ordinal) {
    if (mask == 1) return ordinal == 0 ? effect_player : -1;
    if (mask == 2) return ordinal == 0 ? 1 - effect_player : -1;
    if (mask == 3) return ordinal < 2 ? ordinal : -1;
    return -1;
}

PTCG_OFFICIAL_CONTINUAL_HD inline bool official_continual_no_effect_active(
    const OfficialStatePod& state,
    std::int32_t player,
    OfficialCardRefPod effect_card) {
    if (player < 0 || player > 1 || state.players[player].active.count == 0) return false;
    const OfficialCardStatePod* source = official_pod_card(&state, effect_card);
    if (source != nullptr
        && source->area == static_cast<std::uint8_t>(OfficialArea::kStadium)) return false;
    const OfficialCardStatePod* active = official_pod_card(
        &state, state.players[player].active.values[0]);
    return active != nullptr && official_continual_flag(*active, 12);
}

PTCG_OFFICIAL_CONTINUAL_HD inline void official_continual_sync_runtime_flag(
    OfficialCardStatePod* card,
    std::uint32_t bit) {
    if (bit == 13) card->runtime_flags |= kCardNoSpecialCondition;
    else if (bit == 14) card->runtime_flags |= kCardNoSleepParalyzeConfuse;
    else if (bit == 15) card->runtime_flags |= kCardNoSleep;
    else if (bit == 31) card->runtime_flags |= kCardKoByDamageToHand;
}

PTCG_OFFICIAL_CONTINUAL_HD inline OfficialEffectApplyResult
official_apply_continual_effect_primitive(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialEffectRule& effect,
    std::int32_t effect_player,
    OfficialCardRefPod effect_card) {
    if (!official_pod_ok(state)) return OfficialEffectApplyResult::kError;
    const auto type = static_cast<OfficialEffectTypeId>(effect.values[kEffectType]);
    const std::int32_t value = official_effect_value(*state, effect, 0);

    if (type == OfficialEffectTypeId::kContinualEffectSeparator) {
        return OfficialEffectApplyResult::kApplied;
    }

    if (type >= OfficialEffectTypeId::kMaxHpChange
        && type <= OfficialEffectTypeId::kTakeEnemy4TypePokemonAttackDamageChange) {
        std::uint32_t byte_offset = 0;
        if (type == OfficialEffectTypeId::kDamageChange) byte_offset = 2;
        else if (type == OfficialEffectTypeId::kDamageChangeActive) byte_offset = 4;
        else if (type == OfficialEffectTypeId::kDamageChangeEx) byte_offset = 6;
        else if (type == OfficialEffectTypeId::kDamageChangeAbility) byte_offset = 8;
        else if (type == OfficialEffectTypeId::kDamageChangeEvolved) byte_offset = 10;
        else if (type == OfficialEffectTypeId::kDamageChangeEnemyTakenPrize) byte_offset = 12;
        else if (type == OfficialEffectTypeId::kTakeDamageChange) byte_offset = 14;
        else if (type == OfficialEffectTypeId::kTakeEnemyAttackDamageChange) byte_offset = 16;
        else if (type == OfficialEffectTypeId::kTakeEnemyAbilityPokemonAttackDamageChange) byte_offset = 18;
        else if (type == OfficialEffectTypeId::kTakeEnemyFireOrWaterPokemonAttackDamageChange) byte_offset = 20;
        else if (type == OfficialEffectTypeId::kTakeEnemy4TypePokemonAttackDamageChange) byte_offset = 22;
        for (std::uint16_t index = 0; index < state->targets.count; ++index) {
            const OfficialAreaRefPod target = state->targets.values[index];
            if (!official_pod_area_ref_valid(state, target)) continue;
            OfficialCardStatePod* card = official_pod_card(state, target.card);
            std::int32_t change = value;
            if (type == OfficialEffectTypeId::kMaxHpChangeFighting) {
                change *= official_attached_energy_cards(
                    *state, target.card, 32, false, rules);
            }
            official_continual_add_i16(card, byte_offset, change);
            if (byte_offset == 0) card->hp_change = static_cast<std::int16_t>(
                official_continual_packed_i16(*card, 0));
        }
        return official_pod_ok(state)
            ? OfficialEffectApplyResult::kApplied : OfficialEffectApplyResult::kError;
    }

    switch (type) {
        case OfficialEffectTypeId::kNoDamageGreaterEqual:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)) {
                    official_continual_set_i16(
                        official_pod_card(state, target.card), 24, value);
                }
            }
            break;
        case OfficialEffectTypeId::kRetreatCostChange:
        case OfficialEffectTypeId::kAttackCostChangeColorless:
        case OfficialEffectTypeId::kAttackCostDown:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                const std::uint32_t offset = type == OfficialEffectTypeId::kRetreatCostChange
                    ? 26U
                    : (type == OfficialEffectTypeId::kAttackCostChangeColorless ? 27U : 28U);
                official_continual_add_i8(
                    official_pod_card(state, target.card),
                    offset,
                    value,
                    type == OfficialEffectTypeId::kAttackCostDown ? 0 : -100,
                    100);
            }
            break;
        case OfficialEffectTypeId::kAttackCostDownColorlessTargetCount: {
            OfficialCardStatePod* card = official_pod_card(state, effect_card);
            if (card != nullptr) {
                official_continual_add_i8(
                    card, 27, -static_cast<std::int32_t>(state->targets.count), -100, 100);
            }
            break;
        }
        case OfficialEffectTypeId::kAttackCostDownColorlessOwnAttackEnemyTakenPrize:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                const std::int32_t taken = kOfficialInitialPrizeCount
                    - state->players[1 - effect_player].prize.count;
                official_continual_add_i8(
                    official_pod_card(state, target.card), 29, taken, 0, 100);
            }
            break;
        case OfficialEffectTypeId::kAddEnergyType:
        case OfficialEffectTypeId::kSetWeakness:
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (official_pod_area_ref_valid(state, target)) {
                    official_continual_set_i8(
                        official_pod_card(state, target.card),
                        type == OfficialEffectTypeId::kAddEnergyType ? 30 : 31,
                        value);
                }
            }
            break;
        case OfficialEffectTypeId::kNoAbility:
        case OfficialEffectTypeId::kNoKoMeAbility:
        case OfficialEffectTypeId::kNoDamageEnemyAttack:
        case OfficialEffectTypeId::kNoDamageEnemyAbilityPokemonAttack:
        case OfficialEffectTypeId::kNoDamageEnemyExAttack:
        case OfficialEffectTypeId::kNoDamageEnemyBasicExAttack:
        case OfficialEffectTypeId::kNoDamageAndEffectEnemyTerastalAttack:
        case OfficialEffectTypeId::kNoDamageAndEffectEnemySpecialEnergyAttack:
        case OfficialEffectTypeId::kNoEffectEnemyAttack:
        case OfficialEffectTypeId::kNoDamageAndEffectEnemyAttack:
        case OfficialEffectTypeId::kNoEffectEnemyItem:
        case OfficialEffectTypeId::kNoEffectEnemySupporter:
        case OfficialEffectTypeId::kNoDamageCounterEnemyAttackAbility:
        case OfficialEffectTypeId::kNoEnemyAbility:
        case OfficialEffectTypeId::kNoSpecialCondition:
        case OfficialEffectTypeId::kNoSleepParalyzeConfuse:
        case OfficialEffectTypeId::kNoSleep:
        case OfficialEffectTypeId::kNoRetreatCost:
        case OfficialEffectTypeId::kNoPrizeEx:
        case OfficialEffectTypeId::kNotRecoverConfuseEvolve:
        case OfficialEffectTypeId::kCanUsePreEvolutionAttack:
        case OfficialEffectTypeId::kCanEvolveAppearTurn:
        case OfficialEffectTypeId::kCanEvolveGrassAppearTurn:
        case OfficialEffectTypeId::kCanAttackFirst:
        case OfficialEffectTypeId::kCannotRetreat:
        case OfficialEffectTypeId::kCannotAttack:
        case OfficialEffectTypeId::kCannotToHand:
        case OfficialEffectTypeId::kCannotMoveDamageCounter:
        case OfficialEffectTypeId::kAttackEnergyColoressOne:
        case OfficialEffectTypeId::kAttackEnergyPsychicOne:
        case OfficialEffectTypeId::kDoubleGrassEnergy:
        case OfficialEffectTypeId::kNoDamageCoin:
        case OfficialEffectTypeId::kKoByDamageToHand:
        case OfficialEffectTypeId::kBasicPrizePlus1:
        case OfficialEffectTypeId::kDoubleAttack:
        case OfficialEffectTypeId::kTool2:
        case OfficialEffectTypeId::kTool4:
        case OfficialEffectTypeId::kTechnicalMachine:
        case OfficialEffectTypeId::kSpecialFlagTool:
        case OfficialEffectTypeId::kRainbowDna:
        case OfficialEffectTypeId::kCanPlay: {
            const std::uint32_t type_offset = static_cast<std::uint32_t>(type)
                - static_cast<std::uint32_t>(OfficialEffectTypeId::kNoAbility);
            std::uint32_t bit = type_offset;
            if (type == OfficialEffectTypeId::kNoDamageEnemyAttack
                || type == OfficialEffectTypeId::kNoDamageAndEffectEnemyAttack) {
                bit = 7;
            } else if (type == OfficialEffectTypeId::kNoEffectEnemyAttack) {
                bit = 8;
            } else if (type >= OfficialEffectTypeId::kNoDamageEnemyAbilityPokemonAttack) {
                bit = type_offset - 1U;
            }
            for (std::uint16_t index = 0; index < state->targets.count; ++index) {
                const OfficialAreaRefPod target = state->targets.values[index];
                if (!official_pod_area_ref_valid(state, target)) continue;
                OfficialCardStatePod* card = official_pod_card(state, target.card);
                official_continual_set_flag(card, bit);
                official_continual_sync_runtime_flag(card, bit);
                if (type == OfficialEffectTypeId::kNoDamageAndEffectEnemyAttack) {
                    official_continual_set_flag(card, 8);
                }
            }
            break;
        }
        case OfficialEffectTypeId::kPoisonDamageChange:
        case OfficialEffectTypeId::kBurnDamageChange:
        case OfficialEffectTypeId::kPoisonDamageChangeNotDarkness:
        case OfficialEffectTypeId::kBenchCapacity:
        case OfficialEffectTypeId::kCannotPlayItem:
        case OfficialEffectTypeId::kCannotPlayStadium:
        case OfficialEffectTypeId::kCannotPlayTool:
        case OfficialEffectTypeId::kCannotPlayAceSpec:
        case OfficialEffectTypeId::kCannotPlayAbilityPokemonNotRocket:
        case OfficialEffectTypeId::kCannotTrashToHandAbilityOrTrainers: {
            const std::uint8_t mask = official_effect_player_mask(state, rules, effect);
            for (int ordinal = 0; ordinal < 2; ++ordinal) {
                const int player = official_continual_player(effect_player, mask, ordinal);
                if (player < 0) break;
                const bool blocked_player_effect =
                    type == OfficialEffectTypeId::kPoisonDamageChange
                    || type == OfficialEffectTypeId::kBurnDamageChange
                    || type == OfficialEffectTypeId::kPoisonDamageChangeNotDarkness;
                if (blocked_player_effect
                    && official_continual_no_effect_active(*state, player, effect_card)) continue;
                OfficialPlayerStatePod* ps = &state->players[player];
                if (type == OfficialEffectTypeId::kPoisonDamageChange) {
                    const std::int16_t before = static_cast<std::int16_t>(
                        ps->continual_state & 0xffffULL);
                    ps->continual_state = (ps->continual_state & ~0xffffULL)
                        | static_cast<std::uint16_t>(before + value);
                } else if (type == OfficialEffectTypeId::kBurnDamageChange) {
                    const std::int16_t before = static_cast<std::int16_t>(
                        (ps->continual_state >> 16U) & 0xffffULL);
                    ps->continual_state = (ps->continual_state & ~(0xffffULL << 16U))
                        | (static_cast<std::uint64_t>(
                            static_cast<std::uint16_t>(before + value)) << 16U);
                } else if (type == OfficialEffectTypeId::kPoisonDamageChangeNotDarkness) {
                    const std::int8_t before = static_cast<std::int8_t>(
                        (ps->continual_state >> 32U) & 0xffULL);
                    ps->continual_state = (ps->continual_state & ~(0xffULL << 32U))
                        | (static_cast<std::uint64_t>(
                            static_cast<std::uint8_t>(before + value)) << 32U);
                } else if (type == OfficialEffectTypeId::kBenchCapacity) {
                    const std::uint64_t before = (ps->continual_state >> 40U) & 0xfULL;
                    if (before == 0 || before > static_cast<std::uint64_t>(value)) {
                        ps->continual_state = (ps->continual_state & ~(0xfULL << 40U))
                            | ((static_cast<std::uint64_t>(value) & 0xfULL) << 40U);
                    }
                } else {
                    std::uint32_t bit = 44;
                    if (type == OfficialEffectTypeId::kCannotPlayStadium) bit = 45;
                    else if (type == OfficialEffectTypeId::kCannotPlayTool) bit = 46;
                    else if (type == OfficialEffectTypeId::kCannotPlayAceSpec) bit = 47;
                    else if (type == OfficialEffectTypeId::kCannotPlayAbilityPokemonNotRocket) bit = 48;
                    else if (type == OfficialEffectTypeId::kCannotTrashToHandAbilityOrTrainers) bit = 49;
                    ps->continual_state |= 1ULL << bit;
                }
            }
            break;
        }
        case OfficialEffectTypeId::kNoToolEffect:
            state->continual_state |= 1U;
            break;
        default:
            official_pod_fail(
                state, OfficialPodError::kUnsupportedEffect, effect.values[kEffectType]);
            return OfficialEffectApplyResult::kUnsupported;
    }
    return official_pod_ok(state)
        ? OfficialEffectApplyResult::kApplied : OfficialEffectApplyResult::kError;
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_CONTINUAL_HD
