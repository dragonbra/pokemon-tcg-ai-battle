#pragma once

#include <cstdint>

#include "ptcg_cuda/official_attack_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_MAIN_HD __host__ __device__
#else
#define PTCG_OFFICIAL_MAIN_HD
#endif

namespace ptcg::cuda_engine {

enum class OfficialMainResult : std::int32_t {
    kComplete = 0,
    kNeedsAction = 1,
    kError = 2,
};

constexpr std::uint8_t kOfficialSelectContextMain = 1;
constexpr std::uint8_t kOfficialAttackReturnToMainFlag = 1U << 7U;
constexpr std::uint8_t kOfficialTurnReturnToMainFlag = 1U << 7U;
constexpr std::uint8_t kOfficialRefreshReturnToMainFlag = 1U << 6U;
constexpr std::uint8_t kOfficialPlayEffectReturnToMainFlag = 1U << 5U;
constexpr std::uint8_t kOfficialAbilityReturnToMainFlag = 1U << 4U;
constexpr std::uint8_t kOfficialRetreatFailFlag = 1U << 3U;
constexpr std::uint8_t kOfficialRefreshTriggerReturnToMainFlag = 1U << 2U;
constexpr std::uint8_t kOfficialRetreatTriggerPendingFlag = 1U << 1U;
constexpr std::uint8_t kOfficialRetreatRefreshPendingFlag = 1U << 0U;
constexpr std::uint64_t kOfficialPlayerCannotPlayAceSpecFlag = 1ULL << 47U;
constexpr std::uint64_t kOfficialPlayerCannotPlayAbilityPokemonFlag = 1ULL << 48U;
constexpr std::uint64_t kOfficialPlayerCannotPlayStadiumFlag = 1ULL << 45U;
constexpr std::uint64_t kOfficialPlayerCannotPlayToolFlag = 1ULL << 46U;
constexpr std::uint64_t kOfficialPlayerCannotPlayItemFlag = 1ULL << 44U;
constexpr std::uint32_t kOfficialPlayerThisTurnCannotPlayStadiumFlag = 1U << 19U;
constexpr std::uint32_t kOfficialPlayerThisTurnCannotPlayItemFlag = 1U << 17U;
constexpr std::uint32_t kOfficialPlayerThisTurnCannotPlaySupporterFlag = 1U << 18U;
constexpr std::uint32_t kOfficialPlayerThisTurnCannotPlaySpecialEnergyFlag =
    1U << 20U;
constexpr std::uint8_t kOfficialStadiumPlayedFlag = 1U << 1U;
constexpr std::uint8_t kOfficialSupporterPlayedFlag = 1U << 0U;
constexpr std::uint8_t kOfficialEnergyPlayedFlag = 1U << 2U;
constexpr std::uint64_t kOfficialCardAceSpecFlag = 1ULL << 26U;
constexpr std::uint64_t kOfficialCardToBenchFlag = 1ULL << 6U;
constexpr std::uint64_t kOfficialCardCanPlayFirstTurnFlag = 1ULL << 3U;
constexpr std::uint64_t kOfficialCardTransformOnlyFlag = 1ULL << 4U;
constexpr std::uint64_t kOfficialCardCanTrashFlag = 1ULL << 5U;
constexpr std::uint32_t kOfficialCardCannotHandAttachEnergyFlag = 1U << 1U;
constexpr std::uint8_t kOfficialTriggerHandToBench = 3;
constexpr std::uint8_t kOfficialTriggerToBenchMyTurn = 5;
constexpr std::uint8_t kOfficialTriggerEnergyAttachFromHand = 9;
constexpr std::uint8_t kOfficialTriggerActiveToBench = 4;
constexpr std::uint8_t kOfficialTriggerBenchToActive = 6;
constexpr std::uint8_t kOfficialTriggerPreRetreat = 19;
constexpr std::uint8_t kOfficialTriggerAttach = 20;
constexpr std::uint64_t kOfficialSkillAttachBenchFlag = 1ULL << 5U;
constexpr std::uint32_t kOfficialCardCanEvolveAppearTurnBit =
    static_cast<std::uint32_t>(OfficialEffectTypeId::kCanEvolveAppearTurn)
    - static_cast<std::uint32_t>(OfficialEffectTypeId::kNoAbility) - 1U;
constexpr std::uint32_t kOfficialCardCanEvolveGrassAppearTurnBit =
    static_cast<std::uint32_t>(OfficialEffectTypeId::kCanEvolveGrassAppearTurn)
    - static_cast<std::uint32_t>(OfficialEffectTypeId::kNoAbility) - 1U;
constexpr std::uint32_t kOfficialCardRainbowDnaBit =
    static_cast<std::uint32_t>(OfficialEffectTypeId::kRainbowDna)
    - static_cast<std::uint32_t>(OfficialEffectTypeId::kNoAbility) - 1U;
constexpr std::uint32_t kOfficialPlayerThisTurnCannotEvolveFlag = 1U << 21U;
constexpr std::int32_t kOfficialAngeFloetteCardId = 1429;
constexpr std::uint64_t kOfficialSkillMainAbilityFlag = 1ULL << 0U;
constexpr std::uint64_t kOfficialSkillKoMeAbilityFlag = 1ULL << 6U;

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult official_main_after_flow(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult official_main_begin_refresh(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult
official_main_continue_retreat_after_refresh(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult
official_main_continue_retreat_after_triggers(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

PTCG_OFFICIAL_MAIN_HD inline bool official_main_add_attack_options(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod attacker_ref,
    std::int32_t bench_index) {
    const OfficialCardStatePod* attacker = official_pod_card(state, attacker_ref);
    const OfficialCardRule* master = attacker == nullptr ? nullptr : official_card_rule(
        rules, static_cast<std::uint32_t>(attacker->card_id));
    if (master == nullptr) {
        official_pod_fail(
            state,
            OfficialPodError::kRulePackBounds,
            attacker == nullptr ? 0 : attacker->card_id);
        return false;
    }
    std::int32_t offset = 0;
    std::int32_t count = 0;
    if (!official_attack_card_attack_range(
            state, rules, *master, &offset, &count)) {
        return false;
    }
    const std::int32_t player = attacker->player;

    // The CPU engine's main attack list is not limited to the attacks printed
    // on the active card.  For attacks such as Zoroark's "copy an N Pokemon"
    // and Mimikyu's "copy the active Terastal Pokemon", SetAttackEnergy
    // expands the source attack into one option per eligible copied attack,
    // retaining the source attack id in params[1].  Keep that expansion here
    // as well; otherwise CUDA exposes only the source (e.g. 403/612), which
    // makes the official and CUDA option sets diverge before the attack flow
    // even starts.
    for (std::int32_t index = 0; index < count; ++index) {
        const std::int32_t attack_id = static_cast<std::int32_t>(
            rules.card_attack_ids[offset + index]);
        const OfficialAttackRule* attack = official_attack_rule(
            rules, static_cast<std::uint32_t>(attack_id));
        if (attack == nullptr) {
            official_pod_fail(
                state, OfficialPodError::kRulePackBounds, attack_id);
            return false;
        }
        if (bench_index >= 0 && (attack->flags & kAttackCanUseBench) == 0) {
            continue;
        }

        const std::uint64_t copy_flags = attack->flags
            & (kAttackCopyEnemy | kAttackCopyEnemyCoin
                | kAttackCopyEnemyTerastal | kAttackCopyBenchN);
        if (copy_flags == 0) {
            if (!official_attack_initial_action_is_legal(
                    state, rules, attacker_ref, *attack, 0)) {
                if (!official_pod_ok(state)) return false;
                continue;
            }
            OfficialSelectOptionPod option{};
            option.type = static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kAttack);
            option.params[0] = static_cast<std::int16_t>(attack_id);
            option.params[1] = 0;
            option.params[2] = static_cast<std::int16_t>(bench_index);
            option.option_equiv = static_cast<std::uint16_t>(attack_id);
            if (!official_pod_push(
                    state, &state->options, option,
                    OfficialPodError::kOptionOverflow)) {
                return false;
            }
            continue;
        }

        // Match the CPU SetAttackEnergy extraction rules.  Copy-enemy
        // variants read attacks from the opponent's active Pokemon; the
        // Bench-N variant reads attacks from each benched N Pokemon.  A
        // Terastal copy is available only when the opponent's active master
        // carries the Terastal flag (bit 0 in the packed card flags).
        const OfficialCardRefPod* source_values = nullptr;
        std::uint16_t source_count = 0;
        OfficialCardRefPod enemy_active{};
        if ((copy_flags & (kAttackCopyEnemy | kAttackCopyEnemyCoin
                | kAttackCopyEnemyTerastal)) != 0) {
            const std::int32_t enemy = 1 - player;
            const OfficialPlayerStatePod& enemy_player = state->players[enemy];
            if (enemy_player.active.count == 0) continue;
            enemy_active = enemy_player.active.values[0];
            const OfficialCardStatePod* enemy_card = official_pod_card(
                state, enemy_active);
            const OfficialCardRule* enemy_master = enemy_card == nullptr ? nullptr
                : official_card_rule(
                    rules, static_cast<std::uint32_t>(enemy_card->card_id));
            if (enemy_master == nullptr) {
                official_pod_fail(
                    state, OfficialPodError::kRulePackBounds,
                    enemy_card == nullptr ? 0 : enemy_card->card_id);
                return false;
            }
            if ((copy_flags & kAttackCopyEnemyTerastal) != 0
                && (enemy_master->flags & 1ULL) == 0) {
                continue;
            }
            source_values = enemy_player.active.values;
            source_count = enemy_player.active.count;
        } else {
            const OfficialPlayerStatePod& own_player = state->players[player];
            source_values = own_player.bench.values;
            source_count = own_player.bench.count;
        }

        for (std::uint16_t source_index = 0;
             source_index < source_count;
             ++source_index) {
            const OfficialCardRefPod source_ref = source_values[source_index];
            const OfficialCardStatePod* source_card = official_pod_card(
                state, source_ref);
            const OfficialCardRule* source_master = source_card == nullptr ? nullptr
                : official_card_rule(
                    rules, static_cast<std::uint32_t>(source_card->card_id));
            if (source_master == nullptr) {
                official_pod_fail(
                    state, OfficialPodError::kRulePackBounds,
                    source_card == nullptr ? 0 : source_card->card_id);
                return false;
            }
            if ((copy_flags & kAttackCopyBenchN) != 0
                && (source_master->flags & (1ULL << 16U)) == 0) {
                continue;
            }
            std::int32_t copied_offset = 0;
            std::int32_t copied_count = 0;
            if (!official_attack_card_attack_range(
                    state, rules, *source_master,
                    &copied_offset, &copied_count)) {
                return false;
            }
            for (std::int32_t copied_index = 0;
                 copied_index < copied_count;
                 ++copied_index) {
                const std::int32_t copied_attack_id = static_cast<std::int32_t>(
                    rules.card_attack_ids[copied_offset + copied_index]);
                const OfficialAttackRule* copied_attack = official_attack_rule(
                    rules, static_cast<std::uint32_t>(copied_attack_id));
                if (copied_attack == nullptr) {
                    official_pod_fail(
                        state, OfficialPodError::kRulePackBounds,
                        copied_attack_id);
                    return false;
                }
                if (!official_attack_initial_action_is_legal(
                        state, rules, attacker_ref, *copied_attack, attack_id)) {
                    if (!official_pod_ok(state)) return false;
                    continue;
                }
                OfficialSelectOptionPod option{};
                option.type = static_cast<std::uint8_t>(
                    OfficialSelectOptionTypeId::kAttack);
                option.params[0] = static_cast<std::int16_t>(copied_attack_id);
                option.params[1] = static_cast<std::int16_t>(attack_id);
                option.params[2] = static_cast<std::int16_t>(bench_index);
                option.option_equiv = static_cast<std::uint16_t>(copied_attack_id);
                if (!official_pod_push(
                        state, &state->options, option,
                        OfficialPodError::kOptionOverflow)) {
                    return false;
                }
            }
        }
    }
    return true;
}

PTCG_OFFICIAL_MAIN_HD inline std::int32_t official_main_bench_capacity(
    const OfficialPlayerStatePod& player) {
    const std::int32_t changed = static_cast<std::int32_t>(
        (player.continual_state >> 40U) & 0xfU);
    return changed == 0 ? 5 : changed;
}

PTCG_OFFICIAL_MAIN_HD inline std::int32_t official_main_tool_capacity(
    const OfficialCardStatePod& pokemon) {
    if (official_continual_flag(pokemon, 35)) return 4;
    if (official_continual_flag(pokemon, 34)) return 2;
    return 1;
}

PTCG_OFFICIAL_MAIN_HD inline std::int32_t official_main_attached_tool_count(
    const OfficialStatePod& state,
    std::int32_t player,
    const OfficialCardStatePod& pokemon) {
    if (player < 0 || player > 1) return 0;
    std::int32_t count = 0;
    const OfficialPlayerStatePod& ps = state.players[player];
    for (std::uint16_t index = 0; index < ps.tool.count; ++index) {
        const OfficialCardStatePod* tool = official_pod_card(
            &state, ps.tool.values[index]);
        if (tool != nullptr
            && tool->attach_move_counter == pokemon.move_counter) ++count;
    }
    return count;
}

PTCG_OFFICIAL_MAIN_HD inline bool official_main_can_evolve_candidate(
    const OfficialStatePod& state,
    const OfficialRulePackView& rules,
    const OfficialCardRule& evolution,
    const OfficialCardStatePod& target,
    const OfficialCardRule& target_master);

PTCG_OFFICIAL_MAIN_HD inline bool official_main_add_play_options(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    const OfficialPlayerStatePod& ps = state->players[player];
    for (std::uint16_t hand_index = 0; hand_index < ps.hand.count; ++hand_index) {
        const OfficialCardStatePod* card = official_pod_card(
            state, ps.hand.values[hand_index]);
        const OfficialCardRule* master = card == nullptr ? nullptr
            : official_card_rule(rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) {
            official_pod_fail(
                state,
                OfficialPodError::kRulePackBounds,
                card == nullptr ? 0 : card->card_id);
            return false;
        }
        const bool blocks_ace_spec =
            (ps.continual_state & kOfficialPlayerCannotPlayAceSpecFlag) != 0
            && (master->flags & kOfficialCardAceSpecFlag) != 0;
        if (blocks_ace_spec) continue;

        const std::int32_t card_type = master->values[kCardType];
        if (card_type == 0 && master->values[kCardEvolutionType] != 1) {
            const bool blocked_ability_pokemon =
                (ps.continual_state
                    & kOfficialPlayerCannotPlayAbilityPokemonFlag) != 0
                && master->values[kCardAbilityId] != 0
                && (master->flags & kOfficialCardTeamRocketFlag) == 0;
            if ((ps.this_turn & kOfficialPlayerThisTurnCannotEvolveFlag) == 0
                && !blocked_ability_pokemon
                && (master->flags & kOfficialCardTransformOnlyFlag) == 0) {
                const std::uint16_t target_count = static_cast<std::uint16_t>(
                    ps.active.count + ps.bench.count);
                for (std::uint16_t target_index = 0;
                     target_index < target_count;
                     ++target_index) {
                    const bool active = target_index < ps.active.count;
                    const std::uint16_t zone_index = active
                        ? target_index
                        : static_cast<std::uint16_t>(
                            target_index - ps.active.count);
                    const OfficialCardRefPod target_ref = active
                        ? ps.active.values[zone_index]
                        : ps.bench.values[zone_index];
                    const OfficialCardStatePod* target = official_pod_card(
                        state, target_ref);
                    const OfficialCardRule* target_master = target == nullptr
                        ? nullptr
                        : official_card_rule(
                            rules, static_cast<std::uint32_t>(target->card_id));
                    if (target_master == nullptr) {
                        official_pod_fail(
                            state,
                            OfficialPodError::kRulePackBounds,
                            target == nullptr ? 0 : target->card_id);
                        return false;
                    }
                    const bool can_evolve_appear = official_continual_flag(
                        *target, kOfficialCardCanEvolveAppearTurnBit);
                    if (state->turn <= 2 && !can_evolve_appear) continue;
                    if ((target->runtime_flags & kCardAppear) != 0
                        && !can_evolve_appear) {
                        const bool can_evolve_grass_appear = official_continual_flag(
                            *target,
                            kOfficialCardCanEvolveGrassAppearTurnBit);
                        if (!can_evolve_grass_appear
                            || (master->values[kCardEnergyType] & 1) == 0) {
                            continue;
                        }
                    }
                    if (!official_main_can_evolve_candidate(
                            *state, rules, *master, *target, *target_master)) {
                        continue;
                    }
                    OfficialSelectOptionPod option{};
                    option.type = static_cast<std::uint8_t>(
                        OfficialSelectOptionTypeId::kEvolve);
                    option.params[0] = static_cast<std::int16_t>(OfficialArea::kHand);
                    option.params[1] = static_cast<std::int16_t>(hand_index);
                    option.params[2] = static_cast<std::int16_t>(
                        active ? OfficialArea::kActive : OfficialArea::kBench);
                    option.params[3] = static_cast<std::int16_t>(zone_index);
                    option.resolved_card = ps.hand.values[hand_index].index;
                    option.option_equiv = static_cast<std::uint16_t>(
                        (master->values[kCardId] * 131 + target->card_id) & 0xffff);
                    if (!official_pod_push(
                            state,
                            &state->options,
                            option,
                            OfficialPodError::kOptionOverflow)) {
                        return false;
                    }
                }
            }
        }
        if (card_type == 5 || card_type == 6) {
            if ((state->turn_state & kOfficialEnergyPlayedFlag) != 0
                || (card_type == 6
                    && (ps.this_turn
                        & kOfficialPlayerThisTurnCannotPlaySpecialEnergyFlag) != 0)) {
                continue;
            }
            const bool only_team_rocket =
                (master->flags & kOfficialCardOnlyTeamRocketFlag) != 0;
            const std::uint16_t target_count = static_cast<std::uint16_t>(
                ps.active.count + ps.bench.count);
            for (std::uint16_t target_index = 0;
                 target_index < target_count;
                 ++target_index) {
                const bool active = target_index < ps.active.count;
                const std::uint16_t zone_index = active
                    ? target_index
                    : static_cast<std::uint16_t>(target_index - ps.active.count);
                const OfficialCardRefPod target_ref = active
                    ? ps.active.values[zone_index]
                    : ps.bench.values[zone_index];
                const OfficialCardStatePod* target = official_pod_card(
                    state, target_ref);
                const OfficialCardRule* target_master = target == nullptr
                    ? nullptr
                    : official_card_rule(
                        rules, static_cast<std::uint32_t>(target->card_id));
                if (target_master == nullptr) {
                    official_pod_fail(
                        state,
                        OfficialPodError::kRulePackBounds,
                        target == nullptr ? 0 : target->card_id);
                    return false;
                }
                if ((target->this_turn[3]
                        & kOfficialCardCannotHandAttachEnergyFlag) != 0
                    || (only_team_rocket
                        && (target_master->flags
                            & kOfficialCardTeamRocketFlag) == 0)) {
                    continue;
                }
                OfficialSelectOptionPod option{};
                option.type = static_cast<std::uint8_t>(
                    OfficialSelectOptionTypeId::kAttach);
                option.params[0] = static_cast<std::int16_t>(OfficialArea::kHand);
                option.params[1] = static_cast<std::int16_t>(hand_index);
                option.params[2] = static_cast<std::int16_t>(
                    active ? OfficialArea::kActive : OfficialArea::kBench);
                option.params[3] = static_cast<std::int16_t>(zone_index);
                option.resolved_card = ps.hand.values[hand_index].index;
                option.option_equiv = static_cast<std::uint16_t>(
                    (master->values[kCardId] * 131 + target->card_id) & 0xffff);
                if (!official_pod_push(
                        state,
                        &state->options,
                        option,
                        OfficialPodError::kOptionOverflow)) {
                    return false;
                }
            }
            continue;
        }
        if (card_type == 2) {
            if ((ps.continual_state & kOfficialPlayerCannotPlayToolFlag) != 0) {
                continue;
            }
            const std::uint16_t target_count = static_cast<std::uint16_t>(
                ps.active.count + ps.bench.count);
            for (std::uint16_t target_index = 0;
                 target_index < target_count;
                 ++target_index) {
                const bool active = target_index < ps.active.count;
                const std::uint16_t zone_index = active
                    ? target_index
                    : static_cast<std::uint16_t>(target_index - ps.active.count);
                const OfficialCardRefPod target_ref = active
                    ? ps.active.values[zone_index]
                    : ps.bench.values[zone_index];
                const OfficialCardStatePod* target = official_pod_card(
                    state, target_ref);
                if (target == nullptr) return false;
                if (official_main_attached_tool_count(*state, player, *target)
                    >= official_main_tool_capacity(*target)) {
                    continue;
                }
                OfficialSelectOptionPod option{};
                option.type = static_cast<std::uint8_t>(
                    OfficialSelectOptionTypeId::kAttach);
                option.params[0] = static_cast<std::int16_t>(OfficialArea::kHand);
                option.params[1] = static_cast<std::int16_t>(hand_index);
                option.params[2] = static_cast<std::int16_t>(
                    active ? OfficialArea::kActive : OfficialArea::kBench);
                option.params[3] = static_cast<std::int16_t>(zone_index);
                option.resolved_card = ps.hand.values[hand_index].index;
                option.option_equiv = static_cast<std::uint16_t>(
                    (master->values[kCardId] * 131 + target->card_id) & 0xffff);
                if (!official_pod_push(
                        state,
                        &state->options,
                        option,
                        OfficialPodError::kOptionOverflow)) {
                    return false;
                }
            }
            continue;
        }
        bool can_play = false;
        if (card_type == 0 && master->values[kCardEvolutionType] == 1
            && ps.bench.count < official_main_bench_capacity(ps)) {
            const bool blocks_ability_pokemon =
                (ps.continual_state
                    & kOfficialPlayerCannotPlayAbilityPokemonFlag) != 0
                && master->values[kCardAbilityId] != 0
                && (master->flags & kOfficialCardTeamRocketFlag) == 0;
            can_play = !blocks_ability_pokemon;
        } else if ((card_type == 1 || card_type == 3)
            && master->values[kCardPlayId] > 0) {
            const bool blocked_item = card_type == 1
                && ((ps.continual_state & kOfficialPlayerCannotPlayItemFlag) != 0
                    || (ps.this_turn
                        & kOfficialPlayerThisTurnCannotPlayItemFlag) != 0);
            const bool blocked_supporter = card_type == 3
                && (((state->turn_state & kOfficialSupporterPlayedFlag) != 0)
                    || (ps.this_turn
                        & kOfficialPlayerThisTurnCannotPlaySupporterFlag) != 0
                    || (state->turn <= 1
                        && (master->flags
                            & kOfficialCardCanPlayFirstTurnFlag) == 0));
            if (blocked_item || blocked_supporter) continue;
            const OfficialSkillRule* skill = official_skill_rule(
                rules, static_cast<std::uint32_t>(master->values[kCardPlayId]));
            if (skill == nullptr) {
                official_pod_fail(
                    state,
                    OfficialPodError::kRulePackBounds,
                    master->values[kCardPlayId]);
                return false;
            }
            bool satisfied = true;
            if (skill->values[kSkillSecondEffectStart] == 0
                && !official_resolver_satisfy_skill_conditions(
                    state,
                    rules,
                    *skill,
                    0,
                    official_pod_area_ref(state, ps.hand.values[hand_index]),
                    player,
                    &satisfied)) {
                return false;
            }
            can_play = satisfied;
        } else if (card_type == 4
            && master->values[kCardId] != kOfficialAngeFloetteCardId
            && (state->turn_state & kOfficialStadiumPlayedFlag) == 0
            && (ps.continual_state & kOfficialPlayerCannotPlayStadiumFlag) == 0
            && (ps.this_turn & kOfficialPlayerThisTurnCannotPlayStadiumFlag) == 0) {
            can_play = true;
            if (state->stadium.count > 0) {
                const OfficialCardStatePod* current = official_pod_card(
                    state, state->stadium.values[0]);
                const OfficialCardRule* current_master = current == nullptr ? nullptr
                    : official_card_rule(
                        rules, static_cast<std::uint32_t>(current->card_id));
                if (current_master == nullptr) {
                    official_pod_fail(
                        state,
                        OfficialPodError::kRulePackBounds,
                        current == nullptr ? 0 : current->card_id);
                    return false;
                }
                can_play = current_master->values[kCardNameId]
                    != master->values[kCardNameId];
            }
        }
        if (!can_play) continue;

        OfficialSelectOptionPod option{};
        option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kPlay);
        option.params[0] = static_cast<std::int16_t>(hand_index);
        option.resolved_card = ps.hand.values[hand_index].index;
        option.option_equiv = static_cast<std::uint16_t>(master->values[kCardId]);
        if (!official_pod_push(
                state, &state->options, option, OfficialPodError::kOptionOverflow)) {
            return false;
        }
    }
    return true;
}

PTCG_OFFICIAL_MAIN_HD inline bool official_main_can_evolve_candidate(
    const OfficialStatePod& state,
    const OfficialRulePackView& rules,
    const OfficialCardRule& evolution,
    const OfficialCardStatePod& target,
    const OfficialCardRule& target_master) {
    if (official_evolves_from(evolution, target_master, false)) return true;
    if (!official_continual_flag(target, kOfficialCardRainbowDnaBit)
        || state.turn <= 2
        || (target.runtime_flags & kCardAppear) != 0
        || evolution.values[kCardPokemonType] != 3) {
        return false;
    }
    const std::int32_t evolves_from = evolution.values[kCardEvolvesFromNameId];
    return evolves_from != 0 && official_name_set_contains(
        rules,
        evolves_from,
        kNameContainsOffset,
        kNameContainsCount,
        target.card_id);
}

PTCG_OFFICIAL_MAIN_HD inline bool official_main_skill_matches_area(
    const OfficialSkillRule& skill,
    OfficialArea area) {
    const std::int32_t count = skill.values[kSkillAreaCount];
    for (std::int32_t index = 0; index < count && index < 2; ++index) {
        if (skill.values[kSkillArea0 + index] == static_cast<std::int32_t>(area)) {
            return true;
        }
    }
    return false;
}

PTCG_OFFICIAL_MAIN_HD inline bool official_main_card_can_activate(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    OfficialArea area,
    std::int32_t use_player,
    bool* can_activate) {
    *can_activate = false;
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
    const std::int32_t skill_id = master->values[kCardAbilityId];
    if (skill_id <= 0 || official_continual_flag(*card, 0)) return true;
    const OfficialSkillRule* skill = official_skill_rule(
        rules, static_cast<std::uint32_t>(skill_id));
    if (skill == nullptr) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, skill_id);
        return false;
    }
    if ((skill->flags & kOfficialSkillMainAbilityFlag) == 0
        || !official_main_skill_matches_area(*skill, area)
        || (official_continual_flag(*card, 1)
            && (skill->flags & kOfficialSkillKoMeAbilityFlag) != 0)) {
        return true;
    }
    bool satisfied = false;
    if (!official_resolver_satisfy_skill_conditions(
            state,
            rules,
            *skill,
            0,
            official_pod_area_ref(state, ref),
            use_player,
            &satisfied)) {
        return false;
    }
    *can_activate = satisfied;
    return true;
}

PTCG_OFFICIAL_MAIN_HD inline bool official_main_add_ability_option(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    OfficialArea area,
    std::uint16_t index,
    std::int32_t use_player) {
    bool can_activate = false;
    if (!official_main_card_can_activate(
            state, rules, ref, area, use_player, &can_activate)) {
        return false;
    }
    if (!can_activate) return true;
    const OfficialCardStatePod* card = official_pod_card(state, ref);
    const OfficialCardRule* master = official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    OfficialSelectOptionPod option{};
    option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kAbility);
    option.params[0] = static_cast<std::int16_t>(area);
    option.params[1] = static_cast<std::int16_t>(index);
    option.resolved_card = ref.index;
    option.option_equiv = static_cast<std::uint16_t>(
        master->values[kCardAbilityId]);
    return official_pod_push(
        state, &state->options, option, OfficialPodError::kOptionOverflow);
}

PTCG_OFFICIAL_MAIN_HD inline bool official_main_add_discard_option(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    OfficialArea area,
    std::uint16_t index) {
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
    if ((master->flags & kOfficialCardCanTrashFlag) == 0) return true;
    OfficialSelectOptionPod option{};
    option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kDiscard);
    option.params[0] = static_cast<std::int16_t>(area);
    option.params[1] = static_cast<std::int16_t>(index);
    option.resolved_card = ref.index;
    option.option_equiv = static_cast<std::uint16_t>(master->values[kCardId]);
    return official_pod_push(
        state, &state->options, option, OfficialPodError::kOptionOverflow);
}

PTCG_OFFICIAL_MAIN_HD inline bool official_main_add_ability_options(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    const OfficialPlayerStatePod& ps = state->players[player];
    for (std::uint16_t index = 0; index < ps.active.count; ++index) {
        if (!official_main_add_ability_option(
                state,
                rules,
                ps.active.values[index],
                OfficialArea::kActive,
                index,
                player)) return false;
        if (!official_main_add_discard_option(
                state,
                rules,
                ps.active.values[index],
                OfficialArea::kActive,
                index)) return false;
    }
    for (std::uint16_t index = 0; index < ps.bench.count; ++index) {
        if (!official_main_add_ability_option(
                state,
                rules,
                ps.bench.values[index],
                OfficialArea::kBench,
                index,
                player)) return false;
        if (!official_main_add_discard_option(
                state,
                rules,
                ps.bench.values[index],
                OfficialArea::kBench,
                index)) return false;
    }
    for (std::uint16_t index = 0; index < state->stadium.count; ++index) {
        if (!official_main_add_ability_option(
                state,
                rules,
                state->stadium.values[index],
                OfficialArea::kStadium,
                index,
                player)) return false;
    }
    return true;
}

PTCG_OFFICIAL_MAIN_HD inline std::int32_t official_main_retreat_cost(
    const OfficialStatePod& state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref) {
    const OfficialCardStatePod* card = official_pod_card(&state, ref);
    const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    if (master == nullptr) return -1;
    std::int32_t cost = master->values[kCardRetreatCost]
        + official_continual_i8(*card, 26)
        + official_this_turn_i8(*card, 11);
    if (official_continual_flag(*card, 16)
        || master->values[kCardPokemonType] == 2) {
        cost = 0;
    }
    return cost < 0 ? 0 : cost;
}

PTCG_OFFICIAL_MAIN_HD inline std::int32_t official_main_attached_energy_units(
    const OfficialStatePod& state,
    const OfficialRulePackView& rules,
    std::int32_t player,
    OfficialCardRefPod pokemon_ref) {
    const OfficialCardStatePod* pokemon = official_pod_card(&state, pokemon_ref);
    if (pokemon == nullptr || player < 0 || player > 1) return -1;
    std::int32_t units = 0;
    const OfficialPlayerStatePod& ps = state.players[player];
    for (std::uint16_t index = 0; index < ps.energy.count; ++index) {
        const OfficialCardRefPod ref = ps.energy.values[index];
        const OfficialCardStatePod* energy = official_pod_card(&state, ref);
        if (energy == nullptr
            || energy->attach_move_counter != pokemon->move_counter) continue;
        units += official_energy_info(state, rules, ref, pokemon_ref).count;
    }
    return units;
}

PTCG_OFFICIAL_MAIN_HD inline bool official_main_can_retreat(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player,
    bool* can_retreat) {
    *can_retreat = false;
    if (player < 0 || player > 1) {
        official_pod_fail(state, OfficialPodError::kInvalidPlayer, player);
        return false;
    }
    const OfficialPlayerStatePod& ps = state->players[player];
    if ((state->turn_state & (1U << 3U)) != 0
        || ps.active.count == 0
        || ps.bench.count == 0
        || official_pod_bad_status(ps) == OfficialBadStatus::kAsleep
        || official_pod_bad_status(ps) == OfficialBadStatus::kParalyzed) {
        return true;
    }
    const OfficialCardRefPod active_ref = ps.active.values[0];
    const OfficialCardStatePod* active = official_pod_card(state, active_ref);
    const OfficialCardRule* master = active == nullptr ? nullptr : official_card_rule(
        rules, static_cast<std::uint32_t>(active->card_id));
    if (master == nullptr) {
        official_pod_fail(
            state,
            OfficialPodError::kRulePackBounds,
            active == nullptr ? 0 : active->card_id);
        return false;
    }
    if ((active->this_turn[3] & 1U) != 0
        || official_continual_flag(*active, 23)
        || master->values[kCardPokemonType] == 2) {
        return true;
    }
    if ((ps.this_turn & (1U << 22U)) != 0
        && official_pod_poison_counter(ps) > 0
        && !official_continual_flag(*active, 10)) {
        return true;
    }
    const std::int32_t cost = official_main_retreat_cost(*state, rules, active_ref);
    const std::int32_t energy = official_main_attached_energy_units(
        *state, rules, player, active_ref);
    if (cost < 0 || energy < 0) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, active->card_id);
        return false;
    }
    *can_retreat = energy >= cost;
    return true;
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult official_main_after_refresh(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pod_ok(state)) return OfficialMainResult::kError;
    if (state->temporary_triggers.count == 0 && state->triggers.count == 0) {
        return official_main_after_flow(state, rules);
    }
    state->flow_flags |= kOfficialRefreshTriggerReturnToMainFlag;
    const OfficialTriggerResolverResult trigger = official_begin_trigger_resolution(
        state, rules, 0);
    if (trigger == OfficialTriggerResolverResult::kError) {
        return OfficialMainResult::kError;
    }
    if (trigger == OfficialTriggerResolverResult::kNeedsAction) {
        return OfficialMainResult::kNeedsAction;
    }
    state->flow_flags &= static_cast<std::uint8_t>(
        ~kOfficialRefreshTriggerReturnToMainFlag);
    return official_main_begin_refresh(state, rules);
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult official_main_begin_refresh(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    state->flow_flags |= kOfficialRefreshReturnToMainFlag;
    const OfficialTurnFlowResult refresh = official_begin_refresh(state, rules);
    if (refresh == OfficialTurnFlowResult::kError) {
        return OfficialMainResult::kError;
    }
    if (refresh == OfficialTurnFlowResult::kNeedsAction) {
        return OfficialMainResult::kNeedsAction;
    }
    state->flow_flags &= static_cast<std::uint8_t>(
        ~kOfficialRefreshReturnToMainFlag);
    return official_main_after_refresh(state, rules);
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult official_main_after_play_effect(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    official_clear_resolved_ability(state);
    while (state->playing.count > 0 && official_pod_ok(state)) {
        const OfficialCardRefPod ref = state->playing.values[0];
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        if (card == nullptr || card->player < 0 || card->player > 1) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, ref.index);
            break;
        }
        const std::int32_t owner = card->player;
        official_pod_move_card(
            state,
            owner,
            OfficialArea::kPlaying,
            0,
            OfficialArea::kTrash,
            false);
    }
    state->flow_flags &= static_cast<std::uint8_t>(
        ~kOfficialPlayEffectReturnToMainFlag);
    if (!official_pod_ok(state)) return OfficialMainResult::kError;
    return official_main_begin_refresh(state, rules);
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult official_build_main_options(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pod_ok(state)
        || state->phase != static_cast<std::uint8_t>(OfficialGamePhase::kMain)
        || state->attack_flow_stage != 0
        || state->turn_flow_stage != 0
        || state->refresh_flow_stage != 0
        || state->trigger_resolver.active != 0
        || state->effect_interpreter.active != 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, state->phase);
        }
        return OfficialMainResult::kError;
    }
    if (official_pod_finish_check(state)) return OfficialMainResult::kComplete;
    if (state->turn >= 10000) {
        official_pod_set_result(state, 2, OfficialFinishReason::kOther);
        return OfficialMainResult::kComplete;
    }
    official_clear_effect_selection(state);
    const std::int32_t player = official_active_player(*state);
    state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kMain);
    state->select_context = kOfficialSelectContextMain;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = 1;
    state->select_max = 1;

    const OfficialPlayerStatePod& ps = state->players[player];
    if (!official_main_add_play_options(state, rules, player)) {
        return OfficialMainResult::kError;
    }
    if (!official_main_add_ability_options(state, rules, player)) {
        return OfficialMainResult::kError;
    }
    if (ps.active.count > 0
        && !official_main_add_attack_options(
            state, rules, ps.active.values[0], -1)) {
        return OfficialMainResult::kError;
    }
    if (state->turn >= 2) {
        for (std::uint16_t index = 0; index < ps.bench.count; ++index) {
            if (!official_main_add_attack_options(
                    state, rules, ps.bench.values[index], index)) {
                return OfficialMainResult::kError;
            }
        }
    }

    bool can_retreat = false;
    if (!official_main_can_retreat(state, rules, player, &can_retreat)) {
        return OfficialMainResult::kError;
    }
    if (can_retreat) {
        OfficialSelectOptionPod retreat{};
        retreat.type = static_cast<std::uint8_t>(
            OfficialSelectOptionTypeId::kRetreat);
        if (!official_pod_push(
                state,
                &state->options,
                retreat,
                OfficialPodError::kOptionOverflow)) {
            return OfficialMainResult::kError;
        }
    }

    OfficialSelectOptionPod end{};
    end.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kEnd);
    if (!official_pod_push(
            state, &state->options, end, OfficialPodError::kOptionOverflow)) {
        return OfficialMainResult::kError;
    }
    return OfficialMainResult::kNeedsAction;
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult official_main_after_flow(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pod_ok(state)) return OfficialMainResult::kError;
    if (state->game_result
        != static_cast<std::uint8_t>(OfficialGameResult::kNone)) {
        return OfficialMainResult::kComplete;
    }
    // Effects such as Lumiose City's play skill set the turn-end bit while
    // returning through an ability/refresh continuation rather than the main
    // action dispatcher.  Check it at the shared main boundary so that all
    // such flows enter the same official turn-end sequence.
    if ((state->turn_state & kOfficialTurnEndFlag) != 0
        || state->turn_action_count >= 10000) {
        state->flow_flags |= kOfficialTurnReturnToMainFlag;
        const OfficialTurnFlowResult result = official_begin_turn_end(state, rules);
        if (result == OfficialTurnFlowResult::kError) {
            return OfficialMainResult::kError;
        }
        if (result == OfficialTurnFlowResult::kNeedsAction) {
            return OfficialMainResult::kNeedsAction;
        }
        state->flow_flags &= static_cast<std::uint8_t>(
            ~kOfficialTurnReturnToMainFlag);
        return official_main_after_flow(state, rules);
    }
    return official_build_main_options(state, rules);
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult official_prepare_main_options(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pod_ok(state)
        || state->phase != static_cast<std::uint8_t>(OfficialGamePhase::kMain)
        || state->attack_flow_stage != 0
        || state->turn_flow_stage != 0
        || state->refresh_flow_stage != 0
        || state->trigger_resolver.active != 0
        || state->effect_interpreter.active != 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, state->phase);
        }
        return OfficialMainResult::kError;
    }
    if (official_pod_finish_check(state)) return OfficialMainResult::kComplete;
    return official_main_after_flow(state, rules);
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult
official_main_prepare_retreat_switch(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    bool clear_targets = true) {
    official_clear_effect_selection(state);
    state->energy_cost = 0;
    state->remain_energy_cost = 0;
    state->selected_energy_card_count = 0;
    state->selecting_energy_pokemon = {};
    // A paid retreat reaches SelectSwitchPokemon through
    // SelectedPokemonEnergy, which clears targetList.  A zero-cost retreat
    // calls SelectSwitchPokemon directly and preserves the prior targetList
    // until AfterRetreat runs.
    if (clear_targets) state->targets.count = 0;
    const std::int32_t player = official_active_player(*state);
    OfficialPlayerStatePod& ps = state->players[player];
    if (ps.bench.count == 0) {
        state->selected_list.count = 0;
        return official_main_begin_refresh(state, rules);
    }
    state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kCard);
    state->select_context = 4;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = 1;
    state->select_max = 1;
    for (std::uint16_t index = 0; index < ps.bench.count; ++index) {
        const OfficialCardRefPod ref = ps.bench.values[index];
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        OfficialSelectOptionPod option{};
        option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kCard);
        option.params[0] = static_cast<std::int16_t>(OfficialArea::kBench);
        option.params[1] = static_cast<std::int16_t>(index);
        option.params[2] = static_cast<std::int16_t>(player);
        option.resolved_card = ref.index;
        option.option_equiv = static_cast<std::uint16_t>(
            card == nullptr ? 0 : card->card_id);
        if (!official_pod_push(
                state, &state->options, option, OfficialPodError::kOptionOverflow)) {
            return OfficialMainResult::kError;
        }
    }
    return OfficialMainResult::kNeedsAction;
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult
official_main_build_retreat_energy_options(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    official_clear_effect_selection(state);
    const std::int32_t player = official_active_player(*state);
    const OfficialPlayerStatePod& ps = state->players[player];
    if (ps.active.count == 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, player);
        return OfficialMainResult::kError;
    }
    const OfficialCardRefPod active_ref = ps.active.values[0];
    const OfficialCardStatePod* active = official_pod_card(state, active_ref);
    if (active == nullptr) return OfficialMainResult::kError;
    state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kEnergy);
    state->select_context = 31;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = state->remain_energy_cost <= 0 ? 0 : 1;
    state->select_max = 1;
    std::int32_t energy_index = 0;
    for (std::uint16_t index = 0; index < ps.energy.count; ++index) {
        const OfficialCardRefPod ref = ps.energy.values[index];
        const OfficialCardStatePod* energy = official_pod_card(state, ref);
        if (energy == nullptr
            || energy->attach_move_counter != active->move_counter) continue;
        const OfficialEnergyInfoPod info = official_energy_info(
            *state, rules, ref, active_ref);
        OfficialSelectOptionPod option{};
        option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kEnergy);
        option.params[0] = static_cast<std::int16_t>(OfficialArea::kActive);
        option.params[1] = 0;
        option.params[2] = static_cast<std::int16_t>(player);
        option.params[3] = static_cast<std::int16_t>(energy_index++);
        option.params[4] = static_cast<std::int16_t>(info.count);
        // Official Energy options identify the Pokemon in params[0..2]. The
        // attached Energy itself is selected by params[3], not by getCardPosition().
        option.resolved_card = active_ref.index;
        option.option_equiv = static_cast<std::uint16_t>(active->card_id);
        if (!official_pod_push(
                state, &state->options, option, OfficialPodError::kOptionOverflow)) {
            return OfficialMainResult::kError;
        }
    }
    if (state->options.count == 0) {
        return official_main_prepare_retreat_switch(state, rules);
    }
    return OfficialMainResult::kNeedsAction;
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult
official_main_prepare_retreat_energy(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t cost) {
    official_clear_effect_selection(state);
    state->selected_list.count = 0;
    state->targets.count = 0;
    state->energy_cost = cost;
    state->remain_energy_cost = cost;
    state->selected_energy_card_count = 0;
    state->selecting_energy_pokemon = {};
    const std::int32_t player = official_active_player(*state);
    const OfficialPlayerStatePod& ps = state->players[player];
    if (ps.active.count == 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, player);
        return OfficialMainResult::kError;
    }
    const OfficialCardRefPod active_ref = ps.active.values[0];
    const OfficialCardStatePod* active = official_pod_card(state, active_ref);
    std::int32_t energy_sum = 0;
    for (std::uint16_t index = 0; index < ps.energy.count; ++index) {
        const OfficialCardRefPod ref = ps.energy.values[index];
        const OfficialCardStatePod* energy = official_pod_card(state, ref);
        if (energy == nullptr || active == nullptr
            || energy->attach_move_counter != active->move_counter) continue;
        // Official SelectPokemonEnergy marks the state changed as soon as its
        // candidate AttachedCardList is non-empty, before presenting the first
        // Energy option. Preserve that scratch flag even though no card has
        // moved yet.
        state->control_flags |= kOfficialChangedFlag;
        if (!official_pod_push(
                state,
                &state->targets,
                official_pod_area_ref(state, ref),
                OfficialPodError::kSelectionOverflow)) {
            return OfficialMainResult::kError;
        }
        energy_sum += official_energy_info(*state, rules, ref, active_ref).count;
    }
    if (state->energy_cost > energy_sum) {
        state->energy_cost = energy_sum;
        state->remain_energy_cost = energy_sum;
    }
    if (state->energy_cost <= 0) {
        return official_main_prepare_retreat_switch(state, rules);
    }
    return official_main_build_retreat_energy_options(state, rules);
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult
official_main_continue_retreat_after_triggers(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    state->flow_flags &= static_cast<std::uint8_t>(
        ~kOfficialRetreatTriggerPendingFlag);
    if ((state->control_flags & kOfficialRetreatFailFlag) != 0) {
        state->control_flags &= static_cast<std::uint8_t>(
            ~kOfficialRetreatFailFlag);
        return official_main_begin_refresh(state, rules);
    }
    const std::int32_t player = official_active_player(*state);
    if (state->players[player].active.count == 0) {
        return official_main_begin_refresh(state, rules);
    }
    const std::int32_t cost = official_main_retreat_cost(
        *state, rules, state->players[player].active.values[0]);
    if (cost < 0) {
        official_pod_fail(state, OfficialPodError::kRulePackBounds, player);
        return OfficialMainResult::kError;
    }
    return cost > 0
        ? official_main_prepare_retreat_energy(state, rules, cost)
        : official_main_prepare_retreat_switch(state, rules, false);
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult
official_main_continue_retreat_after_refresh(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    state->flow_flags &= static_cast<std::uint8_t>(
        ~kOfficialRetreatRefreshPendingFlag);
    state->flow_flags |= kOfficialRetreatTriggerPendingFlag;
    const OfficialTriggerResolverResult trigger = official_begin_trigger_resolution(
        state, rules, 0);
    if (trigger == OfficialTriggerResolverResult::kError) {
        return OfficialMainResult::kError;
    }
    if (trigger == OfficialTriggerResolverResult::kNeedsAction) {
        return OfficialMainResult::kNeedsAction;
    }
    return official_main_continue_retreat_after_triggers(state, rules);
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult official_main_resume_retreat_action(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (state->select_context == 31
        && state->select_type == static_cast<std::uint8_t>(
            OfficialSelectTypeId::kEnergy)) {
        if (count < state->select_min || count > state->select_max
            || (count > 0
                && (option_indices == nullptr
                    || option_indices[0] >= state->options.count))) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
            return OfficialMainResult::kError;
        }
        if (count == 0) {
            return official_main_prepare_retreat_switch(state, rules);
        }
        const OfficialSelectOptionPod selected = state->options.values[
            option_indices[0]];
        if (selected.type != static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kEnergy)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, selected.type);
            return OfficialMainResult::kError;
        }
        const std::int32_t player = official_active_player(*state);
        OfficialPlayerStatePod& ps = state->players[player];
        if (selected.params[0] != static_cast<std::int32_t>(OfficialArea::kActive)
            || selected.params[1] != 0
            || selected.params[2] != player
            || ps.active.count == 0) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, selected.params[1]);
            return OfficialMainResult::kError;
        }
        const OfficialCardStatePod* active = official_pod_card(
            state, ps.active.values[0]);
        std::int32_t ordinal = 0;
        std::int32_t zone_index = -1;
        for (std::uint16_t index = 0; index < ps.energy.count; ++index) {
            const OfficialCardStatePod* energy = official_pod_card(
                state, ps.energy.values[index]);
            if (energy == nullptr || active == nullptr
                || energy->attach_move_counter != active->move_counter) continue;
            if (ordinal++ == selected.params[3]) {
                zone_index = index;
                break;
            }
        }
        if (zone_index < 0) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, selected.params[3]);
            return OfficialMainResult::kError;
        }
        const OfficialCardRefPod energy_ref = ps.energy.values[zone_index];
        if (!official_pod_push(
                state,
                &state->selected_list,
                energy_ref,
                OfficialPodError::kSelectionOverflow)) {
            return OfficialMainResult::kError;
        }
        official_pod_move_card(
            state,
            player,
            OfficialArea::kEnergy,
            static_cast<std::uint16_t>(zone_index),
            OfficialArea::kTrash,
            false);
        if (!official_pod_ok(state)) return OfficialMainResult::kError;
        state->remain_energy_cost -= selected.params[4];
        ++state->selected_energy_card_count;
        if (state->energy_cost > state->selected_energy_card_count) {
            return official_main_build_retreat_energy_options(state, rules);
        }
        return official_main_prepare_retreat_switch(state, rules);
    }

    if (state->select_context == 4
        && state->select_type == static_cast<std::uint8_t>(
            OfficialSelectTypeId::kCard)) {
        if (count != 1 || option_indices == nullptr
            || option_indices[0] >= state->options.count) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
            return OfficialMainResult::kError;
        }
        const OfficialSelectOptionPod selected = state->options.values[
            option_indices[0]];
        const std::int32_t player = official_active_player(*state);
        OfficialPlayerStatePod& ps = state->players[player];
        if (selected.type != static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kCard)
            || selected.params[0] != static_cast<std::int32_t>(OfficialArea::kBench)
            || selected.params[1] < 0
            || selected.params[1] >= ps.bench.count
            || selected.params[2] != player
            || ps.active.count == 0) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, selected.params[1]);
            return OfficialMainResult::kError;
        }
        official_clear_effect_selection(state);
        if (!official_switch_pokemon(
                state,
                rules,
                player,
                static_cast<std::uint16_t>(selected.params[1]))) {
            return OfficialMainResult::kError;
        }
        state->selected_list.count = 0;
        state->targets.count = 0;
        return official_main_begin_refresh(state, rules);
    }

    official_pod_fail(state, OfficialPodError::kInvalidAction, state->select_context);
    return OfficialMainResult::kError;
}

PTCG_OFFICIAL_MAIN_HD inline OfficialMainResult official_apply_main_action(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (!official_pod_ok(state)
        || state->select_type != static_cast<std::uint8_t>(OfficialSelectTypeId::kMain)
        || count != 1
        || option_indices == nullptr
        || option_indices[0] >= state->options.count) {
        if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        }
        return OfficialMainResult::kError;
    }
    const OfficialSelectOptionPod selected = state->options.values[option_indices[0]];
    official_clear_effect_selection(state);

    const auto type = static_cast<OfficialSelectOptionTypeId>(selected.type);
    if (type == OfficialSelectOptionTypeId::kAbility) {
        const std::int32_t player = official_active_player(*state);
        const auto area = static_cast<OfficialArea>(selected.params[0]);
        OfficialCardRefPod ref{};
        if (area == OfficialArea::kActive
            && selected.params[1] >= 0
            && selected.params[1] < state->players[player].active.count) {
            ref = state->players[player].active.values[selected.params[1]];
        } else if (area == OfficialArea::kBench
            && selected.params[1] >= 0
            && selected.params[1] < state->players[player].bench.count) {
            ref = state->players[player].bench.values[selected.params[1]];
        } else if (area == OfficialArea::kStadium
            && selected.params[1] >= 0
            && selected.params[1] < state->stadium.count) {
            ref = state->stadium.values[selected.params[1]];
        } else {
            official_pod_fail(state, OfficialPodError::kInvalidAction, selected.params[1]);
            return OfficialMainResult::kError;
        }
        bool can_activate = false;
        if (!official_main_card_can_activate(
                state, rules, ref, area, player, &can_activate)
            || !can_activate) {
            if (official_pod_ok(state)) {
                official_pod_fail(state, OfficialPodError::kInvalidAction, ref.index);
            }
            return OfficialMainResult::kError;
        }
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        const OfficialCardRule* master = official_card_rule(
            rules, static_cast<std::uint32_t>(card->card_id));
        const OfficialSkillRule* skill = official_skill_rule(
            rules, static_cast<std::uint32_t>(master->values[kCardAbilityId]));
        state->trigger_resolver = OfficialTriggerResolverPod{};
        state->trigger_resolver.active = 1;
        state->trigger_resolver.current_valid = 1;
        state->trigger_resolver.current.activate.skill_id =
            master->values[kCardAbilityId];
        state->trigger_resolver.current.activate.effect_card =
            official_pod_area_ref(state, ref);
        state->trigger_resolver.current.activate.use_player =
            static_cast<std::int8_t>(player);
        state->flow_flags |= kOfficialAbilityReturnToMainFlag;
        const OfficialTriggerResolverResult activated =
            official_prepare_resolved_activation(state, rules, *skill);
        if (activated == OfficialTriggerResolverResult::kError) {
            return OfficialMainResult::kError;
        }
        if (activated == OfficialTriggerResolverResult::kNeedsAction) {
            return OfficialMainResult::kNeedsAction;
        }
        const OfficialTriggerResolverResult finished =
            official_finish_resolved_trigger(state, rules);
        if (finished == OfficialTriggerResolverResult::kError) {
            return OfficialMainResult::kError;
        }
        state->trigger_resolver.active = 0;
        state->flow_flags &= static_cast<std::uint8_t>(
            ~kOfficialAbilityReturnToMainFlag);
        return official_main_begin_refresh(state, rules);
    }
    if (type == OfficialSelectOptionTypeId::kEvolve) {
        const std::int32_t player = official_active_player(*state);
        OfficialPlayerStatePod& ps = state->players[player];
        if (selected.params[0] != static_cast<std::int32_t>(OfficialArea::kHand)
            || selected.params[1] < 0
            || selected.params[1] >= ps.hand.count
            || (ps.this_turn & kOfficialPlayerThisTurnCannotEvolveFlag) != 0) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, selected.params[1]);
            return OfficialMainResult::kError;
        }
        const OfficialCardRefPod evolution_ref = ps.hand.values[selected.params[1]];
        const auto target_area = static_cast<OfficialArea>(selected.params[2]);
        OfficialCardRefPod target_ref{};
        if (target_area == OfficialArea::kActive
            && selected.params[3] >= 0
            && selected.params[3] < ps.active.count) {
            target_ref = ps.active.values[selected.params[3]];
        } else if (target_area == OfficialArea::kBench
            && selected.params[3] >= 0
            && selected.params[3] < ps.bench.count) {
            target_ref = ps.bench.values[selected.params[3]];
        } else {
            official_pod_fail(state, OfficialPodError::kInvalidAction, selected.params[3]);
            return OfficialMainResult::kError;
        }
        const OfficialCardStatePod* evolution = official_pod_card(state, evolution_ref);
        const OfficialCardStatePod* target = official_pod_card(state, target_ref);
        const OfficialCardRule* evolution_master = evolution == nullptr
            ? nullptr
            : official_card_rule(
                rules, static_cast<std::uint32_t>(evolution->card_id));
        const OfficialCardRule* target_master = target == nullptr
            ? nullptr
            : official_card_rule(
                rules, static_cast<std::uint32_t>(target->card_id));
        if (evolution_master == nullptr
            || target_master == nullptr
            || evolution_master->values[kCardType] != 0
            || evolution_master->values[kCardEvolutionType] == 1
            || (evolution_master->flags & kOfficialCardTransformOnlyFlag) != 0
            || ((ps.continual_state
                    & kOfficialPlayerCannotPlayAbilityPokemonFlag) != 0
                && evolution_master->values[kCardAbilityId] != 0
                && (evolution_master->flags
                    & kOfficialCardTeamRocketFlag) == 0)
            || !official_main_can_evolve_candidate(
                *state,
                rules,
                *evolution_master,
                *target,
                *target_master)) {
            official_pod_fail(
                state,
                OfficialPodError::kInvalidAction,
                evolution == nullptr ? 0 : evolution->card_id);
            return OfficialMainResult::kError;
        }
        if (state->turn <= 2
            && !official_continual_flag(*target, kOfficialCardCanEvolveAppearTurnBit)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, state->turn);
            return OfficialMainResult::kError;
        }
        if ((target->runtime_flags & kCardAppear) != 0
            && !official_continual_flag(*target, kOfficialCardCanEvolveAppearTurnBit)) {
            const bool can_evolve_grass_appear = official_continual_flag(
                *target, kOfficialCardCanEvolveGrassAppearTurnBit);
            if (!can_evolve_grass_appear
                || (evolution_master->values[kCardEnergyType] & 1) == 0) {
                official_pod_fail(state, OfficialPodError::kInvalidAction, target->card_id);
                return OfficialMainResult::kError;
            }
        }
        if (!official_effect_evolve_card(
                state, rules, evolution_ref, target_ref)) {
            return OfficialMainResult::kError;
        }
        return official_main_begin_refresh(state, rules);
    }
    if (type == OfficialSelectOptionTypeId::kAttach) {
        const std::int32_t player = official_active_player(*state);
        OfficialPlayerStatePod& ps = state->players[player];
        if (selected.params[0] != static_cast<std::int32_t>(OfficialArea::kHand)
            || selected.params[1] < 0
            || selected.params[1] >= ps.hand.count) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, selected.params[1]);
            return OfficialMainResult::kError;
        }
        const OfficialCardRefPod attach_ref = ps.hand.values[selected.params[1]];
        const OfficialCardStatePod* attach_card = official_pod_card(state, attach_ref);
        const OfficialCardRule* attach_master = attach_card == nullptr
            ? nullptr
            : official_card_rule(
                rules, static_cast<std::uint32_t>(attach_card->card_id));
        if (attach_master == nullptr
            || (attach_master->values[kCardType] != 2
                && attach_master->values[kCardType] != 5
                && attach_master->values[kCardType] != 6)) {
            official_pod_fail(
                state,
                OfficialPodError::kInvalidAction,
                attach_card == nullptr ? 0 : attach_card->card_id);
            return OfficialMainResult::kError;
        }
        if ((ps.continual_state & kOfficialPlayerCannotPlayAceSpecFlag) != 0
            && (attach_master->flags & kOfficialCardAceSpecFlag) != 0) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, attach_card->card_id);
            return OfficialMainResult::kError;
        }

        OfficialCardRefPod target_ref{};
        const auto target_area = static_cast<OfficialArea>(selected.params[2]);
        if (target_area == OfficialArea::kActive
            && selected.params[3] >= 0
            && selected.params[3] < ps.active.count) {
            target_ref = ps.active.values[selected.params[3]];
        } else if (target_area == OfficialArea::kBench
            && selected.params[3] >= 0
            && selected.params[3] < ps.bench.count) {
            target_ref = ps.bench.values[selected.params[3]];
        } else {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, selected.params[3]);
            return OfficialMainResult::kError;
        }
        const OfficialCardStatePod* target = official_pod_card(state, target_ref);
        const OfficialCardRule* target_master = target == nullptr
            ? nullptr
            : official_card_rule(
                rules, static_cast<std::uint32_t>(target->card_id));
        if (target_master == nullptr) {
            official_pod_fail(
                state,
                OfficialPodError::kInvalidAction,
                target == nullptr ? 0 : target->card_id);
            return OfficialMainResult::kError;
        }

        if (attach_master->values[kCardType] == 2) {
            if ((ps.continual_state & kOfficialPlayerCannotPlayToolFlag) != 0
                || official_main_attached_tool_count(*state, player, *target)
                    >= official_main_tool_capacity(*target)) {
                official_pod_fail(
                    state, OfficialPodError::kInvalidAction, attach_card->card_id);
                return OfficialMainResult::kError;
            }
            const OfficialCardRefPod moved = official_pod_move_card(
                state,
                player,
                OfficialArea::kHand,
                static_cast<std::uint16_t>(selected.params[1]),
                OfficialArea::kTool,
                false);
            OfficialCardStatePod* attached = official_pod_card(state, moved);
            target = official_pod_card(state, target_ref);
            if (!official_pod_ok(state) || attached == nullptr || target == nullptr) {
                return OfficialMainResult::kError;
            }
            attached->attach_move_counter = target->move_counter;
            const std::int32_t play_id = attach_master->values[kCardPlayId];
            if (play_id > 0 && (state->continual_state & 1U) == 0) {
                const OfficialSkillRule* skill = official_skill_rule(
                    rules, static_cast<std::uint32_t>(play_id));
                if (skill == nullptr) {
                    official_pod_fail(state, OfficialPodError::kRulePackBounds, play_id);
                    return OfficialMainResult::kError;
                }
                if ((skill->flags & kOfficialSkillAttachBenchFlag) == 0
                    || target_area == OfficialArea::kBench) {
                    OfficialTriggeredAbilityPod pending{};
                    pending.trigger.type = kOfficialTriggerAttach;
                    pending.trigger.subject = official_pod_area_ref(state, target_ref);
                    pending.activate.skill_id = play_id;
                    pending.activate.effect_card = official_pod_area_ref(state, moved);
                    pending.activate.use_player = static_cast<std::int8_t>(player);
                    if (!official_pod_push(
                            state,
                            &state->temporary_triggers,
                            pending,
                            OfficialPodError::kTriggerStackOverflow)) {
                        return OfficialMainResult::kError;
                    }
                }
            }
            return official_main_begin_refresh(state, rules);
        }

        if ((state->turn_state & kOfficialEnergyPlayedFlag) != 0
            || (attach_master->values[kCardType] == 6
                && (ps.this_turn
                    & kOfficialPlayerThisTurnCannotPlaySpecialEnergyFlag) != 0)
            || (target->this_turn[3]
                & kOfficialCardCannotHandAttachEnergyFlag) != 0
            || ((attach_master->flags & kOfficialCardOnlyTeamRocketFlag) != 0
                && (target_master->flags & kOfficialCardTeamRocketFlag) == 0)) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, attach_card->card_id);
            return OfficialMainResult::kError;
        }

        const OfficialCardRefPod moved = official_pod_move_card(
            state,
            player,
            OfficialArea::kHand,
            static_cast<std::uint16_t>(selected.params[1]),
            OfficialArea::kEnergy,
            false);
        OfficialCardStatePod* attached = official_pod_card(state, moved);
        target = official_pod_card(state, target_ref);
        if (!official_pod_ok(state) || attached == nullptr || target == nullptr) {
            return OfficialMainResult::kError;
        }
        attached->attach_move_counter = target->move_counter;
        state->turn_state |= kOfficialEnergyPlayedFlag;
        if (!official_pull_trigger(
                state,
                rules,
                kOfficialTriggerEnergyAttachFromHand,
                target_ref)) {
            return OfficialMainResult::kError;
        }
        const std::int32_t play_id = attach_master->values[kCardPlayId];
        if (play_id > 0) {
            const OfficialSkillRule* skill = official_skill_rule(
                rules, static_cast<std::uint32_t>(play_id));
            if (skill == nullptr) {
                official_pod_fail(state, OfficialPodError::kRulePackBounds, play_id);
                return OfficialMainResult::kError;
            }
            if ((skill->flags & kOfficialSkillAttachBenchFlag) == 0
                || target_area == OfficialArea::kBench) {
                OfficialTriggeredAbilityPod pending{};
                pending.trigger.type = kOfficialTriggerAttach;
                pending.trigger.subject = official_pod_area_ref(state, target_ref);
                pending.activate.skill_id = play_id;
                pending.activate.effect_card = official_pod_area_ref(state, moved);
                pending.activate.use_player = static_cast<std::int8_t>(player);
                if (!official_pod_push(
                        state,
                        &state->temporary_triggers,
                        pending,
                        OfficialPodError::kTriggerStackOverflow)) {
                    return OfficialMainResult::kError;
                }
            }
        }
        return official_main_begin_refresh(state, rules);
    }
    if (type == OfficialSelectOptionTypeId::kDiscard) {
        const std::int32_t player = official_active_player(*state);
        OfficialPlayerStatePod& ps = state->players[player];
        const auto area = static_cast<OfficialArea>(selected.params[0]);
        OfficialCardRefPod ref{};
        if (area == OfficialArea::kActive
            && selected.params[1] >= 0
            && selected.params[1] < ps.active.count) {
            ref = ps.active.values[selected.params[1]];
        } else if (area == OfficialArea::kBench
            && selected.params[1] >= 0
            && selected.params[1] < ps.bench.count) {
            ref = ps.bench.values[selected.params[1]];
        } else {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, selected.params[1]);
            return OfficialMainResult::kError;
        }
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        const OfficialCardRule* master = card == nullptr ? nullptr
            : official_card_rule(rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr
            || (master->flags & kOfficialCardCanTrashFlag) == 0) {
            official_pod_fail(
                state,
                OfficialPodError::kInvalidAction,
                card == nullptr ? 0 : card->card_id);
            return OfficialMainResult::kError;
        }
        official_pod_move_ref_complete(
            state, ref, OfficialArea::kTrash, false, false);
        if (!official_pod_ok(state)) return OfficialMainResult::kError;
        return official_main_begin_refresh(state, rules);
    }
    if (type == OfficialSelectOptionTypeId::kRetreat) {
        const std::int32_t player = official_active_player(*state);
        bool can_retreat = false;
        if (!official_main_can_retreat(state, rules, player, &can_retreat)
            || !can_retreat) {
            if (official_pod_ok(state)) {
                official_pod_fail(state, OfficialPodError::kInvalidAction, player);
            }
            return OfficialMainResult::kError;
        }
        const OfficialCardRefPod active_ref = state->players[player].active.values[0];
        state->turn_state |= 1U << 3U;
        state->control_flags &= static_cast<std::uint8_t>(
            ~kOfficialRetreatFailFlag);
        if (!official_pull_trigger(
                state, rules, kOfficialTriggerPreRetreat, active_ref)) {
            return OfficialMainResult::kError;
        }
        state->flow_flags |= kOfficialRetreatRefreshPendingFlag;
        const OfficialTurnFlowResult refresh = official_begin_refresh(state, rules);
        if (refresh == OfficialTurnFlowResult::kError) {
            return OfficialMainResult::kError;
        }
        if (refresh == OfficialTurnFlowResult::kNeedsAction) {
            return OfficialMainResult::kNeedsAction;
        }
        return official_main_continue_retreat_after_refresh(state, rules);
    }
    if (type == OfficialSelectOptionTypeId::kAttack) {
        const std::int32_t player = official_active_player(*state);
        OfficialCardRefPod attacker{};
        if (selected.params[2] < 0) {
            if (state->players[player].active.count == 0) {
                official_pod_fail(state, OfficialPodError::kInvalidAction, player);
                return OfficialMainResult::kError;
            }
            attacker = state->players[player].active.values[0];
        } else if (selected.params[2]
                < static_cast<std::int32_t>(state->players[player].bench.count)) {
            attacker = state->players[player].bench.values[selected.params[2]];
        } else {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, selected.params[2]);
            return OfficialMainResult::kError;
        }
        const OfficialAttackResult result = official_begin_attack(
            state,
            rules,
            attacker,
            selected.params[0],
            selected.params[1]);
        state->attack_flow_flags |= kOfficialAttackReturnToMainFlag;
        if (result == OfficialAttackResult::kError) {
            if (official_pod_ok(state)) {
                official_pod_fail(
                    state,
                    OfficialPodError::kInvalidAction,
                    -1649);
            }
            return OfficialMainResult::kError;
        }
        if (result == OfficialAttackResult::kNeedsAction) {
            return OfficialMainResult::kNeedsAction;
        }
        state->attack_flow_flags &= static_cast<std::uint8_t>(
            ~kOfficialAttackReturnToMainFlag);
        return official_main_after_flow(state, rules);
    }
    if (type == OfficialSelectOptionTypeId::kPlay) {
        const std::int32_t player = official_active_player(*state);
        OfficialPlayerStatePod& ps = state->players[player];
        const std::int32_t hand_index = selected.params[0];
        if (hand_index < 0 || hand_index >= ps.hand.count) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, hand_index);
            return OfficialMainResult::kError;
        }
        const OfficialCardRefPod ref = ps.hand.values[hand_index];
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        const OfficialCardRule* master = card == nullptr ? nullptr
            : official_card_rule(rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction,
                card == nullptr ? 0 : card->card_id);
            return OfficialMainResult::kError;
        }
        const std::int32_t card_type = master->values[kCardType];
        if (card_type == 0
            && master->values[kCardEvolutionType] == 1
            && ps.bench.count < official_main_bench_capacity(ps)) {
            const OfficialCardRefPod moved = official_pod_move_card(
                state,
                player,
                OfficialArea::kHand,
                static_cast<std::uint16_t>(hand_index),
                OfficialArea::kBench,
                false);
            // Official MoveCard refreshes continual effects before emitting
            // ToBenchMyTurn and HandToBench.  AbilityPlay cards such as Meowth
            // ex therefore enter trigger resolution immediately after being
            // benched instead of returning straight to Main.
            if (!official_pod_ok(state)
                || official_refresh_continual_effects(state, rules)
                    != OfficialContinualRefreshResult::kApplied
                || !official_pull_trigger(
                    state,
                    rules,
                    kOfficialTriggerToBenchMyTurn,
                    moved)
                || !official_pull_trigger(
                    state,
                    rules,
                    kOfficialTriggerHandToBench,
                    moved)) {
                return OfficialMainResult::kError;
            }
        } else if ((card_type == 1 || card_type == 3)
            && master->values[kCardPlayId] > 0
            && (card_type != 1
                || ((ps.continual_state & kOfficialPlayerCannotPlayItemFlag) == 0
                    && (ps.this_turn
                        & kOfficialPlayerThisTurnCannotPlayItemFlag) == 0))
            && (card_type != 3
                || (((state->turn_state & kOfficialSupporterPlayedFlag) == 0)
                    && (ps.this_turn
                        & kOfficialPlayerThisTurnCannotPlaySupporterFlag) == 0
                    && (state->turn > 1
                        || (master->flags
                            & kOfficialCardCanPlayFirstTurnFlag) != 0)))) {
            if (card_type == 3) {
                state->turn_state |= kOfficialSupporterPlayedFlag;
                if (!official_pod_push(
                        state,
                        &state->turn_play,
                        ref,
                        OfficialPodError::kTurnRecordOverflow)) {
                    return OfficialMainResult::kError;
                }
            }
            if ((master->flags & kOfficialCardToBenchFlag) != 0) {
                official_pod_move_card(
                    state,
                    player,
                    OfficialArea::kHand,
                    static_cast<std::uint16_t>(hand_index),
                    OfficialArea::kBench,
                    false);
                if (!official_pod_ok(state)) return OfficialMainResult::kError;
                return official_main_begin_refresh(state, rules);
            }
            const std::int32_t play_id = master->values[kCardPlayId];
            const OfficialCardRefPod moved = official_pod_move_card(
                state,
                player,
                OfficialArea::kHand,
                static_cast<std::uint16_t>(hand_index),
                OfficialArea::kPlaying,
                false);
            if (!official_pod_ok(state)) return OfficialMainResult::kError;
            const OfficialSkillRule* play_skill = official_skill_rule(
                rules, static_cast<std::uint32_t>(play_id));
            const OfficialAreaRefPod effect_card = official_pod_area_ref(
                state, moved);
            if (play_skill == nullptr) {
                official_pod_fail(
                    state, OfficialPodError::kRulePackBounds, play_id);
                return OfficialMainResult::kError;
            }
            state->flow_flags |= kOfficialPlayEffectReturnToMainFlag;
            // ActivateAbility clears transient effect state before deciding
            // whether the first branch is available and exposing the choice.
            state->targets.count = 0;
            state->control_flags &= static_cast<std::uint8_t>(
                ~kOfficialChangedFlag);
            const std::int32_t second_effect_start =
                play_skill->values[kSkillSecondEffectStart];
            bool first_effect_satisfied = true;
            if (second_effect_start > 0
                && !official_resolver_satisfy_skill_conditions(
                    state,
                    rules,
                    *play_skill,
                    0,
                    effect_card,
                    player,
                    &first_effect_satisfied)) {
                return OfficialMainResult::kError;
            }

            OfficialEffectInterpreterResult effect =
                OfficialEffectInterpreterResult::kComplete;
            if (second_effect_start > 0 && first_effect_satisfied) {
                // Official ActivateAbility waits for SelectedWhichEffect before
                // ActivateAbility2 records the skill use or starts either
                // branch.  Preserve only the metadata needed by that callback.
                state->effect_state.ability = OfficialActivateAbilityPod{};
                state->effect_state.ability.skill_id = play_id;
                state->effect_state.ability.effect_card = effect_card;
                state->effect_state.ability.use_player =
                    static_cast<std::int8_t>(player);
                state->effect_state.effect_rate = 1;
                state->effect_interpreter = OfficialEffectInterpreterPod{};
                state->effect_interpreter.effect_offset = static_cast<std::uint32_t>(
                    play_skill->values[kSkillEffectOffset]);
                state->effect_interpreter.effect_count = static_cast<std::uint16_t>(
                    play_skill->values[kSkillEffectCount]);
                state->effect_interpreter.first_condition_count =
                    static_cast<std::uint8_t>(
                        play_skill->values[kSkillFirstConditionCount]);
                state->effect_interpreter.active = 1;
                state->effect_interpreter.effect_owner =
                    static_cast<std::int8_t>(player);
                state->effect_interpreter.effect_card = effect_card;
                state->effect_interpreter.step_budget =
                    kOfficialDefaultEffectStepBudget;
                if (!official_begin_effect_selection(
                        state,
                        OfficialSelectTypeId::kYesNo,
                        kOfficialEffectSelectContextFirstEffect,
                        player,
                        1,
                        1,
                        OfficialEffectResumeKind::kSkillChooseEffect)) {
                    return OfficialMainResult::kError;
                }
                OfficialSelectOptionPod yes{};
                yes.type = static_cast<std::uint8_t>(
                    OfficialSelectOptionTypeId::kYes);
                OfficialSelectOptionPod no{};
                no.type = static_cast<std::uint8_t>(
                    OfficialSelectOptionTypeId::kNo);
                if (!official_pod_push(
                        state,
                        &state->options,
                        yes,
                        OfficialPodError::kOptionOverflow)
                    || !official_pod_push(
                        state,
                        &state->options,
                        no,
                        OfficialPodError::kOptionOverflow)) {
                    return OfficialMainResult::kError;
                }
                state->context_card = effect_card.card;
                effect = OfficialEffectInterpreterResult::kNeedsAction;
            } else {
                if (!official_record_resolved_skill(
                        state, *play_skill, effect_card)) {
                    return OfficialMainResult::kError;
                }
                const std::uint16_t start_index =
                    second_effect_start > 0 && !first_effect_satisfied
                    ? static_cast<std::uint16_t>(second_effect_start)
                    : static_cast<std::uint16_t>(
                        play_skill->values[kSkillTriggerStart]);
                effect = official_begin_skill_effects_at(
                    state,
                    rules,
                    play_id,
                    effect_card,
                    player,
                    start_index);
            }
            if (effect == OfficialEffectInterpreterResult::kError) {
                return OfficialMainResult::kError;
            }
            if (effect == OfficialEffectInterpreterResult::kNeedsAction) {
                return OfficialMainResult::kNeedsAction;
            }
            return official_main_after_play_effect(state, rules);
        } else if (card_type == 4
            && master->values[kCardId] != kOfficialAngeFloetteCardId
            && (state->turn_state & kOfficialStadiumPlayedFlag) == 0
            && (ps.continual_state & kOfficialPlayerCannotPlayStadiumFlag) == 0
            && (ps.this_turn & kOfficialPlayerThisTurnCannotPlayStadiumFlag) == 0) {
            if (state->stadium.count > 0) {
                const OfficialCardStatePod* current = official_pod_card(
                    state, state->stadium.values[0]);
                const OfficialCardRule* current_master = current == nullptr ? nullptr
                    : official_card_rule(
                        rules, static_cast<std::uint32_t>(current->card_id));
                if (current_master == nullptr
                    || current_master->values[kCardNameId]
                        == master->values[kCardNameId]) {
                    official_pod_fail(
                        state, OfficialPodError::kInvalidAction, master->values[kCardId]);
                    return OfficialMainResult::kError;
                }
                official_pod_move_card(
                    state,
                    current->player,
                    OfficialArea::kStadium,
                    0,
                    OfficialArea::kTrash,
                    false);
            }
            state->turn_state |= kOfficialStadiumPlayedFlag;
            official_pod_move_card(
                state,
                player,
                OfficialArea::kHand,
                static_cast<std::uint16_t>(hand_index),
                OfficialArea::kStadium,
                false);
        } else {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, master->values[kCardId]);
            return OfficialMainResult::kError;
        }
        if (!official_pod_ok(state)) return OfficialMainResult::kError;
        return official_main_begin_refresh(state, rules);
    }
    if (type == OfficialSelectOptionTypeId::kEnd) {
        state->flow_flags |= kOfficialTurnReturnToMainFlag;
        const OfficialTurnFlowResult result = official_begin_turn_end(state, rules);
        if (result == OfficialTurnFlowResult::kError) {
            return OfficialMainResult::kError;
        }
        if (result == OfficialTurnFlowResult::kNeedsAction) {
            return OfficialMainResult::kNeedsAction;
        }
        state->flow_flags &= static_cast<std::uint8_t>(
            ~kOfficialTurnReturnToMainFlag);
        return official_main_after_flow(state, rules);
    }

    official_pod_fail(
        state, OfficialPodError::kUnsupportedContinuation, selected.type);
    return OfficialMainResult::kError;
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_MAIN_HD
