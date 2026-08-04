#pragma once

#include <cstdint>
#include <limits>
#include <sstream>
#include <type_traits>

#include "All.h"
#include "ptcg_cuda/official_core_pod.cuh"

namespace ptcg::cuda_engine::extractor {

enum class OfficialBridgeError : std::int32_t {
    kNone = 0,
    kNullOutput = 1,
    kMissingGame = 2,
    kMissingRngDrawCount = 3,
    kInvalidRngSerialization = 4,
    kCardRefOutOfRange = 5,
    kZoneOverflow = 6,
    kOptionOverflow = 7,
    kSelectionOverflow = 8,
    kTriggerOverflow = 9,
    kTurnRecordOverflow = 10,
    kContinuationOverflow = 11,
    kInvalidContinuation = 12,
    kNarrowingOverflow = 13,
    kInvalidOptionCard = 14,
};

struct OfficialBridgeResult {
    OfficialBridgeError error = OfficialBridgeError::kNone;
    std::int32_t detail = 0;

    explicit operator bool() const { return error == OfficialBridgeError::kNone; }
};

struct OfficialStateBridgeContext {
    std::uint64_t episode_id = 0;
    std::uint64_t rng_draw_count = 0;
    bool rng_draw_count_valid = false;
};

namespace detail {

inline OfficialBridgeResult bridge_error(
    OfficialBridgeError error,
    std::int32_t detail = 0) {
    return {error, detail};
}

inline OfficialBridgeResult bridge_card_ref(
    CardRef source,
    OfficialCardRefPod* target) {
    if (source.cardIndex >= kOfficialCardCapacity) {
        return bridge_error(
            OfficialBridgeError::kCardRefOutOfRange, source.cardIndex);
    }
    target->index = source.cardIndex;
    return {};
}

inline OfficialBridgeResult bridge_area_ref(
    const AreaRef& source,
    OfficialAreaRefPod* target) {
    OfficialBridgeResult result = bridge_card_ref(source.card, &target->card);
    if (!result) return result;
    target->move_counter = source.moveCounter;
    return {};
}

template <typename SourceList, std::size_t Capacity>
inline OfficialBridgeResult bridge_card_list(
    const SourceList& source,
    OfficialPodList<OfficialCardRefPod, Capacity>* target) {
    if (source.size() > Capacity) {
        return bridge_error(
            OfficialBridgeError::kZoneOverflow,
            static_cast<std::int32_t>(source.size()));
    }
    target->count = static_cast<std::uint16_t>(source.size());
    for (std::size_t index = 0; index < source.size(); ++index) {
        OfficialBridgeResult result = bridge_card_ref(
            source[static_cast<int>(index)], &target->values[index]);
        if (!result) return result;
    }
    return {};
}

template <typename SourceList, std::size_t Capacity>
inline OfficialBridgeResult bridge_area_list(
    const SourceList& source,
    OfficialPodList<OfficialAreaRefPod, Capacity>* target) {
    if (source.size() > Capacity) {
        return bridge_error(
            OfficialBridgeError::kZoneOverflow,
            static_cast<std::int32_t>(source.size()));
    }
    target->count = static_cast<std::uint16_t>(source.size());
    for (std::size_t index = 0; index < source.size(); ++index) {
        OfficialBridgeResult result = bridge_area_ref(
            source[index], &target->values[index]);
        if (!result) return result;
    }
    return {};
}

inline OfficialBridgeResult bridge_activate(
    const ActivateAbilityInfo& source,
    OfficialActivateAbilityPod* target) {
    target->skill_id = source.skillId;
    OfficialBridgeResult result = bridge_area_ref(source.effectCard, &target->effect_card);
    if (!result) return result;
    target->use_player = source.usePlayerIndex;
    target->is_effect_stack = source.isEffectStack;
    target->effect_stack_index = source.effectStackIndex;
    target->is_special_condition = source.isSpecialCondition;
    return {};
}

inline OfficialBridgeResult bridge_trigger_info(
    const TriggerInfo& source,
    OfficialTriggerInfoPod* target) {
    target->value = source.value;
    OfficialBridgeResult result = bridge_area_ref(source.subject, &target->subject);
    if (!result) return result;
    result = bridge_area_ref(source.object, &target->object);
    if (!result) return result;
    target->type = static_cast<std::uint8_t>(source.type);
    target->depth = source.depth;
    return {};
}

inline OfficialBridgeResult bridge_triggered_ability(
    const TriggeredAbility& source,
    OfficialTriggeredAbilityPod* target) {
    OfficialBridgeResult result = bridge_activate(source.activateInfo, &target->activate);
    if (!result) return result;
    return bridge_trigger_info(source.trigger, &target->trigger);
}

inline OfficialBridgeResult bridge_effect_state(
    const EffectState& source,
    OfficialEffectStatePod* target) {
    OfficialBridgeResult result = bridge_activate(source.ability, &target->ability);
    if (!result) return result;
    target->damage_change = source.damageChange;
    target->effect_rate = source.effectRate;
    target->effect_index = source.effectIndex;
    target->on_effect = source.onEffect;
    target->selected_list_index = source.selectedListIndex;
    target->each_list_index = source.eachListIndex;
    return {};
}

inline std::uint64_t bridge_card_runtime_flags(const Card& card) {
    std::uint64_t flags = 0;
    if (card.appear) flags |= kCardAppear;
    if (card.evolved) flags |= kCardEvolved;
    if (card.benchToActive) flags |= kCardBenchToActive;
    if (card.ko) flags |= kCardKo;
    if (card.koAttackDamage) flags |= kCardKoAttackDamage;
    if (card.koEnemyAttackDamage) flags |= kCardKoEnemyAttackDamage;
    if (card.koEnemyAttackDamageActive) flags |= kCardKoEnemyAttackDamageActive;
    if (card.koEnemyExAttackDamage) flags |= kCardKoEnemyExAttackDamage;
    if (card.koEnemyTerastalAttackDamage) flags |= kCardKoEnemyTerastalAttackDamage;
    if (card.koEnemyNAttackDamage) flags |= kCardKoEnemyNAttackDamage;
    if (card.koFull) flags |= kCardKoFull;
    if (card.koPrizePlus1) flags |= kCardKoPrizePlus1;
    if (card.koPrizeDecreaseOnce) flags |= kCardKoPrizeDecreaseOnce;
    if (card.koPrizeZero) flags |= kCardKoPrizeZero;
    if (card.koByDamageToHand) flags |= kCardKoByDamageToHand;
    if (card.noSpecialCondition) flags |= kCardNoSpecialCondition;
    if (card.noSleepParalyzeConfuse) flags |= kCardNoSleepParalyzeConfuse;
    if (card.noSleep) flags |= kCardNoSleep;
    if (card.koNoDamageAndEffectAttackNextEnemyTurn) {
        flags |= kCardKoNoDamageAndEffectAttackNextEnemyTurn;
    }
    return flags;
}

inline OfficialBridgeResult bridge_card(
    const Card& source,
    OfficialCardStatePod* target) {
    target->card_id = source.cardId;
    target->move_counter = source.moveCounter;
    target->attach_move_counter = source.attachMoveCounter;
    target->skill_order = source.skillOrder;
    target->damage = source.damage;
    for (std::size_t index = 0; index < 4; ++index) {
        target->this_turn[index] = source.thisTurn.value[index];
        target->next_turn[index] = source.nextTurn.value[index];
    }
    target->this_turn_enemy = source.thisTurnEnemy.value[0];
    target->next_turn_enemy = source.nextTurnEnemy.value[0];
    target->take_attack_damage_this_turn = source.takeAttackDamageThisTurn;
    target->take_attack_damage_pre_turn = source.takeAttackDamagePreTurn;
    target->cannot_use_attack_id_non_active = source.cannotUseAttackIdNonActive;
    target->player = source.playerIndex;
    target->area = static_cast<std::uint8_t>(source.area);
    target->pre_area = static_cast<std::uint8_t>(source.preArea);
    target->reverse = source.reverse;
    target->ability_used_count = static_cast<std::uint16_t>(source.abilityUsed.size());
    for (int index = 0; index < source.abilityUsed.size(); ++index) {
        target->ability_used[index] = source.abilityUsed[index];
    }
    target->next_enemy_turn_end_battlefield = source.nextEnemyTurnEndStateBattleField;
    target->next_enemy_turn_end = source.nextEnemyTurnEndState;
    for (std::size_t index = 0; index < 3; ++index) {
        target->turn_state[index] = source.turnState[index];
    }
    for (std::size_t index = 0; index < 5; ++index) {
        target->continual_state[index] = source.continualState[index];
    }
    target->runtime_flags = bridge_card_runtime_flags(source);
    target->ko_prize_change_always = source.koPrizeChangeAlways;
    target->ko_prize_change = source.koPrizeChange;
    target->hp_change = source.hpChange;
    return {};
}

inline OfficialBridgeResult bridge_player(
    const PlayerState& source,
    OfficialPlayerStatePod* target) {
    OfficialBridgeResult result = bridge_card_list(source.active, &target->active);
    if (!result) return result;
    result = bridge_card_list(source.bench, &target->bench);
    if (!result) return result;
    result = bridge_card_list(source.prize, &target->prize);
    if (!result) return result;
    result = bridge_card_list(source.hand, &target->hand);
    if (!result) return result;
    result = bridge_card_list(source.deck, &target->deck);
    if (!result) return result;
    result = bridge_card_list(source.trash, &target->trash);
    if (!result) return result;
    result = bridge_card_list(source.energy, &target->energy);
    if (!result) return result;
    result = bridge_card_list(source.tool, &target->tool);
    if (!result) return result;
    result = bridge_card_list(source.preEvolution, &target->pre_evolution);
    if (!result) return result;
    result = bridge_card_list(source.temporary, &target->temporary);
    if (!result) return result;
    target->this_turn = source.thisTurn.value;
    target->next_turn = source.nextTurn.value;
    target->active_state = source.activeState;
    target->continual_state = source.continualState;
    target->turn_state = source.turnState;
    target->player = source.playerIndex;
    target->ko_prize_once_changed = source.koPrizeOnceChanged;
    return {};
}

inline OfficialBridgeResult bridge_rng(
    const std::mt19937& source,
    std::uint64_t draw_count,
    OfficialMt19937* target) {
    std::ostringstream serialized;
    serialized << source;
    std::istringstream input(serialized.str());
    unsigned long long value = 0;
    for (std::size_t index = 0; index < OfficialMt19937::kStateSize; ++index) {
        if (!(input >> value) || value > std::numeric_limits<std::uint32_t>::max()) {
            return bridge_error(
                OfficialBridgeError::kInvalidRngSerialization,
                static_cast<std::int32_t>(index));
        }
        target->words[index] = static_cast<std::uint32_t>(value);
    }
    if (!(input >> value) || value > OfficialMt19937::kStateSize) {
        return bridge_error(OfficialBridgeError::kInvalidRngSerialization, 624);
    }
    target->index = static_cast<std::uint32_t>(value);
    if (input >> value) {
        return bridge_error(OfficialBridgeError::kInvalidRngSerialization, 625);
    }
    target->draw_count = draw_count;
    return {};
}

inline OfficialBridgeResult resolve_option_card(
    const State& state,
    const SelectOption& option,
    CardRef* result) {
    const int active = state.activePlayerIndex();
    try {
        switch (option.type) {
            case SelectOptionType::Card:
            case SelectOptionType::ToolCard:
            case SelectOptionType::EnergyCard:
            case SelectOptionType::Energy:
                *result = state.getCardRef(option.getCardPosition());
                return {};
            case SelectOptionType::Play:
                *result = state.getCardRef(AreaType::Hand, option.param0, active);
                return {};
            case SelectOptionType::Attach:
            case SelectOptionType::Evolve:
                *result = state.getCardRef(
                    static_cast<AreaType>(option.param0), option.param1, active);
                return {};
            case SelectOptionType::Ability:
            case SelectOptionType::Discard:
                *result = state.getCardRef(
                    static_cast<AreaType>(option.param0), option.param1, active);
                return {};
            case SelectOptionType::Skill:
                *result = CardRef(option.param1);
                return {};
            default:
                *result = CardRef(0);
                return {};
        }
    } catch (const std::exception&) {
        return bridge_error(
            OfficialBridgeError::kInvalidOptionCard,
            static_cast<std::int32_t>(option.type));
    }
}

inline OfficialBridgeResult bridge_option(
    const State& state,
    const SelectOption& source,
    OfficialSelectOptionPod* target) {
    target->type = static_cast<std::uint8_t>(source.type);
    target->params[0] = source.param0;
    target->params[1] = source.param1;
    target->params[2] = source.param2;
    target->params[3] = source.param3;
    target->params[4] = source.param4;
    CardRef resolved(0);
    OfficialBridgeResult result = resolve_option_card(state, source, &resolved);
    if (!result) return result;
    OfficialCardRefPod resolved_pod{};
    result = bridge_card_ref(resolved, &resolved_pod);
    if (!result) return result;
    target->resolved_card = resolved_pod.index;
    if (source.type == SelectOptionType::Attach
        || source.type == SelectOptionType::Evolve) {
        const Card& source_card = state.getCard(resolved);
        const CardRef target_ref = state.getCardRef(
            static_cast<AreaType>(source.param2),
            source.param3,
            state.activePlayerIndex());
        const Card& target_card = state.getCard(target_ref);
        target->option_equiv = static_cast<std::uint16_t>(
            (source_card.cardId * 131 + target_card.cardId) & 0xffff);
    } else if (source.type == SelectOptionType::Ability) {
        const Card& card = state.getCard(resolved);
        const Skill* ability = card.getMaster().ability;
        target->option_equiv = static_cast<std::uint16_t>(
            ability == nullptr ? 0 : ability->skillId);
    } else if (!resolved.isNull()) {
        target->option_equiv = static_cast<std::uint16_t>(
            state.getCard(resolved).cardId);
    } else if (source.type == SelectOptionType::Attack) {
        target->option_equiv = static_cast<std::uint16_t>(source.param0);
    } else if (source.type == SelectOptionType::Skill) {
        target->option_equiv = static_cast<std::uint16_t>(source.param0);
    }
    return {};
}

inline OfficialBridgeResult bridge_turn_history(
    const TurnHistory& source,
    OfficialTurnHistoryPod* target) {
    if (source.turnAttackId < std::numeric_limits<std::int16_t>::min()
        || source.turnAttackId > std::numeric_limits<std::int16_t>::max()) {
        return bridge_error(
            OfficialBridgeError::kNarrowingOverflow, source.turnAttackId);
    }
    OfficialBridgeResult result = bridge_card_ref(source.turnAttackCard, &target->attack_card);
    if (!result) return result;
    target->attack_id = static_cast<std::int16_t>(source.turnAttackId);
    target->take_prize_count = source.takePrizeCountTurnPlayer;
    target->flags = static_cast<std::uint8_t>(source.ko)
        | (static_cast<std::uint8_t>(source.koTeamRocket) << 1U)
        | (static_cast<std::uint8_t>(source.koAttackDamage) << 2U)
        | (static_cast<std::uint8_t>(source.koAttackDamageEthan) << 3U)
        | (static_cast<std::uint8_t>(source.koAttackDamageHop) << 4U);
    return {};
}

}  // namespace detail

inline OfficialBridgeResult bridge_official_state(
    const State& source,
    const OfficialStateBridgeContext& context,
    OfficialStatePod* target) {
    if (target == nullptr) {
        return detail::bridge_error(OfficialBridgeError::kNullOutput);
    }
    *target = OfficialStatePod{};
    if (source.game == nullptr) {
        return detail::bridge_error(OfficialBridgeError::kMissingGame);
    }
    if (!context.rng_draw_count_valid) {
        return detail::bridge_error(OfficialBridgeError::kMissingRngDrawCount);
    }
    target->episode_id = context.episode_id;
    OfficialBridgeResult result = detail::bridge_rng(
        source.game->rng, context.rng_draw_count, &target->rng);
    if (!result) return result;

    target->turn = source.turn;
    target->turn_action_count = source.turnActionCount;
    target->effect_action_count = source.effectActionCount;
    target->turn_attack_count = source.turnAttackCount;
    target->current_card_effect_index = source.currentCardEffectIndex;
    target->coin_head_count = source.coinHeadCount;
    target->move_counter = source.moveCounter;
    target->current_skill_order = source.currentSkillOrder;
    target->phase = static_cast<std::uint8_t>(source.phase);
    target->game_result = static_cast<std::uint8_t>(source.gameResult);
    target->finish_reason = static_cast<std::uint8_t>(source.finishReason);
    target->first_player = source.firstPlayer;
    target->looking_player = source.lookingPlayer;
    target->looking_reverse = source.lookingReverse;
    target->setup_done_mask = static_cast<std::uint8_t>(source.setupDone[0])
        | (static_cast<std::uint8_t>(source.setupDone[1]) << 1U);
    target->mulligan_mask = static_cast<std::uint8_t>(source.mulligan[0])
        | (static_cast<std::uint8_t>(source.mulligan[1]) << 1U);
    for (int player = 0; player < 2; ++player) {
        if (source.mulliganCount[player] < std::numeric_limits<std::int16_t>::min()
            || source.mulliganCount[player] > std::numeric_limits<std::int16_t>::max()) {
            return detail::bridge_error(
                OfficialBridgeError::kNarrowingOverflow,
                source.mulliganCount[player]);
        }
        target->mulligan_count[player] = static_cast<std::int16_t>(
            source.mulliganCount[player]);
    }
    target->control_flags = static_cast<std::uint8_t>(source.changed)
        | (static_cast<std::uint8_t>(source.isBreak) << 1U)
        | (static_cast<std::uint8_t>(source.effectLoopStop) << 2U)
        | (static_cast<std::uint8_t>(source.failRetreat) << 3U)
        | (static_cast<std::uint8_t>(source.stateChanged) << 4U)
        | (static_cast<std::uint8_t>(source.updateOrder) << 5U);
    target->turn_state = source.turnState;
    target->continual_state = source.continualState;
    target->last_stadium_player = source.lastStadiumPlayer;

    target->select_type = static_cast<std::uint8_t>(source.selectType);
    target->select_context = static_cast<std::uint8_t>(source.selectContext);
    target->select_player = source.selectPlayer;
    target->select_deck = source.selectDeck;
    if (source.selectMin < std::numeric_limits<std::int16_t>::min()
        || source.selectMin > std::numeric_limits<std::int16_t>::max()
        || source.selectMax < std::numeric_limits<std::int16_t>::min()
        || source.selectMax > std::numeric_limits<std::int16_t>::max()) {
        return detail::bridge_error(OfficialBridgeError::kNarrowingOverflow, source.selectMax);
    }
    target->select_min = static_cast<std::int16_t>(source.selectMin);
    target->select_max = static_cast<std::int16_t>(source.selectMax);
    target->remain_damage_counter = source.remainDamageCounter;
    target->energy_cost = source.energyCost;
    target->remain_energy_cost = source.remainEnergyCost;
    target->selected_energy_card_count = source.selectedEnergyCardCount;
    target->removed_damage_counter = source.removedDamageCounter;
    for (std::size_t index = 0; index < source.selectCounts.size(); ++index) {
        if (source.selectCounts[index] < std::numeric_limits<std::int16_t>::min()
            || source.selectCounts[index] > std::numeric_limits<std::int16_t>::max()) {
            return detail::bridge_error(
                OfficialBridgeError::kNarrowingOverflow, source.selectCounts[index]);
        }
        target->select_counts[index] = static_cast<std::int16_t>(source.selectCounts[index]);
    }
    result = detail::bridge_card_ref(
        source.selectingEnergyPokemonRef, &target->selecting_energy_pokemon);
    if (!result) return result;
    result = detail::bridge_card_ref(source.contextCard, &target->context_card);
    if (!result) return result;
    result = detail::bridge_effect_state(source.effectState, &target->effect_state);
    if (!result) return result;
    result = detail::bridge_trigger_info(source.triggerInfo, &target->trigger_info);
    if (!result) return result;
    target->effect_jump = source.effectJump;
    target->attach_active = source.attachActive;
    target->current_attack_id = source.currentAttackId;
    target->source_attack_id = source.srcAttackId;
    target->attack_damage_change = source.attackDamageChange;
    target->last_attack_damage = source.lastAttackDamage;
    result = detail::bridge_card_ref(source.attacker, &target->attacker);
    if (!result) return result;
    target->post_attack_effect = source.postAttackEffect;
    target->post_effect_activate = source.postEffectActivate;
    target->fail_attack = source.failAttack;
    target->second_attack = source.secondAttack;

    for (std::size_t index = 0; index < source.turnHistories.size(); ++index) {
        result = detail::bridge_turn_history(source.turnHistories[index], &target->turn_histories[index]);
        if (!result) return result;
    }
    result = detail::bridge_card_list(source.stadium, &target->stadium);
    if (!result) return result;
    result = detail::bridge_card_list(source.looking, &target->looking);
    if (!result) return result;
    result = detail::bridge_card_list(source.selectedList, &target->selected_list);
    if (!result) return result;
    result = detail::bridge_card_list(source.eachList, &target->each_list);
    if (!result) return result;
    result = detail::bridge_card_list(source.playing, &target->playing);
    if (!result) return result;
    result = detail::bridge_card_list(source.checkList, &target->check_list);
    if (!result) return result;
    for (int player = 0; player < 2; ++player) {
        result = detail::bridge_player(source.players[player], &target->players[player]);
        if (!result) return result;
        target->log_index[player] = source.logIndex[player];
    }
    for (std::size_t index = 0; index < source.allCard.size(); ++index) {
        result = detail::bridge_card(source.allCard[index], &target->cards[index]);
        if (!result) return result;
    }

    if (source.options.size() > kOfficialOptionCapacity) {
        return detail::bridge_error(
            OfficialBridgeError::kOptionOverflow,
            static_cast<std::int32_t>(source.options.size()));
    }
    target->options.count = static_cast<std::uint16_t>(source.options.size());
    for (std::size_t index = 0; index < source.options.size(); ++index) {
        result = detail::bridge_option(source, source.options[index], &target->options.values[index]);
        if (!result) return result;
    }
    if (source.selected.size() > kOfficialOptionCapacity) {
        return detail::bridge_error(
            OfficialBridgeError::kSelectionOverflow,
            static_cast<std::int32_t>(source.selected.size()));
    }
    target->selected.count = static_cast<std::uint16_t>(source.selected.size());
    for (std::size_t index = 0; index < source.selected.size(); ++index) {
        if (source.selected[index] < 0
            || source.selected[index] > std::numeric_limits<std::uint16_t>::max()) {
            return detail::bridge_error(
                OfficialBridgeError::kNarrowingOverflow, source.selected[index]);
        }
        target->selected.values[index] = static_cast<std::uint16_t>(source.selected[index]);
    }
    result = detail::bridge_area_list(source.preTargetList, &target->pre_targets);
    if (!result) return result;
    result = detail::bridge_area_list(source.targetList, &target->targets);
    if (!result) return result;
    result = detail::bridge_area_list(source.koList, &target->ko_list);
    if (!result) return result;

    auto bridge_triggers = [&](const auto& source_list, auto* target_list) -> OfficialBridgeResult {
        if (source_list.size() > target_list->capacity()) {
            return detail::bridge_error(
                OfficialBridgeError::kTriggerOverflow,
                static_cast<std::int32_t>(source_list.size()));
        }
        target_list->count = static_cast<std::uint16_t>(source_list.size());
        for (std::size_t index = 0; index < source_list.size(); ++index) {
            OfficialBridgeResult item = detail::bridge_triggered_ability(
                source_list[index], &target_list->values[index]);
            if (!item) return item;
        }
        return {};
    };
    result = bridge_triggers(source.delayTriggerStack, &target->delay_triggers);
    if (!result) return result;
    result = bridge_triggers(source.temporaryTriggerStack, &target->temporary_triggers);
    if (!result) return result;
    result = bridge_triggers(source.triggerStack, &target->triggers);
    if (!result) return result;

    if (source.turnUsedSkill.size() > kOfficialTurnRecordCapacity
        || source.turnPlay.size() > kOfficialTurnRecordCapacity
        || source.turnHeal.size() > kOfficialTurnRecordCapacity
        || source.turnEvolve.size() > kOfficialTurnRecordCapacity) {
        return detail::bridge_error(OfficialBridgeError::kTurnRecordOverflow);
    }
    target->turn_used_skills.count = static_cast<std::uint16_t>(source.turnUsedSkill.size());
    for (std::size_t index = 0; index < source.turnUsedSkill.size(); ++index) {
        target->turn_used_skills.values[index] = source.turnUsedSkill[index];
    }
    result = detail::bridge_card_list(source.turnPlay, &target->turn_play);
    if (!result) return result;
    result = detail::bridge_card_list(source.turnHeal, &target->turn_heal);
    if (!result) return result;
    target->turn_evolve.count = static_cast<std::uint16_t>(source.turnEvolve.size());
    for (std::size_t index = 0; index < source.turnEvolve.size(); ++index) {
        result = detail::bridge_card_ref(source.turnEvolve[index].preRef, &target->turn_evolve.values[index].from);
        if (!result) return result;
        result = detail::bridge_card_ref(source.turnEvolve[index].ref, &target->turn_evolve.values[index].to);
        if (!result) return result;
    }

    if (source.functionStack.size() > kOfficialContinuationCapacity) {
        return detail::bridge_error(
            OfficialBridgeError::kContinuationOverflow,
            static_cast<std::int32_t>(source.functionStack.size()));
    }
    target->continuations.count = static_cast<std::uint16_t>(source.functionStack.size());
    for (std::size_t index = 0; index < source.functionStack.size(); ++index) {
        const GameFunction& source_frame = source.functionStack[index];
        if (source_frame.functionIndex < 0
            || source_frame.functionIndex >= static_cast<int>(FunctionTable.size())
            || source_frame.functionIndex > std::numeric_limits<std::uint16_t>::max()) {
            return detail::bridge_error(
                OfficialBridgeError::kInvalidContinuation,
                source_frame.functionIndex);
        }
        OfficialContinuationPod& target_frame = target->continuations.values[index];
        target_frame.args[0] = source_frame.arg0;
        target_frame.args[1] = source_frame.arg1;
        target_frame.args[2] = source_frame.arg2;
        target_frame.opcode = static_cast<std::uint16_t>(source_frame.functionIndex);
        target_frame.arg_type = static_cast<std::uint8_t>(source_frame.argType);
        target_frame.call_count = source_frame.callCount;
        target_frame.called_count = source_frame.calledCount;
    }
    return {};
}

}  // namespace ptcg::cuda_engine::extractor
