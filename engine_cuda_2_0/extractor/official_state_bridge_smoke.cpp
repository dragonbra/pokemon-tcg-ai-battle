#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "official_state_bridge.h"

namespace {

using namespace ptcg::cuda_engine;
using namespace ptcg::cuda_engine::extractor;

void require(bool condition, const std::string& label) {
    if (!condition) throw std::runtime_error("official state bridge smoke: " + label);
}

void populate_state(Game* game, State* state) {
    state->game = game;
    state->turn = 1;
    state->turnActionCount = 2;
    state->effectActionCount = 3;
    state->turnAttackCount = 4;
    state->phase = GamePhase::Main;
    state->gameResult = GameResult::Player1Win;
    state->finishReason = FinishReason::Effect;
    state->setupDone = {true, false};
    state->mulligan = {false, true};
    state->mulliganCount = {5, 6};
    state->firstPlayer = 0;
    state->lookingPlayer = 1;
    state->lookingReverse = true;
    state->changed = true;
    state->isBreak = true;
    state->effectLoopStop = true;
    state->failRetreat = true;
    state->stateChanged = true;
    state->updateOrder = true;
    state->turnState = 0x15;
    state->continualState = 0x01;
    state->currentCardEffectIndex = 7;
    state->coinHeadCount = 8;
    state->lastStadiumPlayer = 1;
    state->moveCounter = 9;
    state->currentSkillOrder = 10;

    state->selectType = SelectType::Card;
    state->selectContext = SelectContext::EffectTarget;
    state->selectPlayer = 0;
    state->selectDeck = true;
    state->selectMin = 1;
    state->selectMax = 2;
    state->remainDamageCounter = 11;
    state->energyCost = 12;
    state->remainEnergyCost = 13;
    state->selectedEnergyCardCount = 14;
    state->removedDamageCounter = 15;
    state->selectCounts[2] = 16;
    state->selectingEnergyPokemonRef = CardRef(5);
    state->contextCard = CardRef(6);

    state->effectState.ability.skillId = 17;
    state->effectState.ability.effectCard = {CardRef(5), 18};
    state->effectState.ability.usePlayerIndex = 1;
    state->effectState.ability.isEffectStack = true;
    state->effectState.ability.effectStackIndex = 2;
    state->effectState.ability.isSpecialCondition = true;
    state->effectState.effectIndex = 3;
    state->effectState.onEffect = true;
    state->effectState.selectedListIndex = 4;
    state->effectState.eachListIndex = 5;
    state->effectState.effectRate = 6;
    state->effectState.damageChange = 19;
    state->triggerInfo.type = static_cast<TriggerType>(7);
    state->triggerInfo.depth = 2;
    state->triggerInfo.value = 20;
    state->triggerInfo.subject = {CardRef(5), 21};
    state->triggerInfo.object = {CardRef(6), 22};
    state->effectJump = 4;
    state->attachActive = true;

    state->currentAttackId = 23;
    state->srcAttackId = 24;
    state->attackDamageChange = 25;
    state->lastAttackDamage = 26;
    state->attacker = CardRef(5);
    state->postAttackEffect = true;
    state->postEffectActivate = true;
    state->failAttack = true;
    state->secondAttack = true;

    for (int player = 0; player < 2; ++player) {
        PlayerState& ps = state->players[player];
        ps.playerIndex = static_cast<signed char>(player);
        ps.koPrizeOnceChanged = player == 1;
        ps.thisTurn.value = 0x100U + player;
        ps.nextTurn.value = 0x200U + player;
        ps.activeState = 0x300U + player;
        ps.continualState = 0x400ULL + player;
        ps.turnState = 0x500ULL + player;
    }
    state->players[0].active.push_back(CardRef(5));
    state->players[0].bench.push_back(CardRef(6));
    state->players[0].hand.push_back(CardRef(7));
    state->players[0].deck.push_back(CardRef(8));
    state->players[0].prize.push_back(CardRef(9));
    state->players[0].trash.push_back(CardRef(10));
    state->players[0].energy.push_back(CardRef(11));
    state->players[0].tool.push_back(CardRef(12));
    state->players[0].preEvolution.push_back(CardRef(13));
    state->players[0].temporary.push_back(CardRef(14));
    state->players[1].active.push_back(CardRef(15));
    state->logIndex = {27, 28};

    for (int index = 1; index <= 15; ++index) {
        Card& card = state->allCard[index];
        card.cardId = 100 + index;
        card.moveCounter = 200 + index;
        card.playerIndex = index < 15 ? 0 : 1;
        card.area = index == 7 ? AreaType::Hand : AreaType::Deck;
    }
    Card& card = state->allCard[5];
    card.attachMoveCounter = 29;
    card.skillOrder = 30;
    card.damage = 40;
    card.thisTurn.value = {1, 2, 3, 4};
    card.nextTurn.value = {5, 6, 7, 8};
    card.thisTurnEnemy.value = {9};
    card.nextTurnEnemy.value = {10};
    card.takeAttackDamageThisTurn = 31;
    card.takeAttackDamagePreTurn = 32;
    card.cannotUseAttackIdNonActive = 33;
    card.preArea = AreaType::Bench;
    card.reverse = true;
    card.abilityUsed.push_back(34);
    card.nextEnemyTurnEndStateBattleField = 35;
    card.nextEnemyTurnEndState = 36;
    card.turnState = {0, 0, 0};
    card.appear = true;
    card.ko = true;
    card.koPrizeChangeAlways = -1;
    card.koPrizeChange = 2;
    card.continualState = {0, 0, 0, 0, 0};
    card.hpChange = 50;
    card.noSpecialCondition = true;

    state->turnHistories[0].ko = true;
    state->turnHistories[0].koAttackDamage = true;
    state->turnHistories[0].turnAttackCard = CardRef(5);
    state->turnHistories[0].takePrizeCountTurnPlayer = 2;
    state->turnHistories[0].turnAttackId = 37;
    state->stadium.push_back(CardRef(6));
    state->looking.push_back(CardRef(7));
    state->selectedList.push_back(CardRef(8));
    state->eachList.push_back(CardRef(9));
    state->playing.push_back(CardRef(10));
    state->checkList.push_back(CardRef(11));
    state->preTargetList.push_back({CardRef(5), 38});
    state->targetList.push_back({CardRef(6), 39});
    state->koList.push_back({CardRef(15), 40});

    AddOptionCard(*state, AreaType::Hand, 0, 0);
    state->selected.push_back(0);

    TriggeredAbility trigger{};
    trigger.activateInfo.skillId = 41;
    trigger.activateInfo.effectCard = {CardRef(5), 42};
    trigger.activateInfo.usePlayerIndex = 0;
    trigger.trigger.type = static_cast<TriggerType>(3);
    trigger.trigger.depth = 1;
    trigger.trigger.value = 43;
    trigger.trigger.subject = {CardRef(5), 44};
    trigger.trigger.object = {CardRef(6), 45};
    state->delayTriggerStack.push_back(trigger);
    state->temporaryTriggerStack.push_back(trigger);
    state->triggerStack.push_back(trigger);
    state->turnUsedSkill.push_back(46);
    state->turnPlay.push_back(CardRef(7));
    state->turnHeal.push_back(CardRef(5));
    state->turnEvolve.push_back({CardRef(6), CardRef(5)});

    GameFunction continuation{};
    continuation.functionIndex = 47;
    continuation.arg0 = 48;
    continuation.arg1 = 49;
    continuation.arg2 = 50;
    continuation.argType = ArgType::III;
    continuation.callCount = 3;
    continuation.calledCount = 2;
    state->functionStack.push_back(continuation);
}

void check_bridge(const State& state, const OfficialStatePod& pod) {
    require(pod.abi_version == 6, "abi");
    require(pod.episode_id == 0x123456789ULL, "episode");
    require(pod.rng.draw_count == 17, "rng draw count");
    require(pod.turn == 1 && pod.turn_action_count == 2, "turn counters");
    require(pod.control_flags == 0x3f, "official control flags");
    require(pod.flow_flags == 0, "flow flags isolated");
    require(pod.setup_done_mask == 1 && pod.mulligan_mask == 2, "masks");
    require(pod.select_type == static_cast<std::uint8_t>(SelectType::Card), "select type");
    require(pod.select_counts[2] == 16, "select counts");
    require(pod.effect_state.ability.skill_id == 17, "effect ability");
    require(pod.trigger_info.object.move_counter == 22, "trigger object");
    require(pod.players[0].hand.count == 1, "hand count");
    require(pod.players[0].hand.values[0].index == 7, "hand ref");
    require(pod.players[1].active.values[0].index == 15, "enemy active");
    require(pod.cards[5].card_id == 105, "card id");
    require(pod.cards[5].ability_used_count == 1, "ability count");
    require(pod.cards[5].ability_used[0] == 34, "ability id");
    require((pod.cards[5].runtime_flags & kCardAppear) != 0, "appear flag");
    require((pod.cards[5].runtime_flags & kCardKo) != 0, "ko flag");
    require((pod.cards[5].runtime_flags & kCardNoSpecialCondition) != 0, "condition flag");
    require(pod.cards[5].hp_change == 50, "hp change");
    require(pod.turn_histories[0].flags == 5, "history flags");
    require(pod.options.count == 1, "option count");
    require(pod.options.values[0].resolved_card == 7, "resolved option card");
    require(pod.options.values[0].option_equiv == 107, "option equivalence");
    require(pod.pre_targets.values[0].move_counter == 38, "pre target");
    require(pod.delay_triggers.values[0].activate.skill_id == 41, "delay trigger");
    require(pod.turn_used_skills.values[0] == 46, "turn skill");
    require(pod.turn_evolve.values[0].from.index == 6, "evolve from");
    require(pod.continuations.count == 1, "continuation count");
    require(pod.continuations.values[0].opcode == 47, "continuation opcode");
    require(pod.continuations.values[0].args[2] == 50, "continuation args");
    require(pod.effect_stack.count == 0, "private effect stack starts empty");
    require(pod.effect_ref_scratch.count == 0, "private scratch starts empty");
}

}  // namespace

int main() {
    try {
        InitializeAll();
        Game game{};
        game.rng = std::mt19937(12345);
        for (std::uint64_t draw = 0; draw < 17; ++draw) game.rng();
        State state{};
        populate_state(&game, &state);

        OfficialStatePod pod{};
        const OfficialBridgeResult result = bridge_official_state(
            state, {0x123456789ULL, 17, true}, &pod);
        require(static_cast<bool>(result), "bridge success");
        check_bridge(state, pod);

        OfficialStatePod rejected{};
        const OfficialBridgeResult missing_rng = bridge_official_state(
            state, {1, 0, false}, &rejected);
        require(
            missing_rng.error == OfficialBridgeError::kMissingRngDrawCount,
            "missing rng rejected");

        State overflow = state;
        overflow.options.resize(kOfficialOptionCapacity + 1);
        const OfficialBridgeResult overflow_result = bridge_official_state(
            overflow, {1, 17, true}, &rejected);
        require(
            overflow_result.error == OfficialBridgeError::kOptionOverflow,
            "option overflow rejected");

        std::cout
            << "{\"passed\":true,\"state_abi\":" << kOfficialStateAbiVersion
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"direct_fields\":true"
            << ",\"continuations\":1"
            << ",\"overflow_rejected\":true"
            << ",\"missing_rng_rejected\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
