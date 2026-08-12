#include <cstdint>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#include "All.h"
#include "official_state_bridge.h"
#include "ptcg_cuda/official_flow_fixture.h"
#include "ptcg_cuda/official_flow_dispatch_pod.cuh"

namespace {

using namespace ptcg::cuda_engine;
using namespace ptcg::cuda_engine::extractor;

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open rule pack: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void require(bool condition, const std::string& label) {
    if (!condition) throw std::runtime_error("official attack oracle failed: " + label);
}

void clear_official(State* state) {
    for (int player = 0; player < 2; ++player) {
        PlayerState& ps = state->players[player];
        ps.active.clear();
        ps.bench.clear();
        ps.prize.clear();
        ps.hand.clear();
        ps.deck.clear();
        ps.trash.clear();
        ps.energy.clear();
        ps.tool.clear();
        ps.preEvolution.clear();
        ps.temporary.clear();
        ps.playerIndex = static_cast<signed char>(player);
        ps.activeState = 0;
        ps.continualState = 0;
        ps.turnState = 0;
        ps.thisTurn.value = 0;
        ps.nextTurn.value = 0;
    }
    state->stadium.clear();
    state->looking.clear();
    state->selectedList.clear();
    state->eachList.clear();
    state->playing.clear();
    state->checkList.clear();
    state->options.clear();
    state->selected.clear();
    state->preTargetList.clear();
    state->targetList.clear();
    state->koList.clear();
    state->delayTriggerStack.clear();
    state->temporaryTriggerStack.clear();
    state->triggerStack.clear();
    state->turnUsedSkill.clear();
    state->turnPlay.clear();
    state->turnHeal.clear();
    state->turnEvolve.clear();
    state->functionStack.clear();
    state->logs.clear();
    state->allCard = {};
    state->selectType = SelectType::None;
    state->selectContext = SelectContext::None;
    state->selectPlayer = -1;
    state->firstPlayer = 0;
    state->lastStadiumPlayer = 0;
    state->turn = 3;
    state->phase = GamePhase::Main;
    state->gameResult = GameResult::None;
    state->finishReason = FinishReason::None;
    state->moveCounter = 1;
    state->attacker = {};
    state->currentAttackId = 0;
    state->srcAttackId = 0;
    state->coinHeadCount = 0;
    state->attackDamageChange = 0;
    state->lastAttackDamage = 0;
    state->turnAttackCount = 0;
    state->secondAttack = false;
    state->failAttack = false;
    state->postAttackEffect = false;
    state->postEffectActivate = false;
}

void place_official(
    State* state,
    int ref_value,
    int card_id,
    int player,
    AreaType area) {
    Card& card = state->getCard(CardRef(ref_value));
    card = {};
    card.init(card_id, state->moveCounter++, player);
    card.area = area;
    PlayerState& ps = state->players[player];
    switch (area) {
        case AreaType::Deck: ps.deck.push_back(CardRef(ref_value)); break;
        case AreaType::Hand: ps.hand.push_back(CardRef(ref_value)); break;
        case AreaType::Trash: ps.trash.push_back(CardRef(ref_value)); break;
        case AreaType::Active: ps.active.push_back(CardRef(ref_value)); break;
        case AreaType::Bench: ps.bench.push_back(CardRef(ref_value)); break;
        case AreaType::Prize: ps.prize.push_back(CardRef(ref_value)); break;
        case AreaType::Energy: ps.energy.push_back(CardRef(ref_value)); break;
        case AreaType::Tool: ps.tool.push_back(CardRef(ref_value)); break;
        case AreaType::Stadium: state->stadium.push_back(CardRef(ref_value)); break;
        default: throw std::runtime_error("unsupported oracle area");
    }
}

void setup_official(BattleData* battle, std::uint64_t seed) {
    GameConfig config{};
    config.seed = static_cast<std::uint32_t>(seed);
    config.recordLog = false;
    config.deviceRand = false;
    for (int player = 0; player < 2; ++player) {
        for (int index = 0; index < DECK_SIZE; ++index) {
            config.decks[player].cards[index] = 7;
        }
    }
    battle->init(config, false);
    battle->game.rng = std::mt19937(static_cast<std::uint32_t>(seed));
    clear_official(&battle->state);
}

void drive_official(State* state, int budget = 10000) {
    for (int step = 0; step < budget; ++step) {
        if (state->selectType != SelectType::None || state->isFinish()) return;
        if (state->functionStack.empty()) return;
        state->callFunction();
    }
    throw std::runtime_error("official function budget exceeded");
}

void choose_official(State* state, std::initializer_list<int> indices) {
    state->selected.clear();
    for (int index : indices) state->selected.push_back(index);
    require(state->selectType != SelectType::None, "official selection exists");
    state->callFunction();
    drive_official(state);
}

void compare_options(
    const State& official,
    const OfficialStatePod& pod,
    const std::string& label) {
    require(official.options.size() == pod.options.count, label + ".option_count");
    require(static_cast<int>(official.selectType) == pod.select_type, label + ".select_type");
    require(static_cast<int>(official.selectContext) == pod.select_context, label + ".select_context");
    require(official.selectPlayer == pod.select_player, label + ".select_player");
    require(official.selectMin == pod.select_min, label + ".select_min");
    require(official.selectMax == pod.select_max, label + ".select_max");
    for (std::size_t index = 0; index < official.options.size(); ++index) {
        const SelectOption& expected = official.options[index];
        const OfficialSelectOptionPod& actual = pod.options.values[index];
        require(static_cast<int>(expected.type) == actual.type, label + ".option_type");
        require(expected.param0 == actual.params[0], label + ".option_param0");
        require(expected.param1 == actual.params[1], label + ".option_param1");
        require(expected.param2 == actual.params[2], label + ".option_param2");
        require(expected.param3 == actual.params[3], label + ".option_param3");
    }
}

void compare_rng(const std::mt19937& official, const OfficialMt19937& pod) {
    std::ostringstream stream;
    stream << official;
    std::istringstream input(stream.str());
    for (std::size_t index = 0; index < OfficialMt19937::kStateSize; ++index) {
        std::uint64_t value = 0;
        input >> value;
        require(static_cast<std::uint32_t>(value) == pod.words[index], "rng word");
    }
    std::uint64_t cursor = 0;
    input >> cursor;
    require(cursor == pod.index, "rng cursor");
}

void prepare_official_case(
    State* state,
    int attacker_card_id,
    int target_card_id,
    int energy_card_id) {
    place_official(state, 10, attacker_card_id, 0, AreaType::Active);
    place_official(state, 11, energy_card_id, 0, AreaType::Energy);
    state->getCard(CardRef(11)).attachMoveCounter =
        state->getCard(CardRef(10)).moveCounter;
    place_official(state, 20, target_card_id, 1, AreaType::Active);
    place_official(state, 30, 1, 0, AreaType::Prize);
    place_official(state, 31, 1, 1, AreaType::Prize);
    place_official(state, 40, 1, 0, AreaType::Deck);
    place_official(state, 41, 1, 1, AreaType::Deck);
}

void prepare_pod_case(
    const State& source,
    OfficialStatePod* state,
    std::uint64_t episode) {
    const OfficialBridgeResult result = bridge_official_state(
        source, OfficialStateBridgeContext{episode, 0, true}, state);
    require(
        static_cast<bool>(result),
        "initial state bridge error="
            + std::to_string(static_cast<std::int32_t>(result.error))
            + ".detail=" + std::to_string(result.detail));
}

void run_damage_case(
    const OfficialRulePackView& rules,
    std::uint64_t seed,
    int attacker_card_id,
    int target_card_id,
    int attack_id,
    int energy_card_id,
    int expected_damage,
    const std::string& label) {
    BattleData battle;
    setup_official(&battle, seed);
    State& official = battle.state;
    prepare_official_case(
        &official, attacker_card_id, target_card_id, energy_card_id);
    OfficialStatePod pod{};
    prepare_pod_case(official, &pod, 800 + seed);
    SelectedAttack(official, attack_id, 0, -1);
    drive_official(&official);

    const OfficialAttackResult result = official_begin_attack(
        &pod, rules, OfficialCardRefPod{10}, attack_id);
    require(result == OfficialAttackResult::kComplete, label + ".pod_complete");
    require(
        official.getCard(CardRef(20)).damage == expected_damage,
        label + ".official_damage expected=" + std::to_string(expected_damage)
            + " actual=" + std::to_string(official.getCard(CardRef(20)).damage));
    require(
        pod.cards[20].damage == expected_damage,
        label + ".pod_damage expected=" + std::to_string(expected_damage)
            + " actual=" + std::to_string(pod.cards[20].damage));
    require(official.turn == pod.turn, label + ".turn");
    require(official.players[1].hand.size() == pod.players[1].hand.count, label + ".hand");
    require(official.players[1].deck.size() == pod.players[1].deck.count, label + ".deck");
    require(official.players[0].prize.size() == pod.players[0].prize.count, label + ".prize0");
    require(official.players[1].prize.size() == pod.players[1].prize.count, label + ".prize1");
    require(official.currentAttackId == pod.current_attack_id, label + ".current_attack");
    require(pod.error == 0, label + ".pod_error");
}

void run_confusion_case(
    const OfficialRulePackView& rules,
    std::uint64_t seed,
    bool expect_head,
    const std::string& label) {
    BattleData battle;
    setup_official(&battle, seed);
    State& official = battle.state;
    prepare_official_case(&official, 39, 24, 1);
    official.players[0].badStatus = BadStatusType::Confused;
    OfficialStatePod pod{};
    prepare_pod_case(official, &pod, 900 + seed);
    SelectedAttack(official, 31, 0, -1);
    drive_official(&official);

    const OfficialAttackResult result = official_begin_attack(
        &pod, rules, OfficialCardRefPod{10}, 31);
    require(result == OfficialAttackResult::kComplete, label + ".pod_complete");
    const int expected_target = expect_head ? 30 : 0;
    const int expected_attacker = expect_head ? 0 : 30;
    require(official.getCard(CardRef(20)).damage == expected_target, label + ".official_target");
    require(pod.cards[20].damage == expected_target, label + ".pod_target");
    require(official.getCard(CardRef(10)).damage == expected_attacker, label + ".official_self");
    require(pod.cards[10].damage == expected_attacker, label + ".pod_self");
    require(official.turn == pod.turn, label + ".turn");
    compare_rng(battle.game.rng, pod.rng);
    require(pod.error == 0, label + ".pod_error");
}

void run_no_damage_coin_case(
    const OfficialRulePackView& rules,
    std::uint64_t seed,
    int attacker_card_id,
    int attack_id,
    int expected_damage,
    const std::string& label) {
    BattleData battle;
    setup_official(&battle, seed);
    State& official = battle.state;
    prepare_official_case(&official, attacker_card_id, 970, 1);
    place_official(&official, 21, 7, 1, AreaType::Energy);
    official.getCard(CardRef(21)).attachMoveCounter =
        official.getCard(CardRef(20)).moveCounter;
    OfficialStatePod pod{};
    prepare_pod_case(official, &pod, 1000 + seed);
    SelectedAttack(official, attack_id, 0, -1);
    drive_official(&official);

    const OfficialAttackResult result = official_begin_attack(
        &pod, rules, OfficialCardRefPod{10}, attack_id);
    require(result == OfficialAttackResult::kComplete, label + ".pod_complete");
    require(official.getCard(CardRef(20)).damage == expected_damage, label + ".official_damage");
    require(pod.cards[20].damage == expected_damage, label + ".pod_damage");
    require(official.turn == pod.turn, label + ".turn");
    compare_rng(battle.game.rng, pod.rng);
    require(pod.error == 0, label + ".pod_error");
}

void run_knockout_case(const OfficialRulePackView& rules) {
    constexpr std::uint64_t seed = 53;
    BattleData battle;
    setup_official(&battle, seed);
    State& official = battle.state;
    prepare_official_case(&official, 39, 160, 1);
    place_official(&official, 22, 24, 1, AreaType::Bench);
    place_official(&official, 32, 1, 0, AreaType::Prize);
    OfficialStatePod pod{};
    prepare_pod_case(official, &pod, 1100);
    SelectedAttack(official, 31, 0, -1);
    drive_official(&official);

    OfficialAttackResult result = official_begin_attack(
        &pod, rules, OfficialCardRefPod{10}, 31);
    require(result == OfficialAttackResult::kNeedsAction, "knockout.prize_needed");
    compare_options(official, pod, "knockout.prize");

    choose_official(&official, {0});
    const std::uint16_t prize = 0;
    OfficialFlowStatus flow = official_apply_pending_action(
        &pod, rules, &prize, 1);
    require(flow == OfficialFlowStatus::kNeedsAction, "knockout.replacement_needed");
    compare_options(official, pod, "knockout.replacement");

    choose_official(&official, {0});
    const std::uint16_t replacement = 0;
    flow = official_apply_pending_action(&pod, rules, &replacement, 1);
    require(flow == OfficialFlowStatus::kIdle, "knockout.complete");
    require(official.turn == pod.turn, "knockout.turn");
    require(official.players[0].hand.size() == pod.players[0].hand.count, "knockout.hand");
    require(official.players[0].prize.size() == pod.players[0].prize.count, "knockout.prize_count");
    require(official.players[1].trash.size() == pod.players[1].trash.count, "knockout.trash");
    require(official.players[1].getActive().cardIndex == 22, "knockout.official_active");
    require(pod.players[1].active.values[0].index == 22, "knockout.pod_active");
    require(pod.error == 0, "knockout.pod_error");
}

void run_copy_enemy_attack_case(
    const OfficialRulePackView& rules,
    OfficialStatePod* fixture_pending,
    OfficialStatePod* fixture_expected) {
    constexpr std::uint64_t seed = 59;
    BattleData battle;
    setup_official(&battle, seed);
    State& official = battle.state;
    prepare_official_case(&official, 615, 39, 1);
    for (int ref = 12; ref <= 13; ++ref) {
        place_official(&official, ref, 1, 0, AreaType::Energy);
        official.getCard(CardRef(ref)).attachMoveCounter =
            official.getCard(CardRef(10)).moveCounter;
    }
    OfficialStatePod pod{};
    prepare_pod_case(official, &pod, 1200);
    SelectedAttack(official, 886, 0, -1);
    drive_official(&official);

    OfficialAttackResult result = official_begin_attack(
        &pod, rules, OfficialCardRefPod{10}, 886);
    require(result == OfficialAttackResult::kNeedsAction, "copy.selection_needed");
    compare_options(official, pod, "copy.selection");
    require(pod.options.count == 2, "copy.option_count");
    require(pod.options.values[0].params[0] == 31, "copy.first_attack_id");
    require(pod.options.values[0].params[1] == 886, "copy.source_attack_id");

    choose_official(&official, {0});
    const std::uint16_t selected = 0;
    if (fixture_pending != nullptr) *fixture_pending = pod;
    const OfficialFlowStatus flow = official_apply_pending_action(
        &pod, rules, &selected, 1);
    require(flow == OfficialFlowStatus::kIdle, "copy.complete");
    if (fixture_expected != nullptr) *fixture_expected = pod;
    require(official.getCard(CardRef(20)).damage == 30, "copy.official_damage");
    require(pod.cards[20].damage == 30, "copy.pod_damage");
    require(official.turnHistories[1].turnAttackId == 886, "copy.official_history");
    require(pod.turn_histories[1].attack_id == 886, "copy.pod_history");
    require(official.turn == pod.turn, "copy.turn");
    require(pod.error == 0, "copy.pod_error");
}

void run_double_attack_case(const OfficialRulePackView& rules) {
    constexpr std::uint64_t seed = 61;
    BattleData battle;
    setup_official(&battle, seed);
    State& official = battle.state;
    prepare_official_case(&official, 93, 24, 1);
    place_official(&official, 22, 24, 0, AreaType::Bench);
    place_official(&official, 50, 1245, 0, AreaType::Stadium);
    OfficialStatePod pod{};
    prepare_pod_case(official, &pod, 1300);
    SelectedAttack(official, 115, 0, -1);
    drive_official(&official);

    OfficialAttackResult result = official_begin_attack(
        &pod, rules, OfficialCardRefPod{10}, 115);
    require(result == OfficialAttackResult::kNeedsAction, "double.selection_needed");
    compare_options(official, pod, "double.selection");
    require(pod.options.count == 1, "double.option_count");
    require(pod.options.values[0].params[0] == 115, "double.attack_id");
    require(official.getCard(CardRef(20)).damage == 20, "double.first_official_damage");
    require(pod.cards[20].damage == 20, "double.first_pod_damage");

    choose_official(&official, {0});
    const std::uint16_t selected = 0;
    const OfficialFlowStatus flow = official_apply_pending_action(
        &pod, rules, &selected, 1);
    require(flow == OfficialFlowStatus::kIdle, "double.complete");
    require(official.getCard(CardRef(20)).damage == 40, "double.official_damage");
    require(pod.cards[20].damage == 40, "double.pod_damage");
    require(official.turn == pod.turn, "double.turn");
    require(pod.error == 0, "double.pod_error");
}

void run_post_effect_selection_case(const OfficialRulePackView& rules) {
    constexpr std::uint64_t seed = 67;
    BattleData battle;
    setup_official(&battle, seed);
    State& official = battle.state;
    prepare_official_case(&official, 22, 24, 6);
    place_official(&official, 22, 38, 1, AreaType::Bench);
    OfficialStatePod pod{};
    prepare_pod_case(official, &pod, 1400);
    SelectedAttack(official, 3, 0, -1);
    drive_official(&official);

    OfficialAttackResult result = official_begin_attack(
        &pod, rules, OfficialCardRefPod{10}, 3);
    require(result == OfficialAttackResult::kNeedsAction, "post_select.selection_needed");
    compare_options(official, pod, "post_select.selection");
    require(official.getCard(CardRef(20)).damage == 20, "post_select.first_damage");
    require(pod.cards[20].damage == 20, "post_select.pod_first_damage");

    choose_official(&official, {0});
    const std::uint16_t selected = 0;
    const OfficialFlowStatus flow = official_apply_pending_action(
        &pod, rules, &selected, 1);
    require(flow == OfficialFlowStatus::kIdle, "post_select.complete");
    require(official.players[1].getActive().cardIndex == 22, "post_select.official_active");
    require(pod.players[1].active.values[0].index == 22, "post_select.pod_active");
    require(official.getCard(CardRef(20)).damage == pod.cards[20].damage, "post_select.damage");
    require(official.turn == pod.turn, "post_select.turn");
    require(pod.error == 0, "post_select.pod_error");
}

void run_deck_top_attack_case(const OfficialRulePackView& rules) {
    constexpr std::uint64_t seed = 71;
    BattleData battle;
    setup_official(&battle, seed);
    State& official = battle.state;
    prepare_official_case(&official, 163, 24, 5);
    place_official(&official, 12, 1, 0, AreaType::Energy);
    official.getCard(CardRef(12)).attachMoveCounter =
        official.getCard(CardRef(10)).moveCounter;
    place_official(&official, 42, 39, 0, AreaType::Deck);
    OfficialStatePod pod{};
    prepare_pod_case(official, &pod, 1500);
    SelectedAttack(official, 213, 0, -1);
    drive_official(&official);

    OfficialAttackResult result = official_begin_attack(
        &pod, rules, OfficialCardRefPod{10}, 213);
    require(result == OfficialAttackResult::kNeedsAction, "deck_top.selection_needed");
    compare_options(official, pod, "deck_top.selection");
    require(pod.options.count == 2, "deck_top.option_count");
    require(pod.players[0].trash.count == 1, "deck_top.pod_trash");

    choose_official(&official, {0});
    const std::uint16_t selected = 0;
    const OfficialFlowStatus flow = official_apply_pending_action(
        &pod, rules, &selected, 1);
    require(flow == OfficialFlowStatus::kIdle, "deck_top.complete");
    require(official.getCard(CardRef(20)).damage == 30, "deck_top.official_damage");
    require(pod.cards[20].damage == 30, "deck_top.pod_damage");
    require(official.players[0].trash.size() == pod.players[0].trash.count, "deck_top.trash");
    require(official.turnHistories[1].turnAttackId == 213, "deck_top.official_history");
    require(pod.turn_histories[1].attack_id == 213, "deck_top.pod_history");
    require(official.turn == pod.turn, "deck_top.turn");
    require(pod.error == 0, "deck_top.pod_error");
}

void run_deck_top_supporter_case(const OfficialRulePackView& rules) {
    constexpr std::uint64_t seed = 73;
    BattleData battle;
    setup_official(&battle, seed);
    State& official = battle.state;
    prepare_official_case(&official, 660, 24, 1);
    for (int ref = 42; ref <= 44; ++ref) {
        place_official(&official, ref, 1, 0, AreaType::Deck);
    }
    place_official(&official, 60, 1224, 0, AreaType::Deck);
    OfficialStatePod pod{};
    prepare_pod_case(official, &pod, 1600);
    SelectedAttack(official, 955, 0, -1);
    drive_official(&official);

    const OfficialAttackResult result = official_begin_attack(
        &pod, rules, OfficialCardRefPod{10}, 955);
    require(result == OfficialAttackResult::kComplete, "deck_supporter.complete");
    require(official.players[0].hand.size() == 3, "deck_supporter.official_hand");
    require(pod.players[0].hand.count == 3, "deck_supporter.pod_hand");
    require(official.players[0].trash.size() == 1, "deck_supporter.official_trash");
    require(pod.players[0].trash.count == 1, "deck_supporter.pod_trash");
    require(official.players[0].deck.size() == pod.players[0].deck.count, "deck_supporter.deck");
    require(official.turn == pod.turn, "deck_supporter.turn");
    require(pod.error == 0, "deck_supporter.pod_error");
}

void run_enemy_deck_top10_case(const OfficialRulePackView& rules) {
    constexpr std::uint64_t seed = 79;
    BattleData battle;
    setup_official(&battle, seed);
    State& official = battle.state;
    prepare_official_case(&official, 471, 24, 1);
    place_official(&official, 12, 1, 0, AreaType::Energy);
    official.getCard(CardRef(12)).attachMoveCounter =
        official.getCard(CardRef(10)).moveCounter;
    for (int ref = 50; ref < 60; ++ref) {
        place_official(
            &official, ref, (ref % 2 == 0) ? 39 : 1, 1, AreaType::Deck);
    }
    OfficialStatePod pod{};
    prepare_pod_case(official, &pod, 1700);
    SelectedAttack(official, 665, 0, -1);
    drive_official(&official);

    OfficialAttackResult result = official_begin_attack(
        &pod, rules, OfficialCardRefPod{10}, 665);
    require(result == OfficialAttackResult::kNeedsAction, "enemy_top10.selection_needed");
    compare_options(official, pod, "enemy_top10.selection");
    require(pod.options.count == 2, "enemy_top10.option_count");
    require(official.players[1].deck.size() == pod.players[1].deck.count, "enemy_top10.deck_count");
    for (std::size_t index = 0; index < official.players[1].deck.size(); ++index) {
        require(
            official.players[1].deck[index].cardIndex == pod.players[1].deck.values[index].index,
            "enemy_top10.deck_order");
    }
    compare_rng(battle.game.rng, pod.rng);

    choose_official(&official, {0});
    const std::uint16_t selected = 0;
    const OfficialFlowStatus flow = official_apply_pending_action(
        &pod, rules, &selected, 1);
    require(flow == OfficialFlowStatus::kIdle, "enemy_top10.complete");
    require(official.getCard(CardRef(20)).damage == 30, "enemy_top10.official_damage");
    require(pod.cards[20].damage == 30, "enemy_top10.pod_damage");
    require(official.turnHistories[1].turnAttackId == 665, "enemy_top10.official_history");
    require(pod.turn_histories[1].attack_id == 665, "enemy_top10.pod_history");
    require(official.turn == pod.turn, "enemy_top10.turn");
    compare_rng(battle.game.rng, pod.rng);
    require(pod.error == 0, "enemy_top10.pod_error");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            throw std::runtime_error(
                "usage: official_attack_oracle <rule-pack> [resume-fixture]");
        }
        InitializeAll();
        const std::vector<std::uint8_t> bytes = read_binary(argv[1]);
        const OfficialRulePackView rules = make_official_rule_pack_view(bytes.data());
        run_damage_case(rules, 43, 39, 24, 31, 1, 30, "simple_damage");
        run_damage_case(rules, 47, 54, 38, 57, 5, 40, "pre_effect_damage_change");
        run_confusion_case(rules, 0, true, "confusion_head");
        run_confusion_case(rules, 1, false, "confusion_tail");
        run_no_damage_coin_case(rules, 0, 39, 31, 0, "no_damage_coin_head");
        run_no_damage_coin_case(rules, 1, 39, 31, 30, "no_damage_coin_tail");
        run_no_damage_coin_case(rules, 0, 625, 901, 40, "no_target_bypass");
        run_knockout_case(rules);
        OfficialStatePod fixture_pending{};
        OfficialStatePod fixture_expected{};
        run_copy_enemy_attack_case(
            rules, &fixture_pending, &fixture_expected);
        run_double_attack_case(rules);
        run_post_effect_selection_case(rules);
        run_deck_top_attack_case(rules);
        run_deck_top_supporter_case(rules);
        run_enemy_deck_top10_case(rules);
        if (argc == 3) {
            OfficialFlowFixtureHeader header{};
            std::memcpy(header.magic, "PTCGFLW1", 8);
            header.action_count = 1;
            header.option_indices[0] = 0;
            std::ofstream fixture(argv[2], std::ios::binary | std::ios::trunc);
            if (!fixture) {
                throw std::runtime_error("cannot create resume fixture");
            }
            fixture.write(
                reinterpret_cast<const char*>(&header), sizeof(header));
            fixture.write(
                reinterpret_cast<const char*>(&fixture_pending),
                sizeof(fixture_pending));
            fixture.write(
                reinterpret_cast<const char*>(&fixture_expected),
                sizeof(fixture_expected));
            if (!fixture) {
                throw std::runtime_error("cannot write resume fixture");
            }
        }
        std::cout
            << "{\"passed\":true,\"scenarios\":14,\"checks\":120,"
               "\"scope\":[\"simple_damage\",\"pre_effect_damage_change\","
               "\"confusion_head\",\"confusion_tail\",\"no_damage_coin_head\","
               "\"no_damage_coin_tail\",\"no_target_bypass\","
               "\"knockout_prize_replacement\",\"copy_enemy_attack\","
               "\"double_attack\",\"post_effect_selection\","
               "\"deck_top_attack\",\"deck_top_supporter\","
               "\"enemy_deck_top10_attack\"]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
