#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "All.h"
#include "official_state_bridge.h"
#include "ptcg_cuda/official_flow_dispatch_pod.cuh"
#include "ptcg_cuda/official_main_fixture.h"

namespace {

using namespace ptcg::cuda_engine;
using namespace ptcg::cuda_engine::extractor;

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open rule pack: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void require(bool condition, const std::string& label) {
    if (!condition) throw std::runtime_error("official main oracle failed: " + label);
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
        ps.koPrizeOnceChanged = false;
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
    state->turnActionCount = 0;
    state->turnState = 0;
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

void setup_official(BattleData* battle, std::uint32_t seed) {
    GameConfig config{};
    config.seed = seed;
    config.recordLog = false;
    config.deviceRand = false;
    for (int player = 0; player < 2; ++player) {
        for (int index = 0; index < DECK_SIZE; ++index) {
            config.decks[player].cards[index] = 7;
        }
    }
    battle->init(config, false);
    battle->game.rng = std::mt19937(seed);
    clear_official(&battle->state);
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
        default: throw std::runtime_error("unsupported official area");
    }
}

void add_pod(
    OfficialStatePod* state,
    std::uint16_t ref,
    std::int32_t card_id,
    std::int32_t player,
    OfficialArea area) {
    OfficialCardStatePod& card = state->cards[ref];
    card = {};
    card.card_id = card_id;
    card.player = static_cast<std::int8_t>(player);
    card.area = static_cast<std::uint8_t>(area);
    card.move_counter = state->move_counter++;
    require(
        official_pod_push_zone_card(state, player, area, OfficialCardRefPod{ref}),
        "pod add card");
}

void prepare_case(
    BattleData* battle,
    OfficialStatePod* pod,
    std::uint32_t seed,
    std::uint64_t episode) {
    try {
        setup_official(battle, seed);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("official battle init: ") + error.what());
    }
    State& official = battle->state;
    try {
        place_official(&official, 10, 169, 0, AreaType::Active);
        place_official(&official, 11, 8, 0, AreaType::Energy);
        official.getCard(CardRef(11)).attachMoveCounter =
            official.getCard(CardRef(10)).moveCounter;
        place_official(&official, 20, 24, 1, AreaType::Active);
        for (int ref = 21; ref <= 25; ++ref) {
            place_official(&official, ref, 39, 1, AreaType::Bench);
        }
        place_official(&official, 30, 1, 0, AreaType::Prize);
        place_official(&official, 31, 1, 1, AreaType::Prize);
    place_official(&official, 40, 1, 0, AreaType::Deck);
        place_official(&official, 41, 39, 1, AreaType::Deck);
        place_official(&official, 42, 39, 0, AreaType::Hand);
        place_official(&official, 43, 1243, 0, AreaType::Hand);
        place_official(&official, 44, 1140, 0, AreaType::Hand);
        place_official(&official, 45, 1211, 0, AreaType::Hand);
        place_official(&official, 46, 1100, 0, AreaType::Hand);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("official card placement: ") + error.what());
    }

    const OfficialBridgeResult bridge = bridge_official_state(
        official,
        OfficialStateBridgeContext{episode, 0, true},
        pod);
    require(
        static_cast<bool>(bridge),
        "initial state bridge error="
            + std::to_string(static_cast<std::int32_t>(bridge.error))
            + ".detail=" + std::to_string(bridge.detail));
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

template <typename OfficialList, std::size_t Capacity>
void compare_zone(
    const OfficialList& official,
    const OfficialPodList<OfficialCardRefPod, Capacity>& pod,
    const std::string& label) {
    require(official.size() == pod.count, label + ".count");
    for (std::size_t index = 0; index < official.size(); ++index) {
        require(
            official[index].cardIndex == pod.values[index].index,
            label + ".ref");
    }
}

void compare_options(
    const State& official,
    const OfficialStatePod& pod,
    const std::string& label) {
    std::string official_types;
    for (const SelectOption& option : official.options) {
        if (!official_types.empty()) official_types += ',';
        official_types += std::to_string(static_cast<int>(option.type));
    }
    std::string pod_types;
    for (std::uint16_t index = 0; index < pod.options.count; ++index) {
        if (!pod_types.empty()) pod_types += ',';
        pod_types += std::to_string(pod.options.values[index].type);
    }
    require(
        official.options.size() == pod.options.count,
        label + ".option_count.official="
            + std::to_string(official.options.size())
            + ".pod=" + std::to_string(pod.options.count)
            + ".official_types=" + official_types
            + ".pod_types=" + pod_types
            + ".official_deck=" + std::to_string(official.players[0].deck.size())
            + ".pod_deck=" + std::to_string(pod.players[0].deck.count)
            + ".official_trash=" + std::to_string(official.players[0].trash.size())
            + ".pod_trash=" + std::to_string(pod.players[0].trash.count)
            + ".official_looking=" + std::to_string(official.looking.size())
            + ".pod_looking=" + std::to_string(pod.looking.count));
    require(static_cast<int>(official.selectType) == pod.select_type, label + ".select_type");
    require(
        static_cast<int>(official.selectContext) == pod.select_context,
        label + ".select_context");
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
        require(expected.param4 == actual.params[4], label + ".option_param4");
    }
}

void compare_state(
    const BattleData& battle,
    const OfficialStatePod& pod,
    const std::string& label) {
    const State& official = battle.state;
    require(official.turn == pod.turn, label + ".turn");
    require(
        official.turnActionCount == pod.turn_action_count,
        label + ".turn_actions.official="
            + std::to_string(official.turnActionCount)
            + ".pod=" + std::to_string(pod.turn_action_count)
            + ".effect_index="
            + std::to_string(pod.effect_interpreter.effect_index)
            + ".effect_count="
            + std::to_string(pod.effect_interpreter.effect_count)
            + ".effect_active="
            + std::to_string(pod.effect_interpreter.active)
            + ".flow_flags=" + std::to_string(pod.flow_flags));
    require(static_cast<int>(official.phase) == pod.phase, label + ".phase");
    require(static_cast<int>(official.gameResult) == pod.game_result, label + ".result");
    require(static_cast<int>(official.finishReason) == pod.finish_reason, label + ".reason");
    require(official.turnState == pod.turn_state, label + ".turn_state");
    compare_options(official, pod, label);
    compare_zone(official.turnPlay, pod.turn_play, label + ".turn_play");
    compare_zone(official.stadium, pod.stadium, label + ".stadium");
    compare_zone(official.looking, pod.looking, label + ".looking");
    compare_zone(official.selectedList, pod.selected_list, label + ".selected_list");
    compare_zone(official.eachList, pod.each_list, label + ".each_list");
    compare_zone(official.playing, pod.playing, label + ".playing");
    compare_zone(official.checkList, pod.check_list, label + ".check_list");
    require(
        official.turnUsedSkill.size() == pod.turn_used_skills.count,
        label + ".turn_used_skills.count");
    for (std::size_t index = 0; index < official.turnUsedSkill.size(); ++index) {
        require(
            official.turnUsedSkill[index] == pod.turn_used_skills.values[index],
            label + ".turn_used_skills.value");
    }
    compare_zone(official.players[0].preEvolution, pod.players[0].pre_evolution,
        label + ".p0.pre_evolution");
    compare_zone(official.players[1].preEvolution, pod.players[1].pre_evolution,
        label + ".p1.pre_evolution");
    require(
        official.turnEvolve.size() == pod.turn_evolve.count,
        label + ".turn_evolve.count");
    for (std::size_t index = 0; index < official.turnEvolve.size(); ++index) {
        require(
            official.turnEvolve[index].preRef.cardIndex
                == pod.turn_evolve.values[index].from.index,
            label + ".turn_evolve.from");
        require(
            official.turnEvolve[index].ref.cardIndex
                == pod.turn_evolve.values[index].to.index,
            label + ".turn_evolve.to");
    }
    for (int player = 0; player < 2; ++player) {
        const PlayerState& expected = official.players[player];
        const OfficialPlayerStatePod& actual = pod.players[player];
        const std::string prefix = label + ".p" + std::to_string(player);
        compare_zone(expected.active, actual.active, prefix + ".active");
        compare_zone(expected.bench, actual.bench, prefix + ".bench");
        compare_zone(expected.prize, actual.prize, prefix + ".prize");
        compare_zone(expected.hand, actual.hand, prefix + ".hand");
        compare_zone(expected.deck, actual.deck, prefix + ".deck");
        compare_zone(expected.trash, actual.trash, prefix + ".trash");
        compare_zone(expected.energy, actual.energy, prefix + ".energy");
        compare_zone(expected.tool, actual.tool, prefix + ".tool");
        require(expected.thisTurn.value == actual.this_turn, prefix + ".this_turn");
        require(expected.nextTurn.value == actual.next_turn, prefix + ".next_turn");
        require(expected.activeState == actual.active_state, prefix + ".active_state");
    }
    for (int ref : {
            10, 11, 20, 21, 22, 23, 24, 25, 30, 31, 40, 41, 42, 43, 44, 45, 46}) {
        const Card& expected = official.getCard(CardRef(ref));
        const OfficialCardStatePod& actual = pod.cards[ref];
        const std::string prefix = label + ".card" + std::to_string(ref);
        require(static_cast<int>(expected.cardId) == actual.card_id, prefix + ".id");
        require(expected.playerIndex == actual.player, prefix + ".player");
        require(static_cast<int>(expected.area) == actual.area, prefix + ".area");
        require(expected.moveCounter == actual.move_counter, prefix + ".move");
        require(expected.attachMoveCounter == actual.attach_move_counter, prefix + ".attach");
        require(expected.damage == actual.damage, prefix + ".damage");
        require(expected.abilityUsed.size() == actual.ability_used_count,
            prefix + ".ability_used.count");
        for (std::size_t index = 0; index < expected.abilityUsed.size(); ++index) {
            require(expected.abilityUsed[index] == actual.ability_used[index],
                prefix + ".ability_used.value");
        }
    }
    for (int ref = 47; ref <= 52; ++ref) {
        const Card& optional_expected = official.getCard(CardRef(ref));
        const OfficialCardStatePod& optional_actual = pod.cards[ref];
        if (optional_actual.card_id == 0) continue;
        const std::string prefix = label + ".card" + std::to_string(ref);
        require(static_cast<int>(optional_expected.cardId) == optional_actual.card_id,
            prefix + ".id");
        require(optional_expected.playerIndex == optional_actual.player,
            prefix + ".player");
        require(static_cast<int>(optional_expected.area) == optional_actual.area,
            prefix + ".area");
        require(optional_expected.moveCounter == optional_actual.move_counter,
            prefix + ".move");
        require(optional_expected.attachMoveCounter == optional_actual.attach_move_counter,
            prefix + ".attach");
        require(optional_expected.damage == optional_actual.damage,
            prefix + ".damage");
        require(optional_expected.abilityUsed.size() == optional_actual.ability_used_count,
            prefix + ".ability_used.count");
    }
    compare_rng(battle.game.rng, pod.rng);
    require(pod.error == 0, label + ".pod_error");
}

int find_option(const State& state, SelectOptionType type, int param0 = -1) {
    for (std::size_t index = 0; index < state.options.size(); ++index) {
        const SelectOption& option = state.options[index];
        if (option.type == type && (param0 < 0 || option.param0 == param0)) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

void prepare_official_decision(State* state) {
    try {
        state->pushFunction(MainSelect);
        require(state->step(), "official main decision");
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("official initial MainSelect: ") + error.what());
    }
}

void apply_official_action(State* state, int option_index) {
    try {
        state->selected.clear();
        state->selected.push_back(option_index);
        require(state->step(), "official next decision");
        // ApiSelect automatically consumes internal zero-card boundaries.
        while (!state->isFinish() && state->selectMax == 0) {
            state->selected.clear();
            require(state->step(), "official automatic selection");
        }
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("official selected Main action: ") + error.what());
    }
}

void apply_official_selection(
    State* state,
    const std::vector<int>& option_indices) {
    try {
        state->selected.clear();
        for (int option_index : option_indices) {
            state->selected.push_back(option_index);
        }
        require(state->step(), "official next selection");
        // Keep fixture generation on the same contract as ApiSelect, which
        // repeatedly advances through selectMax==0 internal boundaries.
        while (!state->isFinish() && state->selectMax == 0) {
            state->selected.clear();
            require(state->step(), "official automatic selection");
        }
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string("official selected effect action: ") + error.what());
    }
}

OfficialMainFixtureRecord make_advance_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    try {
        prepare_case(&battle, &input, 17, 1701);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("case setup: ") + error.what());
    }
    OfficialStatePod expected = input;
    prepare_official_decision(&battle.state);
    OfficialFlowStatus status = OfficialFlowStatus::kError;
    try {
        status = official_advance_idle_state_to_decision(&expected, rules);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("POD MainSelect: ") + error.what());
    }
    require(status == OfficialFlowStatus::kNeedsAction, "prepare pod status");
    try {
        compare_state(battle, expected, "prepare");
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("state compare: ") + error.what());
    }

    OfficialMainFixtureRecord record{};
    record.scenario_id = 1;
    record.operation = OfficialMainFixtureOperation::kAdvance;
    record.expected_status = status;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_action_record(
    const OfficialRulePackView& rules,
    SelectOptionType type,
    int param0,
    std::uint32_t seed,
    std::uint64_t episode,
    std::uint32_t scenario_id,
    const std::string& label) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, seed, episode);
    prepare_official_decision(&battle.state);
    require(
        official_advance_idle_state_to_decision(&input, rules)
            == OfficialFlowStatus::kNeedsAction,
        label + ".prepare");
    compare_state(battle, input, label + ".before");
    const int option_index = find_option(battle.state, type, param0);
    require(option_index >= 0, label + ".option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(option_index);
    apply_official_action(&battle.state, option_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, label + ".pod status");
    compare_state(battle, expected, label + ".after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = scenario_id;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_item_resume_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, 47, 4701);
    prepare_official_decision(&battle.state);
    require(
        official_advance_idle_state_to_decision(&input, rules)
            == OfficialFlowStatus::kNeedsAction,
        "item_resume.prepare");
    const int play_index = find_option(
        battle.state, SelectOptionType::Play, 4);
    require(play_index >= 0, "item_resume.play_option");
    apply_official_action(&battle.state, play_index);
    const std::uint16_t play_selected = static_cast<std::uint16_t>(play_index);
    require(
        official_apply_pending_action(
            &input, rules, &play_selected, 1)
            == OfficialFlowStatus::kNeedsAction,
        "item_resume.pending");
    compare_state(battle, input, "item_resume.before");
    require(input.options.count > 0, "item_resume.effect_option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = 0;
    apply_official_action(&battle.state, selected);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(
        status == OfficialFlowStatus::kNeedsAction,
        "item_resume.status="
            + std::to_string(static_cast<int>(status))
            + ".error=" + std::to_string(expected.error)
            + ".detail=" + std::to_string(expected.error_detail)
            + ".effect_active="
            + std::to_string(expected.effect_interpreter.active)
            + ".awaiting="
            + std::to_string(expected.effect_interpreter.awaiting_selection)
            + ".control_flags=" + std::to_string(expected.control_flags)
            + ".select_type=" + std::to_string(expected.select_type));
    compare_state(battle, expected, "item_resume.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 9;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_attach_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, 47, 4701);
    prepare_official_decision(&battle.state);
    require(
        official_advance_idle_state_to_decision(&input, rules)
            == OfficialFlowStatus::kNeedsAction,
        "attach.prepare");
    const int play_index = find_option(
        battle.state, SelectOptionType::Play, 4);
    require(play_index >= 0, "attach.play_option");
    apply_official_action(&battle.state, play_index);
    const std::uint16_t play_selected = static_cast<std::uint16_t>(play_index);
    require(
        official_apply_pending_action(
            &input, rules, &play_selected, 1)
            == OfficialFlowStatus::kNeedsAction,
        "attach.item_pending");
    const std::uint16_t effect_selected = 0;
    apply_official_action(&battle.state, effect_selected);
    require(
        official_apply_pending_action(
            &input, rules, &effect_selected, 1)
            == OfficialFlowStatus::kNeedsAction,
        "attach.effect_pending");
    compare_state(battle, input, "attach.before");

    const int attach_index = find_option(
        battle.state,
        SelectOptionType::Attach,
        static_cast<int>(AreaType::Hand));
    require(attach_index >= 0, "attach.option");
    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(attach_index);
    apply_official_action(&battle.state, attach_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "attach.status");
    compare_state(battle, expected, "attach.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 10;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_evolve_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, 59, 5901);
    place_official(&battle.state, 47, 170, 0, AreaType::Hand);
    add_pod(&input, 47, 170, 0, OfficialArea::kHand);
    prepare_official_decision(&battle.state);
    require(
        official_advance_idle_state_to_decision(&input, rules)
            == OfficialFlowStatus::kNeedsAction,
        "evolve.prepare");
    compare_state(battle, input, "evolve.before");
    const int evolve_index = find_option(
        battle.state,
        SelectOptionType::Evolve,
        static_cast<int>(AreaType::Hand));
    require(evolve_index >= 0, "evolve.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(evolve_index);
    apply_official_action(&battle.state, evolve_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "evolve.status");
    compare_state(battle, expected, "evolve.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 11;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_ability_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, 61, 6101);
    battle.state.getCard(CardRef(10)).cardId = 351;
    input.cards[10].card_id = 351;
    prepare_official_decision(&battle.state);
    require(
        official_advance_idle_state_to_decision(&input, rules)
            == OfficialFlowStatus::kNeedsAction,
        "ability.prepare");
    compare_state(battle, input, "ability.before");
    const int ability_index = find_option(
        battle.state,
        SelectOptionType::Ability,
        static_cast<int>(AreaType::Active));
    require(ability_index >= 0, "ability.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(ability_index);
    apply_official_action(&battle.state, ability_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "ability.status");
    require(
        find_option(battle.state, SelectOptionType::Ability) < 0,
        "ability.once_turn_consumed");
    compare_state(battle, expected, "ability.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 12;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_discard_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, 67, 6701);
    place_official(&battle.state, 47, 1099, 0, AreaType::Bench);
    add_pod(&input, 47, 1099, 0, OfficialArea::kBench);
    battle.state.getCard(CardRef(11)).attachMoveCounter =
        battle.state.getCard(CardRef(47)).moveCounter;
    input.cards[11].attach_move_counter = input.cards[47].move_counter;
    prepare_official_decision(&battle.state);
    require(
        official_advance_idle_state_to_decision(&input, rules)
            == OfficialFlowStatus::kNeedsAction,
        "discard.prepare");
    compare_state(battle, input, "discard.before");
    const int discard_index = find_option(
        battle.state,
        SelectOptionType::Discard,
        static_cast<int>(AreaType::Bench));
    require(discard_index >= 0, "discard.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(discard_index);
    apply_official_action(&battle.state, discard_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "discard.status");
    compare_state(battle, expected, "discard.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 13;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

void prepare_retreat_main_decision(
    BattleData* battle,
    OfficialStatePod* pod,
    const OfficialRulePackView& rules) {
    prepare_case(battle, pod, 71, 7101);
    battle->state.getCard(CardRef(10)).cardId = 39;
    pod->cards[10].card_id = 39;
    place_official(&battle->state, 47, 39, 0, AreaType::Bench);
    add_pod(pod, 47, 39, 0, OfficialArea::kBench);
    prepare_official_decision(&battle->state);
    require(
        official_advance_idle_state_to_decision(pod, rules)
            == OfficialFlowStatus::kNeedsAction,
        "retreat.prepare");
    compare_state(*battle, *pod, "retreat.main");
}

void advance_retreat_to_energy_decision(
    BattleData* battle,
    OfficialStatePod* pod,
    const OfficialRulePackView& rules) {
    const int retreat_index = find_option(
        battle->state, SelectOptionType::Retreat);
    require(retreat_index >= 0, "retreat.option");
    const std::uint16_t selected = static_cast<std::uint16_t>(retreat_index);
    apply_official_action(&battle->state, retreat_index);
    require(
        official_apply_pending_action(pod, rules, &selected, 1)
            == OfficialFlowStatus::kNeedsAction,
        "retreat.energy.status");
    compare_state(*battle, *pod, "retreat.energy");
}

void advance_retreat_to_switch_decision(
    BattleData* battle,
    OfficialStatePod* pod,
    const OfficialRulePackView& rules) {
    const int energy_index = find_option(
        battle->state, SelectOptionType::Energy);
    require(energy_index >= 0, "retreat.energy.option");
    const std::uint16_t selected = static_cast<std::uint16_t>(energy_index);
    apply_official_action(&battle->state, energy_index);
    require(
        official_apply_pending_action(pod, rules, &selected, 1)
            == OfficialFlowStatus::kNeedsAction,
        "retreat.switch.status");
    compare_state(*battle, *pod, "retreat.switch");
}

OfficialMainFixtureRecord make_retreat_energy_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_retreat_main_decision(&battle, &input, rules);
    const int retreat_index = find_option(battle.state, SelectOptionType::Retreat);
    require(retreat_index >= 0, "retreat_energy.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(retreat_index);
    apply_official_action(&battle.state, retreat_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "retreat_energy.status");
    compare_state(battle, expected, "retreat_energy.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 14;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_retreat_switch_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_retreat_main_decision(&battle, &input, rules);
    advance_retreat_to_energy_decision(&battle, &input, rules);
    const int energy_index = find_option(battle.state, SelectOptionType::Energy);
    require(energy_index >= 0, "retreat_switch.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(energy_index);
    apply_official_action(&battle.state, energy_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "retreat_switch.status");
    compare_state(battle, expected, "retreat_switch.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 15;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_retreat_complete_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_retreat_main_decision(&battle, &input, rules);
    advance_retreat_to_energy_decision(&battle, &input, rules);
    advance_retreat_to_switch_decision(&battle, &input, rules);
    const int switch_index = find_option(
        battle.state,
        SelectOptionType::Card,
        static_cast<int>(AreaType::Bench));
    require(switch_index >= 0, "retreat_complete.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(switch_index);
    apply_official_action(&battle.state, switch_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "retreat_complete.status");
    compare_state(battle, expected, "retreat_complete.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 16;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_tool_attach_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, 73, 7301);
    place_official(&battle.state, 47, 1154, 0, AreaType::Hand);
    add_pod(&input, 47, 1154, 0, OfficialArea::kHand);
    prepare_official_decision(&battle.state);
    require(
        official_advance_idle_state_to_decision(&input, rules)
            == OfficialFlowStatus::kNeedsAction,
        "tool_attach.prepare");
    compare_state(battle, input, "tool_attach.before");
    const int attach_index = find_option(
        battle.state,
        SelectOptionType::Attach,
        static_cast<int>(AreaType::Hand));
    require(attach_index >= 0, "tool_attach.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(attach_index);
    apply_official_action(&battle.state, attach_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "tool_attach.status");
    compare_state(battle, expected, "tool_attach.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 17;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_retreat_zero_cost_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, 79, 7901);
    battle.state.getCard(CardRef(10)).cardId = 39;
    input.cards[10].card_id = 39;
    place_official(&battle.state, 47, 39, 0, AreaType::Bench);
    add_pod(&input, 47, 39, 0, OfficialArea::kBench);
    battle.state.getCard(CardRef(10)).thisTurn.retreatCostChange = -1;
    input.cards[10].this_turn[2] |= 0xffU << 24U;
    prepare_official_decision(&battle.state);
    require(
        official_advance_idle_state_to_decision(&input, rules)
            == OfficialFlowStatus::kNeedsAction,
        "retreat_zero.prepare");
    compare_state(battle, input, "retreat_zero.before");
    const int retreat_index = find_option(battle.state, SelectOptionType::Retreat);
    require(retreat_index >= 0, "retreat_zero.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(retreat_index);
    apply_official_action(&battle.state, retreat_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "retreat_zero.status");
    require(
        battle.state.selectType == SelectType::Card,
        "retreat_zero.switch_decision");
    compare_state(battle, expected, "retreat_zero.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 18;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

void prepare_multi_energy_retreat(
    BattleData* battle,
    OfficialStatePod* pod,
    const OfficialRulePackView& rules) {
    prepare_case(battle, pod, 83, 8301);
    place_official(&battle->state, 47, 39, 0, AreaType::Bench);
    add_pod(pod, 47, 39, 0, OfficialArea::kBench);
    place_official(&battle->state, 48, 8, 0, AreaType::Energy);
    add_pod(pod, 48, 8, 0, OfficialArea::kEnergy);
    battle->state.getCard(CardRef(48)).attachMoveCounter =
        battle->state.getCard(CardRef(10)).moveCounter;
    pod->cards[48].attach_move_counter = pod->cards[10].move_counter;
    prepare_official_decision(&battle->state);
    require(
        official_advance_idle_state_to_decision(pod, rules)
            == OfficialFlowStatus::kNeedsAction,
        "retreat_multi.prepare");
    compare_state(*battle, *pod, "retreat_multi.main");
    advance_retreat_to_energy_decision(battle, pod, rules);
}

void advance_multi_energy_once(
    BattleData* battle,
    OfficialStatePod* pod,
    const OfficialRulePackView& rules) {
    const int energy_index = find_option(
        battle->state, SelectOptionType::Energy);
    require(energy_index >= 0, "retreat_multi.energy.option");
    const std::uint16_t selected = static_cast<std::uint16_t>(energy_index);
    apply_official_action(&battle->state, energy_index);
    require(
        official_apply_pending_action(pod, rules, &selected, 1)
            == OfficialFlowStatus::kNeedsAction,
        "retreat_multi.energy.status");
    compare_state(*battle, *pod, "retreat_multi.energy");
}

OfficialMainFixtureRecord make_retreat_multi_energy_record(
    const OfficialRulePackView& rules,
    bool final_payment) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_multi_energy_retreat(&battle, &input, rules);
    if (final_payment) advance_multi_energy_once(&battle, &input, rules);
    const int energy_index = find_option(battle.state, SelectOptionType::Energy);
    require(energy_index >= 0, "retreat_multi.record.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(energy_index);
    apply_official_action(&battle.state, energy_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "retreat_multi.record.status");
    require(
        battle.state.selectType
            == (final_payment ? SelectType::Card : SelectType::Energy),
        "retreat_multi.record.next_select");
    compare_state(battle, expected, "retreat_multi.record.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = final_payment ? 20 : 19;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

enum class RetreatBlockedCase : std::uint8_t {
    kAlreadyRetreated,
    kAsleep,
    kCannotRetreatThisTurn,
    kCannotRetreatContinual,
    kPoisonRestriction,
    kPokemonItem,
    kInsufficientEnergy,
};

OfficialMainFixtureRecord make_retreat_blocked_record(
    const OfficialRulePackView& rules,
    std::uint32_t scenario_id,
    RetreatBlockedCase blocked_case) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, 89 + scenario_id, 8900 + scenario_id);
    battle.state.getCard(CardRef(10)).cardId = 39;
    input.cards[10].card_id = 39;
    place_official(&battle.state, 47, 39, 0, AreaType::Bench);
    add_pod(&input, 47, 39, 0, OfficialArea::kBench);
    switch (blocked_case) {
        case RetreatBlockedCase::kAlreadyRetreated:
            battle.state.retreated = true;
            input.turn_state |= 1U << 3U;
            break;
        case RetreatBlockedCase::kAsleep:
            battle.state.players[0].badStatus = BadStatusType::Asleep;
            official_pod_set_bad_status(
                &input.players[0], OfficialBadStatus::kAsleep);
            break;
        case RetreatBlockedCase::kCannotRetreatThisTurn:
            battle.state.getCard(CardRef(10)).thisTurn.cannotRetreat = true;
            input.cards[10].this_turn[3] |= 1U;
            break;
        case RetreatBlockedCase::kCannotRetreatContinual:
            battle.state.getCard(CardRef(10)).cannotRetreat = true;
            official_continual_set_flag(&input.cards[10], 23);
            break;
        case RetreatBlockedCase::kPoisonRestriction:
            battle.state.players[0].thisTurn.cannotRetreatPoison = true;
            battle.state.players[0].poisonDamageCounter = 1;
            input.players[0].this_turn |= 1U << 22U;
            official_pod_set_poison_counter(&input.players[0], 1);
            break;
        case RetreatBlockedCase::kPokemonItem:
            battle.state.getCard(CardRef(10)).cardId = 1099;
            input.cards[10].card_id = 1099;
            break;
        case RetreatBlockedCase::kInsufficientEnergy:
            battle.state.getCard(CardRef(10)).cardId = 169;
            input.cards[10].card_id = 169;
            break;
    }

    OfficialStatePod expected = input;
    prepare_official_decision(&battle.state);
    const OfficialFlowStatus status = official_advance_idle_state_to_decision(
        &expected, rules);
    require(status == OfficialFlowStatus::kNeedsAction, "retreat_blocked.status");
    require(
        find_option(battle.state, SelectOptionType::Retreat) < 0,
        "retreat_blocked.option_absent");
    compare_state(
        battle,
        expected,
        "retreat_blocked_" + std::to_string(scenario_id) + ".after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = scenario_id;
    record.operation = OfficialMainFixtureOperation::kAdvance;
    record.expected_status = status;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_rainbow_dna_evolve_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, 127, 12701);
    battle.state.getCard(CardRef(10)).cardId = 249;
    input.cards[10].card_id = 249;
    battle.state.getCard(CardRef(10)).rainbowDna = true;
    official_continual_set_flag(
        &input.cards[10], kOfficialCardRainbowDnaBit);
    place_official(&battle.state, 47, 248, 0, AreaType::Hand);
    add_pod(&input, 47, 248, 0, OfficialArea::kHand);
    prepare_official_decision(&battle.state);
    require(
        official_advance_idle_state_to_decision(&input, rules)
            == OfficialFlowStatus::kNeedsAction,
        "rainbow_dna.prepare");
    compare_state(battle, input, "rainbow_dna.before");
    const int evolve_index = find_option(
        battle.state,
        SelectOptionType::Evolve,
        static_cast<int>(AreaType::Hand));
    require(evolve_index >= 0, "rainbow_dna.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(evolve_index);
    apply_official_action(&battle.state, evolve_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(status == OfficialFlowStatus::kNeedsAction, "rainbow_dna.status");
    compare_state(battle, expected, "rainbow_dna.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 28;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

void prepare_ability_selection(
    BattleData* battle,
    OfficialStatePod* pod,
    const OfficialRulePackView& rules) {
    prepare_case(battle, pod, 131, 13101);
    battle->state.getCard(CardRef(10)).cardId = 36;
    pod->cards[10].card_id = 36;
    prepare_official_decision(&battle->state);
    require(
        official_advance_idle_state_to_decision(pod, rules)
            == OfficialFlowStatus::kNeedsAction,
        "ability_selection.prepare");
    compare_state(*battle, *pod, "ability_selection.main");
}

void advance_ability_to_selection(
    BattleData* battle,
    OfficialStatePod* pod,
    const OfficialRulePackView& rules) {
    const int ability_index = find_option(
        battle->state,
        SelectOptionType::Ability,
        static_cast<int>(AreaType::Active));
    require(ability_index >= 0, "ability_selection.option");
    const std::uint16_t selected = static_cast<std::uint16_t>(ability_index);
    apply_official_action(&battle->state, ability_index);
    require(
        official_apply_pending_action(pod, rules, &selected, 1)
            == OfficialFlowStatus::kNeedsAction,
        "ability_selection.activation_status");
    require(
        battle->state.selectType == SelectType::Card
            && battle->state.selectMin == 0
            && battle->state.selectMax == 1,
        "ability_selection.optional_card_decision");
    compare_state(*battle, *pod, "ability_selection.decision");
}

OfficialMainFixtureRecord make_ability_selection_activation_record(
    const OfficialRulePackView& rules) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_ability_selection(&battle, &input, rules);
    const int ability_index = find_option(
        battle.state,
        SelectOptionType::Ability,
        static_cast<int>(AreaType::Active));
    require(ability_index >= 0, "ability_selection_activation.option");

    OfficialStatePod expected = input;
    const std::uint16_t selected = static_cast<std::uint16_t>(ability_index);
    apply_official_action(&battle.state, ability_index);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected, rules, &selected, 1);
    require(
        status == OfficialFlowStatus::kNeedsAction,
        "ability_selection_activation.status");
    compare_state(battle, expected, "ability_selection_activation.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = 29;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = 1;
    record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

OfficialMainFixtureRecord make_ability_selection_resume_record(
    const OfficialRulePackView& rules,
    bool discard) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_ability_selection(&battle, &input, rules);
    advance_ability_to_selection(&battle, &input, rules);

    std::vector<int> official_selection;
    std::uint16_t selected = 0;
    if (discard) {
        const int card_index = find_option(
            battle.state, SelectOptionType::Card);
        require(card_index >= 0, "ability_selection_resume.option");
        official_selection.push_back(card_index);
        selected = static_cast<std::uint16_t>(card_index);
    }
    OfficialStatePod expected = input;
    apply_official_selection(&battle.state, official_selection);
    const OfficialFlowStatus status = official_apply_pending_action(
        &expected,
        rules,
        discard ? &selected : nullptr,
        discard ? 1 : 0);
    require(
        status == OfficialFlowStatus::kNeedsAction,
        "ability_selection_resume.status");
    require(
        battle.state.selectType == SelectType::Main,
        "ability_selection_resume.main");
    compare_state(battle, expected, "ability_selection_resume.after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = discard ? 31 : 30;
    record.operation = OfficialMainFixtureOperation::kApply;
    record.expected_status = status;
    record.action_count = discard ? 1 : 0;
    if (discard) record.option_indices[0] = selected;
    record.input = input;
    record.expected = expected;
    return record;
}

enum class AbilityBlockedCase : std::uint8_t {
    kWrongArea,
    kConditionFalse,
    kSuppressed,
    kOnceTurnUsed,
};

OfficialMainFixtureRecord make_ability_blocked_record(
    const OfficialRulePackView& rules,
    std::uint32_t scenario_id,
    AbilityBlockedCase blocked_case) {
    BattleData battle;
    OfficialStatePod input{};
    prepare_case(&battle, &input, 137 + scenario_id, 13700 + scenario_id);
    if (blocked_case == AbilityBlockedCase::kWrongArea) {
        place_official(&battle.state, 47, 109, 0, AreaType::Bench);
        add_pod(&input, 47, 109, 0, OfficialArea::kBench);
    } else if (blocked_case == AbilityBlockedCase::kConditionFalse) {
        battle.state.getCard(CardRef(10)).cardId = 36;
        input.cards[10].card_id = 36;
        battle.state.players[0].deck.clear();
        input.players[0].deck.count = 0;
    } else {
        battle.state.getCard(CardRef(10)).cardId = 351;
        input.cards[10].card_id = 351;
        if (blocked_case == AbilityBlockedCase::kSuppressed) {
            battle.state.getCard(CardRef(10)).noAbility = true;
            official_continual_set_flag(&input.cards[10], 0);
        } else {
            battle.state.getCard(CardRef(10)).abilityUsed.push_back(351);
            input.cards[10].ability_used[0] = 351;
            input.cards[10].ability_used_count = 1;
        }
    }

    OfficialStatePod expected = input;
    prepare_official_decision(&battle.state);
    const OfficialFlowStatus status = official_advance_idle_state_to_decision(
        &expected, rules);
    require(status == OfficialFlowStatus::kNeedsAction, "ability_blocked.status");
    require(
        find_option(battle.state, SelectOptionType::Ability) < 0,
        "ability_blocked.option_absent");
    compare_state(
        battle,
        expected,
        "ability_blocked_" + std::to_string(scenario_id) + ".after");

    OfficialMainFixtureRecord record{};
    record.scenario_id = scenario_id;
    record.operation = OfficialMainFixtureOperation::kAdvance;
    record.expected_status = status;
    record.input = input;
    record.expected = expected;
    return record;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error(
                "usage: official_main_oracle <official_rules.bin> <fixture.bin>");
        }
        InitializeAll();
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        require(
            rule_pack.size() >= sizeof(OfficialRulePackHeader),
            "rule pack size");
        const OfficialRulePackView rules = make_official_rule_pack_view(
            rule_pack.data());
        require(std::memcmp(rules.header->magic, "PTCGRUL1", 8) == 0, "rule pack");

        std::vector<OfficialMainFixtureRecord> records;
        records.reserve(kOfficialMainFixtureRecordCount);
        try {
            records.push_back(make_advance_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("prepare scenario: ") + error.what());
        }
        try {
            records.push_back(make_action_record(
                rules, SelectOptionType::Attack, 223, 23, 2301, 2, "attack"));
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("attack scenario: ") + error.what());
        }
        try {
            records.push_back(make_action_record(
                rules, SelectOptionType::End, -1, 29, 2901, 3, "end"));
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("end scenario: ") + error.what());
        }
        try {
            records.push_back(make_action_record(
                rules, SelectOptionType::Play, 0, 31, 3101, 4, "play_basic"));
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("play scenario: ") + error.what());
        }
        try {
            records.push_back(make_action_record(
                rules, SelectOptionType::Play, 1, 37, 3701, 5, "play_stadium"));
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("stadium scenario: ") + error.what());
        }
        try {
            records.push_back(make_action_record(
                rules, SelectOptionType::Play, 2, 41, 4101, 6, "play_item"));
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("item scenario: ") + error.what());
        }
        try {
            records.push_back(make_action_record(
                rules, SelectOptionType::Play, 3, 43, 4301, 7, "play_supporter"));
        } catch (const std::exception& error) {
            throw std::runtime_error(std::string("supporter scenario: ") + error.what());
        }
        try {
            records.push_back(make_action_record(
                rules,
                SelectOptionType::Play,
                4,
                47,
                4701,
                8,
                "play_item_selection"));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("item selection scenario: ") + error.what());
        }
        try {
            records.push_back(make_item_resume_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("item resume scenario: ") + error.what());
        }
        try {
            records.push_back(make_attach_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("attach scenario: ") + error.what());
        }
        try {
            records.push_back(make_evolve_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("evolve scenario: ") + error.what());
        }
        try {
            records.push_back(make_ability_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("ability scenario: ") + error.what());
        }
        try {
            records.push_back(make_discard_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("discard scenario: ") + error.what());
        }
        try {
            records.push_back(make_retreat_energy_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("retreat energy scenario: ") + error.what());
        }
        try {
            records.push_back(make_retreat_switch_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("retreat switch scenario: ") + error.what());
        }
        try {
            records.push_back(make_retreat_complete_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("retreat complete scenario: ") + error.what());
        }
        try {
            records.push_back(make_tool_attach_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("tool attach scenario: ") + error.what());
        }
        try {
            records.push_back(make_retreat_zero_cost_record(rules));
            records.push_back(make_retreat_multi_energy_record(rules, false));
            records.push_back(make_retreat_multi_energy_record(rules, true));
            records.push_back(make_retreat_blocked_record(
                rules, 21, RetreatBlockedCase::kAlreadyRetreated));
            records.push_back(make_retreat_blocked_record(
                rules, 22, RetreatBlockedCase::kAsleep));
            records.push_back(make_retreat_blocked_record(
                rules, 23, RetreatBlockedCase::kCannotRetreatThisTurn));
            records.push_back(make_retreat_blocked_record(
                rules, 24, RetreatBlockedCase::kCannotRetreatContinual));
            records.push_back(make_retreat_blocked_record(
                rules, 25, RetreatBlockedCase::kPoisonRestriction));
            records.push_back(make_retreat_blocked_record(
                rules, 26, RetreatBlockedCase::kPokemonItem));
            records.push_back(make_retreat_blocked_record(
                rules, 27, RetreatBlockedCase::kInsufficientEnergy));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("retreat edge scenario: ") + error.what());
        }
        try {
            records.push_back(make_rainbow_dna_evolve_record(rules));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("rainbow DNA scenario: ") + error.what());
        }
        try {
            records.push_back(make_ability_selection_activation_record(rules));
            records.push_back(make_ability_selection_resume_record(rules, false));
            records.push_back(make_ability_selection_resume_record(rules, true));
            records.push_back(make_ability_blocked_record(
                rules, 32, AbilityBlockedCase::kWrongArea));
            records.push_back(make_ability_blocked_record(
                rules, 33, AbilityBlockedCase::kConditionFalse));
            records.push_back(make_ability_blocked_record(
                rules, 34, AbilityBlockedCase::kSuppressed));
            records.push_back(make_ability_blocked_record(
                rules, 35, AbilityBlockedCase::kOnceTurnUsed));
        } catch (const std::exception& error) {
            throw std::runtime_error(
                std::string("ability edge scenario: ") + error.what());
        }

        OfficialMainFixtureHeader header{};
        std::memcpy(header.magic, "PTCGMAIN", 8);
        header.record_bytes = sizeof(OfficialMainFixtureRecord);
        header.record_count = records.size();
        std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot create main fixture");
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        output.write(
            reinterpret_cast<const char*>(records.data()),
            static_cast<std::streamsize>(records.size() * sizeof(records[0])));
        if (!output) throw std::runtime_error("cannot write main fixture");

        std::cout
            << "{\"passed\":true,\"scenarios\":35,\"checks\":35,"
            << "\"fixture_bytes\":"
            << sizeof(header) + records.size() * sizeof(records[0])
            << ",\"scope\":[\"main_options\",\"main_attack\",\"main_end\","
            << "\"main_play_basic\",\"main_play_stadium\","
            << "\"main_play_item\",\"main_play_supporter\","
            << "\"main_play_item_selection\",\"main_play_item_resume\","
            << "\"main_attach_energy\",\"main_evolve\",\"main_ability\","
            << "\"main_discard\",\"main_retreat_energy\","
            << "\"main_retreat_switch\",\"main_retreat_complete\","
            << "\"main_attach_tool\",\"main_retreat_zero_cost\","
            << "\"main_retreat_multi_energy_continue\","
            << "\"main_retreat_multi_energy_switch\","
            << "\"main_retreat_blocked_already_used\","
            << "\"main_retreat_blocked_asleep\","
            << "\"main_retreat_blocked_this_turn\","
            << "\"main_retreat_blocked_continual\","
            << "\"main_retreat_blocked_poison\","
            << "\"main_retreat_blocked_pokemon_item\","
            << "\"main_retreat_blocked_insufficient_energy\","
            << "\"main_evolve_rainbow_dna\","
            << "\"main_ability_selection_activation\","
            << "\"main_ability_optional_skip\","
            << "\"main_ability_selection_apply\","
            << "\"main_ability_blocked_area\","
            << "\"main_ability_blocked_condition\","
            << "\"main_ability_blocked_suppressed\","
            << "\"main_ability_blocked_once_turn\"]}"
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
