#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "All.h"
#include "ptcg_cuda/official_knockout_pod.cuh"
#include "ptcg_cuda/official_turn_flow_pod.cuh"

namespace {

using namespace ptcg::cuda_engine;

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open rule pack: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void require(bool condition, const std::string& label) {
    if (!condition) throw std::runtime_error("official oracle failed: " + label);
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
    state->selectType = SelectType::None;
    state->selectContext = SelectContext::None;
    state->selectPlayer = -1;
    state->firstPlayer = 0;
    state->lastStadiumPlayer = 0;
    state->turn = 1;
    state->phase = GamePhase::Main;
    state->gameResult = GameResult::None;
    state->finishReason = FinishReason::None;
    state->moveCounter = 1;
    state->attacker = {};
    state->currentAttackId = 0;
    state->srcAttackId = 0;
    state->coinHeadCount = 0;
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
        default: throw std::runtime_error("unsupported oracle area");
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
    card.move_counter = ++state->move_counter;
    require(
        official_pod_push_zone_card(state, player, area, OfficialCardRefPod{ref}),
        "pod add card");
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

void setup_pod(OfficialStatePod* state, std::uint64_t episode, std::uint64_t seed) {
    official_pod_reset(state, episode, seed);
    state->first_player = 0;
    state->turn = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
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
    require(
        official.options.size() == pod.options.count,
        label + ".count official=" + std::to_string(official.options.size())
            + " pod=" + std::to_string(pod.options.count)
            + " select_player=" + std::to_string(official.selectPlayer)
            + " p0_prize=" + std::to_string(official.players[0].prize.size())
            + " p1_prize=" + std::to_string(official.players[1].prize.size()));
    require(
        static_cast<int>(official.selectType) == pod.select_type,
        label + ".select_type");
    require(
        static_cast<int>(official.selectContext) == pod.select_context,
        label + ".select_context");
    require(official.selectPlayer == pod.select_player, label + ".select_player");
    require(
        official.selectMin == pod.select_min,
        label + ".select_min official=" + std::to_string(official.selectMin)
            + " pod=" + std::to_string(pod.select_min));
    require(
        official.selectMax == pod.select_max,
        label + ".select_max official=" + std::to_string(official.selectMax)
            + " pod=" + std::to_string(pod.select_max));
    for (std::size_t index = 0; index < official.options.size(); ++index) {
        const SelectOption& expected = official.options[index];
        const OfficialSelectOptionPod& actual = pod.options.values[index];
        require(
            static_cast<int>(expected.type) == actual.type,
            label + ".option.type[" + std::to_string(index) + "]");
        require(expected.param0 == actual.params[0], label + ".option.param0");
        require(expected.param1 == actual.params[1], label + ".option.param1");
        require(expected.param2 == actual.params[2], label + ".option.param2");
        require(expected.param3 == actual.params[3], label + ".option.param3");
        require(expected.param4 == actual.params[4], label + ".option.param4");
        if (expected.type == SelectOptionType::Card
            || expected.type == SelectOptionType::ToolCard
            || expected.type == SelectOptionType::EnergyCard
            || expected.type == SelectOptionType::Energy) {
            const CardRef ref = expected.type == SelectOptionType::ToolCard
                ? official.attachedCardRefFromOption(expected)
                : official.getCardRef(expected);
            require(
                ref.cardIndex == actual.resolved_card,
                label + ".option.card expected=" + std::to_string(ref.cardIndex)
                    + " actual=" + std::to_string(actual.resolved_card)
                    + " param3=" + std::to_string(expected.param3));
        }
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

void compare_rng_prefix(const std::mt19937& official, OfficialMt19937 pod) {
    std::mt19937 expected_rng = official;
    for (int index = 0; index < 16; ++index) {
        const std::uint32_t expected = expected_rng();
        const std::uint32_t actual = official_mt19937_next(&pod);
        require(expected == actual, "rng prefix");
    }
}

void run_rng_oracle() {
    for (std::uint64_t seed : {0ULL, 1ULL, 2ULL, 5489ULL, 20260730ULL}) {
        BattleData battle;
        setup_official(&battle, seed);
        OfficialStatePod pod{};
        setup_pod(&pod, 100 + seed, seed);
        compare_rng(battle.game.rng, pod.rng);
        compare_rng_prefix(battle.game.rng, pod.rng);
    }
}

void run_bench_overflow_oracle(const OfficialRulePackView& rules) {
    BattleData battle;
    setup_official(&battle, 17);
    State& official = battle.state;
    place_official(&official, 10, 25, 0, AreaType::Active);
    place_official(&official, 20, 22, 1, AreaType::Active);
    for (int index = 0; index < 6; ++index) {
        place_official(&official, 30 + index, 25, 0, AreaType::Bench);
        place_official(&official, 40 + index, 22, 1, AreaType::Bench);
    }
    place_official(&official, 60, 7, 0, AreaType::Prize);
    place_official(&official, 61, 7, 1, AreaType::Prize);
    official.lastStadiumPlayer = 1;
    BenchCheck(official);
    drive_official(&official);

    OfficialStatePod pod{};
    setup_pod(&pod, 200, 17);
    pod.last_stadium_player = 1;
    add_pod(&pod, 10, 25, 0, OfficialArea::kActive);
    add_pod(&pod, 20, 22, 1, OfficialArea::kActive);
    for (std::uint16_t index = 0; index < 6; ++index) {
        add_pod(&pod, 30 + index, 25, 0, OfficialArea::kBench);
        add_pod(&pod, 40 + index, 22, 1, OfficialArea::kBench);
    }
    add_pod(&pod, 60, 7, 0, OfficialArea::kPrize);
    add_pod(&pod, 61, 7, 1, OfficialArea::kPrize);
    require(official_prepare_bench_overflow(&pod), "pod bench overflow");
    compare_options(official, pod, "bench_overflow");
    require(official.selectPlayer == 1, "official bench order");
    require(pod.select_player == 1, "pod bench order");
    (void)rules;
}

void run_tool_overflow_oracle(const OfficialRulePackView& rules) {
    BattleData battle;
    setup_official(&battle, 19);
    State& official = battle.state;
    place_official(&official, 10, 25, 0, AreaType::Active);
    place_official(&official, 20, 22, 1, AreaType::Active);
    place_official(&official, 21, 24, 1, AreaType::Bench);
    place_official(&official, 30, 1, 0, AreaType::Tool);
    place_official(&official, 31, 1, 0, AreaType::Tool);
    official.getCard(CardRef(30)).attachMoveCounter = official.getCard(CardRef(10)).moveCounter;
    official.getCard(CardRef(31)).attachMoveCounter = official.getCard(CardRef(10)).moveCounter;
    for (int ref = 40; ref < 43; ++ref) {
        place_official(&official, ref, 1, 1, AreaType::Tool);
        official.getCard(CardRef(ref)).attachMoveCounter = official.getCard(CardRef(21)).moveCounter;
    }
    official.getCard(CardRef(10)).tool2 = true;
    official.getCard(CardRef(21)).tool2 = true;
    place_official(&official, 60, 7, 0, AreaType::Prize);
    place_official(&official, 61, 7, 1, AreaType::Prize);
    ToolCountProc(official);
    drive_official(&official);

    OfficialStatePod pod{};
    setup_pod(&pod, 201, 19);
    add_pod(&pod, 10, 25, 0, OfficialArea::kActive);
    add_pod(&pod, 20, 22, 1, OfficialArea::kActive);
    add_pod(&pod, 21, 24, 1, OfficialArea::kBench);
    add_pod(&pod, 30, 1, 0, OfficialArea::kTool);
    add_pod(&pod, 31, 1, 0, OfficialArea::kTool);
    pod.cards[30].attach_move_counter = pod.cards[10].move_counter;
    pod.cards[31].attach_move_counter = pod.cards[10].move_counter;
    for (std::uint16_t ref = 40; ref < 43; ++ref) {
        add_pod(&pod, ref, 1, 1, OfficialArea::kTool);
        pod.cards[ref].attach_move_counter = pod.cards[21].move_counter;
    }
    pod.cards[10].continual_state[4] |= 1ULL << 34;
    pod.cards[21].continual_state[4] |= 1ULL << 34;
    add_pod(&pod, 60, 7, 0, OfficialArea::kPrize);
    add_pod(&pod, 61, 7, 1, OfficialArea::kPrize);
    require(official_prepare_tool_overflow(&pod), "pod tool overflow");
    compare_options(official, pod, "tool_overflow");
    require(official.selectPlayer == 1, "official tool order");
    require(pod.select_player == 1, "pod tool order");
    require(pod.options.values[0].params[3] == 0, "tool option attachment index");
    (void)rules;
}

void run_multi_ko_oracle(const OfficialRulePackView& rules) {
    BattleData battle;
    setup_official(&battle, 23);
    State& official = battle.state;
    place_official(&official, 10, 25, 0, AreaType::Active);
    place_official(&official, 20, 22, 1, AreaType::Active);
    place_official(&official, 21, 24, 1, AreaType::Bench);
    for (int ref = 30; ref < 34; ++ref) place_official(&official, ref, 7, 0, AreaType::Prize);
    place_official(&official, 50, 7, 1, AreaType::Prize);
    official.getCard(CardRef(20)).ko = true;
    official.getCard(CardRef(21)).ko = true;
    KOProc(official);
    drive_official(&official);

    OfficialStatePod pod{};
    setup_pod(&pod, 202, 23);
    add_pod(&pod, 10, 25, 0, OfficialArea::kActive);
    add_pod(&pod, 20, 22, 1, OfficialArea::kActive);
    add_pod(&pod, 21, 24, 1, OfficialArea::kBench);
    for (std::uint16_t ref = 30; ref < 34; ++ref) add_pod(&pod, ref, 7, 0, OfficialArea::kPrize);
    add_pod(&pod, 50, 7, 1, OfficialArea::kPrize);
    pod.cards[20].runtime_flags |= kCardKo;
    pod.cards[21].runtime_flags |= kCardKo;
    OfficialKnockoutResult result = official_begin_knockout(&pod, rules);
    require(result == OfficialKnockoutResult::kNeedsAction, "pod first KO decision");
    compare_options(official, pod, "multi_ko.first");
    require(official.selectMin == 1 && pod.select_min == 1, "multi KO first request");
    choose_official(&official, {0});
    std::uint16_t first = 0;
    result = official_resume_knockout(&pod, rules, &first, 1);
    require(result == OfficialKnockoutResult::kNeedsAction, "pod second KO decision");
    compare_options(official, pod, "multi_ko.second");
    require(official.selectMin == 2 && pod.select_min == 2, "multi KO second request");
}

void run_lucky_bonus_oracle() {
    Skill* selected_skill = &SkillTable.begin()->second;
    const bool old_lucky = selected_skill->luckyBonus;
    Skill* old_ability = CardTable.at(1).ability;
    selected_skill->luckyBonus = true;
    CardTable.at(1).ability = selected_skill;

    BattleData battle;
    setup_official(&battle, 2);
    State& official = battle.state;
    place_official(&official, 10, 25, 0, AreaType::Active);
    place_official(&official, 20, 24, 1, AreaType::Active);
    place_official(&official, 30, 1, 0, AreaType::Prize);
    place_official(&official, 31, 1, 0, AreaType::Prize);
    place_official(&official, 32, 7, 0, AreaType::Prize);
    place_official(&official, 33, 7, 0, AreaType::Prize);
    place_official(&official, 50, 7, 1, AreaType::Prize);
    official.getCard(CardRef(30)).reverse = true;
    official.getCard(CardRef(31)).reverse = true;
    official.getCard(CardRef(20)).ko = true;
    KOProc(official);
    drive_official(&official);

    OfficialRulePackHeader header{};
    header.counts[static_cast<std::uint32_t>(OfficialRuleSection::kCards)] = 25;
    header.counts[static_cast<std::uint32_t>(OfficialRuleSection::kSkills)] = 1;
    OfficialCardRule cards[25]{};
    for (int index = 0; index < 25; ++index) cards[index].values[kCardId] = index + 1;
    cards[0].values[kCardAbilityId] = 1;
    cards[6].values[kCardHp] = 100;
    cards[23].values[kCardHp] = 100;
    cards[23].values[kCardPokemonType] = 3;
    cards[24].values[kCardHp] = 100;
    OfficialSkillRule skills[1]{};
    skills[0].values[kSkillId] = 1;
    skills[0].flags = kOfficialSkillLuckyBonusFlag;
    OfficialRulePackView rules{};
    rules.header = &header;
    rules.cards = cards;
    rules.skills = skills;

    OfficialStatePod pod{};
    setup_pod(&pod, 203, 2);
    add_pod(&pod, 10, 25, 0, OfficialArea::kActive);
    add_pod(&pod, 20, 24, 1, OfficialArea::kActive);
    add_pod(&pod, 30, 1, 0, OfficialArea::kPrize);
    add_pod(&pod, 31, 1, 0, OfficialArea::kPrize);
    add_pod(&pod, 32, 7, 0, OfficialArea::kPrize);
    add_pod(&pod, 33, 7, 0, OfficialArea::kPrize);
    add_pod(&pod, 50, 7, 1, OfficialArea::kPrize);
    pod.cards[30].reverse = 1;
    pod.cards[31].reverse = 1;
    pod.cards[20].runtime_flags |= kCardKo;
    OfficialKnockoutResult result = official_begin_knockout(&pod, rules);
    require(result == OfficialKnockoutResult::kNeedsAction, "pod lucky first decision");
    compare_options(official, pod, "lucky.first");
    choose_official(&official, {0, 1});
    const std::uint16_t initial[2]{0, 1};
    result = official_resume_knockout(&pod, rules, initial, 2);
    require(result == OfficialKnockoutResult::kNeedsAction, "pod lucky choice");
    compare_options(official, pod, "lucky.choice");
    require(official.contextCard.cardIndex == pod.context_card.index, "lucky context");
    choose_official(&official, {0});
    const std::uint16_t yes = 0;
    result = official_resume_knockout(&pod, rules, &yes, 1);
    require(result == OfficialKnockoutResult::kNeedsAction, "pod extra prize decision");
    compare_options(official, pod, "lucky.extra_prize");
    require(pod.rng.draw_count == 1, "lucky coin draw");
    choose_official(&official, {0});
    const std::uint16_t extra = 0;
    result = official_resume_knockout(&pod, rules, &extra, 1);
    require(result == OfficialKnockoutResult::kNeedsAction, "pod second lucky choice");
    compare_options(official, pod, "lucky.second_choice");
    choose_official(&official, {1});
    const std::uint16_t no = 1;
    result = official_resume_knockout(&pod, rules, &no, 1);
    require(result == OfficialKnockoutResult::kComplete, "pod lucky complete");
    require(official.players[0].bench.size() == pod.players[0].bench.count, "lucky bench");
    require(official.players[0].hand.size() == pod.players[0].hand.count, "lucky hand");
    require(official.players[0].prize.size() == pod.players[0].prize.count, "lucky prize");

    CardTable.at(1).ability = old_ability;
    selected_skill->luckyBonus = old_lucky;
}

void run_turn_transition_oracle(const OfficialRulePackView& rules) {
    BattleData battle;
    setup_official(&battle, 31);
    State& official = battle.state;
    place_official(&official, 10, 25, 0, AreaType::Active);
    place_official(&official, 20, 22, 1, AreaType::Active);
    place_official(&official, 30, 7, 0, AreaType::Prize);
    place_official(&official, 40, 7, 1, AreaType::Prize);
    place_official(&official, 41, 1, 1, AreaType::Deck);
    place_official(&official, 50, 17, 0, AreaType::Energy);
    official.getCard(CardRef(50)).attachMoveCounter = official.getCard(CardRef(10)).moveCounter;
    official.getCard(CardRef(10)).thisTurn.value[0] = 11;
    official.getCard(CardRef(20)).nextTurn.value[0] = 22;
    official.getCard(CardRef(10)).nextTurnEnemy.value[0] = 33;
    official.players[1].nextTurn.value = 44;
    official.turnHistories[0].turnAttackId = 101;
    official.turnHistories[1].turnAttackId = 102;
    official.turnHistories[2].turnAttackId = 103;
    TurnEnd(official);
    while (official.turn != 2 && !official.isFinish()) {
        drive_official(&official);
    }

    OfficialStatePod pod{};
    setup_pod(&pod, 204, 31);
    add_pod(&pod, 10, 25, 0, OfficialArea::kActive);
    add_pod(&pod, 20, 22, 1, OfficialArea::kActive);
    add_pod(&pod, 30, 7, 0, OfficialArea::kPrize);
    add_pod(&pod, 40, 7, 1, OfficialArea::kPrize);
    add_pod(&pod, 41, 1, 1, OfficialArea::kDeck);
    add_pod(&pod, 50, 17, 0, OfficialArea::kEnergy);
    pod.cards[50].attach_move_counter = pod.cards[10].move_counter;
    pod.cards[10].this_turn[0] = 11;
    pod.cards[20].next_turn[0] = 22;
    pod.cards[10].next_turn_enemy = 33;
    pod.players[1].next_turn = 44;
    pod.turn_histories[0].attack_id = 101;
    pod.turn_histories[1].attack_id = 102;
    pod.turn_histories[2].attack_id = 103;
    const OfficialTurnFlowResult result = official_begin_turn_end(&pod, rules);
    require(result == OfficialTurnFlowResult::kComplete, "pod turn transition");
    require(official.turn == pod.turn, "turn number");
    require(official.players[1].hand.size() == pod.players[1].hand.count, "turn draw");
    require(official.players[0].trash.size() == pod.players[0].trash.count, "turn expiry");
    require(official.turnHistories[1].turnAttackId == pod.turn_histories[1].attack_id, "history 1");
    require(official.turnHistories[2].turnAttackId == pod.turn_histories[2].attack_id, "history 2");
    require(official.getCard(CardRef(10)).thisTurn.value[0] == pod.cards[10].this_turn[0], "card turn state");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: official_turn_flow_oracle <rule-pack>");
        InitializeAll();
        const std::vector<std::uint8_t> bytes = read_binary(argv[1]);
        const OfficialRulePackView rules = make_official_rule_pack_view(bytes.data());
        run_rng_oracle();
        run_bench_overflow_oracle(rules);
        run_tool_overflow_oracle(rules);
        run_multi_ko_oracle(rules);
        run_lucky_bonus_oracle();
        run_turn_transition_oracle(rules);
        std::cout << "{\"passed\":true,\"rng_seeds\":5,\"scenarios\":6,"
                     "\"scope\":[\"std_mt19937\",\"bench_lifo\",\"tool_lifo\","
                     "\"multi_ko_prize_lifo\",\"lucky_bonus_interleave\","
                     "\"turn_transition\"]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
