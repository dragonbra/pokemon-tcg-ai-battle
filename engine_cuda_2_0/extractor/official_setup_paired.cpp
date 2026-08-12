#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "All.h"
#include "official_state_bridge.h"
#include "ptcg_cuda/official_setup_pod.cuh"

namespace {

using ptcg::cuda_engine::SetupBattleState;
using ptcg::cuda_engine::SetupCardMeta;
using ptcg::cuda_engine::SetupCardToken;
using ptcg::cuda_engine::SetupDecision;
using ptcg::cuda_engine::SetupOption;
using ptcg::cuda_engine::SetupTrace;
using ptcg::cuda_engine::OfficialStatePod;
using ptcg::cuda_engine::extractor::OfficialStateBridgeContext;
using ptcg::cuda_engine::extractor::bridge_official_state;

constexpr std::size_t kDeckSize = 60;

std::mt19937 make_official_rng(std::uint64_t seed) {
    return std::mt19937(static_cast<std::uint32_t>(seed));
}

std::vector<std::uint16_t> read_deck(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open deck: " + path);
    }
    std::vector<std::uint16_t> cards;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t comma = line.find(',');
        const std::string token = line.substr(0, comma);
        try {
            std::size_t parsed = 0;
            const unsigned long value = std::stoul(token, &parsed);
            if (parsed == token.size()) {
                cards.push_back(static_cast<std::uint16_t>(value));
            }
        } catch (const std::exception&) {
        }
    }
    if (cards.size() != kDeckSize) {
        throw std::runtime_error("deck must contain exactly 60 card IDs: " + path);
    }
    return cards;
}

std::vector<SetupCardMeta> build_metadata() {
    int max_id = 0;
    for (const auto& [card_id, _] : CardTable) {
        max_id = std::max(max_id, static_cast<int>(card_id));
    }
    std::vector<SetupCardMeta> result(static_cast<std::size_t>(max_id + 1));
    for (const auto& [card_id, card] : CardTable) {
        const bool basic = card.cardType == CardType::Pokemon
            && card.evolutionType == EvolutionType::Basic;
        result.at(card_id) = {
            static_cast<std::uint8_t>(basic),
            static_cast<std::uint8_t>(card.canSetup()),
            static_cast<std::uint8_t>(card.canSetupActive()),
            0,
        };
    }
    return result;
}

void fail(
    std::uint64_t seed,
    const std::string& field,
    long long expected,
    long long actual) {
    throw std::runtime_error(
        "seed=" + std::to_string(seed) + " field=" + field
        + " expected=" + std::to_string(expected)
        + " actual=" + std::to_string(actual));
}

void compare_value(
    std::uint64_t seed,
    const std::string& field,
    long long expected,
    long long actual) {
    if (expected != actual) {
        fail(seed, field, expected, actual);
    }
}

void compare_option(
    std::uint64_t seed,
    std::size_t decision_index,
    std::size_t option_index,
    const SelectOption& official,
    const SetupOption& pod) {
    const std::array<long long, 6> expected = {
        static_cast<int>(official.type), official.param0, official.param1,
        official.param2, official.param3, official.param4};
    const std::array<long long, 6> actual = {
        pod.type, pod.param0, pod.param1, pod.param2, pod.param3, pod.param4};
    for (std::size_t field = 0; field < expected.size(); ++field) {
        compare_value(
            seed,
            "decision[" + std::to_string(decision_index) + "].option["
                + std::to_string(option_index) + "].field[" + std::to_string(field) + "]",
            expected[field],
            actual[field]);
    }
}

void compare_decision(
    std::uint64_t seed,
    std::size_t index,
    const State& official,
    const SetupDecision& pod) {
    compare_value(seed, "decision.select_type", static_cast<int>(official.selectType), pod.select_type);
    compare_value(seed, "decision.select_context", static_cast<int>(official.selectContext), pod.select_context);
    compare_value(seed, "decision.select_player", official.selectPlayer, pod.select_player);
    compare_value(seed, "decision.select_min", official.selectMin, pod.select_min);
    compare_value(seed, "decision.select_max", official.selectMax, pod.select_max);
    compare_value(seed, "decision.option_count", official.options.size(), pod.option_count);
    for (std::size_t option = 0; option < official.options.size(); ++option) {
        compare_option(seed, index, option, official.options[option], pod.options[option]);
    }
}

SetupCardToken official_token(const State& state, CardRef ref) {
    const Card& card = state.getCard(ref);
    return {
        static_cast<std::uint16_t>(card.cardId),
        static_cast<std::uint16_t>(ref.cardIndex),
    };
}

template <typename OfficialList>
void compare_zone(
    std::uint64_t seed,
    const std::string& field,
    const State& state,
    const OfficialList& official,
    const SetupCardToken* pod,
    std::size_t pod_count) {
    compare_value(seed, field + ".count", official.size(), pod_count);
    for (std::size_t i = 0; i < official.size(); ++i) {
        const SetupCardToken expected = official_token(state, official[i]);
        compare_value(seed, field + ".card_id", expected.card_id, pod[i].card_id);
        compare_value(seed, field + ".instance_id", expected.instance_id, pod[i].instance_id);
    }
}

std::vector<std::uint32_t> serialize_rng(const std::mt19937& rng) {
    std::ostringstream stream;
    stream << rng;
    std::istringstream input(stream.str());
    std::vector<std::uint32_t> values;
    unsigned long long value = 0;
    while (input >> value) {
        values.push_back(static_cast<std::uint32_t>(value));
    }
    return values;
}

std::uint64_t count_rng_draws(
    std::uint64_t seed,
    const std::mt19937& final_rng) {
    std::mt19937 candidate = make_official_rng(seed);
    for (std::uint64_t count = 0; count <= 1'000'000; ++count) {
        if (candidate == final_rng) {
            return count;
        }
        candidate();
    }
    throw std::runtime_error("official RNG draw count exceeded search bound");
}

void compare_final_state(
    std::uint64_t seed,
    const ApiData& official,
    const SetupBattleState& pod,
    std::uint64_t rng_draw_count) {
    const State& state = official.state;
    compare_value(seed, "error", 0, pod.error);
    compare_value(seed, "first_player", state.firstPlayer, pod.first_player);
    compare_value(seed, "turn", state.turn, pod.turn);
    compare_value(seed, "actor", state.activePlayerIndex(), pod.actor);
    compare_value(seed, "function_stack_size", state.functionStack.size(), 1);
    compare_value(seed, "continuation_id", state.functionStack.back().functionIndex, pod.continuation_id);

    for (int player = 0; player < 2; ++player) {
        const PlayerState& expected = state.players[player];
        const auto& actual = pod.players[player];
        compare_zone(seed, "deck", state, expected.deck, actual.deck, actual.deck_count);
        compare_zone(seed, "hand", state, expected.hand, actual.hand, actual.hand_count);
        compare_zone(seed, "prize", state, expected.prize, actual.prize, actual.prize_count);
        compare_zone(
            seed, "active", state, expected.active, &actual.active,
            actual.active_present == 0 ? 0 : 1);
        compare_value(seed, "mulligan_count", state.mulliganCount[player], actual.mulligan_count);
    }

    const std::vector<std::uint32_t> rng = serialize_rng(official.game.rng);
    compare_value(
        seed, "rng_serialized_size",
        ptcg::cuda_engine::OfficialMt19937::kStateSize + 1, rng.size());
    for (std::size_t i = 0; i < ptcg::cuda_engine::OfficialMt19937::kStateSize; ++i) {
        compare_value(seed, "rng.words[" + std::to_string(i) + "]", rng[i], pod.rng.words[i]);
    }
    compare_value(seed, "rng.index", rng.back(), pod.rng.index);
    compare_value(seed, "rng.draw_count", rng_draw_count, pod.rng.draw_count);
}

template <typename OfficialList, typename PodList>
void compare_bridge_zone(
    std::uint64_t seed,
    const std::string& field,
    const OfficialList& official,
    const PodList& pod) {
    compare_value(seed, field + ".count", official.size(), pod.count);
    for (std::size_t index = 0; index < official.size(); ++index) {
        compare_value(
            seed, field + ".ref", official[index].cardIndex,
            pod.values[index].index);
    }
}

void compare_full_state_bridge(
    std::uint64_t seed,
    const ApiData& official,
    std::uint64_t rng_draw_count) {
    OfficialStatePod bridged{};
    const auto bridge_result = bridge_official_state(
        official.state,
        OfficialStateBridgeContext{seed, rng_draw_count, true},
        &bridged);
    compare_value(
        seed, "bridge.error", 0,
        static_cast<std::int32_t>(bridge_result.error));
    compare_value(seed, "bridge.abi", 6, bridged.abi_version);
    compare_value(seed, "bridge.error_state", 0, bridged.error);
    compare_value(seed, "bridge.episode", seed, bridged.episode_id);
    compare_value(seed, "bridge.turn", official.state.turn, bridged.turn);
    compare_value(
        seed, "bridge.select_type",
        static_cast<int>(official.state.selectType), bridged.select_type);
    compare_value(
        seed, "bridge.select_context",
        static_cast<int>(official.state.selectContext), bridged.select_context);
    compare_value(
        seed, "bridge.option_count",
        official.state.options.size(), bridged.options.count);
    compare_value(
        seed, "bridge.continuation_count",
        official.state.functionStack.size(), bridged.continuations.count);
    compare_value(seed, "bridge.flow_flags", 0, bridged.flow_flags);

    for (std::size_t index = 0; index < official.state.options.size(); ++index) {
        const SelectOption& expected = official.state.options[index];
        const auto& actual = bridged.options.values[index];
        compare_value(seed, "bridge.option.type", static_cast<int>(expected.type), actual.type);
        compare_value(seed, "bridge.option.param0", expected.param0, actual.params[0]);
        compare_value(seed, "bridge.option.param1", expected.param1, actual.params[1]);
        compare_value(seed, "bridge.option.param2", expected.param2, actual.params[2]);
        compare_value(seed, "bridge.option.param3", expected.param3, actual.params[3]);
        compare_value(seed, "bridge.option.param4", expected.param4, actual.params[4]);
    }
    for (int player = 0; player < 2; ++player) {
        const PlayerState& expected = official.state.players[player];
        const auto& actual = bridged.players[player];
        compare_bridge_zone(seed, "bridge.active", expected.active, actual.active);
        compare_bridge_zone(seed, "bridge.bench", expected.bench, actual.bench);
        compare_bridge_zone(seed, "bridge.prize", expected.prize, actual.prize);
        compare_bridge_zone(seed, "bridge.hand", expected.hand, actual.hand);
        compare_bridge_zone(seed, "bridge.deck", expected.deck, actual.deck);
        compare_bridge_zone(seed, "bridge.trash", expected.trash, actual.trash);
    }
    for (std::size_t index = 0; index < official.state.allCard.size(); ++index) {
        const Card& expected = official.state.allCard[index];
        const auto& actual = bridged.cards[index];
        compare_value(seed, "bridge.card_id", expected.cardId, actual.card_id);
        compare_value(seed, "bridge.card_move", expected.moveCounter, actual.move_counter);
        compare_value(seed, "bridge.card_player", expected.playerIndex, actual.player);
        compare_value(
            seed, "bridge.card_area",
            static_cast<int>(expected.area), actual.area);
    }
    const std::vector<std::uint32_t> rng = serialize_rng(official.game.rng);
    for (std::size_t index = 0; index < ptcg::cuda_engine::OfficialMt19937::kStateSize; ++index) {
        compare_value(seed, "bridge.rng.words", rng[index], bridged.rng.words[index]);
    }
    compare_value(seed, "bridge.rng.index", rng.back(), bridged.rng.index);
    compare_value(seed, "bridge.rng.draw_count", rng_draw_count, bridged.rng.draw_count);
    if (!official.state.functionStack.empty()) {
        const GameFunction& expected = official.state.functionStack.back();
        const auto& actual = bridged.continuations.values[bridged.continuations.count - 1];
        compare_value(seed, "bridge.continuation.opcode", expected.functionIndex, actual.opcode);
        compare_value(seed, "bridge.continuation.arg_type", static_cast<int>(expected.argType), actual.arg_type);
        compare_value(seed, "bridge.continuation.call_count", expected.callCount, actual.call_count);
        compare_value(seed, "bridge.continuation.called_count", expected.calledCount, actual.called_count);
    }
}

void initialize_official_battle(
    ApiData* result,
    const std::vector<std::uint16_t>& decks,
    std::uint64_t seed) {
    GameConfig config{};
    config.seed = static_cast<std::uint32_t>(seed);
    config.recordLog = true;
    config.deviceRand = false;
    for (int player = 0; player < 2; ++player) {
        for (int card = 0; card < static_cast<int>(kDeckSize); ++card) {
            config.decks[player].cards[card] = decks[player * kDeckSize + card];
        }
    }
    result->init(config);
    result->start();
    result->next();
}

void run_seed(
    const std::vector<std::uint16_t>& decks,
    const std::vector<SetupCardMeta>& metadata,
    std::uint64_t seed,
    std::uint64_t* rng_draws,
    std::uint16_t* max_mulligans,
    std::uint16_t* max_decisions,
    std::uint64_t* decisions_compared) {
    SetupBattleState pod{};
    SetupTrace trace{};
    ptcg::cuda_engine::official_setup_first_min(
        &pod, &trace, decks.data(), seed, metadata.data(), metadata.size());
    if (trace.overflow != 0 || pod.error != 0) {
        fail(seed, "pod_setup_error", 0, pod.error);
    }

    ApiData official;
    initialize_official_battle(&official, decks, seed);
    std::size_t decision = 0;
    while (official.state.selectContext != SelectContext::Main) {
        if (decision >= trace.decision_count) {
            fail(seed, "decision_count", decision + 1, trace.decision_count);
        }
        compare_decision(seed, decision, official.state, trace.decisions[decision]);
        std::array<int, ptcg::cuda_engine::kOfficialSetupMaxOptions> selected{};
        for (int i = 0; i < official.state.selectMin; ++i) {
            selected[i] = i;
        }
        const int error = ApiSelect(&official, selected.data(), official.state.selectMin);
        compare_value(seed, "official_select_error", 0, error);
        ++decision;
        if (official.state.isFinish()) {
            throw std::runtime_error("official battle terminated during setup");
        }
    }
    compare_value(seed, "decision_count", decision, trace.decision_count);
    *decisions_compared += decision;
    compare_value(seed, "surfaced_decisions", trace.decision_count, pod.surfaced_decisions);
    const std::uint64_t official_rng_draws = count_rng_draws(seed, official.game.rng);
    compare_final_state(seed, official, pod, official_rng_draws);
    compare_full_state_bridge(seed, official, official_rng_draws);

    *rng_draws = std::max(*rng_draws, pod.rng.draw_count);
    *max_decisions = std::max(*max_decisions, pod.surfaced_decisions);
    for (int player = 0; player < 2; ++player) {
        *max_mulligans = std::max(*max_mulligans, pod.players[player].mulligan_count);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string deck0_path;
        std::string deck1_path;
        std::uint64_t seed_start = 1;
        std::uint64_t seed_count = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            if (argument == "--deck0") {
                deck0_path = argv[++i];
            } else if (argument == "--deck1") {
                deck1_path = argv[++i];
            } else if (argument == "--seed-start") {
                seed_start = std::stoull(argv[++i]);
            } else if (argument == "--seed-count") {
                seed_count = std::stoull(argv[++i]);
            } else {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }
        if (deck0_path.empty() || deck1_path.empty() || seed_count == 0) {
            throw std::runtime_error(
                "--deck0, --deck1, and positive --seed-count are required");
        }

        InitializeAll();
        std::vector<std::uint16_t> decks = read_deck(deck0_path);
        const std::vector<std::uint16_t> deck1 = read_deck(deck1_path);
        decks.insert(decks.end(), deck1.begin(), deck1.end());
        const std::vector<SetupCardMeta> metadata = build_metadata();

        std::uint64_t max_rng_draws = 0;
        std::uint16_t max_mulligans = 0;
        std::uint16_t max_decisions = 0;
        std::uint64_t decisions_compared = 0;
        for (std::uint64_t offset = 0; offset < seed_count; ++offset) {
            run_seed(
                decks, metadata, seed_start + offset,
                &max_rng_draws, &max_mulligans, &max_decisions,
                &decisions_compared);
        }
        std::cout
            << "{\"passed\":true,\"seed_start\":" << seed_start
            << ",\"seed_count\":" << seed_count
            << ",\"states_compared\":" << seed_count
            << ",\"full_state_bridge_states\":" << seed_count
            << ",\"decisions_compared\":" << decisions_compared
            << ",\"max_rng_draws\":" << max_rng_draws
            << ",\"max_mulligans\":" << max_mulligans
            << ",\"max_setup_decisions\":" << max_decisions
            << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
