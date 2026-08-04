#include <algorithm>
#include <array>
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
#include "ptcg_cuda/official_seeded_reset_fixture.h"
#include "ptcg_cuda/official_seeded_setup_pod.cuh"

namespace {

using namespace ptcg::cuda_engine;
using namespace ptcg::cuda_engine::extractor;

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open rule pack: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<std::int32_t> read_deck(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open deck: " + path);
    std::vector<std::int32_t> result;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t comma = line.find(',');
        const std::string token = line.substr(0, comma);
        std::istringstream row(token);
        long value = 0;
        if (row >> value) result.push_back(static_cast<std::int32_t>(value));
    }
    if (result.size() != 60) {
        throw std::runtime_error("deck must contain exactly 60 card IDs: " + path);
    }
    return result;
}

std::mt19937 make_official_rng(std::uint64_t seed) {
    return std::mt19937(static_cast<std::uint32_t>(seed));
}

std::uint64_t count_rng_draws(
    std::uint64_t seed,
    const std::mt19937& final_rng) {
    std::mt19937 candidate = make_official_rng(seed);
    for (std::uint64_t count = 0; count <= 1'000'000; ++count) {
        if (candidate == final_rng) return count;
        candidate();
    }
    throw std::runtime_error("official RNG draw count exceeded search bound");
}

void initialize_official_battle(
    ApiData* result,
    const std::vector<std::int32_t>& decks,
    std::uint64_t seed) {
    GameConfig config{};
    config.seed = static_cast<std::uint32_t>(seed);
    config.recordLog = false;
    config.deviceRand = false;
    for (int player = 0; player < 2; ++player) {
        for (int card = 0; card < 60; ++card) {
            config.decks[player].cards[card] = decks[player * 60 + card];
        }
    }
    result->init(config);
    result->start();
    result->next();
}

OfficialStatePod run_official_first_min(
    const std::vector<std::int32_t>& decks,
    std::uint64_t seed) {
    ApiData official;
    initialize_official_battle(&official, decks, seed);
    std::uint32_t decisions = 0;
    while (official.state.selectContext != SelectContext::Main) {
        if (decisions++ >= 128) {
            throw std::runtime_error("official setup decision limit exceeded");
        }
        std::array<int, 128> selected{};
        for (int index = 0; index < official.state.selectMin; ++index) {
            selected[index] = index;
        }
        const int error = ApiSelect(
            &official, selected.data(), official.state.selectMin);
        if (error != 0 || official.state.isFinish()) {
            throw std::runtime_error("official setup selection failed");
        }
    }
    const std::uint64_t rng_draws = count_rng_draws(seed, official.game.rng);
    OfficialStatePod result{};
    const OfficialBridgeResult bridge = bridge_official_state(
        official.state,
        OfficialStateBridgeContext{seed, rng_draws, true},
        &result);
    if (!bridge) {
        throw std::runtime_error(
            "official state bridge failed: "
            + std::to_string(static_cast<std::int32_t>(bridge.error))
            + "." + std::to_string(bridge.detail));
    }
    return result;
}

OfficialFlowStatus run_cpu_pod_first_min(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::vector<std::int32_t>& decks,
    std::uint64_t seed) {
    if (!official_seeded_setup_first_min_state(
            state, rules, decks.data(), seed, seed)) {
        return OfficialFlowStatus::kError;
    }
    const OfficialMainResult main = official_main_begin_refresh(state, rules);
    if (main == OfficialMainResult::kError) return OfficialFlowStatus::kError;
    if (main == OfficialMainResult::kNeedsAction) {
        const OfficialFlowStatus status = official_yield_decision(state);
        official_setup_normalize_boundary(state);
        return status;
    }
    const OfficialFlowStatus status = official_boundary_status(state);
    official_setup_normalize_boundary(state);
    return status;
}

std::size_t first_mismatch(
    const OfficialStatePod& expected,
    const OfficialStatePod& actual) {
    const auto* left = reinterpret_cast<const std::uint8_t*>(&expected);
    const auto* right = reinterpret_cast<const std::uint8_t*>(&actual);
    for (std::size_t index = 0; index < sizeof(OfficialStatePod); ++index) {
        if (left[index] != right[index]) return index;
    }
    return sizeof(OfficialStatePod);
}

template <typename ListT>
void write_card_ids(
    std::ostream& output,
    const OfficialStatePod& state,
    const ListT& list) {
    output << '[';
    for (std::uint16_t index = 0; index < list.count; ++index) {
        if (index != 0) output << ',';
        const OfficialCardStatePod* card = official_pod_card(
            &state, list.values[index]);
        output << (card == nullptr ? 0 : card->card_id);
    }
    output << ']';
}

void write_target_card_ids(
    std::ostream& output,
    const OfficialStatePod& state) {
    output << '[';
    for (std::uint16_t index = 0; index < state.targets.count; ++index) {
        if (index != 0) output << ',';
        const OfficialCardStatePod* card = official_pod_card(
            &state, state.targets.values[index].card);
        output << (card == nullptr ? 0 : card->card_id);
    }
    output << ']';
}

void write_option_types(std::ostream& output, const OfficialStatePod& state) {
    output << '[';
    for (std::uint16_t index = 0; index < state.options.count; ++index) {
        if (index != 0) output << ',';
        output << static_cast<int>(state.options.values[index].type);
    }
    output << ']';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7) {
            throw std::runtime_error(
                "usage: official_seeded_reset_oracle RULES DECK0 DECK1 "
                "SEED_START SEED_COUNT FIXTURE");
        }
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        const OfficialRulePackView rules = make_official_rule_pack_view(
            rule_pack.data());
        std::vector<std::int32_t> decks = read_deck(argv[2]);
        const std::vector<std::int32_t> deck1 = read_deck(argv[3]);
        decks.insert(decks.end(), deck1.begin(), deck1.end());
        const std::uint64_t seed_start = std::stoull(argv[4]);
        const std::uint32_t seed_count = static_cast<std::uint32_t>(
            std::stoul(argv[5]));
        if (seed_count == 0) throw std::runtime_error("seed count must be positive");

        InitializeAll();
        std::ofstream fixture(argv[6], std::ios::binary | std::ios::trunc);
        if (!fixture) throw std::runtime_error("cannot create fixture");
        OfficialSeededResetFixtureHeader header{};
        std::memcpy(header.magic, "PTCGSR01", 8);
        header.record_bytes = sizeof(OfficialSeededResetFixtureRecord);
        header.record_count = seed_count;
        fixture.write(reinterpret_cast<const char*>(&header), sizeof(header));

        std::uint32_t mismatches = 0;
        std::uint32_t status_mismatches = 0;
        std::uint64_t first_mismatch_seed = 0;
        std::size_t first_mismatch_byte = sizeof(OfficialStatePod);
        std::uint8_t first_expected = 0;
        std::uint8_t first_actual = 0;
        OfficialStatePod first_expected_state{};
        OfficialStatePod first_actual_state{};
        for (std::uint32_t offset = 0; offset < seed_count; ++offset) {
            const std::uint64_t seed = seed_start + offset;
            const OfficialStatePod expected = run_official_first_min(decks, seed);
            OfficialStatePod actual{};
            const OfficialFlowStatus actual_status = run_cpu_pod_first_min(
                &actual, rules, decks, seed);
            const OfficialFlowStatus expected_status = official_flow_status(expected);
            if (actual_status != expected_status) ++status_mismatches;
            const std::size_t mismatch = first_mismatch(expected, actual);
            if (mismatch != sizeof(OfficialStatePod)) {
                ++mismatches;
                if (first_mismatch_seed == 0) {
                    first_mismatch_seed = seed;
                    first_mismatch_byte = mismatch;
                    first_expected = reinterpret_cast<const std::uint8_t*>(&expected)[mismatch];
                    first_actual = reinterpret_cast<const std::uint8_t*>(&actual)[mismatch];
                    first_expected_state = expected;
                    first_actual_state = actual;
                }
            }
            OfficialSeededResetFixtureRecord record{};
            record.seed = seed;
            record.expected_status = expected_status;
            record.expected = expected;
            fixture.write(reinterpret_cast<const char*>(&record), sizeof(record));
        }
        if (!fixture) throw std::runtime_error("failed while writing fixture");
        const bool passed = mismatches == 0 && status_mismatches == 0;
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"seed_start\":" << seed_start
            << ",\"seed_count\":" << seed_count
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"record_bytes\":" << sizeof(OfficialSeededResetFixtureRecord)
            << ",\"state_mismatches\":" << mismatches
            << ",\"status_mismatches\":" << status_mismatches;
        if (mismatches != 0) {
            std::cout
                << ",\"first_mismatch_seed\":" << first_mismatch_seed
                << ",\"first_mismatch_byte\":" << first_mismatch_byte
                << ",\"first_expected\":" << static_cast<int>(first_expected)
                << ",\"first_actual\":" << static_cast<int>(first_actual)
                << ",\"first_expected_targets\":"
                << first_expected_state.targets.count
                << ",\"first_actual_targets\":"
                << first_actual_state.targets.count
                << ",\"first_expected_target_card_ids\":";
            write_target_card_ids(std::cout, first_expected_state);
            std::cout << ",\"first_actual_target_card_ids\":";
            write_target_card_ids(std::cout, first_actual_state);
            std::cout << ",\"first_expected_hand_card_ids\":";
            write_card_ids(
                std::cout,
                first_expected_state,
                first_expected_state.players[official_active_player(
                    first_expected_state)].hand);
            std::cout << ",\"first_actual_hand_card_ids\":";
            write_card_ids(
                std::cout,
                first_actual_state,
                first_actual_state.players[official_active_player(
                    first_actual_state)].hand);
            std::cout << ",\"first_expected_option_types\":";
            write_option_types(std::cout, first_expected_state);
            std::cout << ",\"first_actual_option_types\":";
            write_option_types(std::cout, first_actual_state);
        }
        std::cout << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
