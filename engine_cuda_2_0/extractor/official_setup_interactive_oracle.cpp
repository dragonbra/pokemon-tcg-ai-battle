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
#include "ptcg_cuda/official_setup_interactive_pod.cuh"

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

struct Mismatch {
    std::size_t offset = sizeof(OfficialStatePod);
    std::uint8_t expected = 0;
    std::uint8_t actual = 0;
};

Mismatch compare_state(
    const ApiData& official,
    const OfficialStatePod& pod,
    std::uint64_t seed) {
    OfficialStatePod expected{};
    const std::uint64_t rng_draws = count_rng_draws(seed, official.game.rng);
    const OfficialBridgeResult bridge = bridge_official_state(
        official.state,
        OfficialStateBridgeContext{seed, rng_draws, true},
        &expected);
    if (!bridge) {
        throw std::runtime_error(
            "official state bridge failed: "
            + std::to_string(static_cast<std::int32_t>(bridge.error))
            + "." + std::to_string(bridge.detail));
    }
    const auto* left = reinterpret_cast<const std::uint8_t*>(&expected);
    const auto* right = reinterpret_cast<const std::uint8_t*>(&pod);
    for (std::size_t index = 0; index < sizeof(OfficialStatePod); ++index) {
        if (left[index] != right[index]) {
            return {index, left[index], right[index]};
        }
    }
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::runtime_error(
                "usage: official_setup_interactive_oracle RULES DECK0 DECK1 "
                "SEED_START SEED_COUNT");
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
        std::uint64_t decisions = 0;
        constexpr std::uint32_t policy_count = 2;
        std::uint32_t max_decisions = 0;
        for (std::uint32_t policy = 0; policy < policy_count; ++policy) {
          for (std::uint32_t offset = 0; offset < seed_count; ++offset) {
            const std::uint64_t seed = seed_start + offset;
            ApiData official;
            initialize_official_battle(&official, decks, seed);
            OfficialStatePod pod{};
            OfficialFlowStatus status = official_seeded_setup_interactive_state(
                &pod, rules, decks.data(), seed, seed);
            std::uint32_t seed_decisions = 0;
            while (true) {
                if (status != OfficialFlowStatus::kNeedsAction) {
                    throw std::runtime_error(
                        "interactive POD did not yield a decision at seed "
                        + std::to_string(seed));
                }
                const Mismatch mismatch = compare_state(official, pod, seed);
                if (mismatch.offset != sizeof(OfficialStatePod)) {
                    std::cout
                        << "{\"passed\":false,\"seed\":" << seed
                        << ",\"decision\":" << seed_decisions
                        << ",\"select_context\":"
                        << static_cast<int>(official.state.selectContext)
                        << ",\"first_mismatch_byte\":" << mismatch.offset
                        << ",\"expected\":" << static_cast<int>(mismatch.expected)
                        << ",\"actual\":" << static_cast<int>(mismatch.actual)
                        << "}\n";
                    return 1;
                }
                if (official.state.selectContext == SelectContext::Main) break;
                if (++seed_decisions > 128) {
                    throw std::runtime_error("interactive setup decision limit exceeded");
                }
                std::array<int, kOfficialOptionCapacity> official_selected{};
                std::array<std::uint16_t, kOfficialOptionCapacity> pod_selected{};
                const bool max_cardinality = policy == 1
                    && official.state.selectContext
                        == SelectContext::SetupBenchPokemon;
                const int selected_count_int = max_cardinality
                    ? official.state.selectMax : official.state.selectMin;
                for (int index = 0; index < selected_count_int; ++index) {
                    const int selected = policy == 0
                        ? index
                        : static_cast<int>(official.state.options.size())
                            - selected_count_int + index;
                    official_selected[index] = selected;
                    pod_selected[index] = static_cast<std::uint16_t>(selected);
                }
                const std::uint16_t selected_count = static_cast<std::uint16_t>(
                    selected_count_int);
                const int error = ApiSelect(
                    &official, official_selected.data(), selected_count);
                if (error != 0 || official.state.isFinish()) {
                    throw std::runtime_error("official setup selection failed");
                }
                status = official_setup_apply_action(
                    &pod, rules, pod_selected.data(), selected_count);
            }
            decisions += seed_decisions;
            if (seed_decisions > max_decisions) max_decisions = seed_decisions;
          }
        }
        std::cout
            << "{\"passed\":true,\"seed_start\":" << seed_start
            << ",\"seed_count\":" << seed_count
            << ",\"policy_count\":" << policy_count
            << ",\"decision_boundaries_compared\":"
            << decisions + seed_count * policy_count
            << ",\"setup_actions_applied\":" << decisions
            << ",\"max_setup_actions\":" << max_decisions
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
