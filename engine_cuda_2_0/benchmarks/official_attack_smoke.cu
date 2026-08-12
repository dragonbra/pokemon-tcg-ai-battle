#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptcg_cuda/official_attack_pod.cuh"

namespace {

using namespace ptcg::cuda_engine;

constexpr std::size_t kScenarioCount = 70;
constexpr std::size_t kThreadsPerBlock = 64;
constexpr std::int32_t kAttackerCardId = 39;
constexpr std::int32_t kTargetCardId = 24;
constexpr std::int32_t kBasicEnergyCardId = 1;
constexpr std::int32_t kSimpleAttackId = 31;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::uint8_t> read_binary(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("failed to open rule pack");
    const std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("failed to read rule pack");
    }
    return bytes;
}

__host__ __device__ void add_card(
    OfficialStatePod* state,
    std::uint16_t instance,
    std::int32_t card_id,
    std::int32_t player,
    OfficialArea area) {
    OfficialCardStatePod* card = &state->cards[instance];
    card->card_id = card_id;
    card->player = static_cast<std::int8_t>(player);
    card->area = static_cast<std::uint8_t>(area);
    card->move_counter = ++state->move_counter;
    official_pod_push_zone_card(
        state, player, area, OfficialCardRefPod{instance});
}

__host__ __device__ void run_simple_damage_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t episode,
    std::uint64_t seed,
    std::int32_t attacker_card_id = kAttackerCardId,
    std::int32_t target_card_id = kTargetCardId,
    std::int32_t energy_card_id = kBasicEnergyCardId,
    std::int32_t attack_id = kSimpleAttackId,
    std::int32_t expected_target_damage = 30,
    std::int32_t expected_self_damage = 0,
    bool confused = false,
    bool target_no_damage_coin = false) {
    official_pod_reset(state, episode, seed);
    state->first_player = 0;
    state->turn = 3;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);

    add_card(state, 10, attacker_card_id, 0, OfficialArea::kActive);
    add_card(state, 11, energy_card_id, 0, OfficialArea::kEnergy);
    state->cards[11].attach_move_counter = state->cards[10].move_counter;
    add_card(state, 20, target_card_id, 1, OfficialArea::kActive);
    if (target_no_damage_coin) {
        add_card(state, 21, 7, 1, OfficialArea::kEnergy);
        state->cards[21].attach_move_counter = state->cards[20].move_counter;
    }
    add_card(state, 30, kBasicEnergyCardId, 0, OfficialArea::kPrize);
    add_card(state, 31, kBasicEnergyCardId, 1, OfficialArea::kPrize);
    add_card(state, 40, kBasicEnergyCardId, 0, OfficialArea::kDeck);
    add_card(state, 41, kBasicEnergyCardId, 1, OfficialArea::kDeck);

    if (confused) {
        official_pod_set_bad_status(
            &state->players[0], OfficialBadStatus::kConfused);
    }
    const OfficialAttackResult result = official_begin_attack(
        state, rules, OfficialCardRefPod{10}, attack_id);
    if (result != OfficialAttackResult::kComplete
        || state->cards[20].damage != expected_target_damage
        || state->cards[10].damage != expected_self_damage
        || state->turn != 4
        || state->players[1].hand.count != 1
        || state->players[1].hand.values[0].index != 41
        || state->current_attack_id != 0
        || state->attack_flow_stage != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7101);
    }
}

__host__ __device__ void prepare_flow_case(
    OfficialStatePod* state,
    std::uint64_t episode,
    std::uint64_t seed,
    std::int32_t attacker_card_id,
    std::int32_t target_card_id,
    std::int32_t energy_card_id) {
    official_pod_reset(state, episode, seed);
    state->first_player = 0;
    state->turn = 3;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, attacker_card_id, 0, OfficialArea::kActive);
    add_card(state, 11, energy_card_id, 0, OfficialArea::kEnergy);
    state->cards[11].attach_move_counter = state->cards[10].move_counter;
    add_card(state, 20, target_card_id, 1, OfficialArea::kActive);
    add_card(state, 30, 1, 0, OfficialArea::kPrize);
    add_card(state, 31, 1, 1, OfficialArea::kPrize);
    add_card(state, 40, 1, 0, OfficialArea::kDeck);
    add_card(state, 41, 1, 1, OfficialArea::kDeck);
}

__host__ __device__ void run_copy_enemy_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t episode) {
    prepare_flow_case(state, episode, 59, 615, 39, 1);
    for (std::uint16_t ref = 12; ref <= 13; ++ref) {
        add_card(state, ref, 1, 0, OfficialArea::kEnergy);
        state->cards[ref].attach_move_counter = state->cards[10].move_counter;
    }
    OfficialAttackResult result = official_begin_attack(
        state, rules, OfficialCardRefPod{10}, 886);
    if (result != OfficialAttackResult::kNeedsAction
        || state->options.count != 2
        || state->options.values[0].params[0] != 31
        || state->options.values[0].params[1] != 886) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7201);
        return;
    }
    const std::uint16_t selected = 0;
    result = official_resume_attack(state, rules, &selected, 1);
    if (result != OfficialAttackResult::kComplete
        || state->cards[20].damage != 30
        || state->turn_histories[1].attack_id != 886
        || state->turn != 4) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7202);
    }
}

__host__ __device__ void run_double_attack_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t episode) {
    prepare_flow_case(state, episode, 61, 93, 24, 1);
    add_card(state, 22, 24, 0, OfficialArea::kBench);
    add_card(state, 50, 1245, 0, OfficialArea::kStadium);
    OfficialAttackResult result = official_begin_attack(
        state, rules, OfficialCardRefPod{10}, 115);
    if (result != OfficialAttackResult::kNeedsAction
        || state->options.count != 1
        || state->options.values[0].params[0] != 115
        || state->cards[20].damage != 20) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7301);
        return;
    }
    const std::uint16_t selected = 0;
    result = official_resume_attack(state, rules, &selected, 1);
    if (result != OfficialAttackResult::kComplete
        || state->cards[20].damage != 40
        || state->turn != 4) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7302);
    }
}

__host__ __device__ void run_knockout_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t episode) {
    prepare_flow_case(state, episode, 53, 39, 160, 1);
    add_card(state, 22, 24, 1, OfficialArea::kBench);
    add_card(state, 32, 1, 0, OfficialArea::kPrize);
    OfficialAttackResult result = official_begin_attack(
        state, rules, OfficialCardRefPod{10}, 31);
    if (result != OfficialAttackResult::kNeedsAction) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7401);
        return;
    }
    const std::uint16_t selected = 0;
    result = official_resume_attack(state, rules, &selected, 1);
    if (result != OfficialAttackResult::kNeedsAction) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7402);
        return;
    }
    result = official_resume_attack(state, rules, &selected, 1);
    if (result != OfficialAttackResult::kComplete
        || state->players[1].active.count != 1
        || state->players[1].active.values[0].index != 22
        || state->players[0].hand.count != 1
        || state->players[0].prize.count != 1
        || state->players[1].trash.count != 1
        || state->turn != 4) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7403);
    }
}

__host__ __device__ void run_post_effect_selection_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t episode) {
    prepare_flow_case(state, episode, 67, 22, 24, 6);
    add_card(state, 22, 38, 1, OfficialArea::kBench);
    OfficialAttackResult result = official_begin_attack(
        state, rules, OfficialCardRefPod{10}, 3);
    if (result != OfficialAttackResult::kNeedsAction
        || state->cards[20].damage != 20) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7501);
        return;
    }
    const std::uint16_t selected = 0;
    result = official_resume_attack(state, rules, &selected, 1);
    if (result != OfficialAttackResult::kComplete
        || state->players[1].active.values[0].index != 22
        || state->cards[20].damage != 20
        || state->turn != 4) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7502);
    }
}

__host__ __device__ void run_deck_top_attack_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t episode) {
    prepare_flow_case(state, episode, 71, 163, 24, 5);
    add_card(state, 12, 1, 0, OfficialArea::kEnergy);
    state->cards[12].attach_move_counter = state->cards[10].move_counter;
    add_card(state, 42, 39, 0, OfficialArea::kDeck);
    OfficialAttackResult result = official_begin_attack(
        state, rules, OfficialCardRefPod{10}, 213);
    if (result != OfficialAttackResult::kNeedsAction
        || state->options.count != 2
        || state->players[0].trash.count != 1) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7601);
        return;
    }
    const std::uint16_t selected = 0;
    result = official_resume_attack(state, rules, &selected, 1);
    if (result != OfficialAttackResult::kComplete
        || state->cards[20].damage != 30
        || state->turn_histories[1].attack_id != 213
        || state->turn != 4) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7602);
    }
}

__host__ __device__ void run_deck_top_supporter_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t episode) {
    prepare_flow_case(state, episode, 73, 660, 24, 1);
    for (std::uint16_t ref = 42; ref <= 44; ++ref) {
        add_card(state, ref, 1, 0, OfficialArea::kDeck);
    }
    add_card(state, 60, 1224, 0, OfficialArea::kDeck);
    const OfficialAttackResult result = official_begin_attack(
        state, rules, OfficialCardRefPod{10}, 955);
    if (result != OfficialAttackResult::kComplete
        || state->players[0].hand.count != 3
        || state->players[0].trash.count != 1
        || state->players[0].deck.count != 1
        || state->turn != 4) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7701);
    }
}

__host__ __device__ void run_enemy_deck_top10_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t episode) {
    prepare_flow_case(state, episode, 79, 471, 24, 1);
    add_card(state, 12, 1, 0, OfficialArea::kEnergy);
    state->cards[12].attach_move_counter = state->cards[10].move_counter;
    for (std::uint16_t ref = 50; ref < 60; ++ref) {
        add_card(
            state, ref, (ref % 2 == 0) ? 39 : 1, 1, OfficialArea::kDeck);
    }
    OfficialAttackResult result = official_begin_attack(
        state, rules, OfficialCardRefPod{10}, 665);
    if (result != OfficialAttackResult::kNeedsAction
        || state->options.count != 2
        || state->looking.count != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7801);
        return;
    }
    const std::uint16_t selected = 0;
    result = official_resume_attack(state, rules, &selected, 1);
    if (result != OfficialAttackResult::kComplete
        || state->cards[20].damage != 30
        || state->turn_histories[1].attack_id != 665
        || state->turn != 4) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 7802);
    }
}

__global__ void run_kernel(
    OfficialStatePod* states,
    const std::uint8_t* rule_pack,
    std::uint64_t seed) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kScenarioCount) return;
    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
    switch (index % 14) {
        case 0:
            run_simple_damage_scenario(&states[index], rules, 700 + index, seed + index);
            break;
        case 1:
            run_simple_damage_scenario(
                &states[index], rules, 700 + index, 47, 54, 38, 5, 57, 40);
            break;
        case 2:
            run_simple_damage_scenario(
                &states[index], rules, 700 + index, 0, 39, 24, 1, 31, 30, 0, true);
            break;
        case 3:
            run_simple_damage_scenario(
                &states[index], rules, 700 + index, 1, 39, 24, 1, 31, 0, 30, true);
            break;
        case 4:
            run_simple_damage_scenario(
                &states[index], rules, 700 + index, 0, 39, 970, 1, 31, 0, 0,
                false, true);
            break;
        case 5:
            run_simple_damage_scenario(
                &states[index], rules, 700 + index, 1, 39, 970, 1, 31, 30, 0,
                false, true);
            break;
        case 6:
            run_simple_damage_scenario(
                &states[index], rules, 700 + index, 0, 625, 970, 1, 901, 40, 0,
                false, true);
            break;
        case 7:
            run_copy_enemy_scenario(&states[index], rules, 700 + index);
            break;
        case 8:
            run_double_attack_scenario(&states[index], rules, 700 + index);
            break;
        case 9:
            run_knockout_scenario(&states[index], rules, 700 + index);
            break;
        case 10:
            run_post_effect_selection_scenario(&states[index], rules, 700 + index);
            break;
        case 11:
            run_deck_top_attack_scenario(&states[index], rules, 700 + index);
            break;
        case 12:
            run_deck_top_supporter_scenario(&states[index], rules, 700 + index);
            break;
        default:
            run_enemy_deck_top10_scenario(&states[index], rules, 700 + index);
            break;
    }
}

std::uint64_t digest(const OfficialStatePod* states, std::size_t count) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(states);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < sizeof(OfficialStatePod) * count; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: official_attack_smoke <official_rules.bin>");
        }
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        const OfficialRulePackView host_rules = make_official_rule_pack_view(
            rule_pack.data());
        if (rule_pack.size() < sizeof(OfficialRulePackHeader)
            || std::memcmp(host_rules.header->magic, "PTCGRUL1", 8) != 0) {
            throw std::runtime_error("invalid rule pack");
        }

        constexpr std::uint64_t seed = 20260730ULL;
        std::vector<OfficialStatePod> cpu_states(kScenarioCount);
        for (std::size_t index = 0; index < kScenarioCount; ++index) {
            switch (index % 14) {
                case 0:
                    run_simple_damage_scenario(
                        &cpu_states[index], host_rules, 700 + index, seed + index);
                    break;
                case 1:
                    run_simple_damage_scenario(
                        &cpu_states[index], host_rules, 700 + index, 47,
                        54, 38, 5, 57, 40);
                    break;
                case 2:
                    run_simple_damage_scenario(
                        &cpu_states[index], host_rules, 700 + index, 0,
                        39, 24, 1, 31, 30, 0, true);
                    break;
                case 3:
                    run_simple_damage_scenario(
                        &cpu_states[index], host_rules, 700 + index, 1,
                        39, 24, 1, 31, 0, 30, true);
                    break;
                case 4:
                    run_simple_damage_scenario(
                        &cpu_states[index], host_rules, 700 + index, 0,
                        39, 970, 1, 31, 0, 0, false, true);
                    break;
                case 5:
                    run_simple_damage_scenario(
                        &cpu_states[index], host_rules, 700 + index, 1,
                        39, 970, 1, 31, 30, 0, false, true);
                    break;
                case 6:
                    run_simple_damage_scenario(
                        &cpu_states[index], host_rules, 700 + index, 0,
                        625, 970, 1, 901, 40, 0, false, true);
                    break;
                case 7:
                    run_copy_enemy_scenario(
                        &cpu_states[index], host_rules, 700 + index);
                    break;
                case 8:
                    run_double_attack_scenario(
                        &cpu_states[index], host_rules, 700 + index);
                    break;
                case 9:
                    run_knockout_scenario(
                        &cpu_states[index], host_rules, 700 + index);
                    break;
                case 10:
                    run_post_effect_selection_scenario(
                        &cpu_states[index], host_rules, 700 + index);
                    break;
                case 11:
                    run_deck_top_attack_scenario(
                        &cpu_states[index], host_rules, 700 + index);
                    break;
                case 12:
                    run_deck_top_supporter_scenario(
                        &cpu_states[index], host_rules, 700 + index);
                    break;
                default:
                    run_enemy_deck_top10_scenario(
                        &cpu_states[index], host_rules, 700 + index);
                    break;
            }
        }

        std::uint8_t* device_rules = nullptr;
        OfficialStatePod* device_states = nullptr;
        const std::size_t state_bytes = sizeof(OfficialStatePod) * kScenarioCount;
        check_cuda(cudaMalloc(&device_rules, rule_pack.size()), "cudaMalloc rules");
        check_cuda(cudaMalloc(&device_states, state_bytes), "cudaMalloc states");
        check_cuda(cudaMemcpy(
            device_rules, rule_pack.data(), rule_pack.size(), cudaMemcpyHostToDevice),
            "copy rules");
        check_cuda(cudaMemset(device_states, 0, state_bytes), "clear states");
        const std::size_t block_count =
            (kScenarioCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
        run_kernel<<<block_count, kThreadsPerBlock>>>(
            device_states, device_rules, seed);
        check_cuda(cudaGetLastError(), "launch attack smoke");
        std::vector<OfficialStatePod> gpu_states(kScenarioCount);
        check_cuda(cudaMemcpy(
            gpu_states.data(), device_states, state_bytes, cudaMemcpyDeviceToHost),
            "copy states");
        check_cuda(cudaFree(device_states), "cudaFree states");
        check_cuda(cudaFree(device_rules), "cudaFree rules");

        std::size_t errors = 0;
        for (const OfficialStatePod& state : gpu_states) {
            if (state.error != 0) ++errors;
        }
        const bool equal = std::memcmp(
            cpu_states.data(), gpu_states.data(), state_bytes) == 0;
        const bool passed = equal && errors == 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"device\":\"" << properties.name << "\""
            << ",\"state_abi\":" << kOfficialStateAbiVersion
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"scenario_count\":" << kScenarioCount
            << ",\"cpu_digest\":" << digest(cpu_states.data(), kScenarioCount)
            << ",\"gpu_digest\":" << digest(gpu_states.data(), kScenarioCount)
            << ",\"target_damage\":" << gpu_states[0].cards[20].damage
            << ",\"next_turn\":" << gpu_states[0].turn
            << ",\"errors\":" << errors << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
