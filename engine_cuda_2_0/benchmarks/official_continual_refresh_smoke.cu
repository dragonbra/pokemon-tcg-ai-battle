#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptcg_cuda/official_continual_refresh_pod.cuh"

namespace {

using namespace ptcg::cuda_engine;

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
    official_pod_push_zone_card(state, player, area, OfficialCardRefPod{instance});
}

__host__ __device__ void run_not_stack_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 201, seed);
    state->first_player = 0;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 262, 0, OfficialArea::kActive);
    add_card(state, 11, 262, 0, OfficialArea::kBench);
    add_card(state, 12, 25, 0, OfficialArea::kBench);
    const OfficialContinualRefreshResult result = official_refresh_continual_effects(
        state, rules);
    if (result != OfficialContinualRefreshResult::kApplied
        || state->cards[10].hp_change != 40
        || state->cards[11].hp_change != 40
        || state->cards[12].hp_change != 40) {
        official_pod_fail(state, OfficialPodError::kUnsupportedEffect, 9601);
    }
}

__host__ __device__ void run_no_ability_reorder_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 202, seed);
    state->first_player = 0;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 37, 0, OfficialArea::kActive);
    add_card(state, 11, 56, 1, OfficialArea::kActive);
    const OfficialContinualRefreshResult result = official_refresh_continual_effects(
        state, rules);
    if (result != OfficialContinualRefreshResult::kApplied
        || !official_continual_flag(state->cards[10], 0)
        || state->cards[10].continual_state[4] != 1ULL
        || state->continual_refresh_passes != 2) {
        official_pod_fail(state, OfficialPodError::kUnsupportedEffect, 9602);
    }
}

__global__ void run_kernel(
    OfficialStatePod* states,
    const std::uint8_t* rule_pack,
    std::uint64_t seed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
        run_not_stack_scenario(&states[0], rules, seed);
        run_no_ability_reorder_scenario(&states[1], rules, seed + 1);
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
            throw std::runtime_error(
                "usage: official_continual_refresh_smoke <official_rules.bin>");
        }
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        const OfficialRulePackView host_rules = make_official_rule_pack_view(
            rule_pack.data());
        if (rule_pack.size() < sizeof(OfficialRulePackHeader)
            || std::memcmp(host_rules.header->magic, "PTCGRUL1", 8) != 0) {
            throw std::runtime_error("invalid rule pack");
        }

        constexpr std::uint64_t seed = 20260730ULL;
        OfficialStatePod cpu_states[2]{};
        run_not_stack_scenario(&cpu_states[0], host_rules, seed);
        run_no_ability_reorder_scenario(&cpu_states[1], host_rules, seed + 1);

        std::uint8_t* device_rules = nullptr;
        OfficialStatePod* device_states = nullptr;
        check_cuda(cudaMalloc(&device_rules, rule_pack.size()), "cudaMalloc rules");
        check_cuda(cudaMalloc(&device_states, sizeof(cpu_states)), "cudaMalloc states");
        check_cuda(cudaMemcpy(
            device_rules,
            rule_pack.data(),
            rule_pack.size(),
            cudaMemcpyHostToDevice), "copy rules");
        check_cuda(cudaMemset(device_states, 0, sizeof(cpu_states)), "clear states");
        run_kernel<<<1, 1>>>(device_states, device_rules, seed);
        check_cuda(cudaGetLastError(), "launch continual refresh smoke");
        OfficialStatePod gpu_states[2]{};
        check_cuda(cudaMemcpy(
            gpu_states,
            device_states,
            sizeof(gpu_states),
            cudaMemcpyDeviceToHost), "copy states");
        check_cuda(cudaFree(device_states), "cudaFree states");
        check_cuda(cudaFree(device_rules), "cudaFree rules");

        const bool equal = std::memcmp(cpu_states, gpu_states, sizeof(cpu_states)) == 0;
        const bool passed = equal
            && gpu_states[0].error == 0
            && gpu_states[1].error == 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"device\":\"" << properties.name << "\""
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"cpu_digest\":" << digest(cpu_states, 2)
            << ",\"gpu_digest\":" << digest(gpu_states, 2)
            << ",\"not_stack_hp_change\":" << gpu_states[0].cards[10].hp_change
            << ",\"reordered_flags\":" << gpu_states[1].cards[10].continual_state[4]
            << ",\"refresh_passes\":"
            << static_cast<int>(gpu_states[1].continual_refresh_passes)
            << ",\"suppressed_skill_order\":" << gpu_states[1].cards[10].skill_order
            << ",\"suppressor_skill_order\":" << gpu_states[1].cards[11].skill_order
            << ",\"not_stack_error\":" << gpu_states[0].error
            << ",\"reorder_error\":" << gpu_states[1].error
            << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
