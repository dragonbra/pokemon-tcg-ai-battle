#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptcg_cuda/official_effect_interpreter_pod.cuh"
#include "ptcg_cuda/official_trigger_resolver_pod.cuh"

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

__host__ __device__ void run_selection_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 101, seed);
    state->first_player = 0;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 72, 0, OfficialArea::kActive);
    add_card(state, 11, 1242, 0, OfficialArea::kDeck);
    add_card(state, 12, 7, 0, OfficialArea::kDeck);

    const OfficialEffectInterpreterResult begin = official_begin_skill_effects(
        state,
        rules,
        24,
        official_pod_area_ref(state, OfficialCardRefPod{10}),
        0);
    if (begin != OfficialEffectInterpreterResult::kNeedsAction
        || state->select_type != static_cast<std::uint8_t>(OfficialSelectTypeId::kCard)
        || state->select_min != 0
        || state->select_max != 1
        || state->options.count != 1
        || state->options.values[0].resolved_card != 11) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 2401);
        return;
    }
    const std::uint16_t action = 0;
    const OfficialEffectInterpreterResult resumed = official_apply_effect_action(
        state, rules, &action, 1);
    if (resumed != OfficialEffectInterpreterResult::kComplete
        || state->effect_interpreter.active != 0
        || state->effect_interpreter.awaiting_selection != 0
        || state->players[0].hand.count != 1
        || state->players[0].hand.values[0].index != 11
        || state->players[0].deck.count != 1) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 2402);
    }
}

__host__ __device__ void run_condition_false_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 102, seed);
    state->first_player = 0;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 72, 0, OfficialArea::kActive);
    const OfficialEffectInterpreterResult result = official_begin_skill_effects(
        state,
        rules,
        24,
        official_pod_area_ref(state, OfficialCardRefPod{10}),
        0);
    if (result != OfficialEffectInterpreterResult::kComplete
        || state->effect_interpreter.active != 0
        || state->options.count != 0
        || state->players[0].hand.count != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 2403);
    }
}

__global__ void run_kernel(
    OfficialStatePod* states,
    const std::uint8_t* rule_pack,
    std::uint64_t seed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
        run_selection_scenario(&states[0], rules, seed);
        run_condition_false_scenario(&states[1], rules, seed + 1);
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
                "usage: official_effect_interpreter_smoke <official_rules.bin>");
        }
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        if (rule_pack.size() < sizeof(OfficialRulePackHeader)) {
            throw std::runtime_error("rule pack is truncated");
        }
        const OfficialRulePackView host_rules = make_official_rule_pack_view(
            rule_pack.data());
        if (std::memcmp(host_rules.header->magic, "PTCGRUL1", 8) != 0) {
            throw std::runtime_error("rule pack magic mismatch");
        }

        constexpr std::uint64_t seed = 20260730ULL;
        OfficialStatePod cpu_states[2]{};
        run_selection_scenario(&cpu_states[0], host_rules, seed);
        run_condition_false_scenario(&cpu_states[1], host_rules, seed + 1);

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
        check_cuda(cudaGetLastError(), "launch effect interpreter smoke");

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
            << ",\"state_abi\":" << kOfficialStateAbiVersion
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"cpu_digest\":" << digest(cpu_states, 2)
            << ",\"gpu_digest\":" << digest(gpu_states, 2)
            << ",\"selection_steps\":" << gpu_states[0].interpreter_steps
            << ",\"condition_false_steps\":" << gpu_states[1].interpreter_steps
            << ",\"selected_card\":" << gpu_states[0].players[0].hand.values[0].index
            << ",\"selection_error\":" << gpu_states[0].error
            << ",\"condition_error\":" << gpu_states[1].error
            << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
