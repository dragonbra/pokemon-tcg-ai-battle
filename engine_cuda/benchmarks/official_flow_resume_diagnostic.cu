#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptcg_cuda/official_flow_dispatch_pod.cuh"
#include "ptcg_cuda/official_flow_fixture.h"
#include "ptcg_cuda/official_runtime.h"

namespace {

using namespace ptcg::cuda_engine;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::uint8_t> read_binary(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error(std::string("cannot open ") + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

__device__ OfficialFlowStatus resume_attack_branch(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialActionPod& action) {
    const bool return_to_main =
        (state->attack_flow_flags & kOfficialAttackReturnToMainFlag) != 0;
    const OfficialAttackResult result = official_resume_attack(
        state, rules, action.option_indices, action.count);
    if (result == OfficialAttackResult::kError) return OfficialFlowStatus::kError;
    if (result == OfficialAttackResult::kNeedsAction) {
        return official_yield_decision(state);
    }
    if (return_to_main) {
        state->attack_flow_flags &= static_cast<std::uint8_t>(
            ~kOfficialAttackReturnToMainFlag);
        const OfficialMainResult main = official_main_after_flow(state, rules);
        if (main == OfficialMainResult::kError) return OfficialFlowStatus::kError;
        if (main == OfficialMainResult::kNeedsAction) {
            return official_yield_decision(state);
        }
    }
    return official_flow_status(*state);
}

__global__ void resume_kernel(
    OfficialStatePod* states,
    const std::uint8_t* rule_pack,
    OfficialActionPod action,
    std::uint8_t mode,
    std::uint8_t* status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    OfficialStatePod* state = &states[mode];
    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
    OfficialFlowStatus result = OfficialFlowStatus::kError;
    if (mode == 0) {
        const OfficialAttackResult attack = official_resume_attack(
            state, rules, action.option_indices, action.count);
        result = attack == OfficialAttackResult::kError
            ? OfficialFlowStatus::kError
            : (attack == OfficialAttackResult::kNeedsAction
                ? OfficialFlowStatus::kNeedsAction
                : official_flow_status(*state));
    } else if (mode == 1) {
        result = resume_attack_branch(state, rules, action);
    } else {
        result = official_apply_pending_action(
            state, rules, action.option_indices, action.count);
    }
    status[mode] = static_cast<std::uint8_t>(result);
}

__device__ __noinline__ std::uint8_t apply_action_by_value(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialActionPod action) {
    return static_cast<std::uint8_t>(official_apply_pending_action(
        state, rules, action.option_indices, action.count));
}

__device__ __noinline__ std::uint8_t apply_action_by_pointer(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialActionPod* action) {
    return static_cast<std::uint8_t>(official_apply_pending_action(
        state, rules, action->option_indices, action->count));
}

__global__ void production_shape_kernel(
    OfficialStatePod* states,
    std::uint32_t batch_size,
    const std::uint8_t* rule_pack,
    const OfficialActionPod* actions,
    std::uint8_t mode,
    std::uint8_t* statuses) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) return;
    OfficialStatePod* state = &states[mode + env];
    if (state->abi_version != kOfficialStateAbiVersion) {
        official_pod_fail(
            state,
            OfficialPodError::kInvalidAction,
            static_cast<std::int32_t>(state->abi_version));
        statuses[mode + env] = static_cast<std::uint8_t>(
            OfficialFlowStatus::kError);
        return;
    }
    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
    if (mode == 3) {
        const OfficialActionPod action = actions[env];
        if (action.count > kOfficialOptionCapacity) return;
        statuses[mode + env] = static_cast<std::uint8_t>(
            official_apply_pending_action(
                state, rules, action.option_indices, action.count));
    } else if (mode == 4) {
        const OfficialActionPod action = actions[env];
        if (action.count > kOfficialOptionCapacity) return;
        statuses[mode + env] = apply_action_by_value(state, rules, action);
    } else {
        if (actions[env].count > kOfficialOptionCapacity) return;
        statuses[mode + env] = apply_action_by_pointer(
            state, rules, &actions[env]);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error(
                "usage: official_flow_resume_diagnostic <rules> <fixture>");
        }
        const std::vector<std::uint8_t> rules = read_binary(argv[1]);
        const std::vector<std::uint8_t> fixture = read_binary(argv[2]);
        if (fixture.size() != sizeof(OfficialFlowFixtureHeader)
                + 2 * sizeof(OfficialStatePod)) {
            throw std::runtime_error("invalid fixture size");
        }
        OfficialFlowFixtureHeader header{};
        OfficialStatePod pending{};
        OfficialStatePod expected{};
        std::memcpy(&header, fixture.data(), sizeof(header));
        std::memcpy(&pending, fixture.data() + sizeof(header), sizeof(pending));
        std::memcpy(
            &expected,
            fixture.data() + sizeof(header) + sizeof(pending),
            sizeof(expected));
        OfficialActionPod action{};
        action.count = header.action_count;
        for (std::uint16_t index = 0; index < action.count; ++index) {
            action.option_indices[index] = header.option_indices[index];
        }

        check_cuda(
            cudaDeviceSetLimit(cudaLimitStackSize, kOfficialMinimumDeviceStackBytes),
            "configure device stack");
        std::size_t device_stack_bytes = 0;
        check_cuda(
            cudaDeviceGetLimit(&device_stack_bytes, cudaLimitStackSize),
            "read device stack");
        if (device_stack_bytes < kOfficialMinimumDeviceStackBytes) {
            throw std::runtime_error("device stack limit is below runtime minimum");
        }

        OfficialStatePod host_states[6] = {
            pending, pending, pending, pending, pending, pending};
        OfficialStatePod* device_states = nullptr;
        std::uint8_t* device_rules = nullptr;
        std::uint8_t* device_status = nullptr;
        OfficialActionPod* device_action = nullptr;
        check_cuda(cudaMalloc(&device_states, sizeof(host_states)), "allocate states");
        check_cuda(cudaMalloc(&device_rules, rules.size()), "allocate rules");
        check_cuda(cudaMalloc(&device_status, 6), "allocate status");
        check_cuda(cudaMalloc(&device_action, sizeof(action)), "allocate action");
        check_cuda(cudaMemcpy(
            device_states, host_states, sizeof(host_states), cudaMemcpyHostToDevice),
            "upload states");
        check_cuda(cudaMemcpy(
            device_rules, rules.data(), rules.size(), cudaMemcpyHostToDevice),
            "upload rules");
        check_cuda(cudaMemset(device_status, 0, 6), "clear status");
        check_cuda(cudaMemcpy(
            device_action, &action, sizeof(action), cudaMemcpyHostToDevice),
            "upload action");

        std::uint8_t statuses[6]{};
        std::uint8_t completed = 0;
        for (std::uint8_t mode = 0; mode < 3; ++mode) {
            resume_kernel<<<1, 128>>>(
                device_states, device_rules, action, mode, device_status);
            const cudaError_t sync = cudaDeviceSynchronize();
            if (sync != cudaSuccess) {
                std::cout << "{\"device_stack_bytes\":" << device_stack_bytes
                          << ",\"completed_modes\":" << static_cast<int>(completed)
                          << ",\"failed_mode\":" << static_cast<int>(mode)
                          << ",\"cuda_error\":\"" << cudaGetErrorString(sync)
                          << "\"}\n";
                return 3;
            }
            ++completed;
        }
        for (std::uint8_t mode = 3; mode < 6; ++mode) {
            production_shape_kernel<<<1, 128>>>(
                device_states,
                1,
                device_rules,
                device_action,
                mode,
                device_status);
            const cudaError_t sync = cudaDeviceSynchronize();
            if (sync != cudaSuccess) {
                std::cout << "{\"device_stack_bytes\":" << device_stack_bytes
                          << ",\"completed_modes\":" << static_cast<int>(completed)
                          << ",\"failed_mode\":" << static_cast<int>(mode)
                          << ",\"cuda_error\":\"" << cudaGetErrorString(sync)
                          << "\"}\n";
                return 3;
            }
            ++completed;
        }
        check_cuda(cudaMemcpy(
            host_states, device_states, sizeof(host_states), cudaMemcpyDeviceToHost),
            "download states");
        check_cuda(cudaMemcpy(
            statuses, device_status, sizeof(statuses), cudaMemcpyDeviceToHost),
            "download status");
        std::uint32_t mismatches[6]{};
        std::size_t first_mismatch[6]{};
        std::uint16_t first_actual[6]{};
        std::uint16_t first_expected[6]{};
        for (int mode = 0; mode < 6; ++mode) {
            const auto* actual = reinterpret_cast<const std::uint8_t*>(&host_states[mode]);
            const auto* wanted = reinterpret_cast<const std::uint8_t*>(&expected);
            for (std::size_t index = 0; index < sizeof(expected); ++index) {
                if (actual[index] == wanted[index]) continue;
                if (mismatches[mode] == 0) {
                    first_mismatch[mode] = index;
                    first_actual[mode] = actual[index];
                    first_expected[mode] = wanted[index];
                }
                ++mismatches[mode];
            }
        }
        check_cuda(cudaFree(device_action), "free action");
        check_cuda(cudaFree(device_status), "free status");
        check_cuda(cudaFree(device_rules), "free rules");
        check_cuda(cudaFree(device_states), "free states");
        std::cout
            << "{\"device_stack_bytes\":" << device_stack_bytes
            << ",\"completed_modes\":6,\"statuses\":["
            << static_cast<int>(statuses[0]) << ','
            << static_cast<int>(statuses[1]) << ','
            << static_cast<int>(statuses[2]) << ','
            << static_cast<int>(statuses[3]) << ','
            << static_cast<int>(statuses[4]) << ','
            << static_cast<int>(statuses[5]) << "],\"mismatches\":["
            << mismatches[0] << ',' << mismatches[1] << ',' << mismatches[2] << ','
            << mismatches[3] << ',' << mismatches[4] << ',' << mismatches[5]
            << "],\"first_mismatch\":[";
        for (int mode = 0; mode < 6; ++mode) {
            if (mode != 0) std::cout << ',';
            std::cout << "[" << first_mismatch[mode] << ','
                      << first_actual[mode] << ',' << first_expected[mode] << "]";
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
