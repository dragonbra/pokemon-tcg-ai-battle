#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 4) {
            throw std::runtime_error(
                "usage: official_runtime_smoke <official_rules.bin> [batch] "
                "[resume-fixture]");
        }
        const std::uint32_t batch = argc >= 3
            ? static_cast<std::uint32_t>(std::stoul(argv[2]))
            : 4096U;
        if (batch == 0) throw std::runtime_error("batch must be positive");

        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        bool has_resume_fixture = argc == 4;
        OfficialFlowFixtureHeader fixture_header{};
        OfficialStatePod fixture_pending{};
        OfficialStatePod fixture_expected{};
        if (has_resume_fixture) {
            const std::vector<std::uint8_t> fixture = read_binary(argv[3]);
            const std::size_t expected_bytes = sizeof(OfficialFlowFixtureHeader)
                + 2 * sizeof(OfficialStatePod);
            if (fixture.size() != expected_bytes) {
                throw std::runtime_error("invalid resume fixture size");
            }
            std::memcpy(&fixture_header, fixture.data(), sizeof(fixture_header));
            std::memcpy(
                &fixture_pending,
                fixture.data() + sizeof(fixture_header),
                sizeof(fixture_pending));
            std::memcpy(
                &fixture_expected,
                fixture.data() + sizeof(fixture_header) + sizeof(fixture_pending),
                sizeof(fixture_expected));
            if (std::memcmp(fixture_header.magic, "PTCGFLW1", 8) != 0
                || fixture_header.version != kOfficialFlowFixtureVersion
                || fixture_header.state_abi_version != kOfficialStateAbiVersion
                || fixture_header.state_bytes != sizeof(OfficialStatePod)
                || fixture_header.action_count > kOfficialOptionCapacity) {
                throw std::runtime_error("invalid resume fixture header");
            }
        }
        OfficialRuntimeConfig config{};
        config.batch_size = batch;
        config.rule_pack_bytes = rule_pack.size();

        std::size_t free_before = 0;
        std::size_t total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_before, &total_bytes), "cudaMemGetInfo before");

        OfficialDeviceArena arena{};
        check_cuda(allocate_official_arena(&arena, config), "allocate official arena");
        check_cuda(
            upload_official_rule_pack(&arena, rule_pack.data(), rule_pack.size()),
            "upload official rule pack");

        std::vector<OfficialStatePod> states(batch);
        for (std::uint32_t env = 0; env < batch; ++env) {
            OfficialStatePod& state = states[env];
            state.episode_id = env;
            switch (env % 4U) {
                case 1:
                    state.game_result = static_cast<std::uint8_t>(
                        OfficialGameResult::kPlayer0Win);
                    break;
                case 2:
                    state.select_type = static_cast<std::uint8_t>(1);
                    state.options.count = 2;
                    break;
                case 3:
                    state.error = static_cast<std::int32_t>(
                        OfficialPodError::kInvalidAction);
                    break;
                default:
                    break;
            }
        }
        if (has_resume_fixture) states[0] = fixture_pending;
        check_cuda(
            upload_official_states_async(
                &arena, states.data(), cudaMemcpyHostToDevice, nullptr),
            "upload official states");
        check_cuda(classify_official_states_async(&arena, nullptr), "classify states");

        std::vector<std::uint8_t> statuses(batch);
        check_cuda(cudaMemcpy(
            statuses.data(), arena.statuses, statuses.size(), cudaMemcpyDeviceToHost),
            "copy classified statuses");
        std::size_t classification_mismatches = 0;
        for (std::uint32_t env = 0; env < batch; ++env) {
            const OfficialFlowStatus expected[] = {
                OfficialFlowStatus::kIdle,
                OfficialFlowStatus::kTerminal,
                OfficialFlowStatus::kNeedsAction,
                OfficialFlowStatus::kError,
            };
            const OfficialFlowStatus wanted = has_resume_fixture && env == 0
                ? OfficialFlowStatus::kNeedsAction
                : expected[env % 4U];
            if (statuses[env] != static_cast<std::uint8_t>(wanted)) {
                ++classification_mismatches;
            }
        }

        check_cuda(
            cudaMemset(arena.actions, 0, sizeof(OfficialActionPod) * batch),
            "clear actions");
        if (has_resume_fixture) {
            OfficialActionPod action{};
            action.count = fixture_header.action_count;
            for (std::uint16_t index = 0; index < action.count; ++index) {
                action.option_indices[index] = fixture_header.option_indices[index];
            }
            check_cuda(cudaMemcpy(
                arena.actions,
                &action,
                sizeof(action),
                cudaMemcpyHostToDevice),
                "upload resume fixture action");
        }
        check_cuda(
            apply_official_actions_and_advance_async(&arena, nullptr, nullptr),
            "apply invalid canary actions");
        check_cuda(cudaMemcpy(
            statuses.data(), arena.statuses, statuses.size(), cudaMemcpyDeviceToHost),
            "copy post-action statuses");
        std::size_t fail_closed_mismatches = 0;
        for (std::uint32_t env = 0; env < batch; ++env) {
            const OfficialFlowStatus expected = has_resume_fixture && env == 0
                ? OfficialFlowStatus::kIdle
                : (env % 4U == 1U
                    ? OfficialFlowStatus::kTerminal
                    : OfficialFlowStatus::kError);
            if (statuses[env] != static_cast<std::uint8_t>(expected)) {
                ++fail_closed_mismatches;
            }
        }
        std::size_t resume_state_mismatches = 0;
        if (has_resume_fixture) {
            OfficialStatePod resumed{};
            check_cuda(cudaMemcpy(
                &resumed,
                arena.states,
                sizeof(resumed),
                cudaMemcpyDeviceToHost),
                "copy resumed fixture state");
            if (std::memcmp(
                    &resumed, &fixture_expected, sizeof(OfficialStatePod)) != 0) {
                resume_state_mismatches = 1;
            }
        }

        std::size_t free_after = 0;
        check_cuda(cudaMemGetInfo(&free_after, &total_bytes), "cudaMemGetInfo after");
        const std::size_t measured_delta = free_before > free_after
            ? free_before - free_after
            : 0;
        const std::size_t allocated_bytes = arena.allocated_bytes;
        const std::size_t device_stack_bytes = arena.device_stack_bytes;
        check_cuda(free_official_arena(&arena), "free official arena");

        const bool passed = classification_mismatches == 0
            && fail_closed_mismatches == 0
            && resume_state_mismatches == 0;
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"state_abi\":" << kOfficialStateAbiVersion
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"action_bytes\":" << sizeof(OfficialActionPod)
            << ",\"batch\":" << batch
            << ",\"arena_allocated_bytes\":" << allocated_bytes
            << ",\"device_stack_bytes\":" << device_stack_bytes
            << ",\"cuda_free_delta_bytes\":" << measured_delta
            << ",\"classification_mismatches\":" << classification_mismatches
            << ",\"fail_closed_mismatches\":" << fail_closed_mismatches
            << ",\"resume_fixture_present\":"
            << (has_resume_fixture ? "true" : "false")
            << ",\"resume_state_mismatches\":" << resume_state_mismatches
            << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
