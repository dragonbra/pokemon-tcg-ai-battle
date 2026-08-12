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
#include "ptcg_cuda/official_main_fixture.h"
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

std::uint64_t digest_state(std::uint64_t digest, const OfficialStatePod& state) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&state);
    for (std::size_t index = 0; index < sizeof(state); ++index) {
        digest ^= bytes[index];
        digest *= 1099511628211ULL;
    }
    return digest;
}

OfficialFlowStatus replay_cpu(
    OfficialStatePod* state,
    const OfficialMainFixtureRecord& record,
    const OfficialRulePackView& rules) {
    if (record.operation == OfficialMainFixtureOperation::kAdvance) {
        return official_advance_idle_state_to_decision(state, rules);
    }
    if (record.operation == OfficialMainFixtureOperation::kApply) {
        return official_apply_pending_action(
            state,
            rules,
            record.option_indices,
            record.action_count);
    }
    throw std::runtime_error("unsupported main fixture operation");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error(
                "usage: official_main_smoke <official_rules.bin> <main_fixture.bin>");
        }
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        const std::vector<std::uint8_t> fixture = read_binary(argv[2]);
        if (fixture.size() < sizeof(OfficialMainFixtureHeader)) {
            throw std::runtime_error("main fixture is truncated");
        }
        OfficialMainFixtureHeader header{};
        std::memcpy(&header, fixture.data(), sizeof(header));
        if (std::memcmp(header.magic, "PTCGMAIN", 8) != 0
            || header.version != kOfficialMainFixtureVersion
            || header.state_abi_version != kOfficialStateAbiVersion
            || header.state_bytes != sizeof(OfficialStatePod)
            || header.record_bytes != sizeof(OfficialMainFixtureRecord)
            || header.record_count != kOfficialMainFixtureRecordCount
            || fixture.size() != sizeof(header)
                + header.record_count * sizeof(OfficialMainFixtureRecord)) {
            throw std::runtime_error("invalid main fixture header");
        }
        std::vector<OfficialMainFixtureRecord> records(header.record_count);
        std::memcpy(
            records.data(),
            fixture.data() + sizeof(header),
            records.size() * sizeof(records[0]));

        OfficialRuntimeConfig config{};
        config.batch_size = 1;
        config.rule_pack_bytes = rule_pack.size();
        OfficialDeviceArena arena{};
        check_cuda(allocate_official_arena(&arena, config), "allocate arena");
        check_cuda(
            upload_official_rule_pack(&arena, rule_pack.data(), rule_pack.size()),
            "upload rules");
        const OfficialRulePackView host_rules = make_official_rule_pack_view(
            rule_pack.data());

        std::uint32_t status_mismatches = 0;
        std::uint32_t fixture_mismatches = 0;
        std::uint32_t cuda_mismatches = 0;
        std::uint64_t cpu_digest = 1469598103934665603ULL;
        std::uint64_t gpu_digest = 1469598103934665603ULL;
        for (const OfficialMainFixtureRecord& record : records) {
            OfficialStatePod cpu = record.input;
            const OfficialFlowStatus cpu_status = replay_cpu(
                &cpu, record, host_rules);
            if (cpu_status != record.expected_status) ++status_mismatches;
            if (std::memcmp(&cpu, &record.expected, sizeof(cpu)) != 0) {
                ++fixture_mismatches;
            }

            check_cuda(
                upload_official_states_async(
                    &arena, &record.input, cudaMemcpyHostToDevice, nullptr),
                "upload input state");
            if (record.operation == OfficialMainFixtureOperation::kAdvance) {
                check_cuda(
                    advance_official_states_to_decision_async(&arena, nullptr),
                    "advance main state");
            } else {
                OfficialActionPod action{};
                action.count = record.action_count;
                for (std::uint16_t index = 0; index < action.count; ++index) {
                    action.option_indices[index] = record.option_indices[index];
                }
                check_cuda(cudaMemcpy(
                    arena.actions,
                    &action,
                    sizeof(action),
                    cudaMemcpyHostToDevice), "upload main action");
                check_cuda(
                    apply_official_actions_and_advance_async(
                        &arena, nullptr, nullptr),
                    "apply main action");
            }
            OfficialStatePod gpu{};
            std::uint8_t gpu_status = 0;
            check_cuda(cudaMemcpy(
                &gpu, arena.states, sizeof(gpu), cudaMemcpyDeviceToHost),
                "download main state");
            check_cuda(cudaMemcpy(
                &gpu_status, arena.statuses, sizeof(gpu_status),
                cudaMemcpyDeviceToHost), "download main status");
            if (gpu_status != static_cast<std::uint8_t>(record.expected_status)) {
                ++status_mismatches;
            }
            if (std::memcmp(&gpu, &cpu, sizeof(gpu)) != 0) ++cuda_mismatches;
            cpu_digest = digest_state(cpu_digest, cpu);
            gpu_digest = digest_state(gpu_digest, gpu);
        }
        check_cuda(free_official_arena(&arena), "free arena");

        const bool passed = status_mismatches == 0
            && fixture_mismatches == 0
            && cuda_mismatches == 0
            && cpu_digest == gpu_digest;
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"state_abi\":" << kOfficialStateAbiVersion
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"scenario_count\":" << records.size()
            << ",\"status_mismatches\":" << status_mismatches
            << ",\"fixture_mismatches\":" << fixture_mismatches
            << ",\"cuda_mismatches\":" << cuda_mismatches
            << ",\"cpu_digest\":" << cpu_digest
            << ",\"gpu_digest\":" << gpu_digest << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
