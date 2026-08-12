#include "ptcg_cuda/runtime.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine = ptcg::cuda_engine;

namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
std::uint32_t read_u32(int argc, char** argv, const std::string& flag, std::uint32_t fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == flag) {
            const unsigned long value = std::stoul(argv[index + 1]);
            if (value == 0 || value > 0xFFFFFFFFUL) {
                throw std::invalid_argument(flag + " must be a positive uint32");
            }
            return static_cast<std::uint32_t>(value);
        }
    }
    return fallback;
}

bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int index = 1; index < argc; ++index) {
        if (argv[index] == flag) {
            return true;
        }
    }
    return false;
}

std::vector<engine::Instruction> smoke_instructions() {
    using engine::Opcode;
    using engine::Target;
    return {
        {static_cast<std::uint8_t>(Opcode::kEndTurn), static_cast<std::uint8_t>(Target::kActor), 0, 0, 0, 0},
        {static_cast<std::uint8_t>(Opcode::kHalt), 0, 0, 0, 0, 0},

        {static_cast<std::uint8_t>(Opcode::kDraw), static_cast<std::uint8_t>(Target::kActor), 0, 1, 0, 0},
        {static_cast<std::uint8_t>(Opcode::kEndTurn), static_cast<std::uint8_t>(Target::kActor), 0, 0, 0, 0},
        {static_cast<std::uint8_t>(Opcode::kHalt), 0, 0, 0, 0, 0},

        {static_cast<std::uint8_t>(Opcode::kDamageActive), static_cast<std::uint8_t>(Target::kOpponent), 0, 30, 0, 0},
        {static_cast<std::uint8_t>(Opcode::kCheckKnockout), static_cast<std::uint8_t>(Target::kOpponent), 0, 0, 0, 0},
        {static_cast<std::uint8_t>(Opcode::kEndTurn), static_cast<std::uint8_t>(Target::kActor), 0, 0, 0, 0},
        {static_cast<std::uint8_t>(Opcode::kHalt), 0, 0, 0, 0, 0},

        {static_cast<std::uint8_t>(Opcode::kHealActive), static_cast<std::uint8_t>(Target::kActor), 0, 10, 0, 0},
        {static_cast<std::uint8_t>(Opcode::kEndTurn), static_cast<std::uint8_t>(Target::kActor), 0, 0, 0, 0},
        {static_cast<std::uint8_t>(Opcode::kHalt), 0, 0, 0, 0, 0},

        {static_cast<std::uint8_t>(Opcode::kDelayEffect), static_cast<std::uint8_t>(Target::kActor), 0, 660, 1207, 0},
        {static_cast<std::uint8_t>(Opcode::kHalt), 0, 0, 0, 0, 0},
    };
}

std::vector<engine::ActionDescriptor> smoke_actions() {
    return {
        {0, 2, 0, 1, 0, 0},
        {2, 3, 1, 2, 0, 0},
        {5, 4, 2, 3, 0, 0},
        {9, 3, 3, 4, 0, 0},
        {12, 2, 4, 5, 0, 660},
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t envs = read_u32(argc, argv, "--envs", 4096);
        const std::uint32_t steps = read_u32(argc, argv, "--steps", 1000);
        const std::uint32_t policies = read_u32(argc, argv, "--policies", 12);
        const bool trigger_known_divergence = has_flag(argc, argv, "--trigger-known-divergence");
        if (policies > engine::kMaxPolicies) {
            throw std::invalid_argument("--policies exceeds kMaxPolicies");
        }

        int device = 0;
        check(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");

        engine::DeviceArena arena{};
        const engine::RuntimeConfig config{
            engine::kAbiVersion,
            envs,
            policies,
            envs,
        };
        check(engine::allocate_arena(&arena, config), "allocate_arena");

        const auto instructions = smoke_instructions();
        const auto actions = smoke_actions();
        check(
            engine::upload_rule_pack(
                &arena,
                instructions.data(),
                static_cast<std::uint32_t>(instructions.size()),
                actions.data(),
                static_cast<std::uint32_t>(actions.size())),
            "upload_rule_pack");

        std::vector<engine::ResetSpec> specs(envs);
        std::vector<std::int32_t> host_actions(envs);
        for (std::uint32_t env = 0; env < envs; ++env) {
            specs[env].seed = 12345U + env * 17U;
            specs[env].active_card_id[0] = static_cast<std::uint16_t>(100 + env % 1000);
            specs[env].active_card_id[1] = static_cast<std::uint16_t>(200 + env % 1000);
            specs[env].active_hp[0] = 120;
            specs[env].active_hp[1] = 120;
            specs[env].deck_count[0] = 53;
            specs[env].deck_count[1] = 53;
            specs[env].hand_count[0] = 7;
            specs[env].hand_count[1] = 7;
            specs[env].policy_id[0] = static_cast<std::uint16_t>(env % policies);
            specs[env].policy_id[1] = static_cast<std::uint16_t>((env + 1U) % policies);
            host_actions[env] = trigger_known_divergence ? 4 : static_cast<std::int32_t>(env & 1U);
        }

        std::int32_t* device_actions = nullptr;
        check(cudaMalloc(reinterpret_cast<void**>(&device_actions), sizeof(std::int32_t) * envs), "cudaMalloc actions");
        check(
            cudaMemcpy(
                device_actions,
                host_actions.data(),
                sizeof(std::int32_t) * envs,
                cudaMemcpyHostToDevice),
            "copy actions");
        check(engine::reset_from_host_async(&arena, specs.data(), nullptr), "reset");
        check(cudaDeviceSynchronize(), "reset synchronize");

        if (trigger_known_divergence) {
            check(engine::apply_actions_device_async(&arena, device_actions, nullptr), "known divergence step");
            check(cudaDeviceSynchronize(), "known divergence synchronize");
            engine::BattleState first{};
            check(cudaMemcpy(&first, arena.states, sizeof(first), cudaMemcpyDeviceToHost), "copy known divergence state");
            const bool correct = first.status == static_cast<std::int32_t>(engine::EngineStatus::kError) &&
                                 first.error == static_cast<std::int32_t>(engine::EngineError::kKnownDivergence6601207);
            std::cout << "{\n"
                      << "  \"mode\": \"known_divergence\",\n"
                      << "  \"status\": " << first.status << ",\n"
                      << "  \"error\": " << first.error << ",\n"
                      << "  \"accepted\": " << (correct ? "true" : "false") << "\n"
                      << "}\n";
            check(cudaFree(device_actions), "free actions");
            check(engine::free_arena(&arena), "free arena");
            return correct ? 0 : 2;
        }

        for (int warmup = 0; warmup < 10; ++warmup) {
            check(engine::apply_actions_device_async(&arena, device_actions, nullptr), "warmup apply");
            check(engine::encode_policy_v1_async(&arena, nullptr), "warmup encode");
            check(engine::route_ready_async(&arena, nullptr), "warmup route");
        }
        check(cudaDeviceSynchronize(), "warmup synchronize");

        cudaEvent_t start{};
        cudaEvent_t stop{};
        check(cudaEventCreate(&start), "create start event");
        check(cudaEventCreate(&stop), "create stop event");
        check(cudaEventRecord(start), "record start");
        for (std::uint32_t step = 0; step < steps; ++step) {
            check(engine::apply_actions_device_async(&arena, device_actions, nullptr), "apply actions");
            check(engine::encode_policy_v1_async(&arena, nullptr), "encode policy");
            check(engine::route_ready_async(&arena, nullptr), "route ready");
        }
        check(cudaEventRecord(stop), "record stop");
        check(cudaEventSynchronize(stop), "synchronize stop");
        float elapsed_ms = 0.0F;
        check(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");

        check(engine::digest_async(&arena, nullptr), "digest");
        std::vector<std::uint32_t> route_counts(policies);
        std::uint64_t digest = 0;
        engine::BattleState first{};
        check(
            cudaMemcpy(
                route_counts.data(),
                arena.route_counts,
                sizeof(std::uint32_t) * policies,
                cudaMemcpyDeviceToHost),
            "copy route counts");
        check(cudaMemcpy(&digest, arena.digests, sizeof(digest), cudaMemcpyDeviceToHost), "copy digest");
        check(cudaMemcpy(&first, arena.states, sizeof(first), cudaMemcpyDeviceToHost), "copy first state");
        const std::uint64_t routed = [&]() {
            std::uint64_t total = 0;
            for (const auto count : route_counts) {
                total += count;
            }
            return total;
        }();
        const double env_steps_per_second =
            static_cast<double>(envs) * steps * 1000.0 / elapsed_ms;

        std::cout << std::fixed << std::setprecision(3)
                  << "{\n"
                  << "  \"device\": \"" << properties.name << "\",\n"
                  << "  \"compute_capability\": \"" << properties.major << "." << properties.minor << "\",\n"
                  << "  \"environments\": " << envs << ",\n"
                  << "  \"policies\": " << policies << ",\n"
                  << "  \"steps\": " << steps << ",\n"
                  << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
                  << "  \"environment_steps_per_second\": " << env_steps_per_second << ",\n"
                  << "  \"state_bytes_per_environment\": " << sizeof(engine::BattleState) << ",\n"
                  << "  \"codec_bytes_per_environment\": " << engine::codec_bytes_per_environment() << ",\n"
                  << "  \"arena_allocated_mib\": " << static_cast<double>(arena.allocated_bytes) / (1024.0 * 1024.0) << ",\n"
                  << "  \"last_route_count\": " << routed << ",\n"
                  << "  \"first_state_status\": " << first.status << ",\n"
                  << "  \"first_state_error\": " << first.error << ",\n"
                  << "  \"first_digest\": " << digest << "\n"
                  << "}\n";

        const bool valid = routed == envs &&
                           first.status == static_cast<std::int32_t>(engine::EngineStatus::kNeedsPolicy) &&
                           first.error == static_cast<std::int32_t>(engine::EngineError::kNone);
        check(cudaEventDestroy(start), "destroy start event");
        check(cudaEventDestroy(stop), "destroy stop event");
        check(cudaFree(device_actions), "free actions");
        check(engine::free_arena(&arena), "free arena");
        return valid ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "ptcg_cuda_smoke: " << error.what() << '\n';
        return 1;
    }
}
