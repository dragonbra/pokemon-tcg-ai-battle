#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "ptcg_cuda/state_layout.cuh"

namespace ptcg::cuda_engine {

struct DeviceArena {
    RuntimeConfig config{};
    BattleState* states = nullptr;
    ResetSpec* reset_specs = nullptr;
    Instruction* instructions = nullptr;
    ActionDescriptor* actions = nullptr;
    std::uint32_t instruction_count = 0;
    std::uint32_t action_count = 0;
    std::int32_t* route_indices = nullptr;
    std::uint32_t* route_counts = nullptr;
    std::uint64_t* digests = nullptr;
    PolicyCodecBuffers codec{};
    std::size_t allocated_bytes = 0;
};

cudaError_t allocate_arena(DeviceArena* arena, const RuntimeConfig& config);
cudaError_t free_arena(DeviceArena* arena);

cudaError_t upload_rule_pack(
    DeviceArena* arena,
    const Instruction* host_instructions,
    std::uint32_t instruction_count,
    const ActionDescriptor* host_actions,
    std::uint32_t action_count);

cudaError_t reset_from_host_async(
    DeviceArena* arena,
    const ResetSpec* host_specs,
    cudaStream_t stream);

cudaError_t reset_from_device_async(
    DeviceArena* arena,
    const ResetSpec* device_specs,
    cudaStream_t stream);

cudaError_t apply_actions_device_async(
    DeviceArena* arena,
    const std::int32_t* device_action_indices,
    cudaStream_t stream);

cudaError_t encode_policy_v1_async(DeviceArena* arena, cudaStream_t stream);
cudaError_t route_ready_async(DeviceArena* arena, cudaStream_t stream);
cudaError_t digest_async(DeviceArena* arena, cudaStream_t stream);

std::size_t codec_bytes_per_environment();

}  // namespace ptcg::cuda_engine
