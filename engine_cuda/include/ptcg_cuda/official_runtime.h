#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "ptcg_cuda/official_flow_status.h"
#include "ptcg_cuda/official_rule_layout.cuh"
#include "ptcg_cuda/official_state_pod.cuh"
#include "ptcg_cuda/state_layout.cuh"

namespace ptcg::cuda_engine {

constexpr std::size_t kOfficialMinimumDeviceStackBytes = 32U * 1024U;

struct alignas(16) OfficialActionPod {
    std::uint16_t option_indices[kOfficialOptionCapacity]{};
    std::uint16_t count = 0;
    std::uint16_t reserved[7]{};
};
static_assert(sizeof(OfficialActionPod) == 272);

struct OfficialRuntimeConfig {
    std::uint32_t state_abi_version = kOfficialStateAbiVersion;
    std::uint32_t rule_abi_version = kOfficialRuleAbiVersion;
    std::uint32_t batch_size = 0;
    std::uint32_t reserved = 0;
    std::size_t rule_pack_bytes = 0;
    std::size_t device_stack_bytes = kOfficialMinimumDeviceStackBytes;
};

struct OfficialDeviceArena {
    OfficialRuntimeConfig config{};
    OfficialStatePod* states = nullptr;
    std::uint8_t* rule_pack = nullptr;
    OfficialActionPod* actions = nullptr;
    std::uint8_t* statuses = nullptr;
    PolicyCodecBuffers codec{};
    std::size_t allocated_bytes = 0;
    std::size_t device_stack_bytes = 0;
};

cudaError_t allocate_official_arena(
    OfficialDeviceArena* arena,
    const OfficialRuntimeConfig& config);
cudaError_t free_official_arena(OfficialDeviceArena* arena);

cudaError_t upload_official_rule_pack(
    OfficialDeviceArena* arena,
    const void* host_rule_pack,
    std::size_t bytes);

cudaError_t upload_official_states_async(
    OfficialDeviceArena* arena,
    const OfficialStatePod* states,
    cudaMemcpyKind copy_kind,
    cudaStream_t stream);

// Build complete OfficialStatePod lanes directly on the device from 120 card
// IDs and a seed per lane.  The current setup policy is the frozen official
// first-min oracle contract; it is deliberately separate from the future
// interactive setup-selection entrypoint.
cudaError_t reset_official_states_seeded_first_min_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream);

cudaError_t reset_official_states_seeded_first_min_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream);

// Initialize real setup decisions on device and stop at IsFirst.  Subsequent
// setup choices use the same resident action/apply path as battle decisions.
cudaError_t reset_official_states_seeded_interactive_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream);

cudaError_t reset_official_states_seeded_interactive_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream);

cudaError_t classify_official_states_async(
    OfficialDeviceArena* arena,
    cudaStream_t stream);

cudaError_t advance_official_states_to_decision_async(
    OfficialDeviceArena* arena,
    cudaStream_t stream);

cudaError_t apply_official_actions_and_advance_async(
    OfficialDeviceArena* arena,
    const OfficialActionPod* device_actions,
    cudaStream_t stream);

// Apply actions already packed into arena->actions only for lanes whose most
// recent device status is kNeedsAction. Terminal, idle, and error lanes remain
// untouched, so a heterogeneous resident batch needs no host-side compaction.
cudaError_t apply_official_packed_ready_actions_async(
    OfficialDeviceArena* arena,
    cudaStream_t stream);

// Apply packed actions only while a lane is still in interactive setup.  This
// lets variable-length setup lanes share one resident batch without touching
// lanes that already reached Main.
cudaError_t apply_official_packed_setup_actions_async(
    OfficialDeviceArena* arena,
    cudaStream_t stream);

cudaError_t encode_official_policy_codec_v1_async(
    OfficialDeviceArena* arena,
    cudaStream_t stream);

// Packs policy-decoder output into the resident action POD without a host
// round-trip.  The input tensors are one value per option slot and one count
// per environment; invalid counts are deliberately left for the normal
// fail-closed action validator in apply_official_actions_and_advance_async.
cudaError_t pack_official_actions_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* option_indices,
    const std::int32_t* counts,
    std::uint32_t option_capacity,
    cudaStream_t stream);

cudaError_t pack_official_actions_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* option_indices,
    const std::int64_t* counts,
    std::uint32_t option_capacity,
    cudaStream_t stream);

}  // namespace ptcg::cuda_engine
