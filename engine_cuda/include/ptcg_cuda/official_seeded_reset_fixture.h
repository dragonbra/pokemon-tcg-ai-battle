#pragma once

#include <cstddef>
#include <cstdint>

#include "ptcg_cuda/official_flow_status.h"
#include "ptcg_cuda/official_state_pod.cuh"

namespace ptcg::cuda_engine {

constexpr std::uint32_t kOfficialSeededResetFixtureVersion = 1;

struct alignas(64) OfficialSeededResetFixtureHeader {
    char magic[8]{};
    std::uint32_t version = kOfficialSeededResetFixtureVersion;
    std::uint32_t state_abi_version = kOfficialStateAbiVersion;
    std::uint32_t state_bytes = sizeof(OfficialStatePod);
    std::uint32_t record_bytes = 0;
    std::uint32_t record_count = 0;
    std::uint8_t reserved[36]{};
};

struct alignas(64) OfficialSeededResetFixtureRecord {
    std::uint64_t seed = 0;
    OfficialFlowStatus expected_status = OfficialFlowStatus::kIdle;
    std::uint8_t reserved[55]{};
    OfficialStatePod expected{};
};

static_assert(sizeof(OfficialSeededResetFixtureHeader) == 64);
static_assert(offsetof(OfficialSeededResetFixtureRecord, expected) == 64);
static_assert(sizeof(OfficialSeededResetFixtureRecord) % 64 == 0);

}  // namespace ptcg::cuda_engine
