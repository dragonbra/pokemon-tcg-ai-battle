#pragma once

#include <cstdint>

#include "ptcg_cuda/official_state_pod.cuh"

namespace ptcg::cuda_engine {

constexpr std::uint32_t kOfficialFlowFixtureVersion = 1;

struct alignas(16) OfficialFlowFixtureHeader {
    char magic[8]{};
    std::uint32_t version = kOfficialFlowFixtureVersion;
    std::uint32_t state_abi_version = kOfficialStateAbiVersion;
    std::uint32_t state_bytes = sizeof(OfficialStatePod);
    std::uint16_t action_count = 0;
    std::uint16_t option_indices[kOfficialOptionCapacity]{};
    std::uint8_t reserved[10]{};
};
static_assert(sizeof(OfficialFlowFixtureHeader) == 288);

}  // namespace ptcg::cuda_engine
