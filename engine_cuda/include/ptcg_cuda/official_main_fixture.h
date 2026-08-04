#pragma once

#include <cstddef>
#include <cstdint>

#include "ptcg_cuda/official_flow_status.h"
#include "ptcg_cuda/official_state_pod.cuh"

namespace ptcg::cuda_engine {

constexpr std::uint32_t kOfficialMainFixtureVersion = 1;
constexpr std::uint32_t kOfficialMainFixtureRecordCount = 35;

enum class OfficialMainFixtureOperation : std::uint8_t {
    kAdvance = 1,
    kApply = 2,
};

struct alignas(64) OfficialMainFixtureHeader {
    char magic[8]{};
    std::uint32_t version = kOfficialMainFixtureVersion;
    std::uint32_t state_abi_version = kOfficialStateAbiVersion;
    std::uint32_t state_bytes = sizeof(OfficialStatePod);
    std::uint32_t record_bytes = 0;
    std::uint32_t record_count = 0;
    std::uint8_t reserved[36]{};
};

struct alignas(64) OfficialMainFixtureRecord {
    std::uint32_t scenario_id = 0;
    OfficialMainFixtureOperation operation = OfficialMainFixtureOperation::kAdvance;
    OfficialFlowStatus expected_status = OfficialFlowStatus::kIdle;
    std::uint16_t action_count = 0;
    std::uint16_t option_indices[kOfficialOptionCapacity]{};
    OfficialStatePod input{};
    OfficialStatePod expected{};
};

static_assert(sizeof(OfficialMainFixtureHeader) == 64);
static_assert(offsetof(OfficialMainFixtureRecord, input) % 64 == 0);

}  // namespace ptcg::cuda_engine
