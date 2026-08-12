#pragma once

#include <cstdint>

namespace ptcg::cuda_engine {

enum class OfficialFlowStatus : std::uint8_t {
    kIdle = 0,
    kNeedsAction = 1,
    kTerminal = 2,
    kError = 3,
};

}  // namespace ptcg::cuda_engine
