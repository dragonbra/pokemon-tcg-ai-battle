#pragma once

#include <cstddef>
#include <cstdint>

#include "ptcg_cuda/opcodes.h"

namespace ptcg::cuda_engine {

struct alignas(16) Instruction {
    std::uint8_t opcode;
    std::uint8_t target;
    std::uint16_t flags;
    std::int32_t arg0;
    std::int32_t arg1;
    std::int32_t arg2;
};
static_assert(sizeof(Instruction) == 16);

struct alignas(16) ActionDescriptor {
    std::uint32_t program_offset;
    std::uint16_t program_length;
    std::uint16_t action_id;
    std::uint16_t option_type;
    std::uint16_t flags;
    std::uint32_t card_id;
};
static_assert(sizeof(ActionDescriptor) == 16);

struct alignas(16) ResetSpec {
    std::uint32_t seed;
    std::uint16_t active_card_id[2];
    std::uint16_t active_hp[2];
    std::uint16_t deck_count[2];
    std::uint16_t hand_count[2];
    std::uint16_t policy_id[2];
    std::uint16_t reserved;
};
static_assert(sizeof(ResetSpec) == 32);

struct RuntimeConfig {
    std::uint32_t abi_version = kAbiVersion;
    std::uint32_t batch_size = 0;
    std::uint32_t policy_count = 0;
    std::uint32_t route_capacity = 0;
};

struct RulePackView {
    const Instruction* instructions = nullptr;
    std::uint32_t instruction_count = 0;
    const ActionDescriptor* actions = nullptr;
    std::uint32_t action_count = 0;
};

}  // namespace ptcg::cuda_engine
