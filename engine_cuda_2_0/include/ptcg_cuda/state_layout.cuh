#pragma once

#include <cstdint>
#include <type_traits>

#include "ptcg_cuda/engine_abi.h"
#include "ptcg_cuda/official_rng.cuh"

namespace ptcg::cuda_engine {

struct alignas(16) CardInstance {
    std::uint16_t card_id;
    std::uint16_t max_hp;
    std::uint16_t damage;
    std::int16_t attached_to;
    std::uint8_t owner;
    std::uint8_t zone;
    std::uint8_t slot;
    std::uint8_t flags;
    std::uint32_t aux;
};
static_assert(sizeof(CardInstance) == 16);

struct alignas(16) EffectFrame {
    std::uint32_t program_counter;
    std::uint32_t program_end;
    std::uint16_t source_card_id;
    std::uint8_t actor;
    std::uint8_t flags;
    std::int32_t scratch;
};
static_assert(sizeof(EffectFrame) == 16);

struct alignas(16) DelayedEffect {
    std::uint32_t program_offset;
    std::uint16_t program_length;
    std::uint16_t source_card_id;
    std::uint32_t due_turn;
    std::uint16_t effect_card_id;
    std::uint8_t owner;
    std::uint8_t flags;
};
static_assert(sizeof(DelayedEffect) == 16);

struct ContinuationFrame {
    std::int32_t args[3];
    std::uint16_t opcode;
    std::uint8_t arg_type;
    std::uint8_t call_count;
    std::uint8_t called_count;
    std::uint8_t flags;
    std::uint16_t reserved;
};
static_assert(sizeof(ContinuationFrame) == 20);

struct alignas(32) PlayerState {
    std::uint16_t policy_id;
    std::uint16_t active_entity;
    std::uint16_t deck_count;
    std::uint16_t hand_count;
    std::uint16_t prize_count;
    std::uint16_t discard_count;
    std::uint16_t bench_count;
    std::uint16_t flags;
    std::uint32_t turns_taken;
    std::uint32_t actions_taken;
    std::uint32_t reserved[2];
};
static_assert(sizeof(PlayerState) == 32);

struct alignas(64) BattleState {
    OfficialMt19937 rng;
    std::uint64_t episode_id;
    std::uint32_t turn;
    std::uint32_t decision_count;
    std::int32_t status;
    std::int32_t error;
    std::int32_t winner;
    std::uint16_t actor;
    std::uint16_t action_count;
    std::uint16_t entity_count;
    std::uint16_t effect_stack_size;
    std::uint16_t continuation_stack_size;
    std::uint16_t delayed_count;
    std::uint32_t counters[kMaxCounters];
    PlayerState players[2];
    CardInstance cards[kMaxCardInstances];
    EffectFrame stack[kMaxEffectFrames];
    ContinuationFrame continuations[kMaxContinuationFrames];
    DelayedEffect delayed[kMaxDelayedEffects];
};

static_assert(std::is_trivially_copyable_v<BattleState>);
static_assert(sizeof(BattleState) <= 32768, "state exceeded the planned per-environment budget");

struct PolicyCodecBuffers {
    std::int64_t* global_cat = nullptr;
    float* global_num = nullptr;
    std::int64_t* entity_cat = nullptr;
    float* entity_num = nullptr;
    std::int64_t* entity_parent = nullptr;
    std::uint8_t* entity_mask = nullptr;
    std::int64_t* option_cat = nullptr;
    float* option_num = nullptr;
    std::int64_t* option_equiv = nullptr;
    std::uint8_t* option_mask = nullptr;
    std::int64_t* min_count = nullptr;
    std::int64_t* max_count = nullptr;
};

constexpr std::size_t kGlobalCatWidth = 8;
constexpr std::size_t kGlobalNumWidth = 16;
constexpr std::size_t kEntityCatWidth = 6;
constexpr std::size_t kEntityNumWidth = 10;
constexpr std::size_t kOptionCatWidth = 12;
constexpr std::size_t kOptionNumWidth = 4;

}  // namespace ptcg::cuda_engine
