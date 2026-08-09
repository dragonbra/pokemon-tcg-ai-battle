#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_SEMANTIC_HD __host__ __device__
#else
#define PTCG_OFFICIAL_SEMANTIC_HD
#endif

namespace ptcg::cuda_engine {

struct OfficialStatePod;

constexpr std::uint32_t kOfficialSemanticHistoryEnabled = 0x80000000U;
constexpr std::uint32_t kOfficialStateAbiVersionMask =
    ~kOfficialSemanticHistoryEnabled;
constexpr std::uint32_t kOfficialSemanticHistoryCapacity = 64;
constexpr std::uint32_t kOfficialSemanticHistoryParamCapacity = 7;
constexpr std::uint32_t kOfficialSemanticSerialCapacity = 256;

enum class OfficialSemanticLogType : std::uint8_t {
    kShuffle = 0,
    kHasBasicPokemon = 1,
    kTurnStart = 2,
    kTurnEnd = 3,
    kDraw = 4,
    kDrawReverse = 5,
    kMoveCard = 6,
    kMoveCardReverse = 7,
    kSwitch = 8,
    kChange = 9,
    kPlay = 10,
    kAttach = 11,
    kEvolve = 12,
    kDevolve = 13,
    kMoveAttached = 14,
    kAttack = 15,
    kHpChange = 16,
    kPoisoned = 17,
    kBurned = 18,
    kAsleep = 19,
    kParalyzed = 20,
    kConfused = 21,
    kCoin = 22,
    kResult = 23,
};

struct OfficialSemanticHistoryDeviceView {
    OfficialStatePod* states_base = nullptr;
    std::uint32_t batch_size = 0;
    std::uint32_t capacity = kOfficialSemanticHistoryCapacity;
    std::uint64_t* total_count = nullptr;
    std::uint32_t* write_index = nullptr;
    std::uint8_t* log_type = nullptr;
    std::uint8_t* param_count = nullptr;
    std::int32_t* params = nullptr;
    std::uint8_t* deck_membership_known = nullptr;
    std::uint8_t* prize_membership_known = nullptr;
    std::uint8_t* deck_order_known = nullptr;
    std::uint8_t* known_self_deck_serial = nullptr;
    std::uint64_t* deck_source_event = nullptr;
    std::uint64_t* prize_source_event = nullptr;
    std::uint8_t* known_opponent_hand = nullptr;
    std::uint8_t* possible_opponent_hand = nullptr;
    std::uint8_t* remembered_opponent_cards = nullptr;
    std::uint16_t* unknown_opponent_hand = nullptr;
    std::uint16_t* possible_hand_lower = nullptr;
    std::uint16_t* possible_hand_upper = nullptr;
};

PTCG_OFFICIAL_SEMANTIC_HD inline std::uint32_t official_state_abi_base(
    std::uint32_t abi_version) {
    return abi_version & kOfficialStateAbiVersionMask;
}

PTCG_OFFICIAL_SEMANTIC_HD inline bool official_semantic_history_enabled(
    const OfficialStatePod* state);

PTCG_OFFICIAL_SEMANTIC_HD inline OfficialSemanticHistoryDeviceView*
official_semantic_history_view(OfficialStatePod* state);

PTCG_OFFICIAL_SEMANTIC_HD inline void official_semantic_history_bind(
    OfficialStatePod* state,
    OfficialSemanticHistoryDeviceView* view);

PTCG_OFFICIAL_SEMANTIC_HD inline void official_semantic_history_clear_lane(
    OfficialStatePod* state);

PTCG_OFFICIAL_SEMANTIC_HD inline void official_semantic_history_append_raw(
    OfficialStatePod* state,
    OfficialSemanticLogType type,
    std::uint8_t param_count,
    const std::int32_t* params);

template <typename... Args>
PTCG_OFFICIAL_SEMANTIC_HD inline void official_semantic_history_append(
    OfficialStatePod* state,
    OfficialSemanticLogType type,
    Args... args) {
    static_assert(
        sizeof...(Args) <= kOfficialSemanticHistoryParamCapacity,
        "official semantic log has too many params");
    const std::int32_t params[kOfficialSemanticHistoryParamCapacity] = {
        static_cast<std::int32_t>(args)...};
    official_semantic_history_append_raw(
        state,
        type,
        static_cast<std::uint8_t>(sizeof...(Args)),
        params);
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_SEMANTIC_HD
