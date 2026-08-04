#include <cstdint>
#include <exception>

#include "All.h"
#include "Api.h"

#if defined(_WIN32)
#define PTCG_RAW_API extern "C" __declspec(dllexport)
#else
#define PTCG_RAW_API extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr int kRawSelectMetaSize = 27;

StartData start_raw_seeded_battle(const int* cards, std::uint64_t seed) {
    if (cards == nullptr) return {nullptr, -1, 1};

    auto* data = new ApiData();
    data->apiDataType = 1;
    GameConfig config{};
    config.seed = static_cast<std::uint32_t>(seed);
    if (config.seed == 0) config.seed = 1;
    config.recordLog = false;
    config.deviceRand = false;
    for (int player = 0; player < 2; ++player) {
        for (int card = 0; card < DECK_SIZE; ++card) {
            const CardId id = cards[player * DECK_SIZE + card];
            if (!CardTable.contains(id)) {
                delete data;
                return {nullptr, player, 1};
            }
            config.decks[player].cards[card] = id;
        }
    }
    data->init(config);
    data->start();
    data->next();
    return {data, -1, 0};
}

}  // namespace

PTCG_RAW_API int RawGameInitialize() {
    try {
        InitializeAll();
        return 0;
    } catch (...) {
        return -9001;
    }
}

PTCG_RAW_API StartData RawBattleStartSeeded(
    const int* cards,
    unsigned long long seed) {
    try {
        return start_raw_seeded_battle(cards, static_cast<std::uint64_t>(seed));
    } catch (...) {
        return {nullptr, -1, 9001};
    }
}

PTCG_RAW_API SerialData RawGetBattleData(ApiData* data) {
    try {
        if (data == nullptr) return {nullptr, nullptr, 0, -1};
        return ApiGetBattleData(data);
    } catch (...) {
        return {nullptr, nullptr, 0, -1};
    }
}

PTCG_RAW_API int RawSelect(ApiData* data, int* selected, int count) {
    try {
        if (data == nullptr || count < 0 || (count > 0 && selected == nullptr)) {
            return -1;
        }
        return ApiSelect(data, selected, count);
    } catch (...) {
        return -9001;
    }
}

PTCG_RAW_API int RawWriteSelectMeta(
    const ApiData* data,
    int* output,
    int capacity) {
    try {
        if (data == nullptr || output == nullptr) return 0;
        if (capacity < kRawSelectMetaSize) return -kRawSelectMetaSize;
        for (int index = 0; index < kRawSelectMetaSize; ++index) {
            output[index] = 0;
        }
        const State& state = data->state;
        output[0] = data->apiDataType;
        output[1] = state.turn;
        output[2] = static_cast<int>(state.phase);
        output[3] = static_cast<int>(state.gameResult);
        output[4] = static_cast<int>(state.finishReason);
        output[5] = static_cast<int>(state.selectType);
        output[6] = static_cast<int>(state.selectContext);
        output[7] = static_cast<int>(state.selectPlayer);
        output[8] = state.selectMin;
        output[9] = state.selectMax;
        output[10] = static_cast<int>(state.options.size());
        output[11] = state.turnActionCount;
        output[12] = state.effectActionCount;
        output[13] = state.turnAttackCount;
        output[14] = state.supporterPlayed ? 1 : 0;
        output[15] = state.stadiumPlayed ? 1 : 0;
        output[16] = state.energyPlayed ? 1 : 0;
        output[17] = state.retreated ? 1 : 0;
        output[18] = state.turnEnd ? 1 : 0;
        output[19] = static_cast<int>(state.firstPlayer);
        output[20] = state.coinHeadCount;
        output[21] = state.currentAttackId;
        output[22] = state.srcAttackId;
        output[23] = static_cast<int>(state.logs.size());
        output[24] = static_cast<int>(data->selectCount);
        output[25] = static_cast<int>(data->game.config.seed);
        output[26] = data->game.config.deviceRand ? 1 : 0;
        return kRawSelectMetaSize;
    } catch (...) {
        return -9001;
    }
}

PTCG_RAW_API void RawBattleFinish(ApiData* data) {
    delete data;
}

#undef PTCG_RAW_API
