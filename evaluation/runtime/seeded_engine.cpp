// Local deterministic ABI over the untouched official engine implementation.
// This translation unit is compiled outside engine/source and never ships in Kaggle packages.

#include <cstdint>

#define BattleStart OfficialUnseededBattleStart
#define AgentStart OfficialUnseededAgentStart
#include "Export.cpp"
#undef AgentStart
#undef BattleStart

namespace {

thread_local std::uint32_t configured_engine_seed = 0;
thread_local std::uint32_t configured_search_seed = 0;

std::uint32_t normalized_seed(unsigned long long seed) {
    const auto value = static_cast<std::uint32_t>(seed);
    return value == 0 ? 1U : value;
}
StartData start_seeded_battle(const int* cards, std::uint32_t seed) {
    if (cards == nullptr) {
        return {nullptr, -1, 1};
    }

    auto* data = new ApiData();
    data->apiDataType = 1;
    GameConfig config{};
    config.seed = seed;
    config.recordLog = true;
    config.deviceRand = false;

    for (int player = 0; player < 2; ++player) {
        std::unordered_map<std::u8string, int> name_count;
        bool ace_spec = false;
        bool basic = false;
        for (int card_index = 0; card_index < DECK_SIZE; ++card_index) {
            const CardId id = cards[player * DECK_SIZE + card_index];
            if (!CardTable.contains(id)) {
                delete data;
                return {nullptr, player, 1};
            }

            const CardMaster& master = CardTable.at(id);
            if (master.aceSpec) {
                if (ace_spec) {
                    delete data;
                    return {nullptr, player, 4};
                }
                ace_spec = true;
            }
            if (master.cardType == CardType::Pokemon &&
                master.evolutionType == EvolutionType::Basic) {
                basic = true;
            }
            int& count = name_count[master.name];
            ++count;
            if (count > DECK_SAME_CARD_MAX && master.cardType != CardType::BasicEnergy) {
                delete data;
                return {nullptr, player, 2};
            }
            config.decks[player].cards[card_index] = id;
        }
        if (!basic) {
            delete data;
            return {nullptr, player, 3};
        }
    }

    data->init(config);
    data->start();
    data->next();
    return {data, -1, 0};
}

ApiData* start_seeded_agent(std::uint32_t seed) {
    auto* data = new ApiData();
    data->apiDataType = 2;
    GameConfig& config = data->game.config;
    config.seed = seed;
    config.recordLog = true;
    config.deviceRand = false;
    data->game.rng = std::mt19937(config.seed);
    data->state.game = &data->game;
    return data;
}

}  // namespace

extern "C" {

GAME_API const char* SeededRuntimeAbi() {
    return "seeded_official_engine_abi_v1";
}

GAME_API int ConfigureSeeds(unsigned long long engine_seed, unsigned long long search_seed) {
    configured_engine_seed = normalized_seed(engine_seed);
    configured_search_seed = normalized_seed(search_seed);
    return 0;
}

GAME_API StartData BattleStartSeeded(int* cards, unsigned long long seed) {
    try {
        return start_seeded_battle(cards, normalized_seed(seed));
    } catch (...) {
        return {nullptr, -1, 9001};
    }
}

GAME_API ApiData* AgentStartSeeded(unsigned long long seed) {
    try {
        return start_seeded_agent(normalized_seed(seed));
    } catch (...) {
        return nullptr;
    }
}

GAME_API StartData BattleStart(int* cards) {
    if (configured_engine_seed == 0) {
        return {nullptr, -1, 9002};
    }
    return BattleStartSeeded(cards, configured_engine_seed);
}

GAME_API ApiData* AgentStart() {
    if (configured_search_seed == 0) {
        return nullptr;
    }
    return AgentStartSeeded(configured_search_seed);
}

}
