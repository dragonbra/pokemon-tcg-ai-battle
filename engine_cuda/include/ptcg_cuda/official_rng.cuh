#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define PTCG_CUDA_HD __host__ __device__
#else
#define PTCG_CUDA_HD
#endif

namespace ptcg::cuda_engine {

struct OfficialMt19937 {
    static constexpr std::size_t kStateSize = 624;

    std::uint32_t words[kStateSize];
    std::uint32_t index;
    std::uint64_t draw_count;
};

PTCG_CUDA_HD inline std::uint32_t official_fold_seed32(std::uint64_t seed) {
    std::uint64_t value = seed + 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    value ^= value >> 31U;
    const std::uint32_t folded = static_cast<std::uint32_t>((value >> 32U) ^ value);
    return folded == 0 ? 1U : folded;
}

PTCG_CUDA_HD inline void official_seed_mt19937(
    OfficialMt19937* engine,
    std::uint64_t seed) {
    engine->words[0] = static_cast<std::uint32_t>(seed);
    for (std::size_t index = 1; index < OfficialMt19937::kStateSize; ++index) {
        const std::uint32_t previous = engine->words[index - 1];
        engine->words[index] = 1812433253U * (previous ^ (previous >> 30U))
            + static_cast<std::uint32_t>(index);
    }
    engine->index = static_cast<std::uint32_t>(OfficialMt19937::kStateSize);
    engine->draw_count = 0;
}

PTCG_CUDA_HD inline void official_twist_mt19937(OfficialMt19937* engine) {
    constexpr std::size_t n = OfficialMt19937::kStateSize;
    constexpr std::size_t m = 397;
    constexpr std::uint32_t upper_mask = 0x80000000U;
    constexpr std::uint32_t lower_mask = 0x7FFFFFFFU;
    constexpr std::uint32_t a = 0x9908B0DFU;

    for (std::size_t k = 0; k < n - m; ++k) {
        const std::uint32_t y =
            (engine->words[k] & upper_mask) | (engine->words[k + 1] & lower_mask);
        engine->words[k] =
            engine->words[k + m] ^ (y >> 1U) ^ ((y & 1U) != 0U ? a : 0U);
    }
    for (std::size_t k = n - m; k < n - 1; ++k) {
        const std::uint32_t y =
            (engine->words[k] & upper_mask) | (engine->words[k + 1] & lower_mask);
        engine->words[k] =
            engine->words[k + m - n] ^ (y >> 1U) ^ ((y & 1U) != 0U ? a : 0U);
    }
    const std::uint32_t y =
        (engine->words[n - 1] & upper_mask) | (engine->words[0] & lower_mask);
    engine->words[n - 1] =
        engine->words[m - 1] ^ (y >> 1U) ^ ((y & 1U) != 0U ? a : 0U);
    engine->index = 0;
}

PTCG_CUDA_HD inline std::uint32_t official_mt19937_next(
    OfficialMt19937* engine) {
    if (engine->index >= OfficialMt19937::kStateSize) {
        official_twist_mt19937(engine);
    }
    std::uint32_t value = engine->words[engine->index++];
    ++engine->draw_count;
    value ^= value >> 11U;
    value ^= (value << 7U) & 0x9D2C5680U;
    value ^= (value << 15U) & 0xEFC60000U;
    value ^= value >> 18U;
    return value;
}

PTCG_CUDA_HD inline std::uint32_t official_uniform_below(
    OfficialMt19937* engine,
    std::uint32_t range) {
    std::uint64_t product =
        static_cast<std::uint64_t>(official_mt19937_next(engine)) * range;
    std::uint32_t low = static_cast<std::uint32_t>(product);
    if (low < range) {
        const std::uint32_t threshold = static_cast<std::uint32_t>(-range) % range;
        while (low < threshold) {
            product =
                static_cast<std::uint64_t>(official_mt19937_next(engine)) * range;
            low = static_cast<std::uint32_t>(product);
        }
    }
    return static_cast<std::uint32_t>(product >> 32U);
}

template <typename T>
PTCG_CUDA_HD inline void official_shuffle_60(
    T* cards,
    OfficialMt19937* engine) {
    std::uint32_t i = 1;
    const std::uint32_t first_position = official_uniform_below(engine, 2);
    T temporary = cards[i];
    cards[i++] = cards[first_position];
    cards[first_position] = temporary;

    while (i != 60) {
        const std::uint32_t swap_range = i + 1;
        const std::uint32_t combined =
            official_uniform_below(engine, swap_range * (swap_range + 1));
        const std::uint32_t first = combined / (swap_range + 1);
        const std::uint32_t second = combined % (swap_range + 1);

        temporary = cards[i];
        cards[i++] = cards[first];
        cards[first] = temporary;
        temporary = cards[i];
        cards[i++] = cards[second];
        cards[second] = temporary;
    }
}

}  // namespace ptcg::cuda_engine

#undef PTCG_CUDA_HD
