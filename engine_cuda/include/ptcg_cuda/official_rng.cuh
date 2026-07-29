#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ptcg::cuda_engine {

struct OfficialMt19937 {
    static constexpr std::size_t kStateSize = 624;

    std::uint32_t words[kStateSize];
    std::uint32_t index;
};

__host__ __device__ inline std::uint32_t official_fold_seed32(std::uint64_t seed) {
    std::uint64_t value = seed + 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    value ^= value >> 31U;
    const std::uint32_t folded = static_cast<std::uint32_t>((value >> 32U) ^ value);
    return folded == 0 ? 1U : folded;
}

__host__ __device__ inline void official_seed_mt19937(
    OfficialMt19937* engine,
    std::uint64_t seed) {
    constexpr std::size_t n = OfficialMt19937::kStateSize;
    constexpr std::size_t s = 4;
    constexpr std::size_t t = 11;
    constexpr std::size_t p = (n - t) / 2;
    constexpr std::size_t q = p + t;
    constexpr std::size_t m = n;
    const std::uint32_t seeds[s] = {
        official_fold_seed32(seed),
        official_fold_seed32(seed ^ 0xA5A5A5A55A5A5A5AULL),
        official_fold_seed32(seed + 0xD1B54A32D192ED03ULL),
        official_fold_seed32(seed ^ 0x94D049BB133111EBULL),
    };

    for (std::size_t i = 0; i < n; ++i) {
        engine->words[i] = 0x8B8B8B8BU;
    }

    {
        const std::uint32_t r1 = 1371501266U;
        const std::uint32_t r2 = r1 + static_cast<std::uint32_t>(s);
        engine->words[p] += r1;
        engine->words[q] += r2;
        engine->words[0] = r2;
    }

    for (std::size_t k = 1; k <= s; ++k) {
        const std::size_t kn = k % n;
        const std::size_t kpn = (k + p) % n;
        const std::size_t kqn = (k + q) % n;
        const std::uint32_t arg =
            engine->words[kn] ^ engine->words[kpn] ^ engine->words[(k - 1) % n];
        const std::uint32_t r1 = 1664525U * (arg ^ (arg >> 27U));
        const std::uint32_t r2 =
            r1 + static_cast<std::uint32_t>(kn) + seeds[k - 1];
        engine->words[kpn] += r1;
        engine->words[kqn] += r2;
        engine->words[kn] = r2;
    }

    for (std::size_t k = s + 1; k < m; ++k) {
        const std::size_t kn = k % n;
        const std::size_t kpn = (k + p) % n;
        const std::size_t kqn = (k + q) % n;
        const std::uint32_t arg =
            engine->words[kn] ^ engine->words[kpn] ^ engine->words[(k - 1) % n];
        const std::uint32_t r1 = 1664525U * (arg ^ (arg >> 27U));
        const std::uint32_t r2 = r1 + static_cast<std::uint32_t>(kn);
        engine->words[kpn] += r1;
        engine->words[kqn] += r2;
        engine->words[kn] = r2;
    }

    for (std::size_t k = m; k < m + n; ++k) {
        const std::size_t kn = k % n;
        const std::size_t kpn = (k + p) % n;
        const std::size_t kqn = (k + q) % n;
        const std::uint32_t arg =
            engine->words[kn] + engine->words[kpn] + engine->words[(k - 1) % n];
        const std::uint32_t r3 = 1566083941U * (arg ^ (arg >> 27U));
        const std::uint32_t r4 = r3 - static_cast<std::uint32_t>(kn);
        engine->words[kpn] ^= r3;
        engine->words[kqn] ^= r4;
        engine->words[kn] = r4;
    }
    engine->index = static_cast<std::uint32_t>(n);
}

__host__ __device__ inline void official_twist_mt19937(OfficialMt19937* engine) {
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

__host__ __device__ inline std::uint32_t official_mt19937_next(
    OfficialMt19937* engine) {
    if (engine->index >= OfficialMt19937::kStateSize) {
        official_twist_mt19937(engine);
    }
    std::uint32_t value = engine->words[engine->index++];
    value ^= value >> 11U;
    value ^= (value << 7U) & 0x9D2C5680U;
    value ^= (value << 15U) & 0xEFC60000U;
    value ^= value >> 18U;
    return value;
}

__host__ __device__ inline std::uint32_t official_uniform_below(
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

__host__ __device__ inline void official_shuffle_60(
    std::uint16_t* cards,
    OfficialMt19937* engine) {
    std::uint32_t i = 1;
    const std::uint32_t first_position = official_uniform_below(engine, 2);
    std::uint16_t temporary = cards[i];
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
