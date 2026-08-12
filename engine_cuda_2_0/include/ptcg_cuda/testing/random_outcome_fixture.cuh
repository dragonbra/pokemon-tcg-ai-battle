#pragma once

// Test-only explicit random-outcome adapter.  No production source includes
// this header; parity executables apply a resolved outcome at the randomness
// boundary and then continue through the ordinary rule implementation.

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define PTCG_RANDOM_FIXTURE_HD __host__ __device__
#else
#define PTCG_RANDOM_FIXTURE_HD
#endif

namespace ptcg::cuda_engine::testing {

constexpr std::uint32_t kRandomOutcomeFixtureVersion = 1;

struct RandomOutcomeFixture {
    std::uint32_t version = kRandomOutcomeFixtureVersion;
    std::uint16_t shuffle_count = 0;
    std::uint16_t prize_count = 0;
    std::uint16_t coin_count = 0;
    std::uint16_t target_count = 0;
    std::uint16_t effect_count = 0;
    std::uint16_t reserved = 0;
    std::uint16_t shuffle_permutation[60]{};
    std::uint16_t prize_indices[6]{};
    std::uint8_t coin_results[32]{};
    std::uint16_t random_target_indices[32]{};
    std::int32_t effect_results[32]{};
};

struct RandomOutcomeCursor {
    std::uint16_t coin = 0;
    std::uint16_t target = 0;
    std::uint16_t effect = 0;
};

PTCG_RANDOM_FIXTURE_HD inline bool validate_random_outcome_fixture(
    const RandomOutcomeFixture& fixture) {
    if (fixture.version != kRandomOutcomeFixtureVersion
        || fixture.shuffle_count > 60 || fixture.prize_count > 6
        || fixture.coin_count > 32 || fixture.target_count > 32
        || fixture.effect_count > 32) {
        return false;
    }
    bool seen[60]{};
    for (std::uint16_t index = 0; index < fixture.shuffle_count; ++index) {
        const std::uint16_t value = fixture.shuffle_permutation[index];
        if (value >= fixture.shuffle_count || seen[value]) return false;
        seen[value] = true;
    }
    for (std::uint16_t index = 0; index < fixture.coin_count; ++index) {
        if (fixture.coin_results[index] > 1) return false;
    }
    for (std::uint16_t index = 0; index < fixture.prize_count; ++index) {
        if (fixture.prize_indices[index] >= fixture.shuffle_count) return false;
        for (std::uint16_t previous = 0; previous < index; ++previous) {
            if (fixture.prize_indices[previous] == fixture.prize_indices[index]) return false;
        }
    }
    return true;
}

PTCG_RANDOM_FIXTURE_HD inline bool consume_coin_outcome(
    const RandomOutcomeFixture& fixture,
    RandomOutcomeCursor* cursor,
    bool* head) {
    if (cursor == nullptr || head == nullptr || cursor->coin >= fixture.coin_count) {
        return false;
    }
    *head = fixture.coin_results[cursor->coin++] != 0;
    return true;
}

PTCG_RANDOM_FIXTURE_HD inline bool consume_target_outcome(
    const RandomOutcomeFixture& fixture,
    RandomOutcomeCursor* cursor,
    std::uint16_t option_count,
    std::uint16_t* selected) {
    if (cursor == nullptr || selected == nullptr
        || cursor->target >= fixture.target_count) {
        return false;
    }
    const std::uint16_t value = fixture.random_target_indices[cursor->target++];
    if (value >= option_count) return false;
    *selected = value;
    return true;
}

PTCG_RANDOM_FIXTURE_HD inline bool consume_effect_outcome(
    const RandomOutcomeFixture& fixture,
    RandomOutcomeCursor* cursor,
    std::int32_t* result) {
    if (cursor == nullptr || result == nullptr
        || cursor->effect >= fixture.effect_count) {
        return false;
    }
    *result = fixture.effect_results[cursor->effect++];
    return true;
}

PTCG_RANDOM_FIXTURE_HD inline bool random_outcome_fully_consumed(
    const RandomOutcomeFixture& fixture,
    const RandomOutcomeCursor& cursor) {
    return cursor.coin == fixture.coin_count
        && cursor.target == fixture.target_count
        && cursor.effect == fixture.effect_count;
}

template <typename T>
PTCG_RANDOM_FIXTURE_HD inline bool apply_shuffle_outcome(
    T* values,
    std::uint16_t count,
    const RandomOutcomeFixture& fixture) {
    if (values == nullptr || fixture.shuffle_count != count
        || !validate_random_outcome_fixture(fixture)) {
        return false;
    }
    T original[60];
    for (std::uint16_t index = 0; index < count; ++index) original[index] = values[index];
    for (std::uint16_t index = 0; index < count; ++index) {
        values[index] = original[fixture.shuffle_permutation[index]];
    }
    return true;
}

}  // namespace ptcg::cuda_engine::testing

#undef PTCG_RANDOM_FIXTURE_HD
