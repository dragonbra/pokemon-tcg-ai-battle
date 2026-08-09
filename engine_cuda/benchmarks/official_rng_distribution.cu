#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "ptcg_cuda/official_rng.cuh"

namespace {

using ptcg::cuda_engine::OfficialMt19937;
using ptcg::cuda_engine::official_mt19937_next;
using ptcg::cuda_engine::official_seed_mt19937;
using ptcg::cuda_engine::official_shuffle_60;
using ptcg::cuda_engine::official_uniform_below;

struct Sample {
    std::uint8_t coin;
    std::uint8_t card0_position;
    std::uint8_t opening_contains_card0;
    std::uint8_t prize_contains_card0;
    std::uint8_t random_target;
    std::uint8_t reserved[3]{};
    std::uint64_t fingerprint;
};
static_assert(sizeof(Sample) == 16);

__global__ void sample_cuda(
    Sample* output,
    std::uint64_t seed_base,
    std::uint32_t count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    OfficialMt19937 rng{};
    official_seed_mt19937(&rng, seed_base + index);
    Sample sample{};
    const std::uint32_t first = official_mt19937_next(&rng);
    sample.coin = static_cast<std::uint8_t>((first % 2U) == 0U);
    std::uint8_t cards[60];
    for (std::uint32_t card = 0; card < 60; ++card) cards[card] = card;
    official_shuffle_60(cards, &rng);
    for (std::uint32_t position = 0; position < 60; ++position) {
        if (cards[position] == 0) {
            sample.card0_position = static_cast<std::uint8_t>(position);
            break;
        }
    }
    sample.opening_contains_card0 = sample.card0_position >= 53;
    sample.prize_contains_card0 = sample.card0_position >= 47 && sample.card0_position < 53;
    sample.random_target = static_cast<std::uint8_t>(official_uniform_below(&rng, 8));
    sample.fingerprint = (static_cast<std::uint64_t>(first) << 32U)
        ^ official_mt19937_next(&rng)
        ^ (static_cast<std::uint64_t>(sample.card0_position) << 8U)
        ^ sample.random_target;
    output[index] = sample;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<Sample> cuda_samples(std::uint64_t seed_base, std::uint32_t count) {
    Sample* device = nullptr;
    check_cuda(cudaMalloc(&device, sizeof(Sample) * count), "cudaMalloc");
    sample_cuda<<<(count + 255U) / 256U, 256>>>(device, seed_base, count);
    check_cuda(cudaGetLastError(), "sample_cuda launch");
    std::vector<Sample> result(count);
    check_cuda(cudaMemcpy(result.data(), device, sizeof(Sample) * count, cudaMemcpyDeviceToHost), "cudaMemcpy");
    check_cuda(cudaFree(device), "cudaFree");
    return result;
}

std::vector<Sample> cpu_samples(std::uint64_t seed_base, std::uint32_t count) {
    std::vector<Sample> result(count);
#pragma omp parallel for schedule(static)
    for (std::int64_t raw_index = 0; raw_index < static_cast<std::int64_t>(count); ++raw_index) {
        const std::uint32_t index = static_cast<std::uint32_t>(raw_index);
        std::mt19937 rng(static_cast<std::uint32_t>(seed_base + index));
        Sample sample{};
        const std::uint32_t first = rng();
        sample.coin = static_cast<std::uint8_t>((first % 2U) == 0U);
        std::array<std::uint8_t, 60> cards{};
        for (std::uint32_t card = 0; card < 60; ++card) cards[card] = card;
        std::shuffle(cards.begin(), cards.end(), rng);
        for (std::uint32_t position = 0; position < 60; ++position) {
            if (cards[position] == 0) {
                sample.card0_position = static_cast<std::uint8_t>(position);
                break;
            }
        }
        sample.opening_contains_card0 = sample.card0_position >= 53;
        sample.prize_contains_card0 = sample.card0_position >= 47 && sample.card0_position < 53;
        std::uniform_int_distribution<std::uint32_t> target(0, 7);
        sample.random_target = static_cast<std::uint8_t>(target(rng));
        sample.fingerprint = (static_cast<std::uint64_t>(first) << 32U)
            ^ rng()
            ^ (static_cast<std::uint64_t>(sample.card0_position) << 8U)
            ^ sample.random_target;
        result[index] = sample;
    }
    return result;
}

void emit_summary(
    const char* backend,
    std::uint32_t logical_batch,
    std::uint64_t seed_base,
    const std::vector<Sample>& samples) {
    std::uint64_t coin = 0;
    std::uint64_t opening = 0;
    std::uint64_t prize = 0;
    std::array<std::uint64_t, 60> positions{};
    std::array<std::uint64_t, 8> targets{};
    std::uint64_t adjacent_duplicates = 0;
    double sum = 0.0;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const Sample& sample = samples[index];
        coin += sample.coin;
        opening += sample.opening_contains_card0;
        prize += sample.prize_contains_card0;
        ++positions[sample.card0_position];
        ++targets[sample.random_target];
        sum += sample.coin;
        if (index > 0 && sample.fingerprint == samples[index - 1].fingerprint) {
            ++adjacent_duplicates;
        }
    }
    const double mean = sum / static_cast<double>(samples.size());
    double numerator = 0.0;
    double denominator = 0.0;
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const double centered = static_cast<double>(samples[index].coin) - mean;
        denominator += centered * centered;
        if (index > 0) {
            numerator += centered * (static_cast<double>(samples[index - 1].coin) - mean);
        }
    }
    const double lag1 = denominator == 0.0 ? 0.0 : numerator / denominator;
    std::cout << "{\"backend\":\"" << backend << "\",\"logical_batch_size\":"
              << logical_batch << ",\"seed_base\":" << seed_base
              << ",\"samples\":" << samples.size() << ",\"coin_heads\":" << coin
              << ",\"opening_inclusion\":" << opening << ",\"prize_inclusion\":" << prize
              << ",\"adjacent_fingerprint_duplicates\":" << adjacent_duplicates
              << ",\"coin_lag1\":" << lag1 << ",\"card_position_counts\":[";
    for (std::size_t index = 0; index < positions.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << positions[index];
    }
    std::cout << "],\"random_target_counts\":[";
    for (std::size_t index = 0; index < targets.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << targets[index];
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::uint32_t samples = 262144;
        std::uint64_t base = 0x003800000000ULL;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--samples" && index + 1 < argc) samples = static_cast<std::uint32_t>(std::stoul(argv[++index]));
            else if (argument == "--seed-base" && index + 1 < argc) base = std::stoull(argv[++index]);
            else throw std::runtime_error("unknown or incomplete argument: " + argument);
        }
        const std::array<std::uint32_t, 4> batches = {1, 8, 256, 512};
        for (std::size_t ordinal = 0; ordinal < batches.size(); ++ordinal) {
            const std::uint64_t cpu_seed = base + ordinal * 10'000'000ULL;
            const std::uint64_t cuda_seed = base + 100'000'000ULL + ordinal * 10'000'000ULL;
            emit_summary("official_cpu", batches[ordinal], cpu_seed, cpu_samples(cpu_seed, samples));
            emit_summary("cuda", batches[ordinal], cuda_seed, cuda_samples(cuda_seed, samples));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
