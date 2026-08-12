#include "ptcg_cuda/official_setup_pod.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ptcg::cuda_engine::SetupBattleState;
using ptcg::cuda_engine::SetupCardMeta;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::uint16_t> read_deck(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open deck: " + path);
    }
    std::vector<std::uint16_t> cards;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t comma = line.find(',');
        const std::string token = line.substr(0, comma);
        try {
            std::size_t parsed = 0;
            const unsigned long value = std::stoul(token, &parsed);
            if (parsed == token.size()) {
                cards.push_back(static_cast<std::uint16_t>(value));
            }
        } catch (const std::exception&) {
        }
    }
    if (cards.size() != 2 * ptcg::cuda_engine::kOfficialDeckSize
        && cards.size() != ptcg::cuda_engine::kOfficialDeckSize) {
        throw std::runtime_error("deck file has an invalid card count: " + path);
    }
    return cards;
}

std::vector<SetupCardMeta> read_metadata(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open metadata: " + path);
    }
    std::vector<SetupCardMeta> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream stream(line);
        std::string item;
        std::array<int, 4> values{};
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (!std::getline(stream, item, ',')) {
                throw std::runtime_error("invalid setup metadata row");
            }
            values[i] = std::stoi(item);
        }
        if (values[0] != static_cast<int>(result.size())) {
            throw std::runtime_error("setup metadata IDs must be dense and zero-based");
        }
        result.push_back({
            static_cast<std::uint8_t>(values[1]),
            static_cast<std::uint8_t>(values[2]),
            static_cast<std::uint8_t>(values[3]),
            0,
        });
    }
    if (result.empty()) {
        throw std::runtime_error("setup metadata is empty");
    }
    return result;
}

__global__ void setup_kernel(
    SetupBattleState* states,
    const std::uint16_t* decks,
    std::uint64_t seed_start,
    const SetupCardMeta* metadata,
    std::uint32_t metadata_count,
    std::uint32_t count) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= count) {
        return;
    }
    ptcg::cuda_engine::official_setup_first_min(
        &states[env], nullptr, decks, seed_start + env, metadata, metadata_count);
}

std::size_t first_mismatch_byte(
    const SetupBattleState& expected,
    const SetupBattleState& actual) {
    const auto* left = reinterpret_cast<const std::uint8_t*>(&expected);
    const auto* right = reinterpret_cast<const std::uint8_t*>(&actual);
    for (std::size_t i = 0; i < sizeof(SetupBattleState); ++i) {
        if (left[i] != right[i]) {
            return i;
        }
    }
    return sizeof(SetupBattleState);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string deck0_path;
        std::string deck1_path;
        std::string metadata_path;
        std::uint64_t seed_start = 1;
        std::uint32_t seed_count = 10'000;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            if (argument == "--deck0") {
                deck0_path = argv[++i];
            } else if (argument == "--deck1") {
                deck1_path = argv[++i];
            } else if (argument == "--metadata") {
                metadata_path = argv[++i];
            } else if (argument == "--seed-start") {
                seed_start = std::stoull(argv[++i]);
            } else if (argument == "--seed-count") {
                seed_count = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            } else {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }
        if (deck0_path.empty() || deck1_path.empty() || metadata_path.empty()
            || seed_count == 0) {
            throw std::runtime_error(
                "--deck0, --deck1, --metadata, and positive --seed-count are required");
        }

        std::vector<std::uint16_t> decks = read_deck(deck0_path);
        const std::vector<std::uint16_t> deck1 = read_deck(deck1_path);
        if (decks.size() != ptcg::cuda_engine::kOfficialDeckSize
            || deck1.size() != ptcg::cuda_engine::kOfficialDeckSize) {
            throw std::runtime_error("each deck must contain exactly 60 cards");
        }
        decks.insert(decks.end(), deck1.begin(), deck1.end());
        const std::vector<SetupCardMeta> metadata = read_metadata(metadata_path);

        SetupBattleState* device_states = nullptr;
        std::uint16_t* device_decks = nullptr;
        SetupCardMeta* device_metadata = nullptr;
        check_cuda(
            cudaMalloc(&device_states, seed_count * sizeof(SetupBattleState)),
            "cudaMalloc states");
        check_cuda(
            cudaMalloc(&device_decks, decks.size() * sizeof(std::uint16_t)),
            "cudaMalloc decks");
        check_cuda(
            cudaMalloc(&device_metadata, metadata.size() * sizeof(SetupCardMeta)),
            "cudaMalloc metadata");
        check_cuda(
            cudaMemcpy(
                device_decks, decks.data(), decks.size() * sizeof(std::uint16_t),
                cudaMemcpyHostToDevice),
            "copy decks");
        check_cuda(
            cudaMemcpy(
                device_metadata, metadata.data(), metadata.size() * sizeof(SetupCardMeta),
                cudaMemcpyHostToDevice),
            "copy metadata");

        cudaEvent_t start{};
        cudaEvent_t stop{};
        check_cuda(cudaEventCreate(&start), "create start event");
        check_cuda(cudaEventCreate(&stop), "create stop event");
        check_cuda(cudaEventRecord(start), "record start event");
        setup_kernel<<<(seed_count + 127) / 128, 128>>>(
            device_states,
            device_decks,
            seed_start,
            device_metadata,
            static_cast<std::uint32_t>(metadata.size()),
            seed_count);
        check_cuda(cudaGetLastError(), "setup kernel launch");
        check_cuda(cudaEventRecord(stop), "record stop event");
        check_cuda(cudaEventSynchronize(stop), "synchronize stop event");
        float kernel_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&kernel_ms, start, stop), "elapsed time");

        std::vector<SetupBattleState> gpu_states(seed_count);
        check_cuda(
            cudaMemcpy(
                gpu_states.data(), device_states,
                seed_count * sizeof(SetupBattleState), cudaMemcpyDeviceToHost),
            "copy states");
        check_cuda(cudaEventDestroy(stop), "destroy stop event");
        check_cuda(cudaEventDestroy(start), "destroy start event");
        check_cuda(cudaFree(device_metadata), "free metadata");
        check_cuda(cudaFree(device_decks), "free decks");
        check_cuda(cudaFree(device_states), "free states");

        std::vector<SetupBattleState> cpu_states(seed_count);
        const auto cpu_start = std::chrono::steady_clock::now();
        for (std::uint32_t env = 0; env < seed_count; ++env) {
            ptcg::cuda_engine::official_setup_first_min(
                &cpu_states[env], nullptr, decks.data(), seed_start + env,
                metadata.data(), static_cast<std::uint32_t>(metadata.size()));
        }
        const auto cpu_stop = std::chrono::steady_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(
            cpu_stop - cpu_start).count();

        std::uint32_t mismatches = 0;
        std::uint32_t first_mismatch_env = seed_count;
        std::size_t first_mismatch = sizeof(SetupBattleState);
        for (std::uint32_t env = 0; env < seed_count; ++env) {
            const std::size_t byte = first_mismatch_byte(cpu_states[env], gpu_states[env]);
            if (byte != sizeof(SetupBattleState)) {
                ++mismatches;
                if (first_mismatch_env == seed_count) {
                    first_mismatch_env = env;
                    first_mismatch = byte;
                }
            }
        }

        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "get device properties");
        const double gpu_envs_per_second = seed_count * 1000.0 / kernel_ms;
        const double cpu_envs_per_second = seed_count * 1000.0 / cpu_ms;
        std::cout
            << "{\"passed\":" << (mismatches == 0 ? "true" : "false")
            << ",\"device\":\"" << properties.name << "\""
            << ",\"seed_start\":" << seed_start
            << ",\"seed_count\":" << seed_count
            << ",\"state_bytes\":" << sizeof(SetupBattleState)
            << ",\"state_pool_bytes\":" << seed_count * sizeof(SetupBattleState)
            << ",\"kernel_ms\":" << kernel_ms
            << ",\"cpu_ms\":" << cpu_ms
            << ",\"gpu_envs_per_second\":" << gpu_envs_per_second
            << ",\"cpu_envs_per_second\":" << cpu_envs_per_second
            << ",\"mismatches\":" << mismatches;
        if (mismatches != 0) {
            std::cout
                << ",\"first_mismatch_env\":" << first_mismatch_env
                << ",\"first_mismatch_byte\":" << first_mismatch;
        }
        std::cout << "}\n";
        return mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
