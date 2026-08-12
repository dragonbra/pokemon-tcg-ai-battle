#include "ptcg_cuda/official_rng.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDeckSize = 60;

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
        if (line.empty()) {
            continue;
        }
        cards.push_back(static_cast<std::uint16_t>(std::stoul(line)));
    }
    if (cards.size() != kDeckSize) {
        throw std::runtime_error("deck must contain exactly 60 cards: " + path);
    }
    return cards;
}

std::vector<std::uint64_t> parse_seeds(const std::string& value) {
    std::vector<std::uint64_t> seeds;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            seeds.push_back(std::stoull(item));
        }
    }
    if (seeds.empty()) {
        throw std::runtime_error("at least one seed is required");
    }
    return seeds;
}

std::vector<std::uint16_t> parse_card_ids(const std::string& value) {
    std::vector<std::uint16_t> ids;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            ids.push_back(static_cast<std::uint16_t>(std::stoul(item)));
        }
    }
    if (ids.empty()) {
        throw std::runtime_error("at least one basic card ID is required");
    }
    return ids;
}

__device__ bool is_basic(
    std::uint16_t card_id,
    const std::uint16_t* basic_ids,
    std::uint32_t basic_count) {
    for (std::uint32_t i = 0; i < basic_count; ++i) {
        if (basic_ids[i] == card_id) {
            return true;
        }
    }
    return false;
}

__global__ void setup_kernel(
    const std::uint16_t* input_decks,
    const std::uint64_t* seeds,
    const std::uint16_t* basic_ids,
    std::uint32_t basic_count,
    std::uint16_t* draw_orders,
    std::uint16_t* mulligan_counts,
    std::uint32_t count) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= count) {
        return;
    }

    std::uint16_t decks[2 * kDeckSize];
    for (std::uint32_t player = 0; player < 2; ++player) {
        for (std::uint32_t card = 0; card < kDeckSize; ++card) {
            decks[player * kDeckSize + (kDeckSize - card - 1)] =
                input_decks[player * kDeckSize + card];
        }
    }

    ptcg::cuda_engine::OfficialMt19937 rng;
    ptcg::cuda_engine::official_seed_mt19937(&rng, seeds[env]);
    ptcg::cuda_engine::official_shuffle_60(decks, &rng);
    ptcg::cuda_engine::official_shuffle_60(decks + kDeckSize, &rng);

    std::uint16_t hands[2 * 7];
    std::uint32_t deck_counts[2] = {60, 60};
    bool ready[2] = {false, false};
    for (std::uint32_t player = 0; player < 2; ++player) {
        for (std::uint32_t draw = 0; draw < 7; ++draw) {
            hands[player * 7 + draw] =
                decks[player * kDeckSize + --deck_counts[player]];
        }
        for (std::uint32_t card = 0; card < 7; ++card) {
            ready[player] = ready[player] || is_basic(
                hands[player * 7 + card], basic_ids, basic_count);
        }
    }

    for (std::uint32_t round = 0; round < 47 && (!ready[0] || !ready[1]); ++round) {
        for (std::uint32_t player = 0; player < 2; ++player) {
            if (ready[player]) {
                continue;
            }
            for (std::uint32_t hand_count = 7; hand_count > 0; --hand_count) {
                decks[player * kDeckSize + deck_counts[player]++] =
                    hands[player * 7 + hand_count - 1];
            }
            ptcg::cuda_engine::official_shuffle_60(
                decks + player * kDeckSize, &rng);
            ++mulligan_counts[env * 2 + player];
            for (std::uint32_t draw = 0; draw < 7; ++draw) {
                hands[player * 7 + draw] =
                    decks[player * kDeckSize + --deck_counts[player]];
            }
            for (std::uint32_t card = 0; card < 7; ++card) {
                ready[player] = ready[player] || is_basic(
                    hands[player * 7 + card], basic_ids, basic_count);
            }
        }
    }

    for (std::uint32_t player = 0; player < 2; ++player) {
        for (std::uint32_t draw = 0; draw < 7; ++draw) {
            draw_orders[(env * 2 + player) * kDeckSize + draw] =
                hands[player * 7 + draw];
        }
        for (std::uint32_t draw = 7; draw < kDeckSize; ++draw) {
            draw_orders[(env * 2 + player) * kDeckSize + draw] =
                decks[player * kDeckSize + deck_counts[player] - (draw - 7) - 1];
        }
    }
}

void print_array(const std::vector<std::uint16_t>& values, std::size_t offset) {
    std::cout << '[';
    for (std::size_t i = 0; i < kDeckSize; ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << values[offset + i];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string deck0_path;
        std::string deck1_path;
        std::string seed_text;
        std::string basic_id_text;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            if (argument == "--deck0") {
                deck0_path = argv[++i];
            } else if (argument == "--deck1") {
                deck1_path = argv[++i];
            } else if (argument == "--seeds") {
                seed_text = argv[++i];
            } else if (argument == "--basic-ids") {
                basic_id_text = argv[++i];
            } else {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }
        if (deck0_path.empty() || deck1_path.empty() || seed_text.empty() || basic_id_text.empty()) {
            throw std::runtime_error(
                "--deck0, --deck1, --seeds, and --basic-ids are required");
        }

        std::vector<std::uint16_t> decks = read_deck(deck0_path);
        const std::vector<std::uint16_t> deck1 = read_deck(deck1_path);
        decks.insert(decks.end(), deck1.begin(), deck1.end());
        const std::vector<std::uint64_t> seeds = parse_seeds(seed_text);
        const std::vector<std::uint16_t> basic_ids = parse_card_ids(basic_id_text);
        std::vector<std::uint16_t> draw_orders(seeds.size() * 2 * kDeckSize);
        std::vector<std::uint16_t> mulligan_counts(seeds.size() * 2, 0);

        std::uint16_t* device_decks = nullptr;
        std::uint64_t* device_seeds = nullptr;
        std::uint16_t* device_basic_ids = nullptr;
        std::uint16_t* device_draw_orders = nullptr;
        std::uint16_t* device_mulligan_counts = nullptr;
        check_cuda(cudaMalloc(&device_decks, decks.size() * sizeof(std::uint16_t)), "cudaMalloc decks");
        check_cuda(cudaMalloc(&device_seeds, seeds.size() * sizeof(std::uint64_t)), "cudaMalloc seeds");
        check_cuda(
            cudaMalloc(&device_basic_ids, basic_ids.size() * sizeof(std::uint16_t)),
            "cudaMalloc basic IDs");
        check_cuda(
            cudaMalloc(&device_draw_orders, draw_orders.size() * sizeof(std::uint16_t)),
            "cudaMalloc draw orders");
        check_cuda(
            cudaMalloc(
                &device_mulligan_counts,
                mulligan_counts.size() * sizeof(std::uint16_t)),
            "cudaMalloc mulligan counts");
        check_cuda(
            cudaMemcpy(
                device_decks,
                decks.data(),
                decks.size() * sizeof(std::uint16_t),
                cudaMemcpyHostToDevice),
            "copy decks");
        check_cuda(
            cudaMemcpy(
                device_seeds,
                seeds.data(),
                seeds.size() * sizeof(std::uint64_t),
                cudaMemcpyHostToDevice),
            "copy seeds");
        check_cuda(
            cudaMemcpy(
                device_basic_ids,
                basic_ids.data(),
                basic_ids.size() * sizeof(std::uint16_t),
                cudaMemcpyHostToDevice),
            "copy basic IDs");
        check_cuda(
            cudaMemset(
                device_mulligan_counts,
                0,
                mulligan_counts.size() * sizeof(std::uint16_t)),
            "clear mulligan counts");

        setup_kernel<<<(seeds.size() + 31) / 32, 32>>>(
            device_decks,
            device_seeds,
            device_basic_ids,
            static_cast<std::uint32_t>(basic_ids.size()),
            device_draw_orders,
            device_mulligan_counts,
            static_cast<std::uint32_t>(seeds.size()));
        check_cuda(cudaGetLastError(), "setup kernel launch");
        check_cuda(
            cudaMemcpy(
                draw_orders.data(),
                device_draw_orders,
                draw_orders.size() * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost),
            "copy draw orders");
        check_cuda(
            cudaMemcpy(
                mulligan_counts.data(),
                device_mulligan_counts,
                mulligan_counts.size() * sizeof(std::uint16_t),
                cudaMemcpyDeviceToHost),
            "copy mulligan counts");
        check_cuda(cudaFree(device_mulligan_counts), "cudaFree mulligan counts");
        check_cuda(cudaFree(device_draw_orders), "cudaFree draw orders");
        check_cuda(cudaFree(device_basic_ids), "cudaFree basic IDs");
        check_cuda(cudaFree(device_seeds), "cudaFree seeds");
        check_cuda(cudaFree(device_decks), "cudaFree decks");

        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout << "{\"device\":\"" << properties.name << "\",\"traces\":[";
        for (std::size_t env = 0; env < seeds.size(); ++env) {
            if (env != 0) {
                std::cout << ',';
            }
            std::cout << "{\"seed\":" << seeds[env] << ",\"player0_draw_order\":";
            print_array(draw_orders, (env * 2) * kDeckSize);
            std::cout << ",\"player1_draw_order\":";
            print_array(draw_orders, (env * 2 + 1) * kDeckSize);
            std::cout << ",\"mulligan_counts\":["
                      << mulligan_counts[env * 2] << ','
                      << mulligan_counts[env * 2 + 1] << ']';
            std::cout << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
