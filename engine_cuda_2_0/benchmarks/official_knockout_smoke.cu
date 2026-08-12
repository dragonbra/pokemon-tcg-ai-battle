#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptcg_cuda/official_knockout_pod.cuh"

namespace {

using namespace ptcg::cuda_engine;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::uint8_t> read_binary(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("failed to open rule pack");
    const std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("failed to read rule pack");
    }
    return bytes;
}

__host__ __device__ void add_card(
    OfficialStatePod* state,
    std::uint16_t instance,
    std::int32_t card_id,
    std::int32_t player,
    OfficialArea area) {
    OfficialCardStatePod* card = &state->cards[instance];
    card->card_id = card_id;
    card->player = static_cast<std::int8_t>(player);
    card->area = static_cast<std::uint8_t>(area);
    card->move_counter = ++state->move_counter;
    official_pod_push_zone_card(state, player, area, OfficialCardRefPod{instance});
}

__host__ __device__ void run_prize_and_replacement_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 501, seed);
    state->first_player = 0;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 7, 1, OfficialArea::kActive);
    add_card(state, 21, 8, 1, OfficialArea::kBench);
    add_card(state, 22, 1, 1, OfficialArea::kEnergy);
    state->cards[22].attach_move_counter = state->cards[20].move_counter;
    add_card(state, 30, 1242, 0, OfficialArea::kPrize);
    add_card(state, 31, 7, 0, OfficialArea::kPrize);
    add_card(state, 32, 8, 0, OfficialArea::kPrize);
    add_card(state, 40, 1242, 1, OfficialArea::kPrize);
    state->cards[20].damage = official_pod_max_hp(
        state, rules, OfficialCardRefPod{20});
    state->cards[20].runtime_flags |= kCardKo;

    OfficialKnockoutResult result = official_begin_knockout(state, rules);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->reserved_flow != static_cast<std::uint8_t>(
            OfficialKnockoutStage::kPrizeSelection)
        || state->select_player != 0
        || state->select_min != 1
        || state->options.count != 3) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5101);
        return;
    }
    const std::uint16_t prize = 1;
    result = official_resume_knockout(state, rules, &prize, 1);
    if (result != OfficialKnockoutResult::kComplete
        || state->reserved_flow != 0
        || state->players[0].hand.count != 1
        || state->players[0].hand.values[0].index != 31) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5102);
        return;
    }
    // Official KOProc3 returns to AfterRefresh after prizes. A fresh Refresh /
    // ActiveCheck round then yields the replacement selection.
    result = official_begin_knockout(state, rules);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->reserved_flow != static_cast<std::uint8_t>(
            OfficialKnockoutStage::kActiveReplacement)
        || state->select_player != 1) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5104);
        return;
    }
    const std::uint16_t replacement = 0;
    result = official_resume_knockout(state, rules, &replacement, 1);
    if (result != OfficialKnockoutResult::kComplete
        || state->reserved_flow != 0
        || state->players[1].active.count != 1
        || state->players[1].active.values[0].index != 21
        || state->players[1].bench.count != 0
        || state->players[1].trash.count != 2
        || (state->cards[21].runtime_flags & kCardBenchToActive) == 0
        || (state->cards[21].turn_state[1] & (1U << 26U)) == 0
        || state->game_result != static_cast<std::uint8_t>(OfficialGameResult::kNone)) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5103);
    }
}

__host__ __device__ void run_multi_prize_terminal_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 502, seed);
    state->first_player = 0;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 24, 1, OfficialArea::kActive);
    add_card(state, 30, 1242, 0, OfficialArea::kPrize);
    add_card(state, 31, 7, 0, OfficialArea::kPrize);
    add_card(state, 40, 1242, 1, OfficialArea::kPrize);
    state->cards[20].damage = official_pod_max_hp(
        state, rules, OfficialCardRefPod{20});
    state->cards[20].runtime_flags |= kCardKo;

    OfficialKnockoutResult result = official_begin_knockout(state, rules);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->select_player != 0
        || state->select_min != 2
        || state->select_max != 2
        || state->options.count != 2) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5201);
        return;
    }
    const std::uint16_t prizes[2]{1, 0};
    result = official_resume_knockout(state, rules, prizes, 2);
    if (result != OfficialKnockoutResult::kComplete
        || state->players[0].prize.count != 0
        || state->players[0].hand.count != 2
        || state->players[1].active.count != 0
        || state->players[1].trash.count != 1
        || state->game_result != static_cast<std::uint8_t>(
            OfficialGameResult::kPlayer0Win)
        || state->finish_reason != static_cast<std::uint8_t>(
            OfficialFinishReason::kNoActivePokemon)) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5202);
    }
}

__host__ __device__ void run_lucky_bonus_scenario(
    OfficialStatePod* state) {
    OfficialRulePackHeader header{};
    header.counts[static_cast<std::uint32_t>(OfficialRuleSection::kCards)] = 4;
    header.counts[static_cast<std::uint32_t>(OfficialRuleSection::kSkills)] = 1;
    OfficialCardRule cards[4]{};
    for (std::int32_t index = 0; index < 4; ++index) {
        cards[index].values[kCardId] = index + 1;
    }
    cards[0].values[kCardAbilityId] = 1;
    cards[1].values[kCardHp] = 100;
    OfficialSkillRule skills[1]{};
    skills[0].values[kSkillId] = 1;
    skills[0].flags = kOfficialSkillLuckyBonusFlag;
    OfficialRulePackView rules{};
    rules.header = &header;
    rules.cards = cards;
    rules.skills = skills;

    official_pod_reset(state, 503, 2);
    state->first_player = 0;
    state->turn = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 2, 0, OfficialArea::kActive);
    add_card(state, 20, 2, 1, OfficialArea::kActive);
    add_card(state, 30, 1, 0, OfficialArea::kPrize);
    state->cards[30].reverse = 1;
    add_card(state, 31, 3, 0, OfficialArea::kPrize);
    add_card(state, 40, 3, 1, OfficialArea::kPrize);
    state->cards[20].damage = 100;
    state->cards[20].runtime_flags |= kCardKo;

    OfficialKnockoutResult result = official_begin_knockout(state, rules);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->reserved_flow != static_cast<std::uint8_t>(
            OfficialKnockoutStage::kPrizeSelection)
        || state->options.count != 2) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5301);
        return;
    }
    const std::uint16_t lucky_prize = 0;
    result = official_resume_knockout(state, rules, &lucky_prize, 1);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->reserved_flow != static_cast<std::uint8_t>(
            OfficialKnockoutStage::kLuckyBonusChoice)
        || state->players[0].temporary.count != 1
        || state->players[0].temporary.values[0].index != 30
        || state->options.count != 2) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5302);
        return;
    }
    const std::uint16_t yes = 0;
    result = official_resume_knockout(state, rules, &yes, 1);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->reserved_flow != static_cast<std::uint8_t>(
            OfficialKnockoutStage::kPrizeSelection)
        || state->players[0].bench.count != 1
        || state->players[0].bench.values[0].index != 30
        || state->players[0].temporary.count != 0
        || state->rng.draw_count != 1
        || state->options.count != 1) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5303);
        return;
    }
    const std::uint16_t extra_prize = 0;
    result = official_resume_knockout(state, rules, &extra_prize, 1);
    if (result != OfficialKnockoutResult::kComplete
        || state->players[0].prize.count != 0
        || state->players[0].hand.count != 1
        || state->players[0].hand.values[0].index != 31
        || state->selected_list.count != 0
        || state->pending_prize_count[0] != 0
        || state->error != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5304);
    }
}

__host__ __device__ void run_multi_ko_order_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    official_pod_reset(state, 504, 7);
    state->first_player = 0;
    state->turn = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 7, 1, OfficialArea::kActive);
    add_card(state, 21, 24, 1, OfficialArea::kBench);
    for (std::uint16_t index = 30; index < 34; ++index) {
        add_card(state, index, 7, 0, OfficialArea::kPrize);
    }
    add_card(state, 50, 7, 1, OfficialArea::kPrize);
    state->cards[20].damage = official_pod_max_hp(state, rules, OfficialCardRefPod{20});
    state->cards[20].runtime_flags |= kCardKo;
    state->cards[21].damage = official_pod_max_hp(state, rules, OfficialCardRefPod{21});
    state->cards[21].runtime_flags |= kCardKo;

    OfficialKnockoutResult result = official_begin_knockout(state, rules);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->reserved_flow != static_cast<std::uint8_t>(OfficialKnockoutStage::kPrizeSelection)
        || state->select_min != 1 || state->options.count != 4) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5401);
        state->error_detail = 54010000 + static_cast<int>(result) * 1000000
            + static_cast<int>(state->reserved_flow) * 10000
            + state->select_min * 100 + state->options.count;
        return;
    }
    state->attack_flow_flags = static_cast<std::uint8_t>(state->select_min);
    const std::uint16_t first = 0;
    result = official_resume_knockout(state, rules, &first, 1);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->reserved_flow != static_cast<std::uint8_t>(OfficialKnockoutStage::kPrizeSelection)
        || state->select_min != 2 || state->select_max != 2
        || state->options.count != 3) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5402);
        return;
    }
    const std::uint16_t second[2]{0, 1};
    result = official_resume_knockout(state, rules, second, 2);
    if (result != OfficialKnockoutResult::kComplete
        || state->players[0].hand.count != 3
        || state->players[0].prize.count != 1
        || state->game_result != static_cast<std::uint8_t>(OfficialGameResult::kPlayer0Win)
        || state->finish_reason != static_cast<std::uint8_t>(OfficialFinishReason::kNoActivePokemon)) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5403);
        state->error_detail = 54030000 + static_cast<int>(result) * 1000000
            + static_cast<int>(state->game_result) * 10000
            + state->players[0].hand.count * 100 + state->players[0].prize.count;
    }
}

__host__ __device__ void run_multi_lucky_interleave_scenario(
    OfficialStatePod* state) {
    OfficialRulePackHeader header{};
    header.counts[static_cast<std::uint32_t>(OfficialRuleSection::kCards)] = 4;
    header.counts[static_cast<std::uint32_t>(OfficialRuleSection::kSkills)] = 1;
    OfficialCardRule cards[4]{};
    for (std::int32_t index = 0; index < 4; ++index) cards[index].values[kCardId] = index + 1;
    cards[0].values[kCardAbilityId] = 1;
    cards[1].values[kCardHp] = 100;
    cards[1].values[kCardPokemonType] = 3;
    OfficialSkillRule skills[1]{};
    skills[0].values[kSkillId] = 1;
    skills[0].flags = kOfficialSkillLuckyBonusFlag;
    OfficialRulePackView rules{};
    rules.header = &header;
    rules.cards = cards;
    rules.skills = skills;

    official_pod_reset(state, 505, 2);
    state->first_player = 0;
    state->turn = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 2, 0, OfficialArea::kActive);
    add_card(state, 20, 2, 1, OfficialArea::kActive);
    add_card(state, 30, 1, 0, OfficialArea::kPrize);
    state->cards[30].reverse = 1;
    add_card(state, 31, 1, 0, OfficialArea::kPrize);
    state->cards[31].reverse = 1;
    add_card(state, 32, 3, 0, OfficialArea::kPrize);
    add_card(state, 33, 3, 0, OfficialArea::kPrize);
    state->cards[20].damage = 100;
    state->cards[20].runtime_flags |= kCardKo;

    OfficialKnockoutResult result = official_begin_knockout(state, rules);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->select_min != 2 || state->options.count != 4) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5501);
        return;
    }
    const std::uint16_t initial[2]{0, 1};
    result = official_resume_knockout(state, rules, initial, 2);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->reserved_flow != static_cast<std::uint8_t>(OfficialKnockoutStage::kLuckyBonusChoice)
        || state->context_card.index != 30 || state->selected_list.count != 2) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5502);
        return;
    }
    const std::uint16_t yes = 0;
    result = official_resume_knockout(state, rules, &yes, 1);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->reserved_flow != static_cast<std::uint8_t>(OfficialKnockoutStage::kPrizeSelection)
        || state->select_min != 1
        || state->players[0].bench.count != 1 || state->rng.draw_count != 1
        || state->selected_list.count != 1) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5503);
        state->error_detail = 55030000 + static_cast<int>(result) * 1000000
            + static_cast<int>(state->reserved_flow) * 10000
            + state->context_card.index * 100 + state->selected_list.count;
        return;
    }
    const std::uint16_t extra = 0;
    result = official_resume_knockout(state, rules, &extra, 1);
    if (result != OfficialKnockoutResult::kNeedsAction
        || state->reserved_flow != static_cast<std::uint8_t>(OfficialKnockoutStage::kLuckyBonusChoice)
        || state->context_card.index != 31 || state->selected_list.count != 1) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5504);
        return;
    }
    const std::uint16_t no = 1;
    result = official_resume_knockout(state, rules, &no, 1);
    if (result != OfficialKnockoutResult::kComplete
        || state->players[0].bench.count != 1
        || state->players[0].hand.count != 2
        || state->players[0].prize.count != 1
        || state->error != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 5505);
    }
}

__global__ void run_kernel(
    OfficialStatePod* states,
    const std::uint8_t* rule_pack,
    std::uint64_t seed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
        run_prize_and_replacement_scenario(&states[0], rules, seed);
        run_multi_prize_terminal_scenario(&states[1], rules, seed + 1);
        run_lucky_bonus_scenario(&states[2]);
        run_multi_ko_order_scenario(&states[3], rules);
        run_multi_lucky_interleave_scenario(&states[4]);
    }
}

std::uint64_t digest(const OfficialStatePod* states, std::size_t count) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(states);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t index = 0; index < sizeof(OfficialStatePod) * count; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: official_knockout_smoke <official_rules.bin>");
        }
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        const OfficialRulePackView host_rules = make_official_rule_pack_view(
            rule_pack.data());
        if (rule_pack.size() < sizeof(OfficialRulePackHeader)
            || std::memcmp(host_rules.header->magic, "PTCGRUL1", 8) != 0) {
            throw std::runtime_error("invalid rule pack");
        }

        constexpr std::uint64_t seed = 20260730ULL;
        OfficialStatePod cpu_states[5]{};
        run_prize_and_replacement_scenario(&cpu_states[0], host_rules, seed);
        run_multi_prize_terminal_scenario(&cpu_states[1], host_rules, seed + 1);
        run_lucky_bonus_scenario(&cpu_states[2]);
        run_multi_ko_order_scenario(&cpu_states[3], host_rules);
        run_multi_lucky_interleave_scenario(&cpu_states[4]);

        std::uint8_t* device_rules = nullptr;
        OfficialStatePod* device_states = nullptr;
        check_cuda(cudaMalloc(&device_rules, rule_pack.size()), "cudaMalloc rules");
        check_cuda(cudaMalloc(&device_states, sizeof(cpu_states)), "cudaMalloc states");
        check_cuda(cudaMemcpy(
            device_rules,
            rule_pack.data(),
            rule_pack.size(),
            cudaMemcpyHostToDevice), "copy rules");
        check_cuda(cudaMemset(device_states, 0, sizeof(cpu_states)), "clear states");
        run_kernel<<<1, 1>>>(device_states, device_rules, seed);
        check_cuda(cudaGetLastError(), "launch knockout smoke");
        OfficialStatePod gpu_states[5]{};
        check_cuda(cudaMemcpy(
            gpu_states,
            device_states,
            sizeof(gpu_states),
            cudaMemcpyDeviceToHost), "copy states");
        check_cuda(cudaFree(device_states), "cudaFree states");
        check_cuda(cudaFree(device_rules), "cudaFree rules");

        const bool equal = std::memcmp(cpu_states, gpu_states, sizeof(cpu_states)) == 0;
        const bool passed = equal
            && gpu_states[0].error == 0
            && gpu_states[1].error == 0
            && gpu_states[2].error == 0
            && gpu_states[3].error == 0
            && gpu_states[4].error == 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"device\":\"" << properties.name << "\""
            << ",\"state_abi\":" << kOfficialStateAbiVersion
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"cpu_digest\":" << digest(cpu_states, 5)
            << ",\"gpu_digest\":" << digest(gpu_states, 5)
            << ",\"replacement_active\":"
            << gpu_states[0].players[1].active.values[0].index
            << ",\"replacement_bench_to_active_runtime\":"
            << ((gpu_states[0].cards[21].runtime_flags & kCardBenchToActive) != 0)
            << ",\"replacement_bench_to_active_turn_state\":"
            << ((gpu_states[0].cards[21].turn_state[1] & (1U << 26U)) != 0)
            << ",\"ko_trash_count\":" << gpu_states[0].players[1].trash.count
            << ",\"multi_prize_hand_count\":" << gpu_states[1].players[0].hand.count
            << ",\"terminal_result\":"
            << static_cast<unsigned>(gpu_states[1].game_result)
            << ",\"terminal_reason\":"
            << static_cast<unsigned>(gpu_states[1].finish_reason)
            << ",\"lucky_bonus_bench\":" << gpu_states[2].players[0].bench.count
            << ",\"lucky_bonus_extra_hand\":" << gpu_states[2].players[0].hand.count
            << ",\"lucky_bonus_rng_draws\":" << gpu_states[2].rng.draw_count
            << ",\"multi_ko_first_select_min\":"
            << static_cast<unsigned>(gpu_states[3].attack_flow_flags)
            << ",\"multi_ko_hand_count\":" << gpu_states[3].players[0].hand.count
            << ",\"multi_lucky_extra_before_second\":"
            << (gpu_states[4].rng.draw_count == 1 ? 1 : 0)
            << ",\"errors\":[" << gpu_states[0].error << ','
            << gpu_states[1].error << ',' << gpu_states[2].error << ','
            << gpu_states[3].error << ',' << gpu_states[4].error << "]"
            << ",\"error_details\":[" << gpu_states[0].error_detail << ','
            << gpu_states[1].error_detail << ',' << gpu_states[2].error_detail << ','
            << gpu_states[3].error_detail << ',' << gpu_states[4].error_detail << "]}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
