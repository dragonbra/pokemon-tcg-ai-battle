#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptcg_cuda/official_trigger_resolver_pod.cuh"

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

__host__ __device__ void add_pending(
    OfficialStatePod* state,
    std::int32_t skill_id,
    OfficialCardRefPod effect_card,
    std::uint8_t trigger_type,
    std::int8_t depth = 0) {
    OfficialTriggeredAbilityPod pending{};
    pending.activate.skill_id = skill_id;
    pending.activate.effect_card = official_pod_area_ref(state, effect_card);
    pending.activate.use_player = state->cards[effect_card.index].player;
    pending.trigger.type = trigger_type;
    pending.trigger.depth = depth;
    official_pod_push(
        state,
        &state->temporary_triggers,
        pending,
        OfficialPodError::kTriggerStackOverflow);
}

__host__ __device__ void run_order_and_optional_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 401, seed);
    state->first_player = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 106, 0, OfficialArea::kBench);
    add_card(state, 11, 340, 0, OfficialArea::kActive);
    add_card(state, 20, 7, 0, OfficialArea::kDeck);
    add_pending(state, 39, OfficialCardRefPod{10}, 4);
    add_pending(state, 113, OfficialCardRefPod{11}, 6);

    OfficialTriggerResolverResult result = official_begin_trigger_resolution(
        state, rules, 0);
    if (result != OfficialTriggerResolverResult::kNeedsAction
        || state->select_context != kOfficialSelectContextSkillOrder
        || state->select_player != 0
        || state->options.count != 2) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4101);
        return;
    }
    const std::uint16_t order[2]{1, 0};
    result = official_resume_trigger_resolution(state, rules, order, 2);
    if (result != OfficialTriggerResolverResult::kNeedsAction
        || state->trigger_resolver.current.activate.skill_id != 113
        || state->trigger_resolver.current.activate.effect_card.card.index != 11
        || state->select_context != kOfficialSelectContextActivate) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4102);
        return;
    }
    const std::uint16_t no = 1;
    result = official_resume_trigger_resolution(state, rules, &no, 1);
    if (result != OfficialTriggerResolverResult::kNeedsAction
        || state->trigger_resolver.current.activate.skill_id != 39
        || state->trigger_resolver.current.activate.effect_card.card.index != 10) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4103);
        return;
    }
    result = official_resume_trigger_resolution(state, rules, &no, 1);
    if (result != OfficialTriggerResolverResult::kComplete
        || state->trigger_resolver.active != 0
        || state->temporary_triggers.count != 0
        || state->triggers.count != 0
        || state->turn_used_skills.count != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4104);
    }
}

__host__ __device__ void run_effect_resume_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 402, seed);
    state->first_player = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 72, 0, OfficialArea::kActive);
    add_card(state, 11, 1242, 0, OfficialArea::kDeck);
    add_card(state, 12, 7, 0, OfficialArea::kDeck);
    add_pending(state, 24, OfficialCardRefPod{10}, 1);

    OfficialTriggerResolverResult result = official_begin_trigger_resolution(
        state, rules, 0);
    if (result != OfficialTriggerResolverResult::kNeedsAction
        || state->effect_interpreter.awaiting_selection == 0
        || state->options.count != 1
        || state->options.values[0].resolved_card != 11) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4201);
        return;
    }
    const std::uint16_t action = 0;
    result = official_resume_trigger_resolution(state, rules, &action, 1);
    if (result != OfficialTriggerResolverResult::kComplete
        || state->players[0].hand.count != 1
        || state->players[0].hand.values[0].index != 11
        || state->players[0].deck.count != 1
        || state->turn_used_skills.count != 1
        || state->turn_used_skills.values[0] != 24
        || state->cards[10].ability_used_count != 1) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4202);
    }
}

__host__ __device__ void run_stale_source_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 403, seed);
    state->first_player = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 72, 0, OfficialArea::kActive);
    add_pending(state, 24, OfficialCardRefPod{10}, 1);
    ++state->cards[10].move_counter;
    const OfficialTriggerResolverResult result = official_begin_trigger_resolution(
        state, rules, 0);
    if (result != OfficialTriggerResolverResult::kComplete
        || state->turn_used_skills.count != 0
        || state->options.count != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4301);
    }
}

__host__ __device__ void run_no_ability_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 404, seed);
    state->first_player = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 72, 0, OfficialArea::kActive);
    state->cards[10].continual_state[4] |= 1ULL;
    add_pending(state, 24, OfficialCardRefPod{10}, 1);
    const OfficialTriggerResolverResult result = official_begin_trigger_resolution(
        state, rules, 0);
    if (result != OfficialTriggerResolverResult::kComplete
        || state->turn_used_skills.count != 0
        || state->options.count != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4401);
    }
}

__host__ __device__ void run_second_effect_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 405, seed);
    state->first_player = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 1191, 0, OfficialArea::kPlaying);
    add_card(state, 11, 7, 0, OfficialArea::kBench);
    add_pending(state, 357, OfficialCardRefPod{10}, 1);
    OfficialTriggerResolverResult result = official_begin_trigger_resolution(
        state, rules, 0);
    if (result != OfficialTriggerResolverResult::kNeedsAction
        || state->select_context != kOfficialSelectContextFirstEffect
        || state->select_player != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4501);
        return;
    }
    const std::uint16_t no = 1;
    result = official_resume_trigger_resolution(state, rules, &no, 1);
    if (result != OfficialTriggerResolverResult::kComplete
        || state->turn_used_skills.count != 1
        || state->turn_used_skills.values[0] != 357) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4502);
    }
}

__host__ __device__ void run_enemy_second_effect_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 406, seed);
    state->first_player = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_card(state, 10, 1226, 0, OfficialArea::kPlaying);
    add_card(state, 20, 7, 0, OfficialArea::kDeck);
    add_card(state, 21, 7, 0, OfficialArea::kDeck);
    add_card(state, 22, 7, 0, OfficialArea::kDeck);
    add_card(state, 23, 7, 0, OfficialArea::kDeck);
    add_pending(state, 393, OfficialCardRefPod{10}, 1);
    OfficialTriggerResolverResult result = official_begin_trigger_resolution(
        state, rules, 0);
    if (result != OfficialTriggerResolverResult::kNeedsAction
        || state->select_context != kOfficialSelectContextActivate
        || state->select_player != 1) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4601);
        return;
    }
    const std::uint16_t no = 1;
    result = official_resume_trigger_resolution(state, rules, &no, 1);
    if (result != OfficialTriggerResolverResult::kComplete
        || state->turn_used_skills.count != 1
        || state->turn_used_skills.values[0] != 393
        || state->players[0].hand.count != 4
        || state->players[0].deck.count != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4602);
    }
}

__host__ __device__ void add_special_condition_trigger(OfficialStatePod* state) {
    OfficialTriggeredAbilityPod pending{};
    pending.activate.is_special_condition = 1;
    pending.trigger.depth = 0;
    official_pod_push(
        state,
        &state->temporary_triggers,
        pending,
        OfficialPodError::kTriggerStackOverflow);
}

__host__ __device__ void run_special_condition_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 407, seed);
    state->first_player = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kPokemonCheckup);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 7, 1, OfficialArea::kActive);
    official_pod_set_poison_counter(&state->players[0], 2);
    official_pod_set_burned(&state->players[0], true);
    official_pod_set_bad_status(&state->players[0], OfficialBadStatus::kAsleep);
    official_pod_set_poison_counter(&state->players[1], 1);
    add_special_condition_trigger(state);
    const OfficialTriggerResolverResult result = official_begin_trigger_resolution(
        state, rules, 0);
    if (result != OfficialTriggerResolverResult::kComplete
        || state->cards[10].damage != 40
        || state->cards[20].damage != 10
        || state->rng.draw_count != 2
        || state->trigger_resolver.active != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4701);
    }
}

__host__ __device__ void run_paralyze_recovery_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 408, seed);
    state->first_player = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kPokemonCheckup);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 7, 1, OfficialArea::kActive);
    official_pod_set_bad_status(&state->players[0], OfficialBadStatus::kParalyzed);
    add_special_condition_trigger(state);
    const OfficialTriggerResolverResult result = official_begin_trigger_resolution(
        state, rules, 0);
    if (result != OfficialTriggerResolverResult::kComplete
        || official_pod_bad_status(state->players[0]) != OfficialBadStatus::kNone
        || state->rng.draw_count != 0) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 4801);
    }
}

__global__ void run_kernel(
    OfficialStatePod* states,
    const std::uint8_t* rule_pack,
    std::uint64_t seed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
        run_order_and_optional_scenario(&states[0], rules, seed);
        run_effect_resume_scenario(&states[1], rules, seed + 1);
        run_stale_source_scenario(&states[2], rules, seed + 2);
        run_no_ability_scenario(&states[3], rules, seed + 3);
        run_second_effect_scenario(&states[4], rules, seed + 4);
        run_enemy_second_effect_scenario(&states[5], rules, seed + 5);
        run_special_condition_scenario(&states[6], rules, seed + 6);
        run_paralyze_recovery_scenario(&states[7], rules, seed + 7);
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
            throw std::runtime_error(
                "usage: official_trigger_resolver_smoke <official_rules.bin>");
        }
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        const OfficialRulePackView host_rules = make_official_rule_pack_view(
            rule_pack.data());
        if (rule_pack.size() < sizeof(OfficialRulePackHeader)
            || std::memcmp(host_rules.header->magic, "PTCGRUL1", 8) != 0) {
            throw std::runtime_error("invalid rule pack");
        }

        constexpr std::uint64_t seed = 20260730ULL;
        OfficialStatePod cpu_states[8]{};
        run_order_and_optional_scenario(&cpu_states[0], host_rules, seed);
        run_effect_resume_scenario(&cpu_states[1], host_rules, seed + 1);
        run_stale_source_scenario(&cpu_states[2], host_rules, seed + 2);
        run_no_ability_scenario(&cpu_states[3], host_rules, seed + 3);
        run_second_effect_scenario(&cpu_states[4], host_rules, seed + 4);
        run_enemy_second_effect_scenario(&cpu_states[5], host_rules, seed + 5);
        run_special_condition_scenario(&cpu_states[6], host_rules, seed + 6);
        run_paralyze_recovery_scenario(&cpu_states[7], host_rules, seed + 7);

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
        check_cuda(cudaGetLastError(), "launch trigger resolver smoke");
        OfficialStatePod gpu_states[8]{};
        check_cuda(cudaMemcpy(
            gpu_states,
            device_states,
            sizeof(gpu_states),
            cudaMemcpyDeviceToHost), "copy states");
        check_cuda(cudaFree(device_states), "cudaFree states");
        check_cuda(cudaFree(device_rules), "cudaFree rules");

        const bool equal = std::memcmp(cpu_states, gpu_states, sizeof(cpu_states)) == 0;
        bool passed = equal;
        for (const OfficialStatePod& state : gpu_states) passed = passed && state.error == 0;
        std::size_t first_difference = sizeof(cpu_states);
        std::size_t difference_count = 0;
        const auto* cpu_bytes = reinterpret_cast<const std::uint8_t*>(cpu_states);
        const auto* gpu_bytes = reinterpret_cast<const std::uint8_t*>(gpu_states);
        for (std::size_t index = 0; index < sizeof(cpu_states); ++index) {
            if (cpu_bytes[index] == gpu_bytes[index]) continue;
            if (first_difference == sizeof(cpu_states)) first_difference = index;
            ++difference_count;
        }
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"device\":\"" << properties.name << "\""
            << ",\"state_abi\":" << kOfficialStateAbiVersion
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"cpu_digest\":" << digest(cpu_states, 8)
            << ",\"gpu_digest\":" << digest(gpu_states, 8)
            << ",\"cpu_state_digests\":[" << digest(&cpu_states[0], 1) << ','
            << digest(&cpu_states[1], 1) << ',' << digest(&cpu_states[2], 1) << ','
            << digest(&cpu_states[3], 1) << ',' << digest(&cpu_states[4], 1) << ','
            << digest(&cpu_states[5], 1) << ',' << digest(&cpu_states[6], 1) << ','
            << digest(&cpu_states[7], 1) << ']'
            << ",\"gpu_state_digests\":[" << digest(&gpu_states[0], 1) << ','
            << digest(&gpu_states[1], 1) << ',' << digest(&gpu_states[2], 1) << ','
            << digest(&gpu_states[3], 1) << ',' << digest(&gpu_states[4], 1) << ','
            << digest(&gpu_states[5], 1) << ',' << digest(&gpu_states[6], 1) << ','
            << digest(&gpu_states[7], 1) << ']'
            << ",\"first_difference\":" << first_difference
            << ",\"difference_count\":" << difference_count
            << ",\"first_cpu_byte\":"
            << (equal ? 0U : static_cast<unsigned>(cpu_bytes[first_difference]))
            << ",\"first_gpu_byte\":"
            << (equal ? 0U : static_cast<unsigned>(gpu_bytes[first_difference]))
            << ",\"ordered_first_skill\":113"
            << ",\"effect_resume_hand_count\":"
            << gpu_states[1].players[0].hand.count
            << ",\"stale_source_skipped\":"
            << (gpu_states[2].turn_used_skills.count == 0 ? "true" : "false")
            << ",\"no_ability_skipped\":"
            << (gpu_states[3].turn_used_skills.count == 0 ? "true" : "false")
            << ",\"second_effect_skill\":"
            << gpu_states[4].turn_used_skills.values[0]
            << ",\"enemy_second_effect_hand_count\":"
            << gpu_states[5].players[0].hand.count
            << ",\"checkup_damage\":[" << gpu_states[6].cards[10].damage << ','
            << gpu_states[6].cards[20].damage << ']'
            << ",\"checkup_rng_draws\":" << gpu_states[6].rng.draw_count
            << ",\"paralyze_recovered\":"
            << (official_pod_bad_status(gpu_states[7].players[0])
                    == OfficialBadStatus::kNone ? "true" : "false")
            << ",\"errors\":["
            << gpu_states[0].error << ',' << gpu_states[1].error << ','
            << gpu_states[2].error << ',' << gpu_states[3].error << ','
            << gpu_states[4].error << ',' << gpu_states[5].error << ','
            << gpu_states[6].error << ',' << gpu_states[7].error << "]}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
