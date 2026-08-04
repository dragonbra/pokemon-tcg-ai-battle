#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptcg_cuda/official_flow_dispatch_pod.cuh"
#include "ptcg_cuda/official_runtime.h"
#include "ptcg_cuda/official_turn_flow_pod.cuh"

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

__host__ __device__ void add_player_cards(OfficialStatePod* state) {
    for (std::uint16_t player = 0; player < 2; ++player) {
        OfficialCardStatePod* card = &state->cards[1 + player];
        card->player = static_cast<std::int8_t>(player);
        card->area = static_cast<std::uint8_t>(OfficialArea::kPlayer);
        card->move_counter = ++state->move_counter;
    }
}

__host__ __device__ void run_turn_transition_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 601, seed);
    state->first_player = 0;
    state->turn = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_player_cards(state);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 8, 1, OfficialArea::kActive);
    add_card(state, 30, 1242, 0, OfficialArea::kPrize);
    add_card(state, 40, 1242, 1, OfficialArea::kPrize);
    add_card(state, 41, 1, 1, OfficialArea::kDeck);
    add_card(state, 50, 17, 0, OfficialArea::kEnergy);
    state->cards[50].attach_move_counter = state->cards[10].move_counter;

    state->cards[10].this_turn[0] = 11;
    state->cards[20].next_turn[0] = 22;
    state->cards[10].next_turn_enemy = 33;
    state->players[1].next_turn = 44;
    state->turn_histories[0].attack_id = 101;
    state->turn_histories[1].attack_id = 102;
    state->turn_histories[2].attack_id = 103;

    const OfficialTurnFlowResult result = official_begin_turn_end(state, rules);
    if (result != OfficialTurnFlowResult::kComplete
        || state->turn_flow_stage != 0
        || state->refresh_flow_stage != 0
        || state->turn != 2
        || official_active_player(*state) != 1
        || state->phase != static_cast<std::uint8_t>(OfficialGamePhase::kMain)
        || state->players[1].deck.count != 0
        || state->players[1].hand.count != 1
        || state->players[1].hand.values[0].index != 41
        || state->players[0].energy.count != 0
        || state->players[0].trash.count != 1
        || state->players[0].trash.values[0].index != 50
        || state->cards[10].this_turn[0] != 0
        || state->cards[10].this_turn_enemy != 33
        || state->cards[10].next_turn_enemy != 0
        || state->cards[20].this_turn[0] != 22
        || state->cards[20].next_turn[0] != 0
        || state->players[1].this_turn != 44
        || state->players[1].next_turn != 0
        || state->turn_histories[0].attack_id != 0
        || state->turn_histories[1].attack_id != 101
        || state->turn_histories[2].attack_id != 102) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 6101);
    }
}

__host__ __device__ void run_maintenance_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 602, seed);
    state->first_player = 0;
    state->turn = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_player_cards(state);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 8, 1, OfficialArea::kActive);
    add_card(state, 30, 1242, 0, OfficialArea::kPrize);
    add_card(state, 40, 1242, 1, OfficialArea::kPrize);
    for (std::uint16_t instance = 50; instance < 56; ++instance) {
        add_card(state, instance, 7, 0, OfficialArea::kBench);
    }
    add_card(state, 60, 1, 0, OfficialArea::kTool);
    add_card(state, 61, 1, 0, OfficialArea::kTool);
    state->cards[60].attach_move_counter = state->cards[10].move_counter;
    state->cards[61].attach_move_counter = state->cards[10].move_counter;
    add_card(state, 62, 15, 0, OfficialArea::kEnergy);
    state->cards[62].attach_move_counter = state->cards[10].move_counter;

    OfficialTurnFlowResult result = official_begin_refresh(state, rules);
    if (result != OfficialTurnFlowResult::kNeedsAction
        || state->refresh_flow_stage != static_cast<std::uint8_t>(
            OfficialRefreshFlowStage::kBenchOverflow)
        || state->select_player != 0 || state->select_min != 1
        || state->options.count != 6) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 6201);
        return;
    }
    state->current_card_effect_index = 4;
    const std::uint16_t first = 0;
    result = official_resume_refresh(state, rules, &first, 1);
    if (result != OfficialTurnFlowResult::kNeedsAction
        || state->refresh_flow_stage != static_cast<std::uint8_t>(
            OfficialRefreshFlowStage::kToolOverflow)
        || state->select_player != 0 || state->select_min != 1
        || state->options.count != 2
        || state->current_card_effect_index != 4) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 6202);
        return;
    }
    result = official_resume_refresh(state, rules, &first, 1);
    if (result != OfficialTurnFlowResult::kComplete
        || state->refresh_flow_stage != 0
        || state->players[0].bench.count != 5
        || state->players[0].tool.count != 1
        || state->players[0].energy.count != 0
        || state->players[0].trash.count != 3
        || state->game_result != static_cast<std::uint8_t>(OfficialGameResult::kNone)) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 6203);
    }
}

__host__ __device__ void run_deck_out_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 603, seed);
    state->first_player = 0;
    state->turn = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_player_cards(state);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 8, 1, OfficialArea::kActive);
    add_card(state, 30, 1242, 0, OfficialArea::kPrize);
    add_card(state, 40, 1242, 1, OfficialArea::kPrize);
    const OfficialTurnFlowResult result = official_begin_turn_end(state, rules);
    if (result != OfficialTurnFlowResult::kComplete
        || state->game_result != static_cast<std::uint8_t>(
            OfficialGameResult::kPlayer0Win)
        || state->finish_reason != static_cast<std::uint8_t>(
            OfficialFinishReason::kDeck)) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 6301);
    }
}

void prepare_turn_transition_initial(OfficialStatePod* state, std::uint64_t seed) {
    official_pod_reset(state, 601, seed);
    state->first_player = 0;
    state->turn = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_player_cards(state);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 8, 1, OfficialArea::kActive);
    add_card(state, 30, 1242, 0, OfficialArea::kPrize);
    add_card(state, 40, 1242, 1, OfficialArea::kPrize);
    add_card(state, 41, 1, 1, OfficialArea::kDeck);
    add_card(state, 50, 17, 0, OfficialArea::kEnergy);
    state->cards[50].attach_move_counter = state->cards[10].move_counter;

    state->cards[10].this_turn[0] = 11;
    state->cards[20].next_turn[0] = 22;
    state->cards[10].next_turn_enemy = 33;
    state->players[1].next_turn = 44;
    state->turn_histories[0].attack_id = 101;
    state->turn_histories[1].attack_id = 102;
    state->turn_histories[2].attack_id = 103;
}

void prepare_maintenance_pending(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 602, seed);
    state->first_player = 0;
    state->turn = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_player_cards(state);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 8, 1, OfficialArea::kActive);
    add_card(state, 30, 1242, 0, OfficialArea::kPrize);
    add_card(state, 40, 1242, 1, OfficialArea::kPrize);
    for (std::uint16_t instance = 50; instance < 56; ++instance) {
        add_card(state, instance, 7, 0, OfficialArea::kBench);
    }
    add_card(state, 60, 1, 0, OfficialArea::kTool);
    add_card(state, 61, 1, 0, OfficialArea::kTool);
    state->cards[60].attach_move_counter = state->cards[10].move_counter;
    state->cards[61].attach_move_counter = state->cards[10].move_counter;
    add_card(state, 62, 15, 0, OfficialArea::kEnergy);
    state->cards[62].attach_move_counter = state->cards[10].move_counter;

    const OfficialMainResult result = official_main_begin_refresh(state, rules);
    if (result != OfficialMainResult::kNeedsAction
        || state->refresh_flow_stage != static_cast<std::uint8_t>(
            OfficialRefreshFlowStage::kBenchOverflow)
        || state->select_player != 0 || state->select_min != 1
        || state->options.count != 6) {
        throw std::runtime_error("maintenance fixture did not reach bench overflow");
    }
    state->current_card_effect_index = 4;
    if (official_yield_decision(state) != OfficialFlowStatus::kNeedsAction) {
        throw std::runtime_error("maintenance fixture did not yield decision");
    }
}

void prepare_deck_out_initial(OfficialStatePod* state, std::uint64_t seed) {
    official_pod_reset(state, 603, seed);
    state->first_player = 0;
    state->turn = 1;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);
    add_player_cards(state);
    add_card(state, 10, 7, 0, OfficialArea::kActive);
    add_card(state, 20, 8, 1, OfficialArea::kActive);
    add_card(state, 30, 1242, 0, OfficialArea::kPrize);
    add_card(state, 40, 1242, 1, OfficialArea::kPrize);
}

std::uint16_t find_end_option(const OfficialStatePod& state) {
    for (std::uint16_t index = 0; index < state.options.count; ++index) {
        if (state.options.values[index].type
            == static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kEnd)) {
            return index;
        }
    }
    throw std::runtime_error("End Turn option not found");
}

OfficialActionPod one_option_action(std::uint16_t option_index) {
    OfficialActionPod action{};
    action.count = 1;
    action.option_indices[0] = option_index;
    return action;
}

void apply_cpu_action(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialActionPod& action,
    OfficialFlowStatus expected_status,
    const char* operation) {
    const OfficialFlowStatus status = official_apply_pending_action(
        state, rules, action.option_indices, action.count);
    if (status != expected_status) {
        throw std::runtime_error(operation);
    }
}

void build_cpu_expected_states(
    OfficialStatePod* initial_states,
    OfficialStatePod* expected_states,
    OfficialFlowStatus* expected_statuses,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    prepare_turn_transition_initial(&initial_states[0], seed);
    prepare_maintenance_pending(&initial_states[1], rules, seed + 1);
    prepare_deck_out_initial(&initial_states[2], seed + 2);
    for (std::size_t index = 0; index < 3; ++index) {
        expected_states[index] = initial_states[index];
        const OfficialFlowStatus status = official_advance_idle_state_to_decision(
            &expected_states[index], rules);
        if (status != OfficialFlowStatus::kNeedsAction) {
            throw std::runtime_error("CPU fixture did not advance to decision");
        }
    }

    const OfficialActionPod end_turn = one_option_action(
        find_end_option(expected_states[0]));
    const OfficialActionPod bench_overflow = one_option_action(0);
    const OfficialActionPod deck_out = one_option_action(
        find_end_option(expected_states[2]));
    apply_cpu_action(
        &expected_states[0],
        rules,
        end_turn,
        OfficialFlowStatus::kNeedsAction,
        "CPU turn transition action did not complete");
    apply_cpu_action(
        &expected_states[1],
        rules,
        bench_overflow,
        OfficialFlowStatus::kNeedsAction,
        "CPU bench overflow action did not request tool overflow");
    apply_cpu_action(
        &expected_states[2],
        rules,
        deck_out,
        OfficialFlowStatus::kTerminal,
        "CPU deck-out action did not terminate");
    apply_cpu_action(
        &expected_states[1],
        rules,
        bench_overflow,
        OfficialFlowStatus::kNeedsAction,
        "CPU tool overflow action did not complete");

    for (std::size_t index = 0; index < 3; ++index) {
        expected_statuses[index] = official_flow_status(expected_states[index]);
    }
}

std::size_t first_mismatch_offset(
    const OfficialStatePod& expected,
    const OfficialStatePod& actual) {
    const auto* left = reinterpret_cast<const std::uint8_t*>(&expected);
    const auto* right = reinterpret_cast<const std::uint8_t*>(&actual);
    for (std::size_t index = 0; index < sizeof(OfficialStatePod); ++index) {
        if (left[index] != right[index]) return index;
    }
    return sizeof(OfficialStatePod);
}

bool scenario_enabled(int requested, int lane) {
    return requested < 0 || requested == lane;
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
        if (argc != 2 && argc != 3) {
            throw std::runtime_error(
                "usage: official_turn_flow_smoke <official_rules.bin> [scenario]");
        }
        const int scenario = argc == 3 ? std::stoi(argv[2]) : -1;
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        const OfficialRulePackView host_rules = make_official_rule_pack_view(
            rule_pack.data());
        if (rule_pack.size() < sizeof(OfficialRulePackHeader)
            || std::memcmp(host_rules.header->magic, "PTCGRUL1", 8) != 0) {
            throw std::runtime_error("invalid rule pack");
        }

        if (scenario < -1 || scenario > 2) {
            throw std::runtime_error("scenario must be 0, 1, or 2");
        }

        constexpr std::uint64_t seed = 20260730ULL;
        OfficialStatePod initial_states[3]{};
        OfficialStatePod cpu_states[3]{};
        OfficialFlowStatus cpu_statuses[3]{};
        build_cpu_expected_states(
            initial_states, cpu_states, cpu_statuses, host_rules, seed);

        OfficialRuntimeConfig config{};
        config.batch_size = 3;
        config.rule_pack_bytes = rule_pack.size();
        OfficialDeviceArena arena{};
        check_cuda(allocate_official_arena(&arena, config), "allocate arena");
        check_cuda(
            upload_official_rule_pack(&arena, rule_pack.data(), rule_pack.size()),
            "upload official rule pack");
        check_cuda(
            upload_official_states_async(
                &arena, initial_states, cudaMemcpyHostToDevice, nullptr),
            "upload initial states");
        check_cuda(
            advance_official_states_to_decision_async(&arena, nullptr),
            "advance to decision");
        check_cuda(cudaDeviceSynchronize(), "sync advance to decision");

        OfficialStatePod decision_states[3]{};
        check_cuda(cudaMemcpy(
            decision_states,
            arena.states,
            sizeof(decision_states),
            cudaMemcpyDeviceToHost),
            "copy decision states");

        OfficialActionPod actions[3]{};
        actions[0] = one_option_action(find_end_option(decision_states[0]));
        actions[1] = one_option_action(0);
        actions[2] = one_option_action(find_end_option(decision_states[2]));
        check_cuda(cudaMemcpy(
            arena.actions,
            actions,
            sizeof(actions),
            cudaMemcpyHostToDevice),
            "upload actions");
        check_cuda(
            apply_official_actions_and_advance_async(&arena, nullptr, nullptr),
            "apply first actions");
        check_cuda(cudaDeviceSynchronize(), "sync first actions");

        const std::uint8_t maintenance_only_statuses[3] = {
            static_cast<std::uint8_t>(OfficialFlowStatus::kIdle),
            static_cast<std::uint8_t>(OfficialFlowStatus::kNeedsAction),
            static_cast<std::uint8_t>(OfficialFlowStatus::kTerminal),
        };
        check_cuda(cudaMemcpy(
            arena.statuses,
            maintenance_only_statuses,
            sizeof(maintenance_only_statuses),
            cudaMemcpyHostToDevice),
            "mask maintenance ready lane");
        check_cuda(
            apply_official_packed_ready_actions_async(&arena, nullptr),
            "apply remaining ready action");
        check_cuda(cudaDeviceSynchronize(), "sync remaining ready action");
        check_cuda(
            classify_official_states_async(&arena, nullptr),
            "classify final states");
        check_cuda(cudaDeviceSynchronize(), "sync final classification");

        OfficialStatePod gpu_states[3]{};
        std::uint8_t gpu_statuses[3]{};
        check_cuda(cudaMemcpy(
            gpu_states, arena.states, sizeof(gpu_states), cudaMemcpyDeviceToHost),
            "copy states");
        check_cuda(cudaMemcpy(
            gpu_statuses,
            arena.statuses,
            sizeof(gpu_statuses),
            cudaMemcpyDeviceToHost),
            "copy statuses");
        const std::size_t device_stack_bytes = arena.device_stack_bytes;
        const std::size_t arena_allocated_bytes = arena.allocated_bytes;
        check_cuda(free_official_arena(&arena), "free arena");

        std::size_t state_mismatches = 0;
        std::size_t status_mismatches = 0;
        int first_mismatch_lane = -1;
        std::size_t first_mismatch_byte = sizeof(OfficialStatePod);
        for (int lane = 0; lane < 3; ++lane) {
            if (!scenario_enabled(scenario, lane)) continue;
            if (gpu_statuses[lane] != static_cast<std::uint8_t>(cpu_statuses[lane])) {
                ++status_mismatches;
            }
            if (std::memcmp(
                    &cpu_states[lane],
                    &gpu_states[lane],
                    sizeof(OfficialStatePod)) != 0) {
                ++state_mismatches;
                if (first_mismatch_lane < 0) {
                    first_mismatch_lane = lane;
                    first_mismatch_byte = first_mismatch_offset(
                        cpu_states[lane], gpu_states[lane]);
                }
            }
        }
        const bool passed = state_mismatches == 0
            && status_mismatches == 0
            && (scenario >= 0
                ? gpu_states[scenario].error == 0
                : (gpu_states[0].error == 0
                    && gpu_states[1].error == 0
                    && gpu_states[2].error == 0));
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"device\":\"" << properties.name << "\""
            << ",\"state_abi\":" << kOfficialStateAbiVersion
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"device_stack_bytes\":" << device_stack_bytes
            << ",\"arena_allocated_bytes\":" << arena_allocated_bytes
            << ",\"cpu_digest\":" << digest(cpu_states, 3)
            << ",\"gpu_digest\":" << digest(gpu_states, 3)
            << ",\"state_mismatches\":" << state_mismatches
            << ",\"status_mismatches\":" << status_mismatches
            << ",\"first_mismatch_lane\":" << first_mismatch_lane
            << ",\"first_mismatch_byte\":" << first_mismatch_byte
            << ",\"next_turn\":" << gpu_states[0].turn
            << ",\"drawn_card\":" << gpu_states[0].players[1].hand.values[0].index
            << ",\"expired_energy_trash\":" << gpu_states[0].players[0].trash.count
            << ",\"bench_after_overflow\":" << gpu_states[1].players[0].bench.count
            << ",\"tool_after_overflow\":" << gpu_states[1].players[0].tool.count
            << ",\"rocket_energy_after_cleanup\":"
            << gpu_states[1].players[0].energy.count
            << ",\"deck_out_result\":"
            << static_cast<unsigned>(gpu_states[2].game_result)
            << ",\"deck_out_reason\":"
            << static_cast<unsigned>(gpu_states[2].finish_reason)
            << ",\"statuses\":[" << static_cast<unsigned>(gpu_statuses[0])
            << ',' << static_cast<unsigned>(gpu_statuses[1])
            << ',' << static_cast<unsigned>(gpu_statuses[2]) << ']'
            << ",\"errors\":[" << gpu_states[0].error << ','
            << gpu_states[1].error << ',' << gpu_states[2].error << "]}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
