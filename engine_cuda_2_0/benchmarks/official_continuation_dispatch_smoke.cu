#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "ptcg_cuda/official_flow_dispatch_pod.cuh"
#include "ptcg_cuda/official_runtime.h"

namespace engine = ptcg::cuda_engine;

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open rule pack: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void add_card(
    engine::OfficialStatePod* state,
    std::uint16_t ref,
    std::int32_t card_id,
    std::int32_t player,
    engine::OfficialArea area) {
    engine::OfficialCardStatePod& card = state->cards[ref];
    card.card_id = card_id;
    card.player = static_cast<std::int8_t>(player);
    card.area = static_cast<std::uint8_t>(area);
    card.move_counter = state->move_counter++;
    if (!engine::official_pod_push_zone_card(
            state, player, area, engine::OfficialCardRefPod{ref})) {
        throw std::runtime_error("cannot add smoke card");
    }
}

engine::OfficialStatePod make_state() {
    engine::OfficialStatePod state{};
    engine::official_pod_reset(&state, 4701, 47);
    state.first_player = 0;
    state.turn = 3;
    state.phase = static_cast<std::uint8_t>(engine::OfficialGamePhase::kMain);
    add_card(&state, 10, 169, 0, engine::OfficialArea::kActive);
    add_card(&state, 11, 8, 0, engine::OfficialArea::kEnergy);
    state.cards[11].attach_move_counter = state.cards[10].move_counter;
    add_card(&state, 20, 24, 1, engine::OfficialArea::kActive);
    add_card(&state, 30, 1, 0, engine::OfficialArea::kPrize);
    add_card(&state, 31, 1, 1, engine::OfficialArea::kPrize);
    add_card(&state, 40, 1, 1, engine::OfficialArea::kDeck);
    if (!engine::official_push_continuation(
            &state, engine::OfficialContinuationId::kMainSelect)) {
        throw std::runtime_error("cannot push MainSelect");
    }
    return state;
}

engine::OfficialStatePod make_effect_retreat_overlap_state() {
    engine::OfficialStatePod state{};
    engine::official_pod_reset(&state, 4702, 48);
    state.first_player = 0;
    state.turn = 3;
    state.phase = static_cast<std::uint8_t>(engine::OfficialGamePhase::kMain);
    state.turn_state = 1U << 3U;
    state.select_type = static_cast<std::uint8_t>(
        engine::OfficialSelectTypeId::kCard);
    state.select_context = 4;
    state.select_player = 0;
    state.select_min = 1;
    state.select_max = 1;
    state.options.count = 1;
    state.options.values[0].type = static_cast<std::uint8_t>(
        engine::OfficialSelectOptionTypeId::kCard);
    state.flow_flags = engine::kOfficialPlayEffectReturnToMainFlag;
    state.effect_interpreter.active = 1;
    state.effect_interpreter.awaiting_selection = 1;
    state.effect_interpreter.resume_kind = static_cast<std::uint8_t>(
        engine::OfficialEffectResumeKind::kApplyPrimitive);
    state.effect_interpreter.effect_index = 1;
    state.effect_interpreter.effect_count = 2;
    return state;
}

__global__ void dispatch_kernel(
    engine::OfficialStatePod* state,
    const std::uint8_t* rule_bytes,
    engine::OfficialFlowStatus* status) {
    const engine::OfficialRulePackView rules =
        engine::make_official_rule_pack_view(rule_bytes);
    *status = engine::official_dispatch_no_action_continuation(state, rules);
}

__global__ void apply_kernel(
    engine::OfficialStatePod* state,
    const std::uint8_t* rule_bytes,
    const engine::OfficialActionPod* action,
    engine::OfficialFlowStatus* status) {
    const engine::OfficialRulePackView rules =
        engine::make_official_rule_pack_view(rule_bytes);
    *status = engine::official_apply_pending_action(
        state, rules, action->option_indices, action->count);
}

__global__ void yield_kernel(
    engine::OfficialStatePod* state,
    engine::OfficialFlowStatus* status) {
    *status = engine::official_yield_decision(state);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error(
                "usage: official_continuation_dispatch_smoke <rule-pack>");
        }
        const std::vector<std::uint8_t> rule_bytes = read_binary(argv[1]);
        const engine::OfficialRulePackView host_rules =
            engine::make_official_rule_pack_view(rule_bytes.data());
        if (host_rules.header->counts[10] != engine::kOfficialContinuationCount) {
            throw std::runtime_error("continuation count mismatch");
        }

        const engine::OfficialStatePod initial = make_state();
        engine::OfficialStatePod cpu = initial;
        const engine::OfficialFlowStatus cpu_status =
            engine::official_dispatch_no_action_continuation(&cpu, host_rules);

        engine::OfficialActionPod action{};
        for (std::uint16_t index = 0; index < cpu.options.count; ++index) {
            if (cpu.options.values[index].type == static_cast<std::uint8_t>(
                    engine::OfficialSelectOptionTypeId::kEnd)) {
                action.count = 1;
                action.option_indices[0] = index;
                break;
            }
        }
        if (action.count != 1) throw std::runtime_error("end option missing");
        engine::OfficialStatePod cpu_after = cpu;
        const engine::OfficialFlowStatus cpu_after_status =
            engine::official_apply_pending_action(
                &cpu_after,
                host_rules,
                action.option_indices,
                action.count);
        const engine::OfficialStatePod overlap_initial =
            make_effect_retreat_overlap_state();
        engine::OfficialStatePod overlap_cpu = overlap_initial;
        const engine::OfficialFlowStatus overlap_cpu_status =
            engine::official_yield_decision(&overlap_cpu);

        engine::OfficialStatePod* device_state = nullptr;
        std::uint8_t* device_rules = nullptr;
        engine::OfficialFlowStatus* device_status = nullptr;
        engine::OfficialActionPod* device_action = nullptr;
        check_cuda(cudaMalloc(&device_state, sizeof(cpu)), "allocate state");
        check_cuda(cudaMalloc(&device_rules, rule_bytes.size()), "allocate rules");
        check_cuda(cudaMalloc(&device_status, sizeof(cpu_status)), "allocate status");
        check_cuda(cudaMalloc(&device_action, sizeof(action)), "allocate action");
        check_cuda(
            cudaMemcpy(
                device_state, &initial, sizeof(initial), cudaMemcpyHostToDevice),
            "copy state");
        check_cuda(
            cudaMemcpy(
                device_rules, rule_bytes.data(), rule_bytes.size(),
                cudaMemcpyHostToDevice),
            "copy rules");
        check_cuda(
            cudaMemcpy(device_action, &action, sizeof(action), cudaMemcpyHostToDevice),
            "copy action");
        dispatch_kernel<<<1, 1>>>(device_state, device_rules, device_status);
        check_cuda(cudaGetLastError(), "launch dispatcher");
        engine::OfficialStatePod gpu_after_dispatch{};
        engine::OfficialFlowStatus gpu_status_after_dispatch{};
        check_cuda(
            cudaMemcpy(
                &gpu_after_dispatch,
                device_state,
                sizeof(gpu_after_dispatch),
                cudaMemcpyDeviceToHost),
            "read dispatched state");
        check_cuda(
            cudaMemcpy(
                &gpu_status_after_dispatch,
                device_status,
                sizeof(gpu_status_after_dispatch),
                cudaMemcpyDeviceToHost),
            "read dispatched status");
        apply_kernel<<<1, 1>>>(device_state, device_rules, device_action, device_status);
        check_cuda(cudaGetLastError(), "launch action dispatcher");
        engine::OfficialStatePod gpu_after{};
        engine::OfficialFlowStatus gpu_after_status{};
        check_cuda(
            cudaMemcpy(&gpu_after, device_state, sizeof(gpu_after), cudaMemcpyDeviceToHost),
            "read action state");
        check_cuda(
            cudaMemcpy(
                &gpu_after_status,
                device_status,
                sizeof(gpu_after_status),
                cudaMemcpyDeviceToHost),
            "read action status");
        check_cuda(
            cudaMemcpy(
                device_state,
                &overlap_initial,
                sizeof(overlap_initial),
                cudaMemcpyHostToDevice),
            "copy overlap state");
        yield_kernel<<<1, 1>>>(device_state, device_status);
        check_cuda(cudaGetLastError(), "launch overlap yield");
        engine::OfficialStatePod overlap_gpu{};
        engine::OfficialFlowStatus overlap_gpu_status{};
        check_cuda(
            cudaMemcpy(
                &overlap_gpu,
                device_state,
                sizeof(overlap_gpu),
                cudaMemcpyDeviceToHost),
            "read overlap state");
        check_cuda(
            cudaMemcpy(
                &overlap_gpu_status,
                device_status,
                sizeof(overlap_gpu_status),
                cudaMemcpyDeviceToHost),
            "read overlap status");
        check_cuda(cudaFree(device_status), "free status");
        check_cuda(cudaFree(device_action), "free action");
        check_cuda(cudaFree(device_rules), "free rules");
        check_cuda(cudaFree(device_state), "free state");

        const bool dispatch_stack_ok =
            cpu.continuations.count == 1
            && cpu.continuations.values[0].opcode
                == static_cast<std::uint16_t>(
                    engine::OfficialContinuationId::kSelectedMain);
        const bool action_stack_ok =
            cpu_after.continuations.count == 1
            && cpu_after.continuations.values[0].opcode
                == static_cast<std::uint16_t>(
                    engine::OfficialContinuationId::kSelectedMain);
        const bool overlap_stack_ok =
            overlap_cpu.continuations.count == 3
            && overlap_cpu.continuations.values[0].opcode
                == static_cast<std::uint16_t>(
                    engine::OfficialContinuationId::kToMain)
            && overlap_cpu.continuations.values[1].opcode
                == static_cast<std::uint16_t>(
                    engine::OfficialContinuationId::kAfterPlay)
            && overlap_cpu.continuations.values[2].opcode
                == static_cast<std::uint16_t>(
                    engine::OfficialContinuationId::kSelectedEffectTarget);
        const bool passed =
            cpu_status == engine::OfficialFlowStatus::kNeedsAction
            && gpu_status_after_dispatch == cpu_status
            && cpu_after_status == engine::OfficialFlowStatus::kNeedsAction
            && cpu_after_status == gpu_after_status
            && dispatch_stack_ok
            && action_stack_ok
            && overlap_stack_ok
            && overlap_cpu_status == engine::OfficialFlowStatus::kNeedsAction
            && overlap_gpu_status == overlap_cpu_status
            && overlap_cpu.turn_action_count == 1
            && cpu.select_type
                == static_cast<std::uint8_t>(engine::OfficialSelectTypeId::kMain)
            && cpu.options.count > 0
            && cpu.turn_action_count == 1
            && cpu.error == 0
            && cpu_after.error == 0
            && cpu_after.turn == 4
            && cpu_after.continuations.count == 1
            && cpu_after.select_type
                == static_cast<std::uint8_t>(engine::OfficialSelectTypeId::kMain)
            && std::memcmp(&cpu, &gpu_after_dispatch, sizeof(cpu)) == 0
            && std::memcmp(&cpu_after, &gpu_after, sizeof(cpu_after)) == 0
            && std::memcmp(
                &overlap_cpu, &overlap_gpu, sizeof(overlap_cpu)) == 0;
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"state_abi\":" << engine::kOfficialStateAbiVersion
            << ",\"state_bytes\":" << sizeof(engine::OfficialStatePod)
            << ",\"continuation_count\":" << engine::kOfficialContinuationCount
            << ",\"input_opcode\":47"
            << ",\"output_opcode\":"
            << cpu.continuations.values[0].opcode
            << ",\"option_count\":" << cpu.options.count
            << ",\"cpu_status\":" << static_cast<int>(cpu_status)
            << ",\"gpu_status\":" << static_cast<int>(gpu_status_after_dispatch)
            << ",\"cpu_after_status\":" << static_cast<int>(cpu_after_status)
            << ",\"gpu_after_status\":" << static_cast<int>(gpu_after_status)
            << ",\"action_stack_count\":" << cpu_after.continuations.count
            << ",\"action_output_opcode\":"
            << cpu_after.continuations.values[0].opcode
            << ",\"overlap_cpu_status\":"
            << static_cast<int>(overlap_cpu_status)
            << ",\"overlap_gpu_status\":"
            << static_cast<int>(overlap_gpu_status)
            << ",\"overlap_stack\":["
            << overlap_cpu.continuations.values[0].opcode << ','
            << overlap_cpu.continuations.values[1].opcode << ','
            << overlap_cpu.continuations.values[2].opcode << ']'
            << ",\"overlap_byte_mismatches\":"
            << (std::memcmp(
                    &overlap_cpu, &overlap_gpu, sizeof(overlap_cpu)) == 0
                ? 0 : 1)
            << ",\"byte_mismatches\":"
            << (std::memcmp(&cpu_after, &gpu_after, sizeof(cpu_after)) == 0 ? 0 : 1)
            << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
