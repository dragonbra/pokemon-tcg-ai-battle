#pragma once

#include <cstdint>

#include "ptcg_cuda/state_layout.cuh"

#if defined(__CUDACC__)
#define PTCG_VM_HD __host__ __device__
#else
#define PTCG_VM_HD
#endif

namespace ptcg::cuda_engine {

struct ContinuationCall {
    ContinuationFrame frame{};
    std::uint16_t stack_index = 0;
    bool valid = false;
};

PTCG_VM_HD inline void continuation_fail(
    BattleState* state,
    EngineError error) {
    state->error = static_cast<std::int32_t>(error);
    state->status = static_cast<std::int32_t>(EngineStatus::kError);
}

PTCG_VM_HD inline bool push_continuation(
    BattleState* state,
    std::uint16_t opcode,
    std::uint8_t arg_type = 0,
    std::int32_t arg0 = 0,
    std::int32_t arg1 = 0,
    std::int32_t arg2 = 0,
    std::uint8_t call_count = 1) {
    if (state->continuation_stack_size >= kMaxContinuationFrames) {
        continuation_fail(state, EngineError::kStackOverflow);
        return false;
    }
    if (call_count == 0 || arg_type > 4) {
        continuation_fail(state, EngineError::kRulePackBounds);
        return false;
    }
    ContinuationFrame* frame =
        &state->continuations[state->continuation_stack_size++];
    *frame = {};
    frame->args[0] = arg0;
    frame->args[1] = arg1;
    frame->args[2] = arg2;
    frame->opcode = opcode;
    frame->arg_type = arg_type;
    frame->call_count = call_count;
    return true;
}

PTCG_VM_HD inline ContinuationCall begin_continuation_call(
    BattleState* state) {
    if (state->continuation_stack_size == 0) {
        continuation_fail(state, EngineError::kRulePackBounds);
        return {};
    }
    const std::uint16_t index = state->continuation_stack_size - 1;
    return {state->continuations[index], index, true};
}

PTCG_VM_HD inline bool finish_continuation_call(
    BattleState* state,
    const ContinuationCall& call,
    bool break_call = false) {
    if (!call.valid || call.stack_index >= state->continuation_stack_size) {
        continuation_fail(state, EngineError::kRulePackBounds);
        return false;
    }
    ContinuationFrame* current = &state->continuations[call.stack_index];
    if (current->opcode != call.frame.opcode
        || current->called_count != call.frame.called_count) {
        continuation_fail(state, EngineError::kRulePackBounds);
        return false;
    }
    ++current->called_count;
    if (!break_call && current->called_count < current->call_count) {
        return true;
    }
    for (std::uint16_t i = call.stack_index + 1;
         i < state->continuation_stack_size;
         ++i) {
        state->continuations[i - 1] = state->continuations[i];
    }
    --state->continuation_stack_size;
    return true;
}

}  // namespace ptcg::cuda_engine

#undef PTCG_VM_HD
