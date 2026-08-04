#include "ptcg_cuda/continuation_vm.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace engine = ptcg::cuda_engine;

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__host__ __device__ void exercise(engine::BattleState* state) {
    *state = {};
    state->status = static_cast<std::int32_t>(engine::EngineStatus::kAdvancing);
    engine::push_continuation(state, 10, 1, 7, 0, 0, 2);
    const engine::ContinuationCall first = engine::begin_continuation_call(state);
    engine::push_continuation(state, 20, 2, 11, 13);
    engine::push_continuation(state, 30);
    engine::finish_continuation_call(state, first);

    const engine::ContinuationCall third = engine::begin_continuation_call(state);
    engine::finish_continuation_call(state, third);
    const engine::ContinuationCall second = engine::begin_continuation_call(state);
    engine::finish_continuation_call(state, second, true);
    const engine::ContinuationCall first_again = engine::begin_continuation_call(state);
    engine::finish_continuation_call(state, first_again);
}

__global__ void exercise_kernel(engine::BattleState* state) {
    exercise(state);
}

}  // namespace

int main() {
    try {
        engine::BattleState expected{};
        exercise(&expected);
        engine::BattleState* device = nullptr;
        check_cuda(cudaMalloc(&device, sizeof(engine::BattleState)), "allocate state");
        exercise_kernel<<<1, 1>>>(device);
        check_cuda(cudaGetLastError(), "launch continuation VM");
        engine::BattleState actual{};
        check_cuda(
            cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost),
            "copy state");
        check_cuda(cudaFree(device), "free state");
        const bool passed =
            expected.error == actual.error
            && expected.status == actual.status
            && expected.continuation_stack_size == 0
            && actual.continuation_stack_size == 0;
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"host_error\":" << expected.error
            << ",\"device_error\":" << actual.error
            << ",\"host_stack_size\":" << expected.continuation_stack_size
            << ",\"device_stack_size\":" << actual.continuation_stack_size
            << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
