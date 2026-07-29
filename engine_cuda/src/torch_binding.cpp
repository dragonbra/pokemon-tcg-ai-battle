#include "ptcg_cuda/runtime.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine = ptcg::cuda_engine;

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require_bytes(
    const torch::Tensor& tensor,
    std::int64_t expected,
    const char* name,
    bool allow_cuda) {
    if (!tensor.is_contiguous() || tensor.scalar_type() != torch::kUInt8 ||
        tensor.numel() != expected || (!allow_cuda && tensor.is_cuda())) {
        throw std::invalid_argument(
            std::string(name) + " must be a contiguous uint8 tensor with " +
            std::to_string(expected) + " bytes");
    }
}

class CudaEngine {
public:
    CudaEngine(
        std::int64_t batch_size,
        std::int64_t policy_count,
        std::int64_t route_capacity,
        std::int64_t device_index)
        : device_index_(static_cast<int>(device_index)) {
        if (batch_size <= 0 || batch_size > 0xFFFFFFFFLL ||
            policy_count <= 0 || policy_count > engine::kMaxPolicies ||
            route_capacity <= 0 || route_capacity > 0xFFFFFFFFLL) {
            throw std::invalid_argument("invalid CUDA engine dimensions");
        }
        c10::cuda::CUDAGuard guard(device_index_);
        const engine::RuntimeConfig config{
            engine::kAbiVersion,
            static_cast<std::uint32_t>(batch_size),
            static_cast<std::uint32_t>(policy_count),
            static_cast<std::uint32_t>(route_capacity),
        };
        check_cuda(engine::allocate_arena(&arena_, config), "allocate_arena");
    }

    ~CudaEngine() {
        c10::cuda::CUDAGuard guard(device_index_);
        engine::free_arena(&arena_);
    }

    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    void upload_rule_pack(torch::Tensor instruction_bytes, torch::Tensor action_bytes) {
        if (instruction_bytes.numel() % sizeof(engine::Instruction) != 0 ||
            action_bytes.numel() % sizeof(engine::ActionDescriptor) != 0) {
            throw std::invalid_argument("packed rule tensors do not match the engine ABI");
        }
        require_bytes(instruction_bytes, instruction_bytes.numel(), "instructions", false);
        require_bytes(action_bytes, action_bytes.numel(), "actions", false);
        c10::cuda::CUDAGuard guard(device_index_);
        check_cuda(
            engine::upload_rule_pack(
                &arena_,
                reinterpret_cast<const engine::Instruction*>(instruction_bytes.data_ptr<std::uint8_t>()),
                static_cast<std::uint32_t>(instruction_bytes.numel() / sizeof(engine::Instruction)),
                reinterpret_cast<const engine::ActionDescriptor*>(action_bytes.data_ptr<std::uint8_t>()),
                static_cast<std::uint32_t>(action_bytes.numel() / sizeof(engine::ActionDescriptor))),
            "upload_rule_pack");
    }

    void reset(torch::Tensor reset_bytes) {
        const std::int64_t expected =
            static_cast<std::int64_t>(arena_.config.batch_size) * sizeof(engine::ResetSpec);
        require_bytes(reset_bytes, expected, "reset_specs", true);
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index_).stream();
        if (reset_bytes.is_cuda()) {
            if (reset_bytes.get_device() != device_index_) {
                throw std::invalid_argument("reset_specs is on the wrong CUDA device");
            }
            check_cuda(
                engine::reset_from_device_async(
                    &arena_,
                    reinterpret_cast<const engine::ResetSpec*>(reset_bytes.data_ptr<std::uint8_t>()),
                    stream),
                "reset_from_device_async");
        } else {
            check_cuda(
                engine::reset_from_host_async(
                    &arena_,
                    reinterpret_cast<const engine::ResetSpec*>(reset_bytes.data_ptr<std::uint8_t>()),
                    stream),
                "reset_from_host_async");
            // Reset is outside the decision hot loop. Synchronize so the CPU
            // tensor can be safely released after this call.
            check_cuda(cudaStreamSynchronize(stream), "reset host synchronize");
        }
    }

    void step(torch::Tensor actions) {
        if (!actions.is_cuda() || actions.get_device() != device_index_ ||
            !actions.is_contiguous() || actions.scalar_type() != torch::kInt32 ||
            actions.dim() != 1 || actions.size(0) != arena_.config.batch_size) {
            throw std::invalid_argument(
                "actions must be a contiguous CUDA int32 tensor of shape [batch]");
        }
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index_).stream();
        check_cuda(
            engine::apply_actions_device_async(&arena_, actions.data_ptr<std::int32_t>(), stream),
            "apply_actions_device_async");
    }

    std::unordered_map<std::string, torch::Tensor> encode_policy_v1() {
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index_).stream();
        check_cuda(engine::encode_policy_v1_async(&arena_, stream), "encode_policy_v1_async");
        const auto batch = static_cast<std::int64_t>(arena_.config.batch_size);
        return {
            {"global_cat", view(arena_.codec.global_cat, {batch, 8}, torch::kInt64)},
            {"global_num", view(arena_.codec.global_num, {batch, 16}, torch::kFloat32)},
            {"entity_cat", view(arena_.codec.entity_cat, {batch, 128, 6}, torch::kInt64)},
            {"entity_num", view(arena_.codec.entity_num, {batch, 128, 10}, torch::kFloat32)},
            {"entity_parent", view(arena_.codec.entity_parent, {batch, 128}, torch::kInt64)},
            {"entity_mask", view(arena_.codec.entity_mask, {batch, 128}, torch::kUInt8)},
            {"option_cat", view(arena_.codec.option_cat, {batch, 80, 12}, torch::kInt64)},
            {"option_num", view(arena_.codec.option_num, {batch, 80, 4}, torch::kFloat32)},
            {"option_equiv", view(arena_.codec.option_equiv, {batch, 80}, torch::kInt64)},
            {"option_mask", view(arena_.codec.option_mask, {batch, 80}, torch::kUInt8)},
            {"min_count", view(arena_.codec.min_count, {batch}, torch::kInt64)},
            {"max_count", view(arena_.codec.max_count, {batch}, torch::kInt64)},
        };
    }

    std::vector<torch::Tensor> route_ready() {
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index_).stream();
        check_cuda(engine::route_ready_async(&arena_, stream), "route_ready_async");
        return {
            view(
                arena_.route_indices,
                {
                    static_cast<std::int64_t>(arena_.config.policy_count),
                    static_cast<std::int64_t>(arena_.config.route_capacity),
                },
                torch::kInt32),
            view(
                arena_.route_counts,
                {static_cast<std::int64_t>(arena_.config.policy_count)},
                torch::kInt32),
        };
    }

    torch::Tensor digest() {
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index_).stream();
        check_cuda(engine::digest_async(&arena_, stream), "digest_async");
        return view(
            arena_.digests,
            {static_cast<std::int64_t>(arena_.config.batch_size)},
            torch::kInt64);
    }

    std::int64_t allocated_bytes() const {
        return static_cast<std::int64_t>(arena_.allocated_bytes);
    }

    std::int64_t batch_size() const {
        return static_cast<std::int64_t>(arena_.config.batch_size);
    }

private:
    torch::Tensor view(void* pointer, std::vector<std::int64_t> sizes, torch::ScalarType dtype) const {
        const auto options = torch::TensorOptions()
                                 .dtype(dtype)
                                 .device(torch::Device(torch::kCUDA, device_index_));
        return torch::from_blob(pointer, std::move(sizes), [](void*) {}, options);
    }

    engine::DeviceArena arena_{};
    int device_index_ = 0;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("ABI_VERSION") = engine::kAbiVersion;
    module.attr("STATE_BYTES") = sizeof(engine::BattleState);
    module.attr("KNOWN_DIVERGENCE_660_1207") =
        static_cast<std::int32_t>(engine::EngineError::kKnownDivergence6601207);
    pybind11::class_<CudaEngine>(module, "CudaEngine")
        .def(
            pybind11::init<std::int64_t, std::int64_t, std::int64_t, std::int64_t>(),
            pybind11::arg("batch_size"),
            pybind11::arg("policy_count"),
            pybind11::arg("route_capacity"),
            pybind11::arg("device_index") = 0)
        .def("upload_rule_pack", &CudaEngine::upload_rule_pack)
        .def("reset", &CudaEngine::reset)
        .def("step", &CudaEngine::step)
        .def("encode_policy_v1", &CudaEngine::encode_policy_v1)
        .def("route_ready", &CudaEngine::route_ready)
        .def("digest", &CudaEngine::digest)
        .def_property_readonly("allocated_bytes", &CudaEngine::allocated_bytes)
        .def_property_readonly("batch_size", &CudaEngine::batch_size);
}
