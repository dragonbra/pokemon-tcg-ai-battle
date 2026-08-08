#include "ptcg_cuda/runtime.h"
#include "ptcg_cuda/official_runtime.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <limits>
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

class OfficialCudaEngine {
public:
    OfficialCudaEngine(
        std::int64_t batch_size,
        std::int64_t rule_pack_bytes,
        std::int64_t device_index,
        std::int64_t device_stack_bytes)
        : device_index_(static_cast<int>(device_index)) {
        if (batch_size <= 0 || batch_size > 0xFFFFFFFFLL
            || rule_pack_bytes < static_cast<std::int64_t>(
                sizeof(engine::OfficialRulePackHeader))
            || device_stack_bytes < static_cast<std::int64_t>(
                engine::kOfficialMinimumDeviceStackBytes)) {
            throw std::invalid_argument("invalid official CUDA engine dimensions");
        }
        c10::cuda::CUDAGuard guard(device_index_);
        engine::OfficialRuntimeConfig config{};
        config.batch_size = static_cast<std::uint32_t>(batch_size);
        config.rule_pack_bytes = static_cast<std::size_t>(rule_pack_bytes);
        config.device_stack_bytes = static_cast<std::size_t>(device_stack_bytes);
        check_cuda(
            engine::allocate_official_arena(&arena_, config),
            "allocate_official_arena");
    }

    ~OfficialCudaEngine() {
        c10::cuda::CUDAGuard guard(device_index_);
        engine::free_official_arena(&arena_);
    }

    OfficialCudaEngine(const OfficialCudaEngine&) = delete;
    OfficialCudaEngine& operator=(const OfficialCudaEngine&) = delete;

    void upload_rule_pack(torch::Tensor rule_pack) {
        require_bytes(
            rule_pack,
            static_cast<std::int64_t>(arena_.config.rule_pack_bytes),
            "official_rule_pack",
            false);
        c10::cuda::CUDAGuard guard(device_index_);
        check_cuda(
            engine::upload_official_rule_pack(
                &arena_,
                rule_pack.data_ptr<std::uint8_t>(),
                arena_.config.rule_pack_bytes),
            "upload_official_rule_pack");
    }

    void reset_states(torch::Tensor states) {
        const std::size_t expected_size =
            static_cast<std::size_t>(arena_.config.batch_size)
            * sizeof(engine::OfficialStatePod);
        if (expected_size > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::invalid_argument("official state tensor is too large");
        }
        require_bytes(
            states,
            static_cast<std::int64_t>(expected_size),
            "official_states",
            true);
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();
        if (states.is_cuda()) {
            if (states.get_device() != device_index_) {
                throw std::invalid_argument(
                    "official_states is on the wrong CUDA device");
            }
            check_cuda(
                engine::upload_official_states_async(
                    &arena_,
                    reinterpret_cast<const engine::OfficialStatePod*>(
                        states.data_ptr<std::uint8_t>()),
                    cudaMemcpyDeviceToDevice,
                    stream),
                "upload_official_states_device");
        } else {
            check_cuda(
                engine::upload_official_states_async(
                    &arena_,
                    reinterpret_cast<const engine::OfficialStatePod*>(
                        states.data_ptr<std::uint8_t>()),
                    cudaMemcpyHostToDevice,
                    stream),
                "upload_official_states_host");
            // The caller may release the CPU tensor as soon as this method
            // returns; reset is outside the decision hot path.
            check_cuda(cudaStreamSynchronize(stream), "official reset synchronize");
        }
    }

    void reset_seeded_first_min(torch::Tensor decks, torch::Tensor seeds) {
        reset_seeded_impl(decks, seeds, nullptr, false);
    }

    void reset_seeded_first_min_semantic(torch::Tensor decks, torch::Tensor seeds) {
        reset_seeded_impl(decks, seeds, nullptr, false, true);
    }

    void reset_seeded_first_min_masked(
        torch::Tensor decks,
        torch::Tensor seeds,
        torch::Tensor lane_mask) {
        reset_seeded_impl(decks, seeds, &lane_mask, false);
    }

    void reset_seeded_first_min_semantic_masked(
        torch::Tensor decks,
        torch::Tensor seeds,
        torch::Tensor lane_mask) {
        reset_seeded_impl(decks, seeds, &lane_mask, false, true);
    }

    void reset_seeded_interactive(torch::Tensor decks, torch::Tensor seeds) {
        reset_seeded_impl(decks, seeds, nullptr, true);
    }

    void reset_seeded_interactive_semantic(torch::Tensor decks, torch::Tensor seeds) {
        reset_seeded_impl(decks, seeds, nullptr, true, true);
    }

    void reset_seeded_interactive_masked(
        torch::Tensor decks,
        torch::Tensor seeds,
        torch::Tensor lane_mask) {
        reset_seeded_impl(decks, seeds, &lane_mask, true);
    }

    void reset_seeded_interactive_semantic_masked(
        torch::Tensor decks,
        torch::Tensor seeds,
        torch::Tensor lane_mask) {
        reset_seeded_impl(decks, seeds, &lane_mask, true, true);
    }

    void classify() {
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();
        check_cuda(
            engine::classify_official_states_async(&arena_, stream),
            "classify_official_states_async");
    }

    void advance_to_decision() {
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();
        check_cuda(
            engine::advance_official_states_to_decision_async(&arena_, stream),
            "advance_official_states_to_decision_async");
    }

    void pack_actions(torch::Tensor option_indices, torch::Tensor counts) {
        if (!option_indices.is_cuda() || !counts.is_cuda()
            || option_indices.get_device() != device_index_
            || counts.get_device() != device_index_
            || !option_indices.is_contiguous() || !counts.is_contiguous()
            || option_indices.dim() != 2 || counts.dim() != 1
            || option_indices.size(0) != arena_.config.batch_size
            || counts.size(0) != arena_.config.batch_size
            || option_indices.size(1) <= 0
            || option_indices.size(1) > engine::kOfficialOptionCapacity) {
            throw std::invalid_argument(
                "official action indices/counts have invalid CUDA shapes");
        }
        const auto option_capacity = static_cast<std::uint32_t>(
            option_indices.size(1));
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();
        if (option_indices.scalar_type() == torch::kInt32
            && counts.scalar_type() == torch::kInt32) {
            check_cuda(
                engine::pack_official_actions_i32_async(
                    &arena_,
                    option_indices.data_ptr<std::int32_t>(),
                    counts.data_ptr<std::int32_t>(),
                    option_capacity,
                    stream),
                "pack_official_actions_i32_async");
            return;
        }
        if (option_indices.scalar_type() == torch::kInt64
            && counts.scalar_type() == torch::kInt64) {
            check_cuda(
                engine::pack_official_actions_i64_async(
                    &arena_,
                    option_indices.data_ptr<std::int64_t>(),
                    counts.data_ptr<std::int64_t>(),
                    option_capacity,
                    stream),
                "pack_official_actions_i64_async");
            return;
        }
        throw std::invalid_argument(
            "official action indices/counts must both be int32 or both int64");
    }

    void apply_actions(torch::Tensor action_pods) {
        if (!action_pods.is_cuda() || action_pods.get_device() != device_index_) {
            throw std::invalid_argument("official action PODs must be on the target CUDA device");
        }
        const std::size_t expected_size =
            static_cast<std::size_t>(arena_.config.batch_size)
            * sizeof(engine::OfficialActionPod);
        require_bytes(
            action_pods,
            static_cast<std::int64_t>(expected_size),
            "official_action_pods",
            true);
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();
        check_cuda(
            cudaMemcpyAsync(
                arena_.actions,
                action_pods.data_ptr<std::uint8_t>(),
                expected_size,
                cudaMemcpyDeviceToDevice,
                stream),
            "copy_official_action_pods");
        check_cuda(
            engine::apply_official_actions_and_advance_async(&arena_, nullptr, stream),
            "apply_official_actions_and_advance_async");
    }

    void apply_packed_actions() {
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();
        check_cuda(
            engine::apply_official_packed_ready_actions_async(&arena_, stream),
            "apply_official_packed_ready_actions_async");
    }

    void apply_packed_setup_actions() {
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();
        check_cuda(
            engine::apply_official_packed_setup_actions_async(&arena_, stream),
            "apply_official_packed_setup_actions_async");
    }

    std::unordered_map<std::string, torch::Tensor> encode_policy_v1() {
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();
        check_cuda(
            engine::encode_official_policy_codec_v1_async(&arena_, stream),
            "encode_official_policy_codec_v1_async");
        const auto batch = static_cast<std::int64_t>(arena_.config.batch_size);
        return {
            {"global_cat", view_typed(arena_.codec.global_cat, {batch, 8}, torch::kInt64)},
            {"global_num", view_typed(arena_.codec.global_num, {batch, 16}, torch::kFloat32)},
            {"entity_cat", view_typed(arena_.codec.entity_cat, {batch, 128, 6}, torch::kInt64)},
            {"entity_num", view_typed(arena_.codec.entity_num, {batch, 128, 10}, torch::kFloat32)},
            {"entity_parent", view_typed(arena_.codec.entity_parent, {batch, 128}, torch::kInt64)},
            {"entity_mask", view_typed(arena_.codec.entity_mask, {batch, 128}, torch::kUInt8)},
            {"option_cat", view_typed(arena_.codec.option_cat, {batch, 128, 12}, torch::kInt64)},
            {"option_num", view_typed(arena_.codec.option_num, {batch, 128, 4}, torch::kFloat32)},
            {"option_equiv", view_typed(arena_.codec.option_equiv, {batch, 128}, torch::kInt64)},
            {"option_mask", view_typed(arena_.codec.option_mask, {batch, 128}, torch::kUInt8)},
            {"min_count", view_typed(arena_.codec.min_count, {batch}, torch::kInt64)},
            {"max_count", view_typed(arena_.codec.max_count, {batch}, torch::kInt64)},
        };
    }

    std::unordered_map<std::string, torch::Tensor> encode_semantic0031_v2_lanes(
        torch::Tensor lane_indices) {
        if (!lane_indices.is_cuda() || lane_indices.get_device() != device_index_
            || !lane_indices.is_contiguous() || lane_indices.dim() != 1
            || (lane_indices.scalar_type() != torch::kInt32
                && lane_indices.scalar_type() != torch::kInt64)) {
            throw std::invalid_argument(
                "semantic0031 v2 lane_indices must be contiguous CUDA int32/int64[ready]");
        }
        const auto ready = lane_indices.size(0);
        if (ready < 0 || ready > 0xFFFFFFFFLL) {
            throw std::invalid_argument("semantic0031 v2 ready lane count is invalid");
        }
        c10::cuda::CUDAGuard guard(device_index_);
        const auto long_options = torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(torch::Device(torch::kCUDA, device_index_));
        const auto float_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::Device(torch::kCUDA, device_index_));
        const auto byte_options = torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(torch::Device(torch::kCUDA, device_index_));

        std::unordered_map<std::string, torch::Tensor> result{
            {"global_cat", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031GlobalCatWidth)}, long_options)},
            {"global_num", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031GlobalNumWidth)}, float_options)},
            {"global_state", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031GlobalNumWidth)}, long_options)},
            {"card_cat", torch::empty({ready, static_cast<std::int64_t>(engine::kMaxSemantic0031CardEntities), static_cast<std::int64_t>(engine::kSemantic0031CardCatWidth)}, long_options)},
            {"card_num", torch::empty({ready, static_cast<std::int64_t>(engine::kMaxSemantic0031CardEntities), static_cast<std::int64_t>(engine::kSemantic0031CardNumWidth)}, float_options)},
            {"card_state", torch::empty({ready, static_cast<std::int64_t>(engine::kMaxSemantic0031CardEntities), static_cast<std::int64_t>(engine::kSemantic0031CardNumWidth)}, long_options)},
            {"card_parent", torch::empty({ready, static_cast<std::int64_t>(engine::kMaxSemantic0031CardEntities)}, long_options)},
            {"card_mask", torch::empty({ready, static_cast<std::int64_t>(engine::kMaxSemantic0031CardEntities)}, byte_options)},
            {"resource_cat", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031ResourceCapacity), static_cast<std::int64_t>(engine::kSemantic0031ResourceCatWidth)}, long_options)},
            {"resource_num", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031ResourceCapacity), static_cast<std::int64_t>(engine::kSemantic0031ResourceNumWidth)}, float_options)},
            {"resource_state", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031ResourceCapacity), static_cast<std::int64_t>(engine::kSemantic0031ResourceNumWidth)}, long_options)},
            {"resource_mask", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031ResourceCapacity)}, byte_options)},
            {"event_cat", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031EventCapacity), static_cast<std::int64_t>(engine::kSemantic0031EventCatWidth)}, long_options)},
            {"event_num", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031EventCapacity), static_cast<std::int64_t>(engine::kSemantic0031EventNumWidth)}, float_options)},
            {"event_state", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031EventCapacity), static_cast<std::int64_t>(engine::kSemantic0031EventNumWidth)}, long_options)},
            {"event_mask", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031EventCapacity)}, byte_options)},
            {"event_source", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031EventCapacity)}, long_options)},
            {"event_target", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031EventCapacity)}, long_options)},
            {"event_before", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031EventCapacity)}, long_options)},
            {"event_after", torch::empty({ready, static_cast<std::int64_t>(engine::kSemantic0031EventCapacity)}, long_options)},
            {"option_cat", torch::empty({ready, 128, static_cast<std::int64_t>(engine::kSemantic0031OptionCatWidth)}, long_options)},
            {"option_num", torch::empty({ready, 128, static_cast<std::int64_t>(engine::kSemantic0031OptionNumWidth)}, float_options)},
            {"option_state", torch::empty({ready, 128, static_cast<std::int64_t>(engine::kSemantic0031OptionNumWidth)}, long_options)},
            {"option_mask", torch::empty({ready, 128}, byte_options)},
            {"option_source", torch::empty({ready, 128}, long_options)},
            {"option_target", torch::empty({ready, 128}, long_options)},
            {"option_context", torch::empty({ready, 128}, long_options)},
            {"option_effect_card", torch::empty({ready, 128}, long_options)},
            {"option_skill_id", torch::zeros({ready, 1}, long_options)},
            {"option_skill_role", torch::zeros({ready, 1}, long_options)},
            {"option_skill_parent", torch::zeros({ready, 1}, long_options)},
            {"option_skill_mask", torch::zeros({ready, 1}, byte_options)},
            {"option_effect_id", torch::zeros({ready, 1}, long_options)},
            {"option_effect_role", torch::zeros({ready, 1}, long_options)},
            {"option_effect_parent", torch::zeros({ready, 1}, long_options)},
            {"option_effect_mask", torch::zeros({ready, 1}, byte_options)},
            {"min_count", torch::empty({ready}, long_options)},
            {"max_count", torch::empty({ready}, long_options)},
            {"targets", torch::empty({ready, 1}, long_options)},
        };

        engine::OfficialSemantic0031CodecBuffers output{};
        output.global_cat = result["global_cat"].data_ptr<std::int64_t>();
        output.global_num = result["global_num"].data_ptr<float>();
        output.global_state = result["global_state"].data_ptr<std::int64_t>();
        output.card_cat = result["card_cat"].data_ptr<std::int64_t>();
        output.card_num = result["card_num"].data_ptr<float>();
        output.card_state = result["card_state"].data_ptr<std::int64_t>();
        output.card_parent = result["card_parent"].data_ptr<std::int64_t>();
        output.card_mask = result["card_mask"].data_ptr<std::uint8_t>();
        output.resource_cat = result["resource_cat"].data_ptr<std::int64_t>();
        output.resource_num = result["resource_num"].data_ptr<float>();
        output.resource_state = result["resource_state"].data_ptr<std::int64_t>();
        output.resource_mask = result["resource_mask"].data_ptr<std::uint8_t>();
        output.event_cat = result["event_cat"].data_ptr<std::int64_t>();
        output.event_num = result["event_num"].data_ptr<float>();
        output.event_state = result["event_state"].data_ptr<std::int64_t>();
        output.event_mask = result["event_mask"].data_ptr<std::uint8_t>();
        output.event_source = result["event_source"].data_ptr<std::int64_t>();
        output.event_target = result["event_target"].data_ptr<std::int64_t>();
        output.event_before = result["event_before"].data_ptr<std::int64_t>();
        output.event_after = result["event_after"].data_ptr<std::int64_t>();
        output.option_cat = result["option_cat"].data_ptr<std::int64_t>();
        output.option_num = result["option_num"].data_ptr<float>();
        output.option_state = result["option_state"].data_ptr<std::int64_t>();
        output.option_mask = result["option_mask"].data_ptr<std::uint8_t>();
        output.option_source = result["option_source"].data_ptr<std::int64_t>();
        output.option_target = result["option_target"].data_ptr<std::int64_t>();
        output.option_context = result["option_context"].data_ptr<std::int64_t>();
        output.option_effect_card = result["option_effect_card"].data_ptr<std::int64_t>();
        output.min_count = result["min_count"].data_ptr<std::int64_t>();
        output.max_count = result["max_count"].data_ptr<std::int64_t>();
        output.targets = result["targets"].data_ptr<std::int64_t>();

        const cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();
        if (lane_indices.scalar_type() == torch::kInt32) {
            check_cuda(
                engine::encode_official_semantic0031_codec_v2_i32_async(
                    &arena_,
                    lane_indices.data_ptr<std::int32_t>(),
                    static_cast<std::uint32_t>(ready),
                    output,
                    stream),
                "encode_official_semantic0031_codec_v2_i32_async");
        } else {
            check_cuda(
                engine::encode_official_semantic0031_codec_v2_i64_async(
                    &arena_,
                    lane_indices.data_ptr<std::int64_t>(),
                    static_cast<std::uint32_t>(ready),
                    output,
                    stream),
                "encode_official_semantic0031_codec_v2_i64_async");
        }
        return result;
    }

    torch::Tensor state_bytes() const {
        return view(
            arena_.states,
            {
                static_cast<std::int64_t>(arena_.config.batch_size),
                static_cast<std::int64_t>(sizeof(engine::OfficialStatePod)),
            });
    }

    torch::Tensor statuses() const {
        return view(
            arena_.statuses,
            {static_cast<std::int64_t>(arena_.config.batch_size)});
    }

    torch::Tensor game_results() const {
        auto* pointer = reinterpret_cast<std::uint8_t*>(arena_.states)
            + offsetof(engine::OfficialStatePod, game_result);
        const auto options = torch::TensorOptions()
            .device(torch::kCUDA, device_index_)
            .dtype(torch::kUInt8);
        return torch::from_blob(
            pointer,
            {static_cast<std::int64_t>(arena_.config.batch_size)},
            {static_cast<std::int64_t>(sizeof(engine::OfficialStatePod))},
            [](void*) {},
            options);
    }

    torch::Tensor decision_actors() const {
        auto* pointer = reinterpret_cast<std::uint8_t*>(arena_.states)
            + offsetof(engine::OfficialStatePod, select_player);
        const auto options = torch::TensorOptions()
            .device(torch::kCUDA, device_index_)
            .dtype(torch::kInt8);
        return torch::from_blob(
            pointer,
            {static_cast<std::int64_t>(arena_.config.batch_size)},
            {static_cast<std::int64_t>(sizeof(engine::OfficialStatePod))},
            [](void*) {},
            options);
    }

    torch::Tensor prize_counts() const {
        auto* pointer = reinterpret_cast<std::uint8_t*>(arena_.states)
            + offsetof(engine::OfficialStatePod, players)
            + offsetof(engine::OfficialPlayerStatePod, prize)
            + offsetof(decltype(engine::OfficialPlayerStatePod::prize), count);
        const auto options = torch::TensorOptions()
            .device(torch::kCUDA, device_index_)
            .dtype(torch::kInt16);
        return torch::from_blob(
            pointer,
            {static_cast<std::int64_t>(arena_.config.batch_size), 2},
            {
                static_cast<std::int64_t>(sizeof(engine::OfficialStatePod) / sizeof(std::int16_t)),
                static_cast<std::int64_t>(sizeof(engine::OfficialPlayerStatePod) / sizeof(std::int16_t)),
            },
            [](void*) {},
            options);
    }

    torch::Tensor action_bytes() const {
        return view(
            arena_.actions,
            {
                static_cast<std::int64_t>(arena_.config.batch_size),
                static_cast<std::int64_t>(sizeof(engine::OfficialActionPod)),
            });
    }

    std::unordered_map<std::string, torch::Tensor> semantic_history_raw() const {
        const auto batch = static_cast<std::int64_t>(arena_.config.batch_size);
        const auto capacity =
            static_cast<std::int64_t>(engine::kOfficialSemanticHistoryCapacity);
        const auto param_capacity = static_cast<std::int64_t>(
            engine::kOfficialSemanticHistoryParamCapacity);
        const auto serial_capacity = static_cast<std::int64_t>(
            engine::kOfficialSemanticSerialCapacity);
        return {
            {"total_count", view_typed(
                arena_.semantic_history_total_count,
                {batch},
                torch::kInt64)},
            {"write_index", view_typed(
                arena_.semantic_history_write_index,
                {batch},
                torch::kInt32)},
            {"log_type", view_typed(
                arena_.semantic_history_log_type,
                {batch, capacity},
                torch::kUInt8)},
            {"param_count", view_typed(
                arena_.semantic_history_param_count,
                {batch, capacity},
                torch::kUInt8)},
            {"params", view_typed(
                arena_.semantic_history_params,
                {batch, capacity, param_capacity},
                torch::kInt32)},
            {"known_opponent_hand", view_typed(
                arena_.semantic_known_opponent_hand,
                {batch, 2, serial_capacity},
                torch::kUInt8)},
            {"possible_opponent_hand", view_typed(
                arena_.semantic_possible_opponent_hand,
                {batch, 2, serial_capacity},
                torch::kUInt8)},
            {"remembered_opponent_cards", view_typed(
                arena_.semantic_remembered_opponent_cards,
                {batch, 2, serial_capacity},
                torch::kUInt8)},
            {"unknown_opponent_hand", view_typed(
                arena_.semantic_unknown_opponent_hand,
                {batch, 2},
                torch::kInt16)},
            {"possible_hand_lower", view_typed(
                arena_.semantic_possible_hand_lower,
                {batch, 2},
                torch::kInt16)},
            {"possible_hand_upper", view_typed(
                arena_.semantic_possible_hand_upper,
                {batch, 2},
                torch::kInt16)},
        };
    }

    std::int64_t allocated_bytes() const {
        return static_cast<std::int64_t>(arena_.allocated_bytes);
    }

    std::int64_t batch_size() const {
        return static_cast<std::int64_t>(arena_.config.batch_size);
    }

    std::int64_t rule_pack_bytes() const {
        return static_cast<std::int64_t>(arena_.config.rule_pack_bytes);
    }

private:
    void reset_seeded_impl(
        const torch::Tensor& decks,
        const torch::Tensor& seeds,
        const torch::Tensor* lane_mask,
        bool interactive,
        bool semantic = false) {
        const auto batch = static_cast<std::int64_t>(arena_.config.batch_size);
        if (!decks.is_cuda() || decks.get_device() != device_index_
            || !decks.is_contiguous()
            || decks.numel() != batch * 2 * 60
            || (decks.scalar_type() != torch::kInt32
                && decks.scalar_type() != torch::kInt64)) {
            throw std::invalid_argument(
                "official seeded decks must be contiguous CUDA int32/int64 "
                "with batch*2*60 elements");
        }
        if (!seeds.is_cuda() || seeds.get_device() != device_index_
            || !seeds.is_contiguous() || seeds.dim() != 1
            || seeds.size(0) != batch || seeds.scalar_type() != torch::kInt64) {
            throw std::invalid_argument(
                "official seeded seeds must be contiguous CUDA int64[batch]");
        }
        const std::uint8_t* mask_pointer = nullptr;
        if (lane_mask != nullptr) {
            if (!lane_mask->is_cuda() || lane_mask->get_device() != device_index_
                || !lane_mask->is_contiguous() || lane_mask->dim() != 1
                || lane_mask->size(0) != batch
                || (lane_mask->scalar_type() != torch::kUInt8
                    && lane_mask->scalar_type() != torch::kBool)) {
                throw std::invalid_argument(
                    "official seeded lane mask must be contiguous CUDA bool/uint8[batch]");
            }
            mask_pointer = reinterpret_cast<const std::uint8_t*>(
                lane_mask->data_ptr());
        }
        c10::cuda::CUDAGuard guard(device_index_);
        const cudaStream_t stream =
            at::cuda::getCurrentCUDAStream(device_index_).stream();
        cudaError_t status = cudaErrorInvalidValue;
        if (decks.scalar_type() == torch::kInt32) {
            if (interactive) {
                status = semantic
                    ? engine::reset_official_states_seeded_interactive_semantic_i32_async(
                        &arena_,
                        decks.data_ptr<std::int32_t>(),
                        seeds.data_ptr<std::int64_t>(),
                        mask_pointer,
                        stream)
                    : engine::reset_official_states_seeded_interactive_i32_async(
                        &arena_,
                        decks.data_ptr<std::int32_t>(),
                        seeds.data_ptr<std::int64_t>(),
                        mask_pointer,
                        stream);
            } else {
                status = semantic
                    ? engine::reset_official_states_seeded_first_min_semantic_i32_async(
                        &arena_,
                        decks.data_ptr<std::int32_t>(),
                        seeds.data_ptr<std::int64_t>(),
                        mask_pointer,
                        stream)
                    : engine::reset_official_states_seeded_first_min_i32_async(
                        &arena_,
                        decks.data_ptr<std::int32_t>(),
                        seeds.data_ptr<std::int64_t>(),
                        mask_pointer,
                        stream);
            }
        } else {
            if (interactive) {
                status = semantic
                    ? engine::reset_official_states_seeded_interactive_semantic_i64_async(
                        &arena_,
                        decks.data_ptr<std::int64_t>(),
                        seeds.data_ptr<std::int64_t>(),
                        mask_pointer,
                        stream)
                    : engine::reset_official_states_seeded_interactive_i64_async(
                        &arena_,
                        decks.data_ptr<std::int64_t>(),
                        seeds.data_ptr<std::int64_t>(),
                        mask_pointer,
                        stream);
            } else {
                status = semantic
                    ? engine::reset_official_states_seeded_first_min_semantic_i64_async(
                        &arena_,
                        decks.data_ptr<std::int64_t>(),
                        seeds.data_ptr<std::int64_t>(),
                        mask_pointer,
                        stream)
                    : engine::reset_official_states_seeded_first_min_i64_async(
                        &arena_,
                        decks.data_ptr<std::int64_t>(),
                        seeds.data_ptr<std::int64_t>(),
                        mask_pointer,
                        stream);
            }
        }
        check_cuda(
            status,
            interactive
                ? (semantic
                    ? "reset_official_states_seeded_interactive_semantic_async"
                    : "reset_official_states_seeded_interactive_async")
                : (semantic
                    ? "reset_official_states_seeded_first_min_semantic_async"
                    : "reset_official_states_seeded_first_min_async"));
    }

    torch::Tensor view(void* pointer, std::vector<std::int64_t> sizes) const {
        return view_typed(pointer, std::move(sizes), torch::kUInt8);
    }

    torch::Tensor view_typed(
        void* pointer,
        std::vector<std::int64_t> sizes,
        torch::ScalarType dtype) const {
        const auto options = torch::TensorOptions()
                                 .dtype(dtype)
                                 .device(torch::Device(torch::kCUDA, device_index_));
        return torch::from_blob(pointer, std::move(sizes), [](void*) {}, options);
    }

    engine::OfficialDeviceArena arena_{};
    int device_index_ = 0;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.attr("ABI_VERSION") = engine::kAbiVersion;
    module.attr("MAX_POLICIES") = engine::kMaxPolicies;
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
    module.attr("OFFICIAL_STATE_BYTES") = sizeof(engine::OfficialStatePod);
    module.attr("OFFICIAL_ACTION_BYTES") = sizeof(engine::OfficialActionPod);
    module.attr("OFFICIAL_STATE_ABI_VERSION") = engine::kOfficialStateAbiVersion;
    module.attr("OFFICIAL_RULE_ABI_VERSION") = engine::kOfficialRuleAbiVersion;
    module.attr("OFFICIAL_SELECT_CONTEXT_OFFSET") =
        offsetof(engine::OfficialStatePod, select_context);
    module.attr("OFFICIAL_SEEDED_SETUP_POLICY") = "first_min_v1";
    module.attr("OFFICIAL_INTERACTIVE_SETUP_POLICY") = "device_action_v1";
    pybind11::class_<OfficialCudaEngine>(module, "OfficialCudaEngine")
        .def(
            pybind11::init<std::int64_t, std::int64_t, std::int64_t, std::int64_t>(),
            pybind11::arg("batch_size"),
            pybind11::arg("rule_pack_bytes"),
            pybind11::arg("device_index") = 0,
            pybind11::arg("device_stack_bytes") =
                static_cast<std::int64_t>(engine::kOfficialMinimumDeviceStackBytes))
        .def("upload_rule_pack", &OfficialCudaEngine::upload_rule_pack)
        .def("reset_states", &OfficialCudaEngine::reset_states)
        .def("reset_seeded_first_min", &OfficialCudaEngine::reset_seeded_first_min)
        .def(
            "reset_seeded_first_min_semantic",
            &OfficialCudaEngine::reset_seeded_first_min_semantic)
        .def(
            "reset_seeded_first_min_masked",
            &OfficialCudaEngine::reset_seeded_first_min_masked)
        .def(
            "reset_seeded_first_min_semantic_masked",
            &OfficialCudaEngine::reset_seeded_first_min_semantic_masked)
        .def("reset_seeded_interactive", &OfficialCudaEngine::reset_seeded_interactive)
        .def(
            "reset_seeded_interactive_semantic",
            &OfficialCudaEngine::reset_seeded_interactive_semantic)
        .def(
            "reset_seeded_interactive_masked",
            &OfficialCudaEngine::reset_seeded_interactive_masked)
        .def(
            "reset_seeded_interactive_semantic_masked",
            &OfficialCudaEngine::reset_seeded_interactive_semantic_masked)
        .def("classify", &OfficialCudaEngine::classify)
        .def("advance_to_decision", &OfficialCudaEngine::advance_to_decision)
        .def("pack_actions", &OfficialCudaEngine::pack_actions)
        .def("apply_actions", &OfficialCudaEngine::apply_actions)
        .def("apply_packed_actions", &OfficialCudaEngine::apply_packed_actions)
        .def(
            "apply_packed_setup_actions",
            &OfficialCudaEngine::apply_packed_setup_actions)
        .def("encode_policy_v1", &OfficialCudaEngine::encode_policy_v1)
        .def(
            "encode_semantic0031_v2_lanes",
            &OfficialCudaEngine::encode_semantic0031_v2_lanes)
        .def("state_bytes", &OfficialCudaEngine::state_bytes)
        .def("statuses", &OfficialCudaEngine::statuses)
        .def("game_results", &OfficialCudaEngine::game_results)
        .def("decision_actors", &OfficialCudaEngine::decision_actors)
        .def("prize_counts", &OfficialCudaEngine::prize_counts)
        .def("action_bytes", &OfficialCudaEngine::action_bytes)
        .def("semantic_history_raw", &OfficialCudaEngine::semantic_history_raw)
        .def_property_readonly("allocated_bytes", &OfficialCudaEngine::allocated_bytes)
        .def_property_readonly("batch_size", &OfficialCudaEngine::batch_size)
        .def_property_readonly("rule_pack_bytes", &OfficialCudaEngine::rule_pack_bytes);
}
