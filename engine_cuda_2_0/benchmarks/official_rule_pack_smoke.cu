#include "ptcg_cuda/official_rule_layout.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace engine = ptcg::cuda_engine;

namespace {

struct RuleProbe {
    std::uint64_t fnv1a;
    std::uint32_t card_count;
    std::uint32_t skill_count;
    std::uint32_t attack_count;
    std::uint32_t effect_count;
    std::uint32_t continuation_count;
    std::int32_t last_card_id;
    std::int32_t last_skill_id;
    std::int32_t last_attack_id;
    std::uint32_t lookup_failures;
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

__host__ __device__ RuleProbe probe_rules(const std::uint8_t* data) {
    const engine::OfficialRulePackView view = engine::make_official_rule_pack_view(data);
    RuleProbe result{};
    result.fnv1a = 1469598103934665603ULL;
    for (std::uint64_t i = 0; i < view.header->total_bytes; ++i) {
        result.fnv1a ^= data[i];
        result.fnv1a *= 1099511628211ULL;
    }
    result.card_count = view.header->counts[0];
    result.skill_count = view.header->counts[1];
    result.attack_count = view.header->counts[2];
    result.effect_count = view.header->counts[3];
    result.continuation_count = view.header->counts[10];
    result.last_card_id = view.cards[result.card_count - 1].values[engine::kCardId];
    result.last_skill_id = view.skills[result.skill_count - 1].values[engine::kSkillId];
    result.last_attack_id = view.attacks[result.attack_count - 1].values[engine::kAttackId];
    result.lookup_failures += engine::official_card_rule(view, result.last_card_id) == nullptr;
    result.lookup_failures += engine::official_skill_rule(view, result.last_skill_id) == nullptr;
    result.lookup_failures += engine::official_attack_rule(view, result.last_attack_id) == nullptr;
    result.lookup_failures += engine::official_card_rule(view, 0) != nullptr;
    result.lookup_failures += engine::official_skill_rule(view, 0) != nullptr;
    result.lookup_failures += engine::official_attack_rule(view, 0) != nullptr;
    return result;
}

__global__ void probe_kernel(const std::uint8_t* data, RuleProbe* output) {
    *output = probe_rules(data);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: official_rule_pack_smoke <official_rules.bin>");
        }
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open official rule pack");
        }
        const std::vector<std::uint8_t> bytes(
            (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (bytes.size() < sizeof(engine::OfficialRulePackHeader)) {
            throw std::runtime_error("official rule pack is truncated");
        }
        const RuleProbe expected = probe_rules(bytes.data());

        std::uint8_t* device_bytes = nullptr;
        RuleProbe* device_probe = nullptr;
        check_cuda(cudaMalloc(&device_bytes, bytes.size()), "allocate rule pack");
        check_cuda(cudaMalloc(&device_probe, sizeof(RuleProbe)), "allocate probe");
        check_cuda(
            cudaMemcpy(device_bytes, bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
            "copy rule pack");
        probe_kernel<<<1, 1>>>(device_bytes, device_probe);
        check_cuda(cudaGetLastError(), "launch rule probe");
        RuleProbe actual{};
        check_cuda(
            cudaMemcpy(&actual, device_probe, sizeof(actual), cudaMemcpyDeviceToHost),
            "copy rule probe");
        check_cuda(cudaFree(device_probe), "free probe");
        check_cuda(cudaFree(device_bytes), "free rule pack");

        const bool passed =
            expected.fnv1a == actual.fnv1a
            && expected.card_count == actual.card_count
            && expected.skill_count == actual.skill_count
            && expected.attack_count == actual.attack_count
            && expected.effect_count == actual.effect_count
            && expected.continuation_count == actual.continuation_count
            && expected.last_card_id == actual.last_card_id
            && expected.last_skill_id == actual.last_skill_id
            && expected.last_attack_id == actual.last_attack_id
            && expected.lookup_failures == 0
            && actual.lookup_failures == 0;
        std::cout
            << "{\"passed\":" << (passed ? "true" : "false")
            << ",\"bytes\":" << bytes.size()
            << ",\"fnv1a\":" << actual.fnv1a
            << ",\"cards\":" << actual.card_count
            << ",\"skills\":" << actual.skill_count
            << ",\"attacks\":" << actual.attack_count
            << ",\"effects\":" << actual.effect_count
            << ",\"continuations\":" << actual.continuation_count
            << ",\"lookup_failures\":" << actual.lookup_failures
            << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
