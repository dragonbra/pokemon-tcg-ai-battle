#include "ptcg_cuda/official_effects_pod.cuh"
#include "ptcg_cuda/official_continual_effects_pod.cuh"
#include "ptcg_cuda/official_conditions_pod.cuh"
#include "ptcg_cuda/official_targets_pod.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ptcg::cuda_engine;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open rule pack: " + path);
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
}

__host__ __device__ void add_card(
    OfficialStatePod* state,
    std::uint16_t instance,
    std::int32_t card_id,
    std::int32_t player,
    OfficialArea area) {
    OfficialCardStatePod* card = &state->cards[instance];
    card->card_id = card_id;
    card->move_counter = state->move_counter++;
    card->player = static_cast<std::int8_t>(player);
    card->area = static_cast<std::uint8_t>(area);
    official_pod_push_zone_card(state, player, area, OfficialCardRefPod{instance});
}

__host__ __device__ void apply_effect(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialEffectTypeId type,
    std::int32_t value = 0) {
    OfficialEffectRule effect{};
    effect.values[kEffectType] = static_cast<std::int32_t>(type);
    effect.values[kEffectValue0] = value;
    effect.values[kEffectTargetIndex] = 0;
    const OfficialEffectApplyResult result = official_apply_effect_primitive(
        state, rules, effect);
    if (result != OfficialEffectApplyResult::kApplied) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedEffect,
            static_cast<std::int32_t>(type));
    }
}

__host__ __device__ void apply_continual_effect(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialEffectTypeId type,
    std::int32_t value = 0) {
    OfficialEffectRule effect{};
    effect.values[kEffectType] = static_cast<std::int32_t>(type);
    effect.values[kEffectValue0] = value;
    effect.values[kEffectTargetIndex] = 0;
    const OfficialEffectApplyResult result = official_apply_continual_effect_primitive(
        state, rules, effect, 0, OfficialCardRefPod{29});
    if (result != OfficialEffectApplyResult::kApplied) {
        official_pod_fail(
            state,
            OfficialPodError::kUnsupportedEffect,
            static_cast<std::int32_t>(type));
    }
}

__host__ __device__ void set_single_target(
    OfficialStatePod* state,
    OfficialCardRefPod ref) {
    state->targets.count = 0;
    official_pod_push(
        state,
        &state->targets,
        official_pod_area_ref(state, ref),
        OfficialPodError::kSelectionOverflow);
}

__host__ __device__ void exercise_target_contract(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    OfficialTargetRule target{};
    target.values[kTargetPlayer] = 3;
    target.values[kTargetArea0] = static_cast<std::int32_t>(OfficialArea::kActive);
    target.values[kTargetAreaCount] = 1;
    if (!official_build_target_list(
            state,
            rules,
            target,
            &state->targets,
            official_pod_area_ref(state, OfficialCardRefPod{29}),
            0)
        || state->targets.count != 2) {
        official_pod_fail(state, OfficialPodError::kUnsupportedTarget, 2001);
        return;
    }
    OfficialConditionRule condition{};
    condition.values[kConditionType] = static_cast<std::int32_t>(
        OfficialTargetTypeId::kPokemonCard);
    condition.values[kConditionComparator] = 0;
    if (official_match_target_condition(
            state,
            rules,
            OfficialCardRefPod{29},
            condition,
            official_pod_area_ref(state, OfficialCardRefPod{29}))
        != OfficialTargetMatchResult::kMatch) {
        official_pod_fail(state, OfficialPodError::kUnsupportedTarget, 2002);
    }
    OfficialEffectRule condition_effect{};
    condition_effect.values[kEffectConditionType] = static_cast<std::int32_t>(
        OfficialConditionTypeId::kAlways);
    condition_effect.values[kEffectComparator] = 0;
    condition_effect.values[kEffectTargetIndex] = 0;
    if (official_satisfy_condition(
            state,
            rules,
            &condition_effect,
            1,
            0,
            official_pod_area_ref(state, OfficialCardRefPod{29}),
            0)
        != OfficialConditionResult::kTrue) {
        official_pod_fail(state, OfficialPodError::kUnsupportedCondition, 2003);
    }
}

__host__ __device__ void expect_target_match(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    OfficialTargetTypeId type,
    std::int32_t value,
    std::int32_t name_id,
    std::int32_t detail) {
    OfficialConditionRule condition{};
    condition.values[kConditionType] = static_cast<std::int32_t>(type);
    condition.values[kConditionComparator] = 0;
    condition.values[kConditionValue] = value;
    condition.values[kConditionNameId] = name_id;
    if (official_match_target_condition(
            state,
            rules,
            ref,
            condition,
            official_pod_area_ref(state, ref))
        != OfficialTargetMatchResult::kMatch) {
        official_pod_fail(state, OfficialPodError::kUnsupportedTarget, detail);
    }
}

__host__ __device__ void exercise_energy_target_and_condition_contract(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    const OfficialCardRefPod attacker{30};
    for (std::uint16_t instance = 31; instance < 36; ++instance) {
        add_card(state, instance, 4, 0, OfficialArea::kEnergy);
        state->cards[instance].attach_move_counter = state->cards[attacker.index].move_counter;
    }
    expect_target_match(
        state,
        rules,
        OfficialCardRefPod{31},
        OfficialTargetTypeId::kEnergyTypeAttached,
        8,
        0,
        2101);

    state->attacker = attacker;
    state->source_attack_id = 517;
    OfficialEffectRule condition{};
    condition.values[kEffectConditionType] = static_cast<std::int32_t>(
        OfficialConditionTypeId::kAttackEnergyExtra);
    condition.values[kEffectComparator] = 1;
    condition.values[kEffectValue0] = 2;
    condition.values[kEffectTargetIndex] = 0;
    if (official_satisfy_condition(
            state,
            rules,
            &condition,
            1,
            0,
            official_pod_area_ref(state, attacker),
            0)
        != OfficialConditionResult::kTrue) {
        official_pod_fail(state, OfficialPodError::kUnsupportedCondition, 2102);
    }

    add_card(state, 36, 15, 0, OfficialArea::kEnergy);
    state->cards[36].attach_move_counter = state->cards[attacker.index].move_counter;
    expect_target_match(
        state,
        rules,
        attacker,
        OfficialTargetTypeId::kIsAttachedEnergyName,
        0,
        2269,
        2103);

    add_card(state, 37, 707, 0, OfficialArea::kBench);
    add_card(state, 38, 1, 0, OfficialArea::kEnergy);
    state->cards[38].attach_move_counter = state->cards[37].move_counter;
    state->cards[37].continual_state[4] |= 1ULL << 29;
    const OfficialEnergyInfoPod double_grass = official_energy_info(
        *state, rules, OfficialCardRefPod{38}, OfficialCardRefPod{37});
    if (double_grass.type != 1 || double_grass.count != 2) {
        official_pod_fail(state, OfficialPodError::kUnsupportedTarget, 2104);
    }

    add_card(state, 39, 40, 0, OfficialArea::kBench);
    add_card(state, 40, 10, 0, OfficialArea::kEnergy);
    state->cards[40].attach_move_counter = state->cards[39].move_counter;
    const OfficialEnergyInfoPod neo_upper = official_energy_info(
        *state, rules, OfficialCardRefPod{40}, OfficialCardRefPod{39});
    if (neo_upper.type != 511 || neo_upper.count != 2) {
        official_pod_fail(state, OfficialPodError::kUnsupportedTarget, 2105);
    }

    expect_target_match(
        state,
        rules,
        attacker,
        OfficialTargetTypeId::kSameTypeEnemy,
        0,
        0,
        2106);
}

__host__ __device__ void run_scenario(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::uint64_t seed) {
    official_pod_reset(state, 17, seed);
    state->first_player = 0;
    state->phase = static_cast<std::uint8_t>(OfficialGamePhase::kMain);

    for (std::uint16_t instance = 3; instance < 23; ++instance) {
        add_card(state, instance, 7, 0, OfficialArea::kDeck);
    }
    for (std::uint16_t instance = 23; instance < 29; ++instance) {
        add_card(state, instance, 7, 0, OfficialArea::kPrize);
    }
    add_card(state, 29, 535, 0, OfficialArea::kActive);
    add_card(state, 30, 372, 0, OfficialArea::kBench);

    for (std::uint16_t instance = 63; instance < 83; ++instance) {
        add_card(state, instance, 5, 1, OfficialArea::kDeck);
    }
    for (std::uint16_t instance = 83; instance < 89; ++instance) {
        add_card(state, instance, 5, 1, OfficialArea::kPrize);
    }
    add_card(state, 89, 197, 1, OfficialArea::kActive);
    state->cards[89].continual_state[3] |= 4ULL << 48;

    exercise_target_contract(state, rules);
    exercise_energy_target_and_condition_contract(state, rules);

    set_single_target(state, OfficialCardRefPod{29});
    apply_effect(state, rules, OfficialEffectTypeId::kAttackDamageChange, 30);
    apply_effect(state, rules, OfficialEffectTypeId::kDamageChangeThisTurn, 10);
    apply_effect(state, rules, OfficialEffectTypeId::kCannotAttackNextTurn);
    apply_effect(state, rules, OfficialEffectTypeId::kTakeDamageChangeNextEnemyTurn, -20);
    apply_effect(state, rules, OfficialEffectTypeId::kFailAttack);
    apply_effect(state, rules, OfficialEffectTypeId::kCancelFailAttack);
    apply_continual_effect(state, rules, OfficialEffectTypeId::kMaxHpChange, 10);
    apply_continual_effect(state, rules, OfficialEffectTypeId::kDamageChange, 20);
    apply_continual_effect(state, rules, OfficialEffectTypeId::kRetreatCostChange, -1);
    apply_continual_effect(state, rules, OfficialEffectTypeId::kNoSpecialCondition);
    apply_continual_effect(state, rules, OfficialEffectTypeId::kNoToolEffect);
    if (state->attack_damage_change != 30
        || official_packed_i16(state->cards[29].turn_state, 0) != 10
        || (state->cards[29].next_turn[3] & (1U << 2)) == 0
        || static_cast<std::int16_t>(state->cards[29].next_enemy_turn_end & 0xffffU) != -20
        || state->fail_attack != 0
        || state->cards[29].hp_change != 10
        || official_continual_packed_i16(state->cards[29], 2) != 20
        || official_continual_i8(state->cards[29], 26) != -1
        || !official_continual_flag(state->cards[29], 13)
        || state->continual_state != 1) {
        official_pod_fail(state, OfficialPodError::kUnsupportedEffect, 2201);
    }

    official_pod_shuffle_deck(state, 0);
    official_pod_shuffle_deck(state, 1);
    official_pod_draw(state, 0, 3);
    official_pod_draw(state, 1, 3);

    set_single_target(state, OfficialCardRefPod{29});
    apply_effect(state, rules, OfficialEffectTypeId::kPoison);
    apply_effect(state, rules, OfficialEffectTypeId::kBurn);
    official_pod_set_status(state, 1, OfficialBadStatus::kConfused);
    apply_effect(state, rules, OfficialEffectTypeId::kDamageCounter, 3);
    apply_effect(state, rules, OfficialEffectTypeId::kHeal, 10);

    const std::int32_t defender_hp = official_pod_max_hp(
        state, rules, OfficialCardRefPod{89});
    set_single_target(state, OfficialCardRefPod{89});
    apply_effect(
        state,
        rules,
        OfficialEffectTypeId::kDamageCounter,
        defender_hp / 10);
    official_pod_queue_ko_prizes(state, rules, OfficialCardRefPod{89});
    official_pod_move_card(
        state, 1, OfficialArea::kActive, 0, OfficialArea::kTrash);
    official_pod_finish_check(state);

    state->targets.count = 0;
    apply_effect(state, rules, OfficialEffectTypeId::kCoin, 10);
}

__global__ void run_kernel(
    OfficialStatePod* state,
    const std::uint8_t* rule_pack,
    std::uint64_t seed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        run_scenario(state, make_official_rule_pack_view(rule_pack), seed);
    }
}

std::uint64_t byte_digest(const OfficialStatePod& state) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&state);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < sizeof(state); ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: official_core_pod_smoke <official_rules.bin>");
        }
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        if (rule_pack.size() < sizeof(OfficialRulePackHeader)) {
            throw std::runtime_error("rule pack is truncated");
        }
        const auto host_rules = make_official_rule_pack_view(rule_pack.data());
        if (std::memcmp(host_rules.header->magic, "PTCGRUL1", 8) != 0) {
            throw std::runtime_error("rule pack magic mismatch");
        }

        constexpr std::uint64_t seed = 20260730ULL;
        OfficialStatePod cpu_state{};
        run_scenario(&cpu_state, host_rules, seed);

        std::uint8_t* device_rule_pack = nullptr;
        OfficialStatePod* device_state = nullptr;
        check_cuda(cudaMalloc(&device_rule_pack, rule_pack.size()), "cudaMalloc rule pack");
        check_cuda(cudaMalloc(&device_state, sizeof(OfficialStatePod)), "cudaMalloc state");
        check_cuda(cudaMemcpy(
            device_rule_pack,
            rule_pack.data(),
            rule_pack.size(),
            cudaMemcpyHostToDevice), "copy rule pack");
        check_cuda(cudaMemset(device_state, 0, sizeof(OfficialStatePod)), "clear state");
        run_kernel<<<1, 1>>>(device_state, device_rule_pack, seed);
        check_cuda(cudaGetLastError(), "launch core POD smoke");

        OfficialStatePod gpu_state{};
        check_cuda(cudaMemcpy(
            &gpu_state,
            device_state,
            sizeof(gpu_state),
            cudaMemcpyDeviceToHost), "copy state");
        check_cuda(cudaFree(device_state), "cudaFree state");
        check_cuda(cudaFree(device_rule_pack), "cudaFree rule pack");

        const bool equal = std::memcmp(&cpu_state, &gpu_state, sizeof(cpu_state)) == 0;
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout
            << "{\"passed\":" << (equal ? "true" : "false")
            << ",\"device\":\"" << properties.name << "\""
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"cpu_digest\":" << byte_digest(cpu_state)
            << ",\"gpu_digest\":" << byte_digest(gpu_state)
            << ",\"error\":" << gpu_state.error
            << ",\"rng_draws\":" << gpu_state.rng.draw_count
            << ",\"coin_heads\":" << gpu_state.coin_head_count
            << ",\"player0_hand\":" << gpu_state.players[0].hand.count
            << ",\"player1_hand\":" << gpu_state.players[1].hand.count
            << ",\"pending_prizes_player0\":" << gpu_state.pending_prize_count[0]
            << ",\"game_result\":" << static_cast<int>(gpu_state.game_result)
            << ",\"finish_reason\":" << static_cast<int>(gpu_state.finish_reason)
            << "}\n";
        return equal && gpu_state.error == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
