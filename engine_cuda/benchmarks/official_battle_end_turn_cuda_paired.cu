#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define PTCG_OFFICIAL_BATTLE_END_TURN_LIBRARY
#include "../extractor/official_battle_end_turn_paired.cpp"
#include "ptcg_cuda/official_runtime.h"

namespace {

using namespace ptcg::cuda_engine;
using namespace ptcg::cuda_engine::extractor;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::uint8_t> read_binary_cuda(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open rule pack: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

BattleReplayStats run_seed_cuda(
    const std::vector<std::uint16_t>& deck0,
    const std::vector<std::uint16_t>& deck1,
    const OfficialRulePackView& rules,
    OfficialDeviceArena* arena,
    std::uint64_t seed,
    std::uint32_t decision_limit,
    BattleReplayPolicy policy) {
    ApiData official;
    initialize_official_battle(&official, deck0, deck1, seed);
    const std::uint64_t setup_rng_draws = count_rng_draws(seed, official.game.rng);
    OfficialStatePod pod{};
    OfficialBridgeResult bridge = bridge_official_state(
        official.state,
        OfficialStateBridgeContext{seed, setup_rng_draws, true},
        &pod);
    if (!bridge) {
        throw std::runtime_error("initial bridge failed");
    }
    check_cuda(
        upload_official_states_async(
            arena, &pod, cudaMemcpyHostToDevice, nullptr),
        "upload battle state");
    check_cuda(cudaDeviceSynchronize(), "synchronize initial state");

    BattleReplayStats stats{};
    BattleReplayDriverState driver{};
    while (!official.state.isFinish()) {
        if (stats.decisions >= decision_limit) {
            throw std::runtime_error(
                "decision limit exceeded for seed=" + std::to_string(seed));
        }
        int official_index = -1;
        bool basic_play = false;
        bool basic_energy_attach = false;
        bool evolve = false;
        bool optional_decline = false;
        bool prize_selection = false;
        bool active_replacement = false;
        bool trigger_order = false;
        std::vector<int> official_selected;
        OfficialSelectOptionTypeId expected_type = OfficialSelectOptionTypeId::kEnd;
        BattleReplayChoice coverage_choice{};
        if (is_coverage_policy(policy)) {
            coverage_choice = choose_coverage_action(
                official.state,
                pod,
                seed,
                stats.decisions,
                &driver,
                policy == BattleReplayPolicy::kCoverageRandomLegal,
                policy == BattleReplayPolicy::kAttackFirstLegal);
            official_selected = coverage_choice.indices;
            expected_type = coverage_choice.expected_type;
            official_index = official_selected.empty() ? -1 : official_selected.front();
        } else if (official.state.selectContext == SelectContext::Main
            && pod.select_context == kOfficialSelectContextMain) {
            if (policy != BattleReplayPolicy::kEnd) {
            official_index = find_basic_play_option(official.state);
            basic_play = official_index >= 0;
            }
            if (official_index < 0
                && policy == BattleReplayPolicy::kBasicPlayEvolveAttachThenEnd) {
                official_index = find_evolve_option(official.state);
                evolve = official_index >= 0;
            }
            if (official_index < 0
                && (policy == BattleReplayPolicy::kBasicPlayAttachThenEnd
                    || policy == BattleReplayPolicy::kBasicPlayEvolveAttachThenEnd)) {
                official_index = find_basic_energy_attach_option(official.state);
                basic_energy_attach = official_index >= 0;
            }
            if (official_index < 0) official_index = find_end_option(official.state);
            expected_type = basic_play
            ? OfficialSelectOptionTypeId::kPlay
            : (evolve
                ? OfficialSelectOptionTypeId::kEvolve
                : (basic_energy_attach
                    ? OfficialSelectOptionTypeId::kAttach
                    : OfficialSelectOptionTypeId::kEnd));
        } else if (official.state.selectType == SelectType::YesNo
            && pod.select_type == static_cast<std::uint8_t>(OfficialSelectTypeId::kYesNo)) {
            for (std::size_t index = 0; index < official.state.options.size(); ++index) {
                if (official.state.options[index].type == SelectOptionType::No) {
                    official_index = static_cast<int>(index);
                    break;
                }
            }
            optional_decline = true;
            expected_type = OfficialSelectOptionTypeId::kNo;
        } else if (official.state.selectType == SelectType::Card
            && official.state.selectContext == SelectContext::ToHand
            && pod.select_type == static_cast<std::uint8_t>(OfficialSelectTypeId::kCard)
            && pod.select_context == kOfficialSelectContextToHand) {
            if (official.state.selectMin <= 0
                || official.state.selectMin > std::ssize(official.state.options)
                || official.state.selectMin > static_cast<int>(kOfficialOptionCapacity)) {
                throw std::runtime_error("unsupported prize selection count");
            }
            official_index = 0;
            for (int index = 0; index < official.state.selectMin; ++index) {
                official_selected.push_back(index);
            }
            prize_selection = true;
            expected_type = OfficialSelectOptionTypeId::kCard;
        } else if (official.state.selectType == SelectType::Card
            && official.state.selectContext == SelectContext::ToActive
            && pod.select_type == static_cast<std::uint8_t>(OfficialSelectTypeId::kCard)
            && pod.select_context == kOfficialSelectContextToActive) {
            official_index = 0;
            active_replacement = true;
            expected_type = OfficialSelectOptionTypeId::kCard;
        } else if (official.state.selectType == SelectType::Skill
            && official.state.selectContext == SelectContext::SkillOrder
            && pod.select_type == static_cast<std::uint8_t>(OfficialSelectTypeId::kSkill)
            && pod.select_context == kOfficialSelectContextSkillOrder) {
            if (official.state.selectMin <= 0
                || official.state.selectMin > std::ssize(official.state.options)
                || official.state.selectMin > static_cast<int>(kOfficialOptionCapacity)) {
                throw std::runtime_error("unsupported trigger order count");
            }
            official_index = 0;
            for (int index = 0; index < official.state.selectMin; ++index) {
                official_selected.push_back(index);
            }
            trigger_order = true;
            expected_type = OfficialSelectOptionTypeId::kSkill;
        } else {
            throw std::runtime_error("unsupported policy decision in CUDA harness");
        }
        if (!is_coverage_policy(policy)
            && (official_index < 0
                || official_index >= pod.options.count
                || pod.options.values[official_index].type
                    != static_cast<std::uint8_t>(expected_type))) {
            throw std::runtime_error("policy option mismatch before CUDA action");
        }
        if (official_selected.empty()
            && !is_coverage_policy(policy)) {
            official_selected.push_back(official_index);
        }
        for (const int index : official_selected) {
            if (index < 0 || index >= pod.options.count
                || (!is_coverage_policy(policy)
                    && pod.options.values[index].type
                        != static_cast<std::uint8_t>(expected_type))) {
                throw std::runtime_error("policy multi-option mismatch before CUDA action");
            }
        }
        if (is_coverage_policy(policy)) {
            record_coverage_choice(official.state, coverage_choice, &stats);
        }
        const std::string action_description = selected_action_sequence(
            pod, official_selected);
        if (ApiSelect(&official,
                official_selected.empty() ? nullptr : official_selected.data(),
                static_cast<int>(official_selected.size())) != 0) {
            throw std::runtime_error("official action failed");
        }
        OfficialActionPod action{};
        action.count = static_cast<std::uint16_t>(official_selected.size());
        for (std::size_t index = 0; index < official_selected.size(); ++index) {
            action.option_indices[index] = static_cast<std::uint16_t>(
                official_selected[index]);
        }
        const OfficialFlowStatus cpu_status = official_apply_pending_action(
            &pod, rules, action.option_indices, action.count);
        check_cuda(
            cudaMemcpy(
                arena->actions,
                &action,
                sizeof(action),
                cudaMemcpyHostToDevice),
            "upload policy action");
        check_cuda(
            apply_official_actions_and_advance_async(arena, nullptr, nullptr),
            "launch policy action");
        check_cuda(cudaDeviceSynchronize(), "synchronize policy action");

        OfficialStatePod gpu{};
        std::uint8_t gpu_status_value = 0;
        check_cuda(
            cudaMemcpy(
                &gpu,
                arena->states,
                sizeof(gpu),
                cudaMemcpyDeviceToHost),
            "download CUDA state");
        check_cuda(
            cudaMemcpy(
                &gpu_status_value,
                arena->statuses,
                sizeof(gpu_status_value),
                cudaMemcpyDeviceToHost),
            "download CUDA status");
        const OfficialFlowStatus gpu_status = static_cast<OfficialFlowStatus>(
            gpu_status_value);
        ++stats.decisions;
        if (!is_coverage_policy(policy)) {
            if (basic_play) ++stats.basic_plays;
            else if (evolve) ++stats.evolves;
            else if (basic_energy_attach) ++stats.basic_energy_attaches;
            else if (optional_decline) ++stats.optional_declines;
            else if (prize_selection) {
                ++stats.prize_selections;
                stats.prize_cards_taken += static_cast<std::uint32_t>(
                    official_selected.size());
            }
            else if (active_replacement) ++stats.active_replacements;
            else if (trigger_order) ++stats.trigger_orders;
            else ++stats.ends;
        }
        if (cpu_status != gpu_status
            || std::memcmp(&pod, &gpu, sizeof(pod)) != 0) {
            const std::string cpu_gpu_state = first_byte_mismatch(pod, gpu);
            throw std::runtime_error(
                "CPU POD/CUDA mismatch seed=" + std::to_string(seed)
                + ".decision=" + std::to_string(stats.decisions)
                + ".action=" + action_description
                + ".cpu_status=" + std::to_string(static_cast<int>(cpu_status))
                + ".gpu_status=" + std::to_string(static_cast<int>(gpu_status))
                + ".cpu_gpu_state_mismatch=" + cpu_gpu_state
                + ".cpu_attack_stage=" + std::to_string(pod.attack_flow_stage)
                + ".gpu_attack_stage=" + std::to_string(gpu.attack_flow_stage)
                + ".cpu_select=" + std::to_string(pod.select_type) + ":"
                + std::to_string(pod.select_context) + ":"
                + std::to_string(pod.select_min) + ":"
                + std::to_string(pod.select_max)
                + ".gpu_select=" + std::to_string(gpu.select_type) + ":"
                + std::to_string(gpu.select_context) + ":"
                + std::to_string(gpu.select_min) + ":"
                + std::to_string(gpu.select_max)
                + ".cpu_continuations=" + continuation_sequence(pod)
                + ".gpu_continuations=" + continuation_sequence(gpu)
                + ".cpu_effect=" + effect_sequence(pod)
                + ".gpu_effect=" + effect_sequence(gpu)
                + ".cpu_current_attack=" + std::to_string(pod.current_attack_id)
                + ".gpu_current_attack=" + std::to_string(gpu.current_attack_id)
                + ".cpu_source_attack=" + std::to_string(pod.source_attack_id)
                + ".gpu_source_attack=" + std::to_string(gpu.source_attack_id));
        }
        const bool terminal = official.state.isFinish();
        if ((terminal && cpu_status != OfficialFlowStatus::kTerminal)
            || (!terminal && cpu_status != OfficialFlowStatus::kNeedsAction)) {
            OfficialStatePod status_expected{};
            const OfficialBridgeResult status_bridge = bridge_official_state(
                official.state,
                OfficialStateBridgeContext{seed, pod.rng.draw_count, true},
                &status_expected);
            throw std::runtime_error(
                "CPU status/official terminal mismatch seed="
                + std::to_string(seed)
                + ".decision=" + std::to_string(stats.decisions)
                + ".action=" + action_description
                + ".terminal=" + std::to_string(terminal)
                + ".cpu_status="
                + std::to_string(static_cast<int>(cpu_status))
                + ".gpu_status="
                + std::to_string(static_cast<int>(gpu_status))
                + ".pod_result=" + std::to_string(pod.game_result)
                + ".gpu_result=" + std::to_string(gpu.game_result)
                + ".pod_error=" + std::to_string(pod.error)
                + ".pod_detail=" + std::to_string(pod.error_detail)
                + ".bridge_ok="
                + std::to_string(static_cast<bool>(status_bridge))
                + ".official_result="
                + std::to_string(status_expected.game_result)
                + ".official_flow_flags="
                + std::to_string(status_expected.flow_flags)
                + ".pod_flow_flags=" + std::to_string(pod.flow_flags)
                + ".official_attack_stage="
                + std::to_string(status_expected.attack_flow_stage)
                + ".pod_attack_stage="
                + std::to_string(pod.attack_flow_stage)
                + ".official_continuations="
                + continuation_sequence(status_expected)
                + ".pod_continuations=" + continuation_sequence(pod)
                + ".official_effect=" + effect_sequence(status_expected)
                + ".pod_effect=" + effect_sequence(pod));
        }
        bridge = bridge_official_state(
            official.state,
            OfficialStateBridgeContext{seed, pod.rng.draw_count, true},
            &gpu);
        if (!bridge) throw std::runtime_error("decision bridge failed");
#if defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
        gpu.branch_coverage = pod.branch_coverage;
#endif
        // The official bridge is canonicalized on host to ignore inactive list tails.
        require_state_equal(
            seed,
            stats.decisions,
            action_description,
            official.state,
            gpu,
            pod);
    }
    stats.terminal_result = pod.game_result;
#if defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
    for (std::size_t index = 0;
         index < kBattleReplayEffectCoverageWordCount;
         ++index) {
        stats.effect_offsets_reached[index] = pod.branch_coverage.reached[index];
        stats.effect_offsets_applied[index] = pod.branch_coverage.applied[index];
        stats.effect_offsets_condition_true[index] =
            pod.branch_coverage.condition_true[index];
        stats.effect_offsets_condition_false[index] =
            pod.branch_coverage.condition_false[index];
    }
#endif
    return stats;
}

}  // namespace

#ifndef PTCG_OFFICIAL_BATTLE_CUDA_PAIRED_LIBRARY
int main(int argc, char** argv) {
    try {
        if (argc != 8) {
            throw std::runtime_error(
                "usage: official_battle_end_turn_cuda_paired "
                "<rules> <deck0> <deck1> <seed-start> <seed-count> "
                "<decision-limit> <policy>");
        }
        InitializeAll();
        const std::string rules_path = argv[1];
        const std::vector<std::uint8_t> rule_bytes = read_binary_cuda(rules_path);
        const OfficialRulePackView rules = make_official_rule_pack_view(
            rule_bytes.data());
        const std::vector<std::uint16_t> deck0 = read_deck(argv[2]);
        const std::vector<std::uint16_t> deck1 = read_deck(argv[3]);
        const std::uint64_t seed_start = std::stoull(argv[4]);
        const std::uint64_t seed_count = std::stoull(argv[5]);
        const std::uint32_t decision_limit = static_cast<std::uint32_t>(
            std::stoul(argv[6]));
        const BattleReplayPolicy policy = parse_battle_replay_policy(argv[7]);
        if (seed_count == 0 || decision_limit == 0) {
            throw std::runtime_error("seed-count and decision-limit must be positive");
        }

        OfficialRuntimeConfig config{};
        config.batch_size = 1;
        config.rule_pack_bytes = rule_bytes.size();
        OfficialDeviceArena arena{};
        check_cuda(allocate_official_arena(&arena, config), "allocate arena");
        try {
            check_cuda(
                upload_official_rule_pack(&arena, rule_bytes.data(), rule_bytes.size()),
                "upload rules");
            std::uint64_t total_decisions = 0;
            std::uint64_t total_basic_plays = 0;
            std::uint64_t total_basic_energy_attaches = 0;
            std::uint64_t total_evolves = 0;
            std::uint64_t total_optional_declines = 0;
            std::uint64_t total_prize_selections = 0;
            std::uint64_t total_prize_cards_taken = 0;
            std::uint64_t total_active_replacements = 0;
            std::uint64_t total_trigger_orders = 0;
            std::uint64_t total_ends = 0;
            std::uint64_t total_optional_accepts = 0;
            std::uint64_t total_zero_cardinality_selections = 0;
            std::uint64_t total_max_cardinality_selections = 0;
            std::uint64_t total_player0_wins = 0;
            std::uint64_t total_player1_wins = 0;
            std::uint64_t total_draws = 0;
            std::uint64_t total_unfinished = 0;
            std::array<std::uint64_t, 17> total_option_type_actions{};
            std::array<std::uint64_t, 12> total_select_type_actions{};
            std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
                total_effect_offsets_reached{};
            std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
                total_effect_offsets_applied{};
            std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
                total_effect_offsets_condition_true{};
            std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
                total_effect_offsets_condition_false{};
            std::uint32_t min_decisions = decision_limit;
            std::uint32_t max_decisions = 0;
            for (std::uint64_t offset = 0; offset < seed_count; ++offset) {
                const BattleReplayStats stats = run_seed_cuda(
                    deck0,
                    deck1,
                    rules,
                    &arena,
                    seed_start + offset,
                    decision_limit,
                    policy);
                total_decisions += stats.decisions;
                total_basic_plays += stats.basic_plays;
                total_basic_energy_attaches += stats.basic_energy_attaches;
                total_evolves += stats.evolves;
                total_optional_declines += stats.optional_declines;
                total_prize_selections += stats.prize_selections;
                total_prize_cards_taken += stats.prize_cards_taken;
                total_active_replacements += stats.active_replacements;
                total_trigger_orders += stats.trigger_orders;
                total_ends += stats.ends;
                total_optional_accepts += stats.optional_accepts;
                total_zero_cardinality_selections +=
                    stats.zero_cardinality_selections;
                total_max_cardinality_selections +=
                    stats.max_cardinality_selections;
                add_terminal_result(
                    stats.terminal_result,
                    &total_player0_wins,
                    &total_player1_wins,
                    &total_draws,
                    &total_unfinished);
                for (std::size_t index = 0;
                    index < total_option_type_actions.size(); ++index) {
                    total_option_type_actions[index] += stats.option_type_actions[index];
                }
                for (std::size_t index = 0;
                    index < total_select_type_actions.size(); ++index) {
                    total_select_type_actions[index] += stats.select_type_actions[index];
                }
                for (std::size_t index = 0;
                     index < kBattleReplayEffectCoverageWordCount;
                     ++index) {
                    total_effect_offsets_reached[index] |=
                        stats.effect_offsets_reached[index];
                    total_effect_offsets_applied[index] |=
                        stats.effect_offsets_applied[index];
                    total_effect_offsets_condition_true[index] |=
                        stats.effect_offsets_condition_true[index];
                    total_effect_offsets_condition_false[index] |=
                        stats.effect_offsets_condition_false[index];
                }
                min_decisions = std::min(min_decisions, stats.decisions);
                max_decisions = std::max(max_decisions, stats.decisions);
            }
            check_cuda(free_official_arena(&arena), "free arena");
            std::cout
                << "{\"passed\":true"
                << ",\"scope\":\"" << battle_replay_scope(policy)
                << "_cuda_paired\""
                << ",\"seed_start\":" << seed_start
                << ",\"seed_count\":" << seed_count
                << ",\"decisions_compared\":" << total_decisions
                << ",\"min_decisions_per_battle\":" << min_decisions
                << ",\"max_decisions_per_battle\":" << max_decisions
                << ",\"basic_play_actions\":" << total_basic_plays
                << ",\"basic_energy_attach_actions\":"
                << total_basic_energy_attaches
                << ",\"evolve_actions\":" << total_evolves
                << ",\"optional_decline_actions\":"
                << total_optional_declines
                << ",\"prize_selection_actions\":" << total_prize_selections
                << ",\"prize_cards_taken\":" << total_prize_cards_taken
                << ",\"active_replacement_actions\":"
                << total_active_replacements
                << ",\"trigger_order_actions\":" << total_trigger_orders
                << ",\"end_actions\":" << total_ends
                << ",\"coverage_play_actions\":" << total_option_type_actions[7]
                << ",\"coverage_attach_actions\":" << total_option_type_actions[8]
                << ",\"coverage_evolve_actions\":" << total_option_type_actions[9]
                << ",\"coverage_ability_actions\":" << total_option_type_actions[10]
                << ",\"coverage_discard_actions\":" << total_option_type_actions[11]
                << ",\"coverage_retreat_actions\":" << total_option_type_actions[12]
                << ",\"coverage_attack_actions\":" << total_option_type_actions[13]
                << ",\"coverage_end_actions\":" << total_option_type_actions[14]
                << ",\"coverage_yes_actions\":" << total_optional_accepts
                << ",\"coverage_non_main_actions\":"
                << (is_coverage_policy(policy)
                    ? total_decisions - total_select_type_actions[1]
                    : 0)
                << ",\"coverage_zero_cardinality_actions\":"
                << total_zero_cardinality_selections
                << ",\"coverage_max_cardinality_actions\":"
                << total_max_cardinality_selections
                << ",\"branch_coverage_enabled\":"
                << (kBattleReplayBranchCoverageEnabled ? "true" : "false")
                << ",\"effect_offsets_reached\":";
            write_branch_coverage_offsets(
                std::cout, total_effect_offsets_reached);
            std::cout << ",\"effect_offsets_applied\":";
            write_branch_coverage_offsets(
                std::cout, total_effect_offsets_applied);
            std::cout << ",\"effect_offsets_condition_true\":";
            write_branch_coverage_offsets(
                std::cout, total_effect_offsets_condition_true);
            std::cout << ",\"effect_offsets_condition_false\":";
            write_branch_coverage_offsets(
                std::cout, total_effect_offsets_condition_false);
            std::cout
                << ",\"player0_wins\":" << total_player0_wins
                << ",\"player1_wins\":" << total_player1_wins
                << ",\"draws\":" << total_draws
                << ",\"unfinished_battles\":" << total_unfinished
                << ",\"outcome_mismatches\":0"
                << ",\"state_abi\":" << kOfficialStateAbiVersion
                << ",\"state_bytes\":" << sizeof(OfficialStatePod)
                << ",\"status_mismatches\":0"
                << ",\"state_mismatches\":0}\n";
            return 0;
        } catch (...) {
            free_official_arena(&arena);
            throw;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
#endif
