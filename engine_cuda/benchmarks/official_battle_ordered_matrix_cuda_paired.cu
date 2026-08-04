#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define PTCG_OFFICIAL_BATTLE_CUDA_PAIRED_LIBRARY
#include "official_battle_end_turn_cuda_paired.cu"

namespace {

struct MatrixCaseTotals {
    std::uint64_t decisions = 0;
    std::uint64_t basic_plays = 0;
    std::uint64_t basic_energy_attaches = 0;
    std::uint64_t evolves = 0;
    std::uint64_t optional_declines = 0;
    std::uint64_t prize_selections = 0;
    std::uint64_t prize_cards_taken = 0;
    std::uint64_t active_replacements = 0;
    std::uint64_t trigger_orders = 0;
    std::uint64_t ends = 0;
    std::uint64_t optional_accepts = 0;
    std::uint64_t zero_cardinality_selections = 0;
    std::uint64_t max_cardinality_selections = 0;
    std::uint64_t player0_wins = 0;
    std::uint64_t player1_wins = 0;
    std::uint64_t draws = 0;
    std::uint64_t unfinished = 0;
    std::array<std::uint64_t, 17> option_type_actions{};
    std::array<std::uint64_t, 12> select_type_actions{};
    std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
        effect_offsets_reached{};
    std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
        effect_offsets_applied{};
    std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
        effect_offsets_condition_true{};
    std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
        effect_offsets_condition_false{};
    std::uint32_t min_decisions = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t max_decisions = 0;
};

void add_stats(
    const BattleReplayStats& stats,
    MatrixCaseTotals* totals) {
    totals->decisions += stats.decisions;
    totals->basic_plays += stats.basic_plays;
    totals->basic_energy_attaches += stats.basic_energy_attaches;
    totals->evolves += stats.evolves;
    totals->optional_declines += stats.optional_declines;
    totals->prize_selections += stats.prize_selections;
    totals->prize_cards_taken += stats.prize_cards_taken;
    totals->active_replacements += stats.active_replacements;
    totals->trigger_orders += stats.trigger_orders;
    totals->ends += stats.ends;
    totals->optional_accepts += stats.optional_accepts;
    totals->zero_cardinality_selections += stats.zero_cardinality_selections;
    totals->max_cardinality_selections += stats.max_cardinality_selections;
    add_terminal_result(
        stats.terminal_result,
        &totals->player0_wins,
        &totals->player1_wins,
        &totals->draws,
        &totals->unfinished);
    for (std::size_t index = 0; index < totals->option_type_actions.size(); ++index) {
        totals->option_type_actions[index] += stats.option_type_actions[index];
    }
    for (std::size_t index = 0; index < totals->select_type_actions.size(); ++index) {
        totals->select_type_actions[index] += stats.select_type_actions[index];
    }
    for (std::size_t index = 0;
         index < kBattleReplayEffectCoverageWordCount;
         ++index) {
        totals->effect_offsets_reached[index] |= stats.effect_offsets_reached[index];
        totals->effect_offsets_applied[index] |= stats.effect_offsets_applied[index];
        totals->effect_offsets_condition_true[index] |=
            stats.effect_offsets_condition_true[index];
        totals->effect_offsets_condition_false[index] |=
            stats.effect_offsets_condition_false[index];
    }
    totals->min_decisions = std::min(totals->min_decisions, stats.decisions);
    totals->max_decisions = std::max(totals->max_decisions, stats.decisions);
}

void print_case_result(
    std::uint32_t ordinal,
    const MatrixCaseTotals& totals,
    std::uint64_t seed_start,
    std::uint64_t seed_count,
    BattleReplayPolicy policy) {
    std::cout
        << "CASE\t" << ordinal << "\t{\"passed\":true"
        << ",\"scope\":\"" << battle_replay_scope(policy) << "_cuda_paired\""
        << ",\"seed_start\":" << seed_start
        << ",\"seed_count\":" << seed_count
        << ",\"decisions_compared\":" << totals.decisions
        << ",\"min_decisions_per_battle\":" << totals.min_decisions
        << ",\"max_decisions_per_battle\":" << totals.max_decisions
        << ",\"basic_play_actions\":" << totals.basic_plays
        << ",\"basic_energy_attach_actions\":" << totals.basic_energy_attaches
        << ",\"evolve_actions\":" << totals.evolves
        << ",\"optional_decline_actions\":" << totals.optional_declines
        << ",\"prize_selection_actions\":" << totals.prize_selections
        << ",\"prize_cards_taken\":" << totals.prize_cards_taken
        << ",\"active_replacement_actions\":" << totals.active_replacements
        << ",\"trigger_order_actions\":" << totals.trigger_orders
        << ",\"end_actions\":" << totals.ends
        << ",\"coverage_play_actions\":" << totals.option_type_actions[7]
        << ",\"coverage_attach_actions\":" << totals.option_type_actions[8]
        << ",\"coverage_evolve_actions\":" << totals.option_type_actions[9]
        << ",\"coverage_ability_actions\":" << totals.option_type_actions[10]
        << ",\"coverage_discard_actions\":" << totals.option_type_actions[11]
        << ",\"coverage_retreat_actions\":" << totals.option_type_actions[12]
        << ",\"coverage_attack_actions\":" << totals.option_type_actions[13]
        << ",\"coverage_end_actions\":" << totals.option_type_actions[14]
        << ",\"coverage_yes_actions\":" << totals.optional_accepts
        << ",\"coverage_non_main_actions\":"
        << (is_coverage_policy(policy)
                ? totals.decisions - totals.select_type_actions[1]
                : 0)
        << ",\"coverage_zero_cardinality_actions\":"
        << totals.zero_cardinality_selections
        << ",\"coverage_max_cardinality_actions\":"
        << totals.max_cardinality_selections
        << ",\"branch_coverage_enabled\":"
        << (kBattleReplayBranchCoverageEnabled ? "true" : "false")
        << ",\"effect_offsets_reached\":";
    write_branch_coverage_offsets(std::cout, totals.effect_offsets_reached);
    std::cout << ",\"effect_offsets_applied\":";
    write_branch_coverage_offsets(std::cout, totals.effect_offsets_applied);
    std::cout << ",\"effect_offsets_condition_true\":";
    write_branch_coverage_offsets(
        std::cout, totals.effect_offsets_condition_true);
    std::cout << ",\"effect_offsets_condition_false\":";
    write_branch_coverage_offsets(
        std::cout, totals.effect_offsets_condition_false);
    std::cout
        << ",\"player0_wins\":" << totals.player0_wins
        << ",\"player1_wins\":" << totals.player1_wins
        << ",\"draws\":" << totals.draws
        << ",\"unfinished_battles\":" << totals.unfinished
        << ",\"outcome_mismatches\":0"
        << ",\"state_abi\":" << kOfficialStateAbiVersion
        << ",\"state_bytes\":" << sizeof(OfficialStatePod)
        << ",\"status_mismatches\":0"
        << ",\"state_mismatches\":0}\n"
        << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 7) {
            throw std::runtime_error(
                "usage: official_battle_ordered_matrix_cuda_paired "
                "<rules> <cases-tsv> <seed-start> <seed-count> "
                "<decision-limit> <policy>");
        }
        InitializeAll();
        const std::vector<std::uint8_t> rule_bytes = read_binary_cuda(argv[1]);
        const OfficialRulePackView rules = make_official_rule_pack_view(
            rule_bytes.data());
        std::ifstream cases(argv[2]);
        if (!cases) throw std::runtime_error("cannot open matrix cases TSV");
        const std::uint64_t seed_start = std::stoull(argv[3]);
        const std::uint64_t seed_count = std::stoull(argv[4]);
        const std::uint32_t decision_limit = static_cast<std::uint32_t>(
            std::stoul(argv[5]));
        const BattleReplayPolicy policy = parse_battle_replay_policy(argv[6]);
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
                upload_official_rule_pack(
                    &arena, rule_bytes.data(), rule_bytes.size()),
                "upload rules");
            std::string line;
            while (std::getline(cases, line)) {
                if (line.empty()) continue;
                std::istringstream row(line);
                std::string ordinal_text;
                std::string deck0_path;
                std::string deck1_path;
                if (!std::getline(row, ordinal_text, '\t')
                    || !std::getline(row, deck0_path, '\t')
                    || !std::getline(row, deck1_path, '\t')) {
                    throw std::runtime_error("invalid matrix cases TSV row");
                }
                const std::uint32_t ordinal = static_cast<std::uint32_t>(
                    std::stoul(ordinal_text));
                std::cout << "START\t" << ordinal << '\n' << std::flush;
                const std::vector<std::uint16_t> deck0 = read_deck(deck0_path);
                const std::vector<std::uint16_t> deck1 = read_deck(deck1_path);
                MatrixCaseTotals totals{};
                for (std::uint64_t offset = 0; offset < seed_count; ++offset) {
                    add_stats(
                        run_seed_cuda(
                            deck0,
                            deck1,
                            rules,
                            &arena,
                            seed_start + offset,
                            decision_limit,
                            policy),
                        &totals);
                }
                print_case_result(
                    ordinal, totals, seed_start, seed_count, policy);
            }
            check_cuda(free_official_arena(&arena), "free arena");
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
