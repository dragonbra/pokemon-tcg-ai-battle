#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "All.h"
#include "official_state_bridge.h"
#include "ptcg_cuda/official_flow_dispatch_pod.cuh"

namespace {

using namespace ptcg::cuda_engine;
using namespace ptcg::cuda_engine::extractor;

constexpr std::size_t kDeckSize = 60;
constexpr std::size_t kBattleReplayEffectCoverageWordCount = 64;
#if defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
constexpr bool kBattleReplayBranchCoverageEnabled = true;
#else
constexpr bool kBattleReplayBranchCoverageEnabled = false;
#endif

enum class BattleReplayPolicy : std::uint8_t {
    kEnd = 0,
    kBasicPlayThenEnd = 1,
    kBasicPlayAttachThenEnd = 2,
    kBasicPlayEvolveAttachThenEnd = 3,
    kCoverageFirstLegal = 4,
    kCoverageRandomLegal = 5,
};

bool is_coverage_policy(BattleReplayPolicy policy) {
    return policy == BattleReplayPolicy::kCoverageFirstLegal
        || policy == BattleReplayPolicy::kCoverageRandomLegal;
}

struct BattleReplayStats {
    std::uint32_t decisions = 0;
    std::uint8_t terminal_result = 0;
    std::uint32_t basic_plays = 0;
    std::uint32_t basic_energy_attaches = 0;
    std::uint32_t evolves = 0;
    std::uint32_t optional_declines = 0;
    std::uint32_t prize_selections = 0;
    std::uint32_t prize_cards_taken = 0;
    std::uint32_t active_replacements = 0;
    std::uint32_t trigger_orders = 0;
    std::uint32_t ends = 0;
    std::uint32_t optional_accepts = 0;
    std::uint32_t zero_cardinality_selections = 0;
    std::uint32_t max_cardinality_selections = 0;
    std::array<std::uint32_t, 17> option_type_actions{};
    std::array<std::uint32_t, 12> select_type_actions{};
    std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
        effect_offsets_reached{};
    std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
        effect_offsets_applied{};
    std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
        effect_offsets_condition_true{};
    std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>
        effect_offsets_condition_false{};
};

void write_branch_coverage_offsets(
    std::ostream& output,
    const std::array<std::uint64_t, kBattleReplayEffectCoverageWordCount>& words) {
    output << '[';
    bool first = true;
    for (std::size_t word_index = 0; word_index < words.size(); ++word_index) {
        std::uint64_t word = words[word_index];
        for (std::uint32_t bit = 0; bit < 64U; ++bit) {
            if ((word & (1ULL << bit)) == 0) continue;
            if (!first) output << ',';
            first = false;
            output << word_index * 64U + bit;
        }
    }
    output << ']';
}

void add_terminal_result(
    std::uint8_t result,
    std::uint64_t* player0_wins,
    std::uint64_t* player1_wins,
    std::uint64_t* draws,
    std::uint64_t* unfinished) {
    if (result == static_cast<std::uint8_t>(OfficialGameResult::kPlayer0Win)) {
        ++*player0_wins;
    } else if (result == static_cast<std::uint8_t>(
                   OfficialGameResult::kPlayer1Win)) {
        ++*player1_wins;
    } else if (result == static_cast<std::uint8_t>(OfficialGameResult::kDraw)) {
        ++*draws;
    } else {
        ++*unfinished;
    }
}

struct BattleReplayDriverState {
    std::int32_t turn = -1;
    std::uint32_t main_actions_this_turn = 0;
};

struct BattleReplayChoice {
    std::vector<int> indices;
    OfficialSelectOptionTypeId expected_type = OfficialSelectOptionTypeId::kEnd;
    bool selected_maximum = false;
};

std::uint64_t coverage_choice_hash(
    std::uint64_t seed,
    std::uint32_t decision,
    std::uint64_t salt) {
    std::uint64_t value = seed
        ^ (static_cast<std::uint64_t>(decision) << 32U)
        ^ salt;
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open rule pack: " + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<std::uint16_t> read_deck(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open deck: " + path);
    std::vector<std::uint16_t> cards;
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t comma = line.find(',');
        std::string token = line.substr(0, comma);
        while (!token.empty()
            && (token.back() == '\r' || token.back() == ' '
                || token.back() == '\t')) {
            token.pop_back();
        }
        try {
            std::size_t parsed = 0;
            const unsigned long value = std::stoul(token, &parsed);
            if (parsed == token.size()) {
                cards.push_back(static_cast<std::uint16_t>(value));
            }
        } catch (const std::exception&) {
        }
    }
    if (cards.size() != kDeckSize) {
        throw std::runtime_error("deck must contain 60 card IDs: " + path);
    }
    return cards;
}

std::uint64_t count_rng_draws(
    std::uint64_t seed,
    const std::mt19937& final_rng) {
    std::mt19937 candidate(static_cast<std::uint32_t>(seed));
    for (std::uint64_t count = 0; count <= 1'000'000; ++count) {
        if (candidate == final_rng) return count;
        candidate();
    }
    throw std::runtime_error("initial RNG draw count exceeded search bound");
}

void initialize_official_battle(
    ApiData* battle,
    const std::vector<std::uint16_t>& deck0,
    const std::vector<std::uint16_t>& deck1,
    std::uint64_t seed,
    bool record_log = false) {
    GameConfig config{};
    config.seed = static_cast<std::uint32_t>(seed);
    config.recordLog = record_log;
    config.deviceRand = false;
    for (std::size_t index = 0; index < kDeckSize; ++index) {
        config.decks[0].cards[index] = deck0[index];
        config.decks[1].cards[index] = deck1[index];
    }
    battle->init(config);
    battle->start();
    battle->next();
    while (!battle->state.isFinish()
        && battle->state.selectContext != SelectContext::Main) {
        std::array<int, 128> selected{};
        for (int index = 0; index < battle->state.selectMin; ++index) {
            selected[index] = index;
        }
        const int error = ApiSelect(
            battle,
            battle->state.selectMin == 0 ? nullptr : selected.data(),
            battle->state.selectMin);
        if (error != 0) {
            throw std::runtime_error(
                "official setup selection failed: " + std::to_string(error));
        }
    }
    if (battle->state.isFinish()) {
        throw std::runtime_error("official battle terminated during setup");
    }
}

int find_end_option(const State& state) {
    for (std::size_t index = 0; index < state.options.size(); ++index) {
        if (state.options[index].type == SelectOptionType::End) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

int find_basic_play_option(const State& state) {
    const PlayerState& player = state.players[state.activePlayerIndex()];
    for (std::size_t index = 0; index < state.options.size(); ++index) {
        const SelectOption& option = state.options[index];
        if (option.type != SelectOptionType::Play
            || option.param0 < 0
            || option.param0 >= std::ssize(player.hand)) {
            continue;
        }
        const Card& card = state.getCard(player.hand.at(option.param0));
        const CardMaster& master = card.getMaster();
        if (master.cardType == CardType::Pokemon
            && master.evolutionType == EvolutionType::Basic
            && master.ability == nullptr) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

int find_basic_energy_attach_option(const State& state) {
    const PlayerState& player = state.players[state.activePlayerIndex()];
    for (std::size_t index = 0; index < state.options.size(); ++index) {
        const SelectOption& option = state.options[index];
        if (option.type != SelectOptionType::Attach
            || option.param1 < 0
            || option.param1 >= std::ssize(player.hand)) {
            continue;
        }
        const Card& card = state.getCard(player.hand.at(option.param1));
        if (card.getMaster().cardType == CardType::BasicEnergy) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

int find_evolve_option(const State& state) {
    for (std::size_t index = 0; index < state.options.size(); ++index) {
        if (state.options[index].type == SelectOptionType::Evolve) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

int find_option(const State& state, SelectOptionType type) {
    for (std::size_t index = 0; index < state.options.size(); ++index) {
        if (state.options[index].type == type) return static_cast<int>(index);
    }
    return -1;
}

BattleReplayChoice choose_coverage_action(
    const State& state,
    const OfficialStatePod& pod,
    std::uint64_t seed,
    std::uint32_t decision,
    BattleReplayDriverState* driver,
    bool randomized = false) {
    if (state.options.size() != pod.options.count
        || state.selectMin != pod.select_min
        || state.selectMax != pod.select_max
        || static_cast<std::uint8_t>(state.selectType) != pod.select_type
        || static_cast<std::uint8_t>(state.selectContext) != pod.select_context) {
        throw std::runtime_error("coverage policy received divergent selection contract");
    }
    if (state.options.size() > kOfficialOptionCapacity
        || state.selectMin < 0
        || state.selectMax < state.selectMin
        || state.selectMax > std::ssize(state.options)) {
        throw std::runtime_error(
            "invalid selection bounds for coverage policy: min="
            + std::to_string(state.selectMin)
            + ".max=" + std::to_string(state.selectMax)
            + ".options=" + std::to_string(state.options.size()));
    }

    BattleReplayChoice choice{};
    if (state.selectContext == SelectContext::Main) {
        if (driver->turn != state.turn) {
            driver->turn = state.turn;
            driver->main_actions_this_turn = 0;
        }
        constexpr std::array<SelectOptionType, 6> kCoverageTypes{
            SelectOptionType::Ability,
            SelectOptionType::Play,
            SelectOptionType::Evolve,
            SelectOptionType::Attach,
            SelectOptionType::Discard,
            SelectOptionType::Retreat,
        };
        int selected = -1;
        if (driver->main_actions_this_turn < 4) {
            if (randomized) {
                std::vector<int> candidates;
                for (std::size_t index = 0; index < state.options.size(); ++index) {
                    if (std::find(
                            kCoverageTypes.begin(),
                            kCoverageTypes.end(),
                            state.options[index].type) != kCoverageTypes.end()) {
                        candidates.push_back(static_cast<int>(index));
                    }
                }
                if (!candidates.empty()) {
                    selected = candidates[coverage_choice_hash(
                        seed,
                        decision,
                        static_cast<std::uint64_t>(state.turn)
                            ^ driver->main_actions_this_turn) % candidates.size()];
                }
            } else {
                const std::size_t start = (
                    static_cast<std::size_t>(state.turn)
                    + driver->main_actions_this_turn) % kCoverageTypes.size();
                for (std::size_t offset = 0; offset < kCoverageTypes.size(); ++offset) {
                    selected = find_option(
                        state, kCoverageTypes[(start + offset) % kCoverageTypes.size()]);
                    if (selected >= 0) break;
                }
            }
        }
        if (selected < 0 && randomized) {
            std::vector<int> attacks;
            for (std::size_t index = 0; index < state.options.size(); ++index) {
                if (state.options[index].type == SelectOptionType::Attack) {
                    attacks.push_back(static_cast<int>(index));
                }
            }
            const int end = find_end_option(state);
            const std::uint64_t terminal_hash = coverage_choice_hash(
                seed, decision, 0x7465726d696e616cULL);
            if (end >= 0 && (attacks.empty() || terminal_hash % 5U == 0U)) {
                selected = end;
            } else if (!attacks.empty()) {
                selected = attacks[terminal_hash % attacks.size()];
            }
        }
        if (selected < 0) selected = find_option(state, SelectOptionType::Attack);
        if (selected < 0) selected = find_end_option(state);
        if (selected < 0) {
            throw std::runtime_error("coverage policy found no terminating main option");
        }
        const SelectOptionType type = state.options[selected].type;
        choice.indices.push_back(selected);
        choice.expected_type = static_cast<OfficialSelectOptionTypeId>(
            static_cast<std::uint8_t>(type));
        if (type != SelectOptionType::Attack && type != SelectOptionType::End) {
            ++driver->main_actions_this_turn;
        }
        return choice;
    }

    if (state.selectType == SelectType::YesNo) {
        const int yes = find_option(state, SelectOptionType::Yes);
        if (yes < 0) throw std::runtime_error("YesNo selection has no Yes option");
        const int no = find_option(state, SelectOptionType::No);
        const bool choose_no = randomized && no >= 0
            && (coverage_choice_hash(seed, decision, 0x7965736e6fULL) & 1ULL) != 0;
        choice.indices.push_back(choose_no ? no : yes);
        choice.expected_type = choose_no
            ? OfficialSelectOptionTypeId::kNo
            : OfficialSelectOptionTypeId::kYes;
        return choice;
    }

    const std::uint64_t selection_hash = randomized
        ? coverage_choice_hash(
            seed,
            decision,
            static_cast<std::uint8_t>(state.selectContext))
        : seed + decision + static_cast<std::uint8_t>(state.selectContext);
    const bool choose_maximum = (selection_hash & 1ULL) != 0;
    const int selected_count = choose_maximum ? state.selectMax : state.selectMin;
    choice.selected_maximum = choose_maximum && state.selectMax > state.selectMin;
    if (selected_count == 0) {
        choice.expected_type = state.options.empty()
            ? OfficialSelectOptionTypeId::kNumber
            : static_cast<OfficialSelectOptionTypeId>(
                static_cast<std::uint8_t>(state.options.front().type));
        return choice;
    }
    const std::size_t start = static_cast<std::size_t>(selection_hash)
        % state.options.size();
    for (int offset = 0; offset < selected_count; ++offset) {
        choice.indices.push_back(static_cast<int>(
            (start + static_cast<std::size_t>(offset)) % state.options.size()));
    }
    choice.expected_type = static_cast<OfficialSelectOptionTypeId>(
        static_cast<std::uint8_t>(state.options[choice.indices.front()].type));
    return choice;
}

void record_coverage_choice(
    const State& before,
    const BattleReplayChoice& choice,
    BattleReplayStats* stats) {
    const std::size_t select_type = static_cast<std::uint8_t>(before.selectType);
    if (select_type < stats->select_type_actions.size()) {
        ++stats->select_type_actions[select_type];
    }
    if (choice.indices.empty()) {
        ++stats->zero_cardinality_selections;
    } else {
        const std::size_t option_type = static_cast<std::uint8_t>(
            before.options[choice.indices.front()].type);
        if (option_type < stats->option_type_actions.size()) {
            ++stats->option_type_actions[option_type];
        }
        if (before.selectType == SelectType::YesNo
            && before.options[choice.indices.front()].type == SelectOptionType::Yes) {
            ++stats->optional_accepts;
        } else if (before.selectType == SelectType::YesNo
            && before.options[choice.indices.front()].type == SelectOptionType::No) {
            ++stats->optional_declines;
        }
    }
    if (choice.selected_maximum) ++stats->max_cardinality_selections;
}

BattleReplayPolicy parse_battle_replay_policy(const std::string& value) {
    if (value == "end") return BattleReplayPolicy::kEnd;
    if (value == "basic-play-then-end") {
        return BattleReplayPolicy::kBasicPlayThenEnd;
    }
    if (value == "basic-play-attach-then-end") {
        return BattleReplayPolicy::kBasicPlayAttachThenEnd;
    }
    if (value == "basic-play-evolve-attach-then-end") {
        return BattleReplayPolicy::kBasicPlayEvolveAttachThenEnd;
    }
    if (value == "coverage-first-legal") {
        return BattleReplayPolicy::kCoverageFirstLegal;
    }
    if (value == "coverage-random-legal") {
        return BattleReplayPolicy::kCoverageRandomLegal;
    }
    throw std::runtime_error("unknown battle replay policy: " + value);
}

const char* battle_replay_scope(BattleReplayPolicy policy) {
    if (policy == BattleReplayPolicy::kEnd) {
        return "end_turn_only_full_battle";
    }
    if (policy == BattleReplayPolicy::kBasicPlayThenEnd) {
        return "basic_play_then_end_full_battle";
    }
    if (policy == BattleReplayPolicy::kBasicPlayAttachThenEnd) {
        return "basic_play_attach_then_end_full_battle";
    }
    if (policy == BattleReplayPolicy::kBasicPlayEvolveAttachThenEnd) {
        return "basic_play_evolve_attach_then_end_full_battle";
    }
    if (policy == BattleReplayPolicy::kCoverageFirstLegal) {
        return "coverage_first_legal_full_battle";
    }
    return "coverage_random_legal_full_battle";
}

std::string first_byte_mismatch(
    const OfficialStatePod& official,
    const OfficialStatePod& pod) {
    const auto* expected = reinterpret_cast<const std::uint8_t*>(&official);
    const auto* actual = reinterpret_cast<const std::uint8_t*>(&pod);
    for (std::size_t index = 0; index < sizeof(official); ++index) {
        if (expected[index] != actual[index]) {
            return "offset=" + std::to_string(index)
                + ".official=" + std::to_string(expected[index])
                + ".pod=" + std::to_string(actual[index]);
        }
    }
    return "none";
}

std::string mismatch_location(std::size_t offset) {
    const auto scalar_location = [offset](
        std::size_t begin,
        std::size_t size,
        const char* name) -> std::string {
        if (offset < begin || offset >= begin + size) return {};
        return std::string(name) + ".byte=" + std::to_string(offset - begin);
    };
#define PTCG_SCALAR_LOCATION(field)                                          \
    if (const std::string location = scalar_location(                       \
            offsetof(OfficialStatePod, field),                              \
            sizeof(OfficialStatePod::field),                                \
            #field);                                                        \
        !location.empty()) return location
    PTCG_SCALAR_LOCATION(turn);
    PTCG_SCALAR_LOCATION(turn_action_count);
    PTCG_SCALAR_LOCATION(effect_action_count);
    PTCG_SCALAR_LOCATION(turn_attack_count);
    PTCG_SCALAR_LOCATION(current_card_effect_index);
    PTCG_SCALAR_LOCATION(coin_head_count);
    PTCG_SCALAR_LOCATION(move_counter);
    PTCG_SCALAR_LOCATION(current_skill_order);
#undef PTCG_SCALAR_LOCATION
    const std::size_t cards_begin = offsetof(OfficialStatePod, cards);
    const std::size_t cards_end = cards_begin
        + sizeof(OfficialCardStatePod) * kOfficialCardCapacity;
    if (offset >= cards_begin && offset < cards_end) {
        const std::size_t relative = offset - cards_begin;
        return "card=" + std::to_string(relative / sizeof(OfficialCardStatePod))
            + ".card_offset="
            + std::to_string(relative % sizeof(OfficialCardStatePod));
    }
    const std::size_t players_begin = offsetof(OfficialStatePod, players);
    const std::size_t players_end = players_begin
        + sizeof(OfficialPlayerStatePod) * 2;
    if (offset >= players_begin && offset < players_end) {
        const std::size_t relative = offset - players_begin;
        return "player="
            + std::to_string(relative / sizeof(OfficialPlayerStatePod))
            + ".player_offset="
            + std::to_string(relative % sizeof(OfficialPlayerStatePod));
    }
    const std::size_t options_begin = offsetof(OfficialStatePod, options);
    const std::size_t options_end = options_begin
        + sizeof(OfficialSelectOptionPod) * kOfficialOptionCapacity
        + sizeof(std::uint16_t) * 2;
    if (offset >= options_begin && offset < options_end) {
        const std::size_t relative = offset - options_begin;
        return "options_offset=" + std::to_string(relative)
            + ".option=" + std::to_string(
                relative / sizeof(OfficialSelectOptionPod));
    }
    return "state_offset=" + std::to_string(offset);
}

std::string official_continual_source_sequence(const State& state) {
    std::ostringstream output;
    if (state.game == nullptr) return output.str();
    for (std::size_t index = 0;
         index < state.game->cardEffectList.size();
         ++index) {
        if (index != 0) output << ';';
        const CardEffect& source = state.game->cardEffectList[index];
        const Card& card = state.getCard(source.ref);
        output << index << ':' << static_cast<int>(source.ref.cardIndex)
            << ':' << static_cast<int>(card.cardId)
            << ':' << static_cast<int>(card.area)
            << ':' << static_cast<int>(card.playerIndex)
            << ':' << static_cast<int>(source.priority)
            << ':' << source.skillOrder
            << ':' << source.moveCounter;
    }
    return output.str();
}

std::string option_sequence(const OfficialStatePod& state) {
    std::ostringstream output;
    for (std::uint16_t index = 0; index < state.options.count; ++index) {
        if (index != 0) output << ';';
        const OfficialSelectOptionPod& option = state.options.values[index];
        const std::uint16_t ref = option.resolved_card;
        const std::int32_t card_id = ref < kOfficialCardCapacity
            ? state.cards[ref].card_id
            : -1;
        output << index << ':' << static_cast<int>(option.type)
            << ':' << ref << ':' << card_id
            << ':' << option.params[0]
            << ':' << option.params[1]
            << ':' << option.params[2]
            << ':' << option.params[3];
    }
    return output.str();
}

std::string selected_action_sequence(
    const OfficialStatePod& state,
    const std::vector<int>& selected) {
    std::ostringstream output;
    output << "select=" << static_cast<int>(state.select_type)
        << ':' << static_cast<int>(state.select_context)
        << ':' << state.select_min << ':' << state.select_max
        << ":control=" << static_cast<int>(state.control_flags)
        << ":flow=" << static_cast<int>(state.flow_flags)
        << ':' << static_cast<int>(state.turn_flow_stage)
        << ':' << static_cast<int>(state.refresh_flow_stage)
        << ":attack=" << state.current_attack_id
        << ':' << state.source_attack_id
        << ':' << static_cast<int>(state.attack_flow_stage)
        << ':' << static_cast<int>(state.attack_flow_flags)
        << ":effect=" << static_cast<int>(state.effect_interpreter.active)
        << ':' << static_cast<int>(state.effect_interpreter.awaiting_selection)
        << ':' << static_cast<int>(state.effect_interpreter.resume_kind)
        << ':' << state.effect_interpreter.effect_offset
        << ':' << state.effect_interpreter.effect_index
        << ':' << state.effect_interpreter.effect_count
        << ":continuations=" << state.continuations.count << '[';
    for (std::uint16_t index = 0; index < state.continuations.count; ++index) {
        if (index != 0) output << ';';
        const OfficialContinuationPod& frame = state.continuations.values[index];
        output << frame.opcode
            << ':' << static_cast<int>(frame.arg_type)
            << ':' << static_cast<int>(frame.call_count)
            << ':' << static_cast<int>(frame.called_count)
            << ':' << frame.args[0]
            << ':' << frame.args[1]
            << ':' << frame.args[2];
    }
    output << ']'
        << ":turn_used_skills=";
    for (std::uint16_t index = 0; index < state.turn_used_skills.count; ++index) {
        if (index != 0) output << ',';
        output << state.turn_used_skills.values[index];
    }
    output << ":indices=";
    for (std::size_t ordinal = 0; ordinal < selected.size(); ++ordinal) {
        if (ordinal != 0) output << ',';
        const int index = selected[ordinal];
        output << index;
        if (index < 0 || index >= state.options.count) continue;
        const OfficialSelectOptionPod& option = state.options.values[index];
        const std::uint16_t ref = option.resolved_card;
        const std::int32_t card_id = ref < kOfficialCardCapacity
            ? state.cards[ref].card_id : -1;
        output << '[' << static_cast<int>(option.type)
            << ':' << ref << ':' << card_id
            << ':' << option.params[0]
            << ':' << option.params[1]
            << ':' << option.params[2]
            << ':' << option.params[3] << ']';
    }
    return output.str();
}

std::string continuation_sequence(const OfficialStatePod& state) {
    std::ostringstream output;
    for (std::uint16_t index = 0; index < state.continuations.count; ++index) {
        if (index != 0) output << ';';
        const OfficialContinuationPod& frame = state.continuations.values[index];
        output << index << ':' << frame.opcode
            << ':' << static_cast<int>(frame.arg_type)
            << ':' << static_cast<int>(frame.call_count)
            << ':' << static_cast<int>(frame.called_count)
            << ':' << frame.args[0]
            << ':' << frame.args[1]
            << ':' << frame.args[2];
    }
    return output.str();
}

template <std::size_t Capacity>
std::string integer_sequence(
    const OfficialPodList<std::int32_t, Capacity>& values) {
    std::ostringstream output;
    for (std::uint16_t index = 0; index < values.count; ++index) {
        if (index != 0) output << ',';
        output << values.values[index];
    }
    return output.str();
}

std::string area_ref_sequence(
    const OfficialStatePod& state,
    const OfficialAreaRefPod& area_ref) {
    std::ostringstream output;
    const std::uint16_t ref = area_ref.card.index;
    output << ref << ':' << area_ref.move_counter;
    if (ref < kOfficialCardCapacity) {
        const OfficialCardStatePod& card = state.cards[ref];
        output << ':' << card.card_id
            << ':' << static_cast<int>(card.area)
            << ':' << static_cast<int>(card.player)
            << ':' << card.move_counter;
    } else {
        output << ":-1:-1:-1:-1";
    }
    return output.str();
}

template <std::size_t Capacity>
std::string area_ref_list_sequence(
    const OfficialStatePod& state,
    const OfficialPodList<OfficialAreaRefPod, Capacity>& values) {
    std::ostringstream output;
    for (std::uint16_t index = 0; index < values.count; ++index) {
        if (index != 0) output << ';';
        output << index << ':' << area_ref_sequence(state, values.values[index]);
    }
    return output.str();
}

std::string effect_sequence(const OfficialStatePod& state) {
    std::ostringstream output;
    output << state.effect_state.ability.skill_id
        << ':' << area_ref_sequence(state, state.effect_state.ability.effect_card)
        << ':' << static_cast<int>(state.effect_state.ability.use_player)
        << ':' << static_cast<int>(state.effect_state.effect_index)
        << ':' << static_cast<int>(state.effect_state.on_effect)
        << ":trigger=" << static_cast<int>(state.trigger_info.type)
        << ':' << static_cast<int>(state.trigger_info.depth)
        << ':' << state.trigger_info.value
        << ":subject=" << area_ref_sequence(state, state.trigger_info.subject)
        << ":object=" << area_ref_sequence(state, state.trigger_info.object);
    return output.str();
}

template <std::size_t Capacity>
std::string trigger_sequence(
    const OfficialStatePod& state,
    const OfficialPodList<OfficialTriggeredAbilityPod, Capacity>& triggers) {
    std::ostringstream output;
    for (std::uint16_t index = 0; index < triggers.count; ++index) {
        if (index != 0) output << ';';
        const OfficialTriggeredAbilityPod& ability = triggers.values[index];
        output << index
            << ":skill=" << ability.activate.skill_id
            << ":effect=" << area_ref_sequence(
                state, ability.activate.effect_card)
            << ":use=" << static_cast<int>(ability.activate.use_player)
            << ":trigger=" << static_cast<int>(ability.trigger.type)
            << ':' << static_cast<int>(ability.trigger.depth)
            << ':' << ability.trigger.value
            << ":subject=" << area_ref_sequence(state, ability.trigger.subject)
            << ":object=" << area_ref_sequence(state, ability.trigger.object);
    }
    return output.str();
}

std::string in_play_sequence(const OfficialStatePod& state) {
    std::ostringstream output;
    bool first = true;
    for (int player = 0; player < 2; ++player) {
        const auto append = [&](const auto& zone, const char* name) {
            for (std::uint16_t index = 0; index < zone.count; ++index) {
                const OfficialCardRefPod ref = zone.values[index];
                const OfficialCardStatePod& card = state.cards[ref.index];
                if (!first) output << ';';
                first = false;
                output << player << ':' << name << ':' << index
                    << ':' << ref.index << ':' << card.card_id
                    << ':' << card.damage << ':' << card.continual_state[4]
                    << ':' << card.runtime_flags;
            }
        };
        append(state.players[player].active, "A");
        append(state.players[player].bench, "B");
    }
    return output.str();
}

std::string card_debug_sequence(
    const OfficialStatePod& state,
    OfficialCardRefPod ref) {
    if (ref.index >= kOfficialCardCapacity) return "invalid";
    const OfficialCardStatePod& card = state.cards[ref.index];
    std::ostringstream output;
    output << ref.index
        << ":card=" << card.card_id
        << ":player=" << static_cast<int>(card.player)
        << ":area=" << static_cast<int>(card.area)
        << ":pre_area=" << static_cast<int>(card.pre_area)
        << ":move=" << card.move_counter
        << ":attach=" << card.attach_move_counter
        << ":damage=" << card.damage
        << ":runtime=" << card.runtime_flags
        << ":this=" << card.this_turn[0] << "," << card.this_turn[1]
        << "," << card.this_turn[2] << "," << card.this_turn[3]
        << ":next=" << card.next_turn[0] << "," << card.next_turn[1]
        << "," << card.next_turn[2] << "," << card.next_turn[3]
        << ":enemy=" << card.this_turn_enemy << "," << card.next_turn_enemy
        << ":cannot_non_active=" << card.cannot_use_attack_id_non_active
        << ":ability_used=" << card.ability_used_count;
    return output.str();
}

template <std::size_t Capacity>
std::string card_ref_list_debug_sequence(
    const OfficialStatePod& state,
    const OfficialPodList<OfficialCardRefPod, Capacity>& values) {
    std::ostringstream output;
    for (std::uint16_t index = 0; index < values.count; ++index) {
        if (index != 0) output << ';';
        output << index << ':' << card_debug_sequence(state, values.values[index]);
    }
    return output.str();
}

std::string player_debug_sequence(
    const OfficialStatePod& state,
    std::int32_t player) {
    if (player < 0 || player > 1) return "invalid";
    const OfficialPlayerStatePod& ps = state.players[player];
    std::ostringstream output;
    output << "active=" << card_ref_list_debug_sequence(state, ps.active)
        << ":bench=" << card_ref_list_debug_sequence(state, ps.bench)
        << ":pre_evolution=" << card_ref_list_debug_sequence(
            state, ps.pre_evolution)
        << ":energy=" << card_ref_list_debug_sequence(state, ps.energy)
        << ":tool=" << card_ref_list_debug_sequence(state, ps.tool)
        << ":this_turn=" << ps.this_turn
        << ":continual=" << ps.continual_state;
    return output.str();
}

template <typename T, std::size_t Capacity>
void canonicalize_list(OfficialPodList<T, Capacity>* list) {
    for (std::size_t index = list->count; index < Capacity; ++index) {
        list->values[index] = T{};
    }
    list->reserved = 0;
}

void canonicalize_state(OfficialStatePod* state) {
    // POD-only resumable execution state has no field in official State. The
    // official semantic boundary is represented by continuations, trigger/effect
    // state, selection fields, options, and RNG, all of which remain compared.
    state->interpreter_steps = 0;
    state->pending_prize_count[0] = 0;
    state->pending_prize_count[1] = 0;
    state->pending_active_replacement_mask = 0;
    state->reserved_flow = 0;
    state->flow_flags = 0;
    state->attack_flow_stage = 0;
    state->attack_flow_flags = 0;
    state->turn_flow_stage = 0;
    state->refresh_flow_stage = 0;
    state->effect_interpreter = OfficialEffectInterpreterPod{};
    state->trigger_resolver = OfficialTriggerResolverPod{};
    bool has_after_refresh = false;
    for (std::uint16_t index = 0; index < state->continuations.count; ++index) {
        if (state->continuations.values[index].opcode
            == static_cast<std::uint16_t>(OfficialContinuationId::kAfterRefresh)) {
            has_after_refresh = true;
            break;
        }
    }
    if (has_after_refresh) {
        // stateChanged is CPU Refresh recursion scratch throughout prize taking
        // and active replacement. The POD's explicit refresh/KO stages carry
        // the equivalent resume state, so either retained value is inert while
        // the shared AfterRefresh continuation is present.
        state->control_flags &= static_cast<std::uint8_t>(~kOfficialChangedFlag);
    }
    canonicalize_list(&state->stadium);
    canonicalize_list(&state->looking);
    canonicalize_list(&state->selected_list);
    canonicalize_list(&state->each_list);
    canonicalize_list(&state->playing);
    canonicalize_list(&state->check_list);
    for (OfficialPlayerStatePod& player : state->players) {
        canonicalize_list(&player.active);
        canonicalize_list(&player.bench);
        canonicalize_list(&player.prize);
        canonicalize_list(&player.hand);
        canonicalize_list(&player.deck);
        canonicalize_list(&player.trash);
        canonicalize_list(&player.energy);
        canonicalize_list(&player.tool);
        canonicalize_list(&player.pre_evolution);
        canonicalize_list(&player.temporary);
    }
    canonicalize_list(&state->options);
    canonicalize_list(&state->selected);
    canonicalize_list(&state->pre_targets);
    canonicalize_list(&state->targets);
    canonicalize_list(&state->ko_list);
    state->prize_requests = {};
    canonicalize_list(&state->delay_triggers);
    canonicalize_list(&state->temporary_triggers);
    canonicalize_list(&state->triggers);
    canonicalize_list(&state->turn_used_skills);
    canonicalize_list(&state->turn_play);
    canonicalize_list(&state->turn_heal);
    canonicalize_list(&state->turn_evolve);
    canonicalize_list(&state->effect_stack);
    if (state->game_result != static_cast<std::uint8_t>(
            OfficialGameResult::kNone)) {
        // Selection metadata is inert after a terminal result. Depending on
        // which continuation detects the winner, official State may retain
        // the just-consumed selection while the resumable POD has already
        // cleared it. Neither state can be stepped again, so normalize this
        // stale boundary metadata together with the terminal continuations.
        state->select_type = 0;
        state->select_context = 0;
        state->select_player = 0;
        state->select_min = 0;
        state->select_max = 0;
        state->select_deck = 0;
        state->options = {};
        state->selected = {};
        state->continuations = {};
        state->control_flags &= static_cast<std::uint8_t>(~kOfficialChangedFlag);
    } else {
        canonicalize_list(&state->continuations);
    }
    canonicalize_list(&state->effect_ref_scratch);
}

void require_state_equal(
    std::uint64_t seed,
    std::uint32_t decision,
    const std::string& action,
    const State& official_source,
    const OfficialStatePod& official,
    const OfficialStatePod& pod) {
    OfficialStatePod official_canonical = official;
    OfficialStatePod pod_canonical = pod;
    canonicalize_state(&official_canonical);
    canonicalize_state(&pod_canonical);
    if (std::memcmp(
            &official_canonical,
            &pod_canonical,
            sizeof(pod_canonical)) == 0) return;
    const auto* official_bytes = reinterpret_cast<const std::uint8_t*>(
        &official_canonical);
    const auto* pod_bytes = reinterpret_cast<const std::uint8_t*>(
        &pod_canonical);
    std::size_t mismatch_offset = 0;
    while (mismatch_offset < sizeof(pod_canonical)
        && official_bytes[mismatch_offset] == pod_bytes[mismatch_offset]) {
        ++mismatch_offset;
    }
    const std::size_t options_begin = offsetof(OfficialStatePod, options);
    const std::size_t mismatch_option = mismatch_offset >= options_begin
        && mismatch_offset < options_begin + sizeof(official_canonical.options.values)
        ? (mismatch_offset - options_begin) / sizeof(OfficialSelectOptionPod)
        : kOfficialOptionCapacity;
    const std::size_t cards_begin = offsetof(OfficialStatePod, cards);
    const std::size_t mismatch_card = mismatch_offset >= cards_begin
        && mismatch_offset < cards_begin
            + sizeof(OfficialCardStatePod) * kOfficialCardCapacity
        ? (mismatch_offset - cards_begin) / sizeof(OfficialCardStatePod)
        : kOfficialCardCapacity;
    const OfficialSelectOptionPod empty_option{};
    const OfficialCardStatePod empty_card{};
    const OfficialCardStatePod& official_mismatch_card =
        mismatch_card < kOfficialCardCapacity
        ? official_canonical.cards[mismatch_card] : empty_card;
    const OfficialCardStatePod& pod_mismatch_card =
        mismatch_card < kOfficialCardCapacity
        ? pod_canonical.cards[mismatch_card] : empty_card;
    const OfficialSelectOptionPod& official_mismatch_option =
        mismatch_option < official_canonical.options.count
        ? official_canonical.options.values[mismatch_option]
        : empty_option;
    const OfficialSelectOptionPod& pod_mismatch_option =
        mismatch_option < pod_canonical.options.count
        ? pod_canonical.options.values[mismatch_option]
        : empty_option;
    throw std::runtime_error(
        "seed=" + std::to_string(seed)
        + ".decision=" + std::to_string(decision)
        + ".action=" + action
        + ".state_mismatch."
        + first_byte_mismatch(official_canonical, pod_canonical)
        + ".location=" + mismatch_location(mismatch_offset)
        + ".mismatch_card=" + std::to_string(mismatch_card)
        + ".mismatch_card_official="
        + std::to_string(official_mismatch_card.card_id) + ","
        + std::to_string(official_mismatch_card.move_counter) + ","
        + std::to_string(official_mismatch_card.area) + ","
        + std::to_string(official_mismatch_card.pre_area) + ","
        + std::to_string(official_mismatch_card.reverse) + ","
        + std::to_string(official_mismatch_card.player)
        + ".mismatch_card_pod="
        + std::to_string(pod_mismatch_card.card_id) + ","
        + std::to_string(pod_mismatch_card.move_counter) + ","
        + std::to_string(pod_mismatch_card.area) + ","
        + std::to_string(pod_mismatch_card.pre_area) + ","
        + std::to_string(pod_mismatch_card.reverse) + ","
        + std::to_string(pod_mismatch_card.player)
        + ".card127_off36_official="
        + std::to_string(official_canonical.cards[127].next_turn[0])
        + ".card127_off36_pod="
        + std::to_string(pod_canonical.cards[127].next_turn[0])
        + ".card127_id_official="
        + std::to_string(official_canonical.cards[127].card_id)
        + ".card127_id_pod="
        + std::to_string(pod_canonical.cards[127].card_id)
        + ".card127_area_official="
        + std::to_string(official_canonical.cards[127].area)
        + ".card127_area_pod="
        + std::to_string(pod_canonical.cards[127].area)
        + ".option0_type_official="
        + std::to_string(official_canonical.options.values[0].type)
        + ".option0_type_pod="
        + std::to_string(pod_canonical.options.values[0].type)
        + ".option0_resolved_official="
        + std::to_string(official_canonical.options.values[0].resolved_card)
        + ".option0_resolved_pod="
        + std::to_string(pod_canonical.options.values[0].resolved_card)
        + ".option0_equiv_official="
        + std::to_string(official_canonical.options.values[0].option_equiv)
        + ".option0_equiv_pod="
        + std::to_string(pod_canonical.options.values[0].option_equiv)
        + ".option0_params_official="
        + std::to_string(official_canonical.options.values[0].params[0])
        + "," + std::to_string(official_canonical.options.values[0].params[1])
        + "," + std::to_string(official_canonical.options.values[0].params[2])
        + "," + std::to_string(official_canonical.options.values[0].params[3])
        + ".option0_params_pod="
        + std::to_string(pod_canonical.options.values[0].params[0])
        + "," + std::to_string(pod_canonical.options.values[0].params[1])
        + "," + std::to_string(pod_canonical.options.values[0].params[2])
        + "," + std::to_string(pod_canonical.options.values[0].params[3])
        + ".p1_hand_count_official="
        + std::to_string(official_canonical.players[1].hand.count)
        + ".p1_hand_count_pod="
        + std::to_string(pod_canonical.players[1].hand.count)
        + ".p1_hand0_official="
        + std::to_string(official_canonical.players[1].hand.values[0].index)
        + ".p1_hand0_pod="
        + std::to_string(pod_canonical.players[1].hand.values[0].index)
        + ".option1_type_official="
        + std::to_string(official_canonical.options.values[1].type)
        + ".option1_type_pod="
        + std::to_string(pod_canonical.options.values[1].type)
        + ".option1_resolved_official="
        + std::to_string(official_canonical.options.values[1].resolved_card)
        + ".option1_resolved_pod="
        + std::to_string(pod_canonical.options.values[1].resolved_card)
        + ".option1_equiv_official="
        + std::to_string(official_canonical.options.values[1].option_equiv)
        + ".option1_equiv_pod="
        + std::to_string(pod_canonical.options.values[1].option_equiv)
        + ".option1_params_official="
        + std::to_string(official_canonical.options.values[1].params[0])
        + "," + std::to_string(official_canonical.options.values[1].params[1])
        + "," + std::to_string(official_canonical.options.values[1].params[2])
        + "," + std::to_string(official_canonical.options.values[1].params[3])
        + ".option1_params_pod="
        + std::to_string(pod_canonical.options.values[1].params[0])
        + "," + std::to_string(pod_canonical.options.values[1].params[1])
        + "," + std::to_string(pod_canonical.options.values[1].params[2])
        + "," + std::to_string(pod_canonical.options.values[1].params[3])
        + ".official_turn=" + std::to_string(official.turn)
        + ".pod_turn=" + std::to_string(pod.turn)
        + ".official_result=" + std::to_string(official.game_result)
        + ":" + std::to_string(official.finish_reason)
        + ".pod_result=" + std::to_string(pod.game_result)
        + ":" + std::to_string(pod.finish_reason)
        + ".official_zones="
        + std::to_string(official.players[0].prize.count) + ":"
        + std::to_string(official.players[0].active.count) + ":"
        + std::to_string(official.players[0].bench.count) + ":"
        + std::to_string(official.players[1].prize.count) + ":"
        + std::to_string(official.players[1].active.count) + ":"
        + std::to_string(official.players[1].bench.count)
        + ".pod_zones="
        + std::to_string(pod.players[0].prize.count) + ":"
        + std::to_string(pod.players[0].active.count) + ":"
        + std::to_string(pod.players[0].bench.count) + ":"
        + std::to_string(pod.players[1].prize.count) + ":"
        + std::to_string(pod.players[1].active.count) + ":"
        + std::to_string(pod.players[1].bench.count)
        + ".official_control_flags="
        + std::to_string(official_canonical.control_flags)
        + ".pod_control_flags="
        + std::to_string(pod_canonical.control_flags)
        + ".official_select="
        + std::to_string(official.select_type) + ":"
        + std::to_string(official.select_context) + ":"
        + std::to_string(official.select_player) + ":"
        + std::to_string(official.select_min) + ":"
        + std::to_string(official.select_max)
        + ".pod_select="
        + std::to_string(pod.select_type) + ":"
        + std::to_string(pod.select_context) + ":"
        + std::to_string(pod.select_player) + ":"
        + std::to_string(pod.select_min) + ":"
        + std::to_string(pod.select_max)
        + ".official_turn_action_count="
        + std::to_string(official.turn_action_count)
        + ".pod_turn_action_count="
        + std::to_string(pod.turn_action_count)
        + ".official_raw_attack_stage="
        + std::to_string(official.attack_flow_stage)
        + ".pod_raw_attack_stage="
        + std::to_string(pod.attack_flow_stage)
        + ".official_error=" + std::to_string(official.error)
        + ".pod_error=" + std::to_string(pod.error)
        + ".pod_detail=" + std::to_string(pod.error_detail)
        + ".official_targets=" + std::to_string(official_canonical.targets.count)
        + ".pod_targets=" + std::to_string(pod_canonical.targets.count)
        + ".pod_target0_ref=" + std::to_string(
            pod_canonical.targets.count == 0
                ? 0 : pod_canonical.targets.values[0].card.index)
        + ".pod_target0_card=" + std::to_string(
            pod_canonical.targets.count == 0
                ? 0 : pod_canonical.cards[
                    pod_canonical.targets.values[0].card.index].card_id)
        + ".pod_target0_area=" + std::to_string(
            pod_canonical.targets.count == 0
                ? 0 : pod_canonical.cards[
                    pod_canonical.targets.values[0].card.index].area)
        + ".official_pre_targets=" + std::to_string(official_canonical.pre_targets.count)
        + ".pod_pre_targets=" + std::to_string(pod_canonical.pre_targets.count)
        + ".official_pre_target_refs="
        + area_ref_list_sequence(official_canonical, official_canonical.pre_targets)
        + ".pod_pre_target_refs="
        + area_ref_list_sequence(pod_canonical, pod_canonical.pre_targets)
        + ".official_turn_flow=" + std::to_string(official_canonical.turn_flow_stage)
        + ".pod_turn_flow=" + std::to_string(pod_canonical.turn_flow_stage)
        + ".official_refresh_flow=" + std::to_string(official_canonical.refresh_flow_stage)
        + ".pod_refresh_flow=" + std::to_string(pod_canonical.refresh_flow_stage)
        + ".official_option_count="
        + std::to_string(official_canonical.options.count)
        + ".pod_option_count=" + std::to_string(pod_canonical.options.count)
        + ".mismatch_option=" + std::to_string(mismatch_option)
        + ".mismatch_option_official="
        + std::to_string(official_mismatch_option.type) + ","
        + std::to_string(official_mismatch_option.resolved_card) + ","
        + std::to_string(official_mismatch_option.params[0]) + ","
        + std::to_string(official_mismatch_option.params[1]) + ","
        + std::to_string(official_mismatch_option.params[2]) + ","
        + std::to_string(official_mismatch_option.params[3])
        + ".mismatch_option_pod="
        + std::to_string(pod_mismatch_option.type) + ","
        + std::to_string(pod_mismatch_option.resolved_card) + ","
        + std::to_string(pod_mismatch_option.params[0]) + ","
        + std::to_string(pod_mismatch_option.params[1]) + ","
        + std::to_string(pod_mismatch_option.params[2]) + ","
        + std::to_string(pod_mismatch_option.params[3])
        + ".official_options=" + option_sequence(official_canonical)
        + ".pod_options=" + option_sequence(pod_canonical)
        + ".official_in_play=" + in_play_sequence(official_canonical)
        + ".pod_in_play=" + in_play_sequence(pod_canonical)
        + ".official_player0_debug="
        + player_debug_sequence(official_canonical, 0)
        + ".pod_player0_debug="
        + player_debug_sequence(pod_canonical, 0)
        + ".official_player1_debug="
        + player_debug_sequence(official_canonical, 1)
        + ".pod_player1_debug="
        + player_debug_sequence(pod_canonical, 1)
        + ".official_continuations="
        + std::to_string(official.continuations.count)
        + ".pod_continuations=" + std::to_string(pod.continuations.count)
        + ".official_continuation_frames="
        + continuation_sequence(official_canonical)
        + ".pod_continuation_frames=" + continuation_sequence(pod_canonical)
        + ".official_effect=" + effect_sequence(official_canonical)
        + ".pod_effect=" + effect_sequence(pod_canonical)
        + ".pod_raw_effect_interpreter="
        + std::to_string(pod.effect_interpreter.active) + ":"
        + std::to_string(pod.effect_interpreter.awaiting_selection) + ":"
        + std::to_string(pod.effect_interpreter.resume_kind) + ":"
        + std::to_string(pod.effect_interpreter.effect_index) + ":"
        + std::to_string(pod.effect_interpreter.effect_count)
        + ".pod_raw_flow_flags=" + std::to_string(pod.flow_flags)
        + ".official_turn_used_skills="
        + integer_sequence(official_canonical.turn_used_skills)
        + ".pod_turn_used_skills="
        + integer_sequence(pod_canonical.turn_used_skills)
        + ".official_continual_sources="
        + official_continual_source_sequence(official_source)
        + ".official_delay_triggers="
        + trigger_sequence(official_canonical, official_canonical.delay_triggers)
        + ".pod_delay_triggers="
        + trigger_sequence(pod_canonical, pod_canonical.delay_triggers)
        + ".official_temporary_triggers="
        + trigger_sequence(
            official_canonical, official_canonical.temporary_triggers)
        + ".pod_temporary_triggers="
        + trigger_sequence(pod_canonical, pod_canonical.temporary_triggers)
        + ".official_triggers="
        + trigger_sequence(official_canonical, official_canonical.triggers)
        + ".pod_triggers="
        + trigger_sequence(pod_canonical, pod_canonical.triggers));
}

BattleReplayStats run_seed(
    const std::vector<std::uint16_t>& deck0,
    const std::vector<std::uint16_t>& deck1,
    const OfficialRulePackView& rules,
    std::uint64_t seed,
    std::uint32_t decision_limit,
    BattleReplayPolicy policy,
    std::ostream* trace_output = nullptr) {
    ApiData official;
    initialize_official_battle(
        &official, deck0, deck1, seed, trace_output != nullptr);
    const std::uint64_t setup_rng_draws = count_rng_draws(seed, official.game.rng);
    OfficialStatePod pod{};
    OfficialBridgeResult bridge = bridge_official_state(
        official.state,
        OfficialStateBridgeContext{seed, setup_rng_draws, true},
        &pod);
    if (!bridge) {
        throw std::runtime_error(
            "initial bridge failed: "
            + std::to_string(static_cast<std::int32_t>(bridge.error))
            + ".detail=" + std::to_string(bridge.detail));
    }

    BattleReplayStats stats{};
    BattleReplayDriverState driver{};
    std::array<int, 2> trace_log_index{};
    while (!official.state.isFinish()) {
        if (stats.decisions >= decision_limit) {
            // A coverage policy can legally cycle (for example by repeatedly
            // using recovery/search effects).  The caller already reports a
            // non-terminal game as unfinished, so preserve the semantic
            // comparison performed through the bound instead of aborting an
            // entire multi-deck matrix.
            break;
        }
        std::string actor_observation;
        if (trace_output != nullptr) {
            const int actor = official.state.selectPlayer;
            if (actor < 0 || actor >= static_cast<int>(trace_log_index.size())) {
                throw std::runtime_error("semantic trace has invalid actor");
            }
            const int log_start = trace_log_index[actor];
            trace_log_index[actor] = static_cast<int>(official.state.logs.size());
            official.jsonBuilder.clear();
            ToJsonApi(
                official.state,
                official.jsonBuilder,
                log_start);
            const auto& json = official.jsonBuilder.buf;
            actor_observation.assign(
                reinterpret_cast<const char*>(json.data()), json.size());
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
                policy == BattleReplayPolicy::kCoverageRandomLegal);
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
            && pod.select_type == static_cast<std::uint8_t>(
                OfficialSelectTypeId::kYesNo)) {
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
            && pod.select_type == static_cast<std::uint8_t>(
                OfficialSelectTypeId::kCard)
            && pod.select_context == kOfficialSelectContextToHand) {
            if (official.state.selectMin <= 0
                || official.state.selectMin > std::ssize(official.state.options)
                || official.state.selectMin > static_cast<int>(kOfficialOptionCapacity)) {
                throw std::runtime_error(
                    "unsupported prize selection count seed=" + std::to_string(seed)
                    + ".decision=" + std::to_string(stats.decisions)
                    + ".count=" + std::to_string(official.state.selectMin));
            }
            official_index = 0;
            for (int index = 0; index < official.state.selectMin; ++index) {
                official_selected.push_back(index);
            }
            prize_selection = true;
            expected_type = OfficialSelectOptionTypeId::kCard;
        } else if (official.state.selectType == SelectType::Card
            && official.state.selectContext == SelectContext::ToActive
            && pod.select_type == static_cast<std::uint8_t>(
                OfficialSelectTypeId::kCard)
            && pod.select_context == kOfficialSelectContextToActive) {
            official_index = 0;
            active_replacement = true;
            expected_type = OfficialSelectOptionTypeId::kCard;
        } else if (official.state.selectType == SelectType::Skill
            && official.state.selectContext == SelectContext::SkillOrder
            && pod.select_type == static_cast<std::uint8_t>(
                OfficialSelectTypeId::kSkill)
            && pod.select_context == kOfficialSelectContextSkillOrder) {
            if (official.state.selectMin <= 0
                || official.state.selectMin > std::ssize(official.state.options)
                || official.state.selectMin > static_cast<int>(
                    kOfficialOptionCapacity)) {
                throw std::runtime_error(
                    "unsupported trigger order count seed=" + std::to_string(seed)
                    + ".decision=" + std::to_string(stats.decisions)
                    + ".count=" + std::to_string(official.state.selectMin));
            }
            official_index = 0;
            for (int index = 0; index < official.state.selectMin; ++index) {
                official_selected.push_back(index);
            }
            trigger_order = true;
            expected_type = OfficialSelectOptionTypeId::kSkill;
        } else {
            throw std::runtime_error(
                "unsupported policy decision seed=" + std::to_string(seed)
                + ".decision=" + std::to_string(stats.decisions)
                + ".official_context="
                + std::to_string(static_cast<int>(official.state.selectContext))
                + ".pod_context=" + std::to_string(pod.select_context));
        }
        if (!is_coverage_policy(policy)
            && (official_index < 0
                || official_index >= pod.options.count
                || pod.options.values[official_index].type
                    != static_cast<std::uint8_t>(expected_type))) {
            throw std::runtime_error(
                "policy option mismatch seed=" + std::to_string(seed)
                + ".decision=" + std::to_string(stats.decisions));
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
                throw std::runtime_error(
                    "policy multi-option mismatch seed=" + std::to_string(seed)
                    + ".decision=" + std::to_string(stats.decisions));
            }
        }
        if (is_coverage_policy(policy)) {
            record_coverage_choice(official.state, coverage_choice, &stats);
        }
        if (trace_output != nullptr) {
            *trace_output
                << "{\"seed\":" << seed
                << ",\"decision\":" << stats.decisions
                << ",\"actor_observation\":" << actor_observation
                << ",\"ordered_action\":[";
            for (std::size_t index = 0; index < official_selected.size(); ++index) {
                if (index != 0) *trace_output << ',';
                *trace_output << official_selected[index];
            }
            *trace_output << "]}\n";
        }
        const std::string action = selected_action_sequence(pod, official_selected);
        const int official_error = ApiSelect(
            &official,
            official_selected.empty() ? nullptr : official_selected.data(),
            static_cast<int>(official_selected.size()));
        if (official_error != 0) {
            throw std::runtime_error(
                "official action failed: " + std::to_string(official_error));
        }
        std::array<std::uint16_t, kOfficialOptionCapacity> pod_selected{};
        for (std::size_t index = 0; index < official_selected.size(); ++index) {
            pod_selected[index] = static_cast<std::uint16_t>(
                official_selected[index]);
        }
        const OfficialFlowStatus pod_status = official_apply_pending_action(
            &pod,
            rules,
            pod_selected.data(),
            static_cast<std::uint16_t>(official_selected.size()));
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

        const bool terminal = official.state.isFinish();
        if ((terminal && pod_status != OfficialFlowStatus::kTerminal)
            || (!terminal && pod_status != OfficialFlowStatus::kNeedsAction)) {
            OfficialStatePod status_expected{};
            const OfficialBridgeResult status_bridge = bridge_official_state(
                official.state,
                OfficialStateBridgeContext{seed, pod.rng.draw_count, true},
                &status_expected);
            throw std::runtime_error(
                "status mismatch seed=" + std::to_string(seed)
                + ".decision=" + std::to_string(stats.decisions)
                + ".action=" + action
                + ".terminal=" + std::to_string(terminal)
                + ".pod_status="
                + std::to_string(static_cast<int>(pod_status))
                + ".pod_error=" + std::to_string(pod.error)
                + ".pod_detail=" + std::to_string(pod.error_detail)
                + ".bridge_ok=" + std::to_string(static_cast<bool>(status_bridge))
                + ".official_flow_flags=" + std::to_string(status_expected.flow_flags)
                + ".pod_flow_flags=" + std::to_string(pod.flow_flags)
                + ".official_attack_stage="
                + std::to_string(status_expected.attack_flow_stage)
                + ".pod_attack_stage=" + std::to_string(pod.attack_flow_stage)
                + ".official_trigger="
                + std::to_string(status_expected.trigger_resolver.active) + ":"
                + std::to_string(status_expected.trigger_resolver.awaiting_activation)
                + ".pod_trigger=" + std::to_string(pod.trigger_resolver.active) + ":"
                + std::to_string(pod.trigger_resolver.awaiting_activation)
                + ".official_interpreter="
                + std::to_string(status_expected.effect_interpreter.active) + ":"
                + std::to_string(status_expected.effect_interpreter.awaiting_selection)
                + ".pod_interpreter="
                + std::to_string(pod.effect_interpreter.active) + ":"
                + std::to_string(pod.effect_interpreter.awaiting_selection)
                + ".official_continuations="
                + continuation_sequence(status_expected)
                + ".pod_continuations=" + continuation_sequence(pod)
                + ".official_effect=" + effect_sequence(status_expected)
                + ".pod_effect=" + effect_sequence(pod)
                + ".official_result="
                + std::to_string(status_expected.game_result) + ":"
                + std::to_string(status_expected.finish_reason)
                + ".pod_result=" + std::to_string(pod.game_result) + ":"
                + std::to_string(pod.finish_reason)
                + ".official_zones="
                + std::to_string(status_expected.players[0].prize.count) + ":"
                + std::to_string(status_expected.players[0].active.count) + ":"
                + std::to_string(status_expected.players[0].bench.count) + ":"
                + std::to_string(status_expected.players[1].prize.count) + ":"
                + std::to_string(status_expected.players[1].active.count) + ":"
                + std::to_string(status_expected.players[1].bench.count)
                + ".pod_zones="
                + std::to_string(pod.players[0].prize.count) + ":"
                + std::to_string(pod.players[0].active.count) + ":"
                + std::to_string(pod.players[0].bench.count) + ":"
                + std::to_string(pod.players[1].prize.count) + ":"
                + std::to_string(pod.players[1].active.count) + ":"
                + std::to_string(pod.players[1].bench.count)
                + ".official_attack="
                + std::to_string(status_expected.attacker.index) + ":"
                + std::to_string(status_expected.current_attack_id) + ":"
                + std::to_string(status_expected.source_attack_id)
                + ".pod_attack=" + std::to_string(pod.attacker.index) + ":"
                + std::to_string(pod.current_attack_id) + ":"
                + std::to_string(pod.source_attack_id)
                + ".official_in_play=" + in_play_sequence(status_expected)
                + ".pod_in_play=" + in_play_sequence(pod)
                + ".official_select="
                + std::to_string(status_expected.select_type) + ":"
                + std::to_string(status_expected.select_context) + ":"
                + std::to_string(status_expected.select_player) + ":"
                + std::to_string(status_expected.select_min) + ":"
                + std::to_string(status_expected.select_max)
                + ".pod_select="
                + std::to_string(pod.select_type) + ":"
                + std::to_string(pod.select_context) + ":"
                + std::to_string(pod.select_player) + ":"
                + std::to_string(pod.select_min) + ":"
                + std::to_string(pod.select_max)
                + ".official_options=" + option_sequence(status_expected)
                + ".pod_options=" + option_sequence(pod)
                + ".official_targets="
                + area_ref_list_sequence(status_expected, status_expected.targets)
                + ".pod_targets=" + area_ref_list_sequence(pod, pod.targets)
                + ".pod_pre_targets=" + area_ref_list_sequence(pod, pod.pre_targets)
                + ".pod_triggers=" + trigger_sequence(pod, pod.triggers)
                + ".pod_temp_triggers="
                + trigger_sequence(pod, pod.temporary_triggers));
        }

        OfficialStatePod expected{};
        bridge = bridge_official_state(
            official.state,
            OfficialStateBridgeContext{seed, pod.rng.draw_count, true},
            &expected);
        if (!bridge) {
            throw std::runtime_error(
                "decision bridge failed: "
                + std::to_string(static_cast<std::int32_t>(bridge.error))
                + ".detail=" + std::to_string(bridge.detail));
        }
        require_state_equal(
            seed,
            stats.decisions,
            action,
            official.state,
            expected,
            pod);
    }
    stats.terminal_result = pod.game_result;
    return stats;
}

}  // namespace

#ifndef PTCG_OFFICIAL_BATTLE_END_TURN_LIBRARY
int main(int argc, char** argv) {
    try {
        std::string rules_path;
        std::string deck0_path;
        std::string deck1_path;
        std::string trace_jsonl_path;
        std::uint64_t seed_start = 1;
        std::uint64_t seed_count = 1;
        std::uint32_t decision_limit = 256;
        BattleReplayPolicy policy = BattleReplayPolicy::kEnd;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            if (argument == "--rules") rules_path = argv[++index];
            else if (argument == "--deck0") deck0_path = argv[++index];
            else if (argument == "--deck1") deck1_path = argv[++index];
            else if (argument == "--trace-jsonl") trace_jsonl_path = argv[++index];
            else if (argument == "--seed-start") {
                seed_start = std::stoull(argv[++index]);
            } else if (argument == "--seed-count") {
                seed_count = std::stoull(argv[++index]);
            } else if (argument == "--decision-limit") {
                decision_limit = static_cast<std::uint32_t>(
                    std::stoul(argv[++index]));
            } else if (argument == "--policy") {
                policy = parse_battle_replay_policy(argv[++index]);
            } else {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }
        if (rules_path.empty() || deck0_path.empty() || deck1_path.empty()
            || seed_count == 0 || decision_limit == 0) {
            throw std::runtime_error(
                "rules, both decks, positive seed-count and decision-limit are required");
        }

        InitializeAll();
        const std::vector<std::uint8_t> rule_bytes = read_binary(rules_path);
        const OfficialRulePackView rules = make_official_rule_pack_view(
            rule_bytes.data());
        const std::vector<std::uint16_t> deck0 = read_deck(deck0_path);
        const std::vector<std::uint16_t> deck1 = read_deck(deck1_path);
        std::ofstream trace_jsonl;
        if (!trace_jsonl_path.empty()) {
            trace_jsonl.open(trace_jsonl_path, std::ios::binary | std::ios::trunc);
            if (!trace_jsonl) {
                throw std::runtime_error(
                    "cannot open semantic trace: " + trace_jsonl_path);
            }
        }
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
        std::uint32_t min_decisions = decision_limit;
        std::uint32_t max_decisions = 0;
        for (std::uint64_t offset = 0; offset < seed_count; ++offset) {
            const BattleReplayStats stats = run_seed(
                deck0,
                deck1,
                rules,
                seed_start + offset,
                decision_limit,
                policy,
                trace_jsonl.is_open() ? &trace_jsonl : nullptr);
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
            total_zero_cardinality_selections += stats.zero_cardinality_selections;
            total_max_cardinality_selections += stats.max_cardinality_selections;
            add_terminal_result(
                stats.terminal_result,
                &total_player0_wins,
                &total_player1_wins,
                &total_draws,
                &total_unfinished);
            for (std::size_t index = 0; index < total_option_type_actions.size(); ++index) {
                total_option_type_actions[index] += stats.option_type_actions[index];
            }
            for (std::size_t index = 0; index < total_select_type_actions.size(); ++index) {
                total_select_type_actions[index] += stats.select_type_actions[index];
            }
            min_decisions = std::min(min_decisions, stats.decisions);
            max_decisions = std::max(max_decisions, stats.decisions);
        }
        std::cout
            << "{\"passed\":true"
            << ",\"scope\":\"" << battle_replay_scope(policy) << "\""
            << ",\"seed_start\":" << seed_start
            << ",\"seed_count\":" << seed_count
            << ",\"decisions_compared\":" << total_decisions
            << ",\"min_decisions_per_battle\":" << min_decisions
            << ",\"max_decisions_per_battle\":" << max_decisions
            << ",\"basic_play_actions\":" << total_basic_plays
            << ",\"basic_energy_attach_actions\":"
            << total_basic_energy_attaches
            << ",\"evolve_actions\":" << total_evolves
            << ",\"optional_decline_actions\":" << total_optional_declines
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
            << ",\"player0_wins\":" << total_player0_wins
            << ",\"player1_wins\":" << total_player1_wins
            << ",\"draws\":" << total_draws
            << ",\"unfinished_battles\":" << total_unfinished
            << ",\"outcome_mismatches\":0"
            << ",\"semantic_trace_records\":"
            << (trace_jsonl.is_open() ? total_decisions : 0)
            << ",\"state_abi\":" << kOfficialStateAbiVersion
            << ",\"state_bytes\":" << sizeof(OfficialStatePod)
            << ",\"canonical_byte_mismatches\":0}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
#endif
