#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ptcg_cuda/official_rng.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_HD __host__ __device__
#else
#define PTCG_OFFICIAL_HD
#endif

namespace ptcg::cuda_engine {

constexpr std::uint32_t kOfficialStateAbiVersion = 6;
constexpr std::size_t kOfficialCardCapacity = 128;
constexpr std::size_t kOfficialDeckCapacity = 61;
constexpr std::size_t kOfficialBenchCapacity = 8;
constexpr std::size_t kOfficialOptionCapacity = 128;
constexpr std::size_t kOfficialListCapacity = 128;
constexpr std::size_t kOfficialEffectFrameCapacity = 256;
constexpr std::size_t kOfficialContinuationCapacity = 256;
constexpr std::size_t kOfficialTriggerCapacity = 128;
constexpr std::size_t kOfficialTurnRecordCapacity = 128;
constexpr std::size_t kOfficialEffectRefScratchCapacity = 4096;
#if defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
constexpr std::size_t kOfficialBranchCoverageEffectCapacity = 4096;
constexpr std::size_t kOfficialBranchCoverageWordCount =
    kOfficialBranchCoverageEffectCapacity / 64;
#endif

enum class OfficialPodError : std::int32_t {
    kNone = 0,
    kInvalidPlayer = 1,
    kInvalidCardRef = 2,
    kInvalidArea = 3,
    kInvalidAreaIndex = 4,
    kZoneOverflow = 5,
    kOptionOverflow = 6,
    kSelectionOverflow = 7,
    kEffectStackOverflow = 8,
    kContinuationStackOverflow = 9,
    kTriggerStackOverflow = 10,
    kTurnRecordOverflow = 11,
    kEffectScratchOverflow = 12,
    kRulePackBounds = 13,
    kUnsupportedEffect = 14,
    kInterpreterBudget = 15,
    kInvalidAction = 16,
    kDeckOut = 17,
    kUnsupportedTarget = 18,
    kUnsupportedCondition = 19,
    kUnsupportedContinuation = 20,
    kKnownDivergence6601207 = 6601207,
};

enum class OfficialArea : std::uint8_t {
    kAll = 0,
    kDeck = 1,
    kHand = 2,
    kTrash = 3,
    kActive = 4,
    kBench = 5,
    kPrize = 6,
    kStadium = 7,
    kEnergy = 8,
    kTool = 9,
    kPreEvolution = 10,
    kPlayer = 11,
    kLooking = 12,
    kPlaying = 13,
    kDeckBottom = 14,
    kMe = 15,
    kEffected = 16,
    kEffectedPreTarget = 17,
    kSelectedList = 18,
    kTriggerSubject = 19,
    kTriggerObject = 20,
    kAttach = 21,
    kTurnPlay = 22,
    kAttackPreMyTurn = 23,
    kTemporary = 24,
};

enum class OfficialGamePhase : std::uint8_t {
    kSetup = 0,
    kMain = 1,
    kPokemonCheckup = 2,
    kPokemonCheckupEnd = 3,
};

enum class OfficialGameResult : std::uint8_t {
    kNone = 0,
    kPlayer0Win = 1,
    kPlayer1Win = 2,
    kDraw = 3,
};

enum class OfficialFinishReason : std::uint8_t {
    kNone = 0,
    kPrize = 1,
    kDeck = 2,
    kNoActivePokemon = 3,
    kEffect = 4,
    kOther = 9,
};

enum class OfficialBadStatus : std::uint8_t {
    kNone = 0,
    kAsleep = 1,
    kParalyzed = 2,
    kConfused = 3,
};

struct OfficialCardRefPod {
    std::uint16_t index = 0;
};
static_assert(sizeof(OfficialCardRefPod) == 2);

PTCG_OFFICIAL_HD inline bool operator==(
    OfficialCardRefPod left,
    OfficialCardRefPod right) {
    return left.index == right.index;
}

PTCG_OFFICIAL_HD inline bool operator!=(
    OfficialCardRefPod left,
    OfficialCardRefPod right) {
    return !(left == right);
}

struct OfficialAreaRefPod {
    OfficialCardRefPod card;
    std::uint16_t reserved = 0;
    std::int32_t move_counter = 0;
};
static_assert(sizeof(OfficialAreaRefPod) == 8);

template <typename T, std::size_t Capacity>
struct OfficialPodList {
    T values[Capacity];
    std::uint16_t count = 0;
    std::uint16_t reserved = 0;

    PTCG_OFFICIAL_HD bool empty() const { return count == 0; }
    PTCG_OFFICIAL_HD constexpr std::size_t capacity() const { return Capacity; }
};

struct alignas(16) OfficialSelectOptionPod {
    std::uint8_t type = 0;
    std::uint8_t reserved0 = 0;
    std::int16_t params[5]{};
    std::uint16_t resolved_card = 0;
    std::uint16_t option_equiv = 0;
};
static_assert(sizeof(OfficialSelectOptionPod) == 16);

struct alignas(16) OfficialActivateAbilityPod {
    std::int32_t skill_id = 0;
    OfficialAreaRefPod effect_card{};
    std::int8_t use_player = 0;
    std::uint8_t is_effect_stack = 0;
    std::int8_t effect_stack_index = 0;
    std::uint8_t is_special_condition = 0;
};
static_assert(sizeof(OfficialActivateAbilityPod) == 16);

struct alignas(16) OfficialTriggerInfoPod {
    std::int32_t value = 0;
    OfficialAreaRefPod subject{};
    OfficialAreaRefPod object{};
    std::uint8_t type = 0;
    std::int8_t depth = 0;
    std::uint16_t reserved = 0;
    std::uint32_t reserved2 = 0;
    std::uint32_t reserved3 = 0;
};
static_assert(sizeof(OfficialTriggerInfoPod) == 32);

struct alignas(16) OfficialTriggeredAbilityPod {
    OfficialActivateAbilityPod activate{};
    OfficialTriggerInfoPod trigger{};
};
static_assert(sizeof(OfficialTriggeredAbilityPod) == 48);

struct alignas(16) OfficialEffectStatePod {
    OfficialActivateAbilityPod ability{};
    std::int32_t damage_change = 0;
    std::int16_t effect_rate = 0;
    std::int8_t effect_index = 0;
    std::uint8_t on_effect = 0;
    std::int8_t selected_list_index = 0;
    std::int8_t each_list_index = 0;
    std::uint16_t reserved = 0;
};
static_assert(sizeof(OfficialEffectStatePod) == 32);

struct alignas(16) OfficialCardStatePod {
    std::int32_t card_id = 0;
    std::int32_t move_counter = 0;
    std::int32_t attach_move_counter = 0;
    std::int32_t skill_order = 0;
    std::int32_t damage = 0;
    std::uint32_t this_turn[4]{};
    std::uint32_t next_turn[4]{};
    std::uint32_t this_turn_enemy = 0;
    std::uint32_t next_turn_enemy = 0;
    std::int32_t take_attack_damage_this_turn = 0;
    std::int32_t take_attack_damage_pre_turn = 0;
    std::int16_t cannot_use_attack_id_non_active = 0;
    std::int8_t player = 0;
    std::uint8_t area = 0;
    std::uint8_t pre_area = 0;
    std::uint8_t reverse = 0;
    std::uint16_t ability_used_count = 0;
    std::int16_t ability_used[8]{};
    std::uint16_t next_enemy_turn_end_battlefield = 0;
    std::uint16_t reserved0 = 0;
    std::uint32_t next_enemy_turn_end = 0;
    std::uint32_t turn_state[3]{};
    std::uint64_t continual_state[5]{};
    std::uint64_t runtime_flags = 0;
    std::int16_t ko_prize_change_always = 0;
    std::int16_t ko_prize_change = 0;
    std::int16_t hp_change = 0;
    std::int16_t reserved1 = 0;
};
static_assert(sizeof(OfficialCardStatePod) == 176);

struct alignas(16) OfficialPlayerStatePod {
    OfficialPodList<OfficialCardRefPod, 1> active{};
    OfficialPodList<OfficialCardRefPod, kOfficialBenchCapacity> bench{};
    OfficialPodList<OfficialCardRefPod, kOfficialDeckCapacity> prize{};
    OfficialPodList<OfficialCardRefPod, kOfficialDeckCapacity> hand{};
    OfficialPodList<OfficialCardRefPod, kOfficialDeckCapacity> deck{};
    OfficialPodList<OfficialCardRefPod, kOfficialDeckCapacity> trash{};
    OfficialPodList<OfficialCardRefPod, kOfficialDeckCapacity> energy{};
    OfficialPodList<OfficialCardRefPod, kOfficialDeckCapacity> tool{};
    OfficialPodList<OfficialCardRefPod, kOfficialDeckCapacity> pre_evolution{};
    OfficialPodList<OfficialCardRefPod, kOfficialDeckCapacity> temporary{};
    std::uint32_t this_turn = 0;
    std::uint32_t next_turn = 0;
    std::uint32_t active_state = 0;
    std::uint64_t continual_state = 0;
    std::uint64_t turn_state = 0;
    std::int8_t player = 0;
    std::uint8_t ko_prize_once_changed = 0;
    std::uint16_t reserved = 0;
};

struct OfficialTurnHistoryPod {
    OfficialCardRefPod attack_card{};
    std::int16_t attack_id = 0;
    std::int8_t take_prize_count = 0;
    std::uint8_t flags = 0;
};
static_assert(sizeof(OfficialTurnHistoryPod) == 6);

struct OfficialPrizeRequestPod {
    std::int16_t count = 0;
    std::int8_t player = -1;
    std::uint8_t reserved = 0;
};
static_assert(sizeof(OfficialPrizeRequestPod) == 4);

struct alignas(16) OfficialEffectFramePod {
    OfficialEffectStatePod effect_state{};
    OfficialTriggerInfoPod trigger_info{};
    std::uint32_t selected_offset = 0;
    std::uint32_t pre_target_offset = 0;
    std::uint32_t target_offset = 0;
    std::uint16_t selected_count = 0;
    std::uint16_t pre_target_count = 0;
    std::uint16_t target_count = 0;
    std::uint8_t effect_jump = 0;
    std::uint8_t flags = 0;
};
static_assert(sizeof(OfficialEffectFramePod) == 96);

struct alignas(16) OfficialContinuationPod {
    std::int32_t args[3]{};
    std::uint16_t opcode = 0;
    std::uint8_t arg_type = 0;
    std::uint8_t call_count = 0;
    std::uint8_t called_count = 0;
    std::uint8_t flags = 0;
    std::uint16_t reserved = 0;
    std::uint32_t reserved2 = 0;
};
static_assert(sizeof(OfficialContinuationPod) == 32);

struct OfficialEvolveRecordPod {
    OfficialCardRefPod from{};
    OfficialCardRefPod to{};
};
static_assert(sizeof(OfficialEvolveRecordPod) == 4);

struct alignas(16) OfficialEffectInterpreterPod {
    std::uint32_t effect_offset = 0;
    std::uint16_t effect_count = 0;
    std::uint16_t effect_index = 0;
    std::uint16_t repeat_index = 0;
    std::uint16_t repeat_count = 0;
    std::uint8_t first_condition_count = 0;
    std::uint8_t active = 0;
    std::uint8_t awaiting_selection = 0;
    std::uint8_t resume_kind = 0;
    std::int8_t effect_owner = -1;
    std::uint8_t separator_pending = 0;
    std::uint16_t repeat_continuation = 0;
    OfficialAreaRefPod effect_card{};
    std::uint32_t step_budget = 0;
    std::uint32_t damage_counter_any_total = 0;
    std::uint32_t reserved2[3]{};
};
static_assert(sizeof(OfficialEffectInterpreterPod) == 48);

struct alignas(16) OfficialTriggerResolverPod {
    OfficialTriggeredAbilityPod current{};
    std::uint16_t temporary_first = 0;
    std::uint16_t temporary_count = 0;
    std::int8_t depth = 0;
    std::uint8_t active = 0;
    std::uint8_t awaiting_order = 0;
    std::uint8_t awaiting_activation = 0;
    std::uint8_t current_valid = 0;
    std::uint8_t activation_kind = 0;
    std::uint8_t grouped = 0;
    std::uint8_t reserved0 = 0;
    std::uint32_t reserved1 = 0;
};
static_assert(sizeof(OfficialTriggerResolverPod) == 64);

#if defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
struct alignas(16) OfficialBranchCoveragePod {
    std::uint64_t reached[kOfficialBranchCoverageWordCount]{};
    std::uint64_t applied[kOfficialBranchCoverageWordCount]{};
    std::uint64_t condition_true[kOfficialBranchCoverageWordCount]{};
    std::uint64_t condition_false[kOfficialBranchCoverageWordCount]{};
};
#endif

struct alignas(64) OfficialStatePod {
    std::uint32_t abi_version = kOfficialStateAbiVersion;
    std::int32_t error = 0;
    std::int32_t error_detail = 0;
    std::uint32_t interpreter_steps = 0;
    std::uint64_t episode_id = 0;
    OfficialMt19937 rng{};

    std::int32_t turn = 0;
    std::int32_t turn_action_count = 0;
    std::int32_t effect_action_count = 0;
    std::int32_t turn_attack_count = 0;
    std::int32_t current_card_effect_index = 0;
    std::int32_t coin_head_count = 0;
    std::int32_t move_counter = 0;
    std::int32_t current_skill_order = 0;
    std::int16_t pending_prize_count[2]{};
    std::uint16_t pending_active_replacement_mask = 0;
    std::uint8_t continual_refresh_passes = 0;
    std::uint8_t reserved_flow = 0;

    std::uint8_t phase = 0;
    std::uint8_t game_result = 0;
    std::uint8_t finish_reason = 0;
    std::int8_t first_player = -1;
    std::int8_t looking_player = -1;
    std::uint8_t looking_reverse = 0;
    std::uint8_t setup_done_mask = 0;
    std::uint8_t mulligan_mask = 0;
    std::int16_t mulligan_count[2]{};
    std::uint8_t control_flags = 0;
    std::uint8_t flow_flags = 0;
    std::uint8_t turn_state = 0;
    std::uint8_t continual_state = 0;
    std::int8_t last_stadium_player = -1;

    std::uint8_t select_type = 0;
    std::uint8_t select_context = 0;
    std::int8_t select_player = -1;
    std::uint8_t select_deck = 0;
    std::int16_t select_min = 0;
    std::int16_t select_max = 0;
    std::int32_t remain_damage_counter = 0;
    std::int32_t energy_cost = 0;
    std::int32_t remain_energy_cost = 0;
    std::int32_t selected_energy_card_count = 0;
    std::int32_t removed_damage_counter = 0;
    std::int16_t select_counts[kOfficialBenchCapacity + 1]{};
    OfficialCardRefPod selecting_energy_pokemon{};
    OfficialCardRefPod context_card{};

    OfficialEffectStatePod effect_state{};
    OfficialTriggerInfoPod trigger_info{};
    OfficialEffectInterpreterPod effect_interpreter{};
    OfficialTriggerResolverPod trigger_resolver{};
    std::uint8_t effect_jump = 0;
    std::uint8_t attach_active = 0;
    std::uint8_t attack_flow_stage = 0;
    std::uint8_t attack_flow_flags = 0;

    std::int32_t current_attack_id = 0;
    std::int32_t source_attack_id = 0;
    std::int32_t attack_damage_change = 0;
    std::int32_t last_attack_damage = 0;
    OfficialCardRefPod attacker{};
    std::uint8_t post_attack_effect = 0;
    std::uint8_t post_effect_activate = 0;
    std::uint8_t fail_attack = 0;
    std::uint8_t second_attack = 0;
    std::uint8_t turn_flow_stage = 0;
    std::uint8_t refresh_flow_stage = 0;

    OfficialTurnHistoryPod turn_histories[3]{};
    OfficialPodList<OfficialCardRefPod, 1> stadium{};
    OfficialPodList<OfficialCardRefPod, kOfficialListCapacity> looking{};
    OfficialPodList<OfficialCardRefPod, kOfficialListCapacity> selected_list{};
    OfficialPodList<OfficialCardRefPod, kOfficialListCapacity> each_list{};
    OfficialPodList<OfficialCardRefPod, 2> playing{};
    OfficialPodList<OfficialCardRefPod, 9> check_list{};
    OfficialPlayerStatePod players[2]{};
    std::int32_t log_index[2]{};
    OfficialCardStatePod cards[kOfficialCardCapacity]{};

    OfficialPodList<OfficialSelectOptionPod, kOfficialOptionCapacity> options{};
    OfficialPodList<std::uint16_t, kOfficialOptionCapacity> selected{};
    OfficialPodList<OfficialAreaRefPod, kOfficialListCapacity> pre_targets{};
    OfficialPodList<OfficialAreaRefPod, kOfficialListCapacity> targets{};
    OfficialPodList<OfficialAreaRefPod, kOfficialListCapacity> ko_list{};
    OfficialPodList<OfficialPrizeRequestPod, kOfficialListCapacity> prize_requests{};

    OfficialPodList<OfficialTriggeredAbilityPod, kOfficialTriggerCapacity> delay_triggers{};
    OfficialPodList<OfficialTriggeredAbilityPod, kOfficialTriggerCapacity> temporary_triggers{};
    OfficialPodList<OfficialTriggeredAbilityPod, kOfficialTriggerCapacity> triggers{};
    OfficialPodList<std::int32_t, kOfficialTurnRecordCapacity> turn_used_skills{};
    OfficialPodList<OfficialCardRefPod, kOfficialTurnRecordCapacity> turn_play{};
    OfficialPodList<OfficialCardRefPod, kOfficialTurnRecordCapacity> turn_heal{};
    OfficialPodList<OfficialEvolveRecordPod, kOfficialTurnRecordCapacity> turn_evolve{};

    OfficialPodList<OfficialEffectFramePod, kOfficialEffectFrameCapacity> effect_stack{};
    OfficialPodList<OfficialContinuationPod, kOfficialContinuationCapacity> continuations{};
    OfficialPodList<OfficialAreaRefPod, kOfficialEffectRefScratchCapacity> effect_ref_scratch{};
#if defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
    OfficialBranchCoveragePod branch_coverage{};
#endif
};

#if defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
PTCG_OFFICIAL_HD inline void official_mark_branch_coverage(
    std::uint64_t* words,
    std::uint32_t effect_offset) {
    if (effect_offset >= kOfficialBranchCoverageEffectCapacity) return;
    words[effect_offset / 64U] |= 1ULL << (effect_offset % 64U);
}

PTCG_OFFICIAL_HD inline void official_mark_effect_reached(
    OfficialStatePod* state,
    std::uint32_t effect_offset) {
    official_mark_branch_coverage(state->branch_coverage.reached, effect_offset);
}

PTCG_OFFICIAL_HD inline void official_mark_effect_applied(
    OfficialStatePod* state,
    std::uint32_t effect_offset) {
    official_mark_branch_coverage(state->branch_coverage.applied, effect_offset);
}

PTCG_OFFICIAL_HD inline void official_mark_effect_condition(
    OfficialStatePod* state,
    std::uint32_t effect_offset,
    bool passed) {
    official_mark_effect_reached(state, effect_offset);
    official_mark_branch_coverage(
        passed
            ? state->branch_coverage.condition_true
            : state->branch_coverage.condition_false,
        effect_offset);
}
#endif

static_assert(std::is_trivially_copyable_v<OfficialStatePod>);
#if !defined(PTCG_OFFICIAL_BRANCH_COVERAGE)
static_assert(sizeof(OfficialStatePod) == 119936, "OfficialStatePod ABI v6 changed");
#endif
static_assert(sizeof(OfficialStatePod) <= 131072, "official state exceeded 128 KiB");

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_HD
