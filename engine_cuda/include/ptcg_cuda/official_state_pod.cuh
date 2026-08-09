#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ptcg_cuda/official_rng.cuh"
#include "ptcg_cuda/official_semantic_history.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_HD __host__ __device__
#else
#define PTCG_OFFICIAL_HD
#endif

namespace ptcg::cuda_engine {

constexpr std::uint32_t kOfficialStateAbiVersion = 7;
constexpr std::size_t kOfficialCardCapacity = 128;
constexpr std::size_t kOfficialDeckCapacity = 61;
constexpr std::size_t kOfficialBenchCapacity = 8;
constexpr std::size_t kOfficialOptionCapacity = 128;
constexpr std::size_t kOfficialListCapacity = 128;
constexpr std::size_t kOfficialEffectFrameCapacity = 256;
constexpr std::size_t kOfficialContinuationCapacity = 256;
constexpr std::size_t kOfficialTriggerCapacity = 128;
// Long legal Item/Ability chains can exceed the former 128-record budget
// within one official turn.  Treating that bounded bookkeeping list as an
// engine error invalidates otherwise legal full-pool rollouts; 512 keeps the
// turn-local condition history lossless while retaining a fixed POD layout.
constexpr std::size_t kOfficialTurnRecordCapacity = 512;
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
    union {
        std::int32_t log_index[2];
        OfficialSemanticHistoryDeviceView* semantic_history;
    };
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

PTCG_OFFICIAL_HD inline bool official_semantic_history_enabled(
    const OfficialStatePod* state) {
    return state != nullptr
        && (state->abi_version & kOfficialSemanticHistoryEnabled) != 0;
}

PTCG_OFFICIAL_HD inline OfficialSemanticHistoryDeviceView*
official_semantic_history_view(OfficialStatePod* state) {
    if (!official_semantic_history_enabled(state)) return nullptr;
    return state->semantic_history;
}

PTCG_OFFICIAL_HD inline void official_semantic_history_bind(
    OfficialStatePod* state,
    OfficialSemanticHistoryDeviceView* view) {
    if (state == nullptr) return;
    state->semantic_history = view;
    state->abi_version = official_state_abi_base(state->abi_version)
        | kOfficialSemanticHistoryEnabled;
}

PTCG_OFFICIAL_HD inline std::uint32_t official_semantic_history_lane(
    const OfficialSemanticHistoryDeviceView* view,
    const OfficialStatePod* state) {
    if (view == nullptr || state == nullptr || view->states_base == nullptr) {
        return 0;
    }
    return static_cast<std::uint32_t>(state - view->states_base);
}

PTCG_OFFICIAL_HD inline std::size_t official_semantic_history_slot_offset(
    const OfficialSemanticHistoryDeviceView* view,
    std::uint32_t lane,
    std::uint32_t slot) {
    return (static_cast<std::size_t>(lane) * view->capacity) + slot;
}

PTCG_OFFICIAL_HD inline std::size_t official_semantic_actor_offset(
    std::uint32_t lane,
    std::uint32_t actor) {
    return static_cast<std::size_t>(lane) * 2U + actor;
}

PTCG_OFFICIAL_HD inline std::size_t official_semantic_serial_offset(
    std::uint32_t lane,
    std::uint32_t actor,
    std::uint32_t serial) {
    return (official_semantic_actor_offset(lane, actor)
        * kOfficialSemanticSerialCapacity) + serial;
}

PTCG_OFFICIAL_HD inline void official_semantic_history_clear_lane(
    OfficialStatePod* state) {
    OfficialSemanticHistoryDeviceView* view = official_semantic_history_view(state);
    if (view == nullptr) return;
    const std::uint32_t lane = official_semantic_history_lane(view, state);
    if (lane >= view->batch_size) return;
    view->total_count[lane] = 0;
    view->write_index[lane] = 0;
    for (std::uint32_t actor = 0; actor < 2; ++actor) {
        const std::size_t actor_offset = official_semantic_actor_offset(
            lane, actor);
        view->deck_membership_known[actor_offset] = 0;
        view->prize_membership_known[actor_offset] = 0;
        view->deck_order_known[actor_offset] = 0;
        view->deck_source_event[actor_offset] = 0;
        view->prize_source_event[actor_offset] = 0;
        view->unknown_opponent_hand[actor_offset] = 0;
        view->possible_hand_lower[actor_offset] = 0;
        view->possible_hand_upper[actor_offset] = 0;
        for (std::uint32_t serial = 0;
             serial < kOfficialSemanticSerialCapacity;
             ++serial) {
            const std::size_t serial_offset = official_semantic_serial_offset(
                lane, actor, serial);
            view->known_opponent_hand[serial_offset] = 0;
            view->possible_opponent_hand[serial_offset] = 0;
            view->remembered_opponent_cards[serial_offset] = 0;
            view->known_self_deck_serial[serial_offset] = 0;
        }
    }
    for (std::uint32_t slot = 0; slot < view->capacity; ++slot) {
        const std::size_t offset = official_semantic_history_slot_offset(
            view,
            lane,
            slot);
        view->log_type[offset] = 0;
        view->param_count[offset] = 0;
        for (std::uint32_t index = 0;
             index < kOfficialSemanticHistoryParamCapacity;
             ++index) {
            view->params[
                offset * kOfficialSemanticHistoryParamCapacity + index] = 0;
        }
    }
}

PTCG_OFFICIAL_HD inline bool official_semantic_move_visible(
    std::int32_t player,
    std::int32_t observer,
    std::int32_t open_type) {
    return open_type == 0
        || (open_type == 1 && player == observer)
        || (open_type == 3 && observer == 0)
        || (open_type == 4 && observer == 1);
}

PTCG_OFFICIAL_HD inline bool official_semantic_move_visible_for_observer(
    std::int32_t player,
    std::int32_t observer,
    std::int32_t from_area,
    std::int32_t to_area,
    std::int32_t open_type) {
    if (player != observer
        && (from_area == static_cast<std::int32_t>(OfficialArea::kPrize)
            || to_area == static_cast<std::int32_t>(OfficialArea::kPrize))) {
        return false;
    }
    return official_semantic_move_visible(player, observer, open_type);
}

PTCG_OFFICIAL_HD inline void official_semantic_history_append_raw(
    OfficialStatePod* state,
    OfficialSemanticLogType type,
    std::uint8_t param_count,
    const std::int32_t* params) {
    OfficialSemanticHistoryDeviceView* view = official_semantic_history_view(state);
    if (view == nullptr || params == nullptr || view->capacity == 0) return;
    const std::uint32_t lane = official_semantic_history_lane(view, state);
    if (lane >= view->batch_size) return;
    const std::uint64_t event_index = view->total_count[lane];
    if (param_count >= 1 && params[0] >= 0 && params[0] <= 1) {
        const std::uint32_t player = static_cast<std::uint32_t>(params[0]);
        const std::size_t player_offset = official_semantic_actor_offset(
            lane, player);
        if (type == OfficialSemanticLogType::kShuffle) {
            view->deck_order_known[player_offset] = 0;
        } else if (type == OfficialSemanticLogType::kDraw && param_count >= 3) {
            if (view->deck_membership_known[player_offset] != 0) {
                view->deck_source_event[player_offset] = event_index;
                const std::int32_t serial = params[2];
                if (serial > 0
                    && serial < static_cast<std::int32_t>(
                        kOfficialSemanticSerialCapacity)) {
                    view->known_self_deck_serial[
                        official_semantic_serial_offset(lane, player, serial)] = 0;
                }
            }
        } else if ((type == OfficialSemanticLogType::kMoveCard
                || type == OfficialSemanticLogType::kMoveCardReverse)
            && param_count >= 5) {
            const std::int32_t from_area = params[3];
            const std::int32_t to_area = params[4];
            const bool visible_to_owner =
                type == OfficialSemanticLogType::kMoveCard
                    ? official_semantic_move_visible(
                        params[0], params[0], param_count >= 6 ? params[5] : 0)
                    : to_area != static_cast<std::int32_t>(OfficialArea::kPrize);
            const bool touches_deck =
                from_area == static_cast<std::int32_t>(OfficialArea::kDeck)
                || to_area == static_cast<std::int32_t>(OfficialArea::kDeck);
            const bool touches_deck_bottom =
                from_area == static_cast<std::int32_t>(OfficialArea::kDeckBottom)
                || to_area == static_cast<std::int32_t>(OfficialArea::kDeckBottom);
            const bool touches_prize =
                from_area == static_cast<std::int32_t>(OfficialArea::kPrize)
                || to_area == static_cast<std::int32_t>(OfficialArea::kPrize);
            if (type == OfficialSemanticLogType::kMoveCardReverse
                || !visible_to_owner || touches_deck_bottom) {
                // The official observation reports DeckBottom as area 14,
                // while CausalKnowledge only updates exact membership for
                // area 1.  The next deck-count check therefore invalidates
                // exact membership.  Mirror that loss of authority instead
                // of recomputing hidden truth from the resident state.
                if (touches_deck || touches_deck_bottom) {
                    view->deck_membership_known[player_offset] = 0;
                    view->deck_order_known[player_offset] = 0;
                }
                if (touches_prize) {
                    view->prize_membership_known[player_offset] = 0;
                }
            } else {
                if (touches_deck
                    && view->deck_membership_known[player_offset] != 0) {
                    view->deck_source_event[player_offset] = event_index;
                    const std::int32_t serial = params[2];
                    if (serial > 0
                        && serial < static_cast<std::int32_t>(
                            kOfficialSemanticSerialCapacity)) {
                        view->known_self_deck_serial[
                            official_semantic_serial_offset(
                                lane, player, serial)] =
                            to_area == static_cast<std::int32_t>(
                                OfficialArea::kDeck)
                                ? 1 : 0;
                    }
                }
                if (to_area == static_cast<std::int32_t>(OfficialArea::kDeck)) {
                    view->deck_order_known[player_offset] = 0;
                }
                if (touches_prize
                    && view->prize_membership_known[player_offset] != 0) {
                    view->prize_source_event[player_offset] = event_index;
                }
            }
        }

        const std::uint32_t observer = 1U - player;
        const std::size_t observer_offset = official_semantic_actor_offset(
            lane, observer);
        const bool move_visible = type == OfficialSemanticLogType::kMoveCard
            ? official_semantic_move_visible_for_observer(
                player, static_cast<std::int32_t>(observer),
                param_count >= 4 ? params[3] : -1,
                param_count >= 5 ? params[4] : -1,
                param_count >= 6 ? params[5] : 0)
            : false;
        // CausalKnowledge remembers every actor-visible log carrying the
        // canonical cardId/serial pair, not only MoveCard rows.  The status
        // rows use the API's isRecover placeholder before cardId/serial.
        std::int32_t memory_serial = 0;
        bool memory_identity = false;
        if ((type == OfficialSemanticLogType::kMoveCard && move_visible
                || type == OfficialSemanticLogType::kPlay
                || type == OfficialSemanticLogType::kAttach
                || type == OfficialSemanticLogType::kEvolve
                || type == OfficialSemanticLogType::kDevolve
                || type == OfficialSemanticLogType::kMoveAttached
                || type == OfficialSemanticLogType::kAttack
                || type == OfficialSemanticLogType::kHpChange)
            && param_count >= 3) {
            memory_serial = params[2];
            memory_identity = true;
        } else if ((type == OfficialSemanticLogType::kPoisoned
                || type == OfficialSemanticLogType::kBurned
                || type == OfficialSemanticLogType::kAsleep
                || type == OfficialSemanticLogType::kParalyzed
                || type == OfficialSemanticLogType::kConfused)
            && param_count >= 4) {
            memory_serial = params[3];
            memory_identity = true;
        }
        if (memory_identity
            && memory_serial > 0
            && memory_serial < static_cast<std::int32_t>(
                kOfficialSemanticSerialCapacity)) {
            view->remembered_opponent_cards[
                official_semantic_serial_offset(
                    lane, observer, static_cast<std::uint32_t>(memory_serial))] = 1;
        }
        if (type == OfficialSemanticLogType::kDraw
            || type == OfficialSemanticLogType::kDrawReverse) {
            ++view->unknown_opponent_hand[observer_offset];
        } else if (type == OfficialSemanticLogType::kMoveCard
            && param_count >= 5 && move_visible) {
            const std::int32_t serial = params[2];
            const std::int32_t from_area = params[3];
            const std::int32_t to_area = params[4];
            if (serial > 0
                && serial < static_cast<std::int32_t>(
                    kOfficialSemanticSerialCapacity)) {
                const std::size_t memory = official_semantic_serial_offset(
                    lane, observer, static_cast<std::uint32_t>(serial));
                if (from_area == static_cast<std::int32_t>(OfficialArea::kHand)) {
                    if (view->known_opponent_hand[memory] != 0) {
                        view->known_opponent_hand[memory] = 0;
                    } else if (view->possible_opponent_hand[memory] != 0) {
                        view->possible_opponent_hand[memory] = 0;
                        if (view->possible_hand_lower[observer_offset] > 0) {
                            --view->possible_hand_lower[observer_offset];
                        }
                        if (view->possible_hand_upper[observer_offset] > 0) {
                            --view->possible_hand_upper[observer_offset];
                        }
                    } else if (view->unknown_opponent_hand[observer_offset] > 0) {
                        --view->unknown_opponent_hand[observer_offset];
                    }
                }
                if (to_area == static_cast<std::int32_t>(OfficialArea::kHand)) {
                    view->known_opponent_hand[memory] = 1;
                    view->possible_opponent_hand[memory] = 0;
                }
            }
        } else if ((type == OfficialSemanticLogType::kMoveCard
                || type == OfficialSemanticLogType::kMoveCardReverse)
            && param_count >= 5) {
            const std::int32_t from_area = params[3];
            const std::int32_t to_area = params[4];
            if (from_area == static_cast<std::int32_t>(OfficialArea::kHand)) {
                std::uint32_t known_count = 0;
                for (std::uint32_t serial = 1;
                     serial < kOfficialSemanticSerialCapacity;
                     ++serial) {
                    const std::size_t memory = official_semantic_serial_offset(
                        lane, observer, serial);
                    if (view->known_opponent_hand[memory] == 0) continue;
                    view->known_opponent_hand[memory] = 0;
                    view->possible_opponent_hand[memory] = 1;
                    ++known_count;
                }
                view->possible_hand_lower[observer_offset] =
                    static_cast<std::uint16_t>(
                        view->possible_hand_lower[observer_offset] + known_count);
                view->possible_hand_upper[observer_offset] =
                    static_cast<std::uint16_t>(
                        view->possible_hand_upper[observer_offset] + known_count);
                if (view->possible_hand_lower[observer_offset] > 0) {
                    --view->possible_hand_lower[observer_offset];
                }
                if (view->unknown_opponent_hand[observer_offset] > 0) {
                    --view->unknown_opponent_hand[observer_offset];
                } else if (view->possible_hand_upper[observer_offset] > 0) {
                    --view->possible_hand_upper[observer_offset];
                }
            }
            if (to_area == static_cast<std::int32_t>(OfficialArea::kHand)) {
                ++view->unknown_opponent_hand[observer_offset];
            }
        }
    }
    const std::uint32_t slot = view->write_index[lane] % view->capacity;
    const std::size_t offset = official_semantic_history_slot_offset(
        view,
        lane,
        slot);
    view->log_type[offset] = static_cast<std::uint8_t>(type);
    const std::uint8_t bounded_param_count =
        param_count > kOfficialSemanticHistoryParamCapacity
            ? static_cast<std::uint8_t>(kOfficialSemanticHistoryParamCapacity)
            : param_count;
    view->param_count[offset] = bounded_param_count;
    for (std::uint32_t index = 0;
         index < kOfficialSemanticHistoryParamCapacity;
         ++index) {
        view->params[offset * kOfficialSemanticHistoryParamCapacity + index] =
            index < bounded_param_count ? params[index] : 0;
    }
    view->write_index[lane] = (slot + 1U) % view->capacity;
    ++view->total_count[lane];
}

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
static_assert(sizeof(OfficialStatePod) == 124544, "OfficialStatePod ABI v7 changed");
#endif
static_assert(sizeof(OfficialStatePod) <= 131072, "official state exceeded 128 KiB");

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_HD
