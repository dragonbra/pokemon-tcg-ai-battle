#include "ptcg_cuda/official_runtime.h"

#include "ptcg_cuda/official_flow_dispatch_pod.cuh"
#include "ptcg_cuda/official_seeded_setup_pod.cuh"

#include <cuda_runtime.h>

#include <cstring>

namespace ptcg::cuda_engine {
namespace {

#ifndef PTCG_CUDA_OFFICIAL_THREADS
#define PTCG_CUDA_OFFICIAL_THREADS 128
#endif

static_assert(
    PTCG_CUDA_OFFICIAL_THREADS > 0 && PTCG_CUDA_OFFICIAL_THREADS <= 1024,
    "PTCG_CUDA_OFFICIAL_THREADS must be in [1, 1024]");
// Each lane runs a long, dependent game-state program. Small blocks create
// enough independent warps to hide that latency at rollout batch sizes; the
// CMake cache variable retains a hardware-specific escape hatch.
constexpr std::uint32_t kOfficialThreads = PTCG_CUDA_OFFICIAL_THREADS;
constexpr std::uint32_t kScratchCloneThreads = 256;
constexpr std::uint32_t kMaxOfficialCodecOptions = kOfficialOptionCapacity;

constexpr std::uint32_t official_blocks(std::uint32_t count) {
    return (count + kOfficialThreads - 1) / kOfficialThreads;
}

__global__ void classify_official_states_kernel(
    const OfficialStatePod* states,
    std::uint32_t batch_size,
    std::uint8_t* statuses) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) return;
    statuses[env] = static_cast<std::uint8_t>(official_flow_status(states[env]));
}

template <typename T>
__device__ void copy_lane_span(
    T* destination,
    const T* source,
    std::size_t destination_lane,
    std::size_t source_lane,
    std::size_t span) {
    const std::size_t destination_offset = destination_lane * span;
    const std::size_t source_offset = source_lane * span;
    for (std::size_t index = threadIdx.x; index < span; index += blockDim.x) {
        destination[destination_offset + index] = source[source_offset + index];
    }
}

template <typename IndexT>
__global__ void fork_official_lanes_kernel(
    OfficialDeviceArena source,
    OfficialDeviceArena destination,
    const IndexT* source_lane_indices,
    std::uint32_t lane_count) {
    const std::uint32_t destination_lane = blockIdx.x;
    if (destination_lane >= lane_count) return;
    const std::int64_t source_lane_signed = static_cast<std::int64_t>(
        source_lane_indices[destination_lane]);
    if (source_lane_signed < 0
        || source_lane_signed >= static_cast<std::int64_t>(source.config.batch_size)) {
        if (threadIdx.x == 0) {
            destination.statuses[destination_lane] = static_cast<std::uint8_t>(
                OfficialFlowStatus::kError);
        }
        return;
    }
    const auto source_lane = static_cast<std::size_t>(source_lane_signed);
    const auto destination_index = static_cast<std::size_t>(destination_lane);
    copy_lane_span(
        reinterpret_cast<std::uint8_t*>(destination.states),
        reinterpret_cast<const std::uint8_t*>(source.states),
        destination_index,
        source_lane,
        sizeof(OfficialStatePod));
    copy_lane_span(
        reinterpret_cast<std::uint8_t*>(destination.actions),
        reinterpret_cast<const std::uint8_t*>(source.actions),
        destination_index,
        source_lane,
        sizeof(OfficialActionPod));
    copy_lane_span(
        destination.statuses, source.statuses, destination_index, source_lane, 1U);
    copy_lane_span(
        destination.semantic_history_total_count,
        source.semantic_history_total_count,
        destination_index,
        source_lane,
        1U);
    copy_lane_span(
        destination.semantic_history_write_index,
        source.semantic_history_write_index,
        destination_index,
        source_lane,
        1U);
    copy_lane_span(
        destination.semantic_history_log_type,
        source.semantic_history_log_type,
        destination_index,
        source_lane,
        kOfficialSemanticHistoryCapacity);
    copy_lane_span(
        destination.semantic_history_param_count,
        source.semantic_history_param_count,
        destination_index,
        source_lane,
        kOfficialSemanticHistoryCapacity);
    copy_lane_span(
        destination.semantic_history_params,
        source.semantic_history_params,
        destination_index,
        source_lane,
        kOfficialSemanticHistoryCapacity * kOfficialSemanticHistoryParamCapacity);
    constexpr std::size_t kActorSpan = 2U;
    constexpr std::size_t kSerialSpan =
        kActorSpan * kOfficialSemanticSerialCapacity;
    copy_lane_span(
        destination.semantic_deck_membership_known,
        source.semantic_deck_membership_known,
        destination_index,
        source_lane,
        kActorSpan);
    copy_lane_span(
        destination.semantic_prize_membership_known,
        source.semantic_prize_membership_known,
        destination_index,
        source_lane,
        kActorSpan);
    copy_lane_span(
        destination.semantic_deck_order_known,
        source.semantic_deck_order_known,
        destination_index,
        source_lane,
        kActorSpan);
    copy_lane_span(
        destination.semantic_known_self_deck_serial,
        source.semantic_known_self_deck_serial,
        destination_index,
        source_lane,
        kSerialSpan);
    copy_lane_span(
        destination.semantic_deck_source_event,
        source.semantic_deck_source_event,
        destination_index,
        source_lane,
        kActorSpan);
    copy_lane_span(
        destination.semantic_prize_source_event,
        source.semantic_prize_source_event,
        destination_index,
        source_lane,
        kActorSpan);
    copy_lane_span(
        destination.semantic_known_opponent_hand,
        source.semantic_known_opponent_hand,
        destination_index,
        source_lane,
        kSerialSpan);
    copy_lane_span(
        destination.semantic_possible_opponent_hand,
        source.semantic_possible_opponent_hand,
        destination_index,
        source_lane,
        kSerialSpan);
    copy_lane_span(
        destination.semantic_remembered_opponent_cards,
        source.semantic_remembered_opponent_cards,
        destination_index,
        source_lane,
        kSerialSpan);
    copy_lane_span(
        destination.semantic_unknown_opponent_hand,
        source.semantic_unknown_opponent_hand,
        destination_index,
        source_lane,
        kActorSpan);
    copy_lane_span(
        destination.semantic_possible_hand_lower,
        source.semantic_possible_hand_lower,
        destination_index,
        source_lane,
        kActorSpan);
    copy_lane_span(
        destination.semantic_possible_hand_upper,
        source.semantic_possible_hand_upper,
        destination_index,
        source_lane,
        kActorSpan);
    __syncthreads();
    if (threadIdx.x == 0) {
        OfficialStatePod* cloned = &destination.states[destination_lane];
        if (official_semantic_history_enabled(cloned)) {
            official_semantic_history_bind(cloned, destination.semantic_history_view);
        } else {
            cloned->semantic_history = nullptr;
        }
    }
}

template <std::size_t Capacity>
__device__ void shuffle_official_ref_list(
    OfficialPodList<OfficialCardRefPod, Capacity>* list,
    OfficialMt19937* rng) {
    if (list == nullptr || list->count < 2) return;
    for (std::uint32_t upper = list->count; upper > 1; --upper) {
        const std::uint32_t swap_index = official_uniform_below(rng, upper);
        const OfficialCardRefPod temporary = list->values[upper - 1];
        list->values[upper - 1] = list->values[swap_index];
        list->values[swap_index] = temporary;
    }
}

template <std::size_t Capacity>
__device__ void canonical_sort_official_ref_list(
    OfficialPodList<OfficialCardRefPod, Capacity>* list) {
    if (list == nullptr || list->count < 2) return;
    for (std::uint16_t index = 1; index < list->count; ++index) {
        const OfficialCardRefPod value = list->values[index];
        std::uint16_t position = index;
        while (position > 0 && list->values[position - 1].index > value.index) {
            list->values[position] = list->values[position - 1];
            --position;
        }
        list->values[position] = value;
    }
}

__global__ void redeterminize_official_hidden_order_kernel(
    OfficialStatePod* states,
    std::uint32_t batch_size,
    const std::int64_t* hidden_order_seeds,
    const std::int64_t* future_rng_seeds) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) return;
    OfficialStatePod* state = &states[env];
    OfficialMt19937 shuffle_rng{};
    official_seed_mt19937(
        &shuffle_rng,
        static_cast<std::uint64_t>(hidden_order_seeds[env]));
    for (std::int32_t player = 0; player < 2; ++player) {
        shuffle_official_ref_list(&state->players[player].deck, &shuffle_rng);
        shuffle_official_ref_list(&state->players[player].prize, &shuffle_rng);
    }
    official_seed_mt19937(
        &state->rng,
        static_cast<std::uint64_t>(future_rng_seeds[env]));
}

__device__ bool official_public_belief_hidden_area(
    const OfficialCardStatePod& card,
    std::int32_t observer) {
    const auto area = static_cast<OfficialArea>(card.area);
    if (card.player == observer) {
        return area == OfficialArea::kDeck || area == OfficialArea::kPrize;
    }
    if (card.player == (1 ^ observer)) {
        return area == OfficialArea::kHand
            || area == OfficialArea::kDeck
            || area == OfficialArea::kPrize;
    }
    return false;
}

__device__ void official_public_belief_mark_serial(
    bool* visible_identity,
    std::int32_t serial) {
    if (serial > 0
        && serial < static_cast<std::int32_t>(kOfficialCardCapacity)) {
        visible_identity[serial] = true;
    }
}

__device__ void official_public_belief_mark_visible_event_serials(
    OfficialSemanticLogType type,
    const std::int32_t* params,
    std::uint8_t count,
    std::int32_t observer,
    bool* visible_identity) {
    if (type == OfficialSemanticLogType::kDraw && count >= 3
        && params[0] == observer) {
        official_public_belief_mark_serial(visible_identity, params[2]);
    } else if (type == OfficialSemanticLogType::kMoveCard && count >= 5
        && official_semantic_move_visible_for_observer(
            params[0], observer, params[3], params[4], count >= 6 ? params[5] : 0)) {
        official_public_belief_mark_serial(visible_identity, params[2]);
    } else if (type == OfficialSemanticLogType::kMoveCardReverse && count >= 5
        && params[0] == observer
        && params[4] != static_cast<std::int32_t>(OfficialArea::kPrize)) {
        official_public_belief_mark_serial(visible_identity, params[2]);
    } else if ((type == OfficialSemanticLogType::kSwitch
            || type == OfficialSemanticLogType::kChange)
        && count >= 5) {
        official_public_belief_mark_serial(visible_identity, params[2]);
        official_public_belief_mark_serial(visible_identity, params[4]);
    } else if (type == OfficialSemanticLogType::kPlay && count >= 3) {
        official_public_belief_mark_serial(visible_identity, params[2]);
    } else if ((type == OfficialSemanticLogType::kAttach
            || type == OfficialSemanticLogType::kEvolve
            || type == OfficialSemanticLogType::kDevolve)
        && count >= 5) {
        official_public_belief_mark_serial(visible_identity, params[2]);
        official_public_belief_mark_serial(visible_identity, params[4]);
    } else if (type == OfficialSemanticLogType::kMoveAttached && count >= 7) {
        official_public_belief_mark_serial(visible_identity, params[2]);
        official_public_belief_mark_serial(visible_identity, params[4]);
        official_public_belief_mark_serial(visible_identity, params[6]);
    } else if ((type == OfficialSemanticLogType::kAttack
            || type == OfficialSemanticLogType::kHpChange)
        && count >= 3) {
        official_public_belief_mark_serial(visible_identity, params[2]);
    } else if (type >= OfficialSemanticLogType::kPoisoned
        && type <= OfficialSemanticLogType::kConfused
        && count >= 4) {
        official_public_belief_mark_serial(visible_identity, params[3]);
    }
}

__device__ void official_public_belief_rewrite_pair(
    OfficialStatePod* state,
    std::int32_t* params,
    std::uint8_t count,
    std::int32_t id_position,
    std::int32_t serial_position) {
    if (id_position >= count || serial_position >= count) return;
    const std::int32_t serial = params[serial_position];
    if (serial > 0
        && serial < static_cast<std::int32_t>(kOfficialCardCapacity)) {
        params[id_position] = state->cards[serial].card_id;
    }
}

__device__ void official_public_belief_rewrite_event_ids(
    OfficialStatePod* state,
    OfficialSemanticLogType type,
    std::int32_t* params,
    std::uint8_t count) {
    if ((type == OfficialSemanticLogType::kDraw
            || type == OfficialSemanticLogType::kDrawReverse
            || type == OfficialSemanticLogType::kMoveCard
            || type == OfficialSemanticLogType::kMoveCardReverse
            || type == OfficialSemanticLogType::kPlay
            || type == OfficialSemanticLogType::kAttack
            || type == OfficialSemanticLogType::kHpChange)
        && count >= 3) {
        official_public_belief_rewrite_pair(state, params, count, 1, 2);
    } else if ((type == OfficialSemanticLogType::kSwitch
            || type == OfficialSemanticLogType::kChange
            || type == OfficialSemanticLogType::kAttach
            || type == OfficialSemanticLogType::kEvolve
            || type == OfficialSemanticLogType::kDevolve)
        && count >= 5) {
        official_public_belief_rewrite_pair(state, params, count, 1, 2);
        official_public_belief_rewrite_pair(state, params, count, 3, 4);
    } else if (type == OfficialSemanticLogType::kMoveAttached && count >= 7) {
        official_public_belief_rewrite_pair(state, params, count, 1, 2);
        official_public_belief_rewrite_pair(state, params, count, 3, 4);
        official_public_belief_rewrite_pair(state, params, count, 5, 6);
    } else if (type >= OfficialSemanticLogType::kPoisoned
        && type <= OfficialSemanticLogType::kConfused
        && count >= 4) {
        official_public_belief_rewrite_pair(state, params, count, 2, 3);
    }
}

__global__ void redeterminize_official_public_belief_clean_kernel(
    OfficialStatePod* states,
    std::uint32_t batch_size,
    const std::int32_t* exact_decks,
    const std::int64_t* observer_seats,
    const std::int64_t* hidden_membership_seeds,
    const std::int64_t* future_rng_seeds,
    std::uint8_t* result_codes) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) return;
    OfficialStatePod* state = &states[env];
    const std::int32_t observer = static_cast<std::int32_t>(observer_seats[env]);
    if (observer < 0 || observer > 1 || state->select_player != observer
        || state->select_context != 1 || state->game_result != 0) {
        result_codes[env] = 2;
        return;
    }
    OfficialSemanticHistoryDeviceView* history = official_semantic_history_view(state);
    if (history == nullptr) {
        result_codes[env] = 2;
        return;
    }
    const std::size_t observer_offset = official_semantic_actor_offset(env, observer);
    if (history->deck_membership_known[observer_offset] != 0
        || history->prize_membership_known[observer_offset] != 0
        || history->deck_order_known[observer_offset] != 0) {
        result_codes[env] = 3;
        return;
    }

    bool visible_identity[kOfficialCardCapacity]{};
    for (std::uint32_t serial = 1; serial < kOfficialCardCapacity; ++serial) {
        const OfficialCardStatePod& card = state->cards[serial];
        if (card.card_id <= 0) continue;
        if (!official_public_belief_hidden_area(card, observer)) {
            visible_identity[serial] = true;
        }
        const std::size_t memory = official_semantic_serial_offset(
            env, static_cast<std::uint32_t>(observer), serial);
        if (history->known_opponent_hand[memory] != 0
            || history->possible_opponent_hand[memory] != 0
            || history->remembered_opponent_cards[memory] != 0) {
            visible_identity[serial] = true;
        }
    }
    official_public_belief_mark_serial(visible_identity, state->context_card.index);
    if (state->effect_state.on_effect != 0) {
        official_public_belief_mark_serial(
            visible_identity, state->effect_state.ability.effect_card.card.index);
    }
    const std::uint64_t total = history->total_count[env];
    const std::uint32_t live = total < history->capacity
        ? static_cast<std::uint32_t>(total)
        : history->capacity;
    const std::uint32_t start = total <= history->capacity
        ? 0U : history->write_index[env] % history->capacity;
    for (std::uint32_t pos = 0; pos < live; ++pos) {
        const std::uint32_t slot = (start + pos) % history->capacity;
        const std::size_t base = static_cast<std::size_t>(env) * history->capacity + slot;
        const auto type = static_cast<OfficialSemanticLogType>(history->log_type[base]);
        const std::uint8_t count = history->param_count[base];
        const std::int32_t* params = history->params
            + base * kOfficialSemanticHistoryParamCapacity;
        official_public_belief_mark_visible_event_serials(
            type, params, count, observer, visible_identity);
    }
    for (std::uint32_t serial = 1; serial < kOfficialCardCapacity; ++serial) {
        if (visible_identity[serial]
            && official_public_belief_hidden_area(state->cards[serial], observer)) {
            result_codes[env] = 3;
            return;
        }
    }

    std::int32_t remaining_ids[2][kOfficialSeededDeckSize]{};
    std::uint16_t remaining_count[2]{};
    bool used[2][kOfficialSeededDeckSize]{};
    const std::int32_t* lane_decks = exact_decks
        + static_cast<std::size_t>(env) * 2U * kOfficialSeededDeckSize;
    for (std::uint32_t serial = 3; serial < 3 + 2 * kOfficialSeededDeckSize; ++serial) {
        const OfficialCardStatePod& card = state->cards[serial];
        if (card.player < 0 || card.player > 1 || card.card_id <= 0) {
            result_codes[env] = 4;
            return;
        }
        if (official_public_belief_hidden_area(card, observer)) continue;
        bool matched = false;
        for (std::uint32_t index = 0; index < kOfficialSeededDeckSize; ++index) {
            const std::size_t deck_index = static_cast<std::size_t>(card.player)
                * kOfficialSeededDeckSize + index;
            if (!used[card.player][index] && lane_decks[deck_index] == card.card_id) {
                used[card.player][index] = true;
                matched = true;
                break;
            }
        }
        if (!matched) {
            result_codes[env] = 4;
            return;
        }
    }
    for (std::int32_t player = 0; player < 2; ++player) {
        for (std::uint32_t index = 0; index < kOfficialSeededDeckSize; ++index) {
            if (!used[player][index]) {
                remaining_ids[player][remaining_count[player]++] = lane_decks[
                    static_cast<std::size_t>(player) * kOfficialSeededDeckSize + index];
            }
        }
    }
    std::uint16_t hidden_count[2]{};
    for (std::uint32_t serial = 3; serial < 3 + 2 * kOfficialSeededDeckSize; ++serial) {
        const OfficialCardStatePod& card = state->cards[serial];
        if (official_public_belief_hidden_area(card, observer)) {
            ++hidden_count[card.player];
        }
    }
    if (hidden_count[0] != remaining_count[0] || hidden_count[1] != remaining_count[1]) {
        result_codes[env] = 4;
        return;
    }

    OfficialMt19937 membership_rng{};
    official_seed_mt19937(
        &membership_rng,
        static_cast<std::uint64_t>(hidden_membership_seeds[env]));
    for (std::int32_t player = 0; player < 2; ++player) {
        for (std::uint32_t upper = remaining_count[player]; upper > 1; --upper) {
            const std::uint32_t swap_index = official_uniform_below(&membership_rng, upper);
            const std::int32_t temporary = remaining_ids[player][upper - 1];
            remaining_ids[player][upper - 1] = remaining_ids[player][swap_index];
            remaining_ids[player][swap_index] = temporary;
        }
        std::uint16_t next = 0;
        for (std::uint32_t serial = 3; serial < 3 + 2 * kOfficialSeededDeckSize; ++serial) {
            OfficialCardStatePod& card = state->cards[serial];
            if (card.player == player
                && official_public_belief_hidden_area(card, observer)) {
                card.card_id = remaining_ids[player][next++];
            }
        }
    }
    for (std::uint32_t pos = 0; pos < live; ++pos) {
        const std::uint32_t slot = (start + pos) % history->capacity;
        const std::size_t base = static_cast<std::size_t>(env) * history->capacity + slot;
        const auto type = static_cast<OfficialSemanticLogType>(history->log_type[base]);
        std::int32_t* params = history->params
            + base * kOfficialSemanticHistoryParamCapacity;
        official_public_belief_rewrite_event_ids(
            state, type, params, history->param_count[base]);
    }
    canonical_sort_official_ref_list(&state->players[observer].deck);
    canonical_sort_official_ref_list(&state->players[observer].prize);
    shuffle_official_ref_list(&state->players[observer].deck, &membership_rng);
    shuffle_official_ref_list(&state->players[observer].prize, &membership_rng);
    const std::int32_t opponent = 1 ^ observer;
    canonical_sort_official_ref_list(&state->players[opponent].hand);
    canonical_sort_official_ref_list(&state->players[opponent].deck);
    canonical_sort_official_ref_list(&state->players[opponent].prize);
    shuffle_official_ref_list(&state->players[opponent].hand, &membership_rng);
    shuffle_official_ref_list(&state->players[opponent].deck, &membership_rng);
    shuffle_official_ref_list(&state->players[opponent].prize, &membership_rng);
    official_seed_mt19937(
        &state->rng,
        static_cast<std::uint64_t>(future_rng_seeds[env]));
    result_codes[env] = 1;
}

template <typename DeckT>
__global__ void reset_official_states_seeded_first_min_kernel(
    OfficialStatePod* states,
    std::uint32_t batch_size,
    const std::uint8_t* rule_pack,
    const DeckT* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    std::uint8_t* statuses,
    OfficialSemanticHistoryDeviceView* semantic_history) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size || (lane_mask != nullptr && lane_mask[env] == 0)) return;

    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
    OfficialStatePod* state = &states[env];
    if (semantic_history != nullptr) {
        state->abi_version = kOfficialStateAbiVersion | kOfficialSemanticHistoryEnabled;
        state->semantic_history = semantic_history;
    } else {
        state->abi_version = kOfficialStateAbiVersion;
        state->log_index[0] = 0;
        state->log_index[1] = 0;
    }
    const std::uint64_t seed = static_cast<std::uint64_t>(seeds[env]);
    const DeckT* lane_deck = decks
        + static_cast<std::size_t>(env)
            * 2U * static_cast<std::uint32_t>(kOfficialSeededDeckSize);
    if (!official_seeded_setup_first_min_state(
            state, rules, lane_deck, seed, seed)) {
        statuses[env] = static_cast<std::uint8_t>(OfficialFlowStatus::kError);
        return;
    }
    // The setup function deliberately returns at the shared Main boundary.
    // Advance through refresh and generate the first real policy decision.
    const OfficialMainResult main = official_main_begin_refresh(state, rules);
    if (main == OfficialMainResult::kError) {
        statuses[env] = static_cast<std::uint8_t>(OfficialFlowStatus::kError);
    } else if (main == OfficialMainResult::kNeedsAction) {
        statuses[env] = static_cast<std::uint8_t>(official_yield_decision(state));
    } else {
        statuses[env] = static_cast<std::uint8_t>(official_boundary_status(state));
    }
}

template <typename DeckT>
__global__ void reset_official_states_seeded_interactive_kernel(
    OfficialStatePod* states,
    std::uint32_t batch_size,
    const std::uint8_t* rule_pack,
    const DeckT* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    std::uint8_t* statuses,
    OfficialSemanticHistoryDeviceView* semantic_history) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size || (lane_mask != nullptr && lane_mask[env] == 0)) return;

    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
    OfficialStatePod* state = &states[env];
    if (semantic_history != nullptr) {
        state->abi_version = kOfficialStateAbiVersion | kOfficialSemanticHistoryEnabled;
        state->semantic_history = semantic_history;
    } else {
        state->abi_version = kOfficialStateAbiVersion;
        state->log_index[0] = 0;
        state->log_index[1] = 0;
    }
    const std::uint64_t seed = static_cast<std::uint64_t>(seeds[env]);
    const DeckT* lane_deck = decks
        + static_cast<std::size_t>(env)
            * 2U * static_cast<std::uint32_t>(kOfficialSeededDeckSize);
    statuses[env] = static_cast<std::uint8_t>(
        official_seeded_setup_interactive_state(
            state, rules, lane_deck, seed, seed));
}

__global__ void advance_official_states_kernel(
    OfficialStatePod* states,
    std::uint32_t batch_size,
    const std::uint8_t* rule_pack,
    std::uint8_t* statuses) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) return;
    OfficialStatePod* state = &states[env];
    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
    const OfficialFlowStatus status = official_advance_idle_state_to_decision(
        state, rules);
    statuses[env] = static_cast<std::uint8_t>(status);
}

__device__ __noinline__ std::uint8_t apply_official_action_device(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialActionPod action) {
    return static_cast<std::uint8_t>(official_apply_pending_action(
        state, rules, action.option_indices, action.count));
}

__global__ void apply_official_actions_kernel(
    OfficialStatePod* states,
    std::uint32_t batch_size,
    const std::uint8_t* rule_pack,
    const OfficialActionPod* actions,
    std::uint8_t* statuses,
    bool ready_only,
    bool setup_only) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) return;

    OfficialStatePod* state = &states[env];
    if (setup_only
        && state->phase != static_cast<std::uint8_t>(
            OfficialGamePhase::kSetup)) {
        return;
    }
    if (ready_only
        && statuses[env]
            != static_cast<std::uint8_t>(OfficialFlowStatus::kNeedsAction)) {
        // Preserve the latest classification. This is intentionally a no-op
        // for terminal/error/idle lanes in a heterogeneous resident batch.
        return;
    }
    if (official_state_abi_base(state->abi_version) != kOfficialStateAbiVersion) {
        official_pod_fail(
            state,
            OfficialPodError::kInvalidAction,
            static_cast<std::int32_t>(state->abi_version));
        statuses[env] = static_cast<std::uint8_t>(OfficialFlowStatus::kError);
        return;
    }
    // Snapshot the device action before the deeply inlined dispatcher consumes it.
    const OfficialActionPod action = actions[env];
    if (action.count > kOfficialOptionCapacity) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, action.count);
        statuses[env] = static_cast<std::uint8_t>(OfficialFlowStatus::kError);
        return;
    }
    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
    statuses[env] = apply_official_action_device(state, rules, action);
}

template <typename IndexT>
__global__ void pack_official_actions_kernel(
    OfficialActionPod* actions,
    const IndexT* option_indices,
    const IndexT* counts,
    std::uint32_t batch_size,
    std::uint32_t option_capacity) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) return;

    OfficialActionPod action{};
    const IndexT raw_count = counts[env];
    const bool valid_count = raw_count >= static_cast<IndexT>(0)
        && raw_count <= static_cast<IndexT>(kOfficialOptionCapacity)
        && raw_count <= static_cast<IndexT>(option_capacity);
    action.count = valid_count
        ? static_cast<std::uint16_t>(raw_count)
        : static_cast<std::uint16_t>(kOfficialOptionCapacity + 1U);

    const std::uint32_t copy_count = valid_count
        ? static_cast<std::uint32_t>(raw_count)
        : 0U;
    for (std::uint32_t index = 0; index < copy_count; ++index) {
        const IndexT raw_index = option_indices[
            static_cast<std::size_t>(env) * option_capacity + index];
        const bool valid_index = raw_index >= static_cast<IndexT>(0)
            && raw_index <= static_cast<IndexT>(0xFFFFU);
        action.option_indices[index] = valid_index
            ? static_cast<std::uint16_t>(raw_index)
            : static_cast<std::uint16_t>(0xFFFFU);
    }
    actions[env] = action;
}

constexpr std::int64_t kPolicyOwnerNone = 0;
constexpr std::int64_t kPolicyOwnerSelf = 1;
constexpr std::int64_t kPolicyOwnerOpponent = 2;

constexpr std::int64_t kPolicyZoneUnknown = 0;
constexpr std::int64_t kPolicyZoneOwnActive = 1;
constexpr std::int64_t kPolicyZoneOwnBench = 2;
constexpr std::int64_t kPolicyZoneOwnHand = 3;
constexpr std::int64_t kPolicyZoneOwnDiscard = 4;
constexpr std::int64_t kPolicyZoneOwnPrize = 5;
constexpr std::int64_t kPolicyZoneOpponentActive = 6;
constexpr std::int64_t kPolicyZoneOpponentBench = 7;
constexpr std::int64_t kPolicyZoneOpponentDiscard = 8;
constexpr std::int64_t kPolicyZoneOpponentPrize = 9;
constexpr std::int64_t kPolicyZoneStadium = 10;
constexpr std::int64_t kPolicyZoneLooking = 11;
constexpr std::int64_t kPolicyZoneSelectDeck = 12;
constexpr std::int64_t kPolicyZoneOwnEnergy = 13;
constexpr std::int64_t kPolicyZoneOpponentEnergy = 14;
constexpr std::int64_t kPolicyZoneOwnTool = 15;
constexpr std::int64_t kPolicyZoneOpponentTool = 16;
constexpr std::int64_t kPolicyZoneOwnEvolution = 17;
constexpr std::int64_t kPolicyZoneOpponentEvolution = 18;

constexpr std::int64_t kPolicyKindCard = 1;
constexpr std::int64_t kPolicyKindPokemon = 2;
constexpr std::int64_t kPolicyKindEnergy = 3;
constexpr std::int64_t kPolicyKindTool = 4;
constexpr std::int64_t kPolicyKindEvolution = 5;
constexpr std::int64_t kPolicyKindStadium = 6;
constexpr std::int64_t kPolicyKindLooking = 7;

__device__ std::int64_t official_codec_positive_enum(
    std::int32_t value,
    std::int32_t cap) {
    if (value < 0) return 0;
    const std::int64_t shifted = static_cast<std::int64_t>(value) + 1;
    return shifted > cap ? cap : shifted;
}

__device__ float official_codec_ratio(
    float numerator,
    float denominator,
    float cap) {
    if (numerator < 0.0F) numerator = 0.0F;
    // The frozen Python codecs divide Python numeric values in FP64 and only
    // cast to float32 when collating. Match that rounding boundary exactly.
    const double value = static_cast<double>(numerator)
        / static_cast<double>(denominator);
    return static_cast<float>(
        value > static_cast<double>(cap) ? static_cast<double>(cap) : value);
}

__device__ std::int64_t official_codec_relative_owner(
    std::int32_t player,
    std::int32_t actor) {
    if (player == actor) return kPolicyOwnerSelf;
    if (player == 0 || player == 1) return kPolicyOwnerOpponent;
    return kPolicyOwnerNone;
}

__device__ std::int64_t official_codec_zone_for(
    std::int64_t owner,
    std::int64_t own_zone,
    std::int64_t opponent_zone) {
    return owner == kPolicyOwnerSelf ? own_zone : opponent_zone;
}

__device__ std::int64_t official_codec_status_bits(
    const OfficialPlayerStatePod& player) {
    std::int64_t bits = 0;
    if (official_pod_bad_status(player) == OfficialBadStatus::kAsleep) {
        bits |= 1LL << 0U;
    }
    if (official_pod_burned(player)) {
        bits |= 1LL << 1U;
    }
    if (official_pod_bad_status(player) == OfficialBadStatus::kConfused) {
        bits |= 1LL << 2U;
    }
    if (official_pod_bad_status(player) == OfficialBadStatus::kParalyzed) {
        bits |= 1LL << 3U;
    }
    if (official_pod_poison_counter(player) > 0) {
        bits |= 1LL << 4U;
    }
    return bits;
}

__device__ const OfficialCardStatePod* official_codec_card(
    const OfficialStatePod* state,
    OfficialCardRefPod ref) {
    if (ref.index == 0 || ref.index >= kOfficialCardCapacity) return nullptr;
    return &state->cards[ref.index];
}

template <std::size_t Capacity>
__device__ std::int32_t official_codec_attached_count(
    const OfficialStatePod* state,
    const OfficialPodList<OfficialCardRefPod, Capacity>& list,
    std::int32_t move_counter) {
    std::int32_t count = 0;
    for (std::uint16_t index = 0; index < list.count; ++index) {
        const OfficialCardStatePod* card = official_codec_card(state, list.values[index]);
        if (card != nullptr && card->attach_move_counter == move_counter) {
            ++count;
        }
    }
    return count;
}

__device__ std::int32_t official_codec_find_location(
    const std::int16_t* players,
    const std::int16_t* areas,
    const std::int16_t* slots,
    std::int32_t count,
    std::int32_t player,
    std::int32_t area,
    std::int32_t slot) {
    for (std::int32_t index = 0; index < count; ++index) {
        if (players[index] == player && areas[index] == area && slots[index] == slot) {
            return index;
        }
    }
    return -1;
}

__device__ bool official_codec_add_entity(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    bool hide_reverse,
    std::int64_t owner,
    std::int64_t zone,
    std::int32_t slot,
    std::int64_t kind,
    std::int32_t parent,
    std::int64_t player_status,
    std::int32_t location_player,
    std::int32_t location_area,
    std::int32_t location_slot,
    bool has_location,
    std::int32_t* entity_count,
    std::int16_t* location_players,
    std::int16_t* location_areas,
    std::int16_t* location_slots,
    std::int64_t* entity_cat,
    float* entity_num,
    std::int64_t* entity_parent,
    std::uint8_t* entity_mask,
    std::int32_t* output_index) {
    if (output_index != nullptr) *output_index = -1;
    const OfficialCardStatePod* card = official_codec_card(state, ref);
    if (card == nullptr || (hide_reverse && card->reverse != 0)) return true;
    if (*entity_count >= static_cast<std::int32_t>(kMaxCodecEntities)) {
        official_pod_fail(
            state,
            OfficialPodError::kEffectScratchOverflow,
            static_cast<std::int32_t>(kMaxCodecEntities));
        return false;
    }

    const OfficialCardRule* master = kind == kPolicyKindPokemon
        ? official_card_rule(rules, static_cast<std::uint32_t>(card->card_id))
        : nullptr;
    const std::int32_t max_hp = master == nullptr
        ? 0
        : ((master->values[kCardHp] + card->hp_change) > 0
            ? (master->values[kCardHp] + card->hp_change)
            : 0);
    const std::int32_t damage = kind == kPolicyKindPokemon && card->damage > 0
        ? card->damage
        : 0;
    const std::int32_t hp = max_hp > damage ? max_hp - damage : 0;
    std::int32_t energy_count = 0;
    std::int32_t tool_count = 0;
    std::int32_t pre_evolution_count = 0;
    if (kind == kPolicyKindPokemon && card->player >= 0 && card->player <= 1) {
        const OfficialPlayerStatePod& player = state->players[card->player];
        energy_count = official_codec_attached_count(
            state, player.energy, card->move_counter);
        tool_count = official_codec_attached_count(
            state, player.tool, card->move_counter);
        pre_evolution_count = official_codec_attached_count(
            state, player.pre_evolution, card->move_counter);
    }

    const std::int32_t entity = (*entity_count)++;
    std::int64_t* cat = entity_cat + entity * kEntityCatWidth;
    float* num = entity_num + entity * kEntityNumWidth;
    cat[0] = card->card_id < 0 ? 0 : (card->card_id > 4096 ? 4096 : card->card_id);
    cat[1] = owner < 0 ? 0 : (owner > 3 ? 3 : owner);
    cat[2] = zone < 0 ? 0 : (zone > 31 ? 31 : zone);
    cat[3] = official_codec_positive_enum(slot, 64);
    cat[4] = kind < 0 ? 0 : (kind > 7 ? 7 : kind);
    cat[5] = player_status < 0 ? 0 : (player_status > 63 ? 63 : player_status);
    num[0] = official_codec_ratio(static_cast<float>(hp), 400.0F, 2.0F);
    num[1] = official_codec_ratio(static_cast<float>(max_hp), 400.0F, 2.0F);
    num[2] = official_codec_ratio(static_cast<float>(damage), 400.0F, 2.0F);
    num[3] = official_codec_ratio(static_cast<float>(energy_count), 10.0F, 2.0F);
    num[4] = official_codec_ratio(static_cast<float>(tool_count), 4.0F, 2.0F);
    num[5] = official_codec_ratio(static_cast<float>(pre_evolution_count), 4.0F, 2.0F);
    num[6] = (card->runtime_flags & kCardAppear) != 0 ? 1.0F : 0.0F;
    num[7] = (card->runtime_flags & kCardEvolved) != 0 ? 1.0F : 0.0F;
    num[8] = card->reverse != 0 ? 1.0F : 0.0F;
    num[9] = official_codec_ratio(static_cast<float>(slot), 64.0F, 1.0F);
    entity_parent[entity] = parent;
    entity_mask[entity] = 1;
    if (has_location) {
        location_players[entity] = static_cast<std::int16_t>(location_player);
        location_areas[entity] = static_cast<std::int16_t>(location_area);
        location_slots[entity] = static_cast<std::int16_t>(location_slot);
    }
    if (output_index != nullptr) *output_index = entity;
    return true;
}

template <std::size_t Capacity>
__device__ bool official_codec_add_attached_children(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialPodList<OfficialCardRefPod, Capacity>& list,
    std::int32_t move_counter,
    std::int64_t owner,
    std::int64_t zone,
    std::int64_t kind,
    std::int32_t parent,
    std::int32_t* entity_count,
    std::int16_t* location_players,
    std::int16_t* location_areas,
    std::int16_t* location_slots,
    std::int64_t* entity_cat,
    float* entity_num,
    std::int64_t* entity_parent,
    std::uint8_t* entity_mask) {
    std::int32_t child_slot = 0;
    for (std::uint16_t index = 0; index < list.count; ++index) {
        const OfficialCardStatePod* card = official_codec_card(state, list.values[index]);
        if (card == nullptr || card->attach_move_counter != move_counter) continue;
        if (!official_codec_add_entity(
                state,
                rules,
                list.values[index],
                false,
                owner,
                zone,
                child_slot++,
                kind,
                parent,
                0,
                0,
                0,
                0,
                false,
                entity_count,
                location_players,
                location_areas,
                location_slots,
                entity_cat,
                entity_num,
                entity_parent,
                entity_mask,
                nullptr)) {
            return false;
        }
    }
    return true;
}

__device__ bool official_codec_add_pokemon(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    std::int32_t player,
    std::int32_t actor,
    std::int32_t area,
    std::int32_t slot,
    std::int64_t zone,
    std::int64_t player_status,
    std::int32_t* entity_count,
    std::int16_t* location_players,
    std::int16_t* location_areas,
    std::int16_t* location_slots,
    std::int64_t* entity_cat,
    float* entity_num,
    std::int64_t* entity_parent,
    std::uint8_t* entity_mask) {
    const OfficialCardStatePod* card = official_codec_card(state, ref);
    if (card == nullptr || card->reverse != 0) return true;
    const std::int64_t owner = official_codec_relative_owner(player, actor);
    std::int32_t parent = -1;
    if (!official_codec_add_entity(
            state,
            rules,
            ref,
            false,
            owner,
            zone,
            slot,
            kPolicyKindPokemon,
            -1,
            player_status,
            player,
            area,
            slot,
            true,
            entity_count,
            location_players,
            location_areas,
            location_slots,
            entity_cat,
            entity_num,
            entity_parent,
            entity_mask,
            &parent)) {
        return false;
    }
    if (player < 0 || player > 1) return true;
    const OfficialPlayerStatePod& ps = state->players[player];
    const std::int64_t energy_zone = owner == kPolicyOwnerSelf
        ? kPolicyZoneOwnEnergy
        : kPolicyZoneOpponentEnergy;
    const std::int64_t tool_zone = owner == kPolicyOwnerSelf
        ? kPolicyZoneOwnTool
        : kPolicyZoneOpponentTool;
    const std::int64_t evolution_zone = owner == kPolicyOwnerSelf
        ? kPolicyZoneOwnEvolution
        : kPolicyZoneOpponentEvolution;
    return official_codec_add_attached_children(
            state, rules, ps.energy, card->move_counter, owner, energy_zone,
            kPolicyKindEnergy, parent, entity_count, location_players,
            location_areas, location_slots, entity_cat, entity_num,
            entity_parent, entity_mask)
        && official_codec_add_attached_children(
            state, rules, ps.tool, card->move_counter, owner, tool_zone,
            kPolicyKindTool, parent, entity_count, location_players,
            location_areas, location_slots, entity_cat, entity_num,
            entity_parent, entity_mask)
        && official_codec_add_attached_children(
            state, rules, ps.pre_evolution, card->move_counter, owner,
            evolution_zone, kPolicyKindEvolution, parent, entity_count,
            location_players, location_areas, location_slots, entity_cat,
            entity_num, entity_parent, entity_mask);
}

template <std::size_t Capacity>
__device__ bool official_codec_add_card_list(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialPodList<OfficialCardRefPod, Capacity>& list,
    bool hide_reverse,
    std::int32_t player,
    std::int32_t actor,
    std::int32_t area,
    std::int64_t zone,
    std::int64_t kind,
    std::int32_t* entity_count,
    std::int16_t* location_players,
    std::int16_t* location_areas,
    std::int16_t* location_slots,
    std::int64_t* entity_cat,
    float* entity_num,
    std::int64_t* entity_parent,
    std::uint8_t* entity_mask) {
    const std::int64_t owner = player < 0
        ? kPolicyOwnerNone
        : official_codec_relative_owner(player, actor);
    for (std::uint16_t slot = 0; slot < list.count; ++slot) {
        if (!official_codec_add_entity(
                state,
                rules,
                list.values[slot],
                hide_reverse,
                owner,
                zone,
                slot,
                kind,
                -1,
                0,
                player,
                area,
                slot,
                true,
                entity_count,
                location_players,
                location_areas,
                location_slots,
                entity_cat,
                entity_num,
                entity_parent,
                entity_mask,
                nullptr)) {
            return false;
        }
    }
    return true;
}

__device__ std::int64_t official_codec_action_family(
    std::int32_t context,
    std::int64_t max_count,
    const std::int64_t* first_option_cat) {
    if (max_count > 1) return 10;
    const std::int32_t option_type = static_cast<std::int32_t>(first_option_cat[0] - 1);
    const std::int32_t area = static_cast<std::int32_t>(first_option_cat[1] - 1);
    const std::int32_t in_play_area = static_cast<std::int32_t>(first_option_cat[2] - 1);
    const std::int32_t attack_id = static_cast<std::int32_t>(first_option_cat[6]);
    if (attack_id > 0 || option_type == 13 || context == 35) return 1;
    if (option_type == 12 || option_type == 14) return 2;
    if (option_type == 9 || context == 37) return 3;
    if (option_type == 7) return 4;
    if (context == 30) return 6;
    if (option_type == 5 || option_type == 6) return 5;
    if (context == 1 || context == 2 || context == 4 || context == 13
        || context == 21 || context == 22 || area == 4 || area == 5
        || in_play_area == 4 || in_play_area == 5) {
        return 7;
    }
    if (area == 1 || context == 5 || context == 7) return 8;
    if (option_type == 0 || context == 8) return 9;
    return 0;
}

__global__ void encode_official_policy_codec_v1_kernel(
    OfficialStatePod* states,
    std::uint32_t batch_size,
    const std::uint8_t* rule_pack,
    PolicyCodecBuffers codec) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) return;
    OfficialStatePod* state = &states[env];
    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);

    auto* global_cat = codec.global_cat + env * kGlobalCatWidth;
    auto* global_num = codec.global_num + env * kGlobalNumWidth;
    auto* entity_cat = codec.entity_cat + env * kMaxCodecEntities * kEntityCatWidth;
    auto* entity_num = codec.entity_num + env * kMaxCodecEntities * kEntityNumWidth;
    auto* entity_parent = codec.entity_parent + env * kMaxCodecEntities;
    auto* entity_mask = codec.entity_mask + env * kMaxCodecEntities;
    auto* option_cat = codec.option_cat + env * kMaxOfficialCodecOptions * kOptionCatWidth;
    auto* option_num = codec.option_num + env * kMaxOfficialCodecOptions * kOptionNumWidth;
    auto* option_equiv = codec.option_equiv + env * kMaxOfficialCodecOptions;
    auto* option_mask = codec.option_mask + env * kMaxOfficialCodecOptions;

    for (std::uint32_t index = 0; index < kGlobalCatWidth; ++index) {
        global_cat[index] = 0;
    }
    for (std::uint32_t index = 0; index < kGlobalNumWidth; ++index) {
        global_num[index] = 0.0F;
    }
    for (std::uint32_t index = 0; index < kMaxCodecEntities; ++index) {
        entity_parent[index] = -1;
        entity_mask[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxCodecEntities * kEntityCatWidth; ++index) {
        entity_cat[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxCodecEntities * kEntityNumWidth; ++index) {
        entity_num[index] = 0.0F;
    }
    for (std::uint32_t index = 0; index < kMaxOfficialCodecOptions; ++index) {
        option_equiv[index] = -1;
        option_mask[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxOfficialCodecOptions * kOptionCatWidth; ++index) {
        option_cat[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxOfficialCodecOptions * kOptionNumWidth; ++index) {
        option_num[index] = 0.0F;
    }

    const std::int32_t actor = state->select_player >= 0 && state->select_player <= 1
        ? state->select_player
        : ((state->turn & 1) == 0 ? 0 : 1);
    const std::int32_t opponent = 1 ^ actor;
    const OfficialPlayerStatePod& own = state->players[actor];
    const OfficialPlayerStatePod& opp = state->players[opponent];
    const std::int32_t select_type = state->select_type > 0 ? state->select_type - 1 : 0;
    const std::int32_t context = state->select_context > 0 ? state->select_context - 1 : 0;
    const std::int64_t min_count = state->select_min > 0 ? state->select_min : 0;
    const std::int64_t max_count = state->select_max > min_count ? state->select_max : min_count;
    const std::int32_t flags =
        ((state->turn_state & kOfficialSupporterPlayedFlag) != 0 ? 1 : 0)
        | ((state->turn_state & kOfficialStadiumPlayedFlag) != 0 ? 2 : 0)
        | ((state->turn_state & kOfficialEnergyPlayedFlag) != 0 ? 4 : 0)
        | ((state->turn_state & (1U << 3U)) != 0 ? 8 : 0)
        | ((state->turn_state & kOfficialTurnEndFlag) != 0 ? 16 : 0);

    global_cat[0] = official_codec_positive_enum(select_type, 127);
    global_cat[1] = official_codec_positive_enum(context, 255);
    global_cat[2] = official_codec_positive_enum(state->first_player, 2);
    global_cat[3] = official_codec_positive_enum(actor, 2);
    global_cat[4] = 0;
    global_cat[5] = min_count > 15 ? 15 : min_count;
    global_cat[6] = max_count > 15 ? 15 : max_count;
    global_cat[7] = flags > 63 ? 63 : flags;

    std::int32_t visible_looking_count = 0;
    const bool looking_cards_visible = state->looking.count > 0
        && (state->looking_player == actor || state->looking_player == 2);
    if (looking_cards_visible
        || (state->looking_player >= 3 && state->looking_player == actor + 3)) {
        visible_looking_count = state->looking.count;
    }

    global_num[0] = official_codec_ratio(static_cast<float>(state->turn), 20.0F, 2.0F);
    global_num[1] = official_codec_ratio(static_cast<float>(state->turn_action_count), 50.0F, 2.0F);
    global_num[2] = 0.0F;
    global_num[3] = 0.0F;
    global_num[4] = official_codec_ratio(static_cast<float>(own.deck.count), 60.0F, 1.0F);
    global_num[5] = official_codec_ratio(static_cast<float>(opp.deck.count), 60.0F, 1.0F);
    global_num[6] = official_codec_ratio(static_cast<float>(own.hand.count), 20.0F, 2.0F);
    global_num[7] = official_codec_ratio(static_cast<float>(opp.hand.count), 20.0F, 2.0F);
    global_num[8] = official_codec_ratio(static_cast<float>(own.prize.count), 6.0F, 1.0F);
    global_num[9] = official_codec_ratio(static_cast<float>(opp.prize.count), 6.0F, 1.0F);
    global_num[10] = official_codec_ratio(static_cast<float>(state->options.count), 128.0F, 2.0F);
    global_num[11] = official_codec_ratio(static_cast<float>(state->remain_damage_counter), 300.0F, 2.0F);
    global_num[12] = official_codec_ratio(static_cast<float>(state->remain_energy_cost), 10.0F, 2.0F);
    global_num[13] = official_codec_ratio(static_cast<float>(visible_looking_count), 60.0F, 1.0F);
    global_num[14] = official_codec_ratio(static_cast<float>(own.bench.count), 5.0F, 1.0F);
    global_num[15] = official_codec_ratio(static_cast<float>(opp.bench.count), 5.0F, 1.0F);

    std::int16_t location_players[kMaxCodecEntities]{};
    std::int16_t location_areas[kMaxCodecEntities]{};
    std::int16_t location_slots[kMaxCodecEntities]{};
    for (std::uint32_t index = 0; index < kMaxCodecEntities; ++index) {
        location_players[index] = -99;
        location_areas[index] = -99;
        location_slots[index] = -99;
    }
    std::int32_t entity_count = 0;
    for (std::int32_t pass = 0; pass < 2; ++pass) {
        const std::int32_t player = pass == 0 ? actor : opponent;
        const OfficialPlayerStatePod& ps = state->players[player];
        const std::int64_t owner = official_codec_relative_owner(player, actor);
        const std::int64_t active_zone = official_codec_zone_for(
            owner, kPolicyZoneOwnActive, kPolicyZoneOpponentActive);
        const std::int64_t bench_zone = official_codec_zone_for(
            owner, kPolicyZoneOwnBench, kPolicyZoneOpponentBench);
        const std::int64_t status_bits = official_codec_status_bits(ps);
        for (std::uint16_t slot = 0; slot < ps.active.count; ++slot) {
            if (!official_codec_add_pokemon(
                    state, rules, ps.active.values[slot], player, actor,
                    static_cast<std::int32_t>(OfficialArea::kActive),
                    slot, active_zone, status_bits, &entity_count,
                    location_players, location_areas, location_slots,
                    entity_cat, entity_num, entity_parent, entity_mask)) {
                return;
            }
        }
        for (std::uint16_t slot = 0; slot < ps.bench.count; ++slot) {
            if (!official_codec_add_pokemon(
                    state, rules, ps.bench.values[slot], player, actor,
                    static_cast<std::int32_t>(OfficialArea::kBench),
                    slot, bench_zone, 0, &entity_count, location_players,
                    location_areas, location_slots, entity_cat, entity_num,
                    entity_parent, entity_mask)) {
                return;
            }
        }
        if (player == actor
            && !official_codec_add_card_list(
                state, rules, ps.hand, false, player, actor,
                static_cast<std::int32_t>(OfficialArea::kHand),
                kPolicyZoneOwnHand, kPolicyKindCard, &entity_count,
                location_players, location_areas, location_slots, entity_cat,
                entity_num, entity_parent, entity_mask)) {
            return;
        }
        const std::int64_t discard_zone = official_codec_zone_for(
            owner, kPolicyZoneOwnDiscard, kPolicyZoneOpponentDiscard);
        const std::int64_t prize_zone = official_codec_zone_for(
            owner, kPolicyZoneOwnPrize, kPolicyZoneOpponentPrize);
        if (!official_codec_add_card_list(
                state, rules, ps.trash, false, player, actor,
                static_cast<std::int32_t>(OfficialArea::kTrash),
                discard_zone, kPolicyKindCard, &entity_count,
                location_players, location_areas, location_slots, entity_cat,
                entity_num, entity_parent, entity_mask)
            || !official_codec_add_card_list(
                state, rules, ps.prize, true, player, actor,
                static_cast<std::int32_t>(OfficialArea::kPrize),
                prize_zone, kPolicyKindCard, &entity_count,
                location_players, location_areas, location_slots, entity_cat,
                entity_num, entity_parent, entity_mask)) {
            return;
        }
    }

    if (!official_codec_add_card_list(
            state, rules, state->stadium, false, -1, actor,
            static_cast<std::int32_t>(OfficialArea::kStadium),
            kPolicyZoneStadium, kPolicyKindStadium, &entity_count,
            location_players, location_areas, location_slots, entity_cat,
            entity_num, entity_parent, entity_mask)) {
        return;
    }
    if (looking_cards_visible
        && !official_codec_add_card_list(
            state, rules, state->looking, false, actor, actor,
            static_cast<std::int32_t>(OfficialArea::kLooking),
            kPolicyZoneLooking, kPolicyKindLooking, &entity_count,
            location_players, location_areas, location_slots, entity_cat,
            entity_num, entity_parent, entity_mask)) {
        return;
    }
    if (state->select_deck != 0
        && !official_codec_add_card_list(
            state, rules, own.deck, false, actor, actor,
            static_cast<std::int32_t>(OfficialArea::kDeck),
            kPolicyZoneSelectDeck, kPolicyKindCard, &entity_count,
            location_players, location_areas, location_slots, entity_cat,
            entity_num, entity_parent, entity_mask)) {
        return;
    }

    if (state->options.count > kMaxOfficialCodecOptions) {
        official_pod_fail(
            state,
            OfficialPodError::kOptionOverflow,
            static_cast<std::int32_t>(state->options.count));
        return;
    }
    for (std::uint16_t option_index = 0; option_index < state->options.count; ++option_index) {
        const OfficialSelectOptionPod option = state->options.values[option_index];
        std::int32_t source_player = actor;
        std::int32_t source_area = -1;
        std::int32_t source_slot = -1;
        std::int32_t target_area = -1;
        std::int32_t target_slot = -1;
        std::int32_t target_player = actor;
        std::int32_t explicit_card = 0;
        std::int32_t attack_id = 0;
        std::int32_t number = -1;
        switch (option.type) {
            case 0:  // Number
                number = option.params[0];
                break;
            case 3:  // Card
            case 4:  // ToolCard
            case 5:  // EnergyCard
            case 6:  // Energy
                source_area = option.params[0];
                source_slot = option.params[1];
                source_player = option.params[2];
                break;
            case 7:  // Play
                source_area = static_cast<std::int32_t>(OfficialArea::kHand);
                source_slot = option.params[0];
                break;
            case 8:  // Attach
            case 9:  // Evolve
                source_area = option.params[0];
                source_slot = option.params[1];
                target_area = option.params[2];
                target_slot = option.params[3];
                break;
            case 10:  // Ability
            case 11:  // Discard
                source_area = option.params[0];
                source_slot = option.params[1];
                break;
            case 13:  // Attack
                attack_id = option.params[0] < 0
                    ? 0
                    : (option.params[0] > 4096 ? 4096 : option.params[0]);
                break;
            case 15:  // Skill
                explicit_card = option.params[0];
                break;
            default:
                break;
        }
        std::int32_t source_entity = official_codec_find_location(
            location_players, location_areas, location_slots, entity_count,
            source_player, source_area, source_slot);
        if (source_area == static_cast<std::int32_t>(OfficialArea::kStadium)) {
            const std::int32_t stadium_entity = official_codec_find_location(
                location_players, location_areas, location_slots, entity_count,
                -1, source_area, source_slot);
            if (stadium_entity >= 0) source_entity = stadium_entity;
        }
        std::int32_t source_card = explicit_card;
        if (source_card <= 0 && source_entity >= 0) {
            source_card = static_cast<std::int32_t>(
                entity_cat[source_entity * kEntityCatWidth]);
        }
        const std::int32_t target_entity = official_codec_find_location(
            location_players, location_areas, location_slots, entity_count,
            target_player, target_area, target_slot);
        const std::int32_t target_card = target_entity >= 0
            ? static_cast<std::int32_t>(entity_cat[target_entity * kEntityCatWidth])
            : 0;
        std::int64_t* row = option_cat + option_index * kOptionCatWidth;
        row[0] = official_codec_positive_enum(option.type, 63);
        row[1] = official_codec_positive_enum(source_area, 31);
        row[2] = official_codec_positive_enum(target_area, 31);
        row[3] = official_codec_relative_owner(source_player, actor);
        row[4] = source_card < 0 ? 0 : (source_card > 4096 ? 4096 : source_card);
        row[5] = target_card < 0 ? 0 : (target_card > 4096 ? 4096 : target_card);
        row[6] = attack_id;
        row[7] = official_codec_positive_enum(number, 127);
        row[8] = source_entity + 1;
        row[9] = target_entity + 1;
        row[10] = official_codec_positive_enum(source_slot, 127);
        row[11] = official_codec_positive_enum(target_slot, 127);

        float* num = option_num + option_index * kOptionNumWidth;
        num[0] = official_codec_ratio(
            static_cast<float>(state->remain_damage_counter), 300.0F, 2.0F);
        num[1] = official_codec_ratio(
            static_cast<float>(state->remain_energy_cost), 10.0F, 2.0F);
        num[2] = official_codec_ratio(static_cast<float>(option_index), 128.0F, 2.0F);
        num[3] = official_codec_ratio(static_cast<float>(state->options.count), 128.0F, 2.0F);

        const std::int64_t source_zone = source_entity >= 0
            ? entity_cat[source_entity * kEntityCatWidth + 2]
            : 0;
        std::int64_t source_position = source_entity >= 0
            ? entity_cat[source_entity * kEntityCatWidth + 3]
            : 0;
        if (source_zone == kPolicyZoneOwnHand
            || source_zone == kPolicyZoneOwnDiscard
            || source_zone == kPolicyZoneOpponentDiscard
            || source_zone == kPolicyZoneLooking
            || source_zone == kPolicyZoneSelectDeck) {
            source_position = 0;
        }
        const std::int64_t target_zone = target_entity >= 0
            ? entity_cat[target_entity * kEntityCatWidth + 2]
            : 0;
        const std::int64_t target_position = target_entity >= 0
            ? entity_cat[target_entity * kEntityCatWidth + 3]
            : 0;
        std::int64_t group = 0;
        for (std::uint16_t previous = 0; previous < option_index; ++previous) {
            const std::int64_t* prior = option_cat + previous * kOptionCatWidth;
            const std::int32_t prior_source_entity = static_cast<std::int32_t>(prior[8] - 1);
            const std::int32_t prior_target_entity = static_cast<std::int32_t>(prior[9] - 1);
            const std::int64_t prior_source_zone = prior_source_entity >= 0
                ? entity_cat[prior_source_entity * kEntityCatWidth + 2]
                : 0;
            std::int64_t prior_source_position = prior_source_entity >= 0
                ? entity_cat[prior_source_entity * kEntityCatWidth + 3]
                : 0;
            if (prior_source_zone == kPolicyZoneOwnHand
                || prior_source_zone == kPolicyZoneOwnDiscard
                || prior_source_zone == kPolicyZoneOpponentDiscard
                || prior_source_zone == kPolicyZoneLooking
                || prior_source_zone == kPolicyZoneSelectDeck) {
                prior_source_position = 0;
            }
            const std::int64_t prior_target_zone = prior_target_entity >= 0
                ? entity_cat[prior_target_entity * kEntityCatWidth + 2]
                : 0;
            const std::int64_t prior_target_position = prior_target_entity >= 0
                ? entity_cat[prior_target_entity * kEntityCatWidth + 3]
                : 0;
            bool same = true;
            for (std::uint32_t field = 0; field < 8; ++field) {
                same = same && prior[field] == row[field];
            }
            same = same
                && prior_source_zone == source_zone
                && prior_source_position == source_position
                && prior_target_zone == target_zone
                && prior_target_position == target_position;
            if (same) {
                group = option_equiv[previous];
                break;
            }
            const std::int64_t next_group = option_equiv[previous] + 1;
            if (next_group > group) group = next_group;
        }
        option_equiv[option_index] = group;
        option_mask[option_index] = 1;
    }

    codec.min_count[env] = min_count;
    codec.max_count[env] = max_count;
    if (state->options.count > 0) {
        (void)official_codec_action_family(
            context,
            max_count,
            option_cat);
    }
}

constexpr std::int64_t kSemanticFieldPresent = 1;
constexpr std::int64_t kSemanticFieldUnknown = 2;
constexpr std::int64_t kSemanticFieldNotApplicable = 3;

constexpr std::int64_t kSemanticZoneSelfActive = 1;
constexpr std::int64_t kSemanticZoneSelfBench = 2;
constexpr std::int64_t kSemanticZoneSelfHand = 3;
constexpr std::int64_t kSemanticZoneSelfDiscard = 4;
constexpr std::int64_t kSemanticZoneOpponentActive = 5;
constexpr std::int64_t kSemanticZoneOpponentBench = 6;
constexpr std::int64_t kSemanticZoneOpponentDiscard = 7;
constexpr std::int64_t kSemanticZoneStadium = 8;
constexpr std::int64_t kSemanticZoneLooking = 9;
constexpr std::int64_t kSemanticZoneSelectDeck = 10;
constexpr std::int64_t kSemanticZoneSelfEnergy = 11;
constexpr std::int64_t kSemanticZoneOpponentEnergy = 12;
constexpr std::int64_t kSemanticZoneSelfTool = 13;
constexpr std::int64_t kSemanticZoneOpponentTool = 14;
constexpr std::int64_t kSemanticZoneSelfEvolution = 15;
constexpr std::int64_t kSemanticZoneOpponentEvolution = 16;
constexpr std::int64_t kSemanticZoneSelfPlaying = 18;
constexpr std::int64_t kSemanticZoneSelfResolvedEnergy = 20;
constexpr std::int64_t kSemanticZoneOpponentResolvedEnergy = 21;

__device__ std::int64_t semantic0031_clamp_i64(
    std::int64_t value,
    std::int64_t low,
    std::int64_t high) {
    return value < low ? low : (value > high ? high : value);
}

__device__ std::int64_t semantic0031_positive_enum(
    std::int32_t value,
    std::int32_t cap) {
    if (value < 0) return 0;
    const std::int64_t shifted = static_cast<std::int64_t>(value) + 1;
    return shifted > cap ? cap : shifted;
}

__device__ std::int64_t semantic0031_zone_for(
    std::int64_t owner,
    std::int64_t self_zone,
    std::int64_t opponent_zone) {
    return owner == kPolicyOwnerSelf ? self_zone : opponent_zone;
}

__device__ std::int32_t semantic0031_find_serial(
    const std::int64_t* card_cat,
    const std::uint8_t* card_mask,
    std::int32_t card_count,
    std::int32_t serial) {
    if (serial <= 0) return 0;
    const std::int64_t encoded = static_cast<std::int64_t>(serial) + 1;
    for (std::int32_t index = 0; index < card_count; ++index) {
        if (card_mask[index] != 0
            && card_cat[index * kSemantic0031CardCatWidth + 1] == encoded) {
            return index + 1;
        }
    }
    return 0;
}

// EnergyCard/ToolCard/Energy options identify the parent Pokemon in params
// 0..2 and the attached-card ordinal in param 3.  The semantic contract binds
// the option relation to that actual attached card rather than its parent.
__device__ std::int32_t semantic0031_find_attached_child(
    const std::int64_t* card_cat,
    const std::int64_t* card_parent,
    const std::uint8_t* card_mask,
    std::int32_t card_count,
    std::int32_t parent_entity,
    std::int64_t kind,
    std::int32_t child_slot) {
    if (parent_entity < 0 || child_slot < 0) return -1;
    const std::int64_t encoded_parent = parent_entity + 1;
    const std::int64_t encoded_slot = child_slot + 1;
    for (std::int32_t index = 0; index < card_count; ++index) {
        if (card_mask[index] != 0
            && card_parent[index] == encoded_parent
            && card_cat[index * kSemantic0031CardCatWidth + 5] == kind
            && card_cat[index * kSemantic0031CardCatWidth + 4] == encoded_slot) {
            return index;
        }
    }
    return -1;
}

__device__ bool semantic0031_add_card_entity(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    bool hide_reverse,
    std::int64_t owner,
    std::int64_t zone,
    std::int32_t slot,
    std::int64_t kind,
    std::int32_t parent,
    std::int64_t status_bits,
    std::int32_t location_player,
    std::int32_t location_area,
    std::int32_t location_slot,
    bool has_location,
    std::int32_t* card_count,
    std::int16_t* location_players,
    std::int16_t* location_areas,
    std::int16_t* location_slots,
    std::int64_t* card_cat,
    float* card_num,
    std::int64_t* card_state,
    std::int64_t* card_parent,
    std::uint8_t* card_mask,
    std::int32_t* output_index,
    std::int64_t identity_knowledge = 1) {
    if (output_index != nullptr) *output_index = -1;
    const OfficialCardStatePod* card = official_codec_card(state, ref);
    if (card == nullptr || (hide_reverse && card->reverse != 0)) return true;
    if (*card_count >= static_cast<std::int32_t>(kMaxSemantic0031CardEntities)) {
        official_pod_fail(
            state,
            OfficialPodError::kEffectScratchOverflow,
            static_cast<std::int32_t>(kMaxSemantic0031CardEntities));
        return false;
    }
    const bool is_pokemon = kind == kPolicyKindPokemon;
    const OfficialCardRule* master = is_pokemon
        ? official_card_rule(rules, static_cast<std::uint32_t>(card->card_id))
        : nullptr;
    const std::int32_t max_hp = master == nullptr
        ? 0
        : ((master->values[kCardHp] + card->hp_change) > 0
            ? (master->values[kCardHp] + card->hp_change)
            : 0);
    const std::int32_t damage = is_pokemon && card->damage > 0 ? card->damage : 0;
    const std::int32_t hp = max_hp > damage ? max_hp - damage : 0;
    std::int32_t energy_count = 0;
    std::int32_t tool_count = 0;
    std::int32_t pre_evolution_count = 0;
    if (is_pokemon && card->player >= 0 && card->player <= 1) {
        const OfficialPlayerStatePod& player = state->players[card->player];
        energy_count = official_codec_attached_count(
            state, player.energy, card->move_counter);
        tool_count = official_codec_attached_count(
            state, player.tool, card->move_counter);
        pre_evolution_count = official_codec_attached_count(
            state, player.pre_evolution, card->move_counter);
    }
    const std::int32_t entity = (*card_count)++;
    std::int64_t* cat = card_cat + entity * kSemantic0031CardCatWidth;
    float* num = card_num + entity * kSemantic0031CardNumWidth;
    std::int64_t* state_row = card_state + entity * kSemantic0031CardNumWidth;
    cat[0] = semantic0031_clamp_i64(card->card_id, 0, 2048);
    cat[1] = semantic0031_clamp_i64(ref.index + 1, 0, 256);
    cat[2] = semantic0031_clamp_i64(owner, 0, 3);
    cat[3] = semantic0031_clamp_i64(zone, 0, 23);
    cat[4] = semantic0031_positive_enum(slot, 256);
    cat[5] = semantic0031_clamp_i64(kind, 0, 10);
    cat[6] = semantic0031_clamp_i64(status_bits, 0, 32);
    cat[7] = 0;
    cat[8] = card->card_id > 0 ? identity_knowledge : 0;
    num[0] = static_cast<float>(hp);
    num[1] = static_cast<float>(max_hp);
    num[2] = static_cast<float>(energy_count);
    num[3] = 0.0F;
    num[4] = static_cast<float>(tool_count);
    num[5] = static_cast<float>(pre_evolution_count);
    num[6] = (card->runtime_flags & kCardAppear) != 0 ? 1.0F : 0.0F;
    for (std::uint32_t field = 0; field < kSemantic0031CardNumWidth; ++field) {
        state_row[field] = is_pokemon
            ? kSemanticFieldPresent
            : kSemanticFieldNotApplicable;
    }
    card_parent[entity] = parent >= 0 ? parent + 1 : 0;
    card_mask[entity] = 1;
    if (has_location) {
        location_players[entity] = static_cast<std::int16_t>(location_player);
        location_areas[entity] = static_cast<std::int16_t>(location_area);
        location_slots[entity] = static_cast<std::int16_t>(location_slot);
    }
    if (output_index != nullptr) *output_index = entity;
    return true;
}

template <std::size_t Capacity>
__device__ bool semantic0031_add_attached_children(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialPodList<OfficialCardRefPod, Capacity>& list,
    std::int32_t move_counter,
    std::int64_t owner,
    std::int64_t zone,
    std::int64_t kind,
    std::int32_t parent,
    std::int32_t* card_count,
    std::int16_t* location_players,
    std::int16_t* location_areas,
    std::int16_t* location_slots,
    std::int64_t* card_cat,
    float* card_num,
    std::int64_t* card_state,
    std::int64_t* card_parent,
    std::uint8_t* card_mask) {
    std::int32_t child_slot = 0;
    for (std::uint16_t index = 0; index < list.count; ++index) {
        const OfficialCardStatePod* card = official_codec_card(state, list.values[index]);
        if (card == nullptr || card->attach_move_counter != move_counter) continue;
        if (!semantic0031_add_card_entity(
                state, rules, list.values[index], false, owner, zone,
                child_slot++, kind, parent, 0, 0, 0, 0, false, card_count,
                location_players, location_areas, location_slots, card_cat,
                card_num, card_state, card_parent, card_mask, nullptr)) {
            return false;
        }
    }
    return true;
}

__device__ std::int32_t semantic0031_energy_type_index(std::int32_t type) {
    if (type == 511) return 10;
    if (type == (32 | 64)) return 11;
    const std::int32_t index = official_energy_type_index(type);
    return index < 0 ? 0 : index;
}

__device__ bool semantic0031_add_resolved_energy_units(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod pokemon_ref,
    const OfficialPlayerStatePod& player,
    std::int32_t move_counter,
    std::int64_t owner,
    std::int32_t parent,
    std::int32_t* card_count,
    std::int64_t* card_cat,
    float* card_num,
    std::int64_t* card_state,
    std::int64_t* card_parent,
    std::uint8_t* card_mask) {
    std::int32_t unit_slot = 0;
    const std::int64_t zone = semantic0031_zone_for(
        owner,
        kSemanticZoneSelfResolvedEnergy,
        kSemanticZoneOpponentResolvedEnergy);
    for (std::uint16_t index = 0; index < player.energy.count; ++index) {
        const OfficialCardRefPod energy_ref = player.energy.values[index];
        const OfficialCardStatePod* energy = official_codec_card(state, energy_ref);
        if (energy == nullptr || energy->attach_move_counter != move_counter) continue;
        const OfficialEnergyInfoPod info = official_energy_info(
            *state, rules, energy_ref, pokemon_ref);
        for (std::int32_t unit = 0; unit < info.count; ++unit) {
            if (*card_count >= static_cast<std::int32_t>(kMaxSemantic0031CardEntities)) {
                official_pod_fail(
                    state,
                    OfficialPodError::kEffectScratchOverflow,
                    static_cast<std::int32_t>(kMaxSemantic0031CardEntities));
                return false;
            }
            const std::int32_t entity = (*card_count)++;
            std::int64_t* cat = card_cat + entity * kSemantic0031CardCatWidth;
            float* num = card_num + entity * kSemantic0031CardNumWidth;
            std::int64_t* state_row =
                card_state + entity * kSemantic0031CardNumWidth;
            cat[0] = 0;
            cat[1] = 0;
            cat[2] = owner;
            cat[3] = zone;
            cat[4] = semantic0031_positive_enum(unit_slot, 256);
            cat[5] = 9;
            cat[6] = 0;
            cat[7] = semantic0031_energy_type_index(info.type) + 1;
            cat[8] = 4;
            for (std::uint32_t field = 0; field < kSemantic0031CardNumWidth; ++field) {
                num[field] = 0.0F;
                state_row[field] = kSemanticFieldNotApplicable;
            }
            card_parent[entity] = parent + 1;
            card_mask[entity] = 1;
            ++unit_slot;
        }
    }
    card_num[parent * kSemantic0031CardNumWidth + 3] =
        static_cast<float>(unit_slot);
    return true;
}

__device__ bool semantic0031_add_select_card(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    std::int32_t actor,
    std::int32_t* card_count,
    std::int16_t* location_players,
    std::int16_t* location_areas,
    std::int16_t* location_slots,
    std::int64_t* card_cat,
    float* card_num,
    std::int64_t* card_state,
    std::int64_t* card_parent,
    std::uint8_t* card_mask) {
    if (ref.index == 0
        || semantic0031_find_serial(
            card_cat, card_mask, *card_count, ref.index) != 0) {
        return true;
    }
    const OfficialCardStatePod* card = official_codec_card(state, ref);
    if (card == nullptr || card->card_id <= 0) return true;
    const std::int64_t owner = official_codec_relative_owner(card->player, actor);
    return semantic0031_add_card_entity(
        state, rules, ref, false, owner,
        owner == kPolicyOwnerSelf ? 18 : 19, 0, kPolicyKindLooking, -1, 0,
        card->player, static_cast<std::int32_t>(OfficialArea::kHand), 0, false,
        card_count, location_players, location_areas, location_slots, card_cat,
        card_num, card_state, card_parent, card_mask, nullptr);
}

__device__ void semantic0031_zero_row(
    OfficialSemantic0031CodecBuffers output,
    std::uint32_t row) {
    output.feature_valid[row] = 0;
    std::int64_t* global_cat =
        output.global_cat + row * kSemantic0031GlobalCatWidth;
    float* global_num =
        output.global_num + row * kSemantic0031GlobalNumWidth;
    std::int64_t* global_state =
        output.global_state + row * kSemantic0031GlobalNumWidth;
    std::int64_t* card_cat =
        output.card_cat + row * kMaxSemantic0031CardEntities * kSemantic0031CardCatWidth;
    float* card_num =
        output.card_num + row * kMaxSemantic0031CardEntities * kSemantic0031CardNumWidth;
    std::int64_t* card_state =
        output.card_state + row * kMaxSemantic0031CardEntities * kSemantic0031CardNumWidth;
    std::int64_t* card_parent =
        output.card_parent + row * kMaxSemantic0031CardEntities;
    std::uint8_t* card_mask =
        output.card_mask + row * kMaxSemantic0031CardEntities;
    std::int64_t* resource_cat =
        output.resource_cat + row * kSemantic0031ResourceCapacity * kSemantic0031ResourceCatWidth;
    float* resource_num =
        output.resource_num + row * kSemantic0031ResourceCapacity * kSemantic0031ResourceNumWidth;
    std::int64_t* resource_state =
        output.resource_state + row * kSemantic0031ResourceCapacity * kSemantic0031ResourceNumWidth;
    std::uint8_t* resource_mask =
        output.resource_mask + row * kSemantic0031ResourceCapacity;
    std::int64_t* event_cat =
        output.event_cat + row * kSemantic0031EventCapacity * kSemantic0031EventCatWidth;
    float* event_num =
        output.event_num + row * kSemantic0031EventCapacity * kSemantic0031EventNumWidth;
    std::int64_t* event_state =
        output.event_state + row * kSemantic0031EventCapacity * kSemantic0031EventNumWidth;
    std::uint8_t* event_mask =
        output.event_mask + row * kSemantic0031EventCapacity;
    std::int64_t* event_source =
        output.event_source + row * kSemantic0031EventCapacity;
    std::int64_t* event_target =
        output.event_target + row * kSemantic0031EventCapacity;
    std::int64_t* event_before =
        output.event_before + row * kSemantic0031EventCapacity;
    std::int64_t* event_after =
        output.event_after + row * kSemantic0031EventCapacity;
    std::int64_t* option_cat =
        output.option_cat + row * kMaxOfficialCodecOptions * kSemantic0031OptionCatWidth;
    float* option_num =
        output.option_num + row * kMaxOfficialCodecOptions * kSemantic0031OptionNumWidth;
    std::int64_t* option_state =
        output.option_state + row * kMaxOfficialCodecOptions * kSemantic0031OptionNumWidth;
    std::uint8_t* option_mask =
        output.option_mask + row * kMaxOfficialCodecOptions;
    std::int64_t* option_source =
        output.option_source + row * kMaxOfficialCodecOptions;
    std::int64_t* option_target =
        output.option_target + row * kMaxOfficialCodecOptions;
    std::int64_t* option_context =
        output.option_context + row * kMaxOfficialCodecOptions;
    std::int64_t* option_effect_card =
        output.option_effect_card + row * kMaxOfficialCodecOptions;
    std::int64_t* option_skill_id =
        output.option_skill_id + row * kSemantic0031OptionSkillCapacity;
    std::int64_t* option_skill_role =
        output.option_skill_role + row * kSemantic0031OptionSkillCapacity;
    std::int64_t* option_skill_parent =
        output.option_skill_parent + row * kSemantic0031OptionSkillCapacity;
    std::uint8_t* option_skill_mask =
        output.option_skill_mask + row * kSemantic0031OptionSkillCapacity;
    std::int64_t* option_effect_id =
        output.option_effect_id + row * kSemantic0031OptionEffectCapacity;
    std::int64_t* option_effect_role =
        output.option_effect_role + row * kSemantic0031OptionEffectCapacity;
    std::int64_t* option_effect_parent =
        output.option_effect_parent + row * kSemantic0031OptionEffectCapacity;
    std::uint8_t* option_effect_mask =
        output.option_effect_mask + row * kSemantic0031OptionEffectCapacity;

    for (std::uint32_t index = 0; index < kSemantic0031GlobalCatWidth; ++index) {
        global_cat[index] = 0;
    }
    for (std::uint32_t index = 0; index < kSemantic0031GlobalNumWidth; ++index) {
        global_num[index] = 0.0F;
        global_state[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxSemantic0031CardEntities; ++index) {
        card_parent[index] = 0;
        card_mask[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxSemantic0031CardEntities * kSemantic0031CardCatWidth; ++index) {
        card_cat[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxSemantic0031CardEntities * kSemantic0031CardNumWidth; ++index) {
        card_num[index] = 0.0F;
        card_state[index] = 0;
    }
    for (std::uint32_t index = 0; index < kSemantic0031ResourceCapacity; ++index) {
        resource_mask[index] = 0;
    }
    for (std::uint32_t index = 0; index < kSemantic0031ResourceCapacity * kSemantic0031ResourceCatWidth; ++index) {
        resource_cat[index] = 0;
    }
    for (std::uint32_t index = 0; index < kSemantic0031ResourceCapacity * kSemantic0031ResourceNumWidth; ++index) {
        resource_num[index] = 0.0F;
        resource_state[index] = 0;
    }
    for (std::uint32_t index = 0; index < kSemantic0031EventCapacity; ++index) {
        event_mask[index] = 0;
        event_source[index] = 0;
        event_target[index] = 0;
        event_before[index] = 0;
        event_after[index] = 0;
    }
    for (std::uint32_t index = 0; index < kSemantic0031EventCapacity * kSemantic0031EventCatWidth; ++index) {
        event_cat[index] = 0;
    }
    for (std::uint32_t index = 0; index < kSemantic0031EventCapacity * kSemantic0031EventNumWidth; ++index) {
        event_num[index] = 0.0F;
        event_state[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxOfficialCodecOptions; ++index) {
        option_mask[index] = 0;
        option_source[index] = 0;
        option_target[index] = 0;
        option_context[index] = 0;
        option_effect_card[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxOfficialCodecOptions * kSemantic0031OptionCatWidth; ++index) {
        option_cat[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxOfficialCodecOptions * kSemantic0031OptionNumWidth; ++index) {
        option_num[index] = 0.0F;
        option_state[index] = 0;
    }
    for (std::uint32_t index = 0; index < kSemantic0031OptionSkillCapacity; ++index) {
        option_skill_id[index] = 0;
        option_skill_role[index] = 0;
        option_skill_parent[index] = 0;
        option_skill_mask[index] = 0;
    }
    for (std::uint32_t index = 0; index < kSemantic0031OptionEffectCapacity; ++index) {
        option_effect_id[index] = 0;
        option_effect_role[index] = 0;
        option_effect_parent[index] = 0;
        option_effect_mask[index] = 0;
    }
    output.min_count[row] = 0;
    output.max_count[row] = 0;
    output.targets[row] = -100;
}

__device__ void semantic0031_add_resource(
    std::int64_t* resource_cat,
    float* resource_num,
    std::int64_t* resource_state,
    std::uint8_t* resource_mask,
    std::int32_t* resource_count,
    std::int32_t card_id) {
    if (card_id <= 0 || card_id > 2048) return;
    for (std::int32_t index = 0; index < *resource_count; ++index) {
        if (resource_cat[index * kSemantic0031ResourceCatWidth] == card_id) {
            resource_num[index * kSemantic0031ResourceNumWidth] += 1.0F;
            return;
        }
    }
    if (*resource_count >= static_cast<std::int32_t>(kSemantic0031ResourceCapacity)) return;
    std::int32_t insert = (*resource_count)++;
    while (insert > 0
        && resource_cat[(insert - 1) * kSemantic0031ResourceCatWidth] > card_id) {
        for (std::uint32_t field = 0; field < kSemantic0031ResourceCatWidth; ++field) {
            resource_cat[insert * kSemantic0031ResourceCatWidth + field] =
                resource_cat[(insert - 1) * kSemantic0031ResourceCatWidth + field];
        }
        for (std::uint32_t field = 0; field < kSemantic0031ResourceNumWidth; ++field) {
            resource_num[insert * kSemantic0031ResourceNumWidth + field] =
                resource_num[(insert - 1) * kSemantic0031ResourceNumWidth + field];
            resource_state[insert * kSemantic0031ResourceNumWidth + field] =
                resource_state[(insert - 1) * kSemantic0031ResourceNumWidth + field];
        }
        resource_mask[insert] = resource_mask[insert - 1];
        --insert;
    }
    std::int64_t* cat = resource_cat + insert * kSemantic0031ResourceCatWidth;
    float* num = resource_num + insert * kSemantic0031ResourceNumWidth;
    std::int64_t* state = resource_state + insert * kSemantic0031ResourceNumWidth;
    cat[0] = card_id;
    cat[1] = 6;
    cat[2] = 6;
    cat[3] = 1;
    num[0] = 1.0F;
    state[0] = kSemanticFieldPresent;
    for (std::uint32_t field = 1; field < kSemantic0031ResourceNumWidth; ++field) {
        num[field] = 0.0F;
        state[field] = kSemanticFieldUnknown;
    }
    resource_mask[insert] = 1;
}

__device__ bool semantic0031_add_pokemon_entity(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref,
    std::int32_t player,
    std::int32_t actor,
    std::int32_t area,
    std::int32_t slot,
    std::int64_t zone,
    std::int64_t status_bits,
    std::int32_t* card_count,
    std::int16_t* location_players,
    std::int16_t* location_areas,
    std::int16_t* location_slots,
    std::int64_t* card_cat,
    float* card_num,
    std::int64_t* card_state,
    std::int64_t* card_parent,
    std::uint8_t* card_mask) {
    const OfficialCardStatePod* card = official_codec_card(state, ref);
    if (card == nullptr || card->reverse != 0) return true;
    const std::int64_t owner = official_codec_relative_owner(player, actor);
    std::int32_t parent = -1;
    if (!semantic0031_add_card_entity(
            state, rules, ref, false, owner, zone, slot, kPolicyKindPokemon,
            -1, status_bits, player, area, slot, true, card_count,
            location_players, location_areas, location_slots, card_cat,
            card_num, card_state, card_parent, card_mask, &parent)) {
        return false;
    }
    if (player < 0 || player > 1) return true;
    const OfficialPlayerStatePod& ps = state->players[player];
    const std::int64_t energy_zone = semantic0031_zone_for(
        owner, kSemanticZoneSelfEnergy, kSemanticZoneOpponentEnergy);
    const std::int64_t tool_zone = semantic0031_zone_for(
        owner, kSemanticZoneSelfTool, kSemanticZoneOpponentTool);
    const std::int64_t evolution_zone = semantic0031_zone_for(
        owner, kSemanticZoneSelfEvolution, kSemanticZoneOpponentEvolution);
    if (!semantic0031_add_attached_children(
            state, rules, ps.energy, card->move_counter, owner, energy_zone,
            kPolicyKindEnergy, parent, card_count, location_players,
            location_areas, location_slots, card_cat, card_num, card_state,
            card_parent, card_mask)) {
        return false;
    }
    if (!semantic0031_add_attached_children(
            state, rules, ps.tool, card->move_counter, owner, tool_zone,
            kPolicyKindTool, parent, card_count, location_players,
            location_areas, location_slots, card_cat, card_num, card_state,
            card_parent, card_mask)) {
        return false;
    }
    if (!semantic0031_add_attached_children(
            state, rules, ps.pre_evolution, card->move_counter, owner,
            evolution_zone, kPolicyKindEvolution, parent, card_count,
            location_players, location_areas, location_slots, card_cat,
            card_num, card_state, card_parent, card_mask)) {
        return false;
    }
    return semantic0031_add_resolved_energy_units(
        state, rules, ref, ps, card->move_counter, owner, parent, card_count,
        card_cat, card_num, card_state, card_parent, card_mask);
}

template <std::size_t Capacity>
__device__ bool semantic0031_add_card_list(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialPodList<OfficialCardRefPod, Capacity>& list,
    bool hide_reverse,
    std::int32_t player,
    std::int32_t actor,
    std::int32_t area,
    std::int64_t zone,
    std::int64_t kind,
    std::int32_t* card_count,
    std::int16_t* location_players,
    std::int16_t* location_areas,
    std::int16_t* location_slots,
    std::int64_t* card_cat,
    float* card_num,
    std::int64_t* card_state,
    std::int64_t* card_parent,
    std::uint8_t* card_mask) {
    const std::int64_t owner = player < 0
        ? kPolicyOwnerNone
        : official_codec_relative_owner(player, actor);
    for (std::uint16_t slot = 0; slot < list.count; ++slot) {
        std::int64_t card_owner = owner;
        if (player < 0) {
            const OfficialCardStatePod* card = official_codec_card(
                state, list.values[slot]);
            card_owner = card == nullptr
                ? kPolicyOwnerNone
                : official_codec_relative_owner(card->player, actor);
        }
        if (!semantic0031_add_card_entity(
                state, rules, list.values[slot], hide_reverse, card_owner, zone,
                slot, kind, -1, 0, player, area, slot, true, card_count,
                location_players, location_areas, location_slots, card_cat,
                card_num, card_state, card_parent, card_mask, nullptr)) {
            return false;
        }
    }
    return true;
}

template <typename LaneT>
__global__ void encode_official_semantic0031_codec_v2_kernel(
    OfficialStatePod* states,
    std::uint32_t batch_size,
    const std::uint8_t* rule_pack,
    const LaneT* lane_indices,
    std::uint32_t lane_count,
    OfficialSemanticHistoryDeviceView* history,
    OfficialSemantic0031CodecBuffers output) {
    const std::uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= lane_count) return;
    semantic0031_zero_row(output, row);

    const LaneT raw_lane = lane_indices[row];
    if (raw_lane < static_cast<LaneT>(0)
        || raw_lane >= static_cast<LaneT>(batch_size)) {
        return;
    }
    const std::uint32_t lane = static_cast<std::uint32_t>(raw_lane);
    OfficialStatePod* state = &states[lane];
    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);

    std::int64_t* global_cat =
        output.global_cat + row * kSemantic0031GlobalCatWidth;
    float* global_num =
        output.global_num + row * kSemantic0031GlobalNumWidth;
    std::int64_t* global_state =
        output.global_state + row * kSemantic0031GlobalNumWidth;
    std::int64_t* card_cat =
        output.card_cat + row * kMaxSemantic0031CardEntities * kSemantic0031CardCatWidth;
    float* card_num =
        output.card_num + row * kMaxSemantic0031CardEntities * kSemantic0031CardNumWidth;
    std::int64_t* card_state =
        output.card_state + row * kMaxSemantic0031CardEntities * kSemantic0031CardNumWidth;
    std::int64_t* card_parent =
        output.card_parent + row * kMaxSemantic0031CardEntities;
    std::uint8_t* card_mask =
        output.card_mask + row * kMaxSemantic0031CardEntities;
    std::int64_t* resource_cat =
        output.resource_cat + row * kSemantic0031ResourceCapacity * kSemantic0031ResourceCatWidth;
    float* resource_num =
        output.resource_num + row * kSemantic0031ResourceCapacity * kSemantic0031ResourceNumWidth;
    std::int64_t* resource_state =
        output.resource_state + row * kSemantic0031ResourceCapacity * kSemantic0031ResourceNumWidth;
    std::uint8_t* resource_mask =
        output.resource_mask + row * kSemantic0031ResourceCapacity;
    std::int64_t* event_cat =
        output.event_cat + row * kSemantic0031EventCapacity * kSemantic0031EventCatWidth;
    float* event_num =
        output.event_num + row * kSemantic0031EventCapacity * kSemantic0031EventNumWidth;
    std::int64_t* event_state =
        output.event_state + row * kSemantic0031EventCapacity * kSemantic0031EventNumWidth;
    std::uint8_t* event_mask =
        output.event_mask + row * kSemantic0031EventCapacity;
    std::int64_t* event_source =
        output.event_source + row * kSemantic0031EventCapacity;
    std::int64_t* event_target =
        output.event_target + row * kSemantic0031EventCapacity;
    std::int64_t* event_before =
        output.event_before + row * kSemantic0031EventCapacity;
    std::int64_t* event_after =
        output.event_after + row * kSemantic0031EventCapacity;
    std::int64_t* option_cat =
        output.option_cat + row * kMaxOfficialCodecOptions * kSemantic0031OptionCatWidth;
    float* option_num =
        output.option_num + row * kMaxOfficialCodecOptions * kSemantic0031OptionNumWidth;
    std::int64_t* option_state =
        output.option_state + row * kMaxOfficialCodecOptions * kSemantic0031OptionNumWidth;
    std::uint8_t* option_mask =
        output.option_mask + row * kMaxOfficialCodecOptions;
    std::int64_t* option_source =
        output.option_source + row * kMaxOfficialCodecOptions;
    std::int64_t* option_target =
        output.option_target + row * kMaxOfficialCodecOptions;
    std::int64_t* option_context =
        output.option_context + row * kMaxOfficialCodecOptions;
    std::int64_t* option_effect_card =
        output.option_effect_card + row * kMaxOfficialCodecOptions;
    std::int64_t* option_skill_id =
        output.option_skill_id + row * kSemantic0031OptionSkillCapacity;
    std::int64_t* option_skill_role =
        output.option_skill_role + row * kSemantic0031OptionSkillCapacity;
    std::int64_t* option_skill_parent =
        output.option_skill_parent + row * kSemantic0031OptionSkillCapacity;
    std::uint8_t* option_skill_mask =
        output.option_skill_mask + row * kSemantic0031OptionSkillCapacity;
    std::int64_t* option_effect_id =
        output.option_effect_id + row * kSemantic0031OptionEffectCapacity;
    std::int64_t* option_effect_role =
        output.option_effect_role + row * kSemantic0031OptionEffectCapacity;
    std::int64_t* option_effect_parent =
        output.option_effect_parent + row * kSemantic0031OptionEffectCapacity;
    std::uint8_t* option_effect_mask =
        output.option_effect_mask + row * kSemantic0031OptionEffectCapacity;

    const std::int32_t actor = state->select_player >= 0 && state->select_player <= 1
        ? state->select_player
        : ((state->turn & 1) == 0 ? 0 : 1);
    const std::int32_t opponent = 1 ^ actor;
    const OfficialPlayerStatePod& own = state->players[actor];
    const OfficialPlayerStatePod& opp = state->players[opponent];
    const std::uint64_t history_total =
        history == nullptr ? 0ULL : history->total_count[lane];
    const std::size_t actor_knowledge = official_semantic_actor_offset(
        lane, static_cast<std::uint32_t>(actor));
    if (history != nullptr && state->select_deck != 0) {
        history->deck_membership_known[actor_knowledge] = 1;
        // A full deck view does not by itself make Prize membership exact.
        // The authoritative CPU CausalKnowledge re-anchors the deck first,
        // then infers Prize only when initial - deck - all currently visible
        // cards is non-negative for every identity and sums to Prize count.
        history->prize_membership_known[actor_knowledge] = 0;
        history->deck_order_known[actor_knowledge] = 1;
        history->deck_source_event[actor_knowledge] = history_total;
        history->prize_source_event[actor_knowledge] = 0;
        for (std::uint32_t serial = 0;
             serial < kOfficialSemanticSerialCapacity;
             ++serial) {
            history->known_self_deck_serial[official_semantic_serial_offset(
                lane, static_cast<std::uint32_t>(actor), serial)] = 0;
        }
        for (std::uint16_t index = 0; index < own.deck.count; ++index) {
            const std::uint32_t serial = own.deck.values[index].index;
            if (serial < kOfficialSemanticSerialCapacity) {
                history->known_self_deck_serial[official_semantic_serial_offset(
                    lane, static_cast<std::uint32_t>(actor), serial)] = 1;
            }
        }
    } else if (history != nullptr
        && history->deck_membership_known[actor_knowledge] != 0) {
        std::uint32_t remembered_count = 0;
        bool current_deck_matches_memory = true;
        for (std::uint32_t serial = 1;
             serial < kOfficialSemanticSerialCapacity;
             ++serial) {
            if (history->known_self_deck_serial[official_semantic_serial_offset(
                    lane, static_cast<std::uint32_t>(actor), serial)] != 0) {
                ++remembered_count;
            }
        }
        for (std::uint16_t index = 0; index < own.deck.count; ++index) {
            const std::uint32_t serial = own.deck.values[index].index;
            if (serial >= kOfficialSemanticSerialCapacity
                || history->known_self_deck_serial[official_semantic_serial_offset(
                    lane, static_cast<std::uint32_t>(actor), serial)] == 0) {
                current_deck_matches_memory = false;
                break;
            }
        }
        if (!current_deck_matches_memory || remembered_count != own.deck.count) {
            history->deck_membership_known[actor_knowledge] = 0;
            history->deck_order_known[actor_knowledge] = 0;
        }
    }
    std::int32_t known_opponent_hand_count = 0;
    std::int32_t possible_opponent_hand_count = 0;
    std::uint16_t possible_hand_lower = 0;
    std::uint16_t possible_hand_upper = 0;
    if (history != nullptr) {
        for (std::uint32_t serial = 1;
             serial < kOfficialSemanticSerialCapacity;
             ++serial) {
            const std::size_t memory = official_semantic_serial_offset(
                lane, static_cast<std::uint32_t>(actor), serial);
            if (history->known_opponent_hand[memory] != 0) {
                if (known_opponent_hand_count < opp.hand.count) {
                    ++known_opponent_hand_count;
                } else {
                    history->known_opponent_hand[memory] = 0;
                }
            }
            if (history->possible_opponent_hand[memory] != 0) {
                ++possible_opponent_hand_count;
            }
        }
        const std::int32_t available = opp.hand.count > known_opponent_hand_count
            ? opp.hand.count - known_opponent_hand_count
            : 0;
        possible_hand_upper = history->possible_hand_upper[actor_knowledge];
        if (possible_hand_upper > possible_opponent_hand_count) {
            possible_hand_upper = static_cast<std::uint16_t>(
                possible_opponent_hand_count);
        }
        if (possible_hand_upper > available) {
            possible_hand_upper = static_cast<std::uint16_t>(available);
        }
        possible_hand_lower = history->possible_hand_lower[actor_knowledge];
        if (possible_hand_lower > possible_hand_upper) {
            possible_hand_lower = possible_hand_upper;
        }
        history->possible_hand_lower[actor_knowledge] = possible_hand_lower;
        history->possible_hand_upper[actor_knowledge] = possible_hand_upper;
        if (possible_hand_upper == 0) {
            for (std::uint32_t serial = 1;
                 serial < kOfficialSemanticSerialCapacity;
                 ++serial) {
                history->possible_opponent_hand[official_semantic_serial_offset(
                    lane, static_cast<std::uint32_t>(actor), serial)] = 0;
            }
            possible_opponent_hand_count = 0;
        }
        history->unknown_opponent_hand[actor_knowledge] =
            static_cast<std::uint16_t>(available);
    }
    const std::int32_t select_type = state->select_type > 0 ? state->select_type - 1 : 0;
    const std::int32_t context = state->select_context > 0 ? state->select_context - 1 : 0;
    const std::int64_t min_count = state->select_min > 0 ? state->select_min : 0;
    const std::int64_t max_count = state->select_max > min_count ? state->select_max : min_count;
    const std::int64_t own_status = official_codec_status_bits(own);
    const std::int64_t opp_status = official_codec_status_bits(opp);

    global_cat[0] = semantic0031_positive_enum(select_type, 65);
    global_cat[1] = semantic0031_positive_enum(context, 129);
    global_cat[2] = state->first_player == actor ? 1 : (state->first_player == opponent ? 2 : 0);
    global_cat[3] = (state->turn_state & kOfficialSupporterPlayedFlag) != 0 ? 2 : 1;
    global_cat[4] = (state->turn_state & kOfficialStadiumPlayedFlag) != 0 ? 2 : 1;
    global_cat[5] = (state->turn_state & kOfficialEnergyPlayedFlag) != 0 ? 2 : 1;
    global_cat[6] = (state->turn_state & (1U << 3U)) != 0 ? 2 : 1;
    global_cat[7] = semantic0031_clamp_i64(own_status + 1, 0, 32);
    global_cat[8] = semantic0031_clamp_i64(opp_status + 1, 0, 32);
    const bool deck_membership_known = history != nullptr
        && history->deck_membership_known[actor_knowledge] != 0;
    bool prize_membership_known = history != nullptr
        && history->prize_membership_known[actor_knowledge] != 0;
    const bool deck_order_known = history != nullptr
        && history->deck_order_known[actor_knowledge] != 0;
    global_cat[9] = deck_membership_known ? 2 : 1;
    global_cat[10] = deck_order_known ? 2 : 1;
    const bool looking_visible = state->looking.count > 0
        && (state->looking_player == actor || state->looking_player == 2);
    global_cat[11] = looking_visible ? 3 : 1;

    const float global_values[kSemantic0031GlobalNumWidth] = {
        static_cast<float>(state->turn),
        static_cast<float>(state->turn_action_count),
        static_cast<float>(own.deck.count),
        static_cast<float>(opp.deck.count),
        static_cast<float>(own.hand.count),
        static_cast<float>(opp.hand.count),
        static_cast<float>(own.prize.count),
        static_cast<float>(opp.prize.count),
        static_cast<float>(own.bench.count),
        static_cast<float>(opp.bench.count),
        static_cast<float>(state->options.count),
        static_cast<float>(min_count),
        static_cast<float>(max_count),
        static_cast<float>(state->remain_damage_counter),
        static_cast<float>(state->remain_energy_cost),
        static_cast<float>(known_opponent_hand_count),
        static_cast<float>(opp.hand.count - known_opponent_hand_count),
        static_cast<float>(official_bench_capacity(own)),
        static_cast<float>(official_bench_capacity(opp)),
        static_cast<float>(looking_visible ? state->looking.count : 0),
        static_cast<float>(looking_visible ? state->looking.count : 0),
        static_cast<float>(possible_opponent_hand_count),
        static_cast<float>(possible_hand_lower),
        static_cast<float>(possible_hand_upper),
    };
    for (std::uint32_t field = 0; field < kSemantic0031GlobalNumWidth; ++field) {
        global_num[field] = global_values[field];
        global_state[field] = kSemanticFieldPresent;
    }
    if (!looking_visible) {
        global_state[19] = kSemanticFieldUnknown;
        global_state[20] = kSemanticFieldUnknown;
    }

    std::int16_t location_players[kMaxSemantic0031CardEntities]{};
    std::int16_t location_areas[kMaxSemantic0031CardEntities]{};
    std::int16_t location_slots[kMaxSemantic0031CardEntities]{};
    for (std::uint32_t index = 0; index < kMaxSemantic0031CardEntities; ++index) {
        location_players[index] = -99;
        location_areas[index] = -99;
        location_slots[index] = -99;
    }
    std::int32_t card_count = 0;
    for (std::int32_t pass = 0; pass < 2; ++pass) {
        const std::int32_t player = pass == 0 ? actor : opponent;
        const OfficialPlayerStatePod& ps = state->players[player];
        const std::int64_t owner = official_codec_relative_owner(player, actor);
        const std::int64_t active_zone = semantic0031_zone_for(
            owner, kSemanticZoneSelfActive, kSemanticZoneOpponentActive);
        const std::int64_t bench_zone = semantic0031_zone_for(
            owner, kSemanticZoneSelfBench, kSemanticZoneOpponentBench);
        const std::int64_t status_bits = official_codec_status_bits(ps);
        for (std::uint16_t slot = 0; slot < ps.active.count; ++slot) {
            if (!semantic0031_add_pokemon_entity(
                    state, rules, ps.active.values[slot], player, actor,
                    static_cast<std::int32_t>(OfficialArea::kActive),
                    slot, active_zone, status_bits, &card_count,
                    location_players, location_areas, location_slots,
                    card_cat, card_num, card_state, card_parent, card_mask)) return;
        }
        for (std::uint16_t slot = 0; slot < ps.bench.count; ++slot) {
            if (!semantic0031_add_pokemon_entity(
                    state, rules, ps.bench.values[slot], player, actor,
                    static_cast<std::int32_t>(OfficialArea::kBench),
                    slot, bench_zone, 0, &card_count,
                    location_players, location_areas, location_slots,
                    card_cat, card_num, card_state, card_parent, card_mask)) return;
        }
        if (player == actor
            && !semantic0031_add_card_list(
                state, rules, ps.hand, false, player, actor,
                static_cast<std::int32_t>(OfficialArea::kHand),
                kSemanticZoneSelfHand, kPolicyKindCard, &card_count,
                location_players, location_areas, location_slots,
                card_cat, card_num, card_state, card_parent, card_mask)) return;
        const std::int64_t discard_zone = semantic0031_zone_for(
            owner, kSemanticZoneSelfDiscard, kSemanticZoneOpponentDiscard);
        if (!semantic0031_add_card_list(
                state, rules, ps.trash, false, player, actor,
                static_cast<std::int32_t>(OfficialArea::kTrash),
                discard_zone, kPolicyKindCard, &card_count,
                location_players, location_areas, location_slots,
                card_cat, card_num, card_state, card_parent, card_mask)) return;
    }
    if (!semantic0031_add_card_list(
            state, rules, state->stadium, false, -1, actor,
            static_cast<std::int32_t>(OfficialArea::kStadium),
            kSemanticZoneStadium, kPolicyKindStadium, &card_count,
            location_players, location_areas, location_slots,
            card_cat, card_num, card_state, card_parent, card_mask)) return;
    if (looking_visible
        && !semantic0031_add_card_list(
            state, rules, state->looking, false, actor, actor,
            static_cast<std::int32_t>(OfficialArea::kLooking),
            kSemanticZoneLooking, kPolicyKindLooking, &card_count,
            location_players, location_areas, location_slots,
            card_cat, card_num, card_state, card_parent, card_mask)) return;
    if (state->select_deck != 0
        && !semantic0031_add_card_list(
            state, rules, own.deck, false, actor, actor,
            static_cast<std::int32_t>(OfficialArea::kDeck),
            kSemanticZoneSelectDeck, kPolicyKindCard, &card_count,
            location_players, location_areas, location_slots,
            card_cat, card_num, card_state, card_parent, card_mask)) return;

    if (history != nullptr) {
        if (state->select_deck == 0 && deck_order_known) {
            std::int32_t deck_slot = 0;
            for (std::uint16_t index = 0; index < own.deck.count; ++index) {
                const OfficialCardRefPod ref = own.deck.values[index];
                if (semantic0031_find_serial(
                        card_cat, card_mask, card_count, ref.index) != 0) {
                    continue;
                }
                if (!semantic0031_add_card_entity(
                        state, rules, ref, false, kPolicyOwnerSelf,
                        22, deck_slot++, kPolicyKindCard, -1, 0,
                        actor, static_cast<std::int32_t>(OfficialArea::kDeck),
                        index, false, &card_count, location_players,
                        location_areas, location_slots, card_cat, card_num,
                        card_state, card_parent, card_mask, nullptr, 2)) {
                    return;
                }
            }
        }
        std::int32_t known_slot = 0;
        std::int32_t possible_slot = 0;
        std::int32_t remembered_slot = 0;
        const OfficialCardRefPod context_ref = state->context_card;
        OfficialCardRefPod effect_ref_for_cards{};
        if (state->effect_state.on_effect != 0) {
            effect_ref_for_cards = state->effect_state.ability.effect_card.card;
        }
        // The Python compiler appends causal entities in three independent
        // passes: known hand, possible hand, then remembered hidden cards.
        // Keeping these passes separate is observable because card row
        // positions are also used by event source/target relations.
        for (std::uint32_t serial = 1;
             serial < kOfficialSemanticSerialCapacity;
             ++serial) {
            const OfficialCardRefPod ref{static_cast<std::uint16_t>(serial)};
            const std::size_t memory = official_semantic_serial_offset(
                lane, static_cast<std::uint32_t>(actor), serial);
            const OfficialCardStatePod* card = official_codec_card(state, ref);
            if (card == nullptr || card->card_id <= 0) continue;
            if (history->known_opponent_hand[memory] != 0) {
                const std::int32_t slot = known_slot++;
                if (semantic0031_find_serial(
                        card_cat, card_mask, card_count, serial) == 0
                    && !semantic0031_add_card_entity(
                        state, rules, ref, false, kPolicyOwnerOpponent,
                        17, slot, 8, -1, 0,
                        opponent, static_cast<std::int32_t>(OfficialArea::kHand),
                        slot, false, &card_count, location_players,
                        location_areas, location_slots, card_cat, card_num,
                        card_state, card_parent, card_mask, nullptr, 2)) {
                    return;
                }
            }
        }
        for (std::uint32_t serial = 1;
             serial < kOfficialSemanticSerialCapacity;
             ++serial) {
            const OfficialCardRefPod ref{static_cast<std::uint16_t>(serial)};
            const std::size_t memory = official_semantic_serial_offset(
                lane, static_cast<std::uint32_t>(actor), serial);
            const OfficialCardStatePod* card = official_codec_card(state, ref);
            if (card == nullptr || card->card_id <= 0) continue;
            if (history->possible_opponent_hand[memory] != 0) {
                const std::int32_t slot = possible_slot++;
                if (semantic0031_find_serial(
                        card_cat, card_mask, card_count, serial) == 0
                    && !semantic0031_add_card_entity(
                        state, rules, ref, false, kPolicyOwnerOpponent,
                        17, slot, 8, -1, 0,
                        opponent, static_cast<std::int32_t>(OfficialArea::kHand),
                        slot, false, &card_count, location_players,
                        location_areas, location_slots, card_cat, card_num,
                        card_state, card_parent, card_mask, nullptr, 3)) {
                    return;
                }
            }
        }
        for (std::uint32_t serial = 1;
             serial < kOfficialSemanticSerialCapacity;
             ++serial) {
            const OfficialCardRefPod ref{static_cast<std::uint16_t>(serial)};
            const std::size_t memory = official_semantic_serial_offset(
                lane, static_cast<std::uint32_t>(actor), serial);
            const OfficialCardStatePod* card = official_codec_card(state, ref);
            if (card == nullptr || card->card_id <= 0) continue;
            const bool selected = ref == context_ref || ref == effect_ref_for_cards;
            if (history->remembered_opponent_cards[memory] != 0) {
                const std::int32_t slot = remembered_slot++;
                if (!selected
                    && semantic0031_find_serial(
                        card_cat, card_mask, card_count, serial) == 0
                    && !semantic0031_add_card_entity(
                        state, rules, ref, false, kPolicyOwnerOpponent,
                        23, slot, 10, -1, 0,
                        opponent, static_cast<std::int32_t>(OfficialArea::kHand),
                        slot, false, &card_count, location_players,
                        location_areas, location_slots, card_cat, card_num,
                        card_state, card_parent, card_mask, nullptr, 2)) {
                    return;
                }
            }
        }
    }

    const OfficialCardRefPod context_ref_for_cards = state->context_card;
    OfficialCardRefPod effect_ref_for_cards{};
    if (state->effect_state.on_effect != 0) {
        effect_ref_for_cards = state->effect_state.ability.effect_card.card;
    }
    if (!semantic0031_add_select_card(
            state, rules, context_ref_for_cards, actor, &card_count,
            location_players, location_areas, location_slots, card_cat, card_num,
            card_state, card_parent, card_mask)
        || !semantic0031_add_select_card(
            state, rules, effect_ref_for_cards, actor, &card_count,
            location_players, location_areas, location_slots, card_cat, card_num,
            card_state, card_parent, card_mask)) return;

    std::int32_t resource_count = 0;
    for (std::uint32_t index = 1; index < kOfficialCardCapacity; ++index) {
        const OfficialCardStatePod& card = state->cards[index];
        if (card.card_id > 0 && card.player == actor) {
            semantic0031_add_resource(
                resource_cat, resource_num, resource_state, resource_mask,
                &resource_count, card.card_id);
        }
    }
    const bool prize_reanchor_candidate = history != nullptr
        && state->select_deck != 0 && deck_membership_known;
    bool prize_reanchor_valid = prize_reanchor_candidate;
    float prize_reanchor_total = 0.0F;
    for (std::int32_t resource = 0; resource < resource_count; ++resource) {
        std::int64_t* cat =
            resource_cat + resource * kSemantic0031ResourceCatWidth;
        float* num = resource_num + resource * kSemantic0031ResourceNumWidth;
        std::int64_t* field_state =
            resource_state + resource * kSemantic0031ResourceNumWidth;
        const std::int64_t identity = cat[0];
        float visible_active = 0.0F;
        float visible_bench = 0.0F;
        float visible_hand = 0.0F;
        float visible_discard = 0.0F;
        float visible_stadium = 0.0F;
        float visible_playing = 0.0F;
        float visible_looking = 0.0F;
        for (std::int32_t entity = 0; entity < card_count; ++entity) {
            const std::int64_t* entity_cat =
                card_cat + entity * kSemantic0031CardCatWidth;
            if (card_mask[entity] == 0
                || entity_cat[0] != identity
                || entity_cat[2] != kPolicyOwnerSelf) {
                continue;
            }
            std::int64_t zone = entity_cat[3];
            if ((zone == kSemanticZoneSelfEnergy
                    || zone == kSemanticZoneSelfTool
                    || zone == kSemanticZoneSelfEvolution)
                && card_parent[entity] > 0
                && card_parent[entity] <= card_count) {
                zone = card_cat[
                    (card_parent[entity] - 1) * kSemantic0031CardCatWidth + 3];
            }
            if (zone == kSemanticZoneSelfActive) visible_active += 1.0F;
            else if (zone == kSemanticZoneSelfBench) visible_bench += 1.0F;
            else if (zone == kSemanticZoneSelfHand) visible_hand += 1.0F;
            else if (zone == kSemanticZoneSelfDiscard) visible_discard += 1.0F;
            else if (zone == kSemanticZoneStadium) visible_stadium += 1.0F;
            else if (zone == kSemanticZoneSelfPlaying) visible_playing += 1.0F;
            else if (zone == kSemanticZoneLooking) visible_looking += 1.0F;
        }
        // CausalKnowledge._visible_counts does not enumerate select.deck.
        // It nevertheless counts contextCard/effect as "playing" when their
        // serial has not already appeared in a public current-state zone.
        // During a deck search the relational card entity is de-duplicated
        // against the select-deck row, so reproduce that independent ledger
        // contribution here instead of leaking private Prize membership.
        const OfficialCardRefPod ledger_select_refs[2] = {
            context_ref_for_cards,
            effect_ref_for_cards,
        };
        for (std::uint32_t ref_index = 0; ref_index < 2; ++ref_index) {
            const OfficialCardRefPod ref = ledger_select_refs[ref_index];
            if (ref.index == 0
                || (ref_index == 1 && ref == ledger_select_refs[0])) {
                continue;
            }
            const OfficialCardStatePod* selected = official_codec_card(state, ref);
            if (selected == nullptr || selected->card_id != identity) continue;
            const std::int32_t relation = semantic0031_find_serial(
                card_cat, card_mask, card_count, ref.index);
            if (relation <= 0) continue;
            std::int64_t zone = card_cat[
                (relation - 1) * kSemantic0031CardCatWidth + 3];
            const bool already_visible =
                zone == kSemanticZoneSelfActive
                || zone == kSemanticZoneSelfBench
                || zone == kSemanticZoneSelfHand
                || zone == kSemanticZoneSelfDiscard
                || zone == kSemanticZoneStadium
                || zone == kSemanticZoneSelfPlaying
                || zone == kSemanticZoneLooking;
            if (!already_visible) visible_playing += 1.0F;
        }
        const float visible_total = visible_active + visible_bench + visible_hand
            + visible_discard + visible_stadium + visible_playing
            + visible_looking;
        const float hidden_remaining = num[0] > visible_total
            ? num[0] - visible_total
            : 0.0F;
        const float prize_total = static_cast<float>(own.prize.count);
        const float deck_total = static_cast<float>(own.deck.count);
        float exact_deck = 0.0F;
        float exact_prize = 0.0F;
        if (deck_membership_known) {
            for (std::uint16_t index = 0; index < own.deck.count; ++index) {
                const OfficialCardStatePod* card = official_codec_card(
                    state, own.deck.values[index]);
                if (card != nullptr && card->card_id == identity) exact_deck += 1.0F;
            }
        }
        if (prize_membership_known) {
            for (std::uint16_t index = 0; index < own.prize.count; ++index) {
                const OfficialCardStatePod* card = official_codec_card(
                    state, own.prize.values[index]);
                if (card != nullptr && card->card_id == identity) exact_prize += 1.0F;
            }
        }
        const float inferred_prize = num[0] - exact_deck - visible_total;
        if (prize_reanchor_candidate) {
            if (inferred_prize < 0.0F) {
                prize_reanchor_valid = false;
            } else {
                prize_reanchor_total += inferred_prize;
            }
        }
        cat[1] = deck_membership_known ? 4 : 5;
        cat[2] = prize_membership_known ? 4 : 5;
        cat[3] = deck_order_known ? 2 : 1;
        num[1] = visible_active;
        num[2] = visible_bench;
        num[3] = visible_hand;
        num[4] = visible_discard;
        num[5] = visible_stadium;
        num[6] = visible_playing;
        num[7] = deck_membership_known ? exact_deck : 0.0F;
        num[8] = deck_membership_known
            ? exact_deck
            : (hidden_remaining > prize_total
                ? hidden_remaining - prize_total
                : 0.0F);
        num[9] = deck_membership_known
            ? exact_deck
            : (hidden_remaining < deck_total ? hidden_remaining : deck_total);
        num[10] = prize_membership_known ? exact_prize : 0.0F;
        num[11] = prize_membership_known
            ? exact_prize
            : (hidden_remaining > deck_total
                ? hidden_remaining - deck_total
                : 0.0F);
        num[12] = prize_membership_known
            ? exact_prize
            : (hidden_remaining < prize_total ? hidden_remaining : prize_total);
        const std::uint64_t deck_source = history == nullptr
            ? 0ULL : history->deck_source_event[actor_knowledge];
        const std::uint64_t prize_source = history == nullptr
            ? 0ULL : history->prize_source_event[actor_knowledge];
        num[13] = deck_membership_known
            ? static_cast<float>(history_total - deck_source)
            : 0.0F;
        num[14] = prize_membership_known
            ? static_cast<float>(history_total - prize_source)
            : 0.0F;
        for (std::uint32_t field = 0; field < kSemantic0031ResourceNumWidth; ++field) {
            field_state[field] = kSemanticFieldPresent;
        }
        if (!deck_membership_known) {
            field_state[7] = kSemanticFieldUnknown;
        }
        if (!prize_membership_known) field_state[10] = kSemanticFieldUnknown;
        // Keep the per-identity inference in the otherwise-unknown exact slot
        // until the cross-identity sum has been validated below.
        if (prize_reanchor_candidate) {
            num[10] = inferred_prize < 0.0F ? 0.0F : inferred_prize;
        }
    }
    if (prize_reanchor_candidate) {
        prize_reanchor_valid = prize_reanchor_valid
            && prize_reanchor_total == static_cast<float>(own.prize.count);
        history->prize_membership_known[actor_knowledge] =
            prize_reanchor_valid ? 1 : 0;
        history->prize_source_event[actor_knowledge] =
            prize_reanchor_valid ? history_total : 0ULL;
        prize_membership_known = prize_reanchor_valid;
        for (std::int32_t resource = 0; resource < resource_count; ++resource) {
            std::int64_t* cat =
                resource_cat + resource * kSemantic0031ResourceCatWidth;
            float* num = resource_num + resource * kSemantic0031ResourceNumWidth;
            std::int64_t* field_state =
                resource_state + resource * kSemantic0031ResourceNumWidth;
            if (prize_reanchor_valid) {
                cat[2] = 4;
                num[11] = num[10];
                num[12] = num[10];
                num[14] = 0.0F;
                field_state[10] = kSemanticFieldPresent;
            } else {
                num[10] = 0.0F;
                field_state[10] = kSemanticFieldUnknown;
            }
        }
    }

    const std::uint64_t total = history == nullptr ? 0ULL : history->total_count[lane];
    const std::uint32_t live = total < kSemantic0031EventCapacity
        ? static_cast<std::uint32_t>(total)
        : static_cast<std::uint32_t>(kSemantic0031EventCapacity);
    const std::uint32_t start = total <= kSemantic0031EventCapacity
        ? 0U
        : (history == nullptr ? 0U : history->write_index[lane] % kSemantic0031EventCapacity);
    for (std::uint32_t pos = 0; pos < live; ++pos) {
        const std::uint32_t slot = (start + pos) % kSemantic0031EventCapacity;
        const std::size_t base = static_cast<std::size_t>(lane) * kSemantic0031EventCapacity + slot;
        const std::uint8_t type = history == nullptr ? 255U : history->log_type[base];
        if (type >= 24U) continue;
        const std::uint8_t count = history == nullptr ? 0U : history->param_count[base];
        const std::int32_t* params = history->params
            + base * kOfficialSemanticHistoryParamCapacity;
        std::int64_t* cat = event_cat + pos * kSemantic0031EventCatWidth;
        float* num = event_num + pos * kSemantic0031EventNumWidth;
        std::int64_t* state_row = event_state + pos * kSemantic0031EventNumWidth;
        const bool hidden_draw = type == 4U
            && count >= 1U
            && params[0] >= 0
            && params[0] <= 1
            && params[0] != actor;
        const bool hidden_move = type == 6U
            && count >= 5U
            && params[0] >= 0
            && params[0] <= 1
            && !official_semantic_move_visible_for_observer(
                params[0], actor, params[3], params[4],
                count >= 6U ? params[5] : 0);
        cat[0] = hidden_draw
            ? static_cast<std::int64_t>(OfficialSemanticLogType::kDrawReverse) + 1
            : (hidden_move
                ? static_cast<std::int64_t>(OfficialSemanticLogType::kMoveCardReverse) + 1
                : static_cast<std::int64_t>(type) + 1);
        cat[1] = 3;
        if ((type <= 4U || (type >= 6U && type <= 22U))
            && count >= 1U
            && params[0] >= 0
            && params[0] <= 1) {
            cat[1] = params[0] == actor ? 1 : 2;
        }
        cat[7] = 1;
        cat[8] = 1;
        cat[9] = 1;
        if (type == 1U && count >= 2U) {
            cat[21] = params[1] == 0 ? 1 : 2;
        } else if (type == 4U && count >= 3U && !hidden_draw) {
            cat[2] = semantic0031_clamp_i64(params[1], 0, 2048);
            cat[7] = cat[2] > 0 ? 2 : 1;
            cat[8] = 2;
            cat[14] = semantic0031_clamp_i64(params[2] + 1, 0, 256);
            event_source[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[2]);
        } else if (type == 6U && count >= 5U) {
            if (hidden_move) {
                cat[5] = semantic0031_positive_enum(params[3], 32);
                cat[6] = semantic0031_positive_enum(params[4], 32);
                cat[7] = 1;
                cat[8] = 1;
            } else {
                cat[2] = semantic0031_clamp_i64(params[1], 0, 2048);
                cat[5] = semantic0031_positive_enum(params[3], 32);
                cat[6] = semantic0031_positive_enum(params[4], 32);
                cat[7] = cat[2] > 0 ? 2 : 1;
                cat[8] = 2;
                cat[14] = semantic0031_clamp_i64(params[2] + 1, 0, 256);
                event_source[pos] = semantic0031_find_serial(
                    card_cat, card_mask, card_count, params[2]);
            }
        } else if (type == 7U && count >= 5U) {
            const bool visible_to_owner = params[0] == actor
                && params[4] != static_cast<std::int32_t>(OfficialArea::kPrize);
            if (visible_to_owner) {
                cat[0] = static_cast<std::int64_t>(
                    OfficialSemanticLogType::kMoveCard) + 1;
                cat[2] = semantic0031_clamp_i64(params[1], 0, 2048);
                cat[5] = semantic0031_positive_enum(params[3], 32);
                cat[6] = semantic0031_positive_enum(params[4], 32);
                cat[7] = cat[2] > 0 ? 2 : 1;
                cat[8] = 2;
                cat[14] = semantic0031_clamp_i64(params[2] + 1, 0, 256);
                event_source[pos] = semantic0031_find_serial(
                    card_cat, card_mask, card_count, params[2]);
            } else {
                cat[5] = semantic0031_positive_enum(params[3], 32);
                cat[6] = semantic0031_positive_enum(params[4], 32);
                cat[7] = 1;
                cat[8] = 1;
            }
        } else if (type == 7U && count >= 3U) {
            cat[5] = semantic0031_positive_enum(params[1], 32);
            cat[6] = semantic0031_positive_enum(params[2], 32);
            cat[7] = 1;
            cat[8] = 1;
        } else if (type == 8U && count >= 5U) {
            cat[10] = semantic0031_clamp_i64(params[1], 0, 2048);
            cat[11] = semantic0031_clamp_i64(params[3], 0, 2048);
            cat[16] = semantic0031_clamp_i64(params[2] + 1, 0, 256);
            cat[17] = semantic0031_clamp_i64(params[4] + 1, 0, 256);
            event_source[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[2]);
            event_target[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[4]);
        } else if (type == 9U && count >= 5U) {
            cat[12] = semantic0031_clamp_i64(params[1], 0, 2048);
            cat[13] = semantic0031_clamp_i64(params[3], 0, 2048);
            cat[18] = semantic0031_clamp_i64(params[2] + 1, 0, 256);
            cat[19] = semantic0031_clamp_i64(params[4] + 1, 0, 256);
            event_before[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[2]);
            event_after[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[4]);
        } else if (type == 10U && count >= 3U) {
            cat[2] = semantic0031_clamp_i64(params[1], 0, 2048);
            cat[7] = cat[2] > 0 ? 2 : 1;
            cat[8] = 2;
            cat[14] = semantic0031_clamp_i64(params[2] + 1, 0, 256);
            event_source[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[2]);
        } else if ((type == 11U || type == 12U || type == 13U)
            && count >= 5U) {
            cat[2] = semantic0031_clamp_i64(params[1], 0, 2048);
            cat[3] = semantic0031_clamp_i64(params[3], 0, 2048);
            cat[7] = cat[2] > 0 ? 2 : 1;
            cat[8] = 2;
            cat[9] = 2;
            cat[14] = semantic0031_clamp_i64(params[2] + 1, 0, 256);
            cat[15] = semantic0031_clamp_i64(params[4] + 1, 0, 256);
            event_source[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[2]);
            event_target[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[4]);
        } else if (type == 14U && count >= 7U) {
            cat[2] = semantic0031_clamp_i64(params[1], 0, 2048);
            cat[12] = semantic0031_clamp_i64(params[3], 0, 2048);
            cat[13] = semantic0031_clamp_i64(params[5], 0, 2048);
            cat[7] = cat[2] > 0 ? 2 : 1;
            cat[8] = 2;
            cat[14] = semantic0031_clamp_i64(params[2] + 1, 0, 256);
            cat[18] = semantic0031_clamp_i64(params[4] + 1, 0, 256);
            cat[19] = semantic0031_clamp_i64(params[6] + 1, 0, 256);
            event_source[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[2]);
            event_before[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[4]);
            event_after[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[6]);
        } else if (type == 15U && count >= 4U) {
            cat[2] = semantic0031_clamp_i64(params[1], 0, 2048);
            cat[4] = semantic0031_clamp_i64(params[3], 0, 4096);
            cat[7] = cat[2] > 0 ? 2 : 1;
            cat[8] = 2;
            cat[14] = semantic0031_clamp_i64(params[2] + 1, 0, 256);
            event_source[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[2]);
        } else if (type == 16U && count >= 5U) {
            cat[2] = semantic0031_clamp_i64(params[1], 0, 2048);
            cat[7] = cat[2] > 0 ? 2 : 1;
            cat[8] = 2;
            cat[14] = semantic0031_clamp_i64(params[2] + 1, 0, 256);
            cat[23] = params[4] == 0 ? 1 : 2;
            num[1] = static_cast<float>(params[3]);
            state_row[1] = kSemanticFieldPresent;
            event_source[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[2]);
        } else if (type >= 17U && type <= 21U && count >= 4U) {
            cat[2] = semantic0031_clamp_i64(params[2], 0, 2048);
            cat[7] = cat[2] > 0 ? 2 : 1;
            cat[8] = 2;
            cat[14] = semantic0031_clamp_i64(params[3] + 1, 0, 256);
            cat[20] = params[1] == 0 ? 1 : 2;
            event_source[pos] = semantic0031_find_serial(
                card_cat, card_mask, card_count, params[3]);
        } else if (type == 23U && count >= 2U) {
            cat[24] = semantic0031_positive_enum(params[0], 15);
            cat[25] = semantic0031_positive_enum(params[1], 31);
        } else if (type == 22U && count >= 2U) {
            cat[22] = params[1] == 0 ? 1 : 2;
        }
        num[0] = static_cast<float>(live - 1U - pos);
        state_row[0] = kSemanticFieldPresent;
        for (std::uint32_t field = 1; field < kSemantic0031EventNumWidth; ++field) {
            if (!(type == 16U && field == 1U && count >= 5U)) {
                state_row[field] = kSemanticFieldUnknown;
            }
        }
        event_mask[pos] = 1;
    }

    const std::int32_t context_relation = semantic0031_find_serial(
        card_cat, card_mask, card_count, state->context_card.index);
    OfficialCardRefPod effect_ref{};
    if (state->effect_state.on_effect != 0) {
        effect_ref = state->effect_state.ability.effect_card.card;
    }
    const std::int32_t effect_relation = semantic0031_find_serial(
        card_cat, card_mask, card_count, effect_ref.index);
    const OfficialCardStatePod* context_card = official_codec_card(state, state->context_card);
    const OfficialCardStatePod* effect_card = official_codec_card(state, effect_ref);
    const std::int64_t context_card_id = context_card == nullptr ? 0 : semantic0031_clamp_i64(context_card->card_id, 0, 2048);
    const std::int64_t effect_card_id = effect_card == nullptr ? 0 : semantic0031_clamp_i64(effect_card->card_id, 0, 2048);

    const std::uint16_t option_count = state->options.count > kMaxOfficialCodecOptions
        ? static_cast<std::uint16_t>(kMaxOfficialCodecOptions)
        : state->options.count;
    std::uint32_t skill_relation_count = 0;
    std::uint32_t effect_relation_count = 0;
    for (std::uint16_t option_index = 0; option_index < option_count; ++option_index) {
        const OfficialSelectOptionPod option = state->options.values[option_index];
        std::int32_t source_player = actor;
        std::int32_t source_area = -1;
        std::int32_t source_slot = -1;
        std::int32_t target_player = actor;
        std::int32_t target_area = -1;
        std::int32_t target_slot = -1;
        std::int32_t explicit_card = 0;
        std::int32_t explicit_serial = -1;
        std::int32_t attack_id = 0;
        std::int32_t number = -1;
        std::int32_t count = -1;
        std::int32_t energy_index = -1;
        std::int32_t tool_index = -1;
        std::int32_t special_condition_type = -1;
        switch (option.type) {
            case 0:
                number = option.params[0];
                break;
            case 3:
                source_area = option.params[0];
                source_slot = option.params[1];
                source_player = option.params[2];
                break;
            case 4:
                tool_index = option.params[3];
                source_area = option.params[0];
                source_slot = option.params[1];
                source_player = option.params[2];
                break;
            case 5:
            case 6:
                energy_index = option.params[3];
                if (option.type == 6) count = option.params[4];
                source_area = option.params[0];
                source_slot = option.params[1];
                source_player = option.params[2];
                break;
            case 7:
                source_area = static_cast<std::int32_t>(OfficialArea::kHand);
                source_slot = option.params[0];
                break;
            case 8:
            case 9:
                source_area = option.params[0];
                source_slot = option.params[1];
                target_area = option.params[2];
                target_slot = option.params[3];
                break;
            case 10:
            case 11:
                source_area = option.params[0];
                source_slot = option.params[1];
                break;
            case 13:
                attack_id = option.params[0] < 0 ? 0 : (option.params[0] > 4096 ? 4096 : option.params[0]);
                break;
            case 15:
                explicit_card = option.params[0];
                explicit_serial = option.params[1];
                break;
            case 16:
                special_condition_type = option.params[0];
                break;
            default:
                break;
        }
        std::int32_t source_entity = official_codec_find_location(
            location_players, location_areas, location_slots, card_count,
            source_player, source_area, source_slot);
        if (source_area == static_cast<std::int32_t>(OfficialArea::kStadium)) {
            const std::int32_t stadium_entity = official_codec_find_location(
                location_players, location_areas, location_slots, card_count,
                -1, source_area, source_slot);
            if (stadium_entity >= 0) source_entity = stadium_entity;
        }
        if (option.type == 15 && explicit_serial > 0) {
            const std::int32_t serial_entity = semantic0031_find_serial(
                card_cat, card_mask, card_count, explicit_serial);
            if (serial_entity > 0) source_entity = serial_entity - 1;
        }
        const std::int32_t parent_source_entity = source_entity;
        if (energy_index >= 0) {
            const std::int32_t energy_entity = semantic0031_find_attached_child(
                card_cat, card_parent, card_mask, card_count, parent_source_entity,
                kPolicyKindEnergy, energy_index);
            if (energy_entity >= 0) source_entity = energy_entity;
        } else if (tool_index >= 0) {
            const std::int32_t tool_entity = semantic0031_find_attached_child(
                card_cat, card_parent, card_mask, card_count, parent_source_entity,
                kPolicyKindTool, tool_index);
            if (tool_entity >= 0) source_entity = tool_entity;
        }
        const std::int32_t target_entity = official_codec_find_location(
            location_players, location_areas, location_slots, card_count,
            target_player, target_area, target_slot);
        std::int64_t* cat = option_cat + option_index * kSemantic0031OptionCatWidth;
        float* num = option_num + option_index * kSemantic0031OptionNumWidth;
        std::int64_t* state_row = option_state + option_index * kSemantic0031OptionNumWidth;
        const std::int64_t source_card = source_entity >= 0
            ? card_cat[source_entity * kSemantic0031CardCatWidth]
            : semantic0031_clamp_i64(explicit_card, 0, 2048);
        const std::int64_t target_card = target_entity >= 0
            ? card_cat[target_entity * kSemantic0031CardCatWidth]
            : 0;
        const std::int64_t target_owner = target_entity >= 0
            ? card_cat[target_entity * kSemantic0031CardCatWidth + 2]
            : 3;
        const std::int64_t source_owner = source_entity >= 0
            ? card_cat[source_entity * kSemantic0031CardCatWidth + 2]
            : official_codec_relative_owner(source_player, actor);
        cat[0] = semantic0031_positive_enum(option.type, 65);
        cat[1] = source_owner;
        cat[2] = semantic0031_positive_enum(source_area, 33);
        cat[3] = target_owner;
        cat[4] = semantic0031_positive_enum(target_area, 33);
        cat[5] = source_card;
        cat[6] = target_card;
        cat[7] = semantic0031_clamp_i64(attack_id, 0, 4096);
        cat[8] = semantic0031_positive_enum(special_condition_type, 33);
        cat[9] = global_cat[0];
        cat[10] = global_cat[1];
        cat[11] = context_card_id;
        cat[12] = effect_card_id;
        cat[13] = option_index + 1;
        cat[14] = semantic0031_positive_enum(source_slot, 256);
        cat[15] = semantic0031_positive_enum(target_slot, 256);
        cat[16] = semantic0031_positive_enum(energy_index, 256);
        cat[17] = semantic0031_positive_enum(tool_index, 256);
        cat[18] = explicit_serial < 0
            ? 0
            : semantic0031_clamp_i64(explicit_serial + 1, 0, 256);
        if (option.type == 0) {
            num[0] = static_cast<float>(number);
            state_row[0] = kSemanticFieldPresent;
        } else {
            state_row[0] = kSemanticFieldUnknown;
        }
        if (option.type == 6) {
            num[1] = static_cast<float>(count);
            state_row[1] = kSemanticFieldPresent;
        } else {
            state_row[1] = kSemanticFieldUnknown;
        }
        option_source[option_index] = source_entity + 1;
        option_target[option_index] = target_entity + 1;
        option_context[option_index] = context_relation;
        option_effect_card[option_index] = effect_relation;
        option_mask[option_index] = 1;

        // Match features.compiler.compile_canonical_row exactly: source,
        // context, and effect cards in that order; ability/play/delay within
        // each card; then each skill's effects followed by attack effects.
        // Duplicate (skill_id, role) pairs are removed per option.
        const std::int64_t relation_card_ids[3] = {
            source_card,
            context_card_id,
            effect_card_id,
        };
        std::int32_t seen_skill_ids[9]{};
        std::int32_t seen_skill_roles[9]{};
        std::uint32_t seen_skill_count = 0;
        for (std::uint32_t relation = 0; relation < 3; ++relation) {
            const OfficialCardRule* master = relation_card_ids[relation] > 0
                ? official_card_rule(
                    rules, static_cast<std::uint32_t>(relation_card_ids[relation]))
                : nullptr;
            if (master == nullptr) continue;
            const std::int32_t card_skills[3] = {
                master->values[kCardAbilityId],
                master->values[kCardPlayId],
                master->values[kCardDelayId],
            };
            for (std::uint32_t card_role = 0; card_role < 3; ++card_role) {
                const std::int32_t skill_id = card_skills[card_role];
                if (skill_id <= 0) continue;
                const std::int32_t role = static_cast<std::int32_t>(
                    relation * 3 + card_role + 1);
                bool duplicate = false;
                for (std::uint32_t seen = 0; seen < seen_skill_count; ++seen) {
                    if (seen_skill_ids[seen] == skill_id
                        && seen_skill_roles[seen] == role) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) continue;
                if (seen_skill_count >= 9
                    || skill_relation_count >= kSemantic0031OptionSkillCapacity) {
                    return;
                }
                seen_skill_ids[seen_skill_count] = skill_id;
                seen_skill_roles[seen_skill_count] = role;
                ++seen_skill_count;
                option_skill_id[skill_relation_count] = skill_id;
                option_skill_role[skill_relation_count] = role;
                option_skill_parent[skill_relation_count] = option_index + 1;
                option_skill_mask[skill_relation_count] = 1;
                ++skill_relation_count;

                const OfficialSkillRule* skill = official_skill_rule(
                    rules, static_cast<std::uint32_t>(skill_id));
                if (skill == nullptr) continue;
                const std::int32_t offset = skill->values[kSkillEffectOffset];
                const std::int32_t count = skill->values[kSkillEffectCount];
                if (offset < 0 || count < 0
                    || count > static_cast<std::int32_t>(kSemantic0031MaxSkillEffects)) {
                    return;
                }
                for (std::int32_t effect = 0; effect < count; ++effect) {
                    if (effect_relation_count >= kSemantic0031OptionEffectCapacity) {
                        return;
                    }
                    option_effect_id[effect_relation_count] = offset + effect + 1;
                    option_effect_role[effect_relation_count] = 1;
                    option_effect_parent[effect_relation_count] = option_index + 1;
                    option_effect_mask[effect_relation_count] = 1;
                    ++effect_relation_count;
                }
            }
        }
        if (attack_id > 0) {
            const OfficialAttackRule* attack = official_attack_rule(
                rules, static_cast<std::uint32_t>(attack_id));
            if (attack != nullptr) {
                const std::int32_t offsets[2] = {
                    attack->values[kAttackPreEffectOffset],
                    attack->values[kAttackPostEffectOffset],
                };
                const std::int32_t counts[2] = {
                    attack->values[kAttackPreEffectCount],
                    attack->values[kAttackPostEffectCount],
                };
                if (counts[0] < 0 || counts[1] < 0
                    || counts[0] + counts[1]
                        > static_cast<std::int32_t>(kSemantic0031MaxAttackEffects)) {
                    return;
                }
                for (std::uint32_t phase = 0; phase < 2; ++phase) {
                    if (offsets[phase] < 0) return;
                    for (std::int32_t effect = 0; effect < counts[phase]; ++effect) {
                        if (effect_relation_count >= kSemantic0031OptionEffectCapacity) {
                            return;
                        }
                        option_effect_id[effect_relation_count] =
                            offsets[phase] + effect + 1;
                        option_effect_role[effect_relation_count] = 2;
                        option_effect_parent[effect_relation_count] = option_index + 1;
                        option_effect_mask[effect_relation_count] = 1;
                        ++effect_relation_count;
                    }
                }
            }
        }
    }
    output.min_count[row] = min_count;
    output.max_count[row] = max_count;
    output.feature_valid[row] = 1;
}

template <typename T>
cudaError_t official_allocate(T** output, std::size_t count, std::size_t* bytes) {
    const std::size_t allocation = sizeof(T) * count;
    const cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(output), allocation);
    if (status == cudaSuccess) *bytes += allocation;
    return status;
}

}  // namespace

cudaError_t allocate_official_arena(
    OfficialDeviceArena* arena,
    const OfficialRuntimeConfig& config) {
    if (arena == nullptr
        || config.state_abi_version != kOfficialStateAbiVersion
        || config.rule_abi_version != kOfficialRuleAbiVersion
        || config.batch_size == 0
        || config.device_stack_bytes < kOfficialMinimumDeviceStackBytes
        || config.rule_pack_bytes < sizeof(OfficialRulePackHeader)) {
        return cudaErrorInvalidValue;
    }
    *arena = OfficialDeviceArena{};
    arena->config = config;

    std::size_t device_stack_bytes = 0;
    cudaError_t status = cudaDeviceGetLimit(
        &device_stack_bytes, cudaLimitStackSize);
    if (status != cudaSuccess) return status;
    if (device_stack_bytes < config.device_stack_bytes) {
        status = cudaDeviceSetLimit(
            cudaLimitStackSize, config.device_stack_bytes);
        if (status != cudaSuccess) return status;
        status = cudaDeviceGetLimit(&device_stack_bytes, cudaLimitStackSize);
        if (status != cudaSuccess) return status;
    }
    arena->device_stack_bytes = device_stack_bytes;

    status = official_allocate(
        &arena->states, config.batch_size, &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->rule_pack, config.rule_pack_bytes, &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->actions, config.batch_size, &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->statuses, config.batch_size, &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    const std::size_t batch = config.batch_size;
    const std::size_t history_entries =
        batch * kOfficialSemanticHistoryCapacity;
    const std::size_t semantic_actor_entries = batch * 2U;
    const std::size_t semantic_serial_entries =
        semantic_actor_entries * kOfficialSemanticSerialCapacity;
    status = official_allocate(
        &arena->semantic_history_total_count,
        batch,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_history_write_index,
        batch,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_history_log_type,
        history_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_history_param_count,
        history_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_history_params,
        history_entries * kOfficialSemanticHistoryParamCapacity,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_deck_membership_known,
        semantic_actor_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_prize_membership_known,
        semantic_actor_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_deck_order_known,
        semantic_actor_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_known_self_deck_serial,
        semantic_serial_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_deck_source_event,
        semantic_actor_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_prize_source_event,
        semantic_actor_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_known_opponent_hand,
        semantic_serial_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_possible_opponent_hand,
        semantic_serial_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_remembered_opponent_cards,
        semantic_serial_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_unknown_opponent_hand,
        semantic_actor_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_possible_hand_lower,
        semantic_actor_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->semantic_possible_hand_upper,
        semantic_actor_entries,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    OfficialSemanticHistoryDeviceView history_view{};
    history_view.states_base = arena->states;
    history_view.batch_size = config.batch_size;
    history_view.capacity = kOfficialSemanticHistoryCapacity;
    history_view.total_count = arena->semantic_history_total_count;
    history_view.write_index = arena->semantic_history_write_index;
    history_view.log_type = arena->semantic_history_log_type;
    history_view.param_count = arena->semantic_history_param_count;
    history_view.params = arena->semantic_history_params;
    history_view.deck_membership_known = arena->semantic_deck_membership_known;
    history_view.prize_membership_known = arena->semantic_prize_membership_known;
    history_view.deck_order_known = arena->semantic_deck_order_known;
    history_view.known_self_deck_serial = arena->semantic_known_self_deck_serial;
    history_view.deck_source_event = arena->semantic_deck_source_event;
    history_view.prize_source_event = arena->semantic_prize_source_event;
    history_view.known_opponent_hand = arena->semantic_known_opponent_hand;
    history_view.possible_opponent_hand = arena->semantic_possible_opponent_hand;
    history_view.remembered_opponent_cards = arena->semantic_remembered_opponent_cards;
    history_view.unknown_opponent_hand = arena->semantic_unknown_opponent_hand;
    history_view.possible_hand_lower = arena->semantic_possible_hand_lower;
    history_view.possible_hand_upper = arena->semantic_possible_hand_upper;
    status = official_allocate(
        &arena->semantic_history_view,
        1,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemcpy(
        arena->semantic_history_view,
        &history_view,
        sizeof(history_view),
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(arena->semantic_history_total_count, 0, sizeof(std::uint64_t) * batch);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(arena->semantic_history_write_index, 0, sizeof(std::uint32_t) * batch);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(arena->semantic_history_log_type, 0, sizeof(std::uint8_t) * history_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(arena->semantic_history_param_count, 0, sizeof(std::uint8_t) * history_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_history_params,
        0,
        sizeof(std::int32_t) * history_entries * kOfficialSemanticHistoryParamCapacity);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_deck_membership_known, 0,
        sizeof(std::uint8_t) * semantic_actor_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_prize_membership_known, 0,
        sizeof(std::uint8_t) * semantic_actor_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_deck_order_known, 0,
        sizeof(std::uint8_t) * semantic_actor_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_known_self_deck_serial, 0,
        sizeof(std::uint8_t) * semantic_serial_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_deck_source_event, 0,
        sizeof(std::uint64_t) * semantic_actor_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_prize_source_event, 0,
        sizeof(std::uint64_t) * semantic_actor_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_known_opponent_hand, 0,
        sizeof(std::uint8_t) * semantic_serial_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_possible_opponent_hand, 0,
        sizeof(std::uint8_t) * semantic_serial_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_remembered_opponent_cards, 0,
        sizeof(std::uint8_t) * semantic_serial_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_unknown_opponent_hand, 0,
        sizeof(std::uint16_t) * semantic_actor_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_possible_hand_lower, 0,
        sizeof(std::uint16_t) * semantic_actor_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = cudaMemset(
        arena->semantic_possible_hand_upper, 0,
        sizeof(std::uint16_t) * semantic_actor_entries);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.global_cat, batch * kGlobalCatWidth, &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.global_num, batch * kGlobalNumWidth, &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.entity_cat,
        batch * kMaxCodecEntities * kEntityCatWidth,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.entity_num,
        batch * kMaxCodecEntities * kEntityNumWidth,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.entity_parent,
        batch * kMaxCodecEntities,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.entity_mask,
        batch * kMaxCodecEntities,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.option_cat,
        batch * kMaxOfficialCodecOptions * kOptionCatWidth,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.option_num,
        batch * kMaxOfficialCodecOptions * kOptionNumWidth,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.option_equiv,
        batch * kMaxOfficialCodecOptions,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.option_mask,
        batch * kMaxOfficialCodecOptions,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.min_count, batch, &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.max_count, batch, &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    return cudaSuccess;
}

cudaError_t free_official_arena(OfficialDeviceArena* arena) {
    if (arena == nullptr) return cudaErrorInvalidValue;
    cudaError_t first_error = cudaSuccess;
#define PTCG_OFFICIAL_FREE(field)                        \
    do {                                                 \
        if ((field) != nullptr) {                        \
            const cudaError_t status = cudaFree(field); \
            if (first_error == cudaSuccess) {            \
                first_error = status;                    \
            }                                            \
            (field) = nullptr;                           \
        }                                                \
    } while (false)

    PTCG_OFFICIAL_FREE(arena->states);
    PTCG_OFFICIAL_FREE(arena->rule_pack);
    PTCG_OFFICIAL_FREE(arena->actions);
    PTCG_OFFICIAL_FREE(arena->statuses);
    PTCG_OFFICIAL_FREE(arena->semantic_history_total_count);
    PTCG_OFFICIAL_FREE(arena->semantic_history_write_index);
    PTCG_OFFICIAL_FREE(arena->semantic_history_log_type);
    PTCG_OFFICIAL_FREE(arena->semantic_history_param_count);
    PTCG_OFFICIAL_FREE(arena->semantic_history_params);
    PTCG_OFFICIAL_FREE(arena->semantic_deck_membership_known);
    PTCG_OFFICIAL_FREE(arena->semantic_prize_membership_known);
    PTCG_OFFICIAL_FREE(arena->semantic_deck_order_known);
    PTCG_OFFICIAL_FREE(arena->semantic_known_self_deck_serial);
    PTCG_OFFICIAL_FREE(arena->semantic_deck_source_event);
    PTCG_OFFICIAL_FREE(arena->semantic_prize_source_event);
    PTCG_OFFICIAL_FREE(arena->semantic_known_opponent_hand);
    PTCG_OFFICIAL_FREE(arena->semantic_possible_opponent_hand);
    PTCG_OFFICIAL_FREE(arena->semantic_remembered_opponent_cards);
    PTCG_OFFICIAL_FREE(arena->semantic_unknown_opponent_hand);
    PTCG_OFFICIAL_FREE(arena->semantic_possible_hand_lower);
    PTCG_OFFICIAL_FREE(arena->semantic_possible_hand_upper);
    PTCG_OFFICIAL_FREE(arena->semantic_history_view);
    PTCG_OFFICIAL_FREE(arena->codec.global_cat);
    PTCG_OFFICIAL_FREE(arena->codec.global_num);
    PTCG_OFFICIAL_FREE(arena->codec.entity_cat);
    PTCG_OFFICIAL_FREE(arena->codec.entity_num);
    PTCG_OFFICIAL_FREE(arena->codec.entity_parent);
    PTCG_OFFICIAL_FREE(arena->codec.entity_mask);
    PTCG_OFFICIAL_FREE(arena->codec.option_cat);
    PTCG_OFFICIAL_FREE(arena->codec.option_num);
    PTCG_OFFICIAL_FREE(arena->codec.option_equiv);
    PTCG_OFFICIAL_FREE(arena->codec.option_mask);
    PTCG_OFFICIAL_FREE(arena->codec.min_count);
    PTCG_OFFICIAL_FREE(arena->codec.max_count);
#undef PTCG_OFFICIAL_FREE
    arena->allocated_bytes = 0;
    arena->device_stack_bytes = 0;
    return first_error;
}

cudaError_t upload_official_rule_pack(
    OfficialDeviceArena* arena,
    const void* host_rule_pack,
    std::size_t bytes) {
    if (arena == nullptr || host_rule_pack == nullptr
        || bytes != arena->config.rule_pack_bytes
        || bytes < sizeof(OfficialRulePackHeader)) {
        return cudaErrorInvalidValue;
    }
    const auto* header = static_cast<const OfficialRulePackHeader*>(host_rule_pack);
    if (std::memcmp(header->magic, "PTCGRUL1", 8) != 0
        || header->schema_version != kOfficialRuleAbiVersion
        || header->abi_version != kOfficialRuleAbiVersion
        || header->total_bytes != bytes) {
        return cudaErrorInvalidValue;
    }
    return cudaMemcpy(
        arena->rule_pack, host_rule_pack, bytes, cudaMemcpyHostToDevice);
}

cudaError_t upload_official_states_async(
    OfficialDeviceArena* arena,
    const OfficialStatePod* states,
    cudaMemcpyKind copy_kind,
    cudaStream_t stream) {
    if (arena == nullptr || states == nullptr
        || (copy_kind != cudaMemcpyHostToDevice
            && copy_kind != cudaMemcpyDeviceToDevice)) {
        return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(
        arena->states,
        states,
        sizeof(OfficialStatePod) * arena->config.batch_size,
        copy_kind,
        stream);
}

template <typename IndexT>
cudaError_t fork_official_lanes_async_impl(
    const OfficialDeviceArena* source,
    OfficialDeviceArena* destination,
    const IndexT* source_lane_indices,
    std::uint32_t lane_count,
    cudaStream_t stream) {
    if (source == nullptr || destination == nullptr || source_lane_indices == nullptr
        || lane_count == 0 || destination->config.batch_size != lane_count
        || source->config.state_abi_version != destination->config.state_abi_version
        || source->config.rule_abi_version != destination->config.rule_abi_version
        || source->config.rule_pack_bytes != destination->config.rule_pack_bytes) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = cudaMemcpyAsync(
        destination->rule_pack,
        source->rule_pack,
        source->config.rule_pack_bytes,
        cudaMemcpyDeviceToDevice,
        stream);
    if (status != cudaSuccess) return status;
    fork_official_lanes_kernel<<<lane_count, kScratchCloneThreads, 0, stream>>>(
        *source,
        *destination,
        source_lane_indices,
        lane_count);
    return cudaGetLastError();
}

cudaError_t fork_official_lanes_i32_async(
    const OfficialDeviceArena* source,
    OfficialDeviceArena* destination,
    const std::int32_t* source_lane_indices,
    std::uint32_t lane_count,
    cudaStream_t stream) {
    return fork_official_lanes_async_impl(
        source, destination, source_lane_indices, lane_count, stream);
}

cudaError_t fork_official_lanes_i64_async(
    const OfficialDeviceArena* source,
    OfficialDeviceArena* destination,
    const std::int64_t* source_lane_indices,
    std::uint32_t lane_count,
    cudaStream_t stream) {
    return fork_official_lanes_async_impl(
        source, destination, source_lane_indices, lane_count, stream);
}

cudaError_t redeterminize_official_hidden_order_async(
    OfficialDeviceArena* arena,
    const std::int64_t* hidden_order_seeds,
    const std::int64_t* future_rng_seeds,
    cudaStream_t stream) {
    if (arena == nullptr || arena->states == nullptr
        || hidden_order_seeds == nullptr || future_rng_seeds == nullptr
        || arena->config.batch_size == 0) {
        return cudaErrorInvalidValue;
    }
    redeterminize_official_hidden_order_kernel<<<
        official_blocks(arena->config.batch_size), kOfficialThreads, 0, stream>>>(
        arena->states,
        arena->config.batch_size,
        hidden_order_seeds,
        future_rng_seeds);
    return cudaGetLastError();
}

cudaError_t redeterminize_official_public_belief_clean_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* exact_decks,
    const std::int64_t* observer_seats,
    const std::int64_t* hidden_membership_seeds,
    const std::int64_t* future_rng_seeds,
    std::uint8_t* result_codes,
    cudaStream_t stream) {
    if (arena == nullptr || arena->states == nullptr || exact_decks == nullptr
        || observer_seats == nullptr || hidden_membership_seeds == nullptr
        || future_rng_seeds == nullptr || result_codes == nullptr
        || arena->config.batch_size == 0) {
        return cudaErrorInvalidValue;
    }
    redeterminize_official_public_belief_clean_kernel<<<
        official_blocks(arena->config.batch_size), kOfficialThreads, 0, stream>>>(
        arena->states,
        arena->config.batch_size,
        exact_decks,
        observer_seats,
        hidden_membership_seeds,
        future_rng_seeds,
        result_codes);
    return cudaGetLastError();
}

template <typename DeckT>
cudaError_t reset_official_states_seeded_first_min_async_impl(
    OfficialDeviceArena* arena,
    const DeckT* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    OfficialSemanticHistoryDeviceView* semantic_history,
    cudaStream_t stream) {
    if (arena == nullptr || arena->states == nullptr || arena->rule_pack == nullptr
        || arena->statuses == nullptr || decks == nullptr || seeds == nullptr) {
        return cudaErrorInvalidValue;
    }
    reset_official_states_seeded_first_min_kernel<<<
        official_blocks(arena->config.batch_size), kOfficialThreads, 0, stream>>>(
        arena->states,
        arena->config.batch_size,
        arena->rule_pack,
        decks,
        seeds,
        lane_mask,
        arena->statuses,
        semantic_history);
    return cudaGetLastError();
}

cudaError_t reset_official_states_seeded_first_min_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_first_min_async_impl(
        arena, decks, seeds, lane_mask, nullptr, stream);
}

cudaError_t reset_official_states_seeded_first_min_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_first_min_async_impl(
        arena, decks, seeds, lane_mask, nullptr, stream);
}

cudaError_t reset_official_states_seeded_first_min_semantic_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_first_min_async_impl(
        arena,
        decks,
        seeds,
        lane_mask,
        arena != nullptr ? arena->semantic_history_view : nullptr,
        stream);
}

cudaError_t reset_official_states_seeded_first_min_semantic_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_first_min_async_impl(
        arena,
        decks,
        seeds,
        lane_mask,
        arena != nullptr ? arena->semantic_history_view : nullptr,
        stream);
}

template <typename DeckT>
cudaError_t reset_official_states_seeded_interactive_async_impl(
    OfficialDeviceArena* arena,
    const DeckT* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    OfficialSemanticHistoryDeviceView* semantic_history,
    cudaStream_t stream) {
    if (arena == nullptr || arena->states == nullptr || arena->rule_pack == nullptr
        || arena->statuses == nullptr || decks == nullptr || seeds == nullptr) {
        return cudaErrorInvalidValue;
    }
    reset_official_states_seeded_interactive_kernel<<<
        official_blocks(arena->config.batch_size), kOfficialThreads, 0, stream>>>(
        arena->states,
        arena->config.batch_size,
        arena->rule_pack,
        decks,
        seeds,
        lane_mask,
        arena->statuses,
        semantic_history);
    return cudaGetLastError();
}

cudaError_t reset_official_states_seeded_interactive_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_interactive_async_impl(
        arena, decks, seeds, lane_mask, nullptr, stream);
}

cudaError_t reset_official_states_seeded_interactive_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_interactive_async_impl(
        arena, decks, seeds, lane_mask, nullptr, stream);
}

cudaError_t reset_official_states_seeded_interactive_semantic_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_interactive_async_impl(
        arena,
        decks,
        seeds,
        lane_mask,
        arena != nullptr ? arena->semantic_history_view : nullptr,
        stream);
}

cudaError_t reset_official_states_seeded_interactive_semantic_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_interactive_async_impl(
        arena,
        decks,
        seeds,
        lane_mask,
        arena != nullptr ? arena->semantic_history_view : nullptr,
        stream);
}

cudaError_t classify_official_states_async(
    OfficialDeviceArena* arena,
    cudaStream_t stream) {
    if (arena == nullptr) return cudaErrorInvalidValue;
    classify_official_states_kernel<<<
        official_blocks(arena->config.batch_size),
        kOfficialThreads,
        0,
        stream>>>(arena->states, arena->config.batch_size, arena->statuses);
    return cudaPeekAtLastError();
}

cudaError_t advance_official_states_to_decision_async(
    OfficialDeviceArena* arena,
    cudaStream_t stream) {
    if (arena == nullptr) return cudaErrorInvalidValue;
    advance_official_states_kernel<<<
        official_blocks(arena->config.batch_size),
        kOfficialThreads,
        0,
        stream>>>(
            arena->states,
            arena->config.batch_size,
            arena->rule_pack,
            arena->statuses);
    return cudaPeekAtLastError();
}

cudaError_t apply_official_actions_and_advance_async(
    OfficialDeviceArena* arena,
    const OfficialActionPod* device_actions,
    cudaStream_t stream) {
    if (arena == nullptr) return cudaErrorInvalidValue;
    const OfficialActionPod* actions = device_actions;
    if (actions == nullptr) actions = arena->actions;
    apply_official_actions_kernel<<<
        official_blocks(arena->config.batch_size),
        kOfficialThreads,
        0,
        stream>>>(
            arena->states,
            arena->config.batch_size,
            arena->rule_pack,
            actions,
            arena->statuses,
            false,
            false);
    return cudaPeekAtLastError();
}

cudaError_t apply_official_packed_ready_actions_async(
    OfficialDeviceArena* arena,
    cudaStream_t stream) {
    if (arena == nullptr) return cudaErrorInvalidValue;
    apply_official_actions_kernel<<<
        official_blocks(arena->config.batch_size),
        kOfficialThreads,
        0,
        stream>>>(
            arena->states,
            arena->config.batch_size,
            arena->rule_pack,
            arena->actions,
            arena->statuses,
            true,
            false);
    return cudaPeekAtLastError();
}

cudaError_t apply_official_packed_setup_actions_async(
    OfficialDeviceArena* arena,
    cudaStream_t stream) {
    if (arena == nullptr) return cudaErrorInvalidValue;
    apply_official_actions_kernel<<<
        official_blocks(arena->config.batch_size),
        kOfficialThreads,
        0,
        stream>>>(
            arena->states,
            arena->config.batch_size,
            arena->rule_pack,
            arena->actions,
            arena->statuses,
            true,
            true);
    return cudaPeekAtLastError();
}

cudaError_t encode_official_policy_codec_v1_async(
    OfficialDeviceArena* arena,
    cudaStream_t stream) {
    if (arena == nullptr || arena->rule_pack == nullptr) return cudaErrorInvalidValue;
    encode_official_policy_codec_v1_kernel<<<
        official_blocks(arena->config.batch_size),
        kOfficialThreads,
        0,
        stream>>>(
            arena->states,
            arena->config.batch_size,
            arena->rule_pack,
            arena->codec);
    return cudaPeekAtLastError();
}

template <typename LaneT>
cudaError_t encode_official_semantic0031_codec_v2_async_impl(
    OfficialDeviceArena* arena,
    const LaneT* lane_indices,
    std::uint32_t lane_count,
    OfficialSemantic0031CodecBuffers output,
    cudaStream_t stream) {
    if (arena == nullptr || arena->rule_pack == nullptr
        || (lane_count > 0 && lane_indices == nullptr)
        || output.global_cat == nullptr || output.global_num == nullptr
        || output.global_state == nullptr || output.card_cat == nullptr
        || output.card_num == nullptr || output.card_state == nullptr
        || output.card_parent == nullptr || output.card_mask == nullptr
        || output.resource_cat == nullptr || output.resource_num == nullptr
        || output.resource_state == nullptr || output.resource_mask == nullptr
        || output.event_cat == nullptr || output.event_num == nullptr
        || output.event_state == nullptr || output.event_mask == nullptr
        || output.event_source == nullptr || output.event_target == nullptr
        || output.event_before == nullptr || output.event_after == nullptr
        || output.option_cat == nullptr || output.option_num == nullptr
        || output.option_state == nullptr || output.option_mask == nullptr
        || output.option_source == nullptr || output.option_target == nullptr
        || output.option_context == nullptr || output.option_effect_card == nullptr
        || output.min_count == nullptr || output.max_count == nullptr
        || output.targets == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (lane_count == 0) return cudaSuccess;
    encode_official_semantic0031_codec_v2_kernel<<<
        official_blocks(lane_count),
        kOfficialThreads,
        0,
        stream>>>(
            arena->states,
            arena->config.batch_size,
            arena->rule_pack,
            lane_indices,
            lane_count,
            arena->semantic_history_view,
            output);
    return cudaPeekAtLastError();
}

cudaError_t encode_official_semantic0031_codec_v2_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* lane_indices,
    std::uint32_t lane_count,
    OfficialSemantic0031CodecBuffers output,
    cudaStream_t stream) {
    return encode_official_semantic0031_codec_v2_async_impl(
        arena, lane_indices, lane_count, output, stream);
}

cudaError_t encode_official_semantic0031_codec_v2_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* lane_indices,
    std::uint32_t lane_count,
    OfficialSemantic0031CodecBuffers output,
    cudaStream_t stream) {
    return encode_official_semantic0031_codec_v2_async_impl(
        arena, lane_indices, lane_count, output, stream);
}

cudaError_t pack_official_actions_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* option_indices,
    const std::int32_t* counts,
    std::uint32_t option_capacity,
    cudaStream_t stream) {
    if (arena == nullptr || option_indices == nullptr || counts == nullptr
        || option_capacity == 0
        || option_capacity > kOfficialOptionCapacity) {
        return cudaErrorInvalidValue;
    }
    pack_official_actions_kernel<<<
        official_blocks(arena->config.batch_size),
        kOfficialThreads,
        0,
        stream>>>(
            arena->actions,
            option_indices,
            counts,
            arena->config.batch_size,
            option_capacity);
    return cudaPeekAtLastError();
}

cudaError_t pack_official_actions_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* option_indices,
    const std::int64_t* counts,
    std::uint32_t option_capacity,
    cudaStream_t stream) {
    if (arena == nullptr || option_indices == nullptr || counts == nullptr
        || option_capacity == 0
        || option_capacity > kOfficialOptionCapacity) {
        return cudaErrorInvalidValue;
    }
    pack_official_actions_kernel<<<
        official_blocks(arena->config.batch_size),
        kOfficialThreads,
        0,
        stream>>>(
            arena->actions,
            option_indices,
            counts,
            arena->config.batch_size,
            option_capacity);
    return cudaPeekAtLastError();
}

}  // namespace ptcg::cuda_engine
