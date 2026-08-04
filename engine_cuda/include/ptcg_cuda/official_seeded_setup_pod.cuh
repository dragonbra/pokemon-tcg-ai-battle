#pragma once

#include <cstdint>

#include "ptcg_cuda/official_turn_flow_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_SETUP_STATE_HD __host__ __device__
#else
#define PTCG_OFFICIAL_SETUP_STATE_HD
#endif

namespace ptcg::cuda_engine {

// This is the deterministic setup policy used by the existing official setup
// oracle: choose option indices [0, select_min) at every setup decision.  It is
// intentionally named first_min so callers cannot confuse it with the later
// interactive setup-selection VM.
constexpr std::int32_t kOfficialSeededDeckSize = 60;
constexpr std::int32_t kOfficialSeededFirstHand = 7;
constexpr std::int32_t kOfficialSeededPrizeSize = 6;
constexpr std::uint64_t kOfficialCardSetupBattlefieldOnlyFlag = 1ULL << 7U;
constexpr std::uint64_t kOfficialCardSetupActiveOnlyFlag = 1ULL << 8U;

enum class OfficialSeededSetupError : std::int32_t {
    kInvalidDeckCard = 1001,
    kNoBasicPokemon = 1002,
    kMulliganLimit = 1003,
};

struct OfficialSeededSetupPresence {
    bool basic = false;
    bool doll = false;
};

template <typename T, std::size_t Capacity>
PTCG_OFFICIAL_SETUP_STATE_HD inline void official_seeded_clear_list_tail(
    OfficialPodList<T, Capacity>* list) {
    for (std::size_t index = list->count; index < Capacity; ++index) {
        list->values[index] = T{};
    }
}

PTCG_OFFICIAL_SETUP_STATE_HD inline bool official_seeded_card_can_setup_active(
    const OfficialCardRule& card) {
    const bool basic = card.values[kCardType] == 0
        && card.values[kCardEvolutionType] == 1;
    return basic
        || (card.flags & kOfficialCardSetupBattlefieldOnlyFlag) != 0
        || (card.flags & kOfficialCardSetupActiveOnlyFlag) != 0;
}

PTCG_OFFICIAL_SETUP_STATE_HD inline bool official_seeded_card_can_setup_bench(
    const OfficialCardRule& card) {
    const bool basic = card.values[kCardType] == 0
        && card.values[kCardEvolutionType] == 1;
    return basic || (card.flags & kOfficialCardSetupBattlefieldOnlyFlag) != 0;
}

PTCG_OFFICIAL_SETUP_STATE_HD inline OfficialSeededSetupPresence
official_seeded_setup_presence(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    OfficialSeededSetupPresence result{};
    const OfficialPlayerStatePod& ps = state->players[player];
    for (std::uint16_t index = 0; index < ps.hand.count; ++index) {
        const OfficialCardStatePod* card = official_pod_card(
            state, ps.hand.values[index]);
        const OfficialCardRule* master = card == nullptr ? nullptr
            : official_card_rule(rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) {
            if (official_pod_ok(state)) {
                official_pod_fail(
                    state,
                    OfficialPodError::kRulePackBounds,
                    card == nullptr ? 0 : card->card_id);
            }
            return {};
        }
        const bool basic = master->values[kCardType] == 0
            && master->values[kCardEvolutionType] == 1;
        result.basic = result.basic || basic;
        result.doll = result.doll
            || (!basic
                && ((master->flags & kOfficialCardSetupBattlefieldOnlyFlag) != 0
                    || (master->flags & kOfficialCardSetupActiveOnlyFlag) != 0));
    }
    return result;
}

PTCG_OFFICIAL_SETUP_STATE_HD inline void official_seeded_reset_state(
    OfficialStatePod* state,
    std::uint64_t episode_id,
    std::uint64_t seed) {
#if defined(__CUDA_ARCH__)
    // `*state = OfficialStatePod{}` can make nvcc materialize the complete
    // 119,936-byte aggregate on each thread stack.  Zero the resident global
    // object in place and restore the few non-zero default fields explicitly.
    auto* words = reinterpret_cast<std::uint64_t*>(state);
    for (std::size_t index = 0;
         index < sizeof(OfficialStatePod) / sizeof(std::uint64_t);
         ++index) {
        words[index] = 0;
    }
    state->abi_version = kOfficialStateAbiVersion;
    state->first_player = -1;
    state->looking_player = -1;
    state->select_player = -1;
    state->last_stadium_player = 0;
    state->effect_interpreter.effect_owner = -1;
    for (std::size_t index = 0; index < kOfficialListCapacity; ++index) {
        state->prize_requests.values[index].player = -1;
    }
    state->episode_id = episode_id;
    state->players[0].player = 0;
    state->players[1].player = 1;
    state->move_counter = 1;
    official_seed_mt19937(&state->rng, seed);
#else
    official_pod_reset(state, episode_id, seed);
#endif
}

template <typename DeckT>
PTCG_OFFICIAL_SETUP_STATE_HD inline void official_seeded_initialize_decks(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const DeckT* input_decks,
    std::uint64_t episode_id,
    std::uint64_t seed) {
    official_seeded_reset_state(state, episode_id, seed);

    // The official State reserves index 0 and creates player pseudo-cards at
    // indices 1 and 2 before the 120 physical deck cards.
    for (std::int32_t player = 0; player < 2; ++player) {
        const std::uint16_t card_index = static_cast<std::uint16_t>(player + 1);
        OfficialCardStatePod& card = state->cards[card_index];
        card.card_id = 0;
        card.move_counter = state->move_counter++;
        card.player = static_cast<std::int8_t>(player);
        card.area = static_cast<std::uint8_t>(OfficialArea::kPlayer);
    }

    for (std::int32_t player = 0; player < 2 && official_pod_ok(state); ++player) {
        OfficialPlayerStatePod& ps = state->players[player];
        ps.deck.count = kOfficialSeededDeckSize;
        for (std::int32_t source = 0; source < kOfficialSeededDeckSize; ++source) {
            const DeckT raw_card_id = input_decks[
                player * kOfficialSeededDeckSize + source];
            if (raw_card_id <= static_cast<DeckT>(0)
                || raw_card_id > static_cast<DeckT>(0x7fffffff)
                || official_card_rule(
                       rules, static_cast<std::uint32_t>(raw_card_id)) == nullptr) {
                official_pod_fail(
                    state,
                    OfficialPodError::kInvalidAction,
                    static_cast<std::int32_t>(
                        OfficialSeededSetupError::kInvalidDeckCard));
                state->error_detail = static_cast<std::int32_t>(raw_card_id);
                return;
            }
            const std::int32_t card_id = static_cast<std::int32_t>(raw_card_id);
            const std::uint16_t card_index = static_cast<std::uint16_t>(
                3 + player * kOfficialSeededDeckSize + source);
            OfficialCardStatePod& card = state->cards[card_index];
            card.card_id = card_id;
            card.move_counter = state->move_counter++;
            card.player = static_cast<std::int8_t>(player);
            card.area = static_cast<std::uint8_t>(OfficialArea::kDeck);
            ps.deck.values[kOfficialSeededDeckSize - source - 1] = {card_index};
        }
        official_pod_shuffle_deck(state, player);
    }
}

PTCG_OFFICIAL_SETUP_STATE_HD inline bool official_seeded_setup_active_first(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    OfficialPlayerStatePod& ps = state->players[player];
    for (std::uint16_t index = 0; index < ps.hand.count; ++index) {
        const OfficialCardStatePod* card = official_pod_card(
            state, ps.hand.values[index]);
        const OfficialCardRule* master = card == nullptr ? nullptr
            : official_card_rule(rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) return false;
        if (!official_seeded_card_can_setup_active(*master)) continue;
        official_pod_move_card(
            state,
            player,
            OfficialArea::kHand,
            index,
            OfficialArea::kActive,
            true);
        state->setup_done_mask |= static_cast<std::uint8_t>(1U << player);
        return official_pod_ok(state);
    }
    official_pod_fail(
        state,
        OfficialPodError::kInvalidAction,
        static_cast<std::int32_t>(OfficialSeededSetupError::kNoBasicPokemon));
    return false;
}

PTCG_OFFICIAL_SETUP_STATE_HD inline bool official_seeded_setup_prizes(
    OfficialStatePod* state,
    std::int32_t player) {
    for (std::int32_t index = 0;
         index < kOfficialSeededPrizeSize && official_pod_ok(state);
         ++index) {
        OfficialPlayerStatePod& ps = state->players[player];
        if (ps.deck.count == 0) {
            official_pod_fail(state, OfficialPodError::kDeckOut, player);
            return false;
        }
        official_pod_move_card(
            state,
            player,
            OfficialArea::kDeck,
            static_cast<std::uint16_t>(ps.deck.count - 1),
            OfficialArea::kPrize,
            true);
    }
    return official_pod_ok(state);
}

PTCG_OFFICIAL_SETUP_STATE_HD inline bool official_seeded_return_hand_and_draw(
    OfficialStatePod* state,
    std::int32_t player) {
    OfficialPlayerStatePod& ps = state->players[player];
    while (ps.hand.count > 0 && official_pod_ok(state)) {
        official_pod_move_card(
            state,
            player,
            OfficialArea::kHand,
            static_cast<std::uint16_t>(ps.hand.count - 1),
            OfficialArea::kDeck,
            false);
    }
    if (!official_pod_ok(state)) return false;
    official_pod_shuffle_deck(state, player);
    return official_pod_draw(state, player, kOfficialSeededFirstHand)
            == kOfficialSeededFirstHand
        && official_pod_ok(state);
}

PTCG_OFFICIAL_SETUP_STATE_HD inline bool official_seeded_reset_until_ready(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    for (std::int32_t attempt = 0; attempt < kOfficialSeededDeckSize; ++attempt) {
        if (state->mulligan_count[player]
            < kOfficialSeededDeckSize
                - kOfficialSeededFirstHand
                - kOfficialSeededPrizeSize) {
            ++state->mulligan_count[player];
        }
        if (!official_seeded_return_hand_and_draw(state, player)) return false;
        const OfficialSeededSetupPresence presence = official_seeded_setup_presence(
            state, rules, player);
        if (!official_pod_ok(state)) return false;
        // first-min chooses Yes for the doll-only mulligan decision, so the
        // canonical path is ready only when a real Basic is in hand.
        if (presence.basic) {
            state->mulligan_mask &= static_cast<std::uint8_t>(~(1U << player));
            return official_seeded_setup_active_first(state, rules, player)
                && official_seeded_setup_prizes(state, player);
        }
        state->mulligan_mask |= static_cast<std::uint8_t>(1U << player);
    }
    official_pod_fail(
        state,
        OfficialPodError::kInvalidAction,
        static_cast<std::int32_t>(OfficialSeededSetupError::kMulliganLimit));
    return false;
}

template <typename DeckT>
PTCG_OFFICIAL_SETUP_STATE_HD inline bool official_seeded_setup_first_min_state(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const DeckT* input_decks,
    std::uint64_t episode_id,
    std::uint64_t seed) {
    official_seeded_initialize_decks(
        state, rules, input_decks, episode_id, seed);
    if (!official_pod_ok(state)) return false;

    // Fresh official State storage is zero-initialized; setup never writes the
    // looking-player field before entering Main.
    state->looking_player = 0;

    // first-min selects Yes at IsFirst, therefore player 0 starts.
    state->first_player = 0;
    if (official_pod_draw(state, 0, kOfficialSeededFirstHand)
            != kOfficialSeededFirstHand
        || official_pod_draw(state, 1, kOfficialSeededFirstHand)
            != kOfficialSeededFirstHand) {
        return false;
    }

    while (official_pod_ok(state)) {
        const OfficialSeededSetupPresence p0 = official_seeded_setup_presence(
            state, rules, 0);
        const OfficialSeededSetupPresence p1 = official_seeded_setup_presence(
            state, rules, 1);
        if (!official_pod_ok(state)) return false;
        const bool mulligan0 = !p0.basic;
        const bool mulligan1 = !p1.basic;
        state->mulligan_mask = static_cast<std::uint8_t>(mulligan0)
            | (static_cast<std::uint8_t>(mulligan1) << 1U);

        if (!mulligan0 && !official_seeded_setup_active_first(state, rules, 0)) {
            return false;
        }
        if (!mulligan1 && !official_seeded_setup_active_first(state, rules, 1)) {
            return false;
        }
        if (!mulligan0 && !mulligan1) {
            if (!official_seeded_setup_prizes(state, 0)
                || !official_seeded_setup_prizes(state, 1)) return false;
            break;
        }
        if (!mulligan0) {
            if (!official_seeded_setup_prizes(state, 0)
                || !official_seeded_reset_until_ready(state, rules, 1)) return false;
            break;
        }
        if (!mulligan1) {
            if (!official_seeded_setup_prizes(state, 1)
                || !official_seeded_reset_until_ready(state, rules, 0)) return false;
            break;
        }
        if (!official_seeded_return_hand_and_draw(state, 0)
            || !official_seeded_return_hand_and_draw(state, 1)) return false;
    }
    if (!official_pod_ok(state)) return false;

    // first-min selects zero compensation cards and zero setup-bench cards.
    // The official setup callback clears setup flags and face-down state before
    // entering TurnStart.
    state->setup_done_mask = 0;
    state->mulligan_mask = 0;
    OfficialCardStatePod* active0 = official_pod_card(
        state, state->players[0].active.values[0]);
    OfficialCardStatePod* active1 = official_pod_card(
        state, state->players[1].active.values[0]);
    if (active0 == nullptr || active1 == nullptr) return false;
    active0->reverse = 0;
    active1->reverse = 0;
    if (active0->move_counter > active1->move_counter) {
        official_pod_swap(&active0->move_counter, &active1->move_counter);
    }

    // TurnStart performs the first turn draw.  Main refresh/options are left to
    // the normal official flow dispatcher so setup and battle share one VM.
    if (official_finish_turn_start(state) != OfficialTurnFlowResult::kComplete
        || !official_pod_ok(state)) return false;
    for (std::int32_t player = 0; player < 2; ++player) {
        OfficialPlayerStatePod& ps = state->players[player];
        official_seeded_clear_list_tail(&ps.active);
        official_seeded_clear_list_tail(&ps.bench);
        official_seeded_clear_list_tail(&ps.prize);
        official_seeded_clear_list_tail(&ps.hand);
        official_seeded_clear_list_tail(&ps.deck);
        official_seeded_clear_list_tail(&ps.trash);
        official_seeded_clear_list_tail(&ps.energy);
        official_seeded_clear_list_tail(&ps.tool);
        official_seeded_clear_list_tail(&ps.pre_evolution);
        official_seeded_clear_list_tail(&ps.temporary);
    }
    return true;
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_SETUP_STATE_HD
