#pragma once

#include <cstddef>
#include <cstdint>

#include "ptcg_cuda/official_rng.cuh"

#if defined(__CUDACC__)
#define PTCG_SETUP_HD __host__ __device__
#else
#define PTCG_SETUP_HD
#endif

namespace ptcg::cuda_engine {

constexpr int kOfficialDeckSize = 60;
constexpr int kOfficialFirstHand = 7;
constexpr int kOfficialPrizeSize = 6;
constexpr int kOfficialBenchSize = 5;
constexpr int kOfficialSetupMaxOptions = 80;
constexpr int kOfficialSetupMaxDecisions = 128;

enum class SetupError : std::int32_t {
    kNone = 0,
    kInvalidCardId = 1,
    kNoBasicPokemon = 2,
    kDecisionOverflow = 3,
    kZoneOverflow = 4,
};

struct SetupCardMeta {
    std::uint8_t is_basic;
    std::uint8_t can_setup;
    std::uint8_t can_setup_active;
    std::uint8_t reserved;
};

struct SetupCardToken {
    std::uint16_t card_id;
    std::uint16_t instance_id;
};

struct SetupOption {
    std::uint8_t type;
    std::uint8_t reserved;
    std::int16_t param0;
    std::int16_t param1;
    std::int16_t param2;
    std::int16_t param3;
    std::int16_t param4;
};
static_assert(sizeof(SetupOption) == 12);

struct SetupDecision {
    std::uint8_t select_type;
    std::uint8_t select_context;
    std::uint8_t select_player;
    std::uint8_t reserved;
    std::uint16_t select_min;
    std::uint16_t select_max;
    std::uint16_t option_count;
    std::uint16_t reserved2;
    SetupOption options[kOfficialSetupMaxOptions];
};

struct SetupTrace {
    SetupDecision decisions[kOfficialSetupMaxDecisions];
    std::uint16_t decision_count;
    std::uint16_t overflow;
};

struct SetupPlayerState {
    SetupCardToken deck[kOfficialDeckSize];
    SetupCardToken hand[kOfficialDeckSize];
    SetupCardToken prize[kOfficialPrizeSize];
    SetupCardToken active;
    std::uint16_t deck_count;
    std::uint16_t hand_count;
    std::uint16_t prize_count;
    std::uint16_t mulligan_count;
    std::uint8_t active_present;
    std::uint8_t reserved[3];
};

struct SetupBattleState {
    SetupPlayerState players[2];
    OfficialMt19937 rng;
    std::int32_t error;
    std::int16_t first_player;
    std::int16_t actor;
    std::uint16_t turn;
    std::uint16_t surfaced_decisions;
    std::uint16_t continuation_id;
    std::uint16_t reserved;
};

namespace setup_detail {

constexpr std::uint8_t kSelectCard = 2;
constexpr std::uint8_t kSelectCount = 9;
constexpr std::uint8_t kSelectYesNo = 10;
constexpr std::uint8_t kContextSetupActive = 2;
constexpr std::uint8_t kContextSetupBench = 3;
constexpr std::uint8_t kContextDrawCount = 39;
constexpr std::uint8_t kContextIsFirst = 42;
constexpr std::uint8_t kContextMulligan = 43;
constexpr std::uint8_t kOptionNumber = 0;
constexpr std::uint8_t kOptionYes = 1;
constexpr std::uint8_t kOptionNo = 2;
constexpr std::uint8_t kOptionCard = 3;
constexpr std::int16_t kAreaHand = 2;
constexpr std::uint16_t kSelectedMainContinuation = 79;

PTCG_SETUP_HD inline bool token_equal(SetupCardToken left, SetupCardToken right) {
    return left.card_id == right.card_id && left.instance_id == right.instance_id;
}

PTCG_SETUP_HD inline void swap_token(SetupCardToken* left, SetupCardToken* right) {
    const SetupCardToken temporary = *left;
    *left = *right;
    *right = temporary;
}

PTCG_SETUP_HD inline SetupCardToken pop_token(
    SetupCardToken* values,
    std::uint16_t* count) {
    --(*count);
    return values[*count];
}

PTCG_SETUP_HD inline SetupCardToken remove_token(
    SetupCardToken* values,
    std::uint16_t* count,
    std::uint16_t index) {
    const SetupCardToken selected = values[index];
    for (std::uint16_t i = index + 1; i < *count; ++i) {
        values[i - 1] = values[i];
    }
    --(*count);
    return selected;
}

PTCG_SETUP_HD inline void shuffle_deck(SetupBattleState* state, int player) {
    official_shuffle_60(state->players[player].deck, &state->rng);
}

PTCG_SETUP_HD inline const SetupCardMeta* metadata(
    SetupBattleState* state,
    const SetupCardMeta* card_meta,
    std::uint32_t card_meta_count,
    std::uint16_t card_id) {
    if (card_id >= card_meta_count) {
        state->error = static_cast<std::int32_t>(SetupError::kInvalidCardId);
        return nullptr;
    }
    return &card_meta[card_id];
}

PTCG_SETUP_HD inline void draw_cards(
    SetupBattleState* state,
    int player,
    int count) {
    SetupPlayerState* ps = &state->players[player];
    for (int i = 0; i < count; ++i) {
        if (ps->deck_count == 0 || ps->hand_count >= kOfficialDeckSize) {
            state->error = static_cast<std::int32_t>(SetupError::kZoneOverflow);
            return;
        }
        ps->hand[ps->hand_count++] = pop_token(ps->deck, &ps->deck_count);
    }
}

PTCG_SETUP_HD inline SetupOption make_option(
    std::uint8_t type,
    std::int16_t param0 = 0,
    std::int16_t param1 = 0,
    std::int16_t param2 = 0,
    std::int16_t param3 = 0,
    std::int16_t param4 = 0) {
    return {type, 0, param0, param1, param2, param3, param4};
}

PTCG_SETUP_HD inline void record_decision(
    SetupBattleState* state,
    SetupTrace* trace,
    std::uint8_t select_type,
    std::uint8_t select_context,
    std::uint8_t select_player,
    std::uint16_t select_min,
    std::uint16_t select_max,
    const SetupOption* options,
    std::uint16_t option_count) {
    ++state->surfaced_decisions;
    if (trace == nullptr) {
        return;
    }
    if (trace->decision_count >= kOfficialSetupMaxDecisions
        || option_count > kOfficialSetupMaxOptions) {
        trace->overflow = 1;
        state->error = static_cast<std::int32_t>(SetupError::kDecisionOverflow);
        return;
    }
    SetupDecision* decision = &trace->decisions[trace->decision_count++];
    decision->select_type = select_type;
    decision->select_context = select_context;
    decision->select_player = select_player;
    decision->reserved = 0;
    decision->select_min = select_min;
    decision->select_max = select_max < option_count ? select_max : option_count;
    decision->option_count = option_count;
    decision->reserved2 = 0;
    for (std::uint16_t i = 0; i < option_count; ++i) {
        decision->options[i] = options[i];
    }
}

PTCG_SETUP_HD inline bool has_flagged_card(
    SetupBattleState* state,
    int player,
    const SetupCardMeta* card_meta,
    std::uint32_t card_meta_count,
    int flag) {
    const SetupPlayerState& ps = state->players[player];
    for (std::uint16_t i = 0; i < ps.hand_count; ++i) {
        const SetupCardMeta* meta = metadata(
            state, card_meta, card_meta_count, ps.hand[i].card_id);
        if (meta == nullptr) {
            return false;
        }
        if ((flag == 0 && meta->is_basic != 0)
            || (flag == 1 && meta->can_setup != 0 && meta->is_basic == 0)) {
            return true;
        }
    }
    return false;
}

PTCG_SETUP_HD inline bool pre_setup_active(
    SetupBattleState* state,
    SetupTrace* trace,
    int player,
    const SetupCardMeta* card_meta,
    std::uint32_t card_meta_count) {
    const bool has_basic = has_flagged_card(
        state, player, card_meta, card_meta_count, 0);
    if (state->error != 0 || has_basic) {
        return false;
    }
    const bool has_doll = has_flagged_card(
        state, player, card_meta, card_meta_count, 1);
    if (state->error != 0 || !has_doll) {
        return true;
    }
    const SetupOption options[2] = {
        make_option(kOptionYes), make_option(kOptionNo)};
    record_decision(
        state, trace, kSelectYesNo, kContextMulligan,
        static_cast<std::uint8_t>(player), 1, 1, options, 2);
    return true;
}

PTCG_SETUP_HD inline void setup_active(
    SetupBattleState* state,
    SetupTrace* trace,
    int player,
    const SetupCardMeta* card_meta,
    std::uint32_t card_meta_count) {
    SetupPlayerState* ps = &state->players[player];
    SetupOption options[kOfficialSetupMaxOptions];
    std::uint16_t option_count = 0;
    for (std::uint16_t i = 0; i < ps->hand_count; ++i) {
        const SetupCardMeta* meta = metadata(
            state, card_meta, card_meta_count, ps->hand[i].card_id);
        if (meta == nullptr) {
            return;
        }
        if (meta->can_setup_active != 0) {
            options[option_count++] = make_option(
                kOptionCard, kAreaHand, static_cast<std::int16_t>(i),
                static_cast<std::int16_t>(player));
        }
    }
    if (option_count == 0) {
        state->error = static_cast<std::int32_t>(SetupError::kNoBasicPokemon);
        return;
    }
    record_decision(
        state, trace, kSelectCard, kContextSetupActive,
        static_cast<std::uint8_t>(player), 1, 1, options, option_count);
    const std::uint16_t hand_index = static_cast<std::uint16_t>(options[0].param1);
    ps->active = remove_token(ps->hand, &ps->hand_count, hand_index);
    ps->active_present = 1;
}

PTCG_SETUP_HD inline void setup_prizes(SetupBattleState* state, int player) {
    SetupPlayerState* ps = &state->players[player];
    for (int i = 0; i < kOfficialPrizeSize; ++i) {
        if (ps->deck_count == 0 || ps->prize_count >= kOfficialPrizeSize) {
            state->error = static_cast<std::int32_t>(SetupError::kZoneOverflow);
            return;
        }
        ps->prize[ps->prize_count++] = pop_token(ps->deck, &ps->deck_count);
    }
}

PTCG_SETUP_HD inline void return_hand_and_shuffle(
    SetupBattleState* state,
    int player) {
    SetupPlayerState* ps = &state->players[player];
    while (ps->hand_count > 0) {
        ps->deck[ps->deck_count++] = ps->hand[--ps->hand_count];
    }
    shuffle_deck(state, player);
    draw_cards(state, player, kOfficialFirstHand);
}

PTCG_SETUP_HD inline void reset_until_ready(
    SetupBattleState* state,
    SetupTrace* trace,
    int player,
    const SetupCardMeta* card_meta,
    std::uint32_t card_meta_count) {
    while (state->error == 0) {
        SetupPlayerState* ps = &state->players[player];
        if (ps->mulligan_count < kOfficialDeckSize - kOfficialFirstHand - kOfficialPrizeSize) {
            ++ps->mulligan_count;
        }
        return_hand_and_shuffle(state, player);
        if (!pre_setup_active(state, trace, player, card_meta, card_meta_count)) {
            setup_active(state, trace, player, card_meta, card_meta_count);
            setup_prizes(state, player);
            return;
        }
    }
}

PTCG_SETUP_HD inline void record_compensation_draws(
    SetupBattleState* state,
    SetupTrace* trace) {
    for (int mulligan_player = 0; mulligan_player < 2; ++mulligan_player) {
        const std::uint16_t mulligans =
            state->players[mulligan_player].mulligan_count;
        if (mulligans == 0) {
            continue;
        }
        SetupOption options[kOfficialSetupMaxOptions];
        for (std::uint16_t i = 0; i <= mulligans; ++i) {
            options[i] = make_option(kOptionNumber, static_cast<std::int16_t>(i));
        }
        record_decision(
            state, trace, kSelectCount, kContextDrawCount,
            static_cast<std::uint8_t>(1 - mulligan_player),
            1, 1, options, static_cast<std::uint16_t>(mulligans + 1));
    }
}

PTCG_SETUP_HD inline void record_setup_bench(
    SetupBattleState* state,
    SetupTrace* trace,
    int player,
    const SetupCardMeta* card_meta,
    std::uint32_t card_meta_count) {
    const SetupPlayerState& ps = state->players[player];
    SetupOption options[kOfficialSetupMaxOptions];
    std::uint16_t option_count = 0;
    for (std::uint16_t i = 0; i < ps.hand_count; ++i) {
        const SetupCardMeta* meta = metadata(
            state, card_meta, card_meta_count, ps.hand[i].card_id);
        if (meta == nullptr) {
            return;
        }
        if (meta->can_setup != 0) {
            options[option_count++] = make_option(
                kOptionCard, kAreaHand, static_cast<std::int16_t>(i),
                static_cast<std::int16_t>(player));
        }
    }
    if (option_count == 0) {
        return;
    }
    const std::uint16_t select_max =
        option_count < kOfficialBenchSize ? option_count : kOfficialBenchSize;
    record_decision(
        state, trace, kSelectCard, kContextSetupBench,
        static_cast<std::uint8_t>(player), 0, select_max, options, option_count);
}

}  // namespace setup_detail

PTCG_SETUP_HD inline void official_setup_first_min(
    SetupBattleState* state,
    SetupTrace* trace,
    const std::uint16_t* input_decks,
    std::uint64_t seed,
    const SetupCardMeta* card_meta,
    std::uint32_t card_meta_count) {
    *state = {};
    if (trace != nullptr) {
        *trace = {};
    }
    official_seed_mt19937(&state->rng, seed);
    state->first_player = -1;
    state->actor = -1;

    for (int player = 0; player < 2; ++player) {
        SetupPlayerState* ps = &state->players[player];
        ps->deck_count = kOfficialDeckSize;
        for (int i = 0; i < kOfficialDeckSize; ++i) {
            const int source = kOfficialDeckSize - i - 1;
            ps->deck[i] = {
                input_decks[player * kOfficialDeckSize + source],
                static_cast<std::uint16_t>(3 + player * kOfficialDeckSize + source),
            };
        }
        setup_detail::shuffle_deck(state, player);
    }

    const SetupOption first_options[2] = {
        setup_detail::make_option(setup_detail::kOptionYes),
        setup_detail::make_option(setup_detail::kOptionNo),
    };
    setup_detail::record_decision(
        state, trace, setup_detail::kSelectYesNo, setup_detail::kContextIsFirst,
        0, 1, 1, first_options, 2);
    state->first_player = 0;
    setup_detail::draw_cards(state, 0, kOfficialFirstHand);
    setup_detail::draw_cards(state, 1, kOfficialFirstHand);

    while (state->error == 0) {
        const bool mulligan0 = setup_detail::pre_setup_active(
            state, trace, 0, card_meta, card_meta_count);
        if (!mulligan0) {
            setup_detail::setup_active(state, trace, 0, card_meta, card_meta_count);
        }
        const bool mulligan1 = setup_detail::pre_setup_active(
            state, trace, 1, card_meta, card_meta_count);
        if (!mulligan1) {
            setup_detail::setup_active(state, trace, 1, card_meta, card_meta_count);
        }
        if (!mulligan0 && !mulligan1) {
            setup_detail::setup_prizes(state, 0);
            setup_detail::setup_prizes(state, 1);
            break;
        }
        if (!mulligan0) {
            setup_detail::setup_prizes(state, 0);
            setup_detail::reset_until_ready(
                state, trace, 1, card_meta, card_meta_count);
            break;
        }
        if (!mulligan1) {
            setup_detail::setup_prizes(state, 1);
            setup_detail::reset_until_ready(
                state, trace, 0, card_meta, card_meta_count);
            break;
        }
        setup_detail::return_hand_and_shuffle(state, 0);
        setup_detail::return_hand_and_shuffle(state, 1);
    }

    if (state->error != 0) {
        return;
    }
    setup_detail::record_compensation_draws(state, trace);
    setup_detail::record_setup_bench(state, trace, 0, card_meta, card_meta_count);
    setup_detail::record_setup_bench(state, trace, 1, card_meta, card_meta_count);

    state->turn = 1;
    state->actor = state->first_player;
    setup_detail::draw_cards(state, state->actor, 1);
    state->continuation_id = setup_detail::kSelectedMainContinuation;
}

}  // namespace ptcg::cuda_engine

#undef PTCG_SETUP_HD
