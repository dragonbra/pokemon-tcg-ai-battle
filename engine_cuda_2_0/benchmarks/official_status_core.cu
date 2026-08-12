#include "ptcg_cuda/official_rng.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDeckSize = 60;
constexpr int kPrizeSize = 6;
constexpr int kMaxOptions = 80;
constexpr int kMaxFrames = 64;

constexpr int kVenipede = 535;
constexpr int kDarknessEnergy = 7;
constexpr int kPoisonSpray = 765;
constexpr int kSwablu = 197;
constexpr int kPsychicEnergy = 5;
constexpr int kDisarmingVoice = 266;

constexpr int kPhaseSetup = 0;
constexpr int kPhaseMain = 1;
constexpr int kPhaseCheckupEnd = 3;
constexpr int kResultNone = 0;
constexpr int kFinishNone = 0;
constexpr int kFinishNoActive = 3;

constexpr int kSelectMain = 1;
constexpr int kSelectCard = 2;
constexpr int kSelectCount = 9;
constexpr int kSelectYesNo = 10;
constexpr int kContextMain = 1;
constexpr int kContextSetupActive = 2;
constexpr int kContextSetupBench = 3;
constexpr int kContextToHand = 8;
constexpr int kContextDrawCount = 39;
constexpr int kContextIsFirst = 42;

constexpr int kOptionNumber = 0;
constexpr int kOptionYes = 1;
constexpr int kOptionNo = 2;
constexpr int kOptionCard = 3;
constexpr int kOptionPlay = 7;
constexpr int kOptionAttach = 8;
constexpr int kOptionAttack = 13;
constexpr int kOptionEnd = 14;

constexpr int kAreaHand = 2;
constexpr int kAreaActive = 4;
constexpr int kAreaPrize = 6;

struct CardToken {
    std::uint16_t id;
    std::uint16_t serial;
};

struct OptionRecord {
    int values[8];
};

struct PlayerState {
    CardToken deck[kDeckSize];
    CardToken hand[kDeckSize];
    CardToken prize[kPrizeSize];
    CardToken attached[kDeckSize];
    CardToken discard[kDeckSize];
    CardToken active;
    int deck_count;
    int hand_count;
    int prize_count;
    int attached_count;
    int discard_count;
    int damage;
    int active_present;
    int poisoned;
    int confused;
};

struct BattleState {
    PlayerState players[2];
    ptcg::cuda_engine::OfficialMt19937 rng;
    OptionRecord options[kMaxOptions];
    int option_count;
    int turn;
    int phase;
    int first_player;
    int actor;
    int turn_action_count;
    int energy_attached;
    int coin_head_count;
    int pending_coin;
    int pending_coin_player;
    int game_result;
    int finish_reason;
    int error;
};

struct PlayerSnapshot {
    CardToken hand[kDeckSize];
    CardToken prize[kPrizeSize];
    CardToken attached[kDeckSize];
    CardToken discard[kDeckSize];
    CardToken active;
    int deck_count;
    int hand_count;
    int prize_count;
    int attached_count;
    int discard_count;
    int hp;
    int max_hp;
    int active_present;
    int poisoned;
    int confused;
};

struct Frame {
    PlayerSnapshot players[2];
    OptionRecord options[kMaxOptions];
    int decision;
    int turn;
    int phase;
    int game_result;
    int finish_reason;
    int select_type;
    int select_context;
    int select_player;
    int select_min;
    int select_max;
    int option_count;
    int turn_action_count;
    int energy_attached;
    int first_player;
    int coin_head_count;
    int coin_event;
    int coin_player;
    int action_count;
    int action_index;
};

struct Trace {
    Frame frames[kMaxFrames];
    std::uint64_t seed;
    int frame_count;
    int game_result;
    int finish_reason;
    int error;
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::vector<std::uint16_t> read_deck(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open deck: " + path);
    }
    std::vector<std::uint16_t> cards;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            cards.push_back(static_cast<std::uint16_t>(std::stoul(line)));
        }
    }
    if (cards.size() != kDeckSize) {
        throw std::runtime_error("deck must contain exactly 60 cards: " + path);
    }
    return cards;
}

std::vector<std::uint64_t> parse_seeds(const std::string& value) {
    std::vector<std::uint64_t> seeds;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            seeds.push_back(std::stoull(item));
        }
    }
    if (seeds.empty()) {
        throw std::runtime_error("at least one seed is required");
    }
    return seeds;
}

void validate_controlled_deck(
    const std::vector<std::uint16_t>& deck,
    std::uint16_t pokemon,
    std::uint16_t energy,
    const std::string& path) {
    int pokemon_count = 0;
    int energy_count = 0;
    for (std::uint16_t card : deck) {
        pokemon_count += card == pokemon;
        energy_count += card == energy;
    }
    if (pokemon_count != 4 || energy_count != 56) {
        throw std::runtime_error(
            "controlled status deck must contain 4 expected Pokemon and 56 expected Energy: " +
            path);
    }
}

__host__ __device__ int max_hp_for(int player) {
    return player == 0 ? 80 : 50;
}

__host__ __device__ int pokemon_id_for(int player) {
    return player == 0 ? kVenipede : kSwablu;
}

__host__ __device__ int energy_id_for(int player) {
    return player == 0 ? kDarknessEnergy : kPsychicEnergy;
}

__host__ __device__ int attack_id_for(int player) {
    return player == 0 ? kPoisonSpray : kDisarmingVoice;
}

__device__ void swap_token(CardToken* lhs, CardToken* rhs) {
    const CardToken temporary = *lhs;
    *lhs = *rhs;
    *rhs = temporary;
}

// libstdc++ 11's optimized std::shuffle consumes one downscaled RNG value for
// two adjacent swaps. This is the same numeric contract used by setup parity,
// generalized here to keep card instance serials alongside card IDs.
__device__ void shuffle_60(
    CardToken* cards,
    ptcg::cuda_engine::OfficialMt19937* rng) {
    std::uint32_t i = 1;
    const std::uint32_t first_position =
        ptcg::cuda_engine::official_uniform_below(rng, 2);
    swap_token(&cards[i++], &cards[first_position]);
    while (i != 60) {
        const std::uint32_t swap_range = i + 1;
        const std::uint32_t combined = ptcg::cuda_engine::official_uniform_below(
            rng, swap_range * (swap_range + 1));
        const std::uint32_t first = combined / (swap_range + 1);
        const std::uint32_t second = combined % (swap_range + 1);
        swap_token(&cards[i++], &cards[first]);
        swap_token(&cards[i++], &cards[second]);
    }
}

__device__ CardToken pop_card(CardToken* cards, int* count) {
    return cards[--(*count)];
}

__device__ CardToken remove_card(CardToken* cards, int* count, int index) {
    const CardToken selected = cards[index];
    for (int i = index + 1; i < *count; ++i) {
        cards[i - 1] = cards[i];
    }
    --(*count);
    return selected;
}

__device__ void draw_cards(PlayerState* player, int count) {
    for (int i = 0; i < count; ++i) {
        player->hand[player->hand_count++] = pop_card(player->deck, &player->deck_count);
    }
}

__device__ bool has_basic(const PlayerState& player, int player_index) {
    const int pokemon_id = pokemon_id_for(player_index);
    for (int i = 0; i < player.hand_count; ++i) {
        if (player.hand[i].id == pokemon_id) {
            return true;
        }
    }
    return false;
}

__device__ void return_hand_and_shuffle(BattleState* state, int player_index) {
    PlayerState* player = &state->players[player_index];
    for (int i = player->hand_count; i > 0; --i) {
        player->deck[player->deck_count++] = player->hand[i - 1];
    }
    player->hand_count = 0;
    shuffle_60(player->deck, &state->rng);
    draw_cards(player, 7);
}

__device__ void clear_options(BattleState* state) {
    state->option_count = 0;
}

__device__ void add_option(
    BattleState* state,
    int type,
    int param0 = 0,
    int param1 = 0,
    int param2 = 0,
    int param3 = 0,
    int param4 = 0,
    int resolved_id = 0) {
    if (state->option_count >= kMaxOptions) {
        state->error = 2;
        return;
    }
    OptionRecord* option = &state->options[state->option_count];
    option->values[0] = state->option_count;
    option->values[1] = type;
    option->values[2] = param0;
    option->values[3] = param1;
    option->values[4] = param2;
    option->values[5] = param3;
    option->values[6] = param4;
    option->values[7] = resolved_id;
    ++state->option_count;
}

__device__ void snapshot_player(
    PlayerSnapshot* output,
    const PlayerState& player,
    int player_index) {
    output->deck_count = player.deck_count;
    output->hand_count = player.hand_count;
    output->prize_count = player.prize_count;
    output->attached_count = player.attached_count;
    output->discard_count = player.discard_count;
    output->active = player.active;
    output->active_present = player.active_present;
    output->max_hp = player.active_present ? max_hp_for(player_index) : 0;
    output->hp = player.active_present ? output->max_hp - player.damage : 0;
    output->poisoned = player.poisoned;
    output->confused = player.confused;
    for (int i = 0; i < player.hand_count; ++i) {
        output->hand[i] = player.hand[i];
    }
    for (int i = 0; i < player.prize_count; ++i) {
        output->prize[i] = player.prize[i];
    }
    for (int i = 0; i < player.attached_count; ++i) {
        output->attached[i] = player.attached[i];
    }
    for (int i = 0; i < player.discard_count; ++i) {
        output->discard[i] = player.discard[i];
    }
}

__device__ void record_frame(
    Trace* trace,
    BattleState* state,
    int select_type,
    int select_context,
    int select_player,
    int select_min,
    int select_max,
    int action_count,
    int action_index) {
    if (trace->frame_count >= kMaxFrames) {
        state->error = 3;
        return;
    }
    Frame* frame = &trace->frames[trace->frame_count];
    frame->decision = trace->frame_count;
    frame->turn = state->turn;
    frame->phase = state->phase;
    frame->game_result = state->game_result;
    frame->finish_reason = state->finish_reason;
    frame->select_type = select_type;
    frame->select_context = select_context;
    frame->select_player = select_player;
    frame->select_min = select_min;
    frame->select_max = select_max;
    frame->option_count = state->option_count;
    frame->turn_action_count = state->turn_action_count;
    frame->energy_attached = state->energy_attached;
    frame->first_player = state->first_player;
    frame->coin_head_count = state->coin_head_count;
    frame->coin_event = state->pending_coin;
    frame->coin_player = state->pending_coin_player;
    frame->action_count = action_count;
    frame->action_index = action_index;
    for (int i = 0; i < state->option_count; ++i) {
        frame->options[i] = state->options[i];
    }
    for (int player = 0; player < 2; ++player) {
        snapshot_player(&frame->players[player], state->players[player], player);
    }
    state->pending_coin = -1;
    state->pending_coin_player = -1;
    ++trace->frame_count;
}

__device__ void setup_active_decision(
    Trace* trace,
    BattleState* state,
    int player_index) {
    clear_options(state);
    PlayerState* player = &state->players[player_index];
    for (int i = 0; i < player->hand_count; ++i) {
        if (player->hand[i].id == pokemon_id_for(player_index)) {
            add_option(
                state,
                kOptionCard,
                kAreaHand,
                i,
                player_index,
                0,
                0,
                player->hand[i].id);
        }
    }
    record_frame(
        trace,
        state,
        kSelectCard,
        kContextSetupActive,
        player_index,
        1,
        1,
        1,
        0);
    const int hand_index = state->options[0].values[3];
    player->active = remove_card(player->hand, &player->hand_count, hand_index);
    player->active_present = 1;
    ++state->turn_action_count;
}

__device__ void setup_prizes(BattleState* state, int player_index) {
    PlayerState* player = &state->players[player_index];
    for (int i = 0; i < kPrizeSize; ++i) {
        player->prize[player->prize_count++] = pop_card(player->deck, &player->deck_count);
    }
}

__device__ void setup_one_remaining(
    Trace* trace,
    BattleState* state,
    int player_index,
    int* mulligan_count) {
    while (!has_basic(state->players[player_index], player_index)) {
        ++mulligan_count[player_index];
        return_hand_and_shuffle(state, player_index);
    }
    setup_active_decision(trace, state, player_index);
    setup_prizes(state, player_index);
}

__device__ void setup_battle(
    Trace* trace,
    BattleState* state,
    const std::uint16_t* input_decks,
    std::uint64_t seed,
    int* mulligan_count) {
    state->first_player = -1;
    state->phase = kPhaseSetup;
    state->turn_action_count = 1;
    state->pending_coin = -1;
    state->pending_coin_player = -1;
    state->game_result = kResultNone;
    state->finish_reason = kFinishNone;
    ptcg::cuda_engine::official_seed_mt19937(&state->rng, seed);

    for (int player_index = 0; player_index < 2; ++player_index) {
        PlayerState* player = &state->players[player_index];
        player->deck_count = kDeckSize;
        for (int i = 0; i < kDeckSize; ++i) {
            const int source = kDeckSize - i - 1;
            player->deck[i].id = input_decks[player_index * kDeckSize + source];
            player->deck[i].serial = static_cast<std::uint16_t>(
                3 + player_index * kDeckSize + source);
        }
        shuffle_60(player->deck, &state->rng);
    }

    clear_options(state);
    add_option(state, kOptionYes);
    add_option(state, kOptionNo);
    record_frame(
        trace,
        state,
        kSelectYesNo,
        kContextIsFirst,
        0,
        1,
        1,
        1,
        0);
    state->first_player = 0;
    ++state->turn_action_count;

    draw_cards(&state->players[0], 7);
    draw_cards(&state->players[1], 7);

    while (true) {
        const bool ready0 = has_basic(state->players[0], 0);
        const bool ready1 = has_basic(state->players[1], 1);
        if (ready0 && ready1) {
            setup_active_decision(trace, state, 0);
            setup_active_decision(trace, state, 1);
            setup_prizes(state, 0);
            setup_prizes(state, 1);
            break;
        }
        if (ready0) {
            setup_active_decision(trace, state, 0);
            setup_prizes(state, 0);
            setup_one_remaining(trace, state, 1, mulligan_count);
            break;
        }
        if (ready1) {
            setup_active_decision(trace, state, 1);
            setup_prizes(state, 1);
            setup_one_remaining(trace, state, 0, mulligan_count);
            break;
        }
        // The official setup only counts redraws made after the other player
        // has a valid opening hand. A simultaneous no-Basic redraw advances
        // the RNG but does not award either player compensating draws.
        return_hand_and_shuffle(state, 0);
        return_hand_and_shuffle(state, 1);
    }

    for (int mulligan_player = 0; mulligan_player < 2; ++mulligan_player) {
        if (mulligan_count[mulligan_player] == 0) {
            continue;
        }
        clear_options(state);
        for (int number = 0; number <= mulligan_count[mulligan_player]; ++number) {
            add_option(state, kOptionNumber, number, 0, 0, 0, 0, number);
        }
        record_frame(
            trace,
            state,
            kSelectCount,
            kContextDrawCount,
            1 - mulligan_player,
            1,
            1,
            1,
            0);
        ++state->turn_action_count;
    }

    for (int player_index = 0; player_index < 2; ++player_index) {
        clear_options(state);
        PlayerState* player = &state->players[player_index];
        for (int i = 0; i < player->hand_count; ++i) {
            if (player->hand[i].id == pokemon_id_for(player_index)) {
                add_option(
                    state,
                    kOptionCard,
                    kAreaHand,
                    i,
                    player_index,
                    0,
                    0,
                    player->hand[i].id);
            }
        }
        if (state->option_count > 0) {
            record_frame(
                trace,
                state,
                kSelectCard,
                kContextSetupBench,
                player_index,
                0,
                state->option_count < 5 ? state->option_count : 5,
                0,
                -1);
            ++state->turn_action_count;
        } else {
            // Empty optional setup-bench selections are auto-resolved by the
            // official engine. They do not surface as policy decisions, but
            // they still advance its internal turn-action counter.
            ++state->turn_action_count;
        }
    }
}

__device__ void generate_main_options(BattleState* state) {
    clear_options(state);
    PlayerState* player = &state->players[state->actor];
    for (int i = 0; i < player->hand_count; ++i) {
        const CardToken card = player->hand[i];
        if (card.id == pokemon_id_for(state->actor)) {
            add_option(state, kOptionPlay, i, 0, 0, 0, 0, card.id);
        } else if (card.id == energy_id_for(state->actor) && !state->energy_attached) {
            add_option(
                state,
                kOptionAttach,
                kAreaHand,
                i,
                kAreaActive,
                0,
                0,
                card.id);
        }
    }
    const bool first_player_first_turn = state->turn == 1 && state->actor == state->first_player;
    if (!first_player_first_turn && player->attached_count >= 1) {
        const int attack_id = attack_id_for(state->actor);
        add_option(state, kOptionAttack, attack_id, 0, -1, 0, 0, attack_id);
    }
    add_option(state, kOptionEnd);
}

__device__ int choose_main_action(const BattleState& state) {
    for (int preferred : {kOptionAttach, kOptionAttack, kOptionEnd}) {
        for (int i = 0; i < state.option_count; ++i) {
            if (state.options[i].values[1] == preferred) {
                return i;
            }
        }
    }
    return -1;
}

__device__ void knock_out(BattleState* state, int player_index) {
    PlayerState* player = &state->players[player_index];
    player->discard[player->discard_count++] = player->active;
    for (int i = player->attached_count; i > 0; --i) {
        player->discard[player->discard_count++] = player->attached[i - 1];
    }
    player->attached_count = 0;
    player->active_present = 0;
    player->damage = 0;
    player->poisoned = 0;
    player->confused = 0;
}

__device__ void emit_prize_and_finish(
    Trace* trace,
    BattleState* state,
    int knocked_out_player,
    int phase) {
    knock_out(state, knocked_out_player);
    const int prize_player = 1 - knocked_out_player;
    state->phase = phase;
    clear_options(state);
    PlayerState* player = &state->players[prize_player];
    for (int i = 0; i < player->prize_count; ++i) {
        add_option(
            state,
            kOptionCard,
            kAreaPrize,
            i,
            prize_player,
            0,
            0,
            player->prize[i].id);
    }
    record_frame(
        trace,
        state,
        kSelectCard,
        kContextToHand,
        prize_player,
        1,
        1,
        1,
        0);
    const CardToken selected = remove_card(player->prize, &player->prize_count, 0);
    player->hand[player->hand_count++] = selected;
    ++state->turn_action_count;
    state->game_result = knocked_out_player == 0 ? 2 : 1;
    state->finish_reason = kFinishNoActive;
}

__device__ int resolve_attack(BattleState* state) {
    PlayerState* attacker = &state->players[state->actor];
    PlayerState* defender = &state->players[1 - state->actor];
    if (attacker->confused) {
        const std::uint32_t value = ptcg::cuda_engine::official_mt19937_next(&state->rng);
        const bool head = (value % 2U) == 0U;
        state->coin_head_count = head ? 1 : 0;
        state->pending_coin = head ? 1 : 0;
        state->pending_coin_player = state->actor;
        if (!head) {
            attacker->damage += 30;
            if (attacker->damage > max_hp_for(state->actor)) {
                attacker->damage = max_hp_for(state->actor);
            }
            return attacker->damage >= max_hp_for(state->actor) ? state->actor : -1;
        }
    }
    if (state->actor == 0) {
        defender->poisoned = 1;
    } else {
        defender->damage += 10;
        if (defender->damage > max_hp_for(0)) {
            defender->damage = max_hp_for(0);
        }
        if (defender->damage >= max_hp_for(0)) {
            return 0;
        }
        defender->confused = 1;
    }
    return -1;
}

__device__ int pokemon_checkup(BattleState* state) {
    for (int player_index = 0; player_index < 2; ++player_index) {
        PlayerState* player = &state->players[player_index];
        if (player->active_present && player->poisoned) {
            player->damage += 10;
            const int max_hp = max_hp_for(player_index);
            if (player->damage > max_hp) {
                player->damage = max_hp;
            }
        }
    }
    for (int player_index = 0; player_index < 2; ++player_index) {
        const PlayerState& player = state->players[player_index];
        if (player.active_present && player.damage >= max_hp_for(player_index)) {
            return player_index;
        }
    }
    return -1;
}

__global__ void status_core_kernel(
    const std::uint16_t* input_decks,
    const std::uint64_t* seeds,
    Trace* traces,
    int count) {
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= count) {
        return;
    }
    BattleState state{};
    Trace* trace = &traces[env];
    trace->seed = seeds[env];
    int mulligan_count[2] = {0, 0};
    setup_battle(trace, &state, input_decks, seeds[env], mulligan_count);

    while (state.game_result == kResultNone && !state.error) {
        ++state.turn;
        state.phase = kPhaseMain;
        state.actor = (state.first_player + state.turn - 1) % 2;
        state.turn_action_count = 1;
        state.energy_attached = 0;
        if (state.players[state.actor].deck_count == 0) {
            state.game_result = state.actor == 0 ? 2 : 1;
            state.finish_reason = 2;
            break;
        }
        draw_cards(&state.players[state.actor], 1);

        bool turn_finished = false;
        int knocked_out = -1;
        while (!turn_finished && !state.error) {
            generate_main_options(&state);
            const int selected = choose_main_action(state);
            if (selected < 0) {
                state.error = 4;
                break;
            }
            record_frame(
                trace,
                &state,
                kSelectMain,
                kContextMain,
                state.actor,
                1,
                1,
                1,
                selected);
            ++state.turn_action_count;
            const OptionRecord option = state.options[selected];
            if (option.values[1] == kOptionAttach) {
                PlayerState* player = &state.players[state.actor];
                const CardToken energy = remove_card(
                    player->hand,
                    &player->hand_count,
                    option.values[3]);
                player->attached[player->attached_count++] = energy;
                state.energy_attached = 1;
            } else if (option.values[1] == kOptionAttack) {
                knocked_out = resolve_attack(&state);
                turn_finished = true;
            } else if (option.values[1] == kOptionEnd) {
                turn_finished = true;
            } else {
                state.error = 5;
            }
        }
        if (state.error) {
            break;
        }
        if (knocked_out >= 0) {
            emit_prize_and_finish(trace, &state, knocked_out, kPhaseMain);
            break;
        }
        // TurnEnd2 clears all once-per-turn flags before Pokemon Checkup.
        // A KO reached during the attack itself above still observes the flag.
        state.energy_attached = 0;
        knocked_out = pokemon_checkup(&state);
        if (knocked_out >= 0) {
            emit_prize_and_finish(trace, &state, knocked_out, kPhaseCheckupEnd);
            break;
        }
    }
    trace->game_result = state.game_result;
    trace->finish_reason = state.finish_reason;
    trace->error = state.error;
}

void print_card(CardToken card) {
    std::cout << '[' << card.id << ',' << card.serial << ']';
}

void print_cards(const CardToken* cards, int count) {
    std::cout << '[';
    for (int i = 0; i < count; ++i) {
        if (i) {
            std::cout << ',';
        }
        print_card(cards[i]);
    }
    std::cout << ']';
}

void print_player(const PlayerSnapshot& player) {
    std::cout << "{\"deck_count\":" << player.deck_count
              << ",\"hand\":";
    print_cards(player.hand, player.hand_count);
    std::cout << ",\"prize_count\":" << player.prize_count
              << ",\"attached\":";
    print_cards(player.attached, player.attached_count);
    std::cout << ",\"discard\":";
    print_cards(player.discard, player.discard_count);
    std::cout << ",\"active\":";
    if (player.active_present) {
        std::cout << "{\"card\":";
        print_card(player.active);
        std::cout << ",\"hp\":" << player.hp
                  << ",\"max_hp\":" << player.max_hp << '}';
    } else {
        std::cout << "null";
    }
    std::cout << ",\"poisoned\":" << player.poisoned
              << ",\"confused\":" << player.confused << '}';
}

void print_frame(const Frame& frame) {
    std::cout << "{\"decision\":" << frame.decision
              << ",\"meta\":[" << frame.turn << ',' << frame.phase << ','
              << frame.game_result << ',' << frame.finish_reason << ','
              << frame.select_type << ',' << frame.select_context << ','
              << frame.select_player << ',' << frame.select_min << ','
              << frame.select_max << ',' << frame.turn_action_count << ','
              << frame.energy_attached << ',' << frame.first_player << ','
              << frame.coin_head_count << ']'
              << ",\"options\":[";
    for (int option = 0; option < frame.option_count; ++option) {
        if (option) {
            std::cout << ',';
        }
        std::cout << '[';
        for (int value = 0; value < 8; ++value) {
            if (value) {
                std::cout << ',';
            }
            std::cout << frame.options[option].values[value];
        }
        std::cout << ']';
    }
    std::cout << "],\"action\":[";
    if (frame.action_count) {
        std::cout << frame.action_index;
    }
    std::cout << "],\"coin_event\":" << frame.coin_event
              << ",\"coin_player\":" << frame.coin_player
              << ",\"players\":[";
    print_player(frame.players[0]);
    std::cout << ',';
    print_player(frame.players[1]);
    std::cout << "]}";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string deck0_path;
        std::string deck1_path;
        std::string seed_text;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            if (argument == "--deck0") {
                deck0_path = argv[++i];
            } else if (argument == "--deck1") {
                deck1_path = argv[++i];
            } else if (argument == "--seeds") {
                seed_text = argv[++i];
            } else {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }
        if (deck0_path.empty() || deck1_path.empty() || seed_text.empty()) {
            throw std::runtime_error("--deck0, --deck1, and --seeds are required");
        }
        std::vector<std::uint16_t> decks = read_deck(deck0_path);
        const std::vector<std::uint16_t> deck1 = read_deck(deck1_path);
        validate_controlled_deck(decks, kVenipede, kDarknessEnergy, deck0_path);
        validate_controlled_deck(deck1, kSwablu, kPsychicEnergy, deck1_path);
        decks.insert(decks.end(), deck1.begin(), deck1.end());
        const std::vector<std::uint64_t> seeds = parse_seeds(seed_text);
        std::vector<Trace> traces(seeds.size());

        std::uint16_t* device_decks = nullptr;
        std::uint64_t* device_seeds = nullptr;
        Trace* device_traces = nullptr;
        check_cuda(cudaMalloc(&device_decks, decks.size() * sizeof(std::uint16_t)), "cudaMalloc decks");
        check_cuda(cudaMalloc(&device_seeds, seeds.size() * sizeof(std::uint64_t)), "cudaMalloc seeds");
        check_cuda(cudaMalloc(&device_traces, traces.size() * sizeof(Trace)), "cudaMalloc traces");
        check_cuda(cudaMemcpy(
            device_decks,
            decks.data(),
            decks.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice), "copy decks");
        check_cuda(cudaMemcpy(
            device_seeds,
            seeds.data(),
            seeds.size() * sizeof(std::uint64_t),
            cudaMemcpyHostToDevice), "copy seeds");
        check_cuda(cudaMemset(device_traces, 0, traces.size() * sizeof(Trace)), "clear traces");

        status_core_kernel<<<(seeds.size() + 31) / 32, 32>>>(
            device_decks,
            device_seeds,
            device_traces,
            static_cast<int>(seeds.size()));
        check_cuda(cudaGetLastError(), "status core launch");
        check_cuda(cudaMemcpy(
            traces.data(),
            device_traces,
            traces.size() * sizeof(Trace),
            cudaMemcpyDeviceToHost), "copy traces");
        check_cuda(cudaFree(device_traces), "cudaFree traces");
        check_cuda(cudaFree(device_seeds), "cudaFree seeds");
        check_cuda(cudaFree(device_decks), "cudaFree decks");

        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
        std::cout << "{\"version\":1,\"device\":\"" << properties.name
                  << "\",\"trace_bytes\":" << sizeof(Trace)
                  << ",\"traces\":[";
        for (std::size_t trace_index = 0; trace_index < traces.size(); ++trace_index) {
            if (trace_index) {
                std::cout << ',';
            }
            const Trace& trace = traces[trace_index];
            std::cout << "{\"seed\":" << trace.seed
                      << ",\"decisions\":" << trace.frame_count
                      << ",\"game_result\":" << trace.game_result
                      << ",\"finish_reason\":" << trace.finish_reason
                      << ",\"error\":" << trace.error
                      << ",\"frames\":[";
            for (int frame = 0; frame < trace.frame_count; ++frame) {
                if (frame) {
                    std::cout << ',';
                }
                print_frame(trace.frames[frame]);
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
