#include "ptcg_cuda/runtime.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ptcg::cuda_engine {
namespace {

constexpr int kThreads = 128;

__host__ __device__ constexpr std::uint32_t blocks_for(std::uint32_t count) {
    return (count + kThreads - 1) / kThreads;
}

__device__ std::uint64_t next_random(std::uint64_t* state) {
    std::uint64_t value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * 0x2545F4914F6CDD1DULL;
}

__device__ std::uint8_t resolve_target(std::uint8_t encoded, std::uint16_t actor) {
    switch (static_cast<Target>(encoded)) {
        case Target::kActor:
            return static_cast<std::uint8_t>(actor);
        case Target::kOpponent:
            return static_cast<std::uint8_t>(actor ^ 1U);
        case Target::kPlayer0:
            return 0;
        case Target::kPlayer1:
            return 1;
    }
    return static_cast<std::uint8_t>(actor);
}

__device__ void fail(BattleState* state, EngineError error) {
    state->error = static_cast<std::int32_t>(error);
    state->status = static_cast<std::int32_t>(EngineStatus::kError);
}

__global__ void reset_kernel(
    BattleState* states,
    const ResetSpec* specs,
    std::uint32_t batch_size,
    std::uint32_t policy_count,
    std::uint32_t action_count) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) {
        return;
    }

    BattleState* state = states + env;
    auto* words = reinterpret_cast<std::uint32_t*>(state);
    for (std::size_t i = 0; i < sizeof(BattleState) / sizeof(std::uint32_t); ++i) {
        words[i] = 0;
    }

    const ResetSpec spec = specs[env];
    state->rng_state = (static_cast<std::uint64_t>(spec.seed) << 32U) ^
                       (static_cast<std::uint64_t>(env) + 0x9E3779B97F4A7C15ULL);
    if (state->rng_state == 0) {
        state->rng_state = 0xD1B54A32D192ED03ULL;
    }
    state->episode_id = (static_cast<std::uint64_t>(spec.seed) << 32U) | env;
    state->status = static_cast<std::int32_t>(EngineStatus::kNeedsPolicy);
    state->error = static_cast<std::int32_t>(EngineError::kNone);
    state->winner = -1;
    state->actor = static_cast<std::uint16_t>(next_random(&state->rng_state) & 1ULL);
    state->action_count = static_cast<std::uint16_t>(action_count);
    state->entity_count = 2;

    for (std::uint32_t player = 0; player < 2; ++player) {
        if (spec.policy_id[player] >= policy_count) {
            fail(state, EngineError::kInvalidPolicyId);
            return;
        }
        PlayerState* dst = &state->players[player];
        dst->policy_id = spec.policy_id[player];
        dst->active_entity = static_cast<std::uint16_t>(player);
        dst->deck_count = spec.deck_count[player];
        dst->hand_count = spec.hand_count[player];
        dst->prize_count = 6;

        CardInstance* active = &state->cards[player];
        active->card_id = spec.active_card_id[player];
        active->max_hp = spec.active_hp[player] == 0 ? 100 : spec.active_hp[player];
        active->owner = static_cast<std::uint8_t>(player);
        active->zone = static_cast<std::uint8_t>(Zone::kActive);
        active->slot = 0;
        active->attached_to = -1;
    }
}

__device__ void execute_program(
    BattleState* state,
    const Instruction* instructions,
    std::uint32_t instruction_count,
    const ActionDescriptor& action) {
    std::uint32_t pc = action.program_offset;
    const std::uint32_t end = pc + action.program_length;
    if (end < pc || end > instruction_count) {
        fail(state, EngineError::kRulePackBounds);
        return;
    }

    state->status = static_cast<std::int32_t>(EngineStatus::kAdvancing);
    state->players[state->actor].actions_taken += 1;
    state->decision_count += 1;

    for (std::uint32_t budget = 0; budget < kInterpreterBudget && pc < end; ++budget) {
        const Instruction instruction = instructions[pc++];
        const auto opcode = static_cast<Opcode>(instruction.opcode);
        const std::uint8_t target = resolve_target(instruction.target, state->actor);

        switch (opcode) {
            case Opcode::kNop:
                break;
            case Opcode::kDraw: {
                PlayerState* player = &state->players[target];
                const std::uint32_t requested = instruction.arg0 > 0
                    ? static_cast<std::uint32_t>(instruction.arg0)
                    : 0U;
                const std::uint32_t drawn = requested < player->deck_count
                    ? requested
                    : player->deck_count;
                player->deck_count = static_cast<std::uint16_t>(player->deck_count - drawn);
                player->hand_count = static_cast<std::uint16_t>(player->hand_count + drawn);
                break;
            }
            case Opcode::kDamageActive: {
                CardInstance* active = &state->cards[state->players[target].active_entity];
                const std::uint32_t amount = instruction.arg0 > 0
                    ? static_cast<std::uint32_t>(instruction.arg0)
                    : 0U;
                const std::uint32_t damage = static_cast<std::uint32_t>(active->damage) + amount;
                active->damage = static_cast<std::uint16_t>(damage > 65535U ? 65535U : damage);
                break;
            }
            case Opcode::kHealActive: {
                CardInstance* active = &state->cards[state->players[target].active_entity];
                const std::uint16_t amount = instruction.arg0 > 0
                    ? static_cast<std::uint16_t>(instruction.arg0)
                    : 0U;
                active->damage = active->damage > amount
                    ? static_cast<std::uint16_t>(active->damage - amount)
                    : 0U;
                break;
            }
            case Opcode::kEndTurn:
                state->players[state->actor].turns_taken += 1;
                state->actor ^= 1U;
                state->turn += 1;
                state->status = static_cast<std::int32_t>(EngineStatus::kNeedsPolicy);
                break;
            case Opcode::kCheckKnockout: {
                const CardInstance* active = &state->cards[state->players[target].active_entity];
                if (active->max_hp > 0 && active->damage >= active->max_hp) {
                    state->winner = static_cast<std::int32_t>(target ^ 1U);
                    state->status = static_cast<std::int32_t>(EngineStatus::kTerminal);
                    return;
                }
                break;
            }
            case Opcode::kSetWinner:
                state->winner = static_cast<std::int32_t>(target);
                state->status = static_cast<std::int32_t>(EngineStatus::kTerminal);
                return;
            case Opcode::kEmitDecision:
                state->status = static_cast<std::int32_t>(EngineStatus::kNeedsPolicy);
                return;
            case Opcode::kSetCounter:
                if (instruction.arg0 < 0 || instruction.arg0 >= static_cast<std::int32_t>(kMaxCounters)) {
                    fail(state, EngineError::kRulePackBounds);
                    return;
                }
                state->counters[instruction.arg0] = static_cast<std::uint32_t>(instruction.arg1);
                break;
            case Opcode::kAddCounter:
                if (instruction.arg0 < 0 || instruction.arg0 >= static_cast<std::int32_t>(kMaxCounters)) {
                    fail(state, EngineError::kRulePackBounds);
                    return;
                }
                state->counters[instruction.arg0] += static_cast<std::uint32_t>(instruction.arg1);
                break;
            case Opcode::kDelayEffect:
                if (instruction.arg0 == 660 && instruction.arg1 == 1207) {
                    fail(state, EngineError::kKnownDivergence6601207);
                } else {
                    fail(state, EngineError::kUnsupportedOpcode);
                }
                return;
            case Opcode::kMoveCard: {
                bool moved = false;
                for (std::uint32_t i = 0; i < state->entity_count; ++i) {
                    CardInstance* card = &state->cards[i];
                    if (card->card_id == instruction.arg0 && card->owner == target) {
                        card->zone = static_cast<std::uint8_t>(instruction.arg1);
                        moved = true;
                        break;
                    }
                }
                if (!moved && (instruction.flags & 1U) != 0U) {
                    fail(state, EngineError::kRulePackBounds);
                    return;
                }
                break;
            }
            case Opcode::kConditionalCounterGe: {
                if (instruction.arg0 < 0 || instruction.arg0 >= static_cast<std::int32_t>(kMaxCounters)) {
                    fail(state, EngineError::kRulePackBounds);
                    return;
                }
                if (state->counters[instruction.arg0] >= static_cast<std::uint32_t>(instruction.arg1)) {
                    const std::int64_t destination = static_cast<std::int64_t>(pc) + instruction.arg2;
                    if (destination < static_cast<std::int64_t>(action.program_offset) ||
                        destination > static_cast<std::int64_t>(end)) {
                        fail(state, EngineError::kRulePackBounds);
                        return;
                    }
                    pc = static_cast<std::uint32_t>(destination);
                }
                break;
            }
            case Opcode::kRandomBranch: {
                const std::uint32_t threshold = instruction.arg0 < 0
                    ? 0U
                    : (instruction.arg0 > 10000 ? 10000U : static_cast<std::uint32_t>(instruction.arg0));
                const bool left = (next_random(&state->rng_state) % 10000ULL) < threshold;
                const std::int32_t delta = left ? instruction.arg1 : instruction.arg2;
                const std::int64_t destination = static_cast<std::int64_t>(pc) + delta;
                if (destination < static_cast<std::int64_t>(action.program_offset) ||
                    destination > static_cast<std::int64_t>(end)) {
                    fail(state, EngineError::kRulePackBounds);
                    return;
                }
                pc = static_cast<std::uint32_t>(destination);
                break;
            }
            case Opcode::kHalt:
                if (state->status == static_cast<std::int32_t>(EngineStatus::kAdvancing)) {
                    state->status = static_cast<std::int32_t>(EngineStatus::kNeedsPolicy);
                }
                return;
            default:
                fail(state, EngineError::kUnsupportedOpcode);
                return;
        }

        if (state->status == static_cast<std::int32_t>(EngineStatus::kTerminal) ||
            state->status == static_cast<std::int32_t>(EngineStatus::kError)) {
            return;
        }
    }

    if (pc < end) {
        fail(state, EngineError::kInterpreterBudget);
    } else if (state->status == static_cast<std::int32_t>(EngineStatus::kAdvancing)) {
        state->status = static_cast<std::int32_t>(EngineStatus::kNeedsPolicy);
    }
}

__global__ void apply_actions_kernel(
    BattleState* states,
    std::uint32_t batch_size,
    const std::int32_t* selected_actions,
    const Instruction* instructions,
    std::uint32_t instruction_count,
    const ActionDescriptor* actions,
    std::uint32_t action_count) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) {
        return;
    }
    BattleState* state = states + env;
    if (state->status != static_cast<std::int32_t>(EngineStatus::kNeedsPolicy)) {
        return;
    }
    const std::int32_t selected = selected_actions[env];
    if (selected < 0 || selected >= static_cast<std::int32_t>(action_count)) {
        fail(state, EngineError::kInvalidAction);
        return;
    }
    execute_program(state, instructions, instruction_count, actions[selected]);
}

__device__ std::int64_t actor_relative_owner(std::uint8_t owner, std::uint16_t actor) {
    return owner == actor ? 1 : 2;
}

__device__ std::int64_t actor_relative_zone(
    std::uint8_t zone,
    std::uint8_t owner,
    std::uint16_t actor) {
    const bool self = owner == actor;
    switch (static_cast<Zone>(zone)) {
        case Zone::kActive:
            return self ? 1 : 6;
        case Zone::kBench:
            return self ? 2 : 7;
        case Zone::kHand:
            return self ? 3 : 0;
        case Zone::kDiscard:
            return self ? 4 : 8;
        case Zone::kPrize:
            return self ? 5 : 9;
        case Zone::kStadium:
            return 10;
        default:
            return 0;
    }
}

__global__ void encode_policy_v1_kernel(
    BattleState* states,
    std::uint32_t batch_size,
    const ActionDescriptor* actions,
    std::uint32_t action_count,
    PolicyCodecBuffers codec) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) {
        return;
    }
    BattleState* state = states + env;

    auto* global_cat = codec.global_cat + env * kGlobalCatWidth;
    auto* global_num = codec.global_num + env * kGlobalNumWidth;
    auto* entity_cat = codec.entity_cat + env * kMaxCodecEntities * kEntityCatWidth;
    auto* entity_num = codec.entity_num + env * kMaxCodecEntities * kEntityNumWidth;
    auto* entity_parent = codec.entity_parent + env * kMaxCodecEntities;
    auto* entity_mask = codec.entity_mask + env * kMaxCodecEntities;
    auto* option_cat = codec.option_cat + env * kMaxCodecOptions * kOptionCatWidth;
    auto* option_num = codec.option_num + env * kMaxCodecOptions * kOptionNumWidth;
    auto* option_equiv = codec.option_equiv + env * kMaxCodecOptions;
    auto* option_mask = codec.option_mask + env * kMaxCodecOptions;

    for (std::uint32_t i = 0; i < kGlobalCatWidth; ++i) {
        global_cat[i] = 0;
    }
    for (std::uint32_t i = 0; i < kGlobalNumWidth; ++i) {
        global_num[i] = 0.0F;
    }
    for (std::uint32_t i = 0; i < kMaxCodecEntities; ++i) {
        entity_parent[i] = -1;
        entity_mask[i] = 0;
    }
    for (std::uint32_t i = 0; i < kMaxCodecEntities * kEntityCatWidth; ++i) {
        entity_cat[i] = 0;
    }
    for (std::uint32_t i = 0; i < kMaxCodecEntities * kEntityNumWidth; ++i) {
        entity_num[i] = 0.0F;
    }
    for (std::uint32_t i = 0; i < kMaxCodecOptions; ++i) {
        option_equiv[i] = -1;
        option_mask[i] = 0;
    }
    for (std::uint32_t i = 0; i < kMaxCodecOptions * kOptionCatWidth; ++i) {
        option_cat[i] = 0;
    }
    for (std::uint32_t i = 0; i < kMaxCodecOptions * kOptionNumWidth; ++i) {
        option_num[i] = 0.0F;
    }

    const std::uint16_t actor = state->actor;
    global_cat[0] = actor + 1;
    global_cat[1] = state->status;
    global_cat[2] = state->turn + 1;
    global_cat[3] = state->players[actor].policy_id + 1;
    global_cat[4] = state->players[actor ^ 1U].policy_id + 1;
    global_cat[5] = state->winner + 2;
    global_cat[6] = state->error;
    global_cat[7] = 1;
    global_num[0] = static_cast<float>(state->players[actor].deck_count) / 60.0F;
    global_num[1] = static_cast<float>(state->players[actor ^ 1U].deck_count) / 60.0F;
    global_num[2] = static_cast<float>(state->players[actor].hand_count) / 60.0F;
    global_num[3] = static_cast<float>(state->players[actor ^ 1U].hand_count) / 60.0F;
    global_num[4] = static_cast<float>(state->turn) / 100.0F;
    global_num[5] = static_cast<float>(state->decision_count) / 1000.0F;

    if (state->entity_count > kMaxCodecEntities) {
        fail(state, EngineError::kCodecEntityOverflow);
        return;
    }
    for (std::uint32_t index = 0; index < state->entity_count; ++index) {
        const CardInstance card = state->cards[index];
        auto* cat = entity_cat + index * kEntityCatWidth;
        auto* num = entity_num + index * kEntityNumWidth;
        cat[0] = card.card_id;
        cat[1] = actor_relative_owner(card.owner, actor);
        cat[2] = actor_relative_zone(card.zone, card.owner, actor);
        cat[3] = card.slot + 1;
        cat[4] = 2;
        cat[5] = card.flags + 1;
        num[0] = card.max_hp == 0 ? 0.0F : static_cast<float>(card.damage) / card.max_hp;
        num[1] = static_cast<float>(card.max_hp) / 400.0F;
        num[2] = static_cast<float>(card.damage) / 400.0F;
        entity_mask[index] = 1;
    }

    if (action_count > kMaxCodecOptions) {
        fail(state, EngineError::kCodecOptionOverflow);
        return;
    }
    for (std::uint32_t index = 0; index < action_count; ++index) {
        const ActionDescriptor action = actions[index];
        auto* cat = option_cat + index * kOptionCatWidth;
        cat[0] = action.option_type;
        cat[3] = 1;
        cat[4] = action.card_id;
        cat[11] = index + 1;
        option_equiv[index] = action.action_id;
        option_mask[index] = 1;
    }
    codec.min_count[env] = 1;
    codec.max_count[env] = 1;
}

__global__ void route_kernel(
    BattleState* states,
    std::uint32_t batch_size,
    std::uint32_t policy_count,
    std::uint32_t route_capacity,
    std::uint32_t* route_counts,
    std::int32_t* route_indices) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) {
        return;
    }
    BattleState* state = states + env;
    if (state->status != static_cast<std::int32_t>(EngineStatus::kNeedsPolicy)) {
        return;
    }
    const std::uint32_t policy = state->players[state->actor].policy_id;
    if (policy >= policy_count) {
        fail(state, EngineError::kInvalidPolicyId);
        return;
    }
    const std::uint32_t slot = atomicAdd(route_counts + policy, 1U);
    if (slot >= route_capacity) {
        fail(state, EngineError::kRouteOverflow);
        return;
    }
    route_indices[policy * route_capacity + slot] = static_cast<std::int32_t>(env);
}

__device__ void hash_value(std::uint64_t* digest, std::uint64_t value) {
    *digest ^= value;
    *digest *= 1099511628211ULL;
}

__global__ void digest_kernel(
    const BattleState* states,
    std::uint32_t batch_size,
    std::uint64_t* output) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size) {
        return;
    }
    const BattleState* state = states + env;
    std::uint64_t digest = 1469598103934665603ULL;
    hash_value(&digest, state->rng_state);
    hash_value(&digest, state->turn);
    hash_value(&digest, state->decision_count);
    hash_value(&digest, static_cast<std::uint32_t>(state->status));
    hash_value(&digest, static_cast<std::uint32_t>(state->error));
    hash_value(&digest, static_cast<std::uint32_t>(state->winner));
    hash_value(&digest, state->actor);
    for (std::uint32_t player = 0; player < 2; ++player) {
        hash_value(&digest, state->players[player].policy_id);
        hash_value(&digest, state->players[player].deck_count);
        hash_value(&digest, state->players[player].hand_count);
    }
    for (std::uint32_t index = 0; index < state->entity_count; ++index) {
        hash_value(&digest, state->cards[index].card_id);
        hash_value(&digest, state->cards[index].damage);
        hash_value(&digest, state->cards[index].zone);
    }
    output[env] = digest;
}

template <typename T>
cudaError_t allocate_array(T** pointer, std::size_t count, std::size_t* bytes) {
    const std::size_t size = sizeof(T) * count;
    const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(pointer), size);
    if (status == cudaSuccess) {
        *bytes += size;
    }
    return status;
}

}  // namespace

std::size_t codec_bytes_per_environment() {
    return kGlobalCatWidth * sizeof(std::int64_t) +
           kGlobalNumWidth * sizeof(float) +
           kMaxCodecEntities * kEntityCatWidth * sizeof(std::int64_t) +
           kMaxCodecEntities * kEntityNumWidth * sizeof(float) +
           kMaxCodecEntities * sizeof(std::int64_t) +
           kMaxCodecEntities * sizeof(std::uint8_t) +
           kMaxCodecOptions * kOptionCatWidth * sizeof(std::int64_t) +
           kMaxCodecOptions * kOptionNumWidth * sizeof(float) +
           kMaxCodecOptions * sizeof(std::int64_t) +
           kMaxCodecOptions * sizeof(std::uint8_t) +
           2 * sizeof(std::int64_t);
}

cudaError_t allocate_arena(DeviceArena* arena, const RuntimeConfig& config) {
    if (arena == nullptr || config.abi_version != kAbiVersion || config.batch_size == 0 ||
        config.policy_count == 0 || config.policy_count > kMaxPolicies ||
        config.route_capacity == 0) {
        return cudaErrorInvalidValue;
    }
    *arena = DeviceArena{};
    arena->config = config;
    std::size_t bytes = 0;

#define PTCG_ALLOC(field, count)                                                    \
    do {                                                                            \
        const cudaError_t alloc_status = allocate_array(&(field), (count), &bytes); \
        if (alloc_status != cudaSuccess) {                                           \
            free_arena(arena);                                                       \
            return alloc_status;                                                     \
        }                                                                            \
    } while (false)

    const std::size_t batch = config.batch_size;
    PTCG_ALLOC(arena->states, batch);
    PTCG_ALLOC(arena->reset_specs, batch);
    PTCG_ALLOC(arena->route_indices, static_cast<std::size_t>(config.policy_count) * config.route_capacity);
    PTCG_ALLOC(arena->route_counts, config.policy_count);
    PTCG_ALLOC(arena->digests, batch);
    PTCG_ALLOC(arena->codec.global_cat, batch * kGlobalCatWidth);
    PTCG_ALLOC(arena->codec.global_num, batch * kGlobalNumWidth);
    PTCG_ALLOC(arena->codec.entity_cat, batch * kMaxCodecEntities * kEntityCatWidth);
    PTCG_ALLOC(arena->codec.entity_num, batch * kMaxCodecEntities * kEntityNumWidth);
    PTCG_ALLOC(arena->codec.entity_parent, batch * kMaxCodecEntities);
    PTCG_ALLOC(arena->codec.entity_mask, batch * kMaxCodecEntities);
    PTCG_ALLOC(arena->codec.option_cat, batch * kMaxCodecOptions * kOptionCatWidth);
    PTCG_ALLOC(arena->codec.option_num, batch * kMaxCodecOptions * kOptionNumWidth);
    PTCG_ALLOC(arena->codec.option_equiv, batch * kMaxCodecOptions);
    PTCG_ALLOC(arena->codec.option_mask, batch * kMaxCodecOptions);
    PTCG_ALLOC(arena->codec.min_count, batch);
    PTCG_ALLOC(arena->codec.max_count, batch);

#undef PTCG_ALLOC

    arena->allocated_bytes = bytes;
    return cudaSuccess;
}

cudaError_t free_arena(DeviceArena* arena) {
    if (arena == nullptr) {
        return cudaErrorInvalidValue;
    }
    cudaError_t first_error = cudaSuccess;
#define PTCG_FREE(field)                                  \
    do {                                                   \
        if ((field) != nullptr) {                          \
            const cudaError_t status = cudaFree(field);    \
            if (first_error == cudaSuccess) {              \
                first_error = status;                      \
            }                                              \
            (field) = nullptr;                             \
        }                                                  \
    } while (false)

    PTCG_FREE(arena->states);
    PTCG_FREE(arena->reset_specs);
    PTCG_FREE(arena->instructions);
    PTCG_FREE(arena->actions);
    PTCG_FREE(arena->route_indices);
    PTCG_FREE(arena->route_counts);
    PTCG_FREE(arena->digests);
    PTCG_FREE(arena->codec.global_cat);
    PTCG_FREE(arena->codec.global_num);
    PTCG_FREE(arena->codec.entity_cat);
    PTCG_FREE(arena->codec.entity_num);
    PTCG_FREE(arena->codec.entity_parent);
    PTCG_FREE(arena->codec.entity_mask);
    PTCG_FREE(arena->codec.option_cat);
    PTCG_FREE(arena->codec.option_num);
    PTCG_FREE(arena->codec.option_equiv);
    PTCG_FREE(arena->codec.option_mask);
    PTCG_FREE(arena->codec.min_count);
    PTCG_FREE(arena->codec.max_count);

#undef PTCG_FREE
    *arena = DeviceArena{};
    return first_error;
}

cudaError_t upload_rule_pack(
    DeviceArena* arena,
    const Instruction* host_instructions,
    std::uint32_t instruction_count,
    const ActionDescriptor* host_actions,
    std::uint32_t action_count) {
    if (arena == nullptr || host_instructions == nullptr || instruction_count == 0 ||
        host_actions == nullptr || action_count == 0 || action_count > kMaxCodecOptions) {
        return cudaErrorInvalidValue;
    }
    if (arena->instructions != nullptr) {
        cudaFree(arena->instructions);
        arena->allocated_bytes -= sizeof(Instruction) * arena->instruction_count;
        arena->instructions = nullptr;
    }
    if (arena->actions != nullptr) {
        cudaFree(arena->actions);
        arena->allocated_bytes -= sizeof(ActionDescriptor) * arena->action_count;
        arena->actions = nullptr;
    }

    cudaError_t status = allocate_array(
        &arena->instructions, instruction_count, &arena->allocated_bytes);
    if (status != cudaSuccess) {
        return status;
    }
    status = allocate_array(&arena->actions, action_count, &arena->allocated_bytes);
    if (status != cudaSuccess) {
        cudaFree(arena->instructions);
        arena->instructions = nullptr;
        arena->allocated_bytes -= sizeof(Instruction) * instruction_count;
        return status;
    }
    status = cudaMemcpy(
        arena->instructions,
        host_instructions,
        sizeof(Instruction) * instruction_count,
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaMemcpy(
        arena->actions,
        host_actions,
        sizeof(ActionDescriptor) * action_count,
        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return status;
    }
    arena->instruction_count = instruction_count;
    arena->action_count = action_count;
    return cudaSuccess;
}

cudaError_t reset_from_host_async(
    DeviceArena* arena,
    const ResetSpec* host_specs,
    cudaStream_t stream) {
    if (arena == nullptr || host_specs == nullptr || arena->actions == nullptr) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = cudaMemcpyAsync(
        arena->reset_specs,
        host_specs,
        sizeof(ResetSpec) * arena->config.batch_size,
        cudaMemcpyHostToDevice,
        stream);
    if (status != cudaSuccess) {
        return status;
    }
    reset_kernel<<<blocks_for(arena->config.batch_size), kThreads, 0, stream>>>(
        arena->states,
        arena->reset_specs,
        arena->config.batch_size,
        arena->config.policy_count,
        arena->action_count);
    return cudaPeekAtLastError();
}

cudaError_t reset_from_device_async(
    DeviceArena* arena,
    const ResetSpec* device_specs,
    cudaStream_t stream) {
    if (arena == nullptr || device_specs == nullptr || arena->actions == nullptr) {
        return cudaErrorInvalidValue;
    }
    reset_kernel<<<blocks_for(arena->config.batch_size), kThreads, 0, stream>>>(
        arena->states,
        device_specs,
        arena->config.batch_size,
        arena->config.policy_count,
        arena->action_count);
    return cudaPeekAtLastError();
}

cudaError_t apply_actions_device_async(
    DeviceArena* arena,
    const std::int32_t* device_action_indices,
    cudaStream_t stream) {
    if (arena == nullptr || device_action_indices == nullptr || arena->instructions == nullptr ||
        arena->actions == nullptr) {
        return cudaErrorInvalidValue;
    }
    apply_actions_kernel<<<blocks_for(arena->config.batch_size), kThreads, 0, stream>>>(
        arena->states,
        arena->config.batch_size,
        device_action_indices,
        arena->instructions,
        arena->instruction_count,
        arena->actions,
        arena->action_count);
    return cudaPeekAtLastError();
}

cudaError_t encode_policy_v1_async(DeviceArena* arena, cudaStream_t stream) {
    if (arena == nullptr || arena->actions == nullptr) {
        return cudaErrorInvalidValue;
    }
    encode_policy_v1_kernel<<<blocks_for(arena->config.batch_size), kThreads, 0, stream>>>(
        arena->states,
        arena->config.batch_size,
        arena->actions,
        arena->action_count,
        arena->codec);
    return cudaPeekAtLastError();
}

cudaError_t route_ready_async(DeviceArena* arena, cudaStream_t stream) {
    if (arena == nullptr) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = cudaMemsetAsync(
        arena->route_counts,
        0,
        sizeof(std::uint32_t) * arena->config.policy_count,
        stream);
    if (status != cudaSuccess) {
        return status;
    }
    status = cudaMemsetAsync(
        arena->route_indices,
        0xFF,
        sizeof(std::int32_t) * arena->config.policy_count * arena->config.route_capacity,
        stream);
    if (status != cudaSuccess) {
        return status;
    }
    route_kernel<<<blocks_for(arena->config.batch_size), kThreads, 0, stream>>>(
        arena->states,
        arena->config.batch_size,
        arena->config.policy_count,
        arena->config.route_capacity,
        arena->route_counts,
        arena->route_indices);
    return cudaPeekAtLastError();
}

cudaError_t digest_async(DeviceArena* arena, cudaStream_t stream) {
    if (arena == nullptr) {
        return cudaErrorInvalidValue;
    }
    digest_kernel<<<blocks_for(arena->config.batch_size), kThreads, 0, stream>>>(
        arena->states,
        arena->config.batch_size,
        arena->digests);
    return cudaPeekAtLastError();
}

}  // namespace ptcg::cuda_engine
