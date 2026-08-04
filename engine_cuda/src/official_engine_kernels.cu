#include "ptcg_cuda/official_runtime.h"

#include "ptcg_cuda/official_flow_dispatch_pod.cuh"
#include "ptcg_cuda/official_seeded_setup_pod.cuh"

#include <cuda_runtime.h>

#include <cstring>

namespace ptcg::cuda_engine {
namespace {

constexpr std::uint32_t kOfficialThreads = 128;

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

template <typename DeckT>
__global__ void reset_official_states_seeded_first_min_kernel(
    OfficialStatePod* states,
    std::uint32_t batch_size,
    const std::uint8_t* rule_pack,
    const DeckT* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    std::uint8_t* statuses) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size || (lane_mask != nullptr && lane_mask[env] == 0)) return;

    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
    OfficialStatePod* state = &states[env];
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
    std::uint8_t* statuses) {
    const std::uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= batch_size || (lane_mask != nullptr && lane_mask[env] == 0)) return;

    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
    OfficialStatePod* state = &states[env];
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
    if (state->abi_version != kOfficialStateAbiVersion) {
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

    const OfficialCardRule* master = official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    const std::int32_t max_hp = master == nullptr
        ? 0
        : ((master->values[kCardHp] + card->hp_change) > 0
            ? (master->values[kCardHp] + card->hp_change)
            : 0);
    const std::int32_t damage = card->damage > 0 ? card->damage : 0;
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
    auto* option_cat = codec.option_cat + env * kMaxCodecOptions * kOptionCatWidth;
    auto* option_num = codec.option_num + env * kMaxCodecOptions * kOptionNumWidth;
    auto* option_equiv = codec.option_equiv + env * kMaxCodecOptions;
    auto* option_mask = codec.option_mask + env * kMaxCodecOptions;

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
    for (std::uint32_t index = 0; index < kMaxCodecOptions; ++index) {
        option_equiv[index] = -1;
        option_mask[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxCodecOptions * kOptionCatWidth; ++index) {
        option_cat[index] = 0;
    }
    for (std::uint32_t index = 0; index < kMaxCodecOptions * kOptionNumWidth; ++index) {
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

    if (state->options.count > kMaxCodecOptions) {
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
        std::int64_t group = option_index;
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
        batch * kMaxCodecOptions * kOptionCatWidth,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.option_num,
        batch * kMaxCodecOptions * kOptionNumWidth,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.option_equiv,
        batch * kMaxCodecOptions,
        &arena->allocated_bytes);
    if (status != cudaSuccess) {
        free_official_arena(arena);
        return status;
    }
    status = official_allocate(
        &arena->codec.option_mask,
        batch * kMaxCodecOptions,
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

template <typename DeckT>
cudaError_t reset_official_states_seeded_first_min_async_impl(
    OfficialDeviceArena* arena,
    const DeckT* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
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
        arena->statuses);
    return cudaGetLastError();
}

cudaError_t reset_official_states_seeded_first_min_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_first_min_async_impl(
        arena, decks, seeds, lane_mask, stream);
}

cudaError_t reset_official_states_seeded_first_min_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_first_min_async_impl(
        arena, decks, seeds, lane_mask, stream);
}

template <typename DeckT>
cudaError_t reset_official_states_seeded_interactive_async_impl(
    OfficialDeviceArena* arena,
    const DeckT* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
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
        arena->statuses);
    return cudaGetLastError();
}

cudaError_t reset_official_states_seeded_interactive_i32_async(
    OfficialDeviceArena* arena,
    const std::int32_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_interactive_async_impl(
        arena, decks, seeds, lane_mask, stream);
}

cudaError_t reset_official_states_seeded_interactive_i64_async(
    OfficialDeviceArena* arena,
    const std::int64_t* decks,
    const std::int64_t* seeds,
    const std::uint8_t* lane_mask,
    cudaStream_t stream) {
    return reset_official_states_seeded_interactive_async_impl(
        arena, decks, seeds, lane_mask, stream);
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
