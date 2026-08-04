#pragma once

#include <cstdint>

#include "ptcg_cuda/official_trigger_resolver_pod.cuh"

#if defined(__CUDACC__)
#define PTCG_OFFICIAL_KO_HD __host__ __device__
#else
#define PTCG_OFFICIAL_KO_HD
#endif

namespace ptcg::cuda_engine {

enum class OfficialKnockoutResult : std::int32_t {
    kComplete = 0,
    kNeedsAction = 1,
    kError = 2,
};

enum class OfficialKnockoutStage : std::uint8_t {
    kIdle = 0,
    kPreKoTriggers = 1,
    kPostKoTriggers = 2,
    kPrizeSelection = 3,
    kActiveReplacement = 4,
    kLuckyBonusChoice = 5,
    kActiveTriggers = 6,
};

constexpr std::uint8_t kOfficialSelectContextToActive = 5;
constexpr std::uint8_t kOfficialKoTriggerBenchToActive = 6;
constexpr std::uint8_t kOfficialSelectContextToHand = 8;
constexpr std::uint64_t kOfficialSkillLuckyBonusFlag = 1ULL << 7;
constexpr std::int32_t kOfficialDefaultBenchCapacity = 5;

PTCG_OFFICIAL_KO_HD inline OfficialKnockoutResult official_advance_knockout(
    OfficialStatePod* state,
    const OfficialRulePackView& rules);

template <std::size_t Capacity>
PTCG_OFFICIAL_KO_HD inline bool official_mark_ko_zone(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone) {
    for (std::int32_t index = static_cast<std::int32_t>(zone.count) - 1;
         index >= 0;
         --index) {
        OfficialCardStatePod* card = official_pod_card(state, zone.values[index]);
        if (card == nullptr) return false;
        const std::int32_t max_hp = official_pod_max_hp(state, rules, zone.values[index]);
        if (max_hp > 0 && card->damage >= max_hp) {
            official_pod_set_card_runtime_flag(card, kCardKo);
        }
        if ((card->runtime_flags & kCardKoAttackDamage) == 0) continue;
        if (!official_pull_trigger(
                state, rules, 12, zone.values[index], OfficialCardRefPod{}, 1)) {
            return false;
        }
        if ((card->runtime_flags & kCardKoFull) != 0
            && !official_pull_trigger(
                state, rules, 13, zone.values[index], OfficialCardRefPod{}, 1)) {
            return false;
        }
        if ((card->runtime_flags & kCardKoFull) != 0
            && (card->runtime_flags & kCardKoEnemyAttackDamage) != 0
            && !official_pull_trigger(
                state, rules, 14, zone.values[index], OfficialCardRefPod{}, 1)) {
            return false;
        }
    }
    return official_pod_ok(state);
}

PTCG_OFFICIAL_KO_HD inline bool official_mark_knockouts_and_pull_pre(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    const std::int32_t order[2] = {state->first_player, 1 - state->first_player};
    for (int index = 0; index < 2; ++index) {
        const int player = order[index];
        if (player < 0 || player > 1) continue;
        if (!official_mark_ko_zone(state, rules, state->players[player].active)
            || !official_mark_ko_zone(state, rules, state->players[player].bench)) {
            return false;
        }
    }
    return true;
}

template <std::size_t Capacity>
PTCG_OFFICIAL_KO_HD inline bool official_collect_ko_zone(
    OfficialStatePod* state,
    const OfficialPodList<OfficialCardRefPod, Capacity>& zone) {
    for (std::int32_t index = static_cast<std::int32_t>(zone.count) - 1;
         index >= 0;
         --index) {
        const OfficialCardRefPod ref = zone.values[index];
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        if (card != nullptr && (card->runtime_flags & kCardKo) != 0) {
            if (!official_pod_push(
                    state,
                    &state->ko_list,
                    official_pod_area_ref(state, ref),
                    OfficialPodError::kSelectionOverflow)) {
                return false;
            }
        }
    }
    return true;
}

PTCG_OFFICIAL_KO_HD inline bool official_collect_knockouts(
    OfficialStatePod* state) {
    state->ko_list.count = 0;
    const std::int32_t order[2] = {1 - state->first_player, state->first_player};
    for (int index = 0; index < 2; ++index) {
        const int player = order[index];
        if (player < 0 || player > 1) continue;
        if (!official_collect_ko_zone(state, state->players[player].bench)
            || !official_collect_ko_zone(state, state->players[player].active)) {
            return false;
        }
    }
    return true;
}

PTCG_OFFICIAL_KO_HD inline void official_rebuild_active_replacement_mask(
    OfficialStatePod* state) {
    state->pending_active_replacement_mask = 0;
    for (int player = 0; player < 2; ++player) {
        if (state->players[player].active.count == 0
            && state->players[player].bench.count > 0) {
            state->pending_active_replacement_mask |= static_cast<std::uint16_t>(
                1U << player);
        }
    }
}

PTCG_OFFICIAL_KO_HD inline void official_mark_active_check_changed(
    OfficialStatePod* state) {
    for (int player = 0; player < 2; ++player) {
        if (state->players[player].active.count == 0) {
            state->control_flags |= 1U << 4U;
            return;
        }
    }
}

PTCG_OFFICIAL_KO_HD inline bool official_pull_post_ko_triggers(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    for (std::uint16_t index = 0; index < state->ko_list.count; ++index) {
        const OfficialAreaRefPod ref = state->ko_list.values[index];
        if (!official_pod_area_ref_valid(state, ref)) continue;
        const OfficialCardStatePod* card = official_pod_card(state, ref.card);
        const OfficialCardRefPod cause = state->attacker;
        if (card == nullptr
            || !official_pull_trigger(state, rules, 15, ref.card, cause, 1)) {
            return false;
        }
        if ((card->runtime_flags & kCardKoEnemyAttackDamage) != 0
            && !official_pull_trigger(state, rules, 16, ref.card, cause, 1)) {
            return false;
        }
        if ((card->runtime_flags & kCardKoEnemyExAttackDamage) != 0
            && !official_pull_trigger(state, rules, 17, ref.card, cause, 1)) {
            return false;
        }
        if ((card->runtime_flags & kCardKoEnemyAttackDamageActive) != 0
            && !official_pull_trigger(state, rules, 18, ref.card, cause, 1)) {
            return false;
        }
        if ((card->runtime_flags & kCardKoNoDamageAndEffectAttackNextEnemyTurn) != 0) {
            OfficialCardStatePod* cause_card = official_pod_card(state, cause);
            if (cause_card != nullptr
                && cause_card->area == static_cast<std::uint8_t>(OfficialArea::kActive)) {
                cause_card->next_enemy_turn_end |= 1U << 24U;
            }
        }
    }
    return true;
}

PTCG_OFFICIAL_KO_HD inline bool official_move_knockouts_and_queue_prizes(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    state->pending_prize_count[0] = 0;
    state->pending_prize_count[1] = 0;
    state->prize_requests.count = 0;
    for (std::uint16_t index = 0; index < state->ko_list.count; ++index) {
        const OfficialAreaRefPod ref = state->ko_list.values[index];
        if (!official_pod_area_ref_valid(state, ref)) continue;
        const OfficialCardStatePod* card = official_pod_card(state, ref.card);
        const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
            rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) return false;
        const std::int32_t prize = official_pod_prize_count(state, rules, ref.card);
        if ((master->flags & kOfficialCardNoPrizeFlag) == 0 && prize > 0) {
            const OfficialPrizeRequestPod request{
                static_cast<std::int16_t>(prize),
                static_cast<std::int8_t>(1 - card->player),
                0};
            if (!official_pod_push(
                    state, &state->prize_requests, request,
                    OfficialPodError::kContinuationStackOverflow)) {
                return false;
            }
        }
        if (state->phase == static_cast<std::uint8_t>(OfficialGamePhase::kMain)
            && official_active_player(*state) != card->player) {
            state->turn_histories[0].flags |= kHistoryKo;
            if ((master->flags & (1ULL << 25U)) != 0) {
                state->turn_histories[0].flags |= kHistoryKoTeamRocket;
            }
            if ((card->runtime_flags & kCardKoEnemyAttackDamage) != 0) {
                state->turn_histories[0].flags |= kHistoryKoAttackDamage;
                if ((master->flags & (1ULL << 17U)) != 0) {
                    state->turn_histories[0].flags |= kHistoryKoAttackDamageEthan;
                }
                if ((master->flags & (1ULL << 13U)) != 0) {
                    state->turn_histories[0].flags |= kHistoryKoAttackDamageHop;
                }
            }
        }
    }
    for (std::int32_t index = static_cast<std::int32_t>(state->ko_list.count) - 1;
         index >= 0 && official_pod_ok(state);
         --index) {
        const OfficialAreaRefPod ref = state->ko_list.values[index];
        if (!official_pod_area_ref_valid(state, ref)) continue;
        const OfficialCardStatePod* card = official_pod_card(state, ref.card);
        const bool to_hand = card != nullptr
            && (card->runtime_flags & kCardKoByDamageToHand) != 0
            && (card->runtime_flags & kCardKoEnemyAttackDamage) != 0;
        official_pod_move_ref_complete(
            state,
            ref.card,
            to_hand ? OfficialArea::kHand : OfficialArea::kTrash,
            false,
            false);
    }
    state->ko_list.count = 0;
    official_rebuild_active_replacement_mask(state);
    return official_pod_ok(state);
}

PTCG_OFFICIAL_KO_HD inline OfficialKnockoutResult official_begin_prize_selection(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    std::int32_t player) {
    OfficialPlayerStatePod& ps = state->players[player];
    const std::int32_t pending = state->pending_prize_count[player];
    if (pending <= 0 || ps.prize.count == 0) {
        state->pending_prize_count[player] = 0;
        return OfficialKnockoutResult::kComplete;
    }
    const std::int32_t count = pending < ps.prize.count ? pending : ps.prize.count;
    official_clear_effect_selection(state);
    state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kCard);
    state->select_context = kOfficialSelectContextToHand;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = static_cast<std::int16_t>(count);
    state->select_max = static_cast<std::int16_t>(count);
    for (std::uint16_t index = 0; index < ps.prize.count; ++index) {
        const OfficialCardRefPod ref = ps.prize.values[index];
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
            rules, static_cast<std::uint32_t>(card->card_id));
        if (master == nullptr) return OfficialKnockoutResult::kError;
        OfficialSelectOptionPod option{};
        option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kCard);
        option.params[0] = static_cast<std::int16_t>(OfficialArea::kPrize);
        option.params[1] = static_cast<std::int16_t>(index);
        option.params[2] = static_cast<std::int16_t>(player);
        option.resolved_card = ref.index;
        option.option_equiv = static_cast<std::uint16_t>(card->card_id);
        if (!official_pod_push(
                state, &state->options, option, OfficialPodError::kOptionOverflow)) {
            return OfficialKnockoutResult::kError;
        }
    }
    return OfficialKnockoutResult::kNeedsAction;
}

PTCG_OFFICIAL_KO_HD inline std::int32_t official_ko_bench_capacity(
    const OfficialPlayerStatePod& player) {
    const std::int32_t value = static_cast<std::int32_t>(
        (player.continual_state >> 40U) & 0xfULL);
    return value == 0 ? kOfficialDefaultBenchCapacity : value;
}

PTCG_OFFICIAL_KO_HD inline bool official_lucky_bonus_eligible(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    OfficialCardRefPod ref) {
    const OfficialCardStatePod* card = official_pod_card(state, ref);
    const OfficialCardRule* master = card == nullptr ? nullptr : official_card_rule(
        rules, static_cast<std::uint32_t>(card->card_id));
    if (master == nullptr || card->player < 0 || card->player > 1) return false;
    const std::int32_t ability_id = master->values[kCardAbilityId];
    const OfficialSkillRule* ability = ability_id > 0
        ? official_skill_rule(rules, static_cast<std::uint32_t>(ability_id))
        : nullptr;
    const auto& player = state->players[card->player];
    return card->reverse != 0
        && ability != nullptr
        && (ability->flags & kOfficialSkillLuckyBonusFlag) != 0
        && static_cast<std::int32_t>(player.bench.count)
            < official_ko_bench_capacity(player)
        && state->phase == static_cast<std::uint8_t>(OfficialGamePhase::kMain)
        && official_active_player(*state) == card->player;
}

PTCG_OFFICIAL_KO_HD inline bool official_move_prize_to_temporary_without_state_change(
    OfficialStatePod* state,
    OfficialCardRefPod ref) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr || card->player < 0 || card->player > 1
        || card->area != static_cast<std::uint8_t>(OfficialArea::kPrize)) return false;
    OfficialPlayerStatePod& player = state->players[card->player];
    const std::int32_t index = official_pod_find_in_list(player.prize, ref);
    if (index < 0) return false;
    official_pod_remove(state, &player.prize, static_cast<std::uint16_t>(index));
    return official_pod_push(
        state, &player.temporary, ref, OfficialPodError::kZoneOverflow);
}

PTCG_OFFICIAL_KO_HD inline bool official_move_temporary_lucky_bonus(
    OfficialStatePod* state,
    OfficialCardRefPod ref,
    OfficialArea destination) {
    OfficialCardStatePod* card = official_pod_card(state, ref);
    if (card == nullptr || card->player < 0 || card->player > 1) return false;
    OfficialPlayerStatePod& player = state->players[card->player];
    const std::int32_t index = official_pod_find_in_list(player.temporary, ref);
    if (index < 0) return false;
    official_pod_remove(state, &player.temporary, static_cast<std::uint16_t>(index));
    if (!official_pod_push_zone_card(state, card->player, destination, ref)) return false;
    official_pod_card_moved(state, ref, destination, false);
    return official_pod_ok(state);
}

PTCG_OFFICIAL_KO_HD inline OfficialKnockoutResult official_begin_next_lucky_bonus(
    OfficialStatePod* state) {
    while (state->selected_list.count > 0) {
        const OfficialCardRefPod ref = state->selected_list.values[0];
        OfficialCardStatePod* card = official_pod_card(state, ref);
        if (card == nullptr || card->player < 0 || card->player > 1) {
            return OfficialKnockoutResult::kError;
        }
        const auto& player = state->players[card->player];
        if (static_cast<std::int32_t>(player.bench.count)
            >= official_ko_bench_capacity(player)) {
            official_pod_remove(state, &state->selected_list, 0);
            if (!official_move_temporary_lucky_bonus(
                    state, ref, OfficialArea::kHand)) {
                official_pod_fail(state, OfficialPodError::kInvalidAction, ref.index);
                return OfficialKnockoutResult::kError;
            }
            continue;
        }
        official_clear_effect_selection(state);
        state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kYesNo);
        state->select_context = kOfficialSelectContextActivate;
        state->select_player = card->player;
        state->select_min = 1;
        state->select_max = 1;
        state->context_card = ref;
        OfficialSelectOptionPod yes{};
        yes.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kYes);
        OfficialSelectOptionPod no{};
        no.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kNo);
        if (!official_pod_push(
                state, &state->options, yes, OfficialPodError::kOptionOverflow)
            || !official_pod_push(
                state, &state->options, no, OfficialPodError::kOptionOverflow)) {
            return OfficialKnockoutResult::kError;
        }
        state->reserved_flow = static_cast<std::uint8_t>(
            OfficialKnockoutStage::kLuckyBonusChoice);
        return OfficialKnockoutResult::kNeedsAction;
    }
    state->reserved_flow = static_cast<std::uint8_t>(
        OfficialKnockoutStage::kPrizeSelection);
    return OfficialKnockoutResult::kComplete;
}

PTCG_OFFICIAL_KO_HD inline OfficialKnockoutResult official_begin_active_replacement(
    OfficialStatePod* state,
    std::int32_t player) {
    OfficialPlayerStatePod& ps = state->players[player];
    if ((state->pending_active_replacement_mask & (1U << player)) == 0
        || ps.active.count != 0 || ps.bench.count == 0) {
        state->pending_active_replacement_mask &= static_cast<std::uint16_t>(~(1U << player));
        return OfficialKnockoutResult::kComplete;
    }
    official_clear_effect_selection(state);
    state->select_type = static_cast<std::uint8_t>(OfficialSelectTypeId::kCard);
    state->select_context = kOfficialSelectContextToActive;
    state->select_player = static_cast<std::int8_t>(player);
    state->select_min = 1;
    state->select_max = 1;
    for (std::uint16_t index = 0; index < ps.bench.count; ++index) {
        const OfficialCardRefPod ref = ps.bench.values[index];
        OfficialSelectOptionPod option{};
        option.type = static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kCard);
        option.params[0] = static_cast<std::int16_t>(OfficialArea::kBench);
        option.params[1] = static_cast<std::int16_t>(index);
        option.params[2] = static_cast<std::int16_t>(player);
        option.resolved_card = ref.index;
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        if (card == nullptr) return OfficialKnockoutResult::kError;
        option.option_equiv = static_cast<std::uint16_t>(card->card_id);
        if (!official_pod_push(
                state, &state->options, option, OfficialPodError::kOptionOverflow)) {
            return OfficialKnockoutResult::kError;
        }
    }
    return OfficialKnockoutResult::kNeedsAction;
}

PTCG_OFFICIAL_KO_HD inline OfficialKnockoutResult official_advance_knockout(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pod_ok(state)) return OfficialKnockoutResult::kError;
    while (true) {
        const OfficialKnockoutStage stage = static_cast<OfficialKnockoutStage>(
            state->reserved_flow);
        if (stage == OfficialKnockoutStage::kPreKoTriggers) {
            if (!official_collect_knockouts(state)) return OfficialKnockoutResult::kError;
            if (state->ko_list.count == 0) {
                official_rebuild_active_replacement_mask(state);
                state->reserved_flow = static_cast<std::uint8_t>(
                    OfficialKnockoutStage::kActiveReplacement);
                continue;
            }
            state->control_flags |= 1U << 4U;
            if (!official_pull_post_ko_triggers(state, rules)) {
                return OfficialKnockoutResult::kError;
            }
            state->reserved_flow = static_cast<std::uint8_t>(
                OfficialKnockoutStage::kPostKoTriggers);
            const OfficialTriggerResolverResult triggers =
                official_begin_trigger_resolution(state, rules, 1);
            if (triggers == OfficialTriggerResolverResult::kError) {
                return OfficialKnockoutResult::kError;
            }
            if (triggers == OfficialTriggerResolverResult::kNeedsAction) {
                return OfficialKnockoutResult::kNeedsAction;
            }
            continue;
        }
        if (stage == OfficialKnockoutStage::kPostKoTriggers) {
            if (!official_move_knockouts_and_queue_prizes(state, rules)) {
                return OfficialKnockoutResult::kError;
            }
            state->reserved_flow = static_cast<std::uint8_t>(
                OfficialKnockoutStage::kPrizeSelection);
            continue;
        }
        if (stage == OfficialKnockoutStage::kPrizeSelection) {
            for (int player = 0; player < 2; ++player) {
                if (state->pending_prize_count[player] <= 0) continue;
                const OfficialKnockoutResult pending = official_begin_prize_selection(
                    state, rules, player);
                if (pending == OfficialKnockoutResult::kNeedsAction) return pending;
                state->pending_prize_count[player] = 0;
            }
            if (state->selected_list.count > 0) {
                state->reserved_flow = static_cast<std::uint8_t>(
                    OfficialKnockoutStage::kLuckyBonusChoice);
                const OfficialKnockoutResult lucky = official_begin_next_lucky_bonus(state);
                if (lucky != OfficialKnockoutResult::kComplete) return lucky;
                continue;
            }
            while (state->prize_requests.count > 0) {
                const OfficialPrizeRequestPod request = official_pod_pop_back(
                    state, &state->prize_requests);
                if (!official_pod_ok(state)) return OfficialKnockoutResult::kError;
                if (request.player < 0 || request.player > 1 || request.count <= 0) {
                    official_pod_fail(
                        state, OfficialPodError::kInvalidAction, request.player);
                    return OfficialKnockoutResult::kError;
                }
                state->pending_prize_count[request.player] = request.count;
                const OfficialKnockoutResult request_result =
                    official_begin_prize_selection(state, rules, request.player);
                if (request_result == OfficialKnockoutResult::kNeedsAction) {
                    return request_result;
                }
                state->pending_prize_count[request.player] = 0;
            }
            // Official KOProc3 returns after prize handling. AfterRefresh then
            // starts a new Refresh pass. Bench/tool overflow is resolved in
            // that pass before AfterRefresh performs the finish check.
            state->reserved_flow = 0;
            return OfficialKnockoutResult::kComplete;
        }
        if (stage == OfficialKnockoutStage::kLuckyBonusChoice) {
            return official_begin_next_lucky_bonus(state);
        }
        if (stage == OfficialKnockoutStage::kActiveReplacement) {
            // ActiveCheck calls finishCheck before it queues any
            // SelectActivePokemon callbacks.  This matters after simultaneous
            // knockouts: a player that has already lost must not be offered an
            // otherwise-valid replacement selection for the opponent.
            if (official_pod_finish_check(state)) {
                state->reserved_flow = 0;
                return OfficialKnockoutResult::kComplete;
            }
            official_mark_active_check_changed(state);
            const std::int32_t active = official_active_player(*state);
            const std::int32_t order[2] = {1 - active, active};
            for (int index = 0; index < 2; ++index) {
                const int player = order[index];
                if (player < 0 || player > 1
                    || (state->pending_active_replacement_mask & (1U << player)) == 0) {
                    continue;
                }
                return official_begin_active_replacement(state, player);
            }
            // ActiveCheck pushes ResolveTriggerStack(0) after any required
            // active replacements. This activates shallower triggers left by
            // KOProc's depth-1 resolver.
            state->reserved_flow = static_cast<std::uint8_t>(
                OfficialKnockoutStage::kActiveTriggers);
            const OfficialTriggerResolverResult triggers =
                official_begin_trigger_resolution(state, rules, 0);
            if (triggers == OfficialTriggerResolverResult::kError) {
                return OfficialKnockoutResult::kError;
            }
            if (triggers == OfficialTriggerResolverResult::kNeedsAction) {
                return OfficialKnockoutResult::kNeedsAction;
            }
            continue;
        }
        if (stage == OfficialKnockoutStage::kActiveTriggers) {
            state->reserved_flow = 0;
            return OfficialKnockoutResult::kComplete;
        }
        official_pod_fail(
            state, OfficialPodError::kInvalidAction, state->reserved_flow);
        return OfficialKnockoutResult::kError;
    }
}

PTCG_OFFICIAL_KO_HD inline OfficialKnockoutResult official_begin_knockout(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    if (!official_pod_ok(state) || state->reserved_flow != 0
        || state->trigger_resolver.active != 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, state->reserved_flow);
        }
        return OfficialKnockoutResult::kError;
    }
    state->ko_list.count = 0;
    state->pending_prize_count[0] = 0;
    state->pending_prize_count[1] = 0;
    state->prize_requests.count = 0;
    state->pending_active_replacement_mask = 0;
    if (!official_mark_knockouts_and_pull_pre(state, rules)) {
        return OfficialKnockoutResult::kError;
    }
    state->reserved_flow = static_cast<std::uint8_t>(
        OfficialKnockoutStage::kPreKoTriggers);
    const OfficialTriggerResolverResult triggers = official_begin_trigger_resolution(
        state, rules, 1);
    if (triggers == OfficialTriggerResolverResult::kError) {
        return OfficialKnockoutResult::kError;
    }
    if (triggers == OfficialTriggerResolverResult::kNeedsAction) {
        return OfficialKnockoutResult::kNeedsAction;
    }
    return official_advance_knockout(state, rules);
}

PTCG_OFFICIAL_KO_HD inline bool official_validate_ko_action(
    OfficialStatePod* state,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (count < state->select_min || count > state->select_max) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        return false;
    }
    for (std::uint16_t index = 0; index < count; ++index) {
        if (option_indices[index] >= state->options.count) {
            official_pod_fail(
                state, OfficialPodError::kInvalidAction, option_indices[index]);
            return false;
        }
        for (std::uint16_t prior = 0; prior < index; ++prior) {
            if (option_indices[prior] == option_indices[index]) {
                official_pod_fail(
                    state, OfficialPodError::kInvalidAction, option_indices[index]);
                return false;
            }
        }
    }
    return true;
}

PTCG_OFFICIAL_KO_HD inline OfficialKnockoutResult official_resume_knockout(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    const std::uint16_t* option_indices,
    std::uint16_t count) {
    if (!official_pod_ok(state) || state->reserved_flow == 0) {
        if (official_pod_ok(state)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
        }
        return OfficialKnockoutResult::kError;
    }
    if (state->trigger_resolver.active != 0) {
        const OfficialTriggerResolverResult trigger = official_resume_trigger_resolution(
            state, rules, option_indices, count);
        if (trigger == OfficialTriggerResolverResult::kError) {
            return OfficialKnockoutResult::kError;
        }
        if (trigger == OfficialTriggerResolverResult::kNeedsAction) {
            return OfficialKnockoutResult::kNeedsAction;
        }
        return official_advance_knockout(state, rules);
    }
    if (!official_validate_ko_action(state, option_indices, count)) {
        return OfficialKnockoutResult::kError;
    }
    const OfficialKnockoutStage stage = static_cast<OfficialKnockoutStage>(
        state->reserved_flow);
    if (stage == OfficialKnockoutStage::kPrizeSelection) {
        const std::int32_t player = state->select_player;
        state->targets.count = 0;
        for (std::uint16_t index = 0; index < count; ++index) {
            const OfficialCardRefPod ref{
                state->options.values[option_indices[index]].resolved_card};
            if (!official_pod_push(
                    state,
                    &state->targets,
                    official_pod_area_ref(state, ref),
                    OfficialPodError::kSelectionOverflow)) {
                return OfficialKnockoutResult::kError;
            }
        }
        for (std::int32_t index = static_cast<std::int32_t>(count) - 1;
             index >= 0;
             --index) {
            const OfficialCardRefPod ref{
                state->options.values[option_indices[index]].resolved_card};
            const OfficialCardStatePod* card = official_pod_card(state, ref);
            if (card == nullptr || card->player != player
                || card->area != static_cast<std::uint8_t>(OfficialArea::kPrize)) {
                official_pod_fail(state, OfficialPodError::kInvalidAction, ref.index);
                return OfficialKnockoutResult::kError;
            }
            if (official_lucky_bonus_eligible(state, rules, ref)) {
                if (!official_move_prize_to_temporary_without_state_change(state, ref)
                    || !official_pod_push_front(
                        state, &state->selected_list, ref,
                        OfficialPodError::kSelectionOverflow)) {
                    return OfficialKnockoutResult::kError;
                }
            }
        }
        for (std::uint16_t index = 0; index < count; ++index) {
            const OfficialCardRefPod ref{
                state->options.values[option_indices[index]].resolved_card};
            if (official_pod_find_in_list(state->selected_list, ref) >= 0) {
                continue;
            }
            if (official_pod_move_ref(
                    state, ref, OfficialArea::kHand, false).index == 0) {
                official_pod_fail(state, OfficialPodError::kInvalidAction, ref.index);
                return OfficialKnockoutResult::kError;
            }
        }
        state->pending_prize_count[player] = 0;
        if (state->selected_list.count > 0) {
            official_clear_effect_selection(state);
            state->reserved_flow = static_cast<std::uint8_t>(
                OfficialKnockoutStage::kLuckyBonusChoice);
            return official_begin_next_lucky_bonus(state);
        }
    } else if (stage == OfficialKnockoutStage::kLuckyBonusChoice) {
        if (count != 1 || option_indices[0] >= state->options.count
            || state->selected_list.count == 0) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
            return OfficialKnockoutResult::kError;
        }
        const OfficialSelectOptionPod chosen = state->options.values[option_indices[0]];
        const bool yes = chosen.type
            == static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kYes);
        const bool no = chosen.type
            == static_cast<std::uint8_t>(OfficialSelectOptionTypeId::kNo);
        if (!yes && !no) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, chosen.type);
            return OfficialKnockoutResult::kError;
        }
        const OfficialCardRefPod ref = state->selected_list.values[0];
        const OfficialCardStatePod* card = official_pod_card(state, ref);
        const std::int32_t player = card == nullptr ? -1 : card->player;
        official_pod_remove(state, &state->selected_list, 0);
        if (player < 0 || player > 1
            || !official_move_temporary_lucky_bonus(
                state, ref, yes ? OfficialArea::kBench : OfficialArea::kHand)) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, ref.index);
            return OfficialKnockoutResult::kError;
        }
        if (yes) {
            state->coin_head_count = 0;
            if (official_pod_coin(state)) state->pending_prize_count[player] = 1;
        }
        official_clear_effect_selection(state);
        if (state->pending_prize_count[player] > 0) {
            state->reserved_flow = static_cast<std::uint8_t>(
                OfficialKnockoutStage::kPrizeSelection);
            const OfficialKnockoutResult extra = official_begin_prize_selection(
                state, rules, player);
            if (extra == OfficialKnockoutResult::kNeedsAction) return extra;
            state->pending_prize_count[player] = 0;
        }
        const OfficialKnockoutResult next = official_begin_next_lucky_bonus(state);
        if (next != OfficialKnockoutResult::kComplete) return next;
    } else if (stage == OfficialKnockoutStage::kActiveReplacement) {
        if (count != 1) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, count);
            return OfficialKnockoutResult::kError;
        }
        const std::int32_t player = state->select_player;
        const OfficialCardRefPod ref{
            state->options.values[option_indices[0]].resolved_card};
        const std::int32_t bench_index = official_pod_find_in_list(
            state->players[player].bench, ref);
        if (bench_index < 0
            || !official_switch_pokemon(
                state, rules, player, static_cast<std::uint16_t>(bench_index))) {
            official_pod_fail(state, OfficialPodError::kInvalidAction, ref.index);
            return OfficialKnockoutResult::kError;
        }
        state->pending_active_replacement_mask &= static_cast<std::uint16_t>(
            ~(1U << player));
    } else {
        official_pod_fail(state, OfficialPodError::kInvalidAction, state->reserved_flow);
        return OfficialKnockoutResult::kError;
    }
    official_clear_effect_selection(state);
    return official_advance_knockout(state, rules);
}

}  // namespace ptcg::cuda_engine

#undef PTCG_OFFICIAL_KO_HD
