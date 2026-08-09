#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define PTCG_OFFICIAL_BATTLE_CUDA_PAIRED_LIBRARY
#include "official_battle_end_turn_cuda_paired.cu"
#include "ptcg_cuda/testing/random_outcome_fixture.cuh"

namespace {

using ptcg::cuda_engine::testing::RandomOutcomeFixture;
using ptcg::cuda_engine::testing::RandomOutcomeCursor;
using ptcg::cuda_engine::testing::apply_shuffle_outcome;
using ptcg::cuda_engine::testing::consume_coin_outcome;
using ptcg::cuda_engine::testing::consume_effect_outcome;
using ptcg::cuda_engine::testing::consume_target_outcome;
using ptcg::cuda_engine::testing::random_outcome_fully_consumed;
using ptcg::cuda_engine::testing::validate_random_outcome_fixture;

enum class ContinueOperation : std::uint8_t { kDraw = 0, kDeckToPrize = 1 };

struct AttackOutcomeEvidence {
    std::uint32_t rng_seed = 0;
    std::int32_t observed_effect_result = 0;
    std::uint16_t observed_target = 0;
};

__global__ void apply_fixture_and_continue(
    OfficialStatePod* state,
    RandomOutcomeFixture fixture,
    ContinueOperation operation) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    OfficialPlayerStatePod* player = &state->players[0];
    if (!apply_shuffle_outcome(player->deck.values, player->deck.count, fixture)) {
        official_pod_fail(state, OfficialPodError::kInvalidAction, 9001);
        return;
    }
    if (operation == ContinueOperation::kDraw) {
        official_pod_draw(state, 0, 1);
    } else {
        for (std::uint16_t index = 0; index < fixture.prize_count; ++index) {
            if (player->deck.count == 0) break;
            official_pod_move_card(
                state, 0, OfficialArea::kDeck,
                static_cast<std::uint16_t>(player->deck.count - 1),
                OfficialArea::kPrize, true);
        }
    }
}

RandomOutcomeFixture reverse_fixture(std::uint16_t count, std::uint16_t prize_count) {
    RandomOutcomeFixture fixture{};
    fixture.shuffle_count = count;
    fixture.prize_count = prize_count;
    fixture.coin_count = 2;
    fixture.coin_results[0] = 1;
    fixture.coin_results[1] = 0;
    fixture.target_count = 1;
    fixture.random_target_indices[0] = 2;
    fixture.effect_count = 1;
    fixture.effect_results[0] = 7;
    for (std::uint16_t index = 0; index < count; ++index) {
        fixture.shuffle_permutation[index] = static_cast<std::uint16_t>(count - index - 1);
    }
    for (std::uint16_t index = 0; index < prize_count; ++index) fixture.prize_indices[index] = index;
    if (!validate_random_outcome_fixture(fixture)) throw std::runtime_error("fixture validation failed");
    return fixture;
}

void apply_official_shuffle(State* state, const RandomOutcomeFixture& fixture) {
    ::PlayerState& player = state->players[0];
    if (player.deck.size() != fixture.shuffle_count) throw std::runtime_error("official deck size mismatch");
    std::vector<CardRef> original(player.deck.begin(), player.deck.end());
    for (std::size_t index = 0; index < original.size(); ++index) {
        player.deck[index] = original[fixture.shuffle_permutation[index]];
    }
}

void run_case(
    const std::vector<std::uint16_t>& deck0,
    const std::vector<std::uint16_t>& deck1,
    std::uint64_t seed,
    ContinueOperation operation) {
    ApiData official;
    initialize_official_battle(&official, deck0, deck1, seed, false);
    const std::uint64_t rng_draws = count_rng_draws(seed, official.game.rng);
    OfficialStatePod initial{};
    const OfficialBridgeResult bridge = bridge_official_state(
        official.state, OfficialStateBridgeContext{seed, rng_draws, true}, &initial);
    if (!bridge) throw std::runtime_error("initial bridge failed");
    const std::uint16_t prize_count = operation == ContinueOperation::kDeckToPrize ? 2 : 0;
    const RandomOutcomeFixture fixture = reverse_fixture(initial.players[0].deck.count, prize_count);
    const std::vector<CardRef> original_deck(
        official.state.players[0].deck.begin(), official.state.players[0].deck.end());
    const std::size_t original_prize_count = official.state.players[0].prize.size();
    for (std::uint16_t ordinal = 0; ordinal < fixture.prize_count; ++ordinal) {
        const std::uint16_t top = static_cast<std::uint16_t>(
            fixture.shuffle_count - 1 - ordinal);
        if (fixture.shuffle_permutation[top] != fixture.prize_indices[ordinal]) {
            throw std::runtime_error("prize index is not bound to the shuffled deck top");
        }
    }
    apply_official_shuffle(&official.state, fixture);
    if (operation == ContinueOperation::kDraw) Draw(official.state, 0, 1);
    else DeckToPrize(official.state, 0, prize_count);
    if (operation == ContinueOperation::kDeckToPrize) {
        for (std::uint16_t ordinal = 0; ordinal < fixture.prize_count; ++ordinal) {
            const CardRef expected = original_deck[fixture.prize_indices[ordinal]];
            const CardRef actual = official.state.players[0].prize[
                original_prize_count + ordinal];
            if (actual.cardIndex != expected.cardIndex) {
                throw std::runtime_error("official Prize index fixture was not consumed");
            }
        }
    }

    OfficialStatePod* device = nullptr;
    check_cuda(cudaMalloc(&device, sizeof(OfficialStatePod)), "fixture cudaMalloc");
    check_cuda(cudaMemcpy(device, &initial, sizeof(initial), cudaMemcpyHostToDevice), "fixture H2D");
    apply_fixture_and_continue<<<1, 1>>>(device, fixture, operation);
    check_cuda(cudaGetLastError(), "fixture kernel launch");
    OfficialStatePod actual{};
    check_cuda(cudaMemcpy(&actual, device, sizeof(actual), cudaMemcpyDeviceToHost), "fixture D2H");
    check_cuda(cudaFree(device), "fixture cudaFree");

    OfficialStatePod expected{};
    const OfficialBridgeResult post_bridge = bridge_official_state(
        official.state, OfficialStateBridgeContext{seed, rng_draws, true}, &expected);
    if (!post_bridge) throw std::runtime_error("post bridge failed");
    require_state_equal(seed, static_cast<std::uint32_t>(operation), "explicit_random_outcome",
                        official.state, expected, actual);
}

void clear_official_attack_state(State* state) {
    for (int player = 0; player < 2; ++player) {
        ::PlayerState& ps = state->players[player];
        ps.active.clear();
        ps.bench.clear();
        ps.prize.clear();
        ps.hand.clear();
        ps.deck.clear();
        ps.trash.clear();
        ps.energy.clear();
        ps.tool.clear();
        ps.preEvolution.clear();
        ps.temporary.clear();
        ps.playerIndex = static_cast<signed char>(player);
        ps.koPrizeOnceChanged = false;
        ps.activeState = 0;
        ps.continualState = 0;
        ps.turnState = 0;
        ps.thisTurn.value = 0;
        ps.nextTurn.value = 0;
        ps.badStatus = BadStatusType::None;
    }
    state->stadium.clear();
    state->looking.clear();
    state->selectedList.clear();
    state->eachList.clear();
    state->playing.clear();
    state->checkList.clear();
    state->options.clear();
    state->selected.clear();
    state->preTargetList.clear();
    state->targetList.clear();
    state->koList.clear();
    state->delayTriggerStack.clear();
    state->temporaryTriggerStack.clear();
    state->triggerStack.clear();
    state->turnUsedSkill.clear();
    state->turnPlay.clear();
    state->turnHeal.clear();
    state->turnEvolve.clear();
    state->functionStack.clear();
    state->logs.clear();
    state->allCard = {};
    state->selectType = SelectType::None;
    state->selectContext = SelectContext::None;
    state->selectPlayer = -1;
    state->firstPlayer = 0;
    state->lastStadiumPlayer = 0;
    state->turn = 3;
    state->turnActionCount = 0;
    state->turnState = 0;
    state->phase = GamePhase::Main;
    state->gameResult = GameResult::None;
    state->finishReason = FinishReason::None;
    state->moveCounter = 1;
    state->attacker = {};
    state->currentAttackId = 0;
    state->srcAttackId = 0;
    state->coinHeadCount = 0;
    state->attackDamageChange = 0;
    state->lastAttackDamage = 0;
    state->turnAttackCount = 0;
    state->secondAttack = false;
    state->failAttack = false;
    state->postAttackEffect = false;
    state->postEffectActivate = false;
}

void setup_official_attack_battle(BattleData* battle, std::uint32_t seed) {
    GameConfig config{};
    config.seed = seed;
    config.recordLog = true;
    config.deviceRand = false;
    for (int player = 0; player < 2; ++player) {
        for (int index = 0; index < DECK_SIZE; ++index) {
            config.decks[player].cards[index] = 1;
        }
    }
    battle->init(config, false);
    battle->game.rng = std::mt19937(seed);
    clear_official_attack_state(&battle->state);
}

void place_official_attack_card(
    State* state,
    int ref_value,
    int card_id,
    int player,
    AreaType area) {
    Card& card = state->getCard(CardRef(ref_value));
    card = {};
    card.init(card_id, state->moveCounter++, player);
    card.area = area;
    ::PlayerState& ps = state->players[player];
    switch (area) {
        case AreaType::Deck: ps.deck.push_back(CardRef(ref_value)); break;
        case AreaType::Hand: ps.hand.push_back(CardRef(ref_value)); break;
        case AreaType::Active: ps.active.push_back(CardRef(ref_value)); break;
        case AreaType::Bench: ps.bench.push_back(CardRef(ref_value)); break;
        case AreaType::Prize: ps.prize.push_back(CardRef(ref_value)); break;
        case AreaType::Energy: ps.energy.push_back(CardRef(ref_value)); break;
        default: throw std::runtime_error("unsupported attack fixture area");
    }
}

void prepare_official_attack_case(
    State* state,
    int attacker_card_id,
    int target_card_id,
    int energy_count) {
    place_official_attack_card(state, 10, attacker_card_id, 0, AreaType::Active);
    for (int index = 0; index < energy_count; ++index) {
        place_official_attack_card(state, 11 + index, 3, 0, AreaType::Energy);
        state->getCard(CardRef(11 + index)).attachMoveCounter =
            state->getCard(CardRef(10)).moveCounter;
    }
    place_official_attack_card(state, 20, target_card_id, 1, AreaType::Active);
    place_official_attack_card(state, 30, 1, 0, AreaType::Prize);
    place_official_attack_card(state, 31, 1, 1, AreaType::Prize);
    for (int index = 0; index < 8; ++index) {
        place_official_attack_card(state, 40 + index, 1 + (index % 3), 0, AreaType::Deck);
        place_official_attack_card(state, 60 + index, 1 + (index % 3), 1, AreaType::Deck);
    }
}

std::uint32_t seed_for_coin_fixture(
    const RandomOutcomeFixture& fixture,
    RandomOutcomeCursor* cursor) {
    bool desired[32]{};
    for (std::uint16_t index = 0; index < fixture.coin_count; ++index) {
        if (!consume_coin_outcome(fixture, cursor, &desired[index])) {
            throw std::runtime_error("coin fixture underflow");
        }
    }
    for (std::uint32_t seed = 0; seed < 1'000'000; ++seed) {
        std::mt19937 rng(seed);
        bool matches = true;
        for (std::uint16_t index = 0; index < fixture.coin_count; ++index) {
            if (((rng() % 2U) == 0U) != desired[index]) {
                matches = false;
                break;
            }
        }
        if (matches) return seed;
    }
    throw std::runtime_error("no RNG preimage for coin fixture");
}

std::uint32_t seed_for_target_fixture(
    const RandomOutcomeFixture& fixture,
    RandomOutcomeCursor* cursor,
    std::uint16_t option_count) {
    std::uint16_t desired = 0;
    if (!consume_target_outcome(fixture, cursor, option_count, &desired)) {
        throw std::runtime_error("target fixture underflow");
    }
    for (std::uint32_t seed = 0; seed < 1'000'000; ++seed) {
        std::mt19937 rng(seed);
        std::vector<std::uint16_t> targets;
        for (std::uint16_t index = 0; index < option_count; ++index) {
            targets.push_back(index);
        }
        std::shuffle(targets.begin(), targets.end(), rng);
        if (targets.front() == desired) return seed;
    }
    throw std::runtime_error("no RNG preimage for target fixture");
}

__global__ void execute_attack_fixture_kernel(
    OfficialStatePod* state,
    const std::uint8_t* rule_pack,
    std::int32_t attack_id) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack);
    const OfficialAttackResult result = official_begin_attack(
        state, rules, OfficialCardRefPod{10}, attack_id);
    if (result == OfficialAttackResult::kComplete && official_pod_ok(state)) {
        official_advance_idle_state_to_decision(state, rules);
    }
}

AttackOutcomeEvidence execute_real_attack_fixture(
    const std::vector<std::uint8_t>& rule_pack,
    std::uint32_t seed,
    int attacker_card_id,
    int attack_id,
    int energy_count,
    bool random_target_case,
    std::uint16_t expected_target,
    std::int32_t expected_effect_result) {
    BattleData battle;
    setup_official_attack_battle(&battle, seed);
    State& official = battle.state;
    prepare_official_attack_case(&official, attacker_card_id, 652, energy_count);
    if (random_target_case) {
        for (std::uint16_t index = 0; index < 3; ++index) {
            place_official_attack_card(
                &official, 80 + index, 1 + index, 1, AreaType::Hand);
        }
    }
    OfficialStatePod initial{};
    const OfficialBridgeResult initial_bridge = bridge_official_state(
        official, OfficialStateBridgeContext{seed, 0, true}, &initial);
    if (!initial_bridge) throw std::runtime_error("attack initial bridge failed");

    SelectedAttack(official, attack_id, 0, -1);
    if (!official.step()) {
        throw std::runtime_error("random attack unexpectedly terminated the game");
    }
    if (official.selectType != SelectType::Main
        || official.selectContext != SelectContext::Main) {
        throw std::runtime_error("random attack did not complete to the next main decision");
    }

    const OfficialRulePackView host_rules = make_official_rule_pack_view(rule_pack.data());
    OfficialStatePod host_pod = initial;
    const OfficialAttackResult host_result = official_begin_attack(
        &host_pod, host_rules, OfficialCardRefPod{10}, attack_id);
    if (host_result != OfficialAttackResult::kComplete || host_pod.error != 0) {
        throw std::runtime_error("host POD random attack failed");
    }
    if (official_advance_idle_state_to_decision(&host_pod, host_rules)
        != OfficialFlowStatus::kNeedsAction) {
        throw std::runtime_error("host POD did not reach the next main decision");
    }

    std::uint8_t* device_rules = nullptr;
    OfficialStatePod* device_state = nullptr;
    check_cuda(cudaMalloc(&device_rules, rule_pack.size()), "fixture rule cudaMalloc");
    check_cuda(cudaMalloc(&device_state, sizeof(initial)), "fixture state cudaMalloc");
    check_cuda(cudaMemcpy(
        device_rules, rule_pack.data(), rule_pack.size(), cudaMemcpyHostToDevice),
        "fixture rule H2D");
    check_cuda(cudaMemcpy(device_state, &initial, sizeof(initial), cudaMemcpyHostToDevice),
               "fixture attack H2D");
    execute_attack_fixture_kernel<<<1, 1>>>(device_state, device_rules, attack_id);
    check_cuda(cudaGetLastError(), "fixture attack launch");
    OfficialStatePod gpu{};
    check_cuda(cudaMemcpy(&gpu, device_state, sizeof(gpu), cudaMemcpyDeviceToHost),
               "fixture attack D2H");
    check_cuda(cudaFree(device_state), "fixture attack state free");
    check_cuda(cudaFree(device_rules), "fixture attack rules free");
    if (gpu.error != 0) throw std::runtime_error("CUDA random attack failed");

    const std::uint64_t rng_draws = count_rng_draws(seed, battle.game.rng);
    OfficialStatePod expected{};
    const OfficialBridgeResult final_bridge = bridge_official_state(
        official, OfficialStateBridgeContext{seed, rng_draws, true}, &expected);
    if (!final_bridge) throw std::runtime_error("attack final bridge failed");
    require_state_equal(seed, attack_id, "explicit_random_attack_host_pod",
                        official, expected, host_pod);
    require_state_equal(seed, attack_id, "explicit_random_attack_cuda",
                        official, expected, gpu);

    AttackOutcomeEvidence evidence{};
    evidence.rng_seed = seed;
    if (random_target_case) {
        const std::uint16_t desired_ref = static_cast<std::uint16_t>(80 + expected_target);
        const Card& selected = official.getCard(CardRef(desired_ref));
        const bool moved_from_hand = selected.area != AreaType::Hand
            || selected.preArea == AreaType::Deck;
        if (!moved_from_hand) throw std::runtime_error("random target fixture not consumed");
        evidence.observed_target = expected_target;
        evidence.observed_effect_result = moved_from_hand ? 1 : 0;
    } else if (attacker_card_id == 34) {
        evidence.observed_effect_result =
            official.players[1].badStatus == BadStatusType::Paralyzed ? 1 : 0;
    } else {
        evidence.observed_effect_result = official.getCard(CardRef(20)).damage;
    }
    if (evidence.observed_effect_result != expected_effect_result) {
        throw std::runtime_error("named random effect result mismatch");
    }
    return evidence;
}

AttackOutcomeEvidence run_coin_attack_case(
    const std::vector<std::uint8_t>& rule_pack,
    const RandomOutcomeFixture& fixture,
    int attacker_card_id,
    int attack_id,
    int energy_count) {
    if (!validate_random_outcome_fixture(fixture)) {
        throw std::runtime_error("invalid coin attack fixture");
    }
    RandomOutcomeCursor cursor{};
    const std::uint32_t seed = seed_for_coin_fixture(fixture, &cursor);
    std::int32_t expected_effect = 0;
    if (!consume_effect_outcome(fixture, &cursor, &expected_effect)) {
        throw std::runtime_error("named effect fixture underflow");
    }
    if (!random_outcome_fully_consumed(fixture, cursor)) {
        throw std::runtime_error("coin attack fixture was not fully consumed");
    }
    return execute_real_attack_fixture(
        rule_pack, seed, attacker_card_id, attack_id, energy_count,
        false, 0, expected_effect);
}

AttackOutcomeEvidence run_random_target_attack_case(
    const std::vector<std::uint8_t>& rule_pack,
    const RandomOutcomeFixture& fixture) {
    if (!validate_random_outcome_fixture(fixture)) {
        throw std::runtime_error("invalid target attack fixture");
    }
    RandomOutcomeCursor cursor{};
    std::uint16_t expected_target = fixture.random_target_indices[0];
    const std::uint32_t seed = seed_for_target_fixture(fixture, &cursor, 3);
    std::int32_t expected_effect = 0;
    if (!consume_effect_outcome(fixture, &cursor, &expected_effect)) {
        throw std::runtime_error("target effect fixture underflow");
    }
    if (!random_outcome_fully_consumed(fixture, cursor)) {
        throw std::runtime_error("target attack fixture was not fully consumed");
    }
    return execute_real_attack_fixture(
        rule_pack, seed, 103, 130, 2, true, expected_target, expected_effect);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::runtime_error("usage: fixture <rule-pack> <deck0> <deck1>");
        }
        InitializeAll();
        const std::vector<std::uint8_t> rule_pack = read_binary_cuda(argv[1]);
        const std::vector<std::uint16_t> deck0 = read_deck(argv[2]);
        const std::vector<std::uint16_t> deck1 = read_deck(argv[3]);
        check_cuda(
            cudaDeviceSetLimit(cudaLimitStackSize, kOfficialMinimumDeviceStackBytes),
            "configure fixture device stack");
        run_case(deck0, deck1, 38001, ContinueOperation::kDraw);
        run_case(deck0, deck1, 38002, ContinueOperation::kDeckToPrize);

        RandomOutcomeFixture numbing_head{};
        numbing_head.coin_count = 1;
        numbing_head.coin_results[0] = 1;
        numbing_head.effect_count = 1;
        numbing_head.effect_results[0] = 1;
        const AttackOutcomeEvidence head = run_coin_attack_case(
            rule_pack, numbing_head, 34, 25, 1);

        RandomOutcomeFixture numbing_tail{};
        numbing_tail.coin_count = 1;
        numbing_tail.coin_results[0] = 0;
        numbing_tail.effect_count = 1;
        numbing_tail.effect_results[0] = 0;
        const AttackOutcomeEvidence tail = run_coin_attack_case(
            rule_pack, numbing_tail, 34, 25, 1);

        RandomOutcomeFixture double_hit{};
        double_hit.coin_count = 2;
        double_hit.coin_results[0] = 1;
        double_hit.coin_results[1] = 0;
        double_hit.effect_count = 1;
        double_hit.effect_results[0] = 90;
        const AttackOutcomeEvidence hit = run_coin_attack_case(
            rule_pack, double_hit, 51, 52, 3);

        RandomOutcomeFixture astonish{};
        astonish.target_count = 1;
        astonish.random_target_indices[0] = 2;
        astonish.effect_count = 1;
        astonish.effect_results[0] = 1;
        const AttackOutcomeEvidence target = run_random_target_attack_case(
            rule_pack, astonish);

        std::cout << "{\"passed\":true,\"conditional_cases\":6,"
                     "\"shuffle_permutation\":true,\"prize_indices\":true,"
                     "\"prize_indices_rule_execution\":true,"
                     "\"real_follow_on_rules\":[\"Draw\",\"DeckToPrize\"],"
                     "\"coin_schema\":true,\"random_target_schema\":true,"
                     "\"effect_result_schema\":true,"
                     "\"coin_rule_execution\":true,"
                     "\"random_target_rule_execution\":true,"
                     "\"random_effect_rule_execution\":true,"
                     "\"rule_paths\":[\"Numbing Water(head)\","
                     "\"Numbing Water(tail)\",\"Double Hit(1 head)\","
                     "\"Astonish(target index 2)\"],"
                     "\"injection_adapter\":\"test_only_rng_preimage\","
                     "\"numbing_head_seed\":" << head.rng_seed << ','
                  << "\"numbing_tail_seed\":" << tail.rng_seed << ','
                  << "\"double_hit_seed\":" << hit.rng_seed << ','
                  << "\"astonish_seed\":" << target.rng_seed << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
