#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "All.h"
#include "ptcg_cuda/official_conditions_pod.cuh"
#include "ptcg_cuda/official_continual_effects_pod.cuh"
#include "ptcg_cuda/official_continuation_pod.cuh"

namespace {

using namespace ptcg::cuda_engine;

constexpr int kAttackerRef = 3;
constexpr int kFirstEnergyRef = 4;
constexpr int kNamedEnergyRef = 9;
constexpr int kGrassPokemonRef = 10;
constexpr int kGrassEnergyRef = 11;
constexpr int kStage2PokemonRef = 12;
constexpr int kNeoUpperEnergyRef = 13;
constexpr int kMoveEnergyRef = 14;
constexpr int kEvolutionRef = 15;
constexpr int kTransformRef = 16;
constexpr int kDelaySourceRef = 17;
constexpr int kEnemyBenchRef0 = 61;
constexpr int kEnemyBenchRef1 = 62;
constexpr int kEnemyActiveRef = 63;

std::vector<std::uint8_t> read_binary(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open rule pack: " + path);
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
}

void require(bool value, const std::string& label) {
    if (!value) throw std::runtime_error("semantic fixture failed: " + label);
}

void clear_official_player(PlayerState* player) {
    player->active.clear();
    player->bench.clear();
    player->prize.clear();
    player->hand.clear();
    player->deck.clear();
    player->trash.clear();
    player->energy.clear();
    player->tool.clear();
    player->preEvolution.clear();
    player->temporary.clear();
    player->activeState = 0;
    player->continualState = 0;
    player->turnState = 0;
}

void place_official_card(
    State* state,
    int ref_value,
    int player,
    AreaType area) {
    const CardRef ref(ref_value);
    Card& card = state->getCard(ref);
    card.playerIndex = static_cast<signed char>(player);
    card.area = area;
    PlayerState& ps = state->players[player];
    switch (area) {
        case AreaType::Deck: ps.deck.push_back(ref); break;
        case AreaType::Hand: ps.hand.push_back(ref); break;
        case AreaType::Trash: ps.trash.push_back(ref); break;
        case AreaType::Active: ps.active.push_back(ref); break;
        case AreaType::Bench: ps.bench.push_back(ref); break;
        case AreaType::Energy: ps.energy.push_back(ref); break;
        case AreaType::Tool: ps.tool.push_back(ref); break;
        default: throw std::runtime_error("unsupported official fixture area");
    }
}

void attach_official(State* state, int attached_ref, int pokemon_ref) {
    state->getCard(CardRef(attached_ref)).attachMoveCounter =
        state->getCard(CardRef(pokemon_ref)).moveCounter;
}

void add_pod_card(
    OfficialStatePod* state,
    std::uint16_t instance,
    std::int32_t card_id,
    std::int32_t player,
    OfficialArea area) {
    OfficialCardStatePod* card = &state->cards[instance];
    card->card_id = card_id;
    card->move_counter = state->move_counter++;
    card->player = static_cast<std::int8_t>(player);
    card->area = static_cast<std::uint8_t>(area);
    require(
        official_pod_push_zone_card(
            state, player, area, OfficialCardRefPod{instance}),
        "pod zone push");
}

void attach_pod(OfficialStatePod* state, int attached_ref, int pokemon_ref) {
    state->cards[attached_ref].attach_move_counter = state->cards[pokemon_ref].move_counter;
}

bool official_target(
    const State& state,
    int ref_value,
    TargetType type,
    int value = 0,
    const std::u8string& name = {}) {
    const CardRef ref(ref_value);
    const Card& card = state.getCard(ref);
    const TargetCondition condition{type, ComparatorType::Equal, value, 0, name};
    return IsTargetSingle(
        state,
        ref,
        card,
        card.getMaster(),
        condition,
        state.makeAreaRef(CardRef(kAttackerRef)));
}

bool pod_target(
    OfficialStatePod* state,
    const OfficialRulePackView& rules,
    int ref_value,
    OfficialTargetTypeId type,
    int value = 0,
    int name_id = 0) {
    OfficialConditionRule condition{};
    condition.values[kConditionType] = static_cast<std::int32_t>(type);
    condition.values[kConditionComparator] = 0;
    condition.values[kConditionValue] = value;
    condition.values[kConditionNameId] = name_id;
    const OfficialTargetMatchResult result = official_match_target_condition(
        state,
        rules,
        OfficialCardRefPod{static_cast<std::uint16_t>(ref_value)},
        condition,
        official_pod_area_ref(
            state, OfficialCardRefPod{static_cast<std::uint16_t>(kAttackerRef)}));
    require(result != OfficialTargetMatchResult::kUnsupported, "pod target supported");
    require(result != OfficialTargetMatchResult::kError, "pod target no error");
    return result == OfficialTargetMatchResult::kMatch;
}

bool pod_attack_extra(
    OfficialStatePod* state,
    const OfficialRulePackView& rules) {
    OfficialEffectRule effect{};
    effect.values[kEffectConditionType] = static_cast<std::int32_t>(
        OfficialConditionTypeId::kAttackEnergyExtra);
    effect.values[kEffectComparator] = 1;
    effect.values[kEffectValue0] = 2;
    effect.values[kEffectTargetIndex] = 0;
    const OfficialConditionResult result = official_satisfy_condition(
        state,
        rules,
        &effect,
        1,
        0,
        official_pod_area_ref(
            state, OfficialCardRefPod{static_cast<std::uint16_t>(kAttackerRef)}),
        0);
    require(result != OfficialConditionResult::kUnsupported, "pod condition supported");
    require(result != OfficialConditionResult::kError, "pod condition no error");
    return result == OfficialConditionResult::kTrue;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: official_semantic_micro_fixture <rule-pack>");
        }
        InitializeAll();
        const std::vector<std::uint8_t> rule_pack = read_binary(argv[1]);
        const OfficialRulePackView rules = make_official_rule_pack_view(rule_pack.data());

        GameConfig config{};
        config.deviceRand = false;
        config.recordLog = false;
        for (int player = 0; player < 2; ++player) {
            for (int index = 0; index < DECK_SIZE; ++index) {
                config.decks[player].cards[index] = 197;
            }
        }
        const int player0_cards[] = {
            372, 4, 4, 4, 4, 4, 15, 707, 1, 40, 10, 4, 197, 197, 25};
        for (int index = 0; index < static_cast<int>(std::size(player0_cards)); ++index) {
            config.decks[0].cards[index] = player0_cards[index];
        }
        BattleData official_data;
        official_data.init(config, false);
        State& official = official_data.state;
        clear_official_player(&official.players[0]);
        clear_official_player(&official.players[1]);

        place_official_card(&official, kAttackerRef, 0, AreaType::Active);
        for (int ref = kFirstEnergyRef; ref < kFirstEnergyRef + 4; ++ref) {
            place_official_card(&official, ref, 0, AreaType::Energy);
            attach_official(&official, ref, kAttackerRef);
        }
        place_official_card(&official, kGrassPokemonRef, 0, AreaType::Bench);
        place_official_card(&official, kGrassEnergyRef, 0, AreaType::Energy);
        attach_official(&official, kGrassEnergyRef, kGrassPokemonRef);
        place_official_card(&official, kStage2PokemonRef, 0, AreaType::Bench);
        place_official_card(&official, kNeoUpperEnergyRef, 0, AreaType::Energy);
        attach_official(&official, kNeoUpperEnergyRef, kStage2PokemonRef);
        place_official_card(&official, kEnemyActiveRef, 1, AreaType::Active);
        official.getCard(CardRef(kGrassPokemonRef)).doubleGrassEnergy = true;
        official.attacker = CardRef(kAttackerRef);
        official.srcAttackId = 517;

        OfficialStatePod pod{};
        official_pod_reset(&pod, 99, 20260730);
        add_pod_card(&pod, kAttackerRef, 372, 0, OfficialArea::kActive);
        for (int ref = kFirstEnergyRef; ref < kFirstEnergyRef + 4; ++ref) {
            add_pod_card(&pod, static_cast<std::uint16_t>(ref), 4, 0, OfficialArea::kEnergy);
            attach_pod(&pod, ref, kAttackerRef);
        }
        add_pod_card(&pod, kGrassPokemonRef, 707, 0, OfficialArea::kBench);
        add_pod_card(&pod, kGrassEnergyRef, 1, 0, OfficialArea::kEnergy);
        attach_pod(&pod, kGrassEnergyRef, kGrassPokemonRef);
        add_pod_card(&pod, kStage2PokemonRef, 40, 0, OfficialArea::kBench);
        add_pod_card(&pod, kNeoUpperEnergyRef, 10, 0, OfficialArea::kEnergy);
        attach_pod(&pod, kNeoUpperEnergyRef, kStage2PokemonRef);
        add_pod_card(&pod, kEnemyActiveRef, 197, 1, OfficialArea::kActive);
        pod.cards[kGrassPokemonRef].continual_state[4] |= 1ULL << 29;
        pod.attacker = OfficialCardRefPod{kAttackerRef};
        pod.source_attack_id = 517;

        int checks = 0;
        const auto compare = [&](bool expected, bool actual, const char* label) {
            require(expected == actual, label);
            ++checks;
        };

        compare(
            official_target(official, kFirstEnergyRef, TargetType::EnergyTypeAttached, 8),
            pod_target(&pod, rules, kFirstEnergyRef, OfficialTargetTypeId::kEnergyTypeAttached, 8),
            "EnergyTypeAttached positive");
        compare(
            official_target(official, kFirstEnergyRef, TargetType::EnergyTypeAttached, 2),
            pod_target(&pod, rules, kFirstEnergyRef, OfficialTargetTypeId::kEnergyTypeAttached, 2),
            "EnergyTypeAttached negative");

        const std::u8string named_energy = CardTable.at(15).name;
        compare(
            official_target(
                official, kAttackerRef, TargetType::IsAttachedEnergyName, 0, named_energy),
            pod_target(
                &pod,
                rules,
                kAttackerRef,
                OfficialTargetTypeId::kIsAttachedEnergyName,
                0,
                rules.cards[14].values[kCardNameId]),
            "IsAttachedEnergyName negative");

        compare(
            official_target(official, kAttackerRef, TargetType::SameTypeEnemy),
            pod_target(&pod, rules, kAttackerRef, OfficialTargetTypeId::kSameTypeEnemy),
            "SameTypeEnemy negative");
        official.getCard(CardRef(kEnemyActiveRef)).typeIndex = 4;
        pod.cards[kEnemyActiveRef].continual_state[3] |= 4ULL << 48;
        compare(
            official_target(official, kAttackerRef, TargetType::SameTypeEnemy),
            pod_target(&pod, rules, kAttackerRef, OfficialTargetTypeId::kSameTypeEnemy),
            "SameTypeEnemy positive");

        const EnergyInfo official_grass = official.getEnergyInfo(
            official.getCard(CardRef(kGrassEnergyRef)), CardRef(kGrassPokemonRef));
        const OfficialEnergyInfoPod pod_grass = official_energy_info(
            pod,
            rules,
            OfficialCardRefPod{kGrassEnergyRef},
            OfficialCardRefPod{kGrassPokemonRef});
        require(static_cast<int>(official_grass.type) == pod_grass.type, "double grass type");
        require(official_grass.count == pod_grass.count, "double grass count");
        checks += 2;

        const EnergyInfo official_neo = official.getEnergyInfo(
            official.getCard(CardRef(kNeoUpperEnergyRef)), CardRef(kStage2PokemonRef));
        const OfficialEnergyInfoPod pod_neo = official_energy_info(
            pod,
            rules,
            OfficialCardRefPod{kNeoUpperEnergyRef},
            OfficialCardRefPod{kStage2PokemonRef});
        require(static_cast<int>(official_neo.type) == pod_neo.type, "neo upper type");
        require(official_neo.count == pod_neo.count, "neo upper count");
        checks += 2;

        const Attack& attack = AttackTable.at(517);
        compare(
            SatisfyCondition(
                official,
                attack.preEffects,
                0,
                CardRef(kAttackerRef),
                0),
            pod_attack_extra(&pod, rules),
            "AttackEnergyExtra negative");
        place_official_card(&official, kFirstEnergyRef + 4, 0, AreaType::Energy);
        attach_official(&official, kFirstEnergyRef + 4, kAttackerRef);
        add_pod_card(
            &pod,
            kFirstEnergyRef + 4,
            4,
            0,
            OfficialArea::kEnergy);
        attach_pod(&pod, kFirstEnergyRef + 4, kAttackerRef);
        compare(
            SatisfyCondition(
                official,
                attack.preEffects,
                0,
                CardRef(kAttackerRef),
                0),
            pod_attack_extra(&pod, rules),
            "AttackEnergyExtra positive");

        place_official_card(&official, kNamedEnergyRef, 0, AreaType::Energy);
        attach_official(&official, kNamedEnergyRef, kAttackerRef);
        add_pod_card(&pod, kNamedEnergyRef, 15, 0, OfficialArea::kEnergy);
        attach_pod(&pod, kNamedEnergyRef, kAttackerRef);
        compare(
            official_target(
                official, kAttackerRef, TargetType::IsAttachedEnergyName, 0, named_energy),
            pod_target(
                &pod,
                rules,
                kAttackerRef,
                OfficialTargetTypeId::kIsAttachedEnergyName,
                0,
                rules.cards[14].values[kCardNameId]),
            "IsAttachedEnergyName positive");

        std::vector<AreaRef> official_continual_targets{
            official.makeAreaRef(CardRef(kAttackerRef))};
        pod.targets.count = 0;
        require(
            official_pod_push(
                &pod,
                &pod.targets,
                official_pod_area_ref(&pod, OfficialCardRefPod{kAttackerRef}),
                OfficialPodError::kSelectionOverflow),
            "continual pod target");
        OfficialTargetRule player_target{};
        player_target.values[kTargetPlayer] = 1;
        OfficialRulePackView continual_rules = rules;
        continual_rules.targets = &player_target;

        const auto compare_continual = [&] (
            EffectType official_type,
            OfficialEffectTypeId pod_type,
            int value,
            const char* label) {
            Effect official_effect{};
            official_effect.effectType = official_type;
            official_effect.values[0] = value;
            official_effect.target.targetPlayer = TargetPlayer::Me;
            EffectContinual(
                official,
                official_effect,
                official_continual_targets,
                0,
                CardRef(kAttackerRef));

            OfficialEffectRule pod_effect{};
            pod_effect.values[kEffectType] = static_cast<int>(pod_type);
            pod_effect.values[kEffectValue0] = value;
            pod_effect.values[kEffectTargetIndex] = 0;
            require(
                official_apply_continual_effect_primitive(
                    &pod,
                    continual_rules,
                    pod_effect,
                    0,
                    OfficialCardRefPod{kAttackerRef})
                    == OfficialEffectApplyResult::kApplied,
                label);
        };

        compare_continual(
            EffectType::MaxHpChange,
            OfficialEffectTypeId::kMaxHpChange,
            10,
            "MaxHpChange apply");
        compare_continual(
            EffectType::DamageChange,
            OfficialEffectTypeId::kDamageChange,
            -20,
            "DamageChange apply");
        compare_continual(
            EffectType::RetreatCostChange,
            OfficialEffectTypeId::kRetreatCostChange,
            -1,
            "RetreatCostChange apply");
        compare_continual(
            EffectType::NoSpecialCondition,
            OfficialEffectTypeId::kNoSpecialCondition,
            0,
            "NoSpecialCondition apply");
        compare_continual(
            EffectType::AttackEnergyPsychicOne,
            OfficialEffectTypeId::kAttackEnergyPsychicOne,
            0,
            "AttackEnergyPsychicOne apply");
        compare_continual(
            EffectType::PoisonDamageChange,
            OfficialEffectTypeId::kPoisonDamageChange,
            3,
            "PoisonDamageChange apply");
        compare_continual(
            EffectType::CannotPlayItem,
            OfficialEffectTypeId::kCannotPlayItem,
            0,
            "CannotPlayItem apply");
        compare_continual(
            EffectType::NoToolEffect,
            OfficialEffectTypeId::kNoToolEffect,
            0,
            "NoToolEffect apply");

        const Card& official_attacker = official.getCard(CardRef(kAttackerRef));
        const OfficialCardStatePod& pod_attacker = pod.cards[kAttackerRef];
        require(official_attacker.hpChange == pod_attacker.hp_change, "MaxHpChange state");
        require(
            official_attacker.damageChange
                == official_continual_packed_i16(pod_attacker, 2),
            "DamageChange state");
        require(
            official_attacker.retreatCostChange
                == official_continual_i8(pod_attacker, 26),
            "RetreatCostChange state");
        require(
            official_attacker.noSpecialCondition
                == official_continual_flag(pod_attacker, 13),
            "NoSpecialCondition state");
        require(
            official_attacker.attackEnergyPsychicOne
                == official_continual_flag(pod_attacker, 28),
            "AttackEnergyPsychicOne state");
        require(
            official.players[0].poisonDamageChange
                == static_cast<std::int16_t>(pod.players[0].continual_state & 0xffffULL),
            "PoisonDamageChange state");
        require(
            official.players[0].cannotPlayItem
                == (((pod.players[0].continual_state >> 44U) & 1ULL) != 0),
            "CannotPlayItem state");
        require(
            official.noToolEffect == ((pod.continual_state & 1U) != 0),
            "NoToolEffect state");
        checks += 8;

        official.currentAttackId = 517;
        pod.current_attack_id = 517;
        const int official_damage = CalcDamage(
            official,
            50,
            CardRef(kEnemyActiveRef),
            official.getCard(CardRef(kEnemyActiveRef)),
            CardRef(kAttackerRef),
            official.getCard(CardRef(kAttackerRef)),
            true,
            &AttackTable.at(517));
        const int pod_damage = official_calc_attack_damage(
            &pod,
            continual_rules,
            50,
            OfficialCardRefPod{kEnemyActiveRef},
            OfficialCardRefPod{kAttackerRef},
            true);
        require(
            official_damage == pod_damage,
            "CalcDamage state pipeline official=" + std::to_string(official_damage)
                + " pod=" + std::to_string(pod_damage));
        ++checks;

        {
            State official_ko = official;
            OfficialStatePod pod_ko = pod;
            official_ko.getCard(CardRef(kEnemyActiveRef)).damage = 0;
            pod_ko.cards[kEnemyActiveRef].damage = 0;
            AddDamage(
                official_ko,
                CardRef(kEnemyActiveRef),
                official_ko.getCard(CardRef(kEnemyActiveRef)),
                1000000,
                true,
                CardRef(kAttackerRef),
                false,
                &AttackTable.at(517));
            official_apply_attack_damage_to_card(
                &pod_ko,
                continual_rules,
                OfficialCardRefPod{kEnemyActiveRef},
                1000000,
                true);
            require(
                official_ko.getCard(CardRef(kEnemyActiveRef)).koCauseRef
                    == static_cast<int>(pod_ko.cards[kEnemyActiveRef].turn_state[1]
                        & 0xffU),
                "attack KO cause ref");
            require(
                official_ko.getCard(CardRef(kEnemyActiveRef)).turnState[1]
                        == pod_ko.cards[kEnemyActiveRef].turn_state[1]
                    && official_ko.getCard(CardRef(kEnemyActiveRef)).turnState[2]
                        == pod_ko.cards[kEnemyActiveRef].turn_state[2],
                "attack KO turn-state flags");
            checks += 2;
        }

        {
            OfficialStatePod effect_damage = pod;
            effect_damage.last_attack_damage = 123;
            official_apply_attack_damage_to_card(
                &effect_damage,
                continual_rules,
                OfficialCardRefPod{kEnemyActiveRef},
                10);
            require(
                effect_damage.last_attack_damage == 123,
                "effect attack damage preserves main attack damage");
            ++checks;
        }

        place_official_card(&official, kMoveEnergyRef, 0, AreaType::Hand);
        add_pod_card(&pod, kMoveEnergyRef, 4, 0, OfficialArea::kHand);
        AttachProc(
            official,
            AreaType::Hand,
            0,
            0,
            CardRef(kAttackerRef),
            true);
        require(
            official_effect_attach_card(
                &pod,
                continual_rules,
                OfficialCardRefPod{kMoveEnergyRef},
                OfficialCardRefPod{kAttackerRef}),
            "attach move primitive");
        require(
            official.getCard(CardRef(kMoveEnergyRef)).area
                == static_cast<AreaType>(pod.cards[kMoveEnergyRef].area),
            "attach move area");
        require(
            official.getCard(CardRef(kMoveEnergyRef)).attachMoveCounter
                == official.getCard(CardRef(kAttackerRef)).moveCounter
                && pod.cards[kMoveEnergyRef].attach_move_counter
                    == pod.cards[kAttackerRef].move_counter,
            "attach move counter");
        checks += 2;

        place_official_card(&official, kEvolutionRef, 0, AreaType::Hand);
        add_pod_card(&pod, kEvolutionRef, 197, 0, OfficialArea::kHand);
        EvolveProc(official, AreaType::Hand, 0, AreaType::Bench, 0, 0);
        require(
            official_effect_evolve_card(
                &pod,
                continual_rules,
                OfficialCardRefPod{kEvolutionRef},
                OfficialCardRefPod{kGrassPokemonRef}),
            "evolve move primitive");
        require(
            official.players[0].bench[0].cardIndex
                == pod.players[0].bench.values[0].index,
            "evolve bench replacement");
        require(
            official.players[0].preEvolution.size()
                == pod.players[0].pre_evolution.count,
            "evolve pre-evolution count");
        checks += 2;

        place_official_card(&official, kTransformRef, 0, AreaType::Hand);
        add_pod_card(&pod, kTransformRef, 197, 0, OfficialArea::kHand);
        official.selectedList.clear();
        official.selectedList.push_back(CardRef(kTransformRef));
        official.targetList.clear();
        official.targetList.push_back(official.makeAreaRef(CardRef(kEvolutionRef)));
        const int official_transform_counter =
            official.getCard(CardRef(kEvolutionRef)).moveCounter;
        const int pod_transform_counter = pod.cards[kEvolutionRef].move_counter;
        TransformProc(official, true);
        pod.selected_list.count = 0;
        require(
            official_pod_push(
                &pod,
                &pod.selected_list,
                OfficialCardRefPod{kTransformRef},
                OfficialPodError::kSelectionOverflow),
            "transform selected");
        pod.targets.count = 0;
        require(
            official_pod_push(
                &pod,
                &pod.targets,
                official_pod_area_ref(&pod, OfficialCardRefPod{kEvolutionRef}),
                OfficialPodError::kSelectionOverflow),
            "transform target");
        require(
            official_effect_transform_card(
                &pod,
                OfficialCardRefPod{kEvolutionRef},
                OfficialCardRefPod{kTransformRef},
                OfficialArea::kDeck),
            "transform move primitive");
        require(
            official.players[0].bench[0].cardIndex
                == pod.players[0].bench.values[0].index,
            "transform bench replacement");
        require(
            official.getCard(CardRef(kTransformRef)).moveCounter
                == official_transform_counter
                && pod.cards[kTransformRef].move_counter == pod_transform_counter,
            "transform preserved state");
        checks += 2;

        {
            OfficialStatePod switch_effect = pod;
            const OfficialCardRefPod old_active = switch_effect.players[0].active.values[0];
            const OfficialCardRefPod selected_bench = switch_effect.players[0].bench.values[0];
            switch_effect.control_flags &= static_cast<std::uint8_t>(
                ~kOfficialChangedFlag);
            switch_effect.targets.count = 0;
            require(
                official_pod_push(
                    &switch_effect,
                    &switch_effect.targets,
                    official_pod_area_ref(&switch_effect, selected_bench),
                    OfficialPodError::kSelectionOverflow),
                "switch effect target");
            OfficialEffectRule switch_rule{};
            switch_rule.values[kEffectType] = static_cast<int>(
                OfficialEffectTypeId::kSwitch);
            require(
                official_apply_effect_primitive(
                    &switch_effect, continual_rules, switch_rule)
                    == OfficialEffectApplyResult::kUnsupported,
                "switch rejects incomplete primitive path");
            require(
                official_apply_switch_effect(
                    &switch_effect, continual_rules, switch_rule)
                    == OfficialEffectApplyResult::kApplied,
                "switch effect primitive");
            require(
                (switch_effect.control_flags & kOfficialChangedFlag) != 0,
                "switch effect marks changed");
            require(
                switch_effect.players[0].active.values[0] == selected_bench
                    && switch_effect.players[0].bench.values[0] == old_active,
                "switch effect swaps active and bench");
            checks += 3;
        }

        {
            OfficialStatePod switch_triggers{};
            official_pod_reset(&switch_triggers, 99, 20260730);
            switch_triggers.phase = static_cast<std::uint8_t>(
                OfficialGamePhase::kMain);
            add_pod_card(
                &switch_triggers, 70, 106, 0, OfficialArea::kActive);
            add_pod_card(
                &switch_triggers, 71, 340, 0, OfficialArea::kBench);
            add_pod_card(
                &switch_triggers, 72, 107, 0, OfficialArea::kDeck);
            add_pod_card(
                &switch_triggers, 73, 1, 0, OfficialArea::kDeck);
            add_pod_card(
                &switch_triggers, 74, 197, 1, OfficialArea::kActive);
            require(
                official_switch_pokemon(
                    &switch_triggers, rules, 0, 0),
                "official switch helper");
            bool found_active_to_bench = false;
            bool found_bench_to_active = false;
            for (std::uint16_t index = 0;
                 index < switch_triggers.temporary_triggers.count;
                 ++index) {
                const OfficialTriggeredAbilityPod& trigger =
                    switch_triggers.temporary_triggers.values[index];
                found_active_to_bench |= trigger.activate.skill_id == 39
                    && trigger.trigger.type == kOfficialSwitchTriggerActiveToBench;
                found_bench_to_active |= trigger.activate.skill_id == 113
                    && trigger.trigger.type == kOfficialSwitchTriggerBenchToActive;
            }
            require(
                found_active_to_bench && found_bench_to_active,
                "official switch helper pulls both movement triggers");
            checks += 2;
        }

        {
            OfficialStatePod switch_target = pod;
            const OfficialCardRefPod old_active =
                switch_target.players[0].active.values[0];
            const OfficialCardRefPod selected_bench =
                switch_target.players[0].bench.values[0];
            switch_target.targets.count = 0;
            require(
                official_pod_push(
                    &switch_target,
                    &switch_target.targets,
                    official_pod_area_ref(&switch_target, selected_bench),
                    OfficialPodError::kSelectionOverflow),
                "switch effected target seed");
            OfficialEffectRule switch_rule{};
            switch_rule.values[kEffectType] = static_cast<int>(
                OfficialEffectTypeId::kSwitch);
            switch_rule.flags = kOfficialEffectSetTargetSwitchBench;
            require(
                official_apply_switch_effect(
                    &switch_target, rules, switch_rule)
                    == OfficialEffectApplyResult::kApplied,
                "switch effected target primitive");
            require(
                switch_target.targets.count == 1
                    && switch_target.targets.values[0].card == old_active
                    && official_pod_area_ref_valid(
                        &switch_target, switch_target.targets.values[0]),
                "switch effected target follows old active to bench");
            checks += 3;
        }

        {
            OfficialStatePod core_switch = pod;
            core_switch.players[0].active_state = 1;
            core_switch.cards[kAttackerRef].next_enemy_turn_end = 1U << 24U;
            core_switch.cards[kAttackerRef].next_enemy_turn_end_battlefield = 1;
            core_switch.control_flags &= static_cast<std::uint8_t>(
                ~kOfficialChangedFlag);
            require(
                official_pod_switch_active(&core_switch, 0, 0),
                "core switch primitive");
            require(
                core_switch.players[0].active_state == 0,
                "core switch clears special conditions");
            require(
                (core_switch.control_flags & kOfficialChangedFlag) == 0,
                "core switch status recovery does not imply effect changed");
            require(
                core_switch.cards[kAttackerRef].next_enemy_turn_end == 0,
                "core switch clears next-enemy-turn state");
            require(
                core_switch.cards[kAttackerRef].next_enemy_turn_end_battlefield == 1,
                "core switch preserves battlefield next-enemy-turn state");
            ++checks;
        }

        {
            OfficialStatePod retreat_energy = pod;
            retreat_energy.control_flags &= static_cast<std::uint8_t>(
                ~kOfficialChangedFlag);
            require(
                official_main_prepare_retreat_energy(
                    &retreat_energy, rules, 1)
                    == OfficialMainResult::kNeedsAction,
                "retreat energy selection needs action");
            require(
                (retreat_energy.control_flags & kOfficialChangedFlag) != 0,
                "retreat energy candidates mark changed");
            require(
                retreat_energy.select_type
                        == static_cast<std::uint8_t>(OfficialSelectTypeId::kEnergy)
                    && retreat_energy.options.count > 0,
                "retreat energy candidates preserve official option flow");
            checks += 3;
        }

        {
            OfficialStatePod free_retreat = pod;
            free_retreat.targets.count = 0;
            require(
                official_pod_push(
                    &free_retreat,
                    &free_retreat.targets,
                    official_pod_area_ref(
                        &free_retreat,
                        OfficialCardRefPod{kAttackerRef}),
                    OfficialPodError::kSelectionOverflow),
                "free retreat prior target");
            require(
                official_main_prepare_retreat_switch(
                    &free_retreat, rules, false)
                    == OfficialMainResult::kNeedsAction,
                "free retreat switch needs action");
            require(
                free_retreat.targets.count == 1
                    && free_retreat.targets.values[0].card.index == kAttackerRef,
                "free retreat preserves targetList until AfterRetreat");
            checks += 3;
        }

        {
            OfficialStatePod effect_energy = pod;
            effect_energy.control_flags &= static_cast<std::uint8_t>(
                ~kOfficialChangedFlag);
            effect_energy.targets.count = 0;
            require(
                official_pod_push(
                    &effect_energy,
                    &effect_energy.targets,
                    official_pod_area_ref(
                        &effect_energy,
                        OfficialCardRefPod{kFirstEnergyRef}),
                    OfficialPodError::kSelectionOverflow),
                "effect energy target");
            effect_energy.effect_interpreter.active = 1;
            effect_energy.effect_interpreter.effect_owner = 0;
            effect_energy.effect_interpreter.effect_count = 1;
            effect_energy.flow_flags |= kOfficialPlayEffectReturnToMainFlag;
            OfficialEffectRule energy_effect{};
            energy_effect.values[kEffectSelectContext] = 32;
            require(
                official_prepare_effect_selection(
                    &effect_energy,
                    rules,
                    energy_effect,
                    OfficialEffectSelectTypeId::kEnergy,
                    1)
                    == OfficialEffectInterpreterResult::kNeedsAction,
                "effect energy selection needs action");
            require(
                (effect_energy.control_flags & kOfficialChangedFlag) != 0
                    && effect_energy.options.count == 1
                    && effect_energy.options.values[0].resolved_card
                        == kAttackerRef
                    && effect_energy.options.values[0].option_equiv
                        == effect_energy.cards[kAttackerRef].card_id,
                "effect energy option resolves attached Pokemon and marks changed");
            OfficialCardRefPod selected_energy{};
            require(
                official_energy_ref_from_option(
                    &effect_energy,
                    effect_energy.options.values[0],
                    &selected_energy)
                    && selected_energy.index == kFirstEnergyRef,
                "effect energy option resolves back to Energy");
            require(
                official_build_effect_selection_continuation_mirror(
                    &effect_energy)
                    && effect_energy.continuations.count == 4
                    && effect_energy.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedPokemonEnergy)
                    && effect_energy.continuations.values[3].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedPokemonEnergyLoop),
                "effect energy continuation omits SelectedEnergyTarget");
            effect_energy.energy_cost = 3;
            effect_energy.remain_energy_cost = 2;
            effect_energy.selected_energy_card_count = 1;
            effect_energy.selecting_energy_pokemon =
                OfficialCardRefPod{kAttackerRef};
            official_complete_energy_selection_scratch(&effect_energy);
            require(
                effect_energy.energy_cost == 0
                    && effect_energy.remain_energy_cost == 0
                    && effect_energy.selected_energy_card_count == 0
                    && effect_energy.selecting_energy_pokemon.index == 0
                    && effect_energy.targets.count == 0,
                "SelectedPokemonEnergy clears energy scratch");
            checks += 5;
        }

        {
            OfficialStatePod select_activate = pod;
            select_activate.current_attack_id = 531;
            select_activate.attacker = OfficialCardRefPod{kAttackerRef};
            require(
                official_attack_begin_post_effects(
                    &select_activate, rules)
                    == OfficialAttackResult::kNeedsAction,
                "SelectActivate waits for a YesNo action");
            require(
                select_activate.select_type
                        == static_cast<std::uint8_t>(OfficialSelectTypeId::kYesNo)
                    && select_activate.select_context
                        == kOfficialEffectSelectContextActivate
                    && select_activate.options.count == 2
                    && select_activate.options.values[0].type
                        == static_cast<std::uint8_t>(
                            OfficialSelectOptionTypeId::kYes)
                    && select_activate.options.values[1].type
                        == static_cast<std::uint8_t>(
                            OfficialSelectOptionTypeId::kNo),
                "SelectActivate mirrors official YesNo options");
            require(
                official_build_effect_selection_continuation_mirror(
                    &select_activate)
                    && select_activate.continuations.count == 4
                    && select_activate.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterEffect)
                    && select_activate.continuations.values[3].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedActivate),
                "SelectActivate mirrors AfterEffect and callback");
            const std::uint16_t decline = 1;
            require(
                official_consume_effect_selection_continuation_mirror(
                    &select_activate)
                    && official_apply_effect_action(
                        &select_activate, rules, &decline, 1)
                        == OfficialEffectInterpreterResult::kComplete
                    && select_activate.effect_interpreter.active == 0
                    && select_activate.select_type
                        == static_cast<std::uint8_t>(OfficialSelectTypeId::kNone),
                "SelectActivate No breaks the remaining effect range");
            checks += 4;
        }

        {
            OfficialStatePod damage_counter_any = pod;
            damage_counter_any.attack_flow_stage = static_cast<std::uint8_t>(
                OfficialAttackStage::kPostEffects);
            damage_counter_any.effect_interpreter.active = 1;
            damage_counter_any.effect_interpreter.awaiting_selection = 1;
            damage_counter_any.effect_interpreter.resume_kind =
                static_cast<std::uint8_t>(
                    OfficialEffectResumeKind::kDamageCounterAny);
            damage_counter_any.effect_interpreter.effect_count = 1;
            damage_counter_any.effect_interpreter.damage_counter_any_total = 6;
            damage_counter_any.remain_damage_counter = 6;
            damage_counter_any.select_type = static_cast<std::uint8_t>(
                OfficialSelectTypeId::kCard);
            damage_counter_any.select_context =
                kOfficialSelectContextDamageCounterAny;
            require(
                official_build_effect_selection_continuation_mirror(
                    &damage_counter_any)
                    && damage_counter_any.continuations.count == 4
                    && damage_counter_any.continuations.values[0].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterAttack)
                    && damage_counter_any.continuations.values[1].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterEffect)
                    && damage_counter_any.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectDamageCounterAny)
                    && damage_counter_any.continuations.values[2].call_count == 6
                    && damage_counter_any.continuations.values[2].called_count == 1
                    && damage_counter_any.continuations.values[3].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedDamageCounterAny),
                "DamageCounterAny mirrors official repeated selection stack");
            require(
                official_consume_effect_selection_continuation_mirror(
                    &damage_counter_any)
                    && damage_counter_any.continuations.count == 0,
                "DamageCounterAny continuation consume");
            damage_counter_any.remain_damage_counter = 4;
            require(
                official_build_effect_selection_continuation_mirror(
                    &damage_counter_any)
                    && damage_counter_any.continuations.values[2].call_count == 6
                    && damage_counter_any.continuations.values[2].called_count == 3,
                "DamageCounterAny repeated call progress");
            require(
                official_consume_effect_selection_continuation_mirror(
                    &damage_counter_any),
                "DamageCounterAny repeated continuation consume");
            damage_counter_any.remain_damage_counter = 1;
            require(
                official_build_effect_selection_continuation_mirror(
                    &damage_counter_any)
                    && damage_counter_any.continuations.count == 3
                    && damage_counter_any.continuations.values[1].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterEffect)
                    && damage_counter_any.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedDamageCounterAny),
                "DamageCounterAny final call omits completed repeat frame");

            OfficialStatePod damage_counter_resume = pod;
            damage_counter_resume.effect_interpreter.active = 1;
            damage_counter_resume.effect_interpreter.effect_owner = 0;
            damage_counter_resume.effect_interpreter.effect_count = 1;
            damage_counter_resume.effect_interpreter.damage_counter_any_total = 2;
            damage_counter_resume.effect_state.on_effect = 1;
            damage_counter_resume.effect_state.ability.effect_card =
                official_pod_area_ref(
                    &damage_counter_resume,
                    OfficialCardRefPod{kAttackerRef});
            damage_counter_resume.remain_damage_counter = 2;
            damage_counter_resume.targets.count = 0;
            require(
                official_pod_push(
                    &damage_counter_resume,
                    &damage_counter_resume.targets,
                    official_pod_area_ref(
                        &damage_counter_resume,
                        OfficialCardRefPod{kEnemyActiveRef}),
                    OfficialPodError::kSelectionOverflow),
                "DamageCounterAny resume target");
            const std::int32_t enemy_max_hp = official_pod_max_hp(
                &damage_counter_resume,
                rules,
                OfficialCardRefPod{kEnemyActiveRef});
            damage_counter_resume.cards[kEnemyActiveRef].damage = enemy_max_hp - 10;
            damage_counter_resume.cards[kEnemyActiveRef].runtime_flags &=
                ~static_cast<std::uint64_t>(kCardKo);
            require(
                official_prepare_damage_counter_any_selection(
                    &damage_counter_resume)
                    == OfficialEffectInterpreterResult::kNeedsAction
                    && damage_counter_resume.options.count == 1
                    && damage_counter_resume.options.values[0].resolved_card
                        == kEnemyActiveRef,
                "DamageCounterAny resume selection");
            const std::uint16_t damage_target = 0;
            const OfficialEffectInterpreterResult damage_counter_result =
                official_apply_effect_action(
                    &damage_counter_resume,
                    rules,
                    &damage_target,
                    1);
            require(
                damage_counter_result
                    == OfficialEffectInterpreterResult::kNeedsAction,
                "DamageCounterAny resumes repeated selection");
            require(
                damage_counter_resume.cards[kEnemyActiveRef].damage
                    == enemy_max_hp,
                "DamageCounterAny places damage without clamping early");
            require(
                (damage_counter_resume.cards[kEnemyActiveRef].runtime_flags
                    & kCardKo) == 0,
                "DamageCounterAny defers KO marking to official KO flow");
            checks += 10;
        }

        {
            OfficialStatePod pre_attack_selection = pod;
            pre_attack_selection.attack_flow_stage = static_cast<std::uint8_t>(
                OfficialAttackStage::kPreEffects);
            pre_attack_selection.effect_interpreter.active = 1;
            pre_attack_selection.effect_interpreter.awaiting_selection = 1;
            pre_attack_selection.effect_interpreter.resume_kind =
                static_cast<std::uint8_t>(
                    OfficialEffectResumeKind::kApplyPrimitive);
            pre_attack_selection.effect_interpreter.effect_count = 2;
            pre_attack_selection.effect_interpreter.effect_index = 0;
            pre_attack_selection.select_type = static_cast<std::uint8_t>(
                OfficialSelectTypeId::kAttachedCard);
            pre_attack_selection.options.count = 1;
            pre_attack_selection.options.values[0].type =
                static_cast<std::uint8_t>(
                    OfficialSelectOptionTypeId::kEnergyCard);
            require(
                official_build_effect_selection_continuation_mirror(
                    &pre_attack_selection)
                    && pre_attack_selection.continuations.count == 4
                    && pre_attack_selection.continuations.values[0].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAttackEffects)
                    && pre_attack_selection.continuations.values[0].arg_type == 2
                    && pre_attack_selection.continuations.values[0].args[0] == 0
                    && pre_attack_selection.continuations.values[1].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAttackDamage)
                    && pre_attack_selection.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAttackEffect)
                    && pre_attack_selection.continuations.values[2].call_count == 2
                    && pre_attack_selection.continuations.values[2].called_count == 1
                    && pre_attack_selection.continuations.values[3].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedEnergyTarget),
                "pre-attack effect mirrors attack continuation stack");
            require(
                official_consume_effect_selection_continuation_mirror(
                    &pre_attack_selection)
                    && pre_attack_selection.continuations.count == 0,
                "pre-attack effect continuation consume");
            checks += 2;
        }

        {
            OfficialStatePod moved_effected_target = pod;
            OfficialTargetRule effected_rule{};
            effected_rule.values[kTargetAreaCount] = 1;
            effected_rule.values[kTargetArea0] = static_cast<std::int32_t>(
                OfficialArea::kEffected);
            moved_effected_target.targets.count = 1;
            moved_effected_target.targets.values[0] = official_pod_area_ref(
                &moved_effected_target,
                OfficialCardRefPod{kFirstEnergyRef});
            const std::int32_t old_move_counter =
                moved_effected_target.targets.values[0].move_counter;
            require(
                official_pod_move_card(
                    &moved_effected_target,
                    0,
                    OfficialArea::kEnergy,
                    0,
                    OfficialArea::kTrash,
                    false).index == kFirstEnergyRef,
                "move Effected target before rebuilding target list");
            require(
                official_build_target_list(
                    &moved_effected_target,
                    rules,
                    effected_rule,
                    &moved_effected_target.targets,
                    official_pod_area_ref(
                        &moved_effected_target,
                        OfficialCardRefPod{kAttackerRef}),
                    0)
                    && moved_effected_target.targets.count == 1
                    && moved_effected_target.targets.values[0].card.index
                        == kFirstEnergyRef
                    && moved_effected_target.targets.values[0].move_counter
                        != old_move_counter
                    && official_pod_area_ref_valid(
                        &moved_effected_target,
                        moved_effected_target.targets.values[0]),
                "Effected refreshes AreaRef after the prior effect moves a card");
            checks += 2;
        }

        {
            OfficialStatePod repeat_selection = pod;
            repeat_selection.flow_flags = kOfficialPlayEffectReturnToMainFlag;
            repeat_selection.effect_interpreter.active = 1;
            repeat_selection.effect_interpreter.awaiting_selection = 1;
            repeat_selection.effect_interpreter.resume_kind = static_cast<std::uint8_t>(
                OfficialEffectResumeKind::kApplyPrimitive);
            repeat_selection.effect_interpreter.effect_count = 5;
            repeat_selection.effect_interpreter.effect_index = 3;
            repeat_selection.effect_interpreter.repeat_count = 5;
            repeat_selection.effect_interpreter.repeat_index = 3;
            repeat_selection.effect_interpreter.repeat_continuation =
                static_cast<std::uint16_t>(
                    OfficialContinuationId::kActivateEffectEachSelected);
            repeat_selection.select_type = static_cast<std::uint8_t>(
                OfficialSelectTypeId::kCard);
            require(
                official_build_effect_selection_continuation_mirror(
                    &repeat_selection),
                "each-selected continuation mirror");
            require(
                repeat_selection.continuations.count == 5,
                "each-selected continuation count");
            const OfficialContinuationPod& repeat_frame =
                repeat_selection.continuations.values[3];
            require(
                repeat_frame.opcode == static_cast<std::uint16_t>(
                    OfficialContinuationId::kActivateEffectEachSelected)
                    && repeat_frame.arg_type == 1
                    && repeat_frame.args[0] == 3
                    && repeat_frame.call_count == 5
                    && repeat_frame.called_count == 4,
                "each-selected continuation fields");
            require(
                official_consume_effect_selection_continuation_mirror(
                    &repeat_selection),
                "each-selected continuation consume");
            repeat_selection.effect_interpreter.repeat_index = 4;
            require(
                official_build_effect_selection_continuation_mirror(
                    &repeat_selection),
                "last each-selected continuation mirror");
            require(
                repeat_selection.continuations.count == 4
                    && repeat_selection.continuations.values[3].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedEffectTarget),
                "last each-selected continuation omits completed parent");
            checks += 6;
        }

        {
            OfficialStatePod separator_selection = pod;
            separator_selection.flow_flags = kOfficialPlayEffectReturnToMainFlag;
            separator_selection.effect_interpreter.active = 1;
            separator_selection.effect_interpreter.awaiting_selection = 1;
            separator_selection.effect_interpreter.resume_kind =
                static_cast<std::uint8_t>(
                    OfficialEffectResumeKind::kApplyPrimitive);
            separator_selection.effect_interpreter.effect_count = 3;
            separator_selection.effect_interpreter.effect_index = 1;
            separator_selection.effect_interpreter.repeat_count = 1;
            separator_selection.effect_interpreter.separator_pending = 1;
            separator_selection.select_type = static_cast<std::uint8_t>(
                OfficialSelectTypeId::kCard);
            require(
                official_build_effect_selection_continuation_mirror(
                    &separator_selection),
                "separator continuation mirror");
            require(
                separator_selection.continuations.count == 5
                    && separator_selection.continuations.values[3].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSeparatorProc)
                    && separator_selection.continuations.values[4].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedEffectTarget),
                "separator continuation precedes selection callback");
            require(
                official_consume_effect_selection_continuation_mirror(
                    &separator_selection),
                "separator continuation consume");
            checks += 3;
        }

        {
            constexpr std::uint16_t attach_ref = 18;
            OfficialStatePod attach_from_each = pod;
            add_pod_card(
                &attach_from_each,
                attach_ref,
                4,
                0,
                OfficialArea::kHand);
            attach_from_each.selected_list.count = 0;
            require(
                official_pod_push(
                    &attach_from_each,
                    &attach_from_each.selected_list,
                    OfficialCardRefPod{attach_ref},
                    OfficialPodError::kSelectionOverflow),
                "attach-from-each selected card");
            attach_from_each.effect_state.selected_list_index = 0;
            attach_from_each.targets.count = 0;
            require(
                official_pod_push(
                    &attach_from_each,
                    &attach_from_each.targets,
                    official_pod_area_ref(
                        &attach_from_each,
                        OfficialCardRefPod{kAttackerRef}),
                    OfficialPodError::kSelectionOverflow),
                "attach-from-each target");
            attach_from_each.control_flags &= static_cast<std::uint8_t>(
                ~kOfficialChangedFlag);
            OfficialEffectRule attach_rule{};
            attach_rule.values[kEffectType] = static_cast<int>(
                OfficialEffectTypeId::kAttachFromEach);
            require(
                official_apply_effect_primitive(
                    &attach_from_each, rules, attach_rule)
                    == OfficialEffectApplyResult::kApplied,
                "attach-from-each primitive");
            require(
                (attach_from_each.control_flags & kOfficialChangedFlag) != 0,
                "attach-from-each marks changed");
            require(
                attach_from_each.cards[attach_ref].area
                        == static_cast<std::uint8_t>(OfficialArea::kEnergy)
                    && attach_from_each.cards[attach_ref].attach_move_counter
                        == attach_from_each.cards[kAttackerRef].move_counter,
                "attach-from-each attaches selected card");
            checks += 3;
        }

        {
            OfficialStatePod prevented_effect_damage = pod;
            prevented_effect_damage.current_attack_id = 517;
            prevented_effect_damage.targets.count = 0;
            require(
                official_pod_push(
                    &prevented_effect_damage,
                    &prevented_effect_damage.targets,
                    official_pod_area_ref(
                        &prevented_effect_damage,
                        OfficialCardRefPod{kEnemyActiveRef}),
                    OfficialPodError::kSelectionOverflow),
                "zero effect-attack target");
            prevented_effect_damage.control_flags &= static_cast<std::uint8_t>(
                ~kOfficialChangedFlag);
            const std::int32_t damage_before =
                prevented_effect_damage.cards[kEnemyActiveRef].damage;
            OfficialEffectRule zero_attack_damage{};
            zero_attack_damage.values[kEffectType] = static_cast<int>(
                OfficialEffectTypeId::kAttackDamage);
            zero_attack_damage.values[kEffectValue0] = 0;
            require(
                official_apply_effect_primitive(
                    &prevented_effect_damage, rules, zero_attack_damage)
                    == OfficialEffectApplyResult::kApplied,
                "zero effect-attack primitive");
            require(
                prevented_effect_damage.cards[kEnemyActiveRef].damage == damage_before,
                "zero effect-attack applies no damage");
            require(
                (prevented_effect_damage.control_flags & kOfficialChangedFlag) != 0,
                "effect attack marks changed even when applied damage is zero");
            checks += 3;
        }

        {
            OfficialStatePod no_attack_effect = pod;
            no_attack_effect.targets.count = 0;
            require(
                official_pod_push(
                    &no_attack_effect,
                    &no_attack_effect.targets,
                    official_pod_area_ref(
                        &no_attack_effect,
                        OfficialCardRefPod{kAttackerRef}),
                    OfficialPodError::kSelectionOverflow),
                "no-effect-enemy-attack target");
            OfficialEffectRule no_effect_rule{};
            no_effect_rule.values[kEffectType] = static_cast<std::int32_t>(
                OfficialEffectTypeId::kNoEffectEnemyAttack);
            require(
                official_apply_continual_effect_primitive(
                    &no_attack_effect,
                    rules,
                    no_effect_rule,
                    0,
                    OfficialCardRefPod{kAttackerRef})
                    == OfficialEffectApplyResult::kApplied,
                "no-effect-enemy-attack continual primitive");
            require(
                official_continual_flag(
                    no_attack_effect.cards[kAttackerRef], 8),
                "no-effect-enemy-attack uses official bit");
            require(
                !official_continual_flag(
                    no_attack_effect.cards[kAttackerRef], 7),
                "no-effect-enemy-attack does not set damage-protection bit");
            checks += 4;
        }

        {
            // Rock Fighting Energy's NoEffectEnemyAttack flag blocks attack
            // effects such as Powerful Hand's damage counters.  It does not
            // represent ordinary attack-damage prevention.
            OfficialStatePod protected_from_attack_effect = pod;
            protected_from_attack_effect.current_attack_id = 517;
            protected_from_attack_effect.attacker =
                OfficialCardRefPod{kAttackerRef};
            protected_from_attack_effect.effect_state.on_effect = 1;
            protected_from_attack_effect.effect_state.ability.effect_card =
                official_pod_area_ref(
                    &protected_from_attack_effect,
                    OfficialCardRefPod{kAttackerRef});
            protected_from_attack_effect.cards[kEnemyActiveRef]
                .continual_state[4] |= 1ULL << 8U;
            protected_from_attack_effect.targets.count = 0;
            require(
                official_pod_push(
                    &protected_from_attack_effect,
                    &protected_from_attack_effect.targets,
                    official_pod_area_ref(
                        &protected_from_attack_effect,
                        OfficialCardRefPod{kEnemyActiveRef}),
                    OfficialPodError::kSelectionOverflow),
                "attack-effect immunity target");
            const std::int32_t damage_before =
                protected_from_attack_effect.cards[kEnemyActiveRef].damage;
            OfficialEffectRule damage_counter{};
            damage_counter.values[kEffectType] = static_cast<std::int32_t>(
                OfficialEffectTypeId::kDamageCounter);
            damage_counter.values[kEffectValue0] = 6;
            require(
                official_apply_effect_primitive(
                    &protected_from_attack_effect,
                    rules,
                    damage_counter)
                    == OfficialEffectApplyResult::kApplied,
                "attack-effect immunity primitive");
            require(
                protected_from_attack_effect.cards[kEnemyActiveRef].damage
                    == damage_before,
                "no-effect-enemy-attack blocks damage counters");
            require(
                !official_continual_flag(
                    protected_from_attack_effect.cards[kEnemyActiveRef], 7),
                "attack-effect immunity remains distinct from damage immunity");
            checks += 4;
        }

        {
            OfficialStatePod heal_record = pod;
            heal_record.cards[kAttackerRef].damage = 30;
            require(
                official_pod_heal(
                    &heal_record,
                    OfficialCardRefPod{kAttackerRef},
                    20,
                    true) == 20,
                "effect heal amount");
            require(
                heal_record.turn_heal.count == 1
                    && heal_record.turn_heal.values[0].index == kAttackerRef,
                "effect heal records turn-heal target");
            require(
                official_pod_heal(
                    &heal_record,
                    OfficialCardRefPod{kAttackerRef},
                    10) == 10
                    && heal_record.turn_heal.count == 1,
                "counter removal heal does not record turn-heal target");
            checks += 3;
        }

        {
            OfficialStatePod shadow_bullet = pod;
            shadow_bullet.first_player = 0;
            shadow_bullet.turn = 1;
            add_pod_card(
                &shadow_bullet,
                kEnemyBenchRef0,
                197,
                1,
                OfficialArea::kBench);
            add_pod_card(
                &shadow_bullet,
                kEnemyBenchRef1,
                197,
                1,
                OfficialArea::kBench);
            shadow_bullet.targets.count = 0;
            require(
                official_pod_push(
                    &shadow_bullet,
                    &shadow_bullet.targets,
                    official_pod_area_ref(
                        &shadow_bullet,
                        OfficialCardRefPod{kEnemyActiveRef}),
                    OfficialPodError::kSelectionOverflow),
                "shadow bullet initial attack target");
            const OfficialEffectInterpreterResult shadow_result =
                official_begin_attack_effects(
                    &shadow_bullet,
                    rules,
                    937,
                    true,
                    official_pod_area_ref(
                        &shadow_bullet,
                        OfficialCardRefPod{kAttackerRef}),
                    0);
            require(
                shadow_result == OfficialEffectInterpreterResult::kNeedsAction,
                "shadow bullet bench selection");
            require(
                shadow_bullet.pre_targets.count == 1
                    && shadow_bullet.pre_targets.values[0].card.index
                        == kEnemyActiveRef,
                "attack post effect inherits active pre-target");
            require(
                shadow_bullet.targets.count == 2
                    && shadow_bullet.targets.values[0].card.index
                        == kEnemyBenchRef0
                    && shadow_bullet.targets.values[1].card.index
                        == kEnemyBenchRef1,
                "shadow bullet targets enemy bench");
            checks += 4;
        }

        {
            OfficialStatePod ability_scratch = pod;
            ability_scratch.targets.count = 1;
            ability_scratch.pre_targets.count = 1;
            ability_scratch.selected_list.count = 1;
            ability_scratch.each_list.count = 1;
            ability_scratch.check_list.count = 1;
            ability_scratch.removed_damage_counter = 7;
            ability_scratch.effect_jump = 2;
            ability_scratch.attach_active = 1;
            official_clear_resolved_ability(&ability_scratch);
            require(
                ability_scratch.targets.count == 0
                    && ability_scratch.pre_targets.count == 0
                    && ability_scratch.selected_list.count == 0
                    && ability_scratch.each_list.count == 0
                    && ability_scratch.check_list.count == 0,
                "clear ability reference lists");
            require(
                ability_scratch.removed_damage_counter == 0
                    && ability_scratch.effect_jump == 0
                    && ability_scratch.attach_active == 0,
                "clear ability scalar scratch");
            checks += 2;
        }

        {
            OfficialEffectRule no_same_skill{};
            no_same_skill.values[kEffectConditionType] = static_cast<std::int32_t>(
                OfficialConditionTypeId::kNoSameNameSkillThisTurn);
            no_same_skill.values[kEffectComparator] = 0;
            no_same_skill.values[kEffectSkillId] = 55;

            OfficialStatePod resolving_first_use = pod;
            resolving_first_use.turn_used_skills.count = 1;
            resolving_first_use.turn_used_skills.values[0] = 55;
            resolving_first_use.effect_interpreter.active = 1;
            resolving_first_use.effect_state.on_effect = 1;
            resolving_first_use.effect_state.ability.skill_id = 55;
            require(
                official_satisfy_condition(
                    &resolving_first_use,
                    rules,
                    &no_same_skill,
                    1,
                    0,
                    official_pod_area_ref(
                        &resolving_first_use,
                        OfficialCardRefPod{kAttackerRef}),
                    0)
                    == OfficialConditionResult::kTrue,
                "current skill does not reject itself");

            resolving_first_use.turn_used_skills.count = 2;
            resolving_first_use.turn_used_skills.values[1] = 55;
            require(
                official_satisfy_condition(
                    &resolving_first_use,
                    rules,
                    &no_same_skill,
                    1,
                    0,
                    official_pod_area_ref(
                        &resolving_first_use,
                        OfficialCardRefPod{kAttackerRef}),
                    0)
                    == OfficialConditionResult::kFalse,
                "earlier same-name skill still rejects repeat");

            resolving_first_use.turn_used_skills.count = 1;
            resolving_first_use.effect_interpreter.active = 0;
            resolving_first_use.effect_state.on_effect = 0;
            require(
                official_satisfy_condition(
                    &resolving_first_use,
                    rules,
                    &no_same_skill,
                    1,
                    0,
                    official_pod_area_ref(
                        &resolving_first_use,
                        OfficialCardRefPod{kAttackerRef}),
                    0)
                    == OfficialConditionResult::kFalse,
                "recorded same-name skill rejects outside interpreter");
            checks += 3;
        }

        {
            constexpr std::uint16_t look_ref = 19;
            OfficialStatePod look_deck = pod;
            look_deck.players[0].deck.count = 0;
            look_deck.looking.count = 0;
            add_pod_card(
                &look_deck,
                look_ref,
                4,
                0,
                OfficialArea::kDeck);
            look_deck.effect_state.ability.use_player = 0;
            look_deck.control_flags &= static_cast<std::uint8_t>(
                ~kOfficialChangedFlag);
            const OfficialSkillRule* look_skill = official_skill_rule(rules, 284);
            require(look_skill != nullptr, "look-deck fixture skill");
            const OfficialEffectRule& look_rule = rules.effects[
                look_skill->values[kSkillEffectOffset] + 1];
            require(
                official_apply_effect_primitive(
                    &look_deck, rules, look_rule)
                    == OfficialEffectApplyResult::kApplied,
                "look-deck fixture effect");
            require(
                (look_deck.control_flags & kOfficialChangedFlag) != 0,
                "look-deck marks changed");
            require(
                look_deck.looking.count == 1
                    && look_deck.looking.values[0].index == look_ref,
                "look-deck moves card to looking");
            checks += 3;
        }

        {
            OfficialStatePod bench_selection{};
            bench_selection.abi_version = kOfficialStateAbiVersion;
            bench_selection.effect_interpreter.effect_owner = 0;
            bench_selection.effect_state.ability.use_player = 0;
            for (std::uint16_t ref = 20; ref < 24; ++ref) {
                add_pod_card(
                    &bench_selection,
                    ref,
                    707,
                    0,
                    OfficialArea::kBench);
            }
            for (std::uint16_t ref = 24; ref < 26; ++ref) {
                add_pod_card(
                    &bench_selection,
                    ref,
                    741,
                    0,
                    OfficialArea::kDeck);
                require(
                    official_pod_push(
                        &bench_selection,
                        &bench_selection.targets,
                        official_pod_area_ref(
                            &bench_selection, OfficialCardRefPod{ref}),
                        OfficialPodError::kSelectionOverflow),
                    "bench selection target");
            }
            const OfficialSkillRule* poffin_skill = official_skill_rule(rules, 247);
            require(poffin_skill != nullptr, "Buddy-Buddy Poffin skill");
            const OfficialEffectRule* to_bench = nullptr;
            for (std::int32_t index = 0;
                 index < poffin_skill->values[kSkillEffectCount];
                 ++index) {
                const OfficialEffectRule& candidate = rules.effects[
                    poffin_skill->values[kSkillEffectOffset] + index];
                if (candidate.values[kEffectType]
                    == static_cast<std::int32_t>(OfficialEffectTypeId::kToBench)) {
                    to_bench = &candidate;
                    break;
                }
            }
            require(to_bench != nullptr, "Buddy-Buddy Poffin ToBench effect");
            require(
                official_prepare_effect_selection(
                    &bench_selection,
                    rules,
                    *to_bench,
                    static_cast<OfficialEffectSelectTypeId>(
                        to_bench->values[kEffectSelectType]),
                    2)
                    == OfficialEffectInterpreterResult::kNeedsAction,
                "bench capacity selection needs action");
            require(
                bench_selection.select_min == 0
                    && bench_selection.select_max == 1,
                "bench capacity clamps selection maximum: min="
                    + std::to_string(bench_selection.select_min)
                    + ",max=" + std::to_string(bench_selection.select_max)
                    + ",bench="
                    + std::to_string(bench_selection.players[0].bench.count)
                    + ",mask="
                    + std::to_string(official_effect_player_mask(
                        &bench_selection, rules, *to_bench)));
            checks += 4;
        }

        {
            constexpr std::uint16_t source_ref = 26;
            constexpr std::uint16_t damaged_ref = 27;
            OfficialStatePod remove_damage{};
            remove_damage.abi_version = kOfficialStateAbiVersion;
            add_pod_card(
                &remove_damage,
                source_ref,
                112,
                0,
                OfficialArea::kBench);
            add_pod_card(
                &remove_damage,
                damaged_ref,
                707,
                0,
                OfficialArea::kActive);
            remove_damage.cards[damaged_ref].damage = 30;

            const OfficialSkillRule* adrena_brain = official_skill_rule(rules, 42);
            require(adrena_brain != nullptr, "Adrena-Brain skill");
            std::int32_t remove_index = -1;
            for (std::int32_t index = 0;
                 index < adrena_brain->values[kSkillEffectCount];
                 ++index) {
                const OfficialEffectRule& candidate = rules.effects[
                    adrena_brain->values[kSkillEffectOffset] + index];
                if (candidate.values[kEffectType]
                    == static_cast<std::int32_t>(
                        OfficialEffectTypeId::kRemoveDamageCounter)) {
                    remove_index = index;
                    break;
                }
            }
            require(remove_index >= 0, "Adrena-Brain remove-damage effect");

            remove_damage.effect_interpreter.effect_offset = static_cast<std::uint32_t>(
                adrena_brain->values[kSkillEffectOffset]);
            remove_damage.effect_interpreter.effect_count = static_cast<std::uint16_t>(
                remove_index + 1);
            remove_damage.effect_interpreter.effect_index = static_cast<std::uint16_t>(
                remove_index);
            remove_damage.effect_interpreter.effect_owner = 0;
            remove_damage.effect_interpreter.effect_card = official_pod_area_ref(
                &remove_damage, OfficialCardRefPod{source_ref});
            remove_damage.effect_interpreter.active = 1;
            remove_damage.effect_interpreter.awaiting_selection = 1;
            remove_damage.effect_interpreter.resume_kind = static_cast<std::uint8_t>(
                OfficialEffectResumeKind::kApplyPrimitive);
            remove_damage.effect_interpreter.repeat_count = 1;
            remove_damage.effect_interpreter.step_budget = 32;
            remove_damage.effect_state.ability.effect_card =
                remove_damage.effect_interpreter.effect_card;
            remove_damage.effect_state.ability.use_player = 0;
            remove_damage.select_min = 1;
            remove_damage.select_max = 1;
            remove_damage.options.count = 1;
            remove_damage.options.values[0].type = static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kCard);
            remove_damage.options.values[0].resolved_card = damaged_ref;

            const std::uint16_t target_choice = 0;
            require(
                official_apply_effect_action(
                    &remove_damage, rules, &target_choice, 1)
                    == OfficialEffectInterpreterResult::kNeedsAction,
                "remove damage counter asks for count");
            require(
                remove_damage.select_type
                        == static_cast<std::uint8_t>(OfficialSelectTypeId::kCount)
                    && remove_damage.select_context
                        == kOfficialSelectContextRemoveDamageCounterCount
                    && remove_damage.options.count == 3,
                "remove damage counter count options");
            const std::uint16_t two_counters = 1;
            require(
                official_apply_effect_action(
                    &remove_damage, rules, &two_counters, 1)
                    == OfficialEffectInterpreterResult::kComplete,
                "remove damage counter resumes");
            require(
                remove_damage.cards[damaged_ref].damage == 10
                    && remove_damage.removed_damage_counter == 2,
                "remove damage counter applies selected count");
            checks += 4;
        }

        {
            constexpr std::uint16_t source_ref = 28;
            constexpr std::uint16_t damaged_ref = 29;
            OfficialStatePod remove_all{};
            remove_all.abi_version = kOfficialStateAbiVersion;
            add_pod_card(
                &remove_all,
                source_ref,
                432,
                0,
                OfficialArea::kActive);
            add_pod_card(
                &remove_all,
                damaged_ref,
                432,
                0,
                OfficialArea::kBench);
            remove_all.cards[damaged_ref].damage = 70;

            const OfficialAttackRule* rocket_mirror = official_attack_rule(rules, 609);
            require(rocket_mirror != nullptr, "Rocket Mirror attack");
            const std::uint32_t post_offset = static_cast<std::uint32_t>(
                rocket_mirror->values[kAttackPostEffectOffset]);
            const std::uint16_t post_count = static_cast<std::uint16_t>(
                rocket_mirror->values[kAttackPostEffectCount]);
            require(post_count > 0, "Rocket Mirror post effects");
            std::int32_t remove_all_index = -1;
            for (std::uint16_t index = 0; index < post_count; ++index) {
                if (rules.effects[post_offset + index].values[kEffectType]
                    == static_cast<std::int32_t>(
                        OfficialEffectTypeId::kRemoveDamageCounterAll)) {
                    remove_all_index = index;
                    break;
                }
            }
            require(remove_all_index >= 0, "Rocket Mirror remove-all effect");

            remove_all.effect_interpreter.effect_offset = post_offset;
            remove_all.effect_interpreter.effect_count = static_cast<std::uint16_t>(
                remove_all_index + 1);
            remove_all.effect_interpreter.effect_index = static_cast<std::uint16_t>(
                remove_all_index);
            remove_all.effect_interpreter.effect_owner = 0;
            remove_all.effect_interpreter.effect_card = official_pod_area_ref(
                &remove_all, OfficialCardRefPod{source_ref});
            remove_all.effect_interpreter.active = 1;
            remove_all.effect_interpreter.awaiting_selection = 1;
            remove_all.effect_interpreter.resume_kind = static_cast<std::uint8_t>(
                OfficialEffectResumeKind::kApplyPrimitive);
            remove_all.effect_interpreter.repeat_count = 1;
            remove_all.effect_interpreter.step_budget = 32;
            remove_all.effect_state.ability.effect_card =
                remove_all.effect_interpreter.effect_card;
            remove_all.effect_state.ability.use_player = 0;
            remove_all.select_min = 1;
            remove_all.select_max = 1;
            remove_all.options.count = 1;
            remove_all.options.values[0].type = static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kCard);
            remove_all.options.values[0].resolved_card = damaged_ref;

            const std::uint16_t target_choice = 0;
            require(
                official_apply_effect_action(
                    &remove_all, rules, &target_choice, 1)
                    == OfficialEffectInterpreterResult::kComplete,
                "remove all damage counters completes without count selection");
            require(
                remove_all.cards[damaged_ref].damage == 0
                    && remove_all.removed_damage_counter == 7,
                "remove all damage counters applies full count");
            checks += 5;
        }

        {
            const OfficialSkillRule* janine = official_skill_rule(rules, 361);
            require(janine != nullptr, "Janine's Secret Art skill");
            std::int32_t each_selected_index = -1;
            for (std::int32_t index = 0;
                 index < janine->values[kSkillEffectCount];
                 ++index) {
                const OfficialEffectRule& candidate = rules.effects[
                    janine->values[kSkillEffectOffset] + index];
                if ((candidate.flags & kOfficialEffectEachSelectedList) != 0) {
                    each_selected_index = index;
                    break;
                }
            }
            require(each_selected_index >= 0, "Janine each-selected effect");

            OfficialStatePod empty_repeat{};
            empty_repeat.abi_version = kOfficialStateAbiVersion;
            empty_repeat.flow_flags = 1U << 5U;
            empty_repeat.effect_interpreter.effect_offset = static_cast<std::uint32_t>(
                janine->values[kSkillEffectOffset]);
            empty_repeat.effect_interpreter.effect_count = static_cast<std::uint16_t>(
                each_selected_index + 1);
            empty_repeat.effect_interpreter.effect_index = static_cast<std::uint16_t>(
                each_selected_index);
            empty_repeat.effect_interpreter.effect_owner = 0;
            empty_repeat.effect_interpreter.active = 1;
            empty_repeat.effect_interpreter.repeat_count = 2;
            empty_repeat.effect_interpreter.repeat_index = 0;
            empty_repeat.effect_interpreter.repeat_continuation =
                static_cast<std::uint16_t>(
                    OfficialContinuationId::kActivateEffectEachSelected);
            const OfficialEffectRule& each_selected = rules.effects[
                empty_repeat.effect_interpreter.effect_offset
                + empty_repeat.effect_interpreter.effect_index];
            require(
                official_prepare_effect_selection(
                    &empty_repeat,
                    rules,
                    each_selected,
                    static_cast<OfficialEffectSelectTypeId>(
                        each_selected.values[kEffectSelectType]),
                    1)
                    == OfficialEffectInterpreterResult::kComplete,
                "Janine empty first repeat auto-completes");
            require(
                empty_repeat.turn_action_count == 1,
                "Janine empty first repeat records State::step boundary");
            checks += 4;
        }

        {
            constexpr std::uint16_t old_basic_ref = 30;
            constexpr std::uint16_t new_basic_ref = 31;
            constexpr std::uint16_t stage2_ref = 32;
            OfficialStatePod rare_candy{};
            rare_candy.abi_version = kOfficialStateAbiVersion;
            add_pod_card(
                &rare_candy,
                old_basic_ref,
                741,
                0,
                OfficialArea::kBench);
            add_pod_card(
                &rare_candy,
                new_basic_ref,
                741,
                0,
                OfficialArea::kBench);
            rare_candy.cards[new_basic_ref].runtime_flags |= kCardAppear;
            add_pod_card(
                &rare_candy,
                stage2_ref,
                743,
                0,
                OfficialArea::kHand);
            require(
                official_pod_push(
                    &rare_candy,
                    &rare_candy.targets,
                    official_pod_area_ref(
                        &rare_candy, OfficialCardRefPod{stage2_ref}),
                    OfficialPodError::kSelectionOverflow),
                "Rare Candy evolution target");
            require(
                official_prepare_evolve_options(
                    &rare_candy, rules, 0, true),
                "Rare Candy evolution options");
            require(
                rare_candy.options.count == 1
                    && rare_candy.options.values[0].params[3] == 0,
                "Rare Candy excludes newly appeared Basic Pokemon");
            checks += 2;
        }

        {
            // Hand Trimmer's CardUntil effect is a no-op when the target
            // hand is already at or below the requested limit.  It must keep
            // the target list for the next effect and skip the primitive.
            OfficialStatePod hand_trimmer{};
            official_pod_reset(&hand_trimmer, 99, 20260730);
            for (std::uint16_t ref = 70; ref < 75; ++ref) {
                add_pod_card(&hand_trimmer, ref, 197, 1, OfficialArea::kHand);
                require(
                    official_pod_push(
                        &hand_trimmer,
                        &hand_trimmer.targets,
                        official_pod_area_ref(&hand_trimmer, OfficialCardRefPod{ref}),
                        OfficialPodError::kSelectionOverflow),
                    "Hand Trimmer target");
            }
            const OfficialSkillRule* hand_skill = official_skill_rule(rules, 248);
            require(hand_skill != nullptr, "Hand Trimmer skill");
            const OfficialEffectRule& hand_effect = rules.effects[
                hand_skill->values[kSkillEffectOffset] + 1];
            hand_trimmer.effect_interpreter.effect_index = 1;
            hand_trimmer.effect_interpreter.effect_owner = 0;
            hand_trimmer.turn_action_count = 77;
            require(
                official_prepare_effect_selection(
                    &hand_trimmer,
                    rules,
                    hand_effect,
                    OfficialEffectSelectTypeId::kCardUntil,
                    5)
                    == OfficialEffectInterpreterResult::kComplete,
                "Hand Trimmer <= limit completes without action");
            require(
                hand_trimmer.targets.count == 5
                    && hand_trimmer.targets.values[0].card.index == 70
                    && hand_trimmer.targets.values[4].card.index == 74,
                "Hand Trimmer preserves targets");
            require(
                hand_trimmer.turn_action_count == 77
                    && hand_trimmer.players[1].trash.count == 0,
                "Hand Trimmer skips primitive and action count");
            checks += 3;
        }

        {
            // Skill first conditions are checked while exposing the action;
            // executing the effect range must skip those frames.
            OfficialStatePod first_condition{};
            official_pod_reset(&first_condition, 99, 20260730);
            add_pod_card(&first_condition, 78, 1087, 0, OfficialArea::kHand);
            const OfficialSkillRule* hand_skill = official_skill_rule(rules, 248);
            require(hand_skill != nullptr, "Hand Trimmer first-condition skill");
            const OfficialEffectInterpreterResult result = official_begin_skill_effects(
                &first_condition,
                rules,
                248,
                official_pod_area_ref(&first_condition, OfficialCardRefPod{78}),
                0,
                64);
            require(
                result == OfficialEffectInterpreterResult::kComplete
                    && first_condition.effect_interpreter.active == 0
                    && first_condition.effect_interpreter.effect_index
                        == hand_skill->values[kSkillEffectCount]
                    && first_condition.effect_interpreter.first_condition_count
                        == hand_skill->values[kSkillFirstConditionCount],
                "Hand Trimmer first condition is not re-evaluated");
            checks += 1;
        }

        {
            // Trigger selections reached from AfterAttack use the official
            // AfterAttack2 -> AfterAttackTrigger continuation prefix, not the
            // refresh-to-main prefix used by ordinary main-flow triggers.
            OfficialStatePod attack_trigger_order{};
            official_pod_reset(&attack_trigger_order, 99, 20260730);
            attack_trigger_order.attack_flow_stage = static_cast<std::uint8_t>(
                OfficialAttackStage::kAfterAttackTriggers);
            attack_trigger_order.trigger_resolver.active = 1;
            attack_trigger_order.trigger_resolver.awaiting_order = 1;
            attack_trigger_order.trigger_resolver.depth = 2;
            attack_trigger_order.select_type = static_cast<std::uint8_t>(
                OfficialSelectTypeId::kSkill);
            attack_trigger_order.select_context = kOfficialSelectContextSkillOrder;
            require(
                official_build_trigger_order_continuation_mirror(
                    &attack_trigger_order),
                "after-attack trigger order continuation");
            require(
                attack_trigger_order.continuations.count == 3
                    && attack_trigger_order.continuations.values[0].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterAttack2)
                    && attack_trigger_order.continuations.values[1].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterAttackTrigger)
                    && attack_trigger_order.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedSkillOrder),
                "after-attack trigger order prefix");
            require(
                official_consume_trigger_order_continuation_mirror(
                    &attack_trigger_order)
                    && attack_trigger_order.continuations.count == 0,
                "after-attack trigger order continuation consume");

            OfficialStatePod attack_trigger_activation{};
            official_pod_reset(&attack_trigger_activation, 99, 20260730);
            attack_trigger_activation.attack_flow_stage =
                static_cast<std::uint8_t>(
                    OfficialAttackStage::kAfterAttackTriggers);
            attack_trigger_activation.trigger_resolver.active = 1;
            attack_trigger_activation.trigger_resolver.awaiting_activation = 1;
            attack_trigger_activation.trigger_resolver.activation_kind =
                static_cast<std::uint8_t>(
                    OfficialTriggerActivationKind::kOptional);
            attack_trigger_activation.trigger_resolver.depth = 2;
            require(
                official_build_trigger_activation_continuation_mirror(
                    &attack_trigger_activation),
                "after-attack trigger activation continuation");
            require(
                attack_trigger_activation.continuations.count == 4
                    && attack_trigger_activation.continuations.values[0].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterAttack2)
                    && attack_trigger_activation.continuations.values[1].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterAttackTrigger)
                    && attack_trigger_activation.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterTriggerAbility),
                "after-attack trigger activation prefix");
            require(
                official_consume_trigger_activation_continuation_mirror(
                    &attack_trigger_activation)
                    && attack_trigger_activation.continuations.count == 0,
                "after-attack trigger activation continuation consume");

            OfficialStatePod attack_post_ko_order{};
            official_pod_reset(&attack_post_ko_order, 99, 20260730);
            attack_post_ko_order.attack_flow_stage = static_cast<std::uint8_t>(
                OfficialAttackStage::kRefresh);
            attack_post_ko_order.refresh_flow_stage = static_cast<std::uint8_t>(
                OfficialRefreshFlowStage::kKnockout);
            attack_post_ko_order.reserved_flow = static_cast<std::uint8_t>(
                OfficialKnockoutStage::kPostKoTriggers);
            attack_post_ko_order.trigger_resolver.active = 1;
            attack_post_ko_order.trigger_resolver.awaiting_order = 1;
            attack_post_ko_order.trigger_resolver.depth = 1;
            attack_post_ko_order.select_type = static_cast<std::uint8_t>(
                OfficialSelectTypeId::kSkill);
            attack_post_ko_order.select_context = kOfficialSelectContextSkillOrder;
            require(
                official_build_trigger_order_continuation_mirror(
                    &attack_post_ko_order),
                "attack refresh post-KO trigger order continuation");
            require(
                attack_post_ko_order.continuations.count == 4
                    && attack_post_ko_order.continuations.values[0].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterAttack4)
                    && attack_post_ko_order.continuations.values[1].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterRefresh)
                    && attack_post_ko_order.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kKoProc3)
                    && attack_post_ko_order.continuations.values[3].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedSkillOrder),
                "attack refresh post-KO trigger order prefix");
            require(
                official_consume_trigger_order_continuation_mirror(
                    &attack_post_ko_order),
                "attack refresh post-KO trigger order consume");

            OfficialStatePod attack_pre_ko_order = attack_post_ko_order;
            attack_pre_ko_order.reserved_flow = static_cast<std::uint8_t>(
                OfficialKnockoutStage::kPreKoTriggers);
            require(
                official_build_trigger_order_continuation_mirror(
                    &attack_pre_ko_order),
                "attack refresh pre-KO trigger order continuation");
            require(
                attack_pre_ko_order.continuations.count == 4
                    && attack_pre_ko_order.continuations.values[0].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterAttack4)
                    && attack_pre_ko_order.continuations.values[1].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterRefresh)
                    && attack_pre_ko_order.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kKoProc2)
                    && attack_pre_ko_order.continuations.values[3].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedSkillOrder),
                "attack refresh pre-KO trigger order prefix");
            require(
                official_consume_trigger_order_continuation_mirror(
                    &attack_pre_ko_order),
                "attack refresh pre-KO trigger order consume");

            OfficialStatePod attack_post_ko_activation = attack_post_ko_order;
            attack_post_ko_activation.trigger_resolver.awaiting_order = 0;
            attack_post_ko_activation.trigger_resolver.awaiting_activation = 1;
            attack_post_ko_activation.trigger_resolver.activation_kind =
                static_cast<std::uint8_t>(
                    OfficialTriggerActivationKind::kOptional);
            require(
                official_build_trigger_activation_continuation_mirror(
                    &attack_post_ko_activation)
                    && attack_post_ko_activation.continuations.count == 5
                    && attack_post_ko_activation.continuations.values[0].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterAttack4)
                    && attack_post_ko_activation.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kKoProc3)
                    && attack_post_ko_activation.continuations.values[3].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterTriggerAbility),
                "attack refresh post-KO trigger activation prefix");
            require(
                official_consume_trigger_activation_continuation_mirror(
                    &attack_post_ko_activation),
                "attack refresh post-KO trigger activation consume");

            OfficialStatePod attack_post_ko_effect = attack_post_ko_order;
            attack_post_ko_effect.trigger_resolver.awaiting_order = 0;
            attack_post_ko_effect.effect_interpreter.active = 1;
            attack_post_ko_effect.effect_interpreter.awaiting_selection = 1;
            attack_post_ko_effect.effect_interpreter.effect_count = 1;
            attack_post_ko_effect.effect_interpreter.resume_kind =
                static_cast<std::uint8_t>(
                    OfficialEffectResumeKind::kApplyPrimitive);
            attack_post_ko_effect.select_type = static_cast<std::uint8_t>(
                OfficialSelectTypeId::kCard);
            require(
                official_build_effect_selection_continuation_mirror(
                    &attack_post_ko_effect)
                    && attack_post_ko_effect.continuations.count == 5
                    && attack_post_ko_effect.continuations.values[0].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterAttack4)
                    && attack_post_ko_effect.continuations.values[2].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kKoProc3)
                    && attack_post_ko_effect.continuations.values[3].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kAfterTriggerAbility)
                    && attack_post_ko_effect.continuations.values[4].opcode
                        == static_cast<std::uint16_t>(
                            OfficialContinuationId::kSelectedEffectTarget),
                "attack refresh post-KO effect selection prefix");
            require(
                official_consume_effect_selection_continuation_mirror(
                    &attack_post_ko_effect),
                "attack refresh post-KO effect selection consume");
            checks += 16;
        }

        {
            // Benching an AbilityPlay Basic from hand must pull its optional
            // HandToBench trigger before Main options are rebuilt.
            constexpr std::uint16_t meowth_ref = 91;
            OfficialStatePod hand_to_bench{};
            official_pod_reset(&hand_to_bench, 99, 20260730);
            hand_to_bench.phase = static_cast<std::uint8_t>(
                OfficialGamePhase::kMain);
            hand_to_bench.first_player = 0;
            hand_to_bench.turn = 1;
            add_pod_card(
                &hand_to_bench,
                kAttackerRef,
                372,
                0,
                OfficialArea::kActive);
            add_pod_card(
                &hand_to_bench,
                meowth_ref,
                1071,
                0,
                OfficialArea::kHand);
            add_pod_card(
                &hand_to_bench,
                92,
                197,
                1,
                OfficialArea::kActive);
            for (std::uint16_t ref = 93; ref < 96; ++ref) {
                add_pod_card(
                    &hand_to_bench,
                    ref,
                    197,
                    0,
                    OfficialArea::kPrize);
            }
            for (std::uint16_t ref = 96; ref < 99; ++ref) {
                add_pod_card(
                    &hand_to_bench,
                    ref,
                    197,
                    1,
                    OfficialArea::kPrize);
            }
            // Its search condition requires at least one matching card in
            // the owner's Deck before the optional activation is offered.
            add_pod_card(
                &hand_to_bench,
                99,
                197,
                0,
                OfficialArea::kDeck);
            hand_to_bench.select_type = static_cast<std::uint8_t>(
                OfficialSelectTypeId::kMain);
            hand_to_bench.options.count = 1;
            hand_to_bench.options.values[0].type = static_cast<std::uint8_t>(
                OfficialSelectOptionTypeId::kPlay);
            hand_to_bench.options.values[0].params[0] = 0;
            const std::uint16_t play_meowth = 0;
            const OfficialMainResult meowth_result = official_apply_main_action(
                &hand_to_bench, rules, &play_meowth, 1);
            require(
                meowth_result == OfficialMainResult::kNeedsAction,
                "AbilityPlay hand-to-bench trigger yields");
            require(
                hand_to_bench.cards[meowth_ref].area
                        == static_cast<std::uint8_t>(OfficialArea::kBench)
                    && hand_to_bench.trigger_resolver.active != 0
                    && hand_to_bench.trigger_resolver.awaiting_activation != 0,
                "AbilityPlay trigger enters optional activation");
            require(
                hand_to_bench.select_type == static_cast<std::uint8_t>(
                    OfficialSelectTypeId::kYesNo)
                    && hand_to_bench.options.count == 2,
                "AbilityPlay trigger exposes YesNo");
            checks += 3;
        }

        {
            // Conditions use a separate scratch target list.  An Effected
            // condition must resolve the prior CardRef again after that card
            // moves, just as the official TargetList overload does.
            constexpr std::uint16_t moved_ref = 79;
            OfficialStatePod moved_effected{};
            official_pod_reset(&moved_effected, 99, 20260730);
            add_pod_card(
                &moved_effected,
                moved_ref,
                2,
                0,
                OfficialArea::kDeck);
            require(
                official_pod_push(
                    &moved_effected,
                    &moved_effected.targets,
                    official_pod_area_ref(
                        &moved_effected, OfficialCardRefPod{moved_ref}),
                    OfficialPodError::kSelectionOverflow),
                "moved Effected seed target");
            const std::int32_t old_move_counter =
                moved_effected.targets.values[0].move_counter;
            require(
                official_pod_move_ref(
                    &moved_effected,
                    OfficialCardRefPod{moved_ref},
                    OfficialArea::kHand)
                    .index == moved_ref,
                "moved Effected card moves");
            require(
                moved_effected.targets.values[0].move_counter
                        == old_move_counter
                    && !official_pod_area_ref_valid(
                        &moved_effected, moved_effected.targets.values[0]),
                "serialized Effected target keeps old snapshot");
            OfficialTargetRule effected_target{};
            effected_target.values[kTargetArea0] = static_cast<std::int32_t>(
                OfficialArea::kEffected);
            effected_target.values[kTargetAreaCount] = 1;
            OfficialPodList<OfficialAreaRefPod, kOfficialListCapacity> matches{};
            require(
                official_build_condition_target_list(
                    &moved_effected,
                    rules,
                    effected_target,
                    &matches,
                    OfficialAreaRefPod{},
                    0),
                "moved Effected condition builds targets");
            require(
                matches.count == 1
                    && matches.values[0].card.index == moved_ref
                    && matches.values[0].move_counter
                        == moved_effected.cards[moved_ref].move_counter,
                "moved Effected condition refreshes AreaRef");
            require(
                moved_effected.targets.values[0].move_counter
                    == old_move_counter,
                "moved Effected condition preserves serialized targets");
            checks += 5;
        }

        {
            // Attached-card options identify the carrier Pokemon, while the
            // attachment ordinal resolves the actual Tool card on action.
            OfficialStatePod tool_options{};
            official_pod_reset(&tool_options, 99, 20260730);
            add_pod_card(&tool_options, 80, 756, 0, OfficialArea::kBench);
            add_pod_card(&tool_options, 81, 1159, 0, OfficialArea::kTool);
            attach_pod(&tool_options, 81, 80);
            require(
                official_pod_push(
                    &tool_options,
                    &tool_options.targets,
                    official_pod_area_ref(&tool_options, OfficialCardRefPod{81}),
                    OfficialPodError::kSelectionOverflow),
                "Tool Scrapper target");
            require(
                official_prepare_attached_options(
                    &tool_options,
                    rules,
                    OfficialSelectOptionTypeId::kToolCard),
                "Tool Scrapper attached options");
            require(
                tool_options.options.count == 1
                    && tool_options.options.values[0].resolved_card == 80
                    && tool_options.options.values[0].params[3] == 0,
                "Tool option resolves carrier Pokemon");
            OfficialCardRefPod decoded_tool{};
            require(
                official_attached_ref_from_option(
                    &tool_options, tool_options.options.values[0], &decoded_tool)
                    && decoded_tool.index == 81,
                "Tool option decodes attached card");
            checks += 3;
        }

        place_official_card(&official, kDelaySourceRef, 0, AreaType::Bench);
        add_pod_card(&pod, kDelaySourceRef, 25, 0, OfficialArea::kBench);
        const Attack& delay_attack = AttackTable.at(8);
        int delay_effect_index = -1;
        bool delay_is_post = false;
        for (int index = 0; index < static_cast<int>(delay_attack.preEffects.size()); ++index) {
            if (delay_attack.preEffects[index].effectType == EffectType::DelayEffect) {
                delay_effect_index = index;
                break;
            }
        }
        if (delay_effect_index < 0) {
            for (int index = 0; index < static_cast<int>(delay_attack.postEffects.size()); ++index) {
                if (delay_attack.postEffects[index].effectType == EffectType::DelayEffect) {
                    delay_effect_index = index;
                    delay_is_post = true;
                    break;
                }
            }
        }
        require(delay_effect_index >= 0, "delay effect exists");
        official.currentAttackId = 8;
        official.postAttackEffect = delay_is_post;
        official.setActivateAbility(ActivateAbilityInfo(
            0, official.makeAreaRef(CardRef(kDelaySourceRef)), 0));
        official.effectState.effectIndex = static_cast<signed char>(delay_effect_index);
        official.targetList.clear();
        official.targetList.push_back(official.makeAreaRef(CardRef(kAttackerRef)));
        EffectInstatnt(official);

        pod.current_attack_id = 8;
        pod.effect_state.ability.skill_id = 0;
        pod.effect_state.ability.effect_card = official_pod_area_ref(
            &pod, OfficialCardRefPod{kDelaySourceRef});
        pod.effect_state.ability.use_player = 0;
        pod.targets.count = 0;
        require(
            official_pod_push(
                &pod,
                &pod.targets,
                official_pod_area_ref(&pod, OfficialCardRefPod{kAttackerRef}),
                OfficialPodError::kSelectionOverflow),
            "delay target");
        OfficialEffectRule pod_delay{};
        pod_delay.values[kEffectType] = static_cast<int>(
            OfficialEffectTypeId::kDelayEffect);
        pod_delay.values[kEffectTargetIndex] = 0;
        require(
            official_apply_effect_primitive(
                &pod, continual_rules, pod_delay)
                == OfficialEffectApplyResult::kApplied,
            "delay effect primitive");
        require(
            official.delayTriggerStack.size() == pod.delay_triggers.count,
            "delay trigger count");
        require(
            static_cast<int>(official.delayTriggerStack.back().trigger.type)
                == pod.delay_triggers.values[pod.delay_triggers.count - 1].trigger.type,
            "delay trigger type");
        require(
            official.delayTriggerStack.back().trigger.subject.card.cardIndex
                == pod.delay_triggers.values[pod.delay_triggers.count - 1]
                    .trigger.subject.card.index,
            "delay trigger subject");
        checks += 3;

        {
            TriggeredAbility official_enemy_delay =
                official.delayTriggerStack.back();
            official_enemy_delay.trigger.subject =
                official.makeAreaRef(CardRef(kEnemyActiveRef));
            official.delayTriggerStack.push_back(official_enemy_delay);

            OfficialTriggeredAbilityPod pod_enemy_delay =
                pod.delay_triggers.values[pod.delay_triggers.count - 1];
            pod_enemy_delay.trigger.subject = official_pod_area_ref(
                &pod, OfficialCardRefPod{kEnemyActiveRef});
            require(
                official_pod_push(
                    &pod,
                    &pod.delay_triggers,
                    pod_enemy_delay,
                    OfficialPodError::kTriggerStackOverflow),
                "enemy delay trigger seed");

            official.players[0].activeState = 1;
            pod.players[0].active_state = 1;
            official.cardMoved(CardRef(kAttackerRef), AreaType::Bench);
            official_pod_card_moved(
                &pod, OfficialCardRefPod{kAttackerRef}, OfficialArea::kBench);

            require(
                official.players[0].activeState == pod.players[0].active_state
                    && pod.players[0].active_state == 0,
                "active departure clears active state");
            require(
                official.delayTriggerStack.size() == pod.delay_triggers.count
                    && pod.delay_triggers.count == 1,
                "active departure clears owner delay triggers");
            require(
                official.delayTriggerStack[0].trigger.subject.card.cardIndex
                        == kEnemyActiveRef
                    && pod.delay_triggers.values[0].trigger.subject.card.index
                        == kEnemyActiveRef,
                "active departure preserves enemy delay triggers");
            checks += 4;
        }

        require(pod.error == 0, "pod final error");
        std::cout
            << "{\"passed\":true,\"checks\":" << checks
            << ",\"paired_targets\":[79,85,92],\"paired_conditions\":[9]"
            << ",\"paired_effects\":[172,174,186,207,222,234,238,244]"
            << ",\"paired_damage_pipeline\":true"
            << ",\"paired_move_primitives\":true"
            << ",\"paired_delay_effect\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
