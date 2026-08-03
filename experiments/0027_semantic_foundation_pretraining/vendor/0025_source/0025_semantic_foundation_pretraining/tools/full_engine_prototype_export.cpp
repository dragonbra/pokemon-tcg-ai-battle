// Read-only 0025 extractor. It includes official headers but never modifies engine/source.
#include "All.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

static void quoted(const std::u8string& value) {
    std::cout << '"';
    for (char8_t raw : value) {
        unsigned char value_byte = static_cast<unsigned char>(raw);
        if (value_byte == '"' || value_byte == '\\') {
            std::cout << '\\' << static_cast<char>(value_byte);
        } else if (value_byte < 0x20) {
            static const char* digits = "0123456789abcdef";
            std::cout << "\\u00" << digits[value_byte >> 4] << digits[value_byte & 15];
        } else {
            std::cout << static_cast<char>(value_byte);
        }
    }
    std::cout << '"';
}

template <typename Values>
static void enum_array(const Values& values) {
    std::cout << '[';
    bool first = true;
    for (auto value : values) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << static_cast<int>(value);
    }
    std::cout << ']';
}

static void target_json(const Target& target) {
    std::cout << "{\"target_player\":" << static_cast<int>(target.targetPlayer)
              << ",\"not_me\":" << target.notMe
              << ",\"skip_enemy_target\":" << target.skipEnemyTarget
              << ",\"areas\":";
    enum_array(target.areas);
    std::cout << ",\"conditions\":[";
    for (std::size_t i = 0; i < target.conditions.size(); ++i) {
        if (i) std::cout << ',';
        const auto& item = target.conditions[i];
        std::cout << "{\"target_type\":" << static_cast<int>(item.targetType)
                  << ",\"comparator_type\":" << static_cast<int>(item.comparatorType)
                  << ",\"value\":" << item.val << ",\"value2\":" << item.val2
                  << ",\"name\":";
        quoted(item.name);
        std::cout << '}';
    }
    std::cout << "]}";
}

static void effect_json(const Effect& effect) {
    std::cout << "{\"is_condition\":" << effect.isCondition
              << ",\"effect_type\":" << static_cast<int>(effect.effectType)
              << ",\"select_type\":" << static_cast<int>(effect.effectSelectType)
              << ",\"select_count\":" << static_cast<int>(effect.selectCount)
              << ",\"select_context\":" << static_cast<int>(effect.selectContext)
              << ",\"enemy_select\":" << effect.enemySelect
              << ",\"random_select\":" << effect.randomSelect
              << ",\"each_selected\":" << effect.eachSelectedList
              << ",\"each_list\":" << effect.eachList
              << ",\"add_check_list\":" << effect.addCheckList
              << ",\"keep_selected_list\":" << effect.notClearSelectedList
              << ",\"exclude_previous_target\":" << effect.notPreTarget
              << ",\"keep_target_list\":" << effect.notUpdateTarget
              << ",\"multiply_previous_target_count\":" << effect.multiplyEffectValuePreTargetCount
              << ",\"multiply_coin_heads\":" << effect.multiplyEffectValueCoinHeadCount
              << ",\"can_select_zero\":" << effect.canNoSelect
              << ",\"can_select_zero_if_previous\":" << effect.canNoSelectIfExistPreTarget
              << ",\"cannot_select_zero\":" << effect.cannotNoSelect
              << ",\"energy_max_select\":" << effect.energyMaxSelect
              << ",\"select_target_count\":" << effect.selectTargetCount
              << ",\"select_coin_head_count\":" << effect.selectCoinHeadCount
              << ",\"select_coin_head_count_x2\":" << effect.selectCoinHeadCount2
              << ",\"select_enemy_energy_count\":" << effect.selectEnemyEnergyCount
              << ",\"skip_no_target\":" << effect.skipNoTarget
              << ",\"open\":" << effect.open
              << ",\"switch_bench_as_target\":" << effect.setTargetSwitchBench
              << ",\"active_effect_target\":" << effect.effectTargetActive
              << ",\"bench_effect_target\":" << effect.effectTargetBench
              << ",\"remove_if_no_effect\":" << effect.removeEffectedIfNoEffect
              << ",\"seeing_deck\":" << effect.seeingDeck
              << ",\"separator\":" << effect.separator
              << ",\"loop_count\":" << static_cast<int>(effect.loopCount)
              << ",\"priority\":" << effect.priority
              << ",\"values\":[" << effect.values[0] << ',' << effect.values[1] << ']'
              << ",\"condition_type\":" << static_cast<int>(effect.conditionType)
              << ",\"comparator_type\":" << static_cast<int>(effect.comparatorType)
              << ",\"fail_skip\":" << static_cast<int>(effect.failSkip)
              << ",\"linked_skill_id\":" << effect.skillId
              << ",\"target\":";
    target_json(effect.target);
    std::cout << '}';
}

static void effects_json(const std::vector<Effect>& effects) {
    std::cout << '[';
    for (std::size_t i = 0; i < effects.size(); ++i) {
        if (i) std::cout << ',';
        effect_json(effects[i]);
    }
    std::cout << ']';
}

static void skill_json(const Skill& skill) {
    std::cout << "{\"skill_id\":" << skill.skillId << ",\"card_id\":" << skill.cardId
              << ",\"skill_type\":" << static_cast<int>(skill.skillType)
              << ",\"main_ability\":" << skill.mainAbility
              << ",\"once_turn\":" << skill.onceTurn
              << ",\"select_activation\":" << skill.canSelectActivate
              << ",\"not_stack\":" << skill.notStack
              << ",\"activate_in_discard\":" << skill.canActivateTrash
              << ",\"attach_bench\":" << skill.attachBench
              << ",\"ko_self\":" << skill.koMeAbility
              << ",\"lucky_bonus\":" << skill.luckyBonus
              << ",\"priority\":" << static_cast<int>(skill.priority)
              << ",\"first_condition_count\":" << static_cast<int>(skill.firstConditionCount)
              << ",\"second_effect_start\":" << static_cast<int>(skill.secondEffectStartIndex)
              << ",\"second_effect_start_enemy\":" << static_cast<int>(skill.secondEffectStartIndexEnemy)
              << ",\"trigger_start\":" << static_cast<int>(skill.triggerStartIndex)
              << ",\"areas\":";
    enum_array(skill.areas);
    std::cout << ",\"triggers\":[";
    for (std::size_t i = 0; i < skill.triggers.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"trigger_type\":" << static_cast<int>(skill.triggers[i].triggerType)
                  << ",\"subject\":";
        target_json(skill.triggers[i].subject);
        std::cout << '}';
    }
    std::cout << "],\"effects\":";
    effects_json(skill.effects);
    std::cout << '}';
}

int main() {
    InitializeAll();
    std::vector<int> card_ids, skill_ids, attack_ids;
    for (const auto& [id, _] : CardTable) card_ids.push_back(id);
    for (const auto& [id, _] : SkillTable) skill_ids.push_back(id);
    for (const auto& [id, _] : AttackTable) attack_ids.push_back(id);
    std::sort(card_ids.begin(), card_ids.end());
    std::sort(skill_ids.begin(), skill_ids.end());
    std::sort(attack_ids.begin(), attack_ids.end());
    std::cout << "{\"schema_version\":\"0025_official_full_engine_prototypes_v1\",\"cards\":[";
    for (std::size_t i = 0; i < card_ids.size(); ++i) {
        if (i) std::cout << ',';
        const auto& card = CardTable.at(card_ids[i]);
        std::cout << "{\"card_id\":" << card.cardId
                  << ",\"card_type\":" << static_cast<int>(card.cardType)
                  << ",\"pokemon_type\":" << static_cast<int>(card.pokemonType)
                  << ",\"evolution_type\":" << static_cast<int>(card.evolutionType)
                  << ",\"retreat_cost\":" << static_cast<int>(card.retreatCost)
                  << ",\"hp\":" << card.hp
                  << ",\"weakness\":" << static_cast<int>(card.weakness)
                  << ",\"resistance\":" << static_cast<int>(card.resistance)
                  << ",\"energy_type\":" << static_cast<int>(card.energyType)
                  << ",\"energy_count\":" << static_cast<int>(card.energyCount)
                  << ",\"ability_skill_id\":" << (card.ability ? card.ability->skillId : 0)
                  << ",\"play_skill_id\":" << (card.play ? card.play->skillId : 0)
                  << ",\"delay_skill_id\":" << (card.delay ? card.delay->skillId : 0)
                  << ",\"attack_ids\":[";
        for (std::size_t j = 0; j < card.attacks.size(); ++j) {
            if (j) std::cout << ',';
            std::cout << card.attacks[j]->attackId;
        }
        std::cout << "]}";
    }
    std::cout << "],\"skills\":[";
    for (std::size_t i = 0; i < skill_ids.size(); ++i) {
        if (i) std::cout << ',';
        skill_json(SkillTable.at(skill_ids[i]));
    }
    std::cout << "],\"attacks\":[";
    for (std::size_t i = 0; i < attack_ids.size(); ++i) {
        if (i) std::cout << ',';
        const auto& attack = AttackTable.at(attack_ids[i]);
        std::cout << "{\"attack_id\":" << attack.attackId << ",\"card_id\":" << attack.cardId
                  << ",\"damage\":" << attack.damage << ",\"attack_flags\":" << attack.attackFlags
                  << ",\"energies\":";
        enum_array(attack.energies);
        std::cout << ",\"pre_effects\":";
        effects_json(attack.preEffects);
        std::cout << ",\"post_effects\":";
        effects_json(attack.postEffects);
        std::cout << '}';
    }
    std::cout << "]}\n";
}
