#include <algorithm>
#include <array>
#include <cxxabi.h>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "All.h"

namespace {

const std::u8string kSilcoonOrCascoonSet = u8"__ptcg_target_51__";
const std::u8string kKoffingOrWeezingSet = u8"__ptcg_target_52__";
const std::u8string kHonedgeFamilySet = u8"__ptcg_target_53__";

template <typename T>
void write_number_array(std::ostream& out, const T& values) {
    out << '[';
    bool first = true;
    for (const auto value : values) {
        if (!first) {
            out << ',';
        }
        first = false;
        out << static_cast<long long>(value);
    }
    out << ']';
}

template <typename Map>
std::vector<int> sorted_keys(const Map& values) {
    std::vector<int> result;
    result.reserve(values.size());
    for (const auto& [key, _] : values) {
        result.push_back(static_cast<int>(key));
    }
    std::sort(result.begin(), result.end());
    return result;
}

void collect_target_names(const Target& target, std::set<std::u8string>& names) {
    for (const TargetCondition& condition : target.conditions) {
        if (!condition.name.empty()) {
            names.insert(condition.name);
        }
    }
}

void collect_effect_names(const std::vector<Effect>& effects, std::set<std::u8string>& names) {
    for (const Effect& effect : effects) {
        collect_target_names(effect.target, names);
    }
}

std::map<std::u8string, int> build_name_ids() {
    std::set<std::u8string> names;
    names.insert(kSilcoonOrCascoonSet);
    names.insert(kKoffingOrWeezingSet);
    names.insert(kHonedgeFamilySet);
    for (const auto& [_, card] : CardTable) {
        if (!card.name.empty()) {
            names.insert(card.name);
        }
        if (!card.evolvesFrom.empty()) {
            names.insert(card.evolvesFrom);
        }
        if (!card.evolvesFrom2.empty()) {
            names.insert(card.evolvesFrom2);
        }
    }
    for (const auto& [_, skill] : SkillTable) {
        if (!skill.name.empty()) {
            names.insert(skill.name);
        }
        for (const Trigger& trigger : skill.triggers) {
            collect_target_names(trigger.subject, names);
        }
        collect_effect_names(skill.effects, names);
    }
    for (const auto& [_, attack] : AttackTable) {
        if (!attack.name.empty()) {
            names.insert(attack.name);
        }
        collect_effect_names(attack.preEffects, names);
        collect_effect_names(attack.postEffects, names);
    }

    std::map<std::u8string, int> result;
    int next_id = 1;
    for (const std::u8string& name : names) {
        result.emplace(name, next_id++);
    }
    return result;
}

int name_id(const std::map<std::u8string, int>& names, const std::u8string& name) {
    if (name.empty()) {
        return 0;
    }
    const auto found = names.find(name);
    if (found == names.end()) {
        throw std::runtime_error("missing normalized name ID");
    }
    return found->second;
}

int target_condition_name_id(
    const std::map<std::u8string, int>& names,
    const TargetCondition& condition) {
    if (!condition.name.empty()) {
        return name_id(names, condition.name);
    }
    switch (condition.targetType) {
        case TargetType::SilcoonOrCascoon:
            return name_id(names, kSilcoonOrCascoonSet);
        case TargetType::KoffingOrWeezing:
            return name_id(names, kKoffingOrWeezingSet);
        case TargetType::HonedgeOrDoubladeOrAegislash:
            return name_id(names, kHonedgeFamilySet);
        default:
            return 0;
    }
}

std::uint64_t card_flags(const CardMaster& card) {
    const std::array<bool, 30> values = {
        card.tera,
        card.trashMyTurnEnd,
        card.cannotToHandOrDeckInTrash,
        card.canPlayFirstTurn,
        card.transformOnly,
        card.canTrash,
        card.toBench,
        card.toBattleFieldOnlySetup,
        card.toActiveOnlySetup,
        card.noPrize,
        card.onlyTeamRocket,
        card.ancient,
        card.future,
        card.hop,
        card.lillie,
        card.iono,
        card.n,
        card.ethan,
        card.cynthia,
        card.misty,
        card.arven,
        card.steven,
        card.marnie,
        card.erika,
        card.larry,
        card.teamRocket,
        card.aceSpec,
        card.canUse,
        card.isPokemonCard(),
        card.isRulePokemon(),
    };
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (values[index]) {
            result |= std::uint64_t{1} << index;
        }
    }
    return result;
}

std::uint64_t skill_flags(const Skill& skill) {
    const std::array<bool, 8> values = {
        skill.mainAbility,
        skill.onceTurn,
        skill.canSelectActivate,
        skill.notStack,
        skill.canActivateTrash,
        skill.attachBench,
        skill.koMeAbility,
        skill.luckyBonus,
    };
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (values[index]) {
            result |= std::uint64_t{1} << index;
        }
    }
    return result;
}

std::uint64_t effect_flags(const Effect& effect) {
    const std::array<bool, 31> values = {
        effect.isCondition,
        effect.enemySelect,
        effect.randomSelect,
        effect.eachSelectedList,
        effect.eachList,
        effect.addCheckList,
        effect.notClearSelectedList,
        effect.notPreTarget,
        effect.notUpdateTarget,
        effect.multiplyEffectValuePreTargetCount,
        effect.multiplyEffectValueCoinHeadCount,
        effect.canNoSelect,
        effect.canNoSelectIfExistPreTarget,
        effect.cannotNoSelect,
        effect.energyMaxSelect,
        effect.selectTargetCount,
        effect.selectCoinHeadCount,
        effect.selectCoinHeadCount2,
        effect.selectEnemyEnergyCount,
        effect.skipNoTarget,
        effect.open,
        effect.setTargetSwitchBench,
        effect.effectTargetActive,
        effect.effectTargetBench,
        effect.removeEffectedIfNoEffect,
        effect.seeingDeck,
        effect.separator,
        effect.isInstantEffect(),
        effect.isNotOpenSelect(),
        effect.isNotOpenSelectNoCondition(),
        effect.target.notMe,
    };
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (values[index]) {
            result |= std::uint64_t{1} << index;
        }
    }
    return result;
}

void write_target(
    std::ostream& out,
    const Target& target,
    const std::map<std::u8string, int>& names) {
    out << "{\"player\":" << static_cast<int>(target.targetPlayer)
        << ",\"not_me\":" << static_cast<int>(target.notMe)
        << ",\"skip_enemy_target\":" << static_cast<int>(target.skipEnemyTarget)
        << ",\"areas\":";
    write_number_array(out, target.areas);
    out << ",\"conditions\":[";
    for (std::size_t index = 0; index < target.conditions.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        const TargetCondition& condition = target.conditions[index];
        out << "{\"type\":" << static_cast<int>(condition.targetType)
            << ",\"comparator\":" << static_cast<int>(condition.comparatorType)
            << ",\"value\":" << condition.val
            << ",\"value2\":" << condition.val2
            << ",\"name_id\":" << target_condition_name_id(names, condition) << '}';
    }
    out << "]}";
}

void write_effect(
    std::ostream& out,
    const Effect& effect,
    const std::map<std::u8string, int>& names) {
    out << "{\"type\":" << static_cast<int>(effect.effectType)
        << ",\"select_type\":" << static_cast<int>(effect.effectSelectType)
        << ",\"select_count\":" << static_cast<int>(effect.selectCount)
        << ",\"select_context\":" << static_cast<int>(effect.selectContext)
        << ",\"flags\":" << effect_flags(effect)
        << ",\"loop_count\":" << static_cast<int>(effect.loopCount)
        << ",\"priority\":" << effect.priority
        << ",\"values\":[" << effect.values[0] << ',' << effect.values[1] << ']'
        << ",\"condition_type\":" << static_cast<int>(effect.conditionType)
        << ",\"comparator\":" << static_cast<int>(effect.comparatorType)
        << ",\"fail_skip\":" << static_cast<int>(effect.failSkip)
        << ",\"skill_id\":" << effect.skillId
        << ",\"linked_skill_id\":" << (effect.skill == nullptr ? 0 : effect.skill->skillId)
        << ",\"linked_attack_id\":" << (effect.attack == nullptr ? 0 : effect.attack->attackId)
        << ",\"target\":";
    write_target(out, effect.target, names);
    out << '}';
}

void write_effects(
    std::ostream& out,
    const std::vector<Effect>& effects,
    const std::map<std::u8string, int>& names) {
    out << '[';
    for (std::size_t index = 0; index < effects.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        write_effect(out, effects[index], names);
    }
    out << ']';
}

void write_flag_schema(std::ostream& out) {
    out << "{\"card\":["
        "\"tera\",\"trash_my_turn_end\",\"cannot_to_hand_or_deck_in_trash\","
        "\"can_play_first_turn\",\"transform_only\",\"can_trash\",\"to_bench\","
        "\"to_battle_field_only_setup\",\"to_active_only_setup\",\"no_prize\","
        "\"only_team_rocket\",\"ancient\",\"future\",\"hop\",\"lillie\","
        "\"iono\",\"n\",\"ethan\",\"cynthia\",\"misty\",\"arven\","
        "\"steven\",\"marnie\",\"erika\",\"larry\",\"team_rocket\","
        "\"ace_spec\",\"can_use\",\"is_pokemon_card\",\"is_rule_pokemon\"],"
        "\"skill\":[\"main_ability\",\"once_turn\",\"can_select_activate\","
        "\"not_stack\",\"can_activate_trash\",\"attach_bench\",\"ko_me_ability\","
        "\"lucky_bonus\"],\"effect\":[\"is_condition\",\"enemy_select\","
        "\"random_select\",\"each_selected_list\",\"each_list\",\"add_check_list\","
        "\"not_clear_selected_list\",\"not_pre_target\",\"not_update_target\","
        "\"multiply_pre_target_count\",\"multiply_coin_head_count\",\"can_no_select\","
        "\"can_no_select_if_pre_target\",\"cannot_no_select\",\"energy_max_select\","
        "\"select_target_count\",\"select_coin_head_count\",\"select_coin_head_count2\","
        "\"select_enemy_energy_count\",\"skip_no_target\",\"open\","
        "\"set_target_switch_bench\",\"effect_target_active\",\"effect_target_bench\","
        "\"remove_effected_if_no_effect\",\"seeing_deck\",\"separator\","
        "\"is_instant_effect\",\"is_not_open_select\","
        "\"is_not_open_select_no_condition\",\"target_not_me\"]}";
}

void write_name_sets(
    std::ostream& out,
    const std::map<std::u8string, int>& names) {
    std::vector<const CardMaster*> cards;
    cards.reserve(CardTable.size());
    for (const auto& [_, card] : CardTable) {
        cards.push_back(&card);
    }
    std::sort(cards.begin(), cards.end(), [](const CardMaster* left, const CardMaster* right) {
        return left->cardId < right->cardId;
    });

    out << '[';
    bool first_name = true;
    for (const auto& [name, id] : names) {
        if (!first_name) {
            out << ',';
        }
        first_name = false;
        std::vector<int> equal;
        std::vector<int> contains;
        std::vector<int> ability;
        std::vector<int> attack;
        for (const CardMaster* card : cards) {
            const bool special_equal =
                (name == kSilcoonOrCascoonSet
                    && (card->name == u8"カラサリス" || card->name == u8"マユルド"))
                || (name == kHonedgeFamilySet
                    && (card->name == u8"ヒトツキ" || card->name == u8"ニダンギル"
                        || card->name == u8"ギルガルド"));
            const bool special_contains = name == kKoffingOrWeezingSet
                && (card->name.find(u8"ドガース") != std::u8string::npos
                    || card->name.find(u8"マタドガス") != std::u8string::npos);
            const bool synthetic = name == kSilcoonOrCascoonSet
                || name == kKoffingOrWeezingSet
                || name == kHonedgeFamilySet;
            if (special_equal || (!synthetic && card->name == name)) {
                equal.push_back(card->cardId);
            }
            if (special_contains || (!synthetic && card->name.find(name) != std::u8string::npos)) {
                contains.push_back(card->cardId);
            }
            if (!synthetic && card->ability != nullptr && card->ability->name == name) {
                ability.push_back(card->cardId);
            }
            if (!synthetic) {
                for (const Attack* candidate : card->attacks) {
                    if (candidate->name == name) {
                        attack.push_back(card->cardId);
                        break;
                    }
                }
            }
        }
        out << "{\"id\":" << id << ",\"equal_cards\":";
        write_number_array(out, equal);
        out << ",\"contains_cards\":";
        write_number_array(out, contains);
        out << ",\"ability_cards\":";
        write_number_array(out, ability);
        out << ",\"attack_cards\":";
        write_number_array(out, attack);
        out << '}';
    }
    out << ']';
}

void write_cards(
    std::ostream& out,
    const std::map<std::u8string, int>& names) {
    out << '[';
    const std::vector<int> keys = sorted_keys(CardTable);
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        const CardMaster& card = CardTable.at(keys[index]);
        std::vector<int> attacks;
        attacks.reserve(card.attacks.size());
        for (const Attack* attack : card.attacks) {
            if (attack == nullptr) {
                throw std::runtime_error("null attack pointer");
            }
            attacks.push_back(attack->attackId);
        }
        out << "{\"id\":" << card.cardId
            << ",\"card_type\":" << static_cast<int>(card.cardType)
            << ",\"pokemon_type\":" << static_cast<int>(card.pokemonType)
            << ",\"evolution_type\":" << static_cast<int>(card.evolutionType)
            << ",\"retreat_cost\":" << static_cast<int>(card.retreatCost)
            << ",\"hp\":" << card.hp
            << ",\"weakness\":" << static_cast<int>(card.weakness)
            << ",\"resistance\":" << static_cast<int>(card.resistance)
            << ",\"energy_type\":" << static_cast<int>(card.energyType)
            << ",\"energy_count\":" << static_cast<int>(card.energyCount)
            << ",\"flags\":" << card_flags(card)
            << ",\"number\":" << card.no
            << ",\"name_id\":" << name_id(names, card.name)
            << ",\"evolves_from_name_id\":" << name_id(names, card.evolvesFrom)
            << ",\"evolves_from2_name_id\":" << name_id(names, card.evolvesFrom2)
            << ",\"ability_id\":" << (card.ability == nullptr ? 0 : card.ability->skillId)
            << ",\"play_id\":" << (card.play == nullptr ? 0 : card.play->skillId)
            << ",\"delay_id\":" << (card.delay == nullptr ? 0 : card.delay->skillId)
            << ",\"attack_ids\":";
        write_number_array(out, attacks);
        out << '}';
    }
    out << ']';
}

void write_skills(
    std::ostream& out,
    const std::map<std::u8string, int>& names) {
    out << '[';
    const std::vector<int> keys = sorted_keys(SkillTable);
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        const Skill& skill = SkillTable.at(keys[index]);
        out << "{\"id\":" << skill.skillId
            << ",\"card_id\":" << skill.cardId
            << ",\"skill_type\":" << static_cast<int>(skill.skillType)
            << ",\"flags\":" << skill_flags(skill)
            << ",\"priority\":" << static_cast<int>(skill.priority)
            << ",\"first_condition_count\":" << static_cast<int>(skill.firstConditionCount)
            << ",\"second_effect_start\":" << static_cast<int>(skill.secondEffectStartIndex)
            << ",\"second_effect_start_enemy\":"
            << static_cast<int>(skill.secondEffectStartIndexEnemy)
            << ",\"trigger_start\":" << static_cast<int>(skill.triggerStartIndex)
            << ",\"name_id\":" << name_id(names, skill.name)
            << ",\"areas\":";
        write_number_array(out, skill.areas);
        out << ",\"triggers\":[";
        for (std::size_t trigger_index = 0; trigger_index < skill.triggers.size(); ++trigger_index) {
            if (trigger_index != 0) {
                out << ',';
            }
            const Trigger& trigger = skill.triggers[trigger_index];
            out << "{\"type\":" << static_cast<int>(trigger.triggerType)
                << ",\"subject\":";
            write_target(out, trigger.subject, names);
            out << '}';
        }
        out << "],\"effects\":";
        write_effects(out, skill.effects, names);
        out << '}';
    }
    out << ']';
}

void write_attacks(
    std::ostream& out,
    const std::map<std::u8string, int>& names) {
    out << '[';
    const std::vector<int> keys = sorted_keys(AttackTable);
    for (std::size_t index = 0; index < keys.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        const Attack& attack = AttackTable.at(keys[index]);
        out << "{\"id\":" << attack.attackId
            << ",\"card_id\":" << attack.cardId
            << ",\"damage\":" << attack.damage
            << ",\"flags\":" << attack.attackFlags
            << ",\"last_cancel_fail_attack\":" << static_cast<int>(attack.lastCancelFailAttack)
            << ",\"name_id\":" << name_id(names, attack.name)
            << ",\"energies\":";
        write_number_array(out, attack.energies);
        out << ",\"pre_effects\":";
        write_effects(out, attack.preEffects, names);
        out << ",\"post_effects\":";
        write_effects(out, attack.postEffects, names);
        out << '}';
    }
    out << ']';
}

void write_continuations(std::ostream& out) {
    out << '[';
    for (std::size_t index = 0; index < FunctionTable.size(); ++index) {
        if (index != 0) {
            out << ',';
        }
        Dl_info info{};
        if (dladdr(FunctionTable[index], &info) == 0 || info.dli_sname == nullptr) {
            throw std::runtime_error("cannot resolve continuation symbol " + std::to_string(index));
        }
        int status = 0;
        char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
        const std::string symbol = status == 0 && demangled != nullptr
            ? std::string(demangled)
            : std::string(info.dli_sname);
        std::free(demangled);
        if (symbol.empty() || symbol.find('"') != std::string::npos
            || symbol.find('\\') != std::string::npos) {
            throw std::runtime_error("invalid continuation symbol " + std::to_string(index));
        }
        out << "{\"id\":" << index << ",\"symbol\":\"" << symbol << "\"}";
    }
    out << ']';
}

void export_rules(std::ostream& out) {
    InitializeAll();
    const auto names = build_name_ids();

    out << "{\"schema_version\":1,\"counts\":{\"cards\":" << CardTable.size()
        << ",\"skills\":" << SkillTable.size()
        << ",\"attacks\":" << AttackTable.size()
        << ",\"continuations\":" << FunctionTable.size()
        << ",\"name_sets\":" << names.size() << "},\"flag_schema\":";
    write_flag_schema(out);
    out << ",\"name_sets\":";
    write_name_sets(out, names);
    out << ",\"cards\":";
    write_cards(out, names);
    out << ",\"skills\":";
    write_skills(out, names);
    out << ",\"attacks\":";
    write_attacks(out, names);
    out << ",\"continuations\":";
    write_continuations(out);
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("usage: official_rule_exporter <output-json>");
        }
        std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot open output path");
        }
        export_rules(output);
        output.close();
        if (!output) {
            throw std::runtime_error("failed while writing output");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
