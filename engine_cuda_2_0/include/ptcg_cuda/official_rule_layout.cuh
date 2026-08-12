#pragma once

#include <cstddef>
#include <cstdint>

namespace ptcg::cuda_engine {

constexpr std::uint32_t kOfficialRuleAbiVersion = 1;
constexpr std::size_t kOfficialRuleSectionCount = 12;

enum class OfficialRuleSection : std::uint32_t {
    kCards = 0,
    kSkills = 1,
    kAttacks = 2,
    kEffects = 3,
    kTargets = 4,
    kConditions = 5,
    kTriggers = 6,
    kCardAttackIds = 7,
    kNameSets = 8,
    kNameCardIds = 9,
    kContinuations = 10,
    kReserved = 11,
};

struct alignas(16) OfficialRulePackHeader {
    char magic[8];
    std::uint32_t schema_version;
    std::uint32_t abi_version;
    std::uint64_t total_bytes;
    std::uint32_t counts[kOfficialRuleSectionCount];
    std::uint64_t offsets[kOfficialRuleSectionCount];
    std::uint8_t payload_sha256[32];
    std::uint8_t reserved[56];
};
static_assert(sizeof(OfficialRulePackHeader) == 256);

struct alignas(16) OfficialCardRule {
    std::uint64_t flags;
    std::int32_t values[19];
    std::uint8_t reserved[12];
};
static_assert(sizeof(OfficialCardRule) == 96);

struct alignas(16) OfficialSkillRule {
    std::uint64_t flags;
    std::int32_t values[16];
    std::uint8_t reserved[8];
};
static_assert(sizeof(OfficialSkillRule) == 80);

struct alignas(16) OfficialAttackRule {
    std::uint64_t flags;
    std::int32_t values[17];
    std::uint8_t reserved[20];
};
static_assert(sizeof(OfficialAttackRule) == 96);

struct alignas(16) OfficialEffectRule {
    std::uint64_t flags;
    std::int32_t values[15];
    std::uint8_t reserved[12];
};
static_assert(sizeof(OfficialEffectRule) == 80);

struct alignas(16) OfficialTargetRule {
    std::int32_t values[10];
    std::uint8_t reserved[8];
};
static_assert(sizeof(OfficialTargetRule) == 48);

struct alignas(16) OfficialConditionRule {
    std::int32_t values[5];
    std::uint8_t reserved[12];
};
static_assert(sizeof(OfficialConditionRule) == 32);

struct alignas(16) OfficialTriggerRule {
    std::int32_t type;
    std::int32_t target_index;
    std::uint8_t reserved[8];
};
static_assert(sizeof(OfficialTriggerRule) == 16);

struct alignas(16) OfficialNameSetRule {
    std::int32_t values[9];
    std::uint8_t reserved[12];
};
static_assert(sizeof(OfficialNameSetRule) == 48);

struct alignas(16) OfficialContinuationRule {
    std::uint32_t id;
    std::uint8_t reserved[12];
};
static_assert(sizeof(OfficialContinuationRule) == 16);

enum OfficialCardValueIndex : std::uint32_t {
    kCardId = 0,
    kCardType,
    kCardPokemonType,
    kCardEvolutionType,
    kCardRetreatCost,
    kCardHp,
    kCardWeakness,
    kCardResistance,
    kCardEnergyType,
    kCardEnergyCount,
    kCardNumber,
    kCardNameId,
    kCardEvolvesFromNameId,
    kCardEvolvesFrom2NameId,
    kCardAbilityId,
    kCardPlayId,
    kCardDelayId,
    kCardAttackOffset,
    kCardAttackCount,
};

enum OfficialSkillValueIndex : std::uint32_t {
    kSkillId = 0,
    kSkillCardId,
    kSkillType,
    kSkillPriority,
    kSkillFirstConditionCount,
    kSkillSecondEffectStart,
    kSkillSecondEffectStartEnemy,
    kSkillTriggerStart,
    kSkillNameId,
    kSkillArea0,
    kSkillArea1,
    kSkillAreaCount,
    kSkillTriggerOffset,
    kSkillTriggerCount,
    kSkillEffectOffset,
    kSkillEffectCount,
};

enum OfficialAttackValueIndex : std::uint32_t {
    kAttackId = 0,
    kAttackCardId,
    kAttackDamage,
    kAttackLastCancelFail,
    kAttackNameId,
    kAttackEnergy0,
    kAttackEnergyCount = 12,
    kAttackPreEffectOffset,
    kAttackPreEffectCount,
    kAttackPostEffectOffset,
    kAttackPostEffectCount,
};

enum OfficialEffectValueIndex : std::uint32_t {
    kEffectType = 0,
    kEffectSelectType,
    kEffectSelectCount,
    kEffectSelectContext,
    kEffectLoopCount,
    kEffectPriority,
    kEffectValue0,
    kEffectValue1,
    kEffectConditionType,
    kEffectComparator,
    kEffectFailSkip,
    kEffectSkillId,
    kEffectLinkedSkillId,
    kEffectLinkedAttackId,
    kEffectTargetIndex,
};

enum OfficialTargetValueIndex : std::uint32_t {
    kTargetPlayer = 0,
    kTargetNotMe,
    kTargetSkipEnemy,
    kTargetArea0,
    kTargetArea1,
    kTargetArea2,
    kTargetArea3,
    kTargetAreaCount,
    kTargetConditionOffset,
    kTargetConditionCount,
};

enum OfficialConditionValueIndex : std::uint32_t {
    kConditionType = 0,
    kConditionComparator,
    kConditionValue,
    kConditionValue2,
    kConditionNameId,
};

enum OfficialNameSetValueIndex : std::uint32_t {
    kNameSetId = 0,
    kNameEqualOffset,
    kNameEqualCount,
    kNameContainsOffset,
    kNameContainsCount,
    kNameAbilityOffset,
    kNameAbilityCount,
    kNameAttackOffset,
    kNameAttackCount,
};

struct OfficialRulePackView {
    const OfficialRulePackHeader* header = nullptr;
    const OfficialCardRule* cards = nullptr;
    const OfficialSkillRule* skills = nullptr;
    const OfficialAttackRule* attacks = nullptr;
    const OfficialEffectRule* effects = nullptr;
    const OfficialTargetRule* targets = nullptr;
    const OfficialConditionRule* conditions = nullptr;
    const OfficialTriggerRule* triggers = nullptr;
    const std::uint32_t* card_attack_ids = nullptr;
    const OfficialNameSetRule* name_sets = nullptr;
    const std::uint32_t* name_card_ids = nullptr;
    const OfficialContinuationRule* continuations = nullptr;
};

#if defined(__CUDACC__)
#define PTCG_RULE_HD __host__ __device__
#else
#define PTCG_RULE_HD
#endif

template <typename T>
PTCG_RULE_HD inline const T* official_rule_section(
    const std::uint8_t* base,
    const OfficialRulePackHeader* header,
    OfficialRuleSection section) {
    return reinterpret_cast<const T*>(
        base + header->offsets[static_cast<std::uint32_t>(section)]);
}

PTCG_RULE_HD inline OfficialRulePackView make_official_rule_pack_view(
    const void* data) {
    const auto* base = static_cast<const std::uint8_t*>(data);
    const auto* header = reinterpret_cast<const OfficialRulePackHeader*>(base);
    return {
        header,
        official_rule_section<OfficialCardRule>(base, header, OfficialRuleSection::kCards),
        official_rule_section<OfficialSkillRule>(base, header, OfficialRuleSection::kSkills),
        official_rule_section<OfficialAttackRule>(base, header, OfficialRuleSection::kAttacks),
        official_rule_section<OfficialEffectRule>(base, header, OfficialRuleSection::kEffects),
        official_rule_section<OfficialTargetRule>(base, header, OfficialRuleSection::kTargets),
        official_rule_section<OfficialConditionRule>(base, header, OfficialRuleSection::kConditions),
        official_rule_section<OfficialTriggerRule>(base, header, OfficialRuleSection::kTriggers),
        official_rule_section<std::uint32_t>(base, header, OfficialRuleSection::kCardAttackIds),
        official_rule_section<OfficialNameSetRule>(base, header, OfficialRuleSection::kNameSets),
        official_rule_section<std::uint32_t>(base, header, OfficialRuleSection::kNameCardIds),
        official_rule_section<OfficialContinuationRule>(base, header, OfficialRuleSection::kContinuations),
    };
}

PTCG_RULE_HD inline const OfficialCardRule* official_card_rule(
    const OfficialRulePackView& view,
    std::uint32_t card_id) {
    if (card_id == 0 || card_id > view.header->counts[0]) {
        return nullptr;
    }
    const OfficialCardRule* rule = &view.cards[card_id - 1];
    return rule->values[kCardId] == static_cast<std::int32_t>(card_id) ? rule : nullptr;
}

PTCG_RULE_HD inline const OfficialAttackRule* official_attack_rule(
    const OfficialRulePackView& view,
    std::uint32_t attack_id) {
    if (attack_id == 0 || attack_id > view.header->counts[2]) {
        return nullptr;
    }
    const OfficialAttackRule* rule = &view.attacks[attack_id - 1];
    return rule->values[kAttackId] == static_cast<std::int32_t>(attack_id) ? rule : nullptr;
}

PTCG_RULE_HD inline const OfficialSkillRule* official_skill_rule(
    const OfficialRulePackView& view,
    std::uint32_t skill_id) {
    if (view.header->counts[1] == 0) {
        return nullptr;
    }
    const std::int32_t first_id = view.skills[0].values[kSkillId];
    if (skill_id < static_cast<std::uint32_t>(first_id)) {
        return nullptr;
    }
    const std::uint32_t index = skill_id - static_cast<std::uint32_t>(first_id);
    if (index >= view.header->counts[1]) {
        return nullptr;
    }
    const OfficialSkillRule* rule = &view.skills[index];
    return rule->values[kSkillId] == static_cast<std::int32_t>(skill_id) ? rule : nullptr;
}

#undef PTCG_RULE_HD

}  // namespace ptcg::cuda_engine
