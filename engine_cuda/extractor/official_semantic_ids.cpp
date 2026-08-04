#include "Types.h"
#include "ptcg_cuda/official_semantic_ids.cuh"

#include <iostream>

using ptcg::cuda_engine::OfficialEffectTypeId;
using ptcg::cuda_engine::OfficialConditionTypeId;
using ptcg::cuda_engine::OfficialTargetTypeId;

static_assert(static_cast<int>(OfficialEffectTypeId::kNoEffect) == static_cast<int>(EffectType::NoEffect));
static_assert(static_cast<int>(OfficialEffectTypeId::kKo) == static_cast<int>(EffectType::Ko));
static_assert(static_cast<int>(OfficialEffectTypeId::kDamageCounter) == static_cast<int>(EffectType::DamageCounter));
static_assert(static_cast<int>(OfficialEffectTypeId::kCoin) == static_cast<int>(EffectType::Coin));
static_assert(static_cast<int>(OfficialEffectTypeId::kBurn) == static_cast<int>(EffectType::Burn));
static_assert(static_cast<int>(OfficialEffectTypeId::kDraw) == static_cast<int>(EffectType::Draw));
static_assert(static_cast<int>(OfficialEffectTypeId::kShuffle) == static_cast<int>(EffectType::Shuffle));
static_assert(static_cast<int>(OfficialEffectTypeId::kTurnEnd) == static_cast<int>(EffectType::TurnEnd));
static_assert(static_cast<int>(OfficialEffectTypeId::kNoToolEffect) == static_cast<int>(EffectType::NoToolEffect));
static_assert(static_cast<int>(OfficialTargetTypeId::kAll) == static_cast<int>(TargetType::All));
static_assert(static_cast<int>(OfficialTargetTypeId::kName) == static_cast<int>(TargetType::Name));
static_assert(static_cast<int>(OfficialTargetTypeId::kDamageCounter) == static_cast<int>(TargetType::DamageCounter));
static_assert(static_cast<int>(OfficialTargetTypeId::kNotChecked) == static_cast<int>(TargetType::NotChecked));
static_assert(static_cast<int>(OfficialConditionTypeId::kAlways) == static_cast<int>(ConditionType::Always));
static_assert(static_cast<int>(OfficialConditionTypeId::kCoinHeadCount) == static_cast<int>(ConditionType::CoinHeadCount));
static_assert(static_cast<int>(OfficialConditionTypeId::kLoveBall) == static_cast<int>(ConditionType::LoveBall));

#define EMIT_VALUE(name) \
    std::cout << "\"" #name "\":" << static_cast<int>(name)

int main() {
    std::cout << "{";
    std::cout << "\"effect\":{";
    EMIT_VALUE(EffectType::NoEffect); std::cout << ',';
    EMIT_VALUE(EffectType::SelectCard); std::cout << ',';
    EMIT_VALUE(EffectType::ForEach); std::cout << ',';
    EMIT_VALUE(EffectType::Ko); std::cout << ',';
    EMIT_VALUE(EffectType::ToHand); std::cout << ',';
    EMIT_VALUE(EffectType::PrizeToHand); std::cout << ',';
    EMIT_VALUE(EffectType::ToHandReverse); std::cout << ',';
    EMIT_VALUE(EffectType::ToHandWithAttach); std::cout << ',';
    EMIT_VALUE(EffectType::ToTrash); std::cout << ',';
    EMIT_VALUE(EffectType::ToDeck); std::cout << ',';
    EMIT_VALUE(EffectType::ToDeckWithAttach); std::cout << ',';
    EMIT_VALUE(EffectType::ToDeckReverse); std::cout << ',';
    EMIT_VALUE(EffectType::LookToDeckReverse); std::cout << ',';
    EMIT_VALUE(EffectType::ToDeckAndShuffle); std::cout << ',';
    EMIT_VALUE(EffectType::ToDeckReverseAndShuffle); std::cout << ',';
    EMIT_VALUE(EffectType::ToDeckBottom); std::cout << ',';
    EMIT_VALUE(EffectType::ToDeckBottomReverse); std::cout << ',';
    EMIT_VALUE(EffectType::ToDeckBottomClose); std::cout << ',';
    EMIT_VALUE(EffectType::ToActiveAndTrashActive); std::cout << ',';
    EMIT_VALUE(EffectType::ToBench); std::cout << ',';
    EMIT_VALUE(EffectType::ToPrize); std::cout << ',';
    EMIT_VALUE(EffectType::ToLooking); std::cout << ',';
    EMIT_VALUE(EffectType::ToPlayingFirst); std::cout << ',';
    EMIT_VALUE(EffectType::Switch); std::cout << ',';
    EMIT_VALUE(EffectType::SwitchDeck); std::cout << ',';
    EMIT_VALUE(EffectType::NotMove); std::cout << ',';
    EMIT_VALUE(EffectType::LookDeck); std::cout << ',';
    EMIT_VALUE(EffectType::LookDeckReverse); std::cout << ',';
    EMIT_VALUE(EffectType::LookDeckBottom); std::cout << ',';
    EMIT_VALUE(EffectType::LookAndReturn); std::cout << ',';
    EMIT_VALUE(EffectType::DamageCounter); std::cout << ',';
    EMIT_VALUE(EffectType::AttackDamage); std::cout << ',';
    EMIT_VALUE(EffectType::RemoveDamageCounter); std::cout << ',';
    EMIT_VALUE(EffectType::RemoveDamageCounterAll); std::cout << ',';
    EMIT_VALUE(EffectType::Heal); std::cout << ',';
    EMIT_VALUE(EffectType::HealAll); std::cout << ',';
    EMIT_VALUE(EffectType::ResetHp); std::cout << ',';
    EMIT_VALUE(EffectType::DelayEffect); std::cout << ',';
    EMIT_VALUE(EffectType::Coin); std::cout << ',';
    EMIT_VALUE(EffectType::CoinUntilTail); std::cout << ',';
    EMIT_VALUE(EffectType::Burn); std::cout << ',';
    EMIT_VALUE(EffectType::Poison); std::cout << ',';
    EMIT_VALUE(EffectType::Poison8); std::cout << ',';
    EMIT_VALUE(EffectType::Poison16); std::cout << ',';
    EMIT_VALUE(EffectType::Sleep); std::cout << ',';
    EMIT_VALUE(EffectType::Confuse); std::cout << ',';
    EMIT_VALUE(EffectType::Paralyze); std::cout << ',';
    EMIT_VALUE(EffectType::RecoverSpecialCondition); std::cout << ',';
    EMIT_VALUE(EffectType::RecoverSpecialConditionSingle); std::cout << ',';
    EMIT_VALUE(EffectType::Draw); std::cout << ',';
    EMIT_VALUE(EffectType::DrawTargetCount); std::cout << ',';
    EMIT_VALUE(EffectType::DrawPrizeCount); std::cout << ',';
    EMIT_VALUE(EffectType::DrawUntil); std::cout << ',';
    EMIT_VALUE(EffectType::DrawUntilPsychic); std::cout << ',';
    EMIT_VALUE(EffectType::DrawMirror); std::cout << ',';
    EMIT_VALUE(EffectType::DeckToTrash); std::cout << ',';
    EMIT_VALUE(EffectType::DeckToTrashCoinUntilTail); std::cout << ',';
    EMIT_VALUE(EffectType::DeckBottomToTrash); std::cout << ',';
    EMIT_VALUE(EffectType::DeckToPrize); std::cout << ',';
    EMIT_VALUE(EffectType::Shuffle); std::cout << ',';
    EMIT_VALUE(EffectType::EffectWin); std::cout << ',';
    EMIT_VALUE(EffectType::TurnEnd); std::cout << ',';
    EMIT_VALUE(EffectType::ContinualEffectSeparator); std::cout << ',';
    EMIT_VALUE(EffectType::NoToolEffect);
    std::cout << "},";

    std::cout << "\"domain_max\":{";
    std::cout << "\"effect\":" << static_cast<int>(EffectType::NoToolEffect) << ',';
    std::cout << "\"target\":" << static_cast<int>(TargetType::NotChecked) << ',';
    std::cout << "\"condition\":" << static_cast<int>(ConditionType::LoveBall) << ',';
    std::cout << "\"trigger\":" << static_cast<int>(TriggerType::Attach) << ',';
    std::cout << "\"select_type\":" << static_cast<int>(SelectType::SpecialCondition) << ',';
    std::cout << "\"select_context\":" << static_cast<int>(SelectContext::RecoverSpecialCondition) << ',';
    std::cout << "\"select_option\":" << static_cast<int>(SelectOptionType::SpecialCondition);
    std::cout << "}}\n";
    return 0;
}

#undef EMIT_VALUE
