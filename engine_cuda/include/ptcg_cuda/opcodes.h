#pragma once

#include <cstdint>

namespace ptcg::cuda_engine {

constexpr std::uint32_t kAbiVersion = 1;
constexpr std::uint32_t kRulePackVersion = 1;
constexpr std::uint32_t kMaxPolicies = 64;
constexpr std::uint32_t kMaxCardInstances = 160;
constexpr std::uint32_t kMaxEffectFrames = 128;
constexpr std::uint32_t kMaxContinuationFrames = 256;
constexpr std::uint32_t kMaxDelayedEffects = 32;
constexpr std::uint32_t kMaxCounters = 32;
constexpr std::uint32_t kMaxCodecEntities = 128;
// The semantic0031 schema expands attached cards and resolved Energy units
// into separate ragged card tokens.  Its audited training buckets support 160
// rows, while the legacy POD-native codec remains fixed at 128.
constexpr std::uint32_t kMaxSemantic0031CardEntities = 160;
constexpr std::uint32_t kMaxCodecOptions = 80;
constexpr std::uint32_t kInterpreterBudget = 64;

enum class EngineStatus : std::int32_t {
    kEmpty = 0,
    kNeedsPolicy = 1,
    kAdvancing = 2,
    kTerminal = 3,
    kError = 4,
};

enum class EngineError : std::int32_t {
    kNone = 0,
    kInvalidAction = 1,
    kRulePackBounds = 2,
    kInterpreterBudget = 3,
    kStackOverflow = 4,
    kRouteOverflow = 5,
    kCodecEntityOverflow = 6,
    kCodecOptionOverflow = 7,
    kUnsupportedOpcode = 8,
    kInvalidPolicyId = 9,
    kKnownDivergence6601207 = 6601207,
};

enum class Opcode : std::uint8_t {
    kNop = 0,
    kDraw = 1,
    kDamageActive = 2,
    kHealActive = 3,
    kEndTurn = 4,
    kCheckKnockout = 5,
    kSetWinner = 6,
    kEmitDecision = 7,
    kSetCounter = 8,
    kAddCounter = 9,
    kDelayEffect = 10,
    kMoveCard = 11,
    kConditionalCounterGe = 12,
    kRandomBranch = 13,
    kHalt = 255,
};

enum class Target : std::uint8_t {
    kActor = 0,
    kOpponent = 1,
    kPlayer0 = 2,
    kPlayer1 = 3,
};

enum class Zone : std::uint8_t {
    kNone = 0,
    kDeck = 1,
    kHand = 2,
    kActive = 3,
    kBench = 4,
    kDiscard = 5,
    kPrize = 6,
    kLost = 7,
    kStadium = 8,
};

}  // namespace ptcg::cuda_engine
