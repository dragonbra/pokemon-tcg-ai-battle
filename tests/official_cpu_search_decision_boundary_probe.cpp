// Research-only diagnostic. Including the official Export.cpp keeps this probe
// on the unmodified CPU engine while exposing internal State metadata for audit.
#include "Export.cpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

uint64_t fnv1a(const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string hex_hash(uint64_t hash) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << hash;
  return out.str();
}

std::string state_hash(const State& state) {
  BinaryWriter writer;
  state.serialize(writer);
  return hex_hash(fnv1a(writer.buf.data(), writer.buf.size()));
}

std::string rng_hash(const Game& game) {
  std::ostringstream serialized;
  serialized << game.rng;
  const std::string value = serialized.str();
  return hex_hash(fnv1a(value.data(), value.size()));
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

const char* option_name(SelectOptionType type) {
  switch (type) {
    case SelectOptionType::Number: return "Number";
    case SelectOptionType::Yes: return "Yes";
    case SelectOptionType::No: return "No";
    case SelectOptionType::Card: return "Card";
    case SelectOptionType::ToolCard: return "ToolCard";
    case SelectOptionType::EnergyCard: return "EnergyCard";
    case SelectOptionType::Energy: return "Energy";
    case SelectOptionType::Play: return "Play";
    case SelectOptionType::Attach: return "Attach";
    case SelectOptionType::Evolve: return "Evolve";
    case SelectOptionType::Ability: return "Ability";
    case SelectOptionType::Discard: return "Discard";
    case SelectOptionType::Retreat: return "Retreat";
    case SelectOptionType::Attack: return "Attack";
    case SelectOptionType::End: return "End";
    case SelectOptionType::Skill: return "Skill";
    case SelectOptionType::SpecialCondition: return "SpecialCondition";
  }
  return "Unknown";
}

uint64_t permutation_count(int n, int k) {
  uint64_t result = 1;
  for (int i = 0; i < k; ++i) {
    const uint64_t factor = static_cast<uint64_t>(n - i);
    if (factor != 0 && result > std::numeric_limits<uint64_t>::max() / factor) {
      return std::numeric_limits<uint64_t>::max();
    }
    result *= factor;
  }
  return result;
}

// Counts exactly the distinct ordered index vectors accepted by
// State::checkPlayerSelect(), not unordered semantic equivalence classes.
uint64_t accepted_vector_count(const State& state) {
  const int n = state.options.size();
  const int min_count = std::clamp(static_cast<int>(state.selectMin), 0, n);
  const int max_count = std::clamp(static_cast<int>(state.selectMax), 0, n);
  uint64_t result = 0;
  for (int k = min_count; k <= max_count; ++k) {
    const uint64_t add = permutation_count(n, k);
    if (add == std::numeric_limits<uint64_t>::max() ||
        result > std::numeric_limits<uint64_t>::max() - add) {
      return std::numeric_limits<uint64_t>::max();
    }
    result += add;
  }
  return result;
}

struct LegalSelectionEnumeration {
  std::vector<std::vector<int>> vectors;
  bool truncated = false;
};

void enumerate_length(
    int option_count, int target_length, size_t limit, std::vector<int>& current,
    std::vector<bool>& used, LegalSelectionEnumeration& result) {
  if (result.vectors.size() >= limit) {
    result.truncated = true;
    return;
  }
  if (current.size() == target_length) {
    result.vectors.push_back(current);
    return;
  }
  for (int index = 0; index < option_count; ++index) {
    if (used[index]) continue;
    used[index] = true;
    current.push_back(index);
    enumerate_length(option_count, target_length, limit, current, used, result);
    current.pop_back();
    used[index] = false;
    if (result.truncated) return;
  }
}

LegalSelectionEnumeration enumerate_legal_selections(const State& state, size_t limit = 10000) {
  LegalSelectionEnumeration result;
  const int n = state.options.size();
  const int min_count = std::clamp(static_cast<int>(state.selectMin), 0, n);
  const int max_count = std::clamp(static_cast<int>(state.selectMax), 0, n);
  std::vector<int> current;
  std::vector<bool> used(n, false);
  for (int k = min_count; k <= max_count && !result.truncated; ++k) {
    enumerate_length(n, k, limit, current, used, result);
  }
  return result;
}

enum class DecisionClass {
  Terminal,
  NoSelection,
  ForcedSingleSelection,
  FocalBranchingDecision,
  OpponentBranchingDecision,
  Unknown,
};

const char* decision_class_name(DecisionClass value) {
  switch (value) {
    case DecisionClass::Terminal: return "TERMINAL";
    case DecisionClass::NoSelection: return "NO_SELECTION";
    case DecisionClass::ForcedSingleSelection: return "FORCED_SINGLE_SELECTION";
    case DecisionClass::FocalBranchingDecision: return "FOCAL_BRANCHING_DECISION";
    case DecisionClass::OpponentBranchingDecision: return "OPPONENT_BRANCHING_DECISION";
    case DecisionClass::Unknown: return "UNKNOWN";
  }
  return "UNKNOWN";
}

DecisionClass classify_decision(const State& state, int focal_player) {
  if (state.isFinish()) return DecisionClass::Terminal;
  if (state.selectMax == 0) return DecisionClass::NoSelection;
  const uint64_t count = accepted_vector_count(state);
  if (count == 1) return DecisionClass::ForcedSingleSelection;
  if (state.selectPlayer == focal_player) return DecisionClass::FocalBranchingDecision;
  if (state.selectPlayer == 1 - focal_player) return DecisionClass::OpponentBranchingDecision;
  return DecisionClass::Unknown;
}

std::string options_json(const State& state) {
  std::ostringstream out;
  out << '[';
  for (int i = 0; i < state.options.size(); ++i) {
    if (i) out << ',';
    out << '"' << i << ':' << option_name(state.options[i].type) << '"';
  }
  out << ']';
  return out.str();
}

void dump_decision(const char* label, const State& state, int focal_player) {
  std::cout << label
            << " {hash:" << state_hash(state)
            << ", turn:" << state.turn
            << ", activePlayer:" << state.activePlayerIndex()
            << ", selectPlayer:" << static_cast<int>(state.selectPlayer)
            << ", selectType:" << SelectTypeStr[static_cast<int>(state.selectType)]
            << ", context:" << SelectContextStr[static_cast<int>(state.selectContext)]
            << ", min:" << static_cast<int>(state.selectMin)
            << ", max:" << static_cast<int>(state.selectMax)
            << ", options:" << state.options.size()
            << ", acceptedVectors:" << accepted_vector_count(state)
            << ", class:" << decision_class_name(classify_decision(state, focal_player))
            << ", payload:" << options_json(state)
            << "}\n";
}

std::vector<int> card_ids(const State& state, const CardList& cards) {
  std::vector<int> result;
  result.reserve(cards.size());
  for (CardRef ref : cards) result.push_back(state.getCardId(ref));
  return result;
}

int find_option(const State& state, SelectOptionType type) {
  for (int i = 0; i < state.options.size(); ++i) {
    if (state.options[i].type == type) return i;
  }
  return -1;
}

int find_attack(const State& state, int attack_id) {
  for (int i = 0; i < state.options.size(); ++i) {
    if (state.options[i].type == SelectOptionType::Attack &&
        state.options[i].param0 == attack_id) {
      return i;
    }
  }
  return -1;
}

int find_attach_to_active(const State& state) {
  for (int i = 0; i < state.options.size(); ++i) {
    const SelectOption& option = state.options[i];
    if (option.type == SelectOptionType::Attach &&
        static_cast<AreaType>(option.param2) == AreaType::Active) {
      return i;
    }
  }
  return -1;
}

std::vector<int> simple_basic_ids(int excluded_id, size_t count) {
  std::vector<int> result;
  std::unordered_set<std::u8string> names;
  for (const auto& [id, master] : CardTable) {
    if (id != excluded_id && master.cardType == CardType::Pokemon &&
        master.evolutionType == EvolutionType::Basic && master.canSetup() &&
        master.ability == nullptr && names.insert(master.name).second) {
      result.push_back(id);
      if (result.size() == count) break;
    }
  }
  require(result.size() == count, "could not locate enough simple Basic Pokemon");
  return result;
}

std::vector<int> make_deck(int active_id, int energy_id) {
  std::vector<int> deck;
  for (int copy = 0; copy < 4; ++copy) deck.push_back(active_id);
  for (int id : simple_basic_ids(active_id, 3)) {
    for (int copy = 0; copy < 4; ++copy) deck.push_back(id);
  }
  while (deck.size() < DECK_SIZE) deck.push_back(energy_id);
  return deck;
}

std::vector<int> setup_selection(const State& state, int active_id, int bench_count) {
  if (state.selectContext == SelectContext::SetupActivePokemon) {
    for (int i = 0; i < state.options.size(); ++i) {
      const CardPosition pos = state.options[i].getCardPosition();
      CardRef ref = state.players.at(pos.playerIndex).hand.at(pos.areaIndex);
      if (state.getCardId(ref) == active_id) return {i};
    }
    return {0};
  }
  if (state.selectContext == SelectContext::SetupBenchPokemon) {
    std::vector<int> selected;
    for (int i = 0; i < state.options.size() && selected.size() < bench_count; ++i) {
      selected.push_back(i);
    }
    return selected;
  }
  if (state.selectContext == SelectContext::IsFirst) return {0};
  if (state.selectMin == 0) return {};
  return {0};
}

std::unique_ptr<ApiData> build_first_turn_main(
    int active_id, int energy_id, int bench_count, bool require_play, uint32_t& seed) {
  const std::vector<int> deck = make_deck(active_id, energy_id);
  for (seed = 1; seed <= 50000; ++seed) {
    GameConfig config = {};
    config.seed = seed;
    config.recordLog = true;
    for (int player = 0; player < 2; ++player) {
      for (int i = 0; i < DECK_SIZE; ++i) config.decks[player].cards[i] = deck[i];
    }
    auto battle = std::make_unique<ApiData>();
    battle->apiDataType = 1;
    battle->init(config);
    battle->start();
    battle->next();
    for (int decisions = 0; decisions < 100 &&
         battle->state.selectContext != SelectContext::Main; ++decisions) {
      std::vector<int> selected = setup_selection(battle->state, active_id, bench_count);
      int dummy = 0;
      int error = ApiSelect(battle.get(), selected.empty() ? &dummy : selected.data(), selected.size());
      if (error != 0) throw std::runtime_error("setup selection failed: " + std::to_string(error));
    }
    State& state = battle->state;
    if (state.selectContext != SelectContext::Main) continue;
    const int actor = state.selectPlayer;
    bool both_players_match = true;
    for (int player = 0; player < 2; ++player) {
      if (state.players[player].active.empty() ||
          state.getCardId(state.players[player].getActive()) != active_id ||
          state.players[player].bench.size() < bench_count) {
        both_players_match = false;
      }
    }
    if (!both_players_match) continue;
    if (require_play && find_option(state, SelectOptionType::Play) < 0) continue;
    if (find_attach_to_active(state) < 0) continue;
    return battle;
  }
  throw std::runtime_error("no seed produced the required official setup");
}

void advance_to_second_player_main(ApiData& battle, int expected_active_id) {
  const int first_player = battle.state.selectPlayer;
  const int end = find_option(battle.state, SelectOptionType::End);
  require(end >= 0, "first player has no End option");
  int selected[] = {end};
  require(ApiSelect(&battle, selected, 1) == 0, "ending first turn failed");
  require(battle.state.selectContext == SelectContext::Main, "did not reach second player Main");
  require(battle.state.selectPlayer == 1 - first_player, "turn actor did not switch");
  require(battle.state.getCardId(battle.state.players[battle.state.selectPlayer].getActive()) ==
              expected_active_id,
          "second player's Active is not the requested fixture Pokemon");
}

struct SearchFixture {
  ApiData* agent = nullptr;
  int root_id = -1;
  int focal_player = -1;

  SearchFixture() = default;
  SearchFixture(const SearchFixture&) = delete;
  SearchFixture& operator=(const SearchFixture&) = delete;
  SearchFixture(SearchFixture&& other) noexcept
      : agent(other.agent), root_id(other.root_id), focal_player(other.focal_player) {
    other.agent = nullptr;
  }

  ~SearchFixture() {
    if (agent != nullptr) {
      SearchEnd(agent);
      BattleFinish(agent);
    }
  }
};

int* ptr_or_dummy(std::vector<int>& values) {
  static int dummy = 0;
  return values.empty() ? &dummy : values.data();
}

SearchFixture begin_search(ApiData& battle, uint32_t rng_seed) {
  State& live = battle.state;
  const int me = live.selectPlayer;
  const int enemy = 1 - me;
  SerialData serial = GetBattleData(&battle);
  require(serial.data != nullptr && serial.count > 0, "GetBattleData failed");
  std::vector<int> my_deck = card_ids(live, live.players[me].deck);
  std::vector<int> my_prize = card_ids(live, live.players[me].prize);
  std::vector<int> enemy_deck = card_ids(live, live.players[enemy].deck);
  std::vector<int> enemy_prize = card_ids(live, live.players[enemy].prize);
  std::vector<int> enemy_hand = card_ids(live, live.players[enemy].hand);
  std::vector<int> enemy_active;

  SearchFixture fixture;
  fixture.agent = AgentStart();
  require(fixture.agent != nullptr, "AgentStart failed");
  fixture.agent->game.rng = std::mt19937(rng_seed);
  const char8_t* json = SearchBegin(
      fixture.agent, serial.data, serial.count,
      ptr_or_dummy(my_deck), ptr_or_dummy(my_prize), ptr_or_dummy(enemy_deck),
      ptr_or_dummy(enemy_prize), ptr_or_dummy(enemy_hand), ptr_or_dummy(enemy_active), 0);
  require(json != nullptr, "SearchBegin failed");
  fixture.root_id = fixture.agent->search.lastSearchId();
  fixture.focal_player = me;
  return fixture;
}

const State& search_step(SearchFixture& fixture, int source_id, std::vector<int> selected) {
  int dummy = 0;
  const char8_t* json = SearchStep(
      fixture.agent, source_id, selected.empty() ? &dummy : selected.data(), selected.size());
  require(json != nullptr, "SearchStep returned null");
  return fixture.agent->search.lastState();
}

void test_automatic_forced_and_branching(int energy_id, int active_id) {
  uint32_t seed = 0;
  auto battle = build_first_turn_main(active_id, energy_id, 2, true, seed);
  SearchFixture search = begin_search(*battle, 0xD371A5u);
  const State& root = search.agent->search.lastState();
  const std::string rng_before = rng_hash(search.agent->game);
  dump_decision("TEST1_ROOT", root, search.focal_player);

  const int play = find_option(root, SelectOptionType::Play);
  require(play >= 0, "TEST1 missing Play");
  const State& after_play = search_step(search, search.root_id, {play});
  int id = search.agent->search.lastSearchId();
  dump_decision("TEST1_AFTER_PLAY", after_play, search.focal_player);
  require(after_play.selectContext == SelectContext::Main,
          "TEST1 Play did not auto-resolve to Main");
  std::cout << "TEST 1 PASS ordinary automatic resolution returned at Main\n";

  const int attach = find_attach_to_active(after_play);
  require(attach >= 0, "TEST2 missing Attach-to-Active");
  const State& after_attach = search_step(search, id, {attach});
  id = search.agent->search.lastSearchId();
  const int retreat = find_option(after_attach, SelectOptionType::Retreat);
  require(retreat >= 0, "TEST2 missing Retreat");
  const State& energy_payment = search_step(search, id, {retreat});
  id = search.agent->search.lastSearchId();
  dump_decision("TEST2_FORCED_ENERGY", energy_payment, search.focal_player);
  require(energy_payment.selectContext == SelectContext::DiscardEnergy,
          "TEST2 did not stop at DiscardEnergy");
  require(accepted_vector_count(energy_payment) == 1,
          "TEST2 DiscardEnergy is not a unique accepted vector");
  require(enumerate_legal_selections(energy_payment).vectors ==
              std::vector<std::vector<int>>{{0}},
          "TEST2 enumerator did not return the unique vector [0]");
  std::cout << "TEST 2 PASS forced singleton represented as min=1 max=1 options=1\n";

  const State& switch_target = search_step(search, id, {0});
  id = search.agent->search.lastSearchId();
  dump_decision("TEST3_FOCAL_SWITCH", switch_target, search.focal_player);
  require(switch_target.selectContext == SelectContext::Switch,
          "TEST3 did not stop at Switch");
  require(switch_target.options.size() >= 2 && accepted_vector_count(switch_target) >= 2,
          "TEST3 Switch is not a true multi-way decision");
  require(switch_target.selectPlayer == search.focal_player,
          "TEST3 Switch is not focal-owned");
  std::cout << "TEST 3 PASS focal true branch has multiple accepted vectors\n";

  const State& after_switch = search_step(search, id, {0});
  dump_decision("TEST5_AFTER_CHAIN", after_switch, search.focal_player);
  const std::string rng_after = rng_hash(search.agent->game);
  std::cout << "TEST5_RNG {before:" << rng_before << ", after:" << rng_after << "}\n";
  require(rng_before == rng_after, "TEST5 deterministic chain consumed RNG");
  std::cout << "TEST 5 PASS public SearchStep deterministic chain is RNG-free\n";
}

void test_opponent_owned_decision() {
  constexpr int hippopotas_id = 22;
  constexpr int fighting_energy_id = 6;
  constexpr int push_down_attack_id = 3;
  uint32_t seed = 0;
  auto battle = build_first_turn_main(hippopotas_id, fighting_energy_id, 2, false, seed);
  advance_to_second_player_main(*battle, hippopotas_id);
  SearchFixture search = begin_search(*battle, 0x0A110CEu);
  const State& root = search.agent->search.lastState();
  const int attach = find_attach_to_active(root);
  require(attach >= 0, "TEST4 missing Fighting attachment");
  const State& after_attach = search_step(search, search.root_id, {attach});
  const int attach_id = search.agent->search.lastSearchId();
  const int attack = find_attack(after_attach, push_down_attack_id);
  require(attack >= 0, "TEST4 missing Push Down attack");
  const State& opponent_switch = search_step(search, attach_id, {attack});
  dump_decision("TEST4_OPPONENT_SWITCH", opponent_switch, search.focal_player);
  require(opponent_switch.selectContext == SelectContext::Switch,
          "TEST4 did not stop at Push Down Switch");
  require(opponent_switch.selectPlayer == 1 - search.focal_player,
          "TEST4 selection is not opponent-owned");
  require(opponent_switch.options.size() >= 2,
          "TEST4 opponent Switch does not have multiple targets");
  std::cout << "TEST 4 PASS focal attack stopped at opponent-owned decision\n";
}

void test_rng_consuming_attack(int energy_id) {
  constexpr int hoothoot_id = 172;
  constexpr int triple_stab_attack_id = 228;
  uint32_t seed = 0;
  auto battle = build_first_turn_main(hoothoot_id, energy_id, 0, false, seed);
  advance_to_second_player_main(*battle, hoothoot_id);
  SearchFixture search = begin_search(*battle, 0xC01F11u);
  const State& root = search.agent->search.lastState();
  const int attach = find_attach_to_active(root);
  require(attach >= 0, "TEST6 missing attachment");
  const State& after_attach = search_step(search, search.root_id, {attach});
  const int attach_id = search.agent->search.lastSearchId();
  const int attack = find_attack(after_attach, triple_stab_attack_id);
  require(attack >= 0, "TEST6 missing Triple Stab attack");
  const std::string rng_before = rng_hash(search.agent->game);
  const State& after_attack = search_step(search, attach_id, {attack});
  const std::string rng_after = rng_hash(search.agent->game);
  dump_decision("TEST6_AFTER_COIN_ATTACK", after_attack, search.focal_player);
  std::cout << "TEST6_RNG {before:" << rng_before << ", after:" << rng_after << "}\n";
  require(rng_before != rng_after, "TEST6 coin attack did not advance Game::rng");
  require(after_attack.selectContext == SelectContext::Main,
          "TEST6 attack did not resolve through the next Main decision");
  std::cout << "TEST 6 PASS public SearchStep coin attack consumed RNG\n";
}

void test_ordered_vector_contract() {
  State state;
  state.selectMin = 2;
  state.selectMax = 2;
  state.options.resize(2);
  state.selected = {0, 1};
  require(state.checkPlayerSelect() == 0, "[0,1] unexpectedly rejected");
  state.selected = {1, 0};
  require(state.checkPlayerSelect() == 0, "[1,0] unexpectedly rejected");
  state.selected = {0, 0};
  require(state.checkPlayerSelect() == 6, "duplicate vector unexpectedly accepted");
  const LegalSelectionEnumeration enumeration = enumerate_legal_selections(state);
  require(enumeration.vectors == std::vector<std::vector<int>>{{0, 1}, {1, 0}},
          "ordered-vector enumerator disagrees with checker contract");
  std::cout << "ORDERED_VECTOR_CONTRACT {options:2, min:2, max:2, acceptedVectors:"
            << accepted_vector_count(state)
            << ", accepted:[0,1]|[1,0], rejected:[0,0]}\n";
}

}  // namespace

int main() {
  try {
    GameInitialize();
    int energy_id = 0;
    int ordinary_active_id = 0;
    for (const auto& [id, master] : CardTable) {
      if (energy_id == 0 && master.cardType == CardType::BasicEnergy) energy_id = id;
      if (ordinary_active_id == 0 && master.cardType == CardType::Pokemon &&
          master.evolutionType == EvolutionType::Basic && master.retreatCost == 1 &&
          master.ability == nullptr && master.canSetup()) {
        ordinary_active_id = id;
      }
    }
    require(energy_id != 0 && ordinary_active_id != 0, "base fixture cards unavailable");

    std::cout << "OFFICIAL_CPU_SEARCH_DECISION_BOUNDARY_PROBE_V1\n";
    test_ordered_vector_contract();
    test_automatic_forced_and_branching(energy_id, ordinary_active_id);
    test_opponent_owned_decision();
    test_rng_consuming_attack(energy_id);
    std::cout << "RESULT PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "RESULT FAIL: " << error.what() << '\n';
    return 1;
  }
}
