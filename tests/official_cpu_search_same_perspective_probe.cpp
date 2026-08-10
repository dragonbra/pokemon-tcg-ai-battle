// Research-only Phase 3 probe. Export.cpp is included without modifying the
// official CPU engine, exposing State metadata alongside the public Search API.
#include "Export.cpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

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

std::string copy_json(const char8_t* value) {
  require(value != nullptr, "official API returned null JSON");
  return reinterpret_cast<const char*>(value);
}

std::string observation_json(State& state, JsonBuilder& builder) {
  builder.clear();
  ToJsonApi(state, builder, state.nextLogStart());
  return copy_json(builder.buf.c_str());
}

void emit(const std::string& case_name, const std::string& kind, const std::string& json) {
  std::cout << "PHASE3_JSON\t" << case_name << '\t' << kind << '\t' << json << '\n';
}

void emit_meta(const std::string& case_name, const State& root, const State& branch,
               const std::string& rng_before, const std::string& rng_after,
               int alternatives, bool hidden_perturbed_equal) {
  std::ostringstream out;
  out << "{\"rootTurn\":" << root.turn
      << ",\"branchTurn\":" << branch.turn
      << ",\"rootActivePlayer\":" << root.activePlayerIndex()
      << ",\"branchActivePlayer\":" << branch.activePlayerIndex()
      << ",\"rootMainPhase\":" << (root.isPlayerTurn() ? "true" : "false")
      << ",\"branchMainPhase\":" << (branch.isPlayerTurn() ? "true" : "false")
      << ",\"rootSelectPlayer\":" << static_cast<int>(root.selectPlayer)
      << ",\"branchSelectPlayer\":" << static_cast<int>(branch.selectPlayer)
      << ",\"rootSelectType\":\"" << SelectTypeStr[static_cast<int>(root.selectType)] << "\""
      << ",\"rootSelectContext\":\"" << SelectContextStr[static_cast<int>(root.selectContext)] << "\""
      << ",\"branchSelectType\":\"" << SelectTypeStr[static_cast<int>(branch.selectType)] << "\""
      << ",\"branchSelectContext\":\"" << SelectContextStr[static_cast<int>(branch.selectContext)] << "\""
      << ",\"terminal\":" << (branch.isFinish() ? "true" : "false")
      << ",\"alternatives\":" << alternatives
      << ",\"rootHash\":\"" << state_hash(root) << "\""
      << ",\"branchHash\":\"" << state_hash(branch) << "\""
      << ",\"rngBefore\":\"" << rng_before << "\""
      << ",\"rngAfter\":\"" << rng_after << "\""
      << ",\"rngUnchanged\":" << (rng_before == rng_after ? "true" : "false")
      << ",\"hiddenPerturbedObservationEqual\":"
      << (hidden_perturbed_equal ? "true" : "false") << '}';
  emit(case_name, "META", out.str());
}

std::vector<int> card_ids(const State& state, const CardList& cards) {
  std::vector<int> result;
  result.reserve(cards.size());
  for (CardRef ref : cards) result.push_back(state.getCardId(ref));
  return result;
}

std::string int_array_json(const std::vector<int>& values) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}

int* ptr_or_dummy(std::vector<int>& values) {
  static int dummy = 0;
  return values.empty() ? &dummy : values.data();
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
  require(result.size() == count, "not enough simple Basic Pokemon");
  return result;
}

std::vector<int> make_deck(int active_id, int energy_a, int energy_b) {
  std::vector<int> deck;
  for (int i = 0; i < 4; ++i) deck.push_back(active_id);
  for (int id : simple_basic_ids(active_id, 4)) {
    for (int i = 0; i < 4; ++i) deck.push_back(id);
  }
  for (int i = 0; i < 4; ++i) deck.push_back(1116);  // Energy Switch
  for (int i = 0; i < 4; ++i) deck.push_back(1181);  // Billy & O'Nare
  for (int i = 0; i < 4; ++i) deck.push_back(1182);  // Boss's Orders
  while (deck.size() < 48) deck.push_back(energy_a);
  while (deck.size() < DECK_SIZE) deck.push_back(energy_b);
  return deck;
}

std::vector<int> setup_selection(const State& state, int active_id, int bench_count) {
  if (state.selectContext == SelectContext::SetupActivePokemon) {
    for (int i = 0; i < state.options.size(); ++i) {
      CardPosition pos = state.options[i].getCardPosition();
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

std::unique_ptr<ApiData> build_first_turn(
    const std::vector<int>& deck, int active_id, int bench_count, uint32_t seed) {
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
    require(ApiSelect(battle.get(), ptr_or_dummy(selected), selected.size()) == 0,
            "official setup selection failed");
  }
  require(battle->state.selectContext == SelectContext::Main, "setup did not reach Main");
  return battle;
}

int find_option(const State& state, SelectOptionType type) {
  for (int i = 0; i < state.options.size(); ++i) {
    if (state.options[i].type == type) return i;
  }
  return -1;
}

std::vector<int> play_options_for_card(const State& state, int card_id) {
  std::vector<int> result;
  const PlayerState& player = state.players[state.selectPlayer];
  for (int i = 0; i < state.options.size(); ++i) {
    const SelectOption& option = state.options[i];
    if (option.type == SelectOptionType::Play &&
        state.getCardId(player.hand.at(option.param0)) == card_id) result.push_back(i);
  }
  return result;
}

std::vector<int> basic_play_options(const State& state) {
  std::vector<int> result;
  const PlayerState& player = state.players[state.selectPlayer];
  for (int i = 0; i < state.options.size(); ++i) {
    const SelectOption& option = state.options[i];
    if (option.type != SelectOptionType::Play) continue;
    const CardMaster& master = state.getCard(player.hand.at(option.param0)).getMaster();
    if (master.cardType == CardType::Pokemon && master.evolutionType == EvolutionType::Basic) {
      result.push_back(i);
    }
  }
  return result;
}

std::vector<int> attach_options(const State& state, int energy_id = 0) {
  std::vector<int> result;
  const PlayerState& player = state.players[state.selectPlayer];
  for (int i = 0; i < state.options.size(); ++i) {
    const SelectOption& option = state.options[i];
    if (option.type != SelectOptionType::Attach) continue;
    if (energy_id != 0 && state.getCardId(player.hand.at(option.param1)) != energy_id) continue;
    result.push_back(i);
  }
  return result;
}

int choose_attach(const State& state, int energy_id, AreaType target) {
  for (int index : attach_options(state, energy_id)) {
    if (static_cast<AreaType>(state.options[index].param2) == target) return index;
  }
  return -1;
}

void select_one(ApiData& battle, int option) {
  require(option >= 0, "required legal option was absent");
  int selected[] = {option};
  require(ApiSelect(&battle, selected, 1) == 0, "official live selection failed");
}

void end_turn(ApiData& battle) { select_one(battle, find_option(battle.state, SelectOptionType::End)); }

struct SearchFixture {
  ApiData* agent = nullptr;
  int root_id = -1;
  int focal = -1;
  std::string begin_json;

  SearchFixture() = default;
  SearchFixture(const SearchFixture&) = delete;
  SearchFixture& operator=(const SearchFixture&) = delete;
  SearchFixture(SearchFixture&& other) noexcept
      : agent(other.agent), root_id(other.root_id), focal(other.focal),
        begin_json(std::move(other.begin_json)) { other.agent = nullptr; }
  ~SearchFixture() {
    if (agent != nullptr) {
      SearchEnd(agent);
      BattleFinish(agent);
    }
  }
};

SearchFixture begin_search(ApiData& battle, uint32_t rng_seed, bool perturb_hidden) {
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
  if (perturb_hidden) {
    auto permute_distinct = [](std::vector<int>& values) {
      if (values.size() < 2) return false;
      size_t index = values.size() - 2;
      while (index > 0 && values[index] == values.back()) --index;
      if (values[index] == values.back()) return false;
      std::swap(values[index], values.back());
      return true;
    };
    require(permute_distinct(my_deck), "own hidden deck has no distinct permutation");
    permute_distinct(my_prize);
    permute_distinct(enemy_deck);
    permute_distinct(enemy_prize);
    permute_distinct(enemy_hand);
  }

  SearchFixture fixture;
  fixture.agent = AgentStart();
  require(fixture.agent != nullptr, "AgentStart failed");
  fixture.agent->game.rng = std::mt19937(rng_seed);
  fixture.begin_json = copy_json(SearchBegin(
      fixture.agent, serial.data, serial.count,
      ptr_or_dummy(my_deck), ptr_or_dummy(my_prize), ptr_or_dummy(enemy_deck),
      ptr_or_dummy(enemy_prize), ptr_or_dummy(enemy_hand), ptr_or_dummy(enemy_active), 0));
  fixture.root_id = fixture.agent->search.lastSearchId();
  fixture.focal = me;
  return fixture;
}

std::string search_step(SearchFixture& fixture, int source_id, int option) {
  int selected[] = {option};
  return copy_json(SearchStep(fixture.agent, source_id, selected, 1));
}

void emit_deck(const std::string& case_name, const std::vector<int>& deck) {
  emit(case_name, "REGISTERED_DECK", int_array_json(deck));
}

// Search the official shuffle/setup outcomes for a stable legal fixture.
template <class Predicate>
std::unique_ptr<ApiData> find_fixture(
    const std::vector<int>& deck, int active_id, int bench_count, Predicate predicate,
    uint32_t& selected_seed) {
  for (uint32_t seed = 1; seed <= 50000; ++seed) {
    auto battle = build_first_turn(deck, active_id, bench_count, seed);
    bool board_matches = true;
    for (int player = 0; player < 2; ++player) {
      const PlayerState& ps = battle->state.players[player];
      if (ps.active.empty() || battle->state.getCardId(ps.getActive()) != active_id ||
          ps.bench.size() < bench_count) {
        board_matches = false;
      }
    }
    if (!board_matches) continue;
    if (predicate(*battle)) {
      selected_seed = seed;
      return battle;
    }
  }
  throw std::runtime_error("no seed produced required legal fixture");
}

void run_branch_case(const std::string& name, ApiData& battle, const std::vector<int>& deck,
                     const std::vector<int>& choices, int live_choice) {
  require(choices.size() >= 2, name + " does not have two candidates");
  const std::string root_observation = observation_json(battle.state, battle.jsonBuilder);
  SearchFixture canonical = begin_search(battle, 0x503300u, false);
  SearchFixture perturbed = begin_search(battle, 0x503300u, true);
  emit(name, "ROOT_OBSERVATION", root_observation);
  emit(name, "SEARCH_BEGIN", canonical.begin_json);
  emit(name, "PERTURBED_SEARCH_BEGIN", perturbed.begin_json);
  emit_deck(name, deck);

  const State root_copy = canonical.agent->search.lastState();
  const std::string rng_before = rng_hash(canonical.agent->game);
  std::string first_json;
  std::string first_hash;
  for (size_t i = 0; i < choices.size(); ++i) {
    const std::string branch_json = search_step(canonical, canonical.root_id, choices[i]);
    emit(name, "SEARCH_BRANCH_" + std::to_string(i), branch_json);
    const std::string hash = state_hash(canonical.agent->search.lastState());
    if (i == 0) {
      first_json = branch_json;
      first_hash = hash;
    } else {
      require(hash != first_hash, name + " candidates produced identical internal State hashes");
    }
  }
  const State branch_copy = canonical.agent->search.lastState();
  const std::string rng_after = rng_hash(canonical.agent->game);
  const std::string perturbed_json = search_step(perturbed, perturbed.root_id, choices[0]);
  emit(name, "PERTURBED_SEARCH_BRANCH", perturbed_json);
  const bool hidden_equal = first_json == perturbed_json;

  select_one(battle, live_choice);
  const std::string live_observation = observation_json(battle.state, battle.jsonBuilder);
  emit(name, "LIVE_OBSERVATION", live_observation);
  emit_meta(name, root_copy, branch_copy, rng_before, rng_after,
            static_cast<int>(choices.size()), hidden_equal);
}

std::unique_ptr<ApiData> build_turn_three_with_two_energy(
    const std::vector<int>& deck, int active_id, int energy_a, int energy_b,
    int required_play_card, uint32_t& seed) {
  return find_fixture(deck, active_id, 2,
      [&](ApiData& battle) {
        const int focal = battle.state.selectPlayer;
        int attach_a = choose_attach(battle.state, energy_a, AreaType::Active);
        if (attach_a < 0) return false;
        select_one(battle, attach_a);
        end_turn(battle);
        if (battle.state.selectPlayer == focal) return false;
        end_turn(battle);
        if (battle.state.selectPlayer != focal) return false;
        int attach_b = choose_attach(battle.state, energy_b, AreaType::Active);
        if (attach_b < 0) return false;
        select_one(battle, attach_b);
        return required_play_card == 0 ||
               !play_options_for_card(battle.state, required_play_card).empty();
      }, seed);
}

void test_bench_and_attach(const std::vector<int>& deck, int active_id) {
  uint32_t seed = 0;
  auto bench = find_fixture(deck, active_id, 0,
      [](ApiData& battle) { return basic_play_options(battle.state).size() >= 2; }, seed);
  std::vector<int> plays = basic_play_options(bench->state);
  plays.resize(2);
  run_branch_case("BENCH_PLACEMENT", *bench, deck, plays, plays[0]);

  auto attach = find_fixture(deck, active_id, 2,
      [](ApiData& battle) {
        auto options = attach_options(battle.state);
        if (options.size() < 2) return false;
        const int hand = battle.state.options[options[0]].param1;
        int same_source = 0;
        for (int index : options) same_source += battle.state.options[index].param1 == hand;
        return same_source >= 2;
      }, seed);
  std::vector<int> options = attach_options(attach->state);
  const int hand = attach->state.options[options[0]].param1;
  options.erase(std::remove_if(options.begin(), options.end(), [&](int index) {
    return attach->state.options[index].param1 != hand;
  }), options.end());
  options.resize(2);
  run_branch_case("ENERGY_ATTACH_TARGET", *attach, deck, options, options[0]);
}

void test_retreat_nodes(const std::vector<int>& deck, int active_id, int energy_a, int energy_b) {
  uint32_t seed = 0;
  auto battle = build_turn_three_with_two_energy(deck, active_id, energy_a, energy_b, 0, seed);
  select_one(*battle, find_option(battle->state, SelectOptionType::Retreat));
  require(battle->state.selectContext == SelectContext::DiscardEnergy,
          "Retreat did not reach DiscardEnergy");
  std::vector<int> payments(battle->state.options.size());
  std::iota(payments.begin(), payments.end(), 0);
  require(payments.size() >= 2, "Retreat payment lacks two energy choices");
  payments.resize(2);
  run_branch_case("RETREAT_ENERGY_PAYMENT", *battle, deck, payments, payments[0]);

  require(battle->state.selectContext == SelectContext::Switch,
          "Retreat payment did not reach Switch target");
  std::vector<int> targets(battle->state.options.size());
  std::iota(targets.begin(), targets.end(), 0);
  require(targets.size() >= 2, "Retreat Switch lacks two targets");
  targets.resize(2);
  run_branch_case("RETREAT_SWITCH_TARGET", *battle, deck, targets, targets[0]);
}

void test_boss_target(const std::vector<int>& deck, int active_id, int energy_a, int energy_b) {
  uint32_t seed = 0;
  auto battle = build_turn_three_with_two_energy(deck, active_id, energy_a, energy_b, 1182, seed);
  select_one(*battle, play_options_for_card(battle->state, 1182).at(0));
  require(battle->state.selectContext == SelectContext::Switch,
          "Boss did not reach Switch target");
  std::vector<int> targets(battle->state.options.size());
  std::iota(targets.begin(), targets.end(), 0);
  require(targets.size() >= 2, "Boss lacks two opponent targets");
  targets.resize(2);
  run_branch_case("BOSS_OPPONENT_TARGET", *battle, deck, targets, targets[0]);
}

void test_energy_switch_target(
    const std::vector<int>& deck, int active_id, int energy_a, int energy_b) {
  uint32_t seed = 0;
  auto battle = build_turn_three_with_two_energy(deck, active_id, energy_a, energy_b, 1116, seed);
  select_one(*battle, play_options_for_card(battle->state, 1116).at(0));
  require(battle->state.selectContext == SelectContext::SwitchEnergyCard,
          "Energy Switch did not request an attached Energy");
  select_one(*battle, 0);
  require(battle->state.selectType == SelectType::Card,
          std::string("Energy Switch destination is not Card: ") +
              SelectTypeStr[static_cast<int>(battle->state.selectType)] + "/" +
              SelectContextStr[static_cast<int>(battle->state.selectContext)]);
  std::vector<int> targets(battle->state.options.size());
  std::iota(targets.begin(), targets.end(), 0);
  require(targets.size() >= 2, "Energy Switch lacks two destination targets");
  targets.resize(2);
  run_branch_case("TRAINER_EFFECT_TARGET", *battle, deck, targets, targets[0]);
}

void test_hidden_draw_rejection(
    const std::vector<int>& deck, int active_id, int energy_a, int energy_b) {
  uint32_t seed = 0;
  auto battle = build_turn_three_with_two_energy(deck, active_id, energy_a, energy_b, 1181, seed);
  const int billy = play_options_for_card(battle->state, 1181).at(0);
  // End is a second legal Main selection, but only Billy is used to demonstrate
  // hidden-deck sensitivity of an otherwise same-turn, RNG-free SearchStep.
  std::vector<int> choices = {billy, find_option(battle->state, SelectOptionType::End)};
  const std::string root_observation = observation_json(battle->state, battle->jsonBuilder);
  SearchFixture canonical = begin_search(*battle, 0x503300u, false);
  SearchFixture perturbed = begin_search(*battle, 0x503300u, true);
  emit("HIDDEN_DRAW_COUNTEREXAMPLE", "ROOT_OBSERVATION", root_observation);
  emit_deck("HIDDEN_DRAW_COUNTEREXAMPLE", deck);
  const State root_copy = canonical.agent->search.lastState();
  const std::string rng_before = rng_hash(canonical.agent->game);
  const std::string branch = search_step(canonical, canonical.root_id, billy);
  const State branch_copy = canonical.agent->search.lastState();
  const std::string rng_after = rng_hash(canonical.agent->game);
  const std::string perturbed_branch = search_step(perturbed, perturbed.root_id, billy);
  emit("HIDDEN_DRAW_COUNTEREXAMPLE", "SEARCH_BRANCH_0", branch);
  emit("HIDDEN_DRAW_COUNTEREXAMPLE", "PERTURBED_SEARCH_BRANCH", perturbed_branch);
  require(branch != perturbed_branch, "Billy draw was unexpectedly hidden-invariant");
  emit_meta("HIDDEN_DRAW_COUNTEREXAMPLE", root_copy, branch_copy, rng_before, rng_after,
            static_cast<int>(choices.size()), false);
}

}  // namespace

int main() {
  try {
    GameInitialize();
    int energy_a = 0;
    int energy_b = 0;
    int active_id = 0;
    for (const auto& [id, master] : CardTable) {
      if (master.cardType == CardType::BasicEnergy) {
        if (energy_a == 0) energy_a = id;
        else if (energy_b == 0 && id != energy_a) energy_b = id;
      }
      if (active_id == 0 && master.cardType == CardType::Pokemon &&
          master.evolutionType == EvolutionType::Basic && master.retreatCost == 1 &&
          master.ability == nullptr && master.canSetup()) active_id = id;
    }
    require(energy_a != 0 && energy_b != 0 && active_id != 0, "fixture cards unavailable");
    const std::vector<int> deck = make_deck(active_id, energy_a, energy_b);
    require(deck.size() == DECK_SIZE, "fixture deck is not 60 cards");

    std::cout << "OFFICIAL_CPU_SEARCH_SAME_PERSPECTIVE_PROBE_V1\n";
    test_bench_and_attach(deck, active_id);
    test_retreat_nodes(deck, active_id, energy_a, energy_b);
    test_boss_target(deck, active_id, energy_a, energy_b);
    test_energy_switch_target(deck, active_id, energy_a, energy_b);
    test_hidden_draw_rejection(deck, active_id, energy_a, energy_b);
    std::cout << "RESULT PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "RESULT FAIL: " << error.what() << '\n';
    return 1;
  }
}
