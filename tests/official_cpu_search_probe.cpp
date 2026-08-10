// Standalone diagnostic: compile the unmodified official CPU Export.cpp in this
// translation unit so the public C ABI and its internal state can be audited together.
#include "Export.cpp"

#include <iomanip>
#include <iostream>
#include <memory>
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

std::vector<int> card_ids(const State& state, const CardList& cards) {
  std::vector<int> result;
  result.reserve(cards.size());
  for (CardRef ref : cards) {
    result.push_back(state.getCardId(ref));
  }
  return result;
}

template <typename List>
std::string ids_json(const State& state, const List& cards) {
  std::ostringstream out;
  out << '[';
  for (int i = 0; i < cards.size(); ++i) {
    if (i) out << ',';
    out << state.getCardId(cards[i]);
  }
  out << ']';
  return out.str();
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

void dump_state(const char* label, const State& state) {
  const int actor = state.selectPlayer;
  const PlayerState& player = state.players.at(actor);
  std::cout << label
            << " {hash:" << state_hash(state)
            << ", actor:" << actor
            << ", turn:" << state.turn
            << ", selectType:" << SelectTypeStr[static_cast<int>(state.selectType)]
            << ", context:" << SelectContextStr[static_cast<int>(state.selectContext)]
            << ", legal:" << options_json(state)
            << ", hand:" << ids_json(state, player.hand)
            << ", active:" << ids_json(state, player.active)
            << ", bench:" << ids_json(state, player.bench)
            << ", deckCount:" << player.deck.size()
            << ", prizeCount:" << player.prize.size()
            << ", energyAttached:" << state.energyPlayed
            << ", retreated:" << state.retreated
            << "}\n";
}

int find_option(const State& state, SelectOptionType type) {
  for (int i = 0; i < state.options.size(); ++i) {
    if (state.options[i].type == type) return i;
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

std::vector<int> make_deck(int energy_id, const std::vector<int>& basic_ids) {
  std::vector<int> deck;
  for (int id : basic_ids) {
    for (int copy = 0; copy < 4; ++copy) deck.push_back(id);
  }
  while (deck.size() < DECK_SIZE) deck.push_back(energy_id);
  return deck;
}

struct CardFixture {
  int energy_id = 0;
  std::vector<int> basic_ids;
};

CardFixture find_cards() {
  CardFixture fixture;
  std::unordered_set<std::u8string> names;
  for (const auto& [id, master] : CardTable) {
    if (fixture.energy_id == 0 && master.cardType == CardType::BasicEnergy) {
      fixture.energy_id = id;
    }
    if (master.cardType == CardType::Pokemon &&
        master.evolutionType == EvolutionType::Basic &&
        master.retreatCost == 1 && master.ability == nullptr &&
        master.canSetup() && names.insert(master.name).second) {
      fixture.basic_ids.push_back(id);
    }
  }
  if (fixture.energy_id == 0 || fixture.basic_ids.size() < 4) {
    throw std::runtime_error("could not locate simple official cards");
  }
  fixture.basic_ids.resize(4);
  return fixture;
}

std::vector<int> setup_selection(const State& state) {
  if (state.selectContext == SelectContext::SetupBenchPokemon) return {};
  if (state.selectContext == SelectContext::SetupActivePokemon) {
    for (int i = 0; i < state.options.size(); ++i) {
      const CardPosition position = state.options[i].getCardPosition();
      const CardRef ref = state.players.at(position.playerIndex).hand.at(position.areaIndex);
      if (state.getCard(ref).getMaster().retreatCost == 1) return {i};
    }
  }
  if (state.selectContext == SelectContext::IsFirst) return {0};
  if (state.selectMin == 0) return {};
  return {0};
}

std::unique_ptr<ApiData> build_main_state(const std::vector<int>& deck, uint32_t& seed) {
  for (seed = 1; seed <= 10000; ++seed) {
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
      std::vector<int> selected = setup_selection(battle->state);
      int error = ApiSelect(battle.get(), selected.data(), static_cast<int>(selected.size()));
      if (error != 0) throw std::runtime_error("setup selection failed: " + std::to_string(error));
    }
    State& state = battle->state;
    if (state.selectContext == SelectContext::Main &&
        find_option(state, SelectOptionType::Play) >= 0 &&
        find_attach_to_active(state) >= 0 &&
        state.players.at(state.selectPlayer).bench.empty()) {
      return battle;
    }
  }
  throw std::runtime_error("no deterministic seed produced the required main state");
}

int* ptr_or_dummy(std::vector<int>& values) {
  static int dummy = 0;
  return values.empty() ? &dummy : values.data();
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

State& search_step_checked(ApiData* agent, int source_id, int option_index) {
  int selected[] = {option_index};
  const char8_t* json = SearchStep(agent, source_id, selected, 1);
  require(json != nullptr, "SearchStep returned null JSON");
  return const_cast<State&>(agent->search.lastState());
}

}  // namespace

int main() {
  try {
    GameInitialize();
    const CardFixture cards = find_cards();
    const std::vector<int> deck = make_deck(cards.energy_id, cards.basic_ids);
    uint32_t battle_seed = 0;
    std::unique_ptr<ApiData> battle = build_main_state(deck, battle_seed);
    State& live = battle->state;

    SerialData serial = GetBattleData(battle.get());
    require(serial.data != nullptr && serial.count > 0, "GetBattleData serialization failed");
    const std::string live_state_before = state_hash(live);
    const std::string live_rng_before = rng_hash(battle->game);
    const int me = live.selectPlayer;
    const int enemy = 1 - me;
    std::vector<int> my_deck = card_ids(live, live.players[me].deck);
    std::vector<int> my_prize = card_ids(live, live.players[me].prize);
    std::vector<int> enemy_deck = card_ids(live, live.players[enemy].deck);
    std::vector<int> enemy_prize = card_ids(live, live.players[enemy].prize);
    std::vector<int> enemy_hand = card_ids(live, live.players[enemy].hand);
    std::vector<int> enemy_active;

    ApiData* agent = AgentStart();
    require(agent != nullptr, "AgentStart failed");
    agent->game.rng = std::mt19937(0x5EA2C1u);
    const char8_t* begin_json = SearchBegin(
        agent, serial.data, serial.count,
        ptr_or_dummy(my_deck), ptr_or_dummy(my_prize),
        ptr_or_dummy(enemy_deck), ptr_or_dummy(enemy_prize),
        ptr_or_dummy(enemy_hand), ptr_or_dummy(enemy_active), 0);
    require(begin_json != nullptr && agent->search.lastSearchId() >= 0, "SearchBegin failed");

    const int root_id = agent->search.lastSearchId();
    const State& root = agent->search.lastState();
    const std::string original_hash = state_hash(agent->state);
    const std::string root_hash = state_hash(root);
    const std::string deterministic_rng_before = rng_hash(agent->game);

    std::cout << "OFFICIAL_CPU_SEARCH_PROBE_V1\n";
    std::cout << "fixture {battleSeed:" << battle_seed
              << ", energyId:" << cards.energy_id
              << ", actor:" << static_cast<int>(root.selectPlayer) << "}\n";
    std::cout << "identity {original:" << static_cast<const void*>(&agent->state)
              << ", root:" << static_cast<const void*>(&root)
              << ", sameObject:" << (&agent->state == &root)
              << ", sharedGame:" << (agent->state.game == root.game) << "}\n";
    dump_state("ROOT_S0", root);

    const int play_index = find_option(root, SelectOptionType::Play);
    require(play_index >= 0, "missing Play option");
    State& after_play = search_step_checked(agent, root_id, play_index);
    const int play_id = agent->search.lastSearchId();
    std::cout << "TEST_A action {index:" << play_index << ", type:Play}\n";
    dump_state("TEST_A_S1", after_play);
    require(after_play.players[me].bench.size() == root.players[me].bench.size() + 1,
            "Play did not add exactly one Benched Pokemon");
    require(after_play.selectContext == SelectContext::Main,
            "Play did not auto-resolve back to Main");
    require(state_hash(root) == root_hash, "Play mutated its source search state");

    const int attach_index = find_attach_to_active(after_play);
    require(attach_index >= 0, "missing Attach-to-Active option");
    const int energies_before = after_play.players[me].energy.size();
    const std::string play_hash = state_hash(after_play);
    State& after_attach = search_step_checked(agent, play_id, attach_index);
    const int attach_id = agent->search.lastSearchId();
    std::cout << "TEST_B action {index:" << attach_index << ", type:Attach, target:Active}\n";
    dump_state("TEST_B_S1", after_attach);
    const int energies_after = after_attach.players[me].energy.size();
    require(energies_after == energies_before + 1, "Attach did not attach exactly one card");
    require(after_attach.energyPlayed, "Attach did not consume manual attachment");
    require(after_attach.selectContext == SelectContext::Main,
            "Attach did not auto-resolve back to Main");
    require(state_hash(after_play) == play_hash, "Attach mutated its source search state");

    const int retreat_index = find_option(after_attach, SelectOptionType::Retreat);
    require(retreat_index >= 0, "missing Retreat option");
    const int old_active_id = after_attach.getCardId(after_attach.players[me].getActive());
    const std::string attach_hash = state_hash(after_attach);
    State& retreat_energy = search_step_checked(agent, attach_id, retreat_index);
    const int retreat_energy_id = agent->search.lastSearchId();
    std::cout << "TEST_C1 action {index:" << retreat_index << ", type:Retreat}\n";
    dump_state("TEST_C_ENERGY_DECISION", retreat_energy);
    require(retreat_energy.selectContext == SelectContext::DiscardEnergy,
            "Retreat did not stop at the required Energy payment decision");
    require(state_hash(after_attach) == attach_hash, "Retreat mutated its source search state");
    require(!retreat_energy.options.empty(), "Retreat Energy decision has no option");

    State& retreat_target = search_step_checked(agent, retreat_energy_id, 0);
    const int retreat_target_id = agent->search.lastSearchId();
    std::cout << "TEST_C2 action {index:0, type:Energy, purpose:PayRetreatCost}\n";
    dump_state("TEST_C_SWITCH_DECISION", retreat_target);
    require(retreat_target.selectContext == SelectContext::Switch,
            "Retreat Energy payment did not stop at the switch decision");
    require(retreat_target.players[me].energy.size() == energies_after - 1,
            "Retreat Energy payment did not discard one attached Energy");
    require(!retreat_target.options.empty(), "Retreat switch decision has no target");

    State& after_retreat = search_step_checked(agent, retreat_target_id, 0);
    std::cout << "TEST_C3 action {index:0, type:Card, target:Bench}\n";
    dump_state("TEST_C_S1", after_retreat);
    require(after_retreat.getCardId(after_retreat.players[me].getActive()) != old_active_id,
            "Retreat target choice did not switch Active Pokemon");
    require(after_retreat.retreated, "Retreat flag was not retained");
    require(after_retreat.selectContext == SelectContext::Main,
            "Retreat target did not auto-resolve back to Main");

    const std::string deterministic_rng_after = rng_hash(agent->game);
    std::cout << "mutation_after_deterministic_steps {originalBefore:" << original_hash
              << ", originalAfter:" << state_hash(agent->state)
              << ", rootBefore:" << root_hash
              << ", rootAfter:" << state_hash(root)
              << ", rngBefore:" << deterministic_rng_before
              << ", rngAfter:" << deterministic_rng_after << "}\n";
    require(state_hash(agent->state) == original_hash,
            "deterministic SearchStep polluted reconstructed original State");
    require(state_hash(root) == root_hash,
            "deterministic SearchStep polluted root search State");
    require(deterministic_rng_before == deterministic_rng_after,
            "deterministic actions unexpectedly consumed RNG");

    const std::string shuffle_rng_before = rng_hash(agent->game);
    const std::string root_before_shuffle = state_hash(root);
    SearchInfo shuffled = agent->search.shuffle(root_id, me);
    require(shuffled.errorCode == 0, "Search::shuffle failed");
    const std::string shuffle_rng_after = rng_hash(agent->game);
    std::cout << "rng_shared_state_probe {rootGame:" << static_cast<const void*>(root.game)
              << ", agentGame:" << static_cast<const void*>(&agent->game)
              << ", sameGame:" << (root.game == &agent->game)
              << ", rngBefore:" << shuffle_rng_before
              << ", rngAfter:" << shuffle_rng_after
              << ", sourceRootBefore:" << root_before_shuffle
              << ", sourceRootAfter:" << state_hash(root) << "}\n";
    require(shuffle_rng_before != shuffle_rng_after,
            "Search::shuffle did not advance the shared Game RNG");
    require(state_hash(root) == root_before_shuffle,
            "Search::shuffle mutated its source State instead of its clone");

    std::cout << "live_battle_isolation {liveGame:"
              << static_cast<const void*>(&battle->game)
              << ", searchGame:" << static_cast<const void*>(&agent->game)
              << ", sameGame:" << (&battle->game == &agent->game)
              << ", stateBefore:" << live_state_before
              << ", stateAfter:" << state_hash(live)
              << ", rngBefore:" << live_rng_before
              << ", rngAfter:" << rng_hash(battle->game) << "}\n";
    require(state_hash(live) == live_state_before,
            "search calls polluted the live battle State");
    require(rng_hash(battle->game) == live_rng_before,
            "search calls polluted the live battle RNG");

    std::cout << "RESULT PASS\n";
    SearchEnd(agent);
    BattleFinish(agent);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "RESULT FAIL: " << error.what() << '\n';
    return 1;
  }
}
