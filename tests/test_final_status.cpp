// Black-box final-turn exchange with a controlled in-memory game setup.
#define main sevenwonders_engine_main
#include "../main.cpp"
#undef main

#include <filesystem>
#include <stdexcept>

struct GameTestAccess {
    static std::vector<Player*>& players(Game& game) { return game.player_list; }
    static void final_turn(Game& game) { game.turn = 20; game.era = 3; }
};

class CompletedOlympia : public Olympia_b {
public:
    CompletedOlympia() { stage = 3; }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void build_free(Player* player, const Card& card) {
    player->ReceiveCards({card});
    require(player->BuildStructure(card, {card}, true), "Test setup card failed");
}

int main() {
    try {
        std::filesystem::create_directories("io");
        Game game(0);
        game.Init();
        game.NewGame(3);
        auto& players = GameTestAccess::players(game);
        players[0]->SetWonder(new Babylon_b());
        players[0]->CanPlaySeventh();
        players[1]->SetWonder(new CompletedOlympia());
        players[2]->SetWonder(new Gizah_a());
        build_free(players[0], Card(CARD_ID::lumber_yard, "Lumber Yard", CARD_TYPE::materials,
                                  1, CARD_ID::none, std::vector<int>(8, 0), std::vector<int>(5, 1)));
        build_free(players[2], Card(CARD_ID::workers, "Workers Guild", CARD_TYPE::guild,
                                  3, CARD_ID::none, std::vector<int>(8, 0), std::vector<int>(5, 1)));
        players[0]->AddResource(RESOURCE::shields, 2);
        players[1]->AddResource(RESOURCE::shields, 1);
        Card altar = game.GetCardByName("Altar");
        for (Player* player : players) player->ReceiveCards({altar});
        GameTestAccess::final_turn(game);
        for (int seat = 0; seat < 3; ++seat) {
            json action = {{"turn", 20}, {"command", {{"subcommand", seat == 0 ? "build_structure" : "discard"},
                                                     {"argument", "Altar"}, {"extra", ""}}}};
            std::ofstream("io/player_" + std::to_string(seat + 1) + ".json") << action;
        }
        std::ofstream("io/ready.txt") << "ready\nready\nready\n";
        game.Loop();
        json status;
        std::ifstream("io/game_status.json") >> status;
        require(status["game"]["phase"] == "finished", "Missing final phase");
        require(status["game"]["turn"] == 21, "Missing final turn");
        require(status["players"]["0"]["points"]["civilian"] == 2, "Seventh card was not scored");
        require(status["players"]["0"]["points"]["military"] == 10, "Age-III battles missing or repeated");
        require(status["players"]["1"]["points"]["military"] == 4, "Incorrect final battle score");
        require(status["players"]["2"]["points"]["military"] == -2, "Final defeat tokens missing");
        require(status["players"]["1"]["points"]["guild"] == 1, "Guild copy missing from final JSON");
        require(status["last_actions"]["0"]["status"] == "applied", "Seventh action not acknowledged");
        require(status["last_actions"]["1"]["status"] == "skipped", "Ignored action reported as applied");
        require(status["players"]["1"]["resources"]["coins"] == 3, "Skipped discard changed coins");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
