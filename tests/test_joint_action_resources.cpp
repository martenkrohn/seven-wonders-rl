// Regression for legal choice-resource actions rejected after another player's action.
#define main sevenwonders_engine_main
#include "../main.cpp"
#undef main
#include <filesystem>

struct GameTestAccess {
    static std::vector<Player*>& players(Game& game) { return game.player_list; }
    static void final_turn(Game& game) { game.turn = 20; game.era = 3; }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        std::filesystem::create_directories("io");
        Game game(0);
        game.Init();
        game.NewGame(3);
        auto& players = GameTestAccess::players(game);
        players[0]->SetWonder(new Alexandria_b());
        players[1]->SetWonder(new Gizah_a());
        players[2]->SetWonder(new Ephesos_a());
        // Clear random starting resources, retain only the intended production.
        for (Player* player : players) {
            auto resources = player->GetResources();
            for (auto& entry : resources) entry.second = 0;
            resources[RESOURCE::coins] = 20;
            player->SetResources(resources);
            player->CanPlaySeventh();
        }
        Card clay_pit = game.GetCardByName("Clay Pit");
        players[0]->ReceiveCards({clay_pit});
        require(players[0]->BuildStructure(clay_pit, {clay_pit}, true), "Clay Pit setup failed");
        players[2]->AddResource(RESOURCE::clay, 1);
        Card altar = game.GetCardByName("Altar");
        Card forum = game.GetCardByName("Forum");
        players[0]->ReceiveCards({altar});
        players[1]->ReceiveCards({altar});
        players[2]->ReceiveCards({forum});
        require(players[0]->CanBuildWonder(), "Alexandria stage must be affordable");
        require(!players[2]->GetPlayableCards().empty(), "Forum must be legal before the joint action");
        GameTestAccess::final_turn(game);
        for (int seat = 0; seat < 3; ++seat) {
            std::string command = seat == 0 ? "build_wonder" : seat == 1 ? "discard" : "build_structure";
            json action = {{"turn", 20}, {"command", {{"subcommand", command},
                {"argument", seat == 2 ? "Forum" : "Altar"}, {"extra", ""}}}};
            std::ofstream("io/player_" + std::to_string(seat + 1) + ".json") << action;
        }
        std::ofstream("io/ready.txt") << "ready\nready\nready\n";
        game.Loop();
        json status;
        std::ifstream("io/game_status.json") >> status;
        require(status["players"]["0"]["wonder_stage"] == 1, "First action did not build Alexandria stage");
        require(status["last_actions"]["2"]["status"] == "applied",
                "Earlier player's choice use must not invalidate the next player's legal Forum");
        require(players[2]->HasPlayedCard(forum), "Forum was not built");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
