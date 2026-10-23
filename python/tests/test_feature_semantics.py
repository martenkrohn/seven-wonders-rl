"""Feature regressions use game facts, never strategic reward preferences."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from sevenwonders_rl.data.cards import by_name_strict, load_card_db, get_chain_to_count
from sevenwonders_rl.features.obs_builder import build_obs, GLOBAL_DIM
from sevenwonders_rl.features.player_state import chain_free, science_state, discount_flags
from sevenwonders_rl.features.action_feature_schema import IDX, K
from sevenwonders_rl.utils.ppo_utils import flatten_global


def hand(player, names):
    player['cards_hand'] = names
    player['cards_hand_id'] = [by_name_strict(n)['id'] for n in names]
    player['cards_playable'] = names


@pytest.mark.component
def test_chain_requires_built_prerequisite_and_free_costs(status_age1_turn0):
    status = copy.deepcopy(status_age1_turn0)
    player = status['players']['0']
    player['cards_played'] = []
    hand(player, ['Dispensary', 'Apothecary', 'Forum'])
    row = build_obs(status, 0)['action_feats'][0]
    assert row[IDX['is_chain_free_now']] == 0  # playable does not mean chained
    player['cards_played'] = ['Apothecary']
    obs = build_obs(status, 0)
    row = obs['action_feats'][0]
    assert row[IDX['is_chain_free_now']] == 1
    assert row[IDX['est_trade_cost_left']] == row[IDX['est_trade_cost_right']] == 0
    assert row[IDX['card_cost_ore']] == row[IDX['card_cost_glass']] == 0
    assert row[IDX['coins_after_est_total_cost']] == player['resources']['coins']
    for name in ['East Trading Post', 'West Trading Post']:
        player['cards_played'] = [name]
        assert build_obs(status, 0)['action_feats'][2, IDX['is_chain_free_now']] == 1
    assert get_chain_to_count('apothecary') == 2
    assert obs['action_feats'][1, IDX['enables_chain_to_count']] == 2


@pytest.mark.component
def test_all_chain_relations_and_duplicate_exclusion():
    for card in load_card_db().values():
        prereqs = ([card['chain_from']] if card.get('chain_from') is not None else [])
        prereqs += [by_name_strict(n)['id'] for n in card.get('chain_from_any') or []]
        assert not chain_free(card, [])
        for prerequisite in prereqs:
            assert chain_free(card, [prerequisite])
            assert not chain_free(card, [prerequisite, card['id']])


@pytest.mark.component
def test_science_and_discounts_match_actor_critic_for_every_rotation(status_age1_turn0):
    status = copy.deepcopy(status_age1_turn0)
    cards = [['Workshop', 'Laboratory', 'Scientists Guild', 'East Trading Post'],
             ['Apothecary', 'West Trading Post'], ['Scriptorium', 'Marketplace']]
    for seat, names in enumerate(cards):
        player = status['players'][str(seat)]
        player['cards_played'] = names
        for symbol in ['gear', 'compass', 'tablet']:
            player['resources'][symbol] = 99  # must not trust stale engine counters
        player['wonder_id'] = 0
        player['wonder_stage'] = 0
    for seat in range(3):
        obs = build_obs(status, seat)
        assert len(flatten_global(obs)) == GLOBAL_DIM
        assert obs['action_feats'].shape == (28, K)
        action = np.flatnonzero(obs['action_mask'])[0]
        row = obs['action_feats'][action]
        for index, physical_seat in enumerate([seat, (seat-1)%3, (seat+1)%3]):
            science, jokers = science_state(status['players'][str(physical_seat)])
            np.testing.assert_array_equal(obs['global_table'][index, 2:5],
                                          [science[s] for s in ['gear', 'compass', 'tablet']])
            assert obs['science_jokers'][index] == jokers
            role = ['own', 'left', 'right'][index]
            for symbol, count in science.items():
                assert row[IDX[f'{role}_science_{symbol}']] == count
            assert row[IDX[f'{role}_science_jokers']] == jokers
        assert obs['science_jokers'].sum() == 1
    player = status['players']['0']
    player['wonder_id'] = 1  # Babylon A; current engine grants joker at stage 2
    player['wonder_stage'] = 2
    assert science_state(player) == ({'gear': 2, 'compass': 0, 'tablet': 0}, 2)
    player['wonder_id'] = 8
    player['wonder_stage'] = 2
    assert science_state(player)[1] == 1
    player['wonder_stage'] = 3
    assert science_state(player)[1] == 2
    player['wonder_id'] = 9  # Olympia B current-engine raw discounts
    player['wonder_stage'] = 1
    assert {'raw_cheap_east', 'raw_cheap_west'} <= discount_flags(player)


@pytest.mark.component
def test_discount_deltas_choice_production_and_free_hand_costs(status_age1_turn0):
    status = copy.deepcopy(status_age1_turn0)
    player = status['players']['0']
    player['cards_played'] = ['Marketplace']
    player['can_build_hand_free'] = True
    hand(player, ['East Trading Post', 'West Trading Post', 'Marketplace', 'Clay Pit', 'Forum'])
    obs = build_obs(status, 0)
    for i, flag in enumerate(['raw_cheap_east', 'raw_cheap_west', 'manuf_cheap']):
        assert obs['action_feats'][i, IDX[f'adds_discount_{flag}']] == 1
    assert obs['action_feats'][3, IDX['delta_prod_choice_raw']] == 1
    assert obs['action_feats'][4, IDX['delta_prod_choice_manuf']] == 1
    assert obs['action_feats'][17, IDX['coins_after_est_total_cost']] == player['resources']['coins']
    assert obs['action_feats'][17, IDX['card_cost_coins']] == 0
    assert obs['action_mask'][16] == 0  # cannot build Marketplace twice, even for free


@pytest.mark.component
def test_previous_feature_semantics_are_rejected(tmp_path):
    from sevenwonders_rl.utils.checkpoints import save_checkpoint, load_model, FEATURE_SEMANTICS_VERSION
    from sevenwonders_rl.utils.ppo_model import ActorCritic
    path = tmp_path / 'stale.pt'
    save_checkpoint(path, ActorCritic(GLOBAL_DIM, K), None, 0,
                    {'global_dim': GLOBAL_DIM, 'action_feat_dim': K, 'hidden': 128},
                    {'feature_semantics_version': FEATURE_SEMANTICS_VERSION - 1})
    with pytest.raises(ValueError, match='fresh training'):
        load_model(path)


PROBE = r'''
#define main sevenwonders_engine_main
#include "ENGINE_MAIN"
#undef main
#include <filesystem>
struct GameTestAccess {
    static auto& players(Game& g) { return g.player_list; }
    static auto& deck(Game& g, int age) { return g.deck[age]; }
};
json read_status() {
    json s; std::ifstream("io/game_status.json") >> s; return s;
}
void build(Game& g, Player* p, std::string name) {
    Card c = g.GetCardByName(name); p->ReceiveCards({c});
    if (!p->BuildStructure(c, {c}, true)) throw std::runtime_error("setup build failed");
}
int main(int argc, char** argv) {
    std::filesystem::create_directories("io");
    json output = json::array();
    for (int scenario = 0; scenario < 6; ++scenario) {
        Game g(0); g.Init(); g.NewGame(3);
        auto& players = GameTestAccess::players(g);
        for (auto* p : players) {
            p->SetWonder(new Gizah_a());
            auto r = p->GetResources(); for (auto& x:r) x.second=0;
            r[RESOURCE::coins]=20; r[RESOURCE::stone]=1; p->SetResources(r);
        }
        auto* p = players[0];
        std::string target;
        if (scenario < 2) {
            build(g,p,scenario==0 ? "East Trading Post" : "West Trading Post");
            build(g,players[scenario==0 ? 2 : 1],"Lumber Yard"); target="Stockade";
        } else if (scenario == 2) {
            build(g,p,"Marketplace"); build(g,players[2],"Glassworks"); target="Workshop";
        } else if (scenario == 3) {
            build(g,p,"Apothecary"); target="Dispensary";
        } else if (scenario == 4) {
            build(g,p,"West Trading Post"); target="Forum";
        } else {
            build(g,p,"Workshop"); build(g,p,"Apothecary");
            build(g,p,"Scriptorium"); build(g,p,"Scientists Guild"); target="Altar";
        }
        auto c = g.GetCardByName(target); p->ReceiveCards({c});
        g.WriteGameStatus(); json before=read_status();
        if (!p->BuildStructure(c,{c},false)) throw std::runtime_error("tested build failed");
        g.WriteGameStatus();
        output.push_back({{"scenario",scenario},{"before",before},{"after",read_status()}});
    }
    std::ofstream(argv[1]) << output;
}
'''


@pytest.mark.integration
def test_features_match_real_engine_free_chains_trade_direction_and_science(require_engine, tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler required for controlled engine probe')
    root = Path(__file__).parents[2]
    source = tmp_path / 'probe.cpp'
    source.write_text(PROBE.replace('ENGINE_MAIN', str(root / 'main.cpp')))
    binary = tmp_path / 'probe'
    # Compile a temporary probe against unchanged engine sources. No C++ repo mutation.
    subprocess.run([compiler, '-std=c++17', '-O0', '-I', str(root/'include'), str(source),
                    *[str(root/'src'/n) for n in ['player.cpp','card.cpp','wonder.cpp','filer.cpp']],
                    '-o', str(binary)], check=True, capture_output=True, timeout=120)
    output = tmp_path / 'probe.json'
    subprocess.run([str(binary), str(output)], cwd=tmp_path, check=True, capture_output=True, timeout=15)
    for case in json.loads(output.read_text()):
        before, after = case['before'], case['after']
        obs = build_obs(before, 0)
        assert obs['action_mask'][0] == 1
        features = obs['action_feats'][0]
        paid = before['players']['0']['resources']['coins'] - after['players']['0']['resources']['coins']
        assert features[IDX['coins_after_est_total_cost']] == after['players']['0']['resources']['coins']
        assert features[IDX['est_trade_cost_left']] == after['players']['2']['resources']['coins'] - before['players']['2']['resources']['coins']
        assert features[IDX['est_trade_cost_right']] == after['players']['1']['resources']['coins'] - before['players']['1']['resources']['coins']
        if case['scenario'] < 3:
            assert paid == 1
        if case['scenario'] in (3, 4):
            assert paid == 0 and features[IDX['is_chain_free_now']] == 1
        if case['scenario'] == 5:
            assert obs['global_table'][0, 2:5].tolist() == [1, 1, 1]
            assert obs['science_jokers'][0] == 1
            assert before['players']['0']['points']['scientific'] == 13
