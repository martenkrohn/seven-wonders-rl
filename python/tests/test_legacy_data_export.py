"""Source-backed regressions for the legacy exporters, never BGA target data."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "engine"
DATA = ROOT / "python/src/sevenwonders_rl/data"
sys.path.insert(0, str(ROOT / "python/scripts"))
try:
    import extract_card_db_from_cpp as cards
    import extract_wonder_db_from_cpp as wonders
finally:
    sys.path.pop(0)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module", autouse=True)
def historical_source(legacy_runtime):
    global ENGINE
    original=ENGINE;ENGINE=legacy_runtime
    yield
    ENGINE=original


def card_declarations():
    identities, _ = cards.parse_enum(ENGINE / "include/card.h", "CARD_ID")
    types, _ = cards.parse_enum(ENGINE / "include/card.h", "CARD_TYPE")
    return cards.parse_cards_from_main(ENGINE / "main.cpp", identities, types)


@pytest.mark.parametrize("module", [cards, wonders])
@pytest.mark.parametrize("declaration", ["enum{", "enum {", "enum ID{"])
def test_enum_syntax_never_becomes_an_identity(module, declaration, tmp_path):
    header = tmp_path / "ids.h"
    header.write_text("namespace EXAMPLE {\n" + declaration +
                      "\n// Initial ID must stay zero.\nfirst,\nsecond,\nlast = 99,\n};\n}\n")
    forward, reverse = module.parse_enum(header, "EXAMPLE")
    assert forward == {"first": 0, "second": 1, "last": 99}
    assert reverse == {0: "first", 1: "second", 99: "last"}


def test_wonder_zero_and_all_stage_plans_match_maintained_legacy_data():
    identities, _ = wonders.parse_enum(ENGINE / "include/wonder.h", "WONDER_ID")
    _, resources = wonders.parse_enum(ENGINE / "include/resources.h", "RESOURCE")
    _, effects = wonders.parse_enum(ENGINE / "include/wonder.h", "EFFECT")
    parsed = []
    for name, constructor, addstage in wonders.parse_wonder_classes(ENGINE / "src/wonder.cpp"):
        base = wonders.parse_wonder_constructor(constructor, identities, resources, effects)
        stages = wonders.parse_addstage_method(addstage, base["stage_0_cost"], base["effects_raw"])
        parsed.append((name, base, stages))
    actual = wonders.build_wonder_db(parsed)
    assert set(actual) == {str(i) for i in range(14)}
    assert actual["0"]["id_name"] == "gizah_a"
    assert actual == json.loads((DATA / "wonder_db.json").read_text())


@pytest.mark.parametrize("symbol,resource", [("sawmill", "wood"), ("quarry", "stone"),
                                             ("brickyard", "clay"), ("foundry", "ore")])
def test_double_production_keeps_quantity_from_build_structure(symbol, resource):
    identities, _ = cards.parse_enum(ENGINE / "include/card.h", "CARD_ID")
    resources = cards.parse_resource_enum(ENGINE / "include/resources.h")
    effects = cards.parse_build_structure_effects(ENGINE / "src/player.cpp", identities, resources)
    assert effects[identities[symbol]]["produces_fixed"] == {resource: 2}


@pytest.mark.parametrize("identity", [15, 16, 17])
def test_grey_variants_keep_both_ages_and_all_five_player_counts(identity):
    merged = cards.merge_card_definitions(card_declarations())[identity]
    assert merged["eras"] == [1, 2]
    assert merged["amount_per_player_by_era"] == {
        "1": {"3": 1, "4": 1, "5": 1, "6": 2, "7": 2},
        "2": {"3": 1, "4": 1, "5": 2, "6": 2, "7": 2},
    }


@pytest.mark.parametrize("field,value", [("cost_vec", [1, 0, 0, 0, 0, 0, 0, 0]),
                                        ("free_with_raw", 18)])
def test_different_age_cost_or_chain_cannot_be_silently_overwritten(field, value):
    rows = [copy.deepcopy(r) for r in card_declarations() if r["id"] == 15]
    rows[1][field] = value
    with pytest.raises(ValueError, match="[Cc]onflict|mismatch"):
        cards.merge_card_definitions(rows)


def test_duplicate_same_age_cannot_be_silently_overwritten():
    row = card_declarations()[0]
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        cards.merge_card_definitions([row, copy.deepcopy(row)])


def test_every_exported_age_has_cpp_cost_chain_and_quantity():
    data = json.loads((DATA / "card_db.json").read_text())
    resource_names = ["wood", "ore", "clay", "stone", "loom", "glass", "papyrus", "coins"]
    declarations = card_declarations()
    assert len(declarations) == 78
    assert len(data) == 75
    for row in declarations:
        actual = data[str(row["id"])]
        assert row["era"] in actual["eras"]
        assert actual["cost"] == dict(zip(resource_names, row["cost_vec"]))
        assert actual["amount_per_player_by_era"][str(row["era"])] == {
            str(n): row["amount_vec"][n - 3] for n in range(3, 8)
        }
        if row["free_with_raw"] == -1:
            assert actual["chain_from"] is None
            assert actual["chain_from_any"] == ["east_trading_post", "west_trading_post"]
        else:
            assert actual["chain_from"] == row["free_with_raw"]


@pytest.mark.component
@pytest.mark.parametrize("identity,resource", [(11, "wood"), (12, "stone"), (13, "clay"), (14, "ore")])
def test_existing_production_consumer_reads_two_without_feature_changes(identity, resource):
    from sevenwonders_rl.features.action_features import _get_card_production

    assert _get_card_production(identity) == {resource: 2}


def export(script, *arguments, cwd):
    if '--engine-root' not in arguments:
        arguments=(*arguments,'--engine-root',ENGINE)
    return subprocess.run([sys.executable, str(ROOT / 'python/scripts' / script), *map(str, arguments)],
                          cwd=cwd, capture_output=True, text=True, timeout=15)


def test_cli_exports_twice_byte_identically_from_unrelated_directory(tmp_path):
    for destination in (tmp_path / 'first', tmp_path / 'second'):
        for script in ('extract_card_db_from_cpp.py', 'extract_wonder_db_from_cpp.py'):
            result = export(script, '--output-dir', destination, '--engine-root', ENGINE, cwd=tmp_path)
            assert result.returncode == 0, result.stdout + result.stderr
    for name in ('card_db.json', 'wonder_db.json', 'mechanics.json'):
        first = (tmp_path / 'first' / name).read_bytes()
        assert first == (tmp_path / 'second' / name).read_bytes()
        assert json.loads(first) == json.loads((DATA / name).read_text())


@pytest.mark.parametrize('script', ['extract_card_db_from_cpp.py', 'extract_wonder_db_from_cpp.py'])
def test_cli_check_never_writes_and_returns_failure_for_missing_or_changed_data(script, tmp_path):
    destination = tmp_path / 'missing'
    result = export(script, '--output-dir', destination, '--check', cwd=tmp_path)
    assert result.returncode == 1, result.stderr
    assert not destination.exists()
    result = export(script, '--output-dir', destination, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    result = export(script, '--output-dir', destination, '--check', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    snapshots = {p.name: p.read_bytes() for p in destination.iterdir()}
    for name in snapshots:
        with (destination / name).open('w') as output:
            json.dump({}, output)
    changed = {p.name: p.read_bytes() for p in destination.iterdir()}
    result = export(script, '--output-dir', destination, '--check', cwd=tmp_path)
    assert result.returncode == 1, result.stderr
    assert changed == {p.name: p.read_bytes() for p in destination.iterdir()}


@pytest.mark.parametrize('script', ['extract_card_db_from_cpp.py', 'extract_wonder_db_from_cpp.py'])
def test_cli_rejects_a_nonlegacy_ruleset_before_creating_output(script, tmp_path):
    engine = tmp_path / 'engine'
    engine.mkdir()
    (engine / 'engine-contract.json').write_text(json.dumps({
        'ruleset': 'bga-second-edition-development', 'second_edition_implemented': False}))
    result = export(script, '--engine-root', engine, '--output-dir', tmp_path / 'output', cwd=tmp_path)
    assert result.returncode == 2
    assert 'require the legacy engine contract' in result.stderr
    assert not (tmp_path / 'output').exists()
