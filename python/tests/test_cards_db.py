import json
from pathlib import Path
import pytest

pytestmark = pytest.mark.unit

ALLOWED_COLORS = {"brown", "grey", "blue", "yellow", "red", "green", "purple"}


def _load_json_from_pkg_or_src(rel: Path) -> dict:
    # Preferred: installed/editable package path
    try:
        import sevenwonders_rl  # type: ignore
        base = Path(sevenwonders_rl.__file__).resolve().parent
        p = base / rel
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass

    # Fallback: source tree under python/src/...
    p2 = Path(__file__).resolve().parents[1] / "src" / "sevenwonders_rl" / rel
    if p2.exists():
        return json.loads(p2.read_text())

    raise FileNotFoundError(f"Could not locate {rel} (expected under sevenwonders_rl/).")


def load_card_db() -> dict:
    return _load_json_from_pkg_or_src(Path("data") / "card_db.json")


def extract_cards(db: object) -> list[dict]:
    """
    Return a list[card_dict] regardless of whether card_db.json is:
      - a list of cards
      - {"cards": [...]}
      - {"by_id": {...}}
      - { "<id>": {...}, ... }
    """
    if isinstance(db, list):
        assert all(isinstance(x, dict) for x in db), "card_db.json list must contain dict entries"
        return db

    assert isinstance(db, dict), f"card_db.json must be dict or list, got {type(db).__name__}"

    if "cards" in db and isinstance(db["cards"], list):
        cards = db["cards"]
        assert all(isinstance(x, dict) for x in cards), "card_db['cards'] must contain dict entries"
        return cards

    if "by_id" in db and isinstance(db["by_id"], dict):
        cards = list(db["by_id"].values())
        assert all(isinstance(x, dict) for x in cards), "card_db['by_id'] must contain dict values"
        return cards

    # dict-of-cards fallback (string/int keys)
    if all(isinstance(v, dict) for v in db.values()):
        return list(db.values())

    raise AssertionError(f"Unrecognized card_db.json structure. Top-level keys={sorted(db.keys())}")


def _get_first(d: dict, keys: list[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _leaf_strings(obj) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_leaf_strings(k))
            out.extend(_leaf_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_leaf_strings(v))
    elif isinstance(obj, str):
        out.append(obj.lower())
    return out


def _has_list_containing(obj, required: set[str]) -> bool:
    """
    Heuristic: return True if anywhere inside obj there is a list of strings that contains all required strings.
    """
    req = {x.lower() for x in required}

    def rec(x) -> bool:
        if isinstance(x, list):
            strs = {s.lower() for s in x if isinstance(s, str)}
            if req.issubset(strs):
                return True
            return any(rec(v) for v in x)
        if isinstance(x, dict):
            return any(rec(v) for v in x.values())
        return False

    return rec(obj)


def find_card(cards: list[dict], name: str) -> dict:
    for c in cards:
        n = _get_first(c, ["name", "card_name", "title"])
        if isinstance(n, str) and n == name:
            return c
    raise AssertionError(f"Card not found by exact name: {name}. (Check name casing/spelling in DB.)")


def test_card_db_loads_and_has_cards():
    db = load_card_db()
    cards = extract_cards(db)
    assert cards, "card_db.json contained no cards"


def test_card_db_schema_minimum_fields():
    db = load_card_db()
    cards = extract_cards(db)

    for i, c in enumerate(cards):
        ctx = f"card[{i}] name={_get_first(c, ['name','card_name','title'])!r}"

        name = _get_first(c, ["name", "card_name", "title"])
        assert isinstance(name, str) and name, f"{ctx}: missing/invalid name field"

        cid = _get_first(c, ["id", "card_id"])
        assert isinstance(cid, int), f"{ctx}: missing/invalid id field (expected int), got {cid!r}"

        # Era can be encoded as single int "era", list "eras", or not present at all
        era = _get_first(c, ["era", "age", "eras"])
        if era is not None:
            if isinstance(era, list):
                assert all(isinstance(e, int) and e in (1, 2, 3) for e in era), f"{ctx}: invalid eras list: {era!r}"
            else:
                assert isinstance(era, int) and era in (1, 2, 3), f"{ctx}: invalid era field: {era!r}"

        color = _get_first(c, ["color", "card_color", "group"])
        assert isinstance(color, str), f"{ctx}: missing/invalid color field"
        assert color.lower() in ALLOWED_COLORS, f"{ctx}: unexpected color={color!r}"

        # cost/effects are used downstream; schema can vary but should exist.
        has_cost = any(k in c for k in ("cost", "cost_coins", "cost_resources", "coins"))
        assert has_cost, f"{ctx}: missing cost-related fields. keys={sorted(c.keys())}"

        has_effects = any(k in c for k in ("effects", "effect", "vp", "science", "shields", "produces", "type"))
        assert has_effects, f"{ctx}: missing effects-related fields. keys={sorted(c.keys())}"


def test_canonical_science_cards_exist_and_encode_symbol():
    db = load_card_db()
    cards = extract_cards(db)

    # These are stable Age I science cards in 7 Wonders:
    workshop = find_card(cards, "Workshop")
    apoth = find_card(cards, "Apothecary")
    scriptorium = find_card(cards, "Scriptorium")

    ws = _leaf_strings(workshop)
    ap = _leaf_strings(apoth)
    sc = _leaf_strings(scriptorium)

    assert any("gear" in s for s in ws), f"Workshop does not appear to encode science symbol 'gear'. keys={sorted(workshop.keys())}"
    assert any("compass" in s for s in ap), f"Apothecary does not appear to encode science symbol 'compass'. keys={sorted(apoth.keys())}"
    assert any("tablet" in s for s in sc), f"Scriptorium does not appear to encode science symbol 'tablet'. keys={sorted(scriptorium.keys())}"


def test_tree_farm_is_choice_between_wood_and_clay():
    db = load_card_db()
    cards = extract_cards(db)
    tree_farm = find_card(cards, "Tree Farm")

    leaves = _leaf_strings(tree_farm)
    assert any("wood" in s for s in leaves), "Tree Farm does not mention wood anywhere"
    assert any("clay" in s for s in leaves), "Tree Farm does not mention clay anywhere"

    # Stronger heuristic: a list somewhere contains BOTH options => indicates "choice" not "both"
    assert _has_list_containing(tree_farm, {"wood", "clay"}), (
        "Tree Farm did not contain an obvious choice-list with both wood and clay. "
        "If your schema encodes choices differently, adjust _has_list_containing() to match it."
    )


def test_forum_is_yellow_and_mentions_all_three_manufactured_goods():
    db = load_card_db()
    cards = extract_cards(db)
    forum = find_card(cards, "Forum")

    color = _get_first(forum, ["color", "card_color", "group"])
    assert isinstance(color, str) and color.lower() == "yellow", f"Forum color expected yellow, got {color!r}"

    leaves = _leaf_strings(forum)
    for res in ("loom", "glass", "papyrus"):
        assert any(res in s for s in leaves), f"Forum does not mention {res} anywhere (expected manuf choice encoding)"
