import json
from pathlib import Path
import pytest

pytestmark = pytest.mark.unit


def _load_json_from_pkg_or_src(rel: Path) -> dict:
    try:
        import sevenwonders_rl  # type: ignore
        base = Path(sevenwonders_rl.__file__).resolve().parent
        p = base / rel
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass

    p2 = Path(__file__).resolve().parents[1] / "src" / "sevenwonders_rl" / rel
    if p2.exists():
        return json.loads(p2.read_text())

    raise FileNotFoundError(f"Could not locate {rel} (expected under sevenwonders_rl/).")


def load_wonder_db() -> dict:
    return _load_json_from_pkg_or_src(Path("data") / "wonder_db.json")


def extract_wonders(db: object) -> list[dict]:
    if isinstance(db, list):
        assert all(isinstance(x, dict) for x in db), "wonder_db.json list must contain dict entries"
        return db

    assert isinstance(db, dict), f"wonder_db.json must be dict or list, got {type(db).__name__}"

    if "wonders" in db and isinstance(db["wonders"], list):
        w = db["wonders"]
        assert all(isinstance(x, dict) for x in w), "wonder_db['wonders'] must contain dict entries"
        return w

    if "by_id" in db and isinstance(db["by_id"], dict):
        w = list(db["by_id"].values())
        assert all(isinstance(x, dict) for x in w), "wonder_db['by_id'] must contain dict values"
        return w

    if all(isinstance(v, dict) for v in db.values()):
        return list(db.values())

    raise AssertionError(f"Unrecognized wonder_db.json structure. Top-level keys={sorted(db.keys())}")


def _get_first(d: dict, keys: list[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def test_wonder_db_loads_and_has_wonders():
    db = load_wonder_db()
    wonders = extract_wonders(db)
    assert wonders, "wonder_db.json contained no wonders"


def test_wonder_db_minimum_schema():
    db = load_wonder_db()
    wonders = extract_wonders(db)

    for i, w in enumerate(wonders):
        ctx = f"wonder[{i}] name={_get_first(w, ['name','wonder_name','title'])!r}"

        wid = _get_first(w, ["id", "wonder_id"])
        assert isinstance(wid, int), f"{ctx}: missing/invalid id field (expected int), got {wid!r}"

        name = _get_first(w, ["name", "wonder_name", "title"])
        assert isinstance(name, str) and name, f"{ctx}: missing/invalid name field"

        # Side can be encoded as "A/B", "side", "variant", etc.
        side = _get_first(w, ["side", "variant"])
        if side is not None:
            assert isinstance(side, str), f"{ctx}: side/variant must be str if present"

        start = _get_first(w, ["starting_resource", "start_resource", "starting"])
        assert start is not None, f"{ctx}: missing starting resource field (name may differ)"

        stages = _get_first(w, ["stages", "stage_costs", "wonder_stages"])
        assert isinstance(stages, list) and stages, f"{ctx}: missing/invalid stages list"
        assert 2 <= len(stages) <= 4, f"{ctx}: unexpected number of stages: {len(stages)}"

        # Each stage should have at least a cost or an effect; schema varies so keep it permissive.
        for si, st in enumerate(stages):
            assert isinstance(st, dict), f"{ctx}: stage[{si}] is not a dict"
            has_cost = any(k in st for k in ("cost", "cost_resources", "cost_coins", "resources", "coins"))
            has_effect = any(k in st for k in ("effects", "effect", "vp", "points", "shields", "science", "coins_gain"))
            assert has_cost or has_effect, f"{ctx}: stage[{si}] missing cost/effect fields. keys={sorted(st.keys())}"
