import json
from pathlib import Path
import pytest

pytestmark = pytest.mark.unit


def load_mechanics_json() -> dict:
    """
    Load mechanics.json from the installed package layout or from source tree.
    """
    # Installed/editable package path
    try:
        import sevenwonders_rl
        base = Path(sevenwonders_rl.__file__).resolve().parent
        p = base / "data" / "mechanics.json"
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass

    # Source-tree fallback (python/ is parents[1])
    p2 = Path(__file__).resolve().parents[1] / "src" / "sevenwonders_rl" / "data" / "mechanics.json"
    if p2.exists():
        return json.loads(p2.read_text())

    raise FileNotFoundError("Could not locate mechanics.json (expected under sevenwonders_rl/data/).")


def test_mechanics_loads():
    mech = load_mechanics_json()
    assert isinstance(mech, dict), f"mechanics.json must be a JSON object, got {type(mech).__name__}"
    assert mech, "mechanics.json is empty"


def test_mechanics_encodes_base_trade_cost_is_2():
    mech = load_mechanics_json()
    assert "trade" in mech, "Expected 'trade' key in mechanics.json"
    trade = mech["trade"]
    assert isinstance(trade, dict), "Expected trade to be a dict"
    assert "base_cost_per_resource" in trade, "Expected 'base_cost_per_resource' in trade"
    assert trade["base_cost_per_resource"] == 2, (
        f"Expected base_cost_per_resource == 2, got {trade['base_cost_per_resource']}"
    )


def test_mechanics_encodes_trade_discount_is_1():
    mech = load_mechanics_json()
    assert "trade" in mech, "Expected 'trade' key in mechanics.json"
    trade = mech["trade"]
    assert isinstance(trade, dict), "Expected trade to be a dict"
    assert "discount_cost_per_resource" in trade, "Expected 'discount_cost_per_resource' in trade"
    assert trade["discount_cost_per_resource"] == 1, (
        f"Expected discount_cost_per_resource == 1, got {trade['discount_cost_per_resource']}"
    )
