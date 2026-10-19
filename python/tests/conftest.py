import json
from pathlib import Path

import numpy as np
import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def rng():
    """Deterministic RNG for any test that needs randomness."""
    return np.random.default_rng(123)


def load_fixture_json(filename: str) -> dict:
    path = FIXTURES / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fixture: {path}\n"
            f"Create it under python/tests/fixtures/ (see python/scripts/README.md and capture_fixtures.py)."
        )
    return json.loads(path.read_text())


def repo_root_from_tests() -> Path:
    # .../python/tests/conftest.py -> parents[0]=tests, [1]=python, [2]=repo_root
    return Path(__file__).resolve().parents[2]


def engine_path() -> Path:
    return repo_root_from_tests() / "build" / "7Wonders"


def engine_exists() -> bool:
    p = engine_path()
    return p.exists() and p.is_file()


@pytest.fixture
def require_engine():
    """Skip cleanly if the C++ engine is not built."""
    if not engine_exists():
        pytest.skip(f"C++ engine not built (missing {engine_path()})")


# ---- Frozen status fixtures (engine-free backbone) ----
# These filenames are your contract: keep them stable once created.

@pytest.fixture
def status_age1_turn0():
    return load_fixture_json("status_age1_turn0.json")


@pytest.fixture
def status_age2_turn0():
    return load_fixture_json("status_age2_turn0.json")


@pytest.fixture
def status_age3_turn0():
    return load_fixture_json("status_age3_turn0.json")


@pytest.fixture
def status_can_build_wonder():
    return load_fixture_json("status_can_build_wonder.json")


@pytest.fixture
def status_can_build_hand_free():
    return load_fixture_json("status_can_build_hand_free.json")


@pytest.fixture
def status_choice_production():
    return load_fixture_json("status_choice_production.json")


@pytest.fixture
def status_near_endgame():
    return load_fixture_json("status_near_endgame.json")


@pytest.fixture
def status_pool():
    """
    Returns a list of (path, status_dict) for all frozen statuses in fixtures/pool/.
    Useful for invariant-style component tests.
    """
    pool_dir = FIXTURES / "pool"
    if not pool_dir.exists():
        pytest.skip("No fixture pool found (python/tests/fixtures/pool). Run capture_fixtures.py --capture-pool.")
    paths = sorted(pool_dir.glob("*.json"))
    if not paths:
        pytest.skip("Fixture pool directory is empty.")
    return [(p, json.loads(p.read_text())) for p in paths]


@pytest.fixture(scope="session")
def card_db() -> dict:
    """Load card_db.json from package or source tree."""
    try:
        import sevenwonders_rl

        base = Path(sevenwonders_rl.__file__).resolve().parent
        p = base / "data" / "card_db.json"
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass

    src_path = repo_root_from_tests() / "python" / "src" / "sevenwonders_rl" / "data" / "card_db.json"
    if not src_path.exists():
        raise FileNotFoundError(f"card_db.json not found at {src_path}")
    return json.loads(src_path.read_text())
