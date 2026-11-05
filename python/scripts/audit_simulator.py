"""Compatibility entry point for the standalone engine's provenance audit."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    script = Path(__file__).resolve().parents[2] / "engine/scripts/audit_simulator.py"
    if not script.is_file():
        raise SystemExit("Engine missing: run git submodule update --init --recursive")
    runpy.run_path(str(script), run_name="__main__")
