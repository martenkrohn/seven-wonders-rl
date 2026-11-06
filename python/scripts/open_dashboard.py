#!/usr/bin/env python3
"""Open the latest successful active-agent report, rebuilding HTML if needed."""

import argparse
import json
from pathlib import Path
import webbrowser

from sevenwonders_rl.evaluation.dashboard import write_dashboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = [
        args.run_dir / name / "latest.json"
        for name in ("evaluation", "intermediate_evaluation")
    ]
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        parser.exit(
            2, "No successful evaluation exists yet. Run eval_decision.py first.\n"
        )
    # Prefer the independent full evaluation to the quick training control.
    latest = candidates[0]
    session = latest.parent / json.loads(latest.read_text())["directory"]
    path = write_dashboard(session).resolve()
    print(path)
    webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
