#!/usr/bin/env python3
"""Render a Decision evaluation without rerunning games or loading weights."""

import argparse
import json
from pathlib import Path

from sevenwonders_rl.evaluation.dashboard import (
    dashboard_data,
    render_dashboard,
    write_dashboard,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        required=True,
        help="Directory containing summary.json",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        help="Export only index.html and sanitized snapshot.json here",
    )
    args = parser.parse_args()
    if args.publish_dir:
        summary = json.loads((args.evaluation_dir / "summary.json").read_text())
        manifest = json.loads((args.evaluation_dir / "manifest.json").read_text())
        if not summary["valid"]:
            parser.error(
                "An incomplete evaluation cannot be exported as the public example"
            )
        data = dashboard_data(summary, manifest)
        args.publish_dir.mkdir(parents=True, exist_ok=True)
        (args.publish_dir / "snapshot.json").write_text(
            json.dumps(data, indent=2, allow_nan=False) + "\n"
        )
        path = args.publish_dir / "index.html"
        path.write_text(render_dashboard(data))
    else:
        path = write_dashboard(args.evaluation_dir)
    print(path)


if __name__ == "__main__":
    main()
