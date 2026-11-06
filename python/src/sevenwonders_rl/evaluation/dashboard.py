"""Render portable dashboards from the active evaluator's public aggregate data."""

import hashlib
import json
from pathlib import Path


def dashboard_data(summary, manifest=None):
    """Deliberately exclude local paths, raw games and training manifests."""
    manifest = manifest or {}
    if (
        not isinstance(summary, dict)
        or not {"plan_id", "candidate", "models", "valid"} <= summary.keys()
    ):
        raise ValueError("Expected a Decision evaluation summary.json")
    if "current" not in summary["models"]:
        raise ValueError("The evaluation has no current model")
    return {
        "format": "decision-dashboard-v1",
        "evaluation": {
            k: summary.get(k)
            for k in (
                "plan_id",
                "candidate",
                "previous",
                "blocks",
                "references",
                "models",
                "paired_change",
                "valid",
            )
        },
        "provenance": {
            "ruleset": manifest.get("contract", {}).get(
                "ruleset", "bga-second-edition"
            ),
            "schema_hash": manifest.get("contract", {}).get("schema_hash"),
            "engine_commit": manifest.get("engine", {}).get("source_commit"),
            "source_summary_sha256": hashlib.sha256(
                json.dumps(
                    summary, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest(),
        },
    }


def render_dashboard(data):
    if data.get("format") != "decision-dashboard-v1":
        raise ValueError("Unsupported dashboard format")
    assets = Path(__file__).with_name("assets")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
    for source, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e")):
        payload = payload.replace(source, escaped)
    return (
        (assets / "dashboard.html")
        .read_text()
        .replace("/* DASHBOARD_CSS */", (assets / "dashboard.css").read_text())
        .replace("/* DASHBOARD_JS */", (assets / "dashboard.js").read_text())
        .replace("DASHBOARD_DATA", payload)
    )


def write_dashboard(session):
    session = Path(session)
    summary = json.loads((session / "summary.json").read_text())
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    data = dashboard_data(summary, manifest)
    target = session / "dashboard.html"
    target.write_text(render_dashboard(data))
    return target
