"""Public exports exclude local provenance and remain inert when embedded in HTML."""

import json
import pytest

from sevenwonders_rl.evaluation.dashboard import (
    dashboard_data,
    render_dashboard,
    write_dashboard,
)

pytestmark = pytest.mark.component


def summary():
    return {
        "plan_id": "fixed-panel",
        "candidate": {"step": 64},
        "valid": True,
        "models": {"current": {"own_points": 42}},
        "directory": "/private/session",
    }


def test_export_excludes_local_training_paths_and_embeds_inert_json(tmp_path):
    report = summary()
    report["plan_id"] = "</script><script>alert(1)</script>"
    manifest = {
        "engine_binary": "/private/binary",
        "training_provenance": {"secret": "local"},
        "contract": {"ruleset": "bga-second-edition", "schema_hash": "schema"},
        "engine": {"source_commit": "engine-commit"},
    }
    (tmp_path / "summary.json").write_text(json.dumps(report))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    data = dashboard_data(report, manifest)
    assert "/private" not in json.dumps(data) and "secret" not in json.dumps(data)
    assert data["provenance"]["engine_commit"] == "engine-commit"
    html = write_dashboard(tmp_path).read_text()
    assert report["plan_id"] not in html
    payload = html.split('<script id="report-data" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(payload) == data
    assert "DASHBOARD_DATA" not in html and "DASHBOARD_JS" not in html
    assert "https://" not in html.split("<style>")[1].split("</style>")[0]


def test_rejects_unrelated_historical_reports():
    with pytest.raises(ValueError, match="Decision evaluation"):
        dashboard_data({"analysis": {"models": {}}})
    with pytest.raises(ValueError, match="Unsupported"):
        render_dashboard({"format": "historical"})
