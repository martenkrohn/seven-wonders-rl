import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def opener():
    path = Path(__file__).parents[1] / 'scripts/open_dashboard.py'
    spec = importlib.util.spec_from_file_location('open_dashboard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dashboard_config(root):
    from sevenwonders_rl.utils.checkpoints import file_hash
    source = root / 'runs' / 'old-final'
    source.mkdir(parents=True)
    report = source / 'analysis.json'
    report.write_text(json.dumps({'models': {'candidate': {'win_share': .6}, 'initial': {'win_share': .1}},
                                  'paired_progress': {'initial': {'paired_valid': True}}}))
    checkpoint = source / 'model.pt'
    checkpoint.write_bytes(b'fixed reference')
    config = dict(version=1, active_run='runs/current', reference=dict(label='Fixed reference',
        model='candidate', report=str(report), report_sha256=file_hash(report),
        checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint)))
    (root / 'configs').mkdir()
    (root / 'configs' / 'dashboard.json').write_text(json.dumps(config))
    return config, report


def test_only_pinned_final_model_is_loaded_and_historical_discovery_is_removed(tmp_path):
    module = opener()
    config, report = dashboard_config(tmp_path)
    original = report.read_bytes()
    for name in ('long-old', 'green-priority-old', 'science-common-holdout-old'):
        folder = tmp_path / 'runs' / name
        folder.mkdir()
        (folder / 'dashboard.html').write_text('obsolete')
        (folder / 'analysis.json').write_text('must not be loaded')
    path = module.dashboard_path(tmp_path)
    from test_dashboard import DashboardParser
    parser = DashboardParser(); parser.feed(path.read_text())
    entry, = json.loads(parser.payload)
    assert list(entry['report']['models']) == ['candidate']
    assert entry['report']['models']['candidate']['win_share'] == .6
    assert entry['report']['paired_progress'] == {}
    assert report.read_bytes() == original


def test_wsl_browser_handoff_preserves_path(tmp_path, monkeypatch):
    module = opener()
    path = tmp_path / 'some dashboard.html'
    commands = []
    monkeypatch.setattr(module.shutil, 'which', lambda name: {'explorer.exe': '/explorer', 'wslpath': '/wslpath'}.get(name))
    monkeypatch.setattr(module.subprocess, 'check_output', lambda args, **kwargs: '\\\\wsl.localhost\\Ubuntu\\some dashboard.html\n')
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **kwargs: (commands.append(args) or SimpleNamespace(returncode=1)))
    module.open_browser(path)
    assert commands == [['/explorer', '\\\\wsl.localhost\\Ubuntu\\some dashboard.html']]


def test_opener_selects_only_configured_current_run_without_overwriting_it(tmp_path):
    module = opener()
    dashboard_config(tmp_path)
    current = tmp_path / 'runs' / 'current'
    current.mkdir()
    path = current / 'dashboard.html'
    path.write_text('live training dashboard')
    assert module.dashboard_path(tmp_path) == path
    assert path.read_text() == 'live training dashboard'


def test_generic_browser_and_missing_dashboard(tmp_path, monkeypatch):
    module = opener()
    with pytest.raises(FileNotFoundError, match='Noch kein'):
        module.dashboard_path(tmp_path)
    monkeypatch.setattr(module.shutil, 'which', lambda name: None)
    urls = []
    monkeypatch.setattr(module.webbrowser, 'open', lambda url: urls.append(url) or True)
    module.open_browser(tmp_path / 'some dashboard.html')
    assert urls[0].endswith('/some%20dashboard.html')
