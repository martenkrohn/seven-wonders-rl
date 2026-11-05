import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from sevenwonders_rl.evaluation.dashboard import write_dashboard, write_science_dashboard, render_dashboard


class DashboardParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_data = False
        self.payload = ''
        self.remote_scripts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'script':
            self.in_data = attrs.get('id') == 'evaluation-data'
            if 'src' in attrs:
                self.remote_scripts.append(attrs['src'])

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_data = False

    def handle_data(self, text):
        if self.in_data:
            self.payload += text


@pytest.mark.unit
def test_embedded_artwork_and_training_context_are_safe_and_portable(monkeypatch):
    import re
    import sevenwonders_rl.evaluation.dashboard as module
    pictures = {'resources': {'wood': 'data:image/png;base64,aGVsbG8='},
                'cards': {'Workshop': 'data:image/png;base64,Y2FyZA=='}}
    monkeypatch.setattr(module, 'artwork', lambda: pictures)
    malicious = '</script><script>alert(1)</script>'
    training = {'plan_id': malicious, 'evaluations': [{'step': 64, 'vp_diff_mean': 1.5}]}
    # A literal template token in saved data must not be replaced inside the payload.
    text = render_dashboard([{'title':'DASHBOARD_CONTEXT','report':{'models':{'candidate':{}}}}], training=training)
    assert malicious not in text
    match = re.search(r'<script id="dashboard-context" type="application/json">(.*?)</script>', text, re.S)
    payload = json.loads(match[1])
    assert payload == dict(artwork=pictures, training=training)
    parser = DashboardParser();parser.feed(text)
    assert json.loads(parser.payload)[0]['title'] == 'DASHBOARD_CONTEXT'
    assert "img-src data:" in text and not parser.remote_scripts


@pytest.mark.unit
def test_self_contained_dashboard_escapes_data_and_preserves_missing_values(tmp_path):
    malicious = '</script><script>alert("x")</script>'
    report = {'models': {'candidate': {'categories': {'science': {'mean': 0}, 'yellow': None},
                                     'cards': [{'card': malicious, 'built': 0}]}},
              'model_descriptors': {'candidate': {'checkpoint': 'latest.pt', 'timestep': 262144}}}
    other = tmp_path / 'other'
    other.mkdir()
    (tmp_path / 'manifest.json').write_text(json.dumps({'feature_semantics_version': 3}))
    (other / 'analysis.json').write_text(json.dumps(report))
    destination = write_dashboard(tmp_path, report, comparisons=[other])
    text = destination.read_text()
    assert malicious not in text
    assert 'DASHBOARD_DATA' not in text
    assert '/* DASHBOARD_JS */' not in text
    assert '/* DASHBOARD_I18N */' not in text
    assert '<html lang="en">' in text
    assert 'id="language-select"' in text
    assert '<option value="en" lang="en">English</option>' in text
    assert 'fetch(' not in text
    assert 'Trainingsende – Schritt 262.144' in text
    parser = DashboardParser()
    parser.feed(text)
    entries = json.loads(parser.payload)
    assert len(entries) == 2
    assert entries[0]['provenance']['feature_semantics_version'] == 3
    assert entries[0]['report']['models']['candidate']['cards'][0]['card'] == malicious
    assert entries[0]['report']['models']['candidate']['categories']['yellow'] is None
    assert entries[0]['report']['models']['candidate']['categories']['science']['mean'] == 0
    assert not parser.remote_scripts
    assert destination.name == 'dashboard.html'


@pytest.mark.unit
def test_build_command_uses_saved_statistics_without_loading_weights(tmp_path):
    import importlib.util
    path = Path(__file__).parents[1] / 'scripts/build_dashboard.py'
    spec = importlib.util.spec_from_file_location('build_dashboard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = {'models': {'initial': {}, 'candidate': {}}, 'model_descriptors': {}}
    (tmp_path / 'analysis.json').write_text(json.dumps(report))
    assert module.main(['--evaluation-dir', str(tmp_path)]) == tmp_path / 'dashboard.html'


@pytest.mark.unit
def test_science_dashboard_preserves_training_identity_and_can_combine_with_evaluation(tmp_path):
    run = {'run': '/tmp/entropy-seed31', 'seed': 31, 'ent_coef': .03, 'gae_lambda': .95,
           'windows': [], 'training_result': None, 'completed_games': 0,
           'first_early_green_nonloss_discoveries': {}, 'note': '</script><script>alert(1)</script>'}
    summary = {'schema_version': 1, 'runs': [run]}
    source = tmp_path / 'summary.json'
    source.write_text(json.dumps(summary))
    destination = write_science_dashboard(tmp_path / 'science.html', summary)
    parser = DashboardParser()
    parser.feed(destination.read_text())
    entry, = json.loads(parser.payload)
    assert entry['kind'] == 'science_diagnostics' and 'report' not in entry
    assert entry['science']['training_result'] is None
    assert entry['title'] == 'entropy-seed31'
    assert run['note'] not in destination.read_text()
    assert not parser.remote_scripts
    # The existing builder accepts both standalone and combined saved reports.
    import importlib.util
    path = Path(__file__).parents[1] / 'scripts/build_dashboard.py'
    spec = importlib.util.spec_from_file_location('build_science_dashboard', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    standalone = module.main(['--science-summary', str(source), '--out-dir', str(tmp_path / 'browser')])
    assert standalone == tmp_path / 'browser' / 'dashboard.html'
    (tmp_path / 'analysis.json').write_text(json.dumps({'models': {'candidate': {}}}))
    combined = module.main(['--evaluation-dir', str(tmp_path), '--science-summary', str(source)])
    parser = DashboardParser()
    parser.feed(combined.read_text())
    entries = json.loads(parser.payload)
    assert len(entries) == 2 and entries[0]['report']['models'] == {'candidate': {}}
    assert entries[1]['science']['ent_coef'] == .03
    with pytest.raises(SystemExit):
        module.main(['--evaluation-dir', str(tmp_path), '--out-dir', str(tmp_path / 'bad')])
