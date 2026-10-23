import json
from html.parser import HTMLParser
from pathlib import Path

import pytest

from sevenwonders_rl.evaluation.dashboard import write_dashboard


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
