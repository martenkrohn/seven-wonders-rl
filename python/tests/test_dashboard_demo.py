import importlib.util
import json
from pathlib import Path

import pytest

from test_dashboard import DashboardParser


def exporter():
    path = Path(__file__).parents[1] / 'scripts/prepare_dashboard_demo.py'
    spec = importlib.util.spec_from_file_location('prepare_dashboard_demo', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detailed_report():
    return {
        'models': {'candidate': {
            'attempted': 12, 'completed': 12, 'failed': 0, 'truncated': 0,
            'coverage': {'final_states': 12, 'complete_age_sequences': 12, 'action_traces': 12},
            'categories': {'science': {'mean': 0}},
        }},
        'model_descriptors': {'candidate': {'checkpoint': '/home/private/run/latest.pt', 'timestep': 262144}},
        'references': [{'checkpoint': '/home/private/run/checkpoint_65536.pt', 'sha256': 'abc'}],
        'seed': 39001,
    }


@pytest.mark.unit
def test_demo_exports_statistics_without_local_paths_or_raw_artifacts(tmp_path):
    source, output = tmp_path / 'private-run', tmp_path / 'public-demo'
    source.mkdir()
    raw = json.dumps(detailed_report())
    (source / 'analysis.json').write_text(raw)
    (source / 'candidate.jsonl').write_text('private raw games')
    (source / 'manifest.json').write_text(json.dumps({'feature_semantics_version': 3, 'io_root': '/home/private/io'}))
    destination = exporter().prepare_demo(source, output, title='Example <experiment>', description='Context <script>',
                                         title_de='Beispiel <Experiment>', description_de='Kontext <script>')
    text = destination.read_text()
    assert '/home/private' not in text
    assert 'private-run' not in text
    assert 'Context &lt;script&gt;' in text
    assert 'data-language-de="Kontext &lt;script&gt;"' in text
    assert '<html lang="en">' in text
    assert 'Public experiment snapshot' in text
    assert 'Öffentliche Beispieldaten' in text
    parser = DashboardParser()
    parser.feed(text)
    entry, = json.loads(parser.payload)
    assert entry['report']['references'][0]['checkpoint'] == 'checkpoint_65536.pt'
    assert entry['provenance']['feature_semantics_version'] == 3
    assert entry['title_de'] == 'Beispiel <Experiment>'
    assert entry['report']['models']['candidate']['categories']['science']['mean'] == 0
    assert {p.name for p in output.iterdir()} == {'index.html', 'snapshot.json', '.nojekyll'}
    assert (source / 'analysis.json').read_text() == raw


@pytest.mark.unit
@pytest.mark.parametrize('change', ['incomplete', 'compact', 'score_error', 'embedded_path'])
def test_demo_refuses_incomplete_or_unreviewed_data(tmp_path, change):
    source, output = tmp_path / 'source', tmp_path / 'demo'
    source.mkdir()
    report = detailed_report()
    candidate = report['models']['candidate']
    if change == 'incomplete':
        candidate['attempted'] = 13
    elif change == 'compact':
        candidate['coverage']['action_traces'] = 0
    elif change == 'score_error':
        candidate['coverage']['score_sum_mismatches'] = 1
    else:
        candidate['unexpected'] = '/home/private/location'
    (source / 'analysis.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        exporter().prepare_demo(source, output, title='Example', description='Context')
    assert not output.exists()


@pytest.mark.unit
def test_explicit_failure_override_retains_counts_and_visible_warning(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    report = detailed_report()
    report['models']['candidate'].update(attempted=13, failed=1)
    (source / 'analysis.json').write_text(json.dumps(report))
    destination = exporter().prepare_demo(source, tmp_path / 'demo', title='Example', description='Context',
                                         allow_evaluation_failures=True)
    assert '12/13 games completed' in destination.read_text()
    parser = DashboardParser()
    parser.feed(destination.read_text())
    assert json.loads(parser.payload)[0]['report']['models']['candidate']['failed'] == 1
