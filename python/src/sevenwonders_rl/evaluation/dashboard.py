"""Generate a self-contained, browser-local evaluation dashboard."""
import json
from html import escape
from pathlib import Path

from sevenwonders_rl.evaluation.model_labels import model_label


def dashboard_entry(report, title):
    # Presentation labels use the same identities as the console report.
    labels = {name: model_label(name, report.get('model_descriptors', {}).get(name))
              for name in report['models']}
    return {'title': title, 'labels': labels, 'report': report}


def entry_from_directory(report, directory):
    entry = dashboard_entry(report, directory.name)
    manifest_path = directory / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        entry['provenance'] = {key: manifest.get(key) for key in (
            'feature_semantics_version', 'model_architecture_version', 'training_version',
            'engine_sha256', 'card_database_sha256', 'git_commit', 'dirty')}
    return entry


def render_dashboard(entries, *, demo=None):
    """Render local reports or a deliberately selected public snapshot."""
    assets = Path(__file__).with_name('assets')
    payload = json.dumps(entries, ensure_ascii=False, allow_nan=False)
    # JSON is embedded in a script data block, never interpreted as HTML/code.
    payload = payload.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    template = (assets / 'dashboard.html').read_text(encoding='utf-8')
    introduction = ''
    if demo:
        introduction = (
            '<section class="notice demo-context" aria-label="Public demo context">'
            f'<h2>{escape(demo["title"])}</h2><p>{escape(demo["description"])}</p>'
            '<p>Public experiment snapshot · exploratory results, not a general playing-strength benchmark. '
            f'Exported {escape(demo["exported_at"])}. '
            '<a href="https://github.com/martenkrohn/seven-wonders-rl">Source code &amp; experiment documentation</a>.</p>'
            '<p>The example data is public. Additional files you open are processed locally in your browser.</p>'
            '</section>')
    document = (template.replace('/* DASHBOARD_CSS */', (assets / 'dashboard.css').read_text(encoding='utf-8'))
                .replace('/* DASHBOARD_THEME */', (assets / 'theme.js').read_text(encoding='utf-8'))
                .replace('/* DASHBOARD_JS */', (assets / 'dashboard.js').read_text(encoding='utf-8'))
                .replace('<!-- DASHBOARD_INTRODUCTION -->', introduction)
                .replace('DASHBOARD_DATA', payload))
    if demo:
        document = document.replace('Lokale Auswertung', 'Öffentliche Beispieldaten')
        document = document.replace('Alle Daten bleiben in deinem Browser.', 'Zusätzliche Dateien bleiben im Browser.')
        document = document.replace('Lokale Trainingsanalyse', 'Öffentliche Demo · Trainingsanalyse')
    return document


def write_dashboard(output, report, *, comparisons=()):
    output = Path(output)
    entries = [entry_from_directory(report, output)]
    for directory in comparisons:
        directory = Path(directory)
        other = json.loads((directory / 'analysis.json').read_text(encoding='utf-8'))
        entries.append(entry_from_directory(other, directory))
    document = render_dashboard(entries)
    destination = output / 'dashboard.html'
    destination.write_text(document, encoding='utf-8')
    return destination
