"""Export an evaluated experiment as a public, self-contained Pages snapshot.

Only derived statistics are exported. No training, model loading or deployment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from sevenwonders_rl.evaluation.dashboard import entry_from_directory, render_dashboard


def public_copy(value):
    """Remove local file references while retaining model names and hashes."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == 'checkpoint':
                result[key] = str(item).replace('\\', '/').rsplit('/', 1)[-1]
            elif key not in ('jsonl', 'jsonl_path', 'io_root', 'run_dir', 'out_dir') and not key.endswith('_path'):
                result[key] = public_copy(item)
        return result
    if isinstance(value, list):
        return [public_copy(item) for item in value]
    if isinstance(value, str) and re.search(r'(^|[\s\"\'=])(/(?:home|tmp|dev|mnt|Users|var|workspace)/|[A-Za-z]:[\\/])', value):
        raise ValueError('A local path remains in the public data; review the source statistics before export')
    return value


def prepare_demo(evaluation_dir, output, *, title, description, allow_evaluation_failures=False):
    evaluation_dir, output = Path(evaluation_dir).resolve(), Path(output).resolve()
    if output == evaluation_dir or evaluation_dir in output.parents or output in evaluation_dir.parents:
        raise ValueError('The public output and private evaluation directories must be separate')
    if not title.strip() or not description.strip():
        raise ValueError('A public title and experiment description are required')
    raw = (evaluation_dir / 'analysis.json').read_bytes()
    report = json.loads(raw)
    models = report.get('models', {})
    if 'candidate' not in models:
        raise ValueError('A completed candidate evaluation is required')
    for model in models.values():
        completed = model.get('completed', 0)
        coverage = model.get('coverage', {})
        if not completed or (not allow_evaluation_failures and (
                model.get('attempted') != completed or model.get('failed', 0) or model.get('truncated', 0))):
            raise ValueError('All selected evaluations must be completed without failures or truncations')
        if any(coverage.get(key) != completed for key in ('final_states', 'complete_age_sequences', 'action_traces')):
            raise ValueError('A detailed evaluation with final states, all ages and action traces is required')
        if coverage.get('score_sum_mismatches', 0) or coverage.get('invalid_age_sequences', 0):
            raise ValueError('Resolve evaluation data-quality errors before exporting the demo')
    entry = public_copy(entry_from_directory(report, evaluation_dir))
    entry['title'] = title.strip()
    metadata = public_copy({
        'title': title.strip(), 'description': description.strip(),
        'exported_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source_analysis_sha256': hashlib.sha256(raw).hexdigest(),
        'evaluation_seed': report.get('seed'),
        'provenance': entry.get('provenance', {}),
        'evaluation_failures': {key: {'attempted': model.get('attempted'),
                                     'completed': model.get('completed'),
                                     'failed': model.get('failed', 0),
                                     'truncated': model.get('truncated', 0)}
                                for key, model in models.items()},
    })
    incomplete = [f'{key}: {model.get("completed", 0)}/{model.get("attempted", 0)} games completed'
                  for key, model in models.items() if model.get('attempted') != model.get('completed')]
    if incomplete:
        metadata['description'] += ' Data-quality limitation: ' + '; '.join(incomplete) + '. See the data-quality view; incomplete results are not a fully paired baseline comparison.'
    document = render_dashboard([entry], demo=metadata)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'index.html').write_text(document, encoding='utf-8')
    (output / 'snapshot.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (output / '.nojekyll').touch()
    return output / 'index.html'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--out-dir', default=str(Path(__file__).resolve().parents[2] / 'demo'))
    parser.add_argument('--title', required=True)
    parser.add_argument('--description', required=True, help='Public experiment context and limitations')
    parser.add_argument('--allow-evaluation-failures', action='store_true',
                        help='Explicitly publish a report with failed games, retaining counts and a visible warning')
    args = parser.parse_args(argv)
    try:
        destination = prepare_demo(args.evaluation_dir, args.out_dir, title=args.title, description=args.description,
                                   allow_evaluation_failures=args.allow_evaluation_failures)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f'Öffentliche Demo vorbereitet: {destination}\nVorschau: {destination.as_uri()}\nNoch nicht veröffentlicht: Export prüfen, committen und pushen.')
    return destination


if __name__ == '__main__':
    main()
