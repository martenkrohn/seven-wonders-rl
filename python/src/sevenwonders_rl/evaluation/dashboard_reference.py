"""One explicitly pinned historical reference for the current dashboard."""
import copy
import json
from pathlib import Path

from sevenwonders_rl.evaluation.dashboard import dashboard_entry, render_dashboard
from sevenwonders_rl.utils.checkpoints import file_hash


def read_dashboard_config(path, root):
    config = json.loads(Path(path).read_text())
    if config.get('version') != 1:
        raise ValueError('Unsupported dashboard configuration')
    ref = dict(config['reference'])
    for key in ('report', 'checkpoint'):
        source = (Path(root) / ref[key]).resolve()
        if file_hash(source) != ref[key + '_sha256']:
            raise ValueError(f'Pinned dashboard {key} changed: {source}')
        ref[key] = str(source)
    return {**config, 'reference': ref, 'active_run': str((Path(root) / config['active_run']).resolve())}


def reference_entry(config):
    ref = config['reference']
    report = copy.deepcopy(json.loads(Path(ref['report']).read_text()))
    key = ref['model']
    # Retain precisely the selected model, with its original evaluation data.
    report['models'] = {'candidate': report['models'][key]}
    report['model_descriptors'] = {'candidate': dict(checkpoint=ref['checkpoint'],
        sha256=ref['checkpoint_sha256'], label=ref['label'])}
    report['paired_progress'] = {}
    report.pop('without_science_specialist', None)
    return {**dashboard_entry(report, 'Feste Referenz · letzte bisherige Evaluation'), 'historical_reference': True}


def workspace_dashboard(root):
    root = Path(root)
    config = read_dashboard_config(root / 'configs' / 'dashboard.json', root)
    active = Path(config['active_run']) / 'dashboard.html'
    if active.is_file():
        return active
    destination = root / 'runs' / 'analysis-dashboard' / 'dashboard.html'
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = render_dashboard([reference_entry(config)])
    temporary = destination.with_suffix('.html.tmp')
    temporary.write_text(document, encoding='utf-8')
    temporary.replace(destination)
    return destination


def ensure_reference_evaluation(config, specs, plan_id, args, output, repo_root, io_root):
    """Evaluate the same fixed model once per run on exactly its current panel."""
    from datetime import datetime, timezone
    from sevenwonders_rl.evaluation.runner import evaluate_model
    from sevenwonders_rl.evaluation.report import write_report
    from sevenwonders_rl.utils.checkpoints import load_model
    from sevenwonders_rl.utils.runtime import atomic_json

    destination = Path(output) / 'reference_evaluation'
    ref = config['reference']
    complete = all((destination / p).is_file() for p in ('analysis.json', 'summary.json', 'candidate.jsonl'))
    if complete:
        report = json.loads((destination / 'analysis.json').read_text())
        summary = json.loads((destination / 'summary.json').read_text())
        if (report.get('plan_id') != plan_id or report.get('reference_checkpoint_sha256') != ref['checkpoint_sha256']
                or not summary.get('valid_for_selection')):
            raise ValueError('Cached dashboard reference does not match the current evaluation plan/model')
        return destination
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        destination.rename(destination.with_name('incomplete_reference_' + stamp))
    destination.mkdir()
    print('Feste Dashboard-Referenz: einmalige Auswertung auf dem neuen Kontrollplan.', flush=True)
    model = load_model(ref['checkpoint'])
    result = evaluate_model(model, 'cpu', specs, repo_root, Path(io_root) / 'reference',
        destination / 'candidate.jsonl', args.eval_action_mode, args.engine_poll_us, args.poll_seconds,
        'fixed_dashboard_reference', detailed=True, workers=args.eval_workers,
        batch_opponents=args.opponent_inference == 'batched')
    if not result['valid_for_selection'] or file_hash(ref['checkpoint']) != ref['checkpoint_sha256']:
        raise RuntimeError('Dashboard reference evaluation incomplete or source changed')
    from sevenwonders_rl.evaluation.selfplay import read_records
    write_report(destination, {'candidate': read_records(destination / 'candidate.jsonl')},
        dict(seed=args.eval_seed, action_mode=args.eval_action_mode, plan_id=plan_id,
             references=list({r['checkpoint']: r for spec in specs for r in (spec['left'], spec['right'])}.values()),
             reference_checkpoint_sha256=ref['checkpoint_sha256'], evaluation_role='fixed_dashboard_reference',
             model_descriptors={'candidate': dict(checkpoint=ref['checkpoint'], sha256=ref['checkpoint_sha256'],
                                                 label=ref['label'])}), {})
    atomic_json(destination / 'summary.json', result)
    return destination
