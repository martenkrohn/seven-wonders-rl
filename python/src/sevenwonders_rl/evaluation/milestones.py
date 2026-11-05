"""Fixed-panel evaluations published during training, without training RNG use."""
import json
from pathlib import Path

from sevenwonders_rl.evaluation.dashboard import dashboard_entry, render_dashboard
from sevenwonders_rl.evaluation.report import write_report, summarize_model
from sevenwonders_rl.evaluation.runner import evaluate_model, aggregate
from sevenwonders_rl.evaluation.selfplay import paired_progress, plan_identity, read_records
from sevenwonders_rl.utils.runtime import atomic_json
from sevenwonders_rl.utils.seeding import derive_seed
from sevenwonders_rl.evaluation.dashboard_reference import reference_entry, ensure_reference_evaluation


def milestone_steps(total, n_envs):
    if total % (4 * n_envs):
        raise ValueError('Quarterly evaluations require total-timesteps divisible by 4*n-envs')
    return [total * i // 4 for i in range(1, 5)]


def fixed_panel(references, panel, seed, blocks):
    indexed = {}
    for ref in references:
        group = ref['group']
        index = sum(key.startswith(group + ':') for key in indexed)
        indexed[f'{group}:{index}'] = ref
    pairs = json.loads(Path(panel).read_text())['pairs']
    if not pairs or any(len(pair) != 2 or any(k not in indexed for k in pair) for pair in pairs):
        raise ValueError('Evaluation panel must name existing group:index reference pairs')
    return [dict(left=indexed[left], right=indexed[right], agent_seat=seat,
                 engine_seed=derive_seed(seed, 'evaluation', block))
            for block in range(blocks) for left, right in pairs for seat in range(3)]


class MilestoneEvaluations:
    def __init__(self, output, args, references, repo_root, io_root, checkpoint_save, *, resume_step=None):
        self.output, self.args = Path(output), args
        self.repo_root, self.io_root, self.save = repo_root, io_root, checkpoint_save
        self.specs = fixed_panel(references, args.eval_panel, args.eval_seed, args.eval_games_per_matchup)
        self.plan_id = plan_identity(self.specs, args.eval_action_mode)
        self.continuous = getattr(args, 'continuous', False)
        self.interval = args.eval_every if self.continuous else None
        self.steps = [] if self.continuous else milestone_steps(args.total_timesteps, args.n_envs)
        self.entries, self.curve = [], []
        self.baseline = None
        self.previous = None
        self.reference_config = getattr(args, 'dashboard_reference_spec', None)
        if resume_step is not None:
            plan_path = self.output / 'evaluation_plan.json'
            saved = json.loads(plan_path.read_text()) if plan_path.is_file() else None
            if saved is None and resume_step != 0:
                raise ValueError('Resume requires the evaluation plan')
            if saved is not None and (saved['plan_id'] != self.plan_id or saved['milestones'] != self.steps
                                      or saved.get('evaluation_interval') != self.interval):
                raise ValueError('Resume evaluation plan changed')
            from datetime import datetime, timezone
            for directory in sorted((self.output / 'evaluations').glob('step_*'),
                                    key=lambda p: int(p.name.removeprefix('step_'))):
                step = int(directory.name.removeprefix('step_'))
                if step > resume_step or not all((directory / name).is_file()
                        for name in ('analysis.json', 'summary.json', 'candidate.jsonl', 'model.pt')):
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
                    directory.rename(directory.with_name('incomplete_' + directory.name + '_' + stamp))
                    continue
                report = json.loads((directory / 'analysis.json').read_text())
                percent = self.percent(step)
                self.remember_entry(dashboard_entry(report, self.label(step)))
                m = report['models']['candidate']
                self.curve.append(dict(step=step, percent=percent, win_share=m['win_share'],
                    vp_diff_mean=m['vp_diff_mean'], directory=str(directory),
                    agent_vp_mean=m.get('agent_vp_mean'), vp_diff_ci95=m.get('vp_diff_ci95'),
                    win_share_ci95=m.get('win_share_ci95'), previous_progress=report.get('previous_progress'),
                    without_science_margin=report.get('without_science_specialist', {}).get('vp_diff_mean')))
                if step == 0:
                    self.baseline = directory / 'candidate.jsonl'
                self.previous = directory / 'candidate.jsonl'
            if self.baseline is None and resume_step != 0:
                raise ValueError('Resume requires the preserved initial evaluation')
        atomic_json(self.output / 'evaluation_plan.json', dict(plan=self.specs, plan_id=self.plan_id,
            milestones=self.steps, evaluation_interval=self.interval,
            action_mode=args.eval_action_mode, seed=args.eval_seed,
            purpose='Repeated training control; not an untouched final holdout'))
        self.update_status(resume_step or 0, 'training' if resume_step is not None else 'initial_evaluation')
        self.publish_dashboard()

    def percent(self, step):
        return None if self.continuous else round(100 * step / self.args.total_timesteps)

    def label(self, step):
        return f'{step:,} Schritte' if self.continuous else f'{self.percent(step)}% · {step:,} Schritte'

    def next_step(self, step):
        return ((step // self.interval + 1) * self.interval if self.continuous else
                next((s for s in self.steps if s > step), None))

    def is_due(self, step):
        return (step > 0 and step % self.interval == 0) if self.continuous else step in self.steps

    def finished(self, step):
        return not self.continuous and step == self.args.total_timesteps

    def remember_entry(self, entry):
        self.entries.append(entry)
        if self.continuous and len(self.entries) > 13:
            del self.entries[1]  # Keep the start and the twelve latest reports in HTML.

    def update_status(self, step, status='training'):
        atomic_json(self.output / 'training_status.json', dict(timestep=step,
            total_timesteps=None if self.continuous else self.args.total_timesteps,
            percent=None if self.continuous else 100 * step / self.args.total_timesteps,
            status=status, completed_evaluations=self.curve,
            next_evaluation=self.next_step(step)))

    def publish_dashboard(self):
        """Publish compact selectors; the full curve belongs in its own view."""
        visible = list(reversed(self.entries))
        if self.reference_config:
            visible.append(reference_entry(self.reference_config))
        status = json.loads((self.output / 'training_status.json').read_text())
        reference_path = self.output / 'reference_evaluation' / 'summary.json'
        league_path = self.output / 'league_status.json'
        training = dict(evaluations=self.curve,
            status={k: v for k, v in status.items() if k != 'completed_evaluations'}, plan_id=self.plan_id,
            continuous=self.continuous, interval=self.interval,
            reference=json.loads(reference_path.read_text()) if reference_path.is_file() else None,
            league=json.loads(league_path.read_text()) if league_path.is_file() else None)
        doc = render_dashboard(visible, training=training)
        destination = self.output / 'dashboard.html'
        temporary = destination.with_suffix('.html.tmp')
        temporary.write_text(doc, encoding='utf-8')
        temporary.replace(destination)

    def evaluate(self, model, step):
        reference_directory = (ensure_reference_evaluation(self.reference_config, self.specs, self.plan_id,
            self.args, self.output, self.repo_root, self.io_root) if self.reference_config else None)
        percent = self.percent(step)
        destination = self.output / 'evaluations' / f'step_{step:07d}'
        destination.mkdir(parents=True)
        checkpoint = destination / 'model.pt'
        self.save(checkpoint)
        self.update_status(step, 'evaluating')
        raw = destination / 'candidate.jsonl'
        result = evaluate_model(model, 'cpu', self.specs, self.repo_root,
            self.io_root / 'milestones' / str(step), raw, self.args.eval_action_mode,
            self.args.engine_poll_us, self.args.poll_seconds, f'step:{step}',
            detailed=True, workers=self.args.eval_workers,
            batch_opponents=self.args.opponent_inference == 'batched')
        records = read_records(raw)
        if not result['valid_for_selection']:
            atomic_json(destination / 'failure_summary.json', result)
            raise RuntimeError(f'Milestone evaluation contains failed/truncated games: {destination}')
        models = {'candidate': records}
        descriptors = {'candidate': dict(checkpoint=str(checkpoint), label=self.label(step))}
        paired = {}
        if self.baseline is None:
            self.baseline = raw
        if reference_directory is not None:
            ref = self.reference_config['reference']
            models['reference'] = read_records(reference_directory / 'candidate.jsonl')
            descriptors['reference'] = dict(checkpoint=ref['checkpoint'], sha256=ref['checkpoint_sha256'], label=ref['label'])
            paired['reference'] = paired_progress(raw, reference_directory / 'candidate.jsonl')
        elif self.baseline != raw:
            models['initial'] = read_records(self.baseline)
            descriptors['initial'] = dict(checkpoint=str(self.baseline.parent / 'model.pt'), label='Start · 0 Schritte')
            paired['initial'] = paired_progress(raw, self.baseline)
        no_science = [r for r in records if not any(
            v.get('group') == 'science' for v in r['opponents'].values())]
        metadata = dict(seed=self.args.eval_seed, action_mode=self.args.eval_action_mode,
            plan_id=self.plan_id, references=list({r['checkpoint']: r for spec in self.specs
                for r in (spec['left'], spec['right'])}.values()),
            model_descriptors=descriptors, training_step=step, training_percent=percent,
            evaluation_role='repeated_training_control', without_science_specialist=summarize_model(no_science))
        if self.previous is not None and self.previous != raw:
            metadata['previous_progress'] = paired_progress(raw, self.previous)
            metadata['previous_step'] = int(self.previous.parent.name.removeprefix('step_'))
        write_report(destination, models, metadata, paired)
        report = json.loads((destination / 'analysis.json').read_text())
        self.remember_entry(dashboard_entry(report, self.label(step)))
        self.curve.append(dict(step=step, percent=percent, win_share=result['win_share'],
            vp_diff_mean=result['vp_diff_mean'], directory=str(destination),
            agent_vp_mean=result.get('agent_vp_mean'),
            vp_diff_ci95=report['models']['candidate'].get('vp_diff_ci95'),
            win_share_ci95=report['models']['candidate'].get('win_share_ci95'),
            previous_progress=metadata.get('previous_progress'),
            without_science_margin=metadata['without_science_specialist']['vp_diff_mean']))
        self.previous = raw
        atomic_json(self.output / 'learning_curve.json', dict(plan_id=self.plan_id, evaluations=self.curve))
        self.update_status(step, 'completed' if self.finished(step) else 'training')
        self.publish_dashboard()
        result.update(step=step, plan_id=self.plan_id,
                      without_science_specialist=aggregate(no_science),
                      paired_progress=paired.get('reference', paired.get('initial')), dashboard=str(self.output / 'dashboard.html'))
        if paired:
            key = 'reference' if 'reference' in paired else 'initial'
            result.update(paired[key], comparison_label='Gegenüber festem Referenzmodell' if key == 'reference' else 'Gegenüber Start (0%)')
        atomic_json(destination / 'summary.json', result)
        return result
