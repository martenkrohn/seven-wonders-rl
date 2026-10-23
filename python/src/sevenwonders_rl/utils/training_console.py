"""Readable, append-only training output, also suitable for redirected logs."""
from __future__ import annotations

import math
import time

from sevenwonders_rl.evaluation.telemetry import CATEGORIES, LABELS


def number(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def duration(seconds: float) -> str:
    seconds = max(0, math.ceil(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def decimal(value, digits=2, signed=False):
    if value is None:
        return 'nicht erfasst'
    return format(value, f'+.{digits}f' if signed else f'.{digits}f').replace('.', ',')


def category_lines(categories, *, detailed=False):
    lines = ['\n PUNKTEHERKUNFT — Punkte je Spiel', '-' * 78,
             f" {'Kategorie':25} {'Mittel':>9} {'Median':>9} {'P10–P90':>17} {'Maximum':>9}"
             if detailed else f" {'Kategorie':25} {'Mittel':>9} {'P10–P90':>17} {'Maximum':>9}"]
    for key in (*CATEGORIES, 'commercial_combined', 'total'):
        stat = categories.get(key)
        if key == 'commercial_combined' and stat is None:
            continue
        label = LABELS.get(key, 'Handel + Münzen (gesamt)' if key == 'commercial_combined' else 'Gesamt')
        if stat is None:
            lines.append(f' {label:25} nicht erfasst')
            continue
        interval = f"{decimal(stat['p10'], 1)}–{decimal(stat['p90'], 1)}"
        middle = f" {decimal(stat['median']):>9}" if detailed else ''
        lines.append(f" {label:25} {decimal(stat['mean']):>9}{middle} {interval:>17} {decimal(stat['max'], 1):>9}")
    sci = categories.get('science')
    if sci:
        lines.extend(['\n WISSENSCHAFT',
                      f" Ohne Punkte: {decimal(100 * sci['zero_fraction'], 1)} % der Spiele | "
                      f"Mit Punkten: {decimal(100 * sci['positive_fraction'], 1)} %",
                      (f" Bei vorhandenen Wissenschaftspunkten: Ø {decimal(sci['positive_mean'])}"
                       if sci['positive_mean'] is not None else ' Keine Spiele mit Wissenschaftspunkten.')])
    lines.extend([' P10–P90: mittlerer Bereich der Einzelspiele, kein Konfidenzintervall.',
                  ' Kategorien-Maxima können aus verschiedenen Spielen stammen.'])
    return lines


def evaluation_lines(summary, *, new_best=False):
    def metric(key, *, percent=False, signed=False):
        value = summary.get(key)
        if value is None:
            return "n/v"
        return (f"{value:.1%}" if percent else f"{value:+.2f}" if signed else f"{value:.2f}")

    lines = [
        f"  Spiele:          {summary['completed']}/{summary['attempted']} abgeschlossen | "
        f"{summary['failed']} Fehler | {summary['truncated']} abgebrochen",
        f"  Sieganteil:      {metric('win_share', percent=True)} (geteilte Siege anteilig)",
        f"  Eigene Punkte:   {metric('agent_vp_mean')} im Mittel",
        f"  Punktevorsprung: {metric('vp_diff_mean', signed=True)} gegen den besten Gegner",
    ]
    if summary.get('paired_valid'):
        label = summary.get('comparison_label', 'Seit Start')
        lines.append(f"  {label}: {metric('vp_diff_gain', signed=True)} Punkte | "
                     f"{summary['win_share_gain'] * 100:+.1f} Prozentpunkte Sieganteil")
        interval = summary.get('vp_diff_gain_ci95')
        if interval is not None:
            lines.append(f"  95%-Intervall:   [{interval[0]:+.2f}, {interval[1]:+.2f}] Punkte Zuwachs")
    if not summary.get('valid_for_selection'):
        lines.append("  Status:          UNGÜLTIG für Modellauswahl")
    elif new_best:
        lines.append("  Modell:          Neuer Bestwert -> best.pt gespeichert")
    if summary.get('saved_evaluated_checkpoint'):
        lines.append('  Modell:          Erfolgreich geprüft -> evaluated.pt gespeichert')
    if summary.get('failure_reasons'):
        lines.append(f"  Fehlerursachen:  {summary['failure_reasons']}")
    if summary.get('categories'):
        lines.extend(category_lines(summary['categories']))
    return lines


class TrainingConsole:
    def __init__(self, args, out_dir, io_root):
        self.args = args
        self.started = time.perf_counter()
        self.eval_seconds = 0.0
        self.eval_count = 0
        self.previous_evaluation = None
        print(f"\n{'=' * 72}\n"
              f"  {'SELF-PLAY' if args.mode == 'selfplay' else 'FIXED'} TRAINING | Seed {args.seed}\n"
              f"{'=' * 72}\n"
              f"  Ziel:          {number(args.total_timesteps)} Schritte | {args.n_envs} Umgebungen\n"
              f"  Fortschritt:   alle {number(args.log_every)} Schritte\n"
              f"  Evaluation:    alle {number(args.eval_every)} Schritte\n"
              f"  Gegner:        {args.opponent_inference}\n"
              f"  Ergebnisse:    {out_dir}\n"
              f"  Spielaustausch: {io_root}\n"
              f"  Restzeit:      Schätzung inkl. Evaluationen; passt sich im Lauf an.\n"
              f"{'=' * 72}", flush=True)

    def evaluation_start(self, step, games, *, references=None):
        self.eval_started = time.perf_counter()
        label = "Anfangsevaluation" if step == 0 else f"Evaluation bei Schritt {number(step)}"
        opponents = ('trainierte Vorgänger, Schritte ' + ', '.join(number(r['step']) for r in references)
                     if references else 'feste Botreferenzen')
        print(f"\n{'=' * 78}\n CHECKPOINT-EVALUATION | {label}\n"
              f" Gegner: {opponents} | {number(games)} Spiele laufen ...\n{'=' * 78}", flush=True)

    def evaluation_end(self, summary, *, new_best=False):
        elapsed = time.perf_counter() - self.eval_started
        self.eval_seconds += elapsed
        self.eval_count += 1

        lines = evaluation_lines(summary, new_best=new_best)
        previous_plan = self.previous_evaluation.get('plan_id') if self.previous_evaluation else None
        comparable = (previous_plan is not None and previous_plan == summary.get('plan_id')
                      and self.previous_evaluation.get('valid_for_selection')
                      and summary.get('valid_for_selection'))
        if self.previous_evaluation is not None and comparable:
            lines.append('\n ÄNDERUNG ZUR VORHERIGEN EVALUATION (gleicher Testplan)')
            for key, label, scale in (('win_share', 'Sieganteil / Prozentpunkte', 100),
                                      ('agent_vp_mean', 'Eigene Punkte', 1),
                                      ('vp_diff_mean', 'Punktevorsprung', 1)):
                previous = self.previous_evaluation.get(key)
                current = summary.get(key)
                if current is not None and previous is not None:
                    lines.append(f" {label:30} {decimal(scale * (current - previous), signed=True)}")
        elif self.previous_evaluation is not None:
            reason = ('Gegner/Testplan geändert' if previous_plan != summary.get('plan_id')
                      else 'Unvollständige Evaluation')
            lines.append(f'\n {reason} — kein direkter Fortschrittsvergleich zur letzten Evaluation.')
        self.previous_evaluation = dict(summary)
        lines.append(f"  Dauer:           {duration(elapsed)}\n")
        print("\n".join(lines), flush=True)

    def progress(self, step, updates):
        elapsed = time.perf_counter() - self.started
        training_seconds = max(0.0, elapsed - self.eval_seconds)
        speed = step / training_seconds if training_seconds else 0.0
        remaining = max(0, self.args.total_timesteps - step)
        # Initial evaluation is already complete. Count scheduled future evaluations
        # and the mandatory final self-play evaluation, without double counting.
        future_evals = self.args.total_timesteps // self.args.eval_every - step // self.args.eval_every
        if self.args.mode == 'selfplay' and self.args.total_timesteps % self.args.eval_every and remaining:
            future_evals += 1
        eta = remaining / speed if speed else 0.0
        if self.eval_count:
            eta += future_evals * self.eval_seconds / self.eval_count
        fraction = step / self.args.total_timesteps
        filled = min(20, int(20 * fraction))
        print(f"[{('#' * filled).ljust(20, '-')}] {fraction:6.1%} | "
              f"{number(step)}/{number(self.args.total_timesteps)} Schritte | {updates} Updates\n"
              f"  Laufzeit {duration(elapsed)} | Rest ~{duration(eta)} | "
              f"{speed:.1f} Schritte/s (ohne Evaluationen)", flush=True)

    def finish(self, out_dir):
        print(f"\n{'=' * 72}\n"
              f"Training abgeschlossen | Dauer {duration(time.perf_counter() - self.started)}\n"
              f"  Letztes Modell: {out_dir / 'latest.pt'}\n"
              f"  Ergebnisse:    {out_dir}\n"
              f"{'=' * 72}", flush=True)


class HoldoutConsole:
    """Present standalone evaluation using the same metrics as training."""

    def __init__(self, seed, blocks, games, candidate, references, output):
        self.started = time.perf_counter()
        print(f"\n{'=' * 72}\n"
              f"  SELF-PLAY AUSWERTUNG | Seed {seed}\n"
              f"{'=' * 72}\n"
              f"  Testumfang:    {blocks} Seed-Blöcke | {number(games)} Spiele je Modell\n"
              f"  Testmodell:    {candidate}\n"
              f"  Ergebnisse:    {output}", flush=True)
        for i, ref in enumerate(references, 1):
            print(f"  Gegner {i}:      {ref['checkpoint']}", flush=True)
        print("  Vergleich:     Untrainiertes und trainiertes Modell auf gleichem Spielplan\n"
              f"{'=' * 72}", flush=True)

    def evaluation_start(self, name, games):
        self.phase_started = time.perf_counter()
        label = {'initial': 'Untrainiertes Ausgangsmodell', 'candidate': 'Trainiertes Testmodell'}.get(name, name)
        print(f"\n{label} | {number(games)} Spiele laufen ...", flush=True)

    def progress(self, done, total):
        if done % 60 == 0 or done == total:
            elapsed = time.perf_counter() - self.phase_started
            eta = elapsed * (total - done) / done
            print(f"  Spiele {done}/{total} | {done / total:.0%} | "
                  f"Dauer {duration(elapsed)} | Rest ~{duration(eta)}", flush=True)

    def evaluation_end(self, summary):
        print("\n".join(evaluation_lines(summary)) +
              f"\n  Dauer:           {duration(time.perf_counter() - self.phase_started)}", flush=True)

    def finish(self, summaries, output):
        progress = summaries['progress']
        print(f"\n{'=' * 72}\n  VERGLEICH ZUM UNTRAINIERTEN AUSGANGSMODELL\n{'=' * 72}", flush=True)
        if progress.get('paired_valid'):
            print(f"  Punktevorsprung: {progress['vp_diff_gain']:+.2f} Punkte Zuwachs\n"
                  f"  Sieganteil:      {progress['win_share_gain'] * 100:+.1f} Prozentpunkte Zuwachs",
                  flush=True)
            for key, label, scale, unit in (
                ('vp_diff_gain_ci95', 'Punkte-Zuwachs', 1, 'Punkte'),
                ('win_share_gain_ci95', 'Sieganteil-Zuwachs', 100, 'Prozentpunkte'),
            ):
                interval = progress.get(key)
                if interval is not None:
                    print(f"  95%-Intervall ({label}): "
                          f"[{interval[0] * scale:+.2f}, {interval[1] * scale:+.2f}] {unit}", flush=True)
        else:
            print("  Vergleich ungültig: unvollständige Spiele.", flush=True)
        complete = all(summaries[name]['valid_for_selection'] for name in ('initial', 'candidate'))
        print(f"\n  Status:          {'Vollständig abgeschlossen' if complete else 'UNVOLLSTÄNDIG – Fehler prüfen'}\n"
              f"  Gesamtdauer:     {duration(time.perf_counter() - self.started)}\n"
              f"  Detailergebnisse: {output / 'summary.json'}\n"
              f"{'=' * 72}", flush=True)
