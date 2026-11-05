"""Build a local dashboard from saved evaluations or science summaries; no games are run."""
import argparse
import json
from pathlib import Path

from sevenwonders_rl.evaluation.dashboard import write_dashboard, write_science_dashboard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir')
    parser.add_argument('--compare-dir', nargs='*', default=[])
    parser.add_argument('--science-summary', help='Science diagnostics JSON; can also accompany an evaluation')
    parser.add_argument('--out-dir', help='Output directory for a science-only dashboard')
    args = parser.parse_args(argv)
    if not args.evaluation_dir and not args.science_summary:
        parser.error('provide --evaluation-dir or --science-summary')
    if (args.compare_dir and not args.evaluation_dir) or (args.out_dir and args.evaluation_dir):
        parser.error('--compare-dir requires --evaluation-dir; --out-dir is for science-only dashboards')
    science = json.loads(Path(args.science_summary).read_text(encoding='utf-8')) if args.science_summary else None
    if args.evaluation_dir:
        output = Path(args.evaluation_dir).expanduser().resolve()
        report = json.loads((output / 'analysis.json').read_text(encoding='utf-8'))
        destination = write_dashboard(output, report, comparisons=[Path(p).expanduser().resolve() for p in args.compare_dir],
                                      science_summary=science)
    else:
        output = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(args.science_summary).resolve().parent
        destination = write_science_dashboard(output / 'dashboard.html', science)
    print(f'Dashboard erstellt: {destination}\nIm Browser öffnen: {destination.as_uri()}')
    return destination


if __name__ == '__main__':
    main()
