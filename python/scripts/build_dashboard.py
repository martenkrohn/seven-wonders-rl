"""Build a local dashboard from saved analysis.json files; no games are run."""
import argparse
import json
from pathlib import Path

from sevenwonders_rl.evaluation.dashboard import write_dashboard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', required=True)
    parser.add_argument('--compare-dir', nargs='*', default=[])
    args = parser.parse_args(argv)
    output = Path(args.evaluation_dir).expanduser().resolve()
    report = json.loads((output / 'analysis.json').read_text(encoding='utf-8'))
    destination = write_dashboard(output, report, comparisons=[Path(p).expanduser().resolve() for p in args.compare_dir])
    print(f'Dashboard erstellt: {destination}\nIm Browser öffnen: {destination.as_uri()}')
    return destination


if __name__ == '__main__':
    main()
