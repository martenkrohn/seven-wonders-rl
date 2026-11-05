"""Build and open a dashboard from available completed evaluation reports."""
import argparse
from pathlib import Path
import shutil
import subprocess
import webbrowser

from sevenwonders_rl.evaluation.dashboard_reference import workspace_dashboard


def dashboard_path(root):
    if not (Path(root) / 'configs' / 'dashboard.json').is_file():
        raise FileNotFoundError('Noch keine Dashboard-Konfiguration vorhanden: configs/dashboard.json')
    return workspace_dashboard(root)


def open_browser(path):
    path = path.resolve()
    wslview = shutil.which('wslview')
    if wslview:
        subprocess.run([wslview, str(path)], check=True)
        return
    explorer, wslpath = shutil.which('explorer.exe'), shutil.which('wslpath')
    if explorer and wslpath:
        windows_path = subprocess.check_output([wslpath, '-w', str(path)], text=True).strip()
        # Explorer hands HTML files to their default Windows application.
        # It may return 1 even when the hand-off succeeds.
        result = subprocess.run([explorer, windows_path], check=False)
        if result.returncode not in (0, 1):
            raise RuntimeError(f'Windows-Browser konnte nicht geöffnet werden: {path}')
        return
    if not webbrowser.open(path.as_uri()):
        raise RuntimeError(f'Kein Browser gefunden. Diese Datei manuell öffnen: {path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-path', action='store_true', help='Show the selected file without opening a browser')
    parser.add_argument('--run-dir', type=Path, help='Open this training run dashboard directly')
    args = parser.parse_args(argv)
    path = (args.run_dir.resolve() / 'dashboard.html' if args.run_dir else
            dashboard_path(Path(__file__).resolve().parents[1]))
    if args.run_dir and not path.is_file():
        from sevenwonders_rl.evaluation.dashboard_reference import read_dashboard_config
        root = Path(__file__).resolve().parents[1]
        config = read_dashboard_config(root / 'configs' / 'dashboard.json', root)
        if args.run_dir.resolve() == Path(config['active_run']):
            path = dashboard_path(root)
    if not path.is_file():
        raise FileNotFoundError(f'Dashboard noch nicht vorhanden: {path}')
    print(f'Analyse-Dashboard: {path}', flush=True)
    if not args.print_path:
        open_browser(path)


if __name__ == '__main__':
    main()
