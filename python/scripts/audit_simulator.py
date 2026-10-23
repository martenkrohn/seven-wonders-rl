"""Maintain an exact C++/build/test comparison against the pinned upstream."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASELINE = "a0bc333a7a7f21f83156de8a2d522e7baa913f0a"
ARTIFACTS = ROOT / "docs/simulator"


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=check)


def in_scope(path):
    p = Path(path)
    return path in ("main.cpp", "CMakeLists.txt") or (
        p.parts[0] in ("src", "include", "tests") and p.suffix in (".cpp", ".h", ".hpp", ".cc"))


def current_paths():
    paths = {"main.cpp", "CMakeLists.txt"}
    for folder in ("src", "include", "tests"):
        paths.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*")
                     if p.is_file() and in_scope(str(p.relative_to(ROOT))))
    return paths


def sha(data):
    return hashlib.sha256(data).hexdigest()


def current(path):
    p = ROOT / path
    return ({"sha256": sha(p.read_bytes()), "mode": "100755" if p.stat().st_mode & 0o111 else "100644"}
            if p.is_file() else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify inventory and patch integrity without fetching Git history")
    args = parser.parse_args(argv)
    patch_path, manifest_path = ARTIFACTS / "upstream.patch", ARTIFACTS / "manifest.json"
    if args.check:
        recorded = json.loads(manifest_path.read_text())
        if recorded["baseline_commit"] != BASELINE or sha(patch_path.read_bytes()) != recorded["patch_sha256"]:
            raise SystemExit("Simulator baseline/patch integrity mismatch")
        entries = {row["path"]: row for row in recorded["files"]}
        actual = current_paths()
        missing = actual - set(entries)
        changed = [path for path, row in entries.items() if current(path) != row["current"]]
        if missing or changed:
            raise SystemExit(f"Simulator audit is stale: new={sorted(missing)}, changed={changed}. Update SIMULATOR_CHANGES.md and regenerate.")
        print(f"Simulator audit matches {len(entries)} C++/build/test files; patch SHA-256 verified")
        return
    if git("cat-file", "-e", f"{BASELINE}^{{commit}}", check=False).returncode:
        raise SystemExit(f"Pinned upstream commit missing locally: fetch {BASELINE} from the documented upstream before regeneration")
    tree = git("ls-tree", "-r", BASELINE).stdout.decode().splitlines()
    baseline = {}
    for line in tree:
        metadata, path = line.split("\t", 1)
        if in_scope(path):
            mode, _, blob = metadata.split()
            baseline[path] = {"mode": mode, "sha256": sha(git("cat-file", "blob", blob).stdout)}
    paths = sorted(set(baseline) | current_paths())
    entries = []
    for path in paths:
        before, after = baseline.get(path), current(path)
        status = "added" if before is None else "deleted" if after is None else "unchanged" if before == after else "modified"
        entries.append({"path": path, "status": status, "baseline": before, "current": after})
    tracked = set(git("ls-files").stdout.decode().splitlines())
    tracked_paths = [p for p in paths if p in tracked or p in baseline]
    patch = git("diff", "--no-ext-diff", "--binary", "--full-index", "--no-color", BASELINE, "--", *tracked_paths).stdout
    for path in paths:
        if path not in tracked and path not in baseline:
            result = git("diff", "--no-index", "--binary", "--full-index", "--no-color", "--", "/dev/null", path, check=False)
            if result.returncode not in (0, 1):
                raise RuntimeError(result.stderr.decode())
            patch += result.stdout
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(patch)
    manifest_path.write_text(json.dumps({"format_version": 1,
        "upstream": "https://github.com/dmag-ufsm/7Wonders", "baseline_commit": BASELINE,
        "patch_sha256": sha(patch), "files": entries}, indent=2) + "\n")
    print(f"Recorded {len(entries)} files; {sum(e['status'] != 'unchanged' for e in entries)} differ from upstream")


if __name__ == "__main__":
    main()
