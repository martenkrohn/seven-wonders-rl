"""Run the dependency-free JavaScript contracts when Node is available (including CI)."""
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.unit
def test_dashboard_language_runtime():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node unavailable; the same JS harness can run in an external V8 isolate')
    root = Path(__file__).parents[1]
    harness = root / 'tests/js/dashboard_language.js'
    assets = root / 'src/sevenwonders_rl/evaluation/assets'
    subprocess.run([node, '-e', '''
const fs=require('node:fs'),path=require('node:path');
const run=require(process.argv[1]);
const names={i18n:'i18n.js',theme:'theme.js',dashboard:'dashboard.js'};
const sources=Object.fromEntries(Object.entries(names).map(([k,v])=>[k,fs.readFileSync(path.join(process.argv[2],v),'utf8')]));
run(sources).then(result=>console.log(JSON.stringify(result))).catch(error=>{console.error(error);process.exitCode=1;});
''', str(harness), str(assets)], check=True, timeout=30)
