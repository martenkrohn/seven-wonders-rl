from pathlib import Path

import pytest

from sevenwonders_rl.utils import runtime
from test_training_contracts import trainer_module

pytestmark = pytest.mark.unit


def test_automatic_ram_and_explicit_disk_paths(tmp_path, monkeypatch):
    ram = tmp_path / "shared-memory" / "sevenwonders"
    ram.parent.mkdir()
    monkeypatch.setattr(runtime, "RAM_IO_ROOT", ram)
    auto = runtime.exchange_io_root(None, tmp_path / "disk")
    explicit = runtime.exchange_io_root(tmp_path / "explicit", tmp_path / "disk")
    assert auto.is_relative_to(ram) and auto.is_dir()
    assert explicit.is_relative_to(tmp_path / "explicit") and explicit.is_dir()
    disk = runtime.exchange_io_root(None, tmp_path / "disk", prefer_ram=False)
    assert disk.is_relative_to(tmp_path / "disk")


def test_unavailable_ram_falls_back_but_explicit_failures_propagate(tmp_path, monkeypatch):
    ram = tmp_path / "shared-memory" / "sevenwonders"
    ram.parent.mkdir()
    monkeypatch.setattr(runtime, "RAM_IO_ROOT", ram)
    original = runtime.run_io_root
    def denied(root, prefix):
        if Path(root) == ram:
            raise PermissionError("Shared memory unavailable")
        return original(root, prefix)
    monkeypatch.setattr(runtime, "run_io_root", denied)
    assert runtime.exchange_io_root(None, tmp_path / "disk").is_relative_to(tmp_path / "disk")
    with pytest.raises(PermissionError):
        runtime.exchange_io_root(ram, tmp_path / "disk")


def test_sparse_evaluation_default_and_explicit_override():
    trainer = trainer_module()
    assert trainer.parse_args([]).eval_every == 32768
    assert trainer.parse_args([]).opponent_inference == "batched"
    assert trainer.parse_args(["--eval-every", "1024"]).eval_every == 1024
