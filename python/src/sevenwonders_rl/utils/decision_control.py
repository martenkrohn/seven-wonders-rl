"""Cooperative stop requests for a locked active training run."""

from contextlib import contextmanager
import fcntl
from pathlib import Path
import signal
import threading


class RunControl:
    def __init__(self, directory):
        self.path = Path(directory) / "stop.request"
        self.requested = False

    def signal_stop(self, signum, frame):
        self.requested = True

    def should_stop(self):
        return self.requested or self.path.exists()


@contextmanager
def control_run(directory):
    """Call only while holding the exclusive training-directory lock."""
    control = RunControl(directory)
    control.path.unlink(missing_ok=True)  # A previous run's request is stale.
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, control.signal_stop)
    try:
        yield control
    finally:
        control.path.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def request_stop(directory):
    """Request a boundary stop; never start a trainer or signal unrelated PIDs."""
    directory = Path(directory)
    lock_path = directory / ".training.lock"
    if not lock_path.exists():
        return False
    with lock_path.open("r") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            (directory / "stop.request").touch()
            return True
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            return False
