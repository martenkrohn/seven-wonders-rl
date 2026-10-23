"""Human-readable model identities without exposing internal result keys."""
from pathlib import Path


def model_label(name, descriptor=None, *, short=False):
    descriptor = descriptor or {}
    step = descriptor.get('timestep')
    stem = Path(descriptor.get('checkpoint', '')).stem
    if step is None and stem.startswith('checkpoint_') and stem[11:].isdigit():
        step = int(stem[11:])
    formatted = f'{step:,}'.replace(',', '.') if step is not None else None
    if name == 'initial':
        return 'Untrainiert' if short else 'Untrainiertes Ausgangsmodell (Schritt 0)'
    if name == 'candidate':
        # Explicit --checkpoint can evaluate a non-final model.
        role = 'Testmodell' if stem.startswith('checkpoint_') else 'Trainingsende'
        return f"{'Test' if role == 'Testmodell' else 'Ende'} {formatted}" if formatted and short else (
            f'{role} – Schritt {formatted}' if formatted else role)
    return f'Schritt {formatted}' if formatted and short else (
        f'Zwischenstand – Schritt {formatted}' if formatted else f"Zusätzlicher Zwischenstand {name.removeprefix('comparison_')}")
