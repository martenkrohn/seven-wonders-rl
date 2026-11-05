"""Human-readable model identities without exposing internal result keys."""
from pathlib import Path


def model_label(name, descriptor=None, *, short=False):
    descriptor = descriptor or {}
    if descriptor.get('label'):
        return descriptor['label']
    if descriptor.get('head_id') is not None:
        return f"Head {descriptor['head_id']}"
    if 'action_constraint' in descriptor:
        return {'none': 'Freie Policy', 'green_priority': 'Grün-Priorität'}[descriptor['action_constraint']]
    step = descriptor.get('timestep')
    stem = Path(descriptor.get('checkpoint', '')).stem
    if step is None and stem.startswith('checkpoint_') and stem[11:].isdigit():
        step = int(stem[11:])
    formatted = f'{step:,}'.replace(',', '.') if step is not None else None
    if name == 'initial':
        if descriptor.get('prior_strength', 0) > 0:
            return 'Start mit Priors' if short else 'Startmodell mit Priors (Schritt 0)'
        if descriptor.get('initialization', {}).get('kind') == 'checkpoint':
            if descriptor.get('num_heads', 1) > 1:
                return 'Startmodell' if short else 'Vortrainiertes Ausgangsmodell (Zusatztraining Schritt 0)'
            return 'Startmodell' if short else 'Vortrainiertes Ausgangsmodell (Spezialisierungsschritt 0)'
        return 'Untrainiert' if short else 'Untrainiertes Ausgangsmodell (Schritt 0)'
    if name == 'candidate':
        # Explicit --checkpoint can evaluate a non-final model.
        role = 'Testmodell' if stem.startswith('checkpoint_') else 'Trainingsende'
        return f"{'Test' if role == 'Testmodell' else 'Ende'} {formatted}" if formatted and short else (
            f'{role} – Schritt {formatted}' if formatted else role)
    return f'Schritt {formatted}' if formatted and short else (
        f'Zwischenstand – Schritt {formatted}' if formatted else f"Zusätzlicher Zwischenstand {name.removeprefix('comparison_')}")
