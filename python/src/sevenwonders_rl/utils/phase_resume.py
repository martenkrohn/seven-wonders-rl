"""Resume completed PPO updates; explicitly restart unfinished engine games."""
import json
from pathlib import Path


def runtime_snapshot(updates, episodes, active_games, env_rngs, head_rngs, minibatch_rng, action_generator):
    return dict(version=1, updates=updates, episodes=list(episodes), active_games=active_games,
        opponent_rngs=[r.bit_generator.state for r in env_rngs],
        head_rngs=[r.bit_generator.state for r in head_rngs],
        minibatch_rng=minibatch_rng.bit_generator.state, action_rng=action_generator.get_state(),
        engine_resume='unfinished games are abandoned; fresh games use new episode seeds')


def restore_runtime(state, episodes, env_rngs, head_rngs, minibatch_rng, action_generator):
    if not state or state.get('version') != 1 or len(state['episodes']) != len(episodes):
        raise ValueError('Checkpoint has no compatible phase resume state')
    # Move beyond every game active at the checkpoint; never replay its engine seed.
    episodes[:] = [int(i) + 1 for i in state['episodes']]
    for rng, saved in zip(env_rngs, state['opponent_rngs']):
        rng.bit_generator.state = saved
    for rng, saved in zip(head_rngs, state['head_rngs']):
        rng.bit_generator.state = saved
    minibatch_rng.bit_generator.state = state['minibatch_rng']
    action_generator.set_state(state['action_rng'])


def trim_uncommitted_logs(directory, step, updates):
    """Logs after the last durable update are copied aside before continuing."""
    directory = Path(directory)
    for name, key, boundary in [('updates.jsonl', 'update', updates),
                                ('episodes.jsonl', 'completed_update', updates),
                                ('opponents.jsonl', 'selection_step', step - 1)]:
        path = directory / name
        if not path.exists():
            continue
        temporary = path.with_suffix('.jsonl.tmp')
        rolled_back = directory / (name + '.rolled_back')
        with path.open() as source, temporary.open('w') as output, rolled_back.open('a') as archive:
            line = source.readline()
            while line:
                following = source.readline()
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A hard kill may leave only the final append incomplete.
                    # Interior corruption and complete malformed lines still fail.
                    if following or line.endswith('\n'):
                        raise
                    archive.write(line + '\n')
                else:
                    (output if row.get(key, 0) <= boundary else archive).write(line)
                line = following
        temporary.replace(path)
