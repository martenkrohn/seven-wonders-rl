# Environment and engine interface

## Turn exchange

`SevenWondersEnv` starts `build/7Wonders` with a per-environment working directory. The existing simulator expects an `io/` directory relative to that working directory.

1. C++ publishes `game_status.json` with players, cards, resources and legal/available actions.
2. Python builds relative-seat observations and a mask for each controlled seat.
3. The learner or opponent policy selects an encoded action.
4. Python writes `player_1.json`, `player_2.json` and `player_3.json`, then signals readiness through `ready.txt`.
5. The engine advances and publishes the next status; Python waits for the turn change.

The file protocol is inherited from the simulator. The Python environment is the RL bridge around it. A RAM-backed `--io-root` retains the same protocol and must be kept separate from persistent logs and checkpoints.

## Action and observation contracts

The action index is `action_type * 7 + hand_slot`. Types represent structure building, wonder building, free building from the hand and discarding. Unused slots and unavailable actions are masked. Feature ordering is defined in `features/action_feature_schema.py`; observation construction is in `features/obs_builder.py`.

`SevenWondersEnv` controls all seats to advance the simulator, but exposes the learner's selected seat as the Gymnasium agent. Other seats use heuristic or checkpoint-backed policies. A terminal reward uses the learner's victory-point difference against its strongest opponent, divided by `reward_scale`.

## Training

The shared actor-critic scores the action representation and estimates state value. `train_unified.py` supports fixed heuristic mixtures and an opponent pool of frozen snapshots. Evaluation tools report game outcomes and action statistics. These are implemented experimental components; [known issues](KNOWN_ISSUES.md) currently prevent a validated learning claim.
