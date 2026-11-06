"""Seven Wonders RL: Second Edition decisions, visible memory and PPO."""

__version__ = "0.2.0"
from gymnasium.envs.registration import register, registry

if "SevenWondersBGA-v0" not in registry:
    register(id="SevenWondersBGA-v0", entry_point="sevenwonders_rl.envs:DecisionEnv")
