"""Active BGA environment; legacy V2 APIs remain explicitly available."""
from .decision_env import DecisionEnv
SevenWondersEnv=DecisionEnv
__all__=['DecisionEnv','SevenWondersEnv','LegacySevenWondersEnv','SeatPolicy']

def __getattr__(name):
    if name in ('LegacySevenWondersEnv','SeatPolicy'):
        from .sevenwonders_env import SevenWondersEnv as legacy,SeatPolicy
        return legacy if name=='LegacySevenWondersEnv' else SeatPolicy
    raise AttributeError(name)
