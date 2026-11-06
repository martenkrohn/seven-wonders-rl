# Research questions

The current agent learns with shared own-hand/public-history context, one PPO
learner and an own-snapshot pair-PFSP league. Upcoming experiments should answer
one specific question with a fixed plan and explicit success criteria.

1. Does explicit history improve learning or robustness at a matched compute
   budget? The current information contract provides access; it is not an
   isolated causal test of the memory benefit.
2. How does opponent selection affect policy coverage and strategic diversity?
   PFSP estimates difficulty from training outcomes; it does not measure
   independent strategies or solve the multiplayer meta-game.
3. Do apparent gains survive new opponents, independent training seeds and a
   final evaluation plan that was not used to choose checkpoints?

No additional model family or compatibility path is required just to keep an
option open. Add an experimental variant only for an actual comparison, and
retire its implementation if it is no longer used.

[Results](../results/README.md) distinguish observations from conclusions.
The [historical literature review](LITERATURE.md) retains the October research
and sources. Its implementation audit concerns the retired agent; current
contracts are in [architecture](../reference/ARCHITECTURE.md).
